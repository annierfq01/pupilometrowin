"""
pupil_adaptive_analyzer.py - Adaptive Binary Search Pupil Analyzer
==================================================================
Modulo de analisis acelerado semiautomatico que reduce la cantidad de frames
a procesar usando busqueda binaria adaptativa con validacion por curva esperada.

El video ya esta segmentado en 3 fases conocidas (pre-iluminacion, iluminacion,
post-iluminacion) con timestamps exactos.

Arquitectura:
  1. Data structures: PupilSample, PhaseResult, PupilMetrics (dataclasses)
  2. AdaptiveBinaryAnalyzer class with configurable parameters
  3. analyze_pre_phase(): Flat model baseline detection with CV stability check
  4. analyze_illumination_phase(): Binary search for contraction minimum +
     dense post-minimum sampling + sigmoid fitting + escape detection
  5. analyze_post_phase(): Exponential model-guided binary search for
     redilation + PIPR/T75/T90 calculation
  6. compute_metrics(): Derive clinical metrics from phase results
  7. run_adaptive_analysis(): Main orchestration function with fallback
     to full analysis if poor fit
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, interp1d
import os
import csv


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PupilSample:
    """Representa una medicion o estimacion del diametro pupilar en un frame."""
    frame_index: int
    timestamp_ms: float
    diameter_px: float
    confidence: float
    is_interpolated: bool


@dataclass
class PhaseResult:
    """Resultado del analisis de una fase del video."""
    phase_name: str
    samples: List[PupilSample] = field(default_factory=list)
    measured_frames: List[int] = field(default_factory=list)
    interpolated_frames: List[int] = field(default_factory=list)
    curve_model: str = ""
    model_params: Dict = field(default_factory=dict)
    rmse: float = float('inf')
    is_valid: bool = False
    warnings: List[str] = field(default_factory=list)

    def get_diameter_at_ms(self, target_ms: float) -> Optional[float]:
        """Interpola el diametro en un timestamp dado usando los samples."""
        if not self.samples:
            return None
        times = [s.timestamp_ms for s in self.samples]
        diams = [s.diameter_px for s in self.samples]
        if target_ms <= times[0]:
            return diams[0]
        if target_ms >= times[-1]:
            return diams[-1]
        return float(np.interp(target_ms, times, diams))


@dataclass
class PupilMetrics:
    """Metricas clinicas derivadas del analisis de las tres fases."""
    baseline_diameter: float
    baseline_std: float
    min_diameter: float
    min_diameter_time_ms: float
    constriction_amplitude: float
    constriction_amplitude_pct: float
    constriction_latency_ms: float
    max_constriction_velocity: float
    pupillary_escape_detected: bool
    escape_onset_ms: Optional[float]
    escape_amplitude_px: Optional[float]
    redilation_velocity_mean: float
    t75_ms: Optional[float]
    t90_ms: Optional[float]
    pipr_1s: Optional[float]
    pipr_3s: Optional[float]
    pipr_6s: Optional[float]
    total_frames_analyzed: int
    total_frames_measured: int
    compression_ratio: float


class PupilDetectionError(Exception):
    """Error cuando el detector falla en demasiados frames."""
    pass


# ═══════════════════════════════════════════════════════════════════════
#  Module 2 — AdaptiveBinaryAnalyzer
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveBinaryAnalyzer:
    """
    Analizador adaptativo de diametro pupilar usando busqueda binaria.

    Reduce drasticamente el numero de frames a procesar midiendo solo
    puntos clave e interpolando el resto cuando el modelo lo permite.
    """

    def __init__(self,
                 get_pupil_diameter_fn: Callable[[int], Tuple[float, float]],
                 total_frames: int,
                 fps: float,
                 phase_boundaries: Dict[str, Tuple[int, int]],
                 error_margin_px: float = 2.0,
                 min_confidence: float = 0.7,
                 max_iterations: int = 8):
        """
        Inicializa el analizador adaptativo.

        Args:
            get_pupil_diameter_fn: callable(frame_index) -> (diameter_px, confidence)
            total_frames: Numero total de frames en el video
            fps: Frames por segundo del video
            phase_boundaries: {"pre": (0, f1), "ilum": (f1, f2), "post": (f2, fn)}
            error_margin_px: Margen de error aceptable en pixeles
            min_confidence: Confianza minima para aceptar una medicion
            max_iterations: Maximo de divisiones binarias por intervalo
        """
        self.get_diameter = get_pupil_diameter_fn
        self.total_frames = total_frames
        self.fps = fps
        self.phase_boundaries = phase_boundaries
        self.error_margin_px = error_margin_px
        self.min_confidence = min_confidence
        self.max_iterations = max_iterations
        self._baseline = None
        self._indeterminate_frames: set = set()

    def set_indeterminate_frames(self, indices: set):
        """Establece los frames indeterminados (ojo cerrado/pestan~eando)."""
        self._indeterminate_frames = indices

    def _frame_to_ms(self, frame_idx: int) -> float:
        """Convierte un indice de frame a milisegundos."""
        return frame_idx / self.fps * 1000.0

    def _ms_to_frame(self, ms: float) -> int:
        """Convierte milisegundos a un indice de frame."""
        return int(ms / 1000.0 * self.fps)

    def _measure_frame(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """
        Mide el diametro de un frame respetando frames indeterminados.

        Returns:
            (diameter_px, confidence) or None if frame is indeterminate
        """
        if frame_idx in self._indeterminate_frames:
            return None
        try:
            return self.get_diameter(frame_idx)
        except Exception:
            return None

    def _check_confidence_rate(self, measured: int, failed: int, phase_name: str):
        """Verifica que no mas del 30% de frames medidos fallen."""
        total = measured + failed
        if total > 0 and failed / total > 0.30:
            raise PupilDetectionError(
                f"Demasiados frames con baja confianza en fase {phase_name}: "
                f"{failed}/{total} ({failed/total*100:.1f}%)"
            )

    # ═══════════════════════════════════════════════════════════════════
    #  Module 3 — analyze_pre_phase()
    # ═══════════════════════════════════════════════════════════════════

    def analyze_pre_phase(self) -> PhaseResult:
        """
        Analiza la fase pre-iluminacion para confirmar estabilidad basal.

        Algoritmo:
          1. Toma 5 frames distribuidos uniformemente en la fase pre.
          2. Mide diametro en cada uno.
          3. Calcula media, desviacion estandar, coeficiente de variacion (CV).
          4. Si CV < 0.05: ajusta modelo flat, interpola resto.
          5. Si CV >= 0.05: agrega 5 frames mas, repite hasta 20 maximo.
          6. Retorna PhaseResult con modelo flat.

        Returns:
            PhaseResult con curva_model="flat" y baseline establecido.
        """
        pre_start, pre_end = self.phase_boundaries["pre"]
        result = PhaseResult(phase_name="pre", curve_model="flat")

        measured_indices = []
        measured_diams = []
        measured_confs = []
        failed_count = 0
        max_samples = 20
        batch_sizes = [5, 5, 5, 5]

        for batch_idx, batch_size in enumerate(batch_sizes):
            current_total = len(measured_indices)
            if current_total >= max_samples:
                break

            if batch_idx == 0:
                indices = np.linspace(pre_start, pre_end, batch_size, dtype=int)
            else:
                existing = set(measured_indices)
                all_candidates = np.linspace(pre_start, pre_end, max_samples, dtype=int)
                new_candidates = [c for c in all_candidates if c not in existing]
                indices = new_candidates[:batch_size]

            batch_diams = []
            batch_indices = []
            batch_confs = []

            for idx in indices:
                measurement = self._measure_frame(idx)
                if measurement is None:
                    failed_count += 1
                    continue
                diam, conf = measurement
                batch_diams.append(diam)
                batch_indices.append(idx)
                batch_confs.append(conf)

            if not batch_diams:
                result.warnings.append("no_valid_measurements_in_batch")
                continue

            measured_indices.extend(batch_indices)
            measured_diams.extend(batch_diams)
            measured_confs.extend(batch_confs)

            diam_arr = np.array(measured_diams)
            mean_diam = float(np.mean(diam_arr))
            std_diam = float(np.std(diam_arr))
            cv = std_diam / mean_diam if mean_diam > 0 else float('inf')

            if cv < 0.05 or len(measured_indices) >= max_samples:
                baseline = mean_diam
                self._baseline = baseline

                samples = []
                for idx, diam, conf in zip(measured_indices, measured_diams, measured_confs):
                    ts = self._frame_to_ms(idx)
                    samples.append(PupilSample(
                        frame_index=idx, timestamp_ms=ts,
                        diameter_px=diam, confidence=conf,
                        is_interpolated=False
                    ))

                for idx in range(pre_start, pre_end + 1):
                    if idx not in measured_indices:
                        ts = self._frame_to_ms(idx)
                        noise = np.random.normal(0, std_diam) if std_diam > 0 else 0
                        samples.append(PupilSample(
                            frame_index=idx, timestamp_ms=ts,
                            diameter_px=baseline + noise, confidence=0.5,
                            is_interpolated=True
                        ))

                samples.sort(key=lambda s: s.frame_index)
                result.samples = samples
                result.measured_frames = measured_indices
                result.interpolated_frames = [s.frame_index for s in samples if s.is_interpolated]
                result.model_params = {"baseline": baseline, "std": std_diam}
                result.rmse = std_diam
                result.is_valid = True

                if cv >= 0.05:
                    result.warnings.append("baseline_unstable")

                self._check_confidence_rate(len(measured_indices), failed_count, "pre")
                return result

        baseline = float(np.mean(measured_diams)) if measured_diams else 0.0
        std_diam = float(np.std(measured_diams)) if measured_diams else 0.0
        self._baseline = baseline

        samples = []
        for idx, diam, conf in zip(measured_indices, measured_diams, measured_confs):
            ts = self._frame_to_ms(idx)
            samples.append(PupilSample(
                frame_index=idx, timestamp_ms=ts,
                diameter_px=diam, confidence=conf,
                is_interpolated=False
            ))

        for idx in range(pre_start, pre_end + 1):
            if idx not in measured_indices:
                ts = self._frame_to_ms(idx)
                noise = np.random.normal(0, std_diam) if std_diam > 0 else 0
                samples.append(PupilSample(
                    frame_index=idx, timestamp_ms=ts,
                    diameter_px=baseline + noise, confidence=0.5,
                    is_interpolated=True
                ))

        samples.sort(key=lambda s: s.frame_index)
        result.samples = samples
        result.measured_frames = measured_indices
        result.interpolated_frames = [s.frame_index for s in samples if s.is_interpolated]
        result.model_params = {"baseline": baseline, "std": std_diam}
        result.rmse = std_diam
        result.is_valid = True
        result.warnings.append("baseline_unstable")

        self._check_confidence_rate(len(measured_indices), failed_count, "pre")
        return result

    # ═══════════════════════════════════════════════════════════════════
    #  Module 4 — analyze_illumination_phase()
    # ═══════════════════════════════════════════════════════════════════

    def analyze_illumination_phase(self, baseline: Optional[float] = None) -> PhaseResult:
        """
        Analiza la fase de iluminacion para detectar contraccion, minimo y escape.

        Etapas:
          3.1 - Deteccion de latencia de contraccion (primeros 200ms)
          3.2 - Busqueda binaria del minimo en la bajada monotona
          3.3 - Muestreo denso post-minimo (obligatorio)
          3.4 - Ajuste de modelo sigmoidal

        Args:
            baseline: Diametro basal de la fase pre (si None usa self._baseline)

        Returns:
            PhaseResult con curva_model="sigmoid_down"
        """
        if baseline is not None:
            self._baseline = baseline
        baseline = self._baseline if self._baseline is not None else 0.0

        ilum_start, ilum_end = self.phase_boundaries["ilum"]
        result = PhaseResult(phase_name="ilum", curve_model="sigmoid_down")

        frames_200ms = max(1, int(0.200 * self.fps))
        latency_indices = list(range(ilum_start, min(ilum_start + frames_200ms, ilum_end + 1)))

        measured_indices = []
        measured_diams = []
        measured_confs = []
        failed_count = 0
        constriction_start_frame = None

        for idx in latency_indices:
            measurement = self._measure_frame(idx)
            if measurement is None:
                failed_count += 1
                continue
            diam, conf = measurement
            measured_indices.append(idx)
            measured_diams.append(diam)
            measured_confs.append(conf)

            if constriction_start_frame is None and diam < (baseline - self.error_margin_px):
                constriction_start_frame = idx

        if constriction_start_frame is None:
            result.warnings.append("no_constriction_detected")
            constriction_start_frame = ilum_start

        measured_set = set(measured_indices)

        if len(measured_indices) >= 2:
            left_idx = measured_indices[-1]
            left_diam = measured_diams[-1]
            right_idx = ilum_end
            right_diam = baseline

            measurement = self._measure_frame(right_idx)
            if measurement is not None:
                right_diam = measurement[0]
                measured_indices.append(right_idx)
                measured_diams.append(measurement[0])
                measured_confs.append(measurement[1])
                measured_set.add(right_idx)
            else:
                failed_count += 1

            bs_measured, bs_interpolated = self._binary_search_minimum(
                left_idx, right_idx, left_diam, right_diam
            )

            for idx, diam, conf, is_interp in bs_measured:
                if not is_interp and idx not in measured_set:
                    measured_indices.append(idx)
                    measured_diams.append(diam)
                    measured_confs.append(conf)
                    measured_set.add(idx)

            for idx, diam, conf in bs_interpolated:
                if idx not in measured_set:
                    measured_indices.append(idx)
                    measured_diams.append(diam)
                    measured_confs.append(conf)
                    measured_set.add(idx)

        min_diam_idx = ilum_end
        min_diam_val = baseline
        for idx, diam in zip(measured_indices, measured_diams):
            if constriction_start_frame <= idx <= ilum_end:
                if diam < min_diam_val:
                    min_diam_val = diam
                    min_diam_idx = idx

        post_min_indices = list(range(min_diam_idx + 1, ilum_end + 1))
        if len(post_min_indices) > 0 and self.fps > 60:
            post_min_indices = post_min_indices[::2]

        for idx in post_min_indices:
            if idx in measured_set:
                continue
            measurement = self._measure_frame(idx)
            if measurement is None:
                failed_count += 1
                continue
            diam, conf = measurement
            measured_indices.append(idx)
            measured_diams.append(diam)
            measured_confs.append(conf)
            measured_set.add(idx)

        escape_detected = False
        escape_onset_ms = None
        escape_amplitude_px = None

        if len(measured_diams) > 1:
            sorted_pairs = sorted(zip(measured_indices, measured_diams))
            idxs, diams = zip(*sorted_pairs)
            diams = list(diams)
            for i in range(1, len(diams)):
                if diams[i] > diams[i - 1] + self.error_margin_px:
                    escape_detected = True
                    escape_onset_ms = self._frame_to_ms(idxs[i])
                    post_min_diams = [d for idx, d in sorted_pairs if idx >= min_diam_idx]
                    if post_min_diams:
                        escape_amplitude_px = max(post_min_diams) - min_diam_val
                    break

        all_timestamps = [self._frame_to_ms(idx) for idx in measured_indices]
        measured_ts_arr = np.array(all_timestamps)
        measured_diam_arr = np.array(measured_diams)

        if len(measured_ts_arr) >= 4:
            try:
                U = baseline
                L = min_diam_val

                def sigmoid_inv(t, k, t0):
                    return L + (U - L) / (1 + np.exp(k * (t - t0)))

                t_start_ms = measured_ts_arr[0]
                t_min_ms = self._frame_to_ms(min_diam_idx)

                popt, _ = curve_fit(
                    sigmoid_inv, measured_ts_arr, measured_diam_arr,
                    p0=[0.01, (t_start_ms + t_min_ms) / 2],
                    bounds=([0.001, t_start_ms], [1.0, t_min_ms]),
                    maxfev=5000
                )
                k_fit, t0_fit = popt
                predicted = sigmoid_inv(measured_ts_arr, k_fit, t0_fit)
                rmse = float(np.sqrt(np.mean((measured_diam_arr - predicted) ** 2)))

                model_params = {"U": U, "L": L, "k": k_fit, "t0": t0_fit}
                curve_model = "sigmoid_down"

                if rmse > 3 * self.error_margin_px:
                    result.warnings.append("poor_sigmoid_fit")
                    curve_model = "cubic_spline"
                    if len(measured_ts_arr) >= 4:
                        cs = CubicSpline(measured_ts_arr, measured_diam_arr)
                        model_params = {"cubic_spline": cs}
                        rmse = float(np.sqrt(np.mean(
                            (measured_diam_arr - cs(measured_ts_arr)) ** 2
                        )))
                    else:
                        model_params = {"U": U, "L": L, "k": 0.01, "t0": t_start_ms}
            except Exception:
                result.warnings.append("sigmoid_fit_failed")
                model_params = {"U": baseline, "L": min_diam_val, "k": 0.01, "t0": t_start_ms}
                curve_model = "sigmoid_down"
                rmse = float('inf')
        else:
            model_params = {"U": baseline, "L": min_diam_val, "k": 0.01, "t0": 0}
            curve_model = "sigmoid_down"
            rmse = float('inf')

        samples = []
        all_frames = set(range(ilum_start, ilum_end + 1))
        measured_set_final = set()

        for idx, diam, conf in zip(measured_indices, measured_diams, measured_confs):
            ts = self._frame_to_ms(idx)
            samples.append(PupilSample(
                frame_index=idx, timestamp_ms=ts,
                diameter_px=diam, confidence=conf,
                is_interpolated=False
            ))
            measured_set_final.add(idx)

        for idx in all_frames:
            if idx in measured_set_final:
                continue
            ts = self._frame_to_ms(idx)
            if curve_model == "sigmoid_down" and "k" in model_params:
                diam = model_params["L"] + (model_params["U"] - model_params["L"]) / \
                       (1 + np.exp(model_params["k"] * (ts - model_params["t0"])))
            elif curve_model == "cubic_spline" and "cubic_spline" in model_params:
                diam = float(model_params["cubic_spline"](ts))
            else:
                diam = float(np.interp(ts,
                                       [self._frame_to_ms(i) for i in measured_indices],
                                       measured_diams))

            samples.append(PupilSample(
                frame_index=idx, timestamp_ms=ts,
                diameter_px=float(diam), confidence=0.5,
                is_interpolated=True
            ))

        samples.sort(key=lambda s: s.frame_index)
        result.samples = samples
        result.measured_frames = list(measured_set_final)
        result.interpolated_frames = [s.frame_index for s in samples if s.is_interpolated]
        result.curve_model = curve_model
        result.model_params = model_params
        result.rmse = rmse
        result.is_valid = len(measured_indices) >= 3
        result.model_params["constriction_start_frame"] = constriction_start_frame
        result.model_params["constriction_latency_ms"] = \
            self._frame_to_ms(constriction_start_frame) - self._frame_to_ms(ilum_start)
        result.model_params["min_diameter"] = min_diam_val
        result.model_params["min_diameter_frame"] = min_diam_idx
        result.model_params["escape_detected"] = escape_detected
        result.model_params["escape_onset_ms"] = escape_onset_ms
        result.model_params["escape_amplitude_px"] = escape_amplitude_px

        self._check_confidence_rate(len(measured_indices), failed_count, "ilum")
        return result

    def _binary_search_minimum(self, frame_left: int, frame_right: int,
                                diameter_left: float, diameter_right: float
                                ) -> Tuple[List[Tuple[int, float, float, bool]],
                                           List[Tuple[int, float, float]]]:
        """
        Busqueda binaria del minimo pupilar en la bajada monotona.

        Args:
            frame_left: Indice del frame izquierdo (ya medido)
            frame_right: Indice del frame derecho (ya medido o estimado)
            diameter_left: Diametro en frame_left
            diameter_right: Diametro en frame_right

        Returns:
            Tuple de (measured_points, interpolated_points)
            measured: (frame_idx, diameter, confidence, is_interpolated=True/False)
            interpolated: (frame_idx, diameter, confidence)
        """
        measured = []
        interpolated = []
        iteration = 0

        left = frame_left
        right = frame_right
        d_left = diameter_left
        d_right = diameter_right

        measured.append((left, d_left, 1.0, False))

        while (right - left) > 3 and iteration < self.max_iterations:
            mid = (left + right) // 2
            measurement = self._measure_frame(mid)

            if measurement is None:
                for offset in [-1, 1]:
                    alt = mid + offset
                    if left < alt < right:
                        measurement = self._measure_frame(alt)
                        if measurement is not None:
                            mid = alt
                            break

            if measurement is None:
                left = mid
                iteration += 1
                continue

            d_mid, conf_mid = measurement

            if d_mid > d_left + self.error_margin_px:
                measured.append((mid, d_mid, conf_mid, False))
                interpolated.extend(self._interpolate_range(left, mid, d_left, d_mid))
                left = mid
                d_left = d_mid
                iteration += 1
                continue

            expected_mid = d_left + (d_right - d_left) * 0.5
            deviation = abs(d_mid - expected_mid)

            if deviation <= self.error_margin_px:
                measured.append((mid, d_mid, conf_mid, False))
                interpolated.extend(self._interpolate_range(left, mid, d_left, d_mid))
                interpolated.extend(self._interpolate_range(mid, right, d_mid, d_right))
                return measured, interpolated

            if d_mid < expected_mid - self.error_margin_px:
                measured.append((mid, d_mid, conf_mid, False))
                right = mid
                d_right = d_mid
            else:
                measured.append((mid, d_mid, conf_mid, False))
                left = mid
                d_left = d_mid

            iteration += 1

        measured.append((right, d_right, 1.0, False))
        return measured, interpolated

    def _interpolate_range(self, left: int, right: int,
                            d_left: float, d_right: float
                            ) -> List[Tuple[int, float, float]]:
        """Interpola linealmente todos los frames entre left y right."""
        result = []
        if right - left <= 1:
            return result
        for idx in range(left + 1, right):
            t = (idx - left) / (right - left)
            diam = d_left + t * (d_right - d_left)
            result.append((idx, diam, 0.5))
        return result

    # ═══════════════════════════════════════════════════════════════════
    #  Module 5 — analyze_post_phase()
    # ═══════════════════════════════════════════════════════════════════

    def analyze_post_phase(self, min_diameter: Optional[float] = None,
                            flash_end_ms: Optional[float] = None) -> PhaseResult:
        """
        Analiza la fase post-iluminacion para trazar redilatacion y encontrar T75, T90, PIPR.

        Etapas:
          4.1 - Estimacion inicial de tau con 4 frames + busqueda binaria guiada
          4.2 - Calculo de PIPR en 1s, 3s, 6s post-flash

        Args:
            min_diameter: Diametro minimo detectado en fase de iluminacion
            flash_end_ms: Timestamp de fin del flash (para PIPR)

        Returns:
            PhaseResult con curva_model="exponential_up"
        """
        if min_diameter is None:
            min_diameter = 0.0
        baseline = self._baseline if self._baseline is not None else 0.0

        post_start, post_end = self.phase_boundaries["post"]
        result = PhaseResult(phase_name="post", curve_model="exponential_up")

        initial_indices = np.linspace(post_start, post_end, min(4, post_end - post_start + 1), dtype=int)
        initial_indices = initial_indices.astype(int)

        measured_indices = []
        measured_diams = []
        measured_confs = []
        failed_count = 0

        for idx in initial_indices:
            measurement = self._measure_frame(idx)
            if measurement is None:
                failed_count += 1
                continue
            diam, conf = measurement
            measured_indices.append(int(idx))
            measured_diams.append(diam)
            measured_confs.append(conf)

        t_rel = np.array([self._frame_to_ms(idx) - self._frame_to_ms(post_start)
                          for idx in measured_indices])
        d_arr = np.array(measured_diams)

        tau_estimated = 1000.0

        def exp_model(t, tau):
            return baseline - (baseline - min_diameter) * np.exp(-t / tau)

        if len(t_rel) >= 3:
            try:
                popt, _ = curve_fit(
                    exp_model, t_rel, d_arr,
                    p0=[1000.0],
                    bounds=([10.0], [30000.0]),
                    maxfev=5000
                )
                tau_estimated = float(popt[0])
            except Exception:
                pass

        measured_set = set(measured_indices)

        self._binary_search_redilation(
            post_start, post_end, measured_indices, measured_diams,
            baseline, min_diameter, tau_estimated, result, measured_set
        )

        measured_set_final = set()
        samples = []

        for idx, diam, conf in zip(measured_indices, measured_diams, measured_confs):
            ts = self._frame_to_ms(idx)
            samples.append(PupilSample(
                frame_index=idx, timestamp_ms=ts,
                diameter_px=diam, confidence=conf,
                is_interpolated=False
            ))
            measured_set_final.add(idx)

        all_post_frames = set(range(post_start, post_end + 1))
        remaining_time = self._frame_to_ms(post_end) - self._frame_to_ms(post_start)
        recovery_target = min_diameter + 0.90 * (baseline - min_diameter)

        for idx in sorted(all_post_frames - measured_set_final):
            ts = self._frame_to_ms(idx)
            t_rel_val = ts - self._frame_to_ms(post_start)

            if remaining_time > 8000 and ts > self._frame_to_ms(post_start) + remaining_time * 0.7:
                diam = baseline - (baseline - min_diameter) * np.exp(-t_rel_val / tau_estimated)
            elif measured_indices:
                all_ts = [self._frame_to_ms(i) for i in measured_indices]
                diam = float(np.interp(ts, all_ts, measured_diams))
            else:
                diam = baseline

            samples.append(PupilSample(
                frame_index=idx, timestamp_ms=ts,
                diameter_px=float(diam), confidence=0.5,
                is_interpolated=True
            ))

        samples.sort(key=lambda s: s.frame_index)
        result.samples = samples
        result.measured_frames = list(measured_set_final)
        result.interpolated_frames = [s.frame_index for s in samples if s.is_interpolated]
        result.curve_model = "exponential_up"
        result.model_params = {
            "baseline": baseline, "min_diameter": min_diameter,
            "tau": tau_estimated, "flash_end_ms": flash_end_ms
        }

        def get_diam_at_ms(target_ms: float) -> float:
            if not result.samples:
                return baseline
            times = [s.timestamp_ms for s in result.samples]
            diams = [s.diameter_px for s in result.samples]
            return float(np.interp(target_ms, times, diams))

        if flash_end_ms is not None:
            result.model_params["pipr_1s"] = get_diam_at_ms(flash_end_ms + 1000)
            result.model_params["pipr_3s"] = get_diam_at_ms(flash_end_ms + 3000)
            result.model_params["pipr_6s"] = get_diam_at_ms(flash_end_ms + 6000)
        else:
            post_start_ms = self._frame_to_ms(post_start)
            result.model_params["pipr_1s"] = get_diam_at_ms(post_start_ms + 1000)
            result.model_params["pipr_3s"] = get_diam_at_ms(post_start_ms + 3000)
            result.model_params["pipr_6s"] = get_diam_at_ms(post_start_ms + 6000)

        result.rmse = float(np.std(measured_diams)) if measured_diams else float('inf')
        result.is_valid = len(measured_indices) >= 3

        all_post_ts = [s.timestamp_ms for s in result.samples]
        all_post_diams = [s.diameter_px for s in result.samples]
        if len(all_post_diams) > 1:
            grads = np.gradient(all_post_diams, all_post_ts)
            pos_grads = grads[grads > 0]
            result.model_params["redilation_velocity_mean"] = \
                float(np.mean(pos_grads)) if len(pos_grads) > 0 else 0.0
        else:
            result.model_params["redilation_velocity_mean"] = 0.0

        self._check_confidence_rate(len(measured_indices), failed_count, "post")
        return result

    def _binary_search_redilation(self, frame_left: int, frame_right: int,
                                   anchor_indices: List[int], anchor_diams: List[float],
                                   baseline: float, min_d: float, tau: float,
                                   result: PhaseResult, measured_set: set,
                                   depth: int = 0):
        """
        Busqueda binaria guiada por modelo exponencial para la redilatacion.

        Subdivide intervalos donde el modelo exponencial no predice bien
        y re-ajusta tau con cada nuevo punto anchor.
        """
        if depth > self.max_iterations or (frame_right - frame_left) <= 3:
            return

        mid = (frame_left + frame_right) // 2
        if mid in measured_set:
            return

        t_mid = self._frame_to_ms(mid)
        t_start = self._frame_to_ms(frame_left)
        t_rel = t_mid - t_start

        expected = baseline - (baseline - min_d) * np.exp(-t_rel / tau)

        measurement = self._measure_frame(mid)
        if measurement is None:
            self._binary_search_redilation(
                frame_left, mid, anchor_indices, anchor_diams,
                baseline, min_d, tau, result, measured_set, depth + 1
            )
            self._binary_search_redilation(
                mid, frame_right, anchor_indices, anchor_diams,
                baseline, min_d, tau, result, measured_set, depth + 1
            )
            return

        diam_mid, conf_mid = measurement
        deviation = abs(diam_mid - expected)

        anchor_indices.append(mid)
        anchor_diams.append(diam_mid)
        measured_set.add(mid)

        if deviation <= self.error_margin_px:
            for idx in range(frame_left + 1, frame_right):
                if idx not in measured_set and idx != mid:
                    t_val = self._frame_to_ms(idx) - t_start
                    interp_diam = baseline - (baseline - min_d) * np.exp(-t_val / tau)
                    anchor_indices.append(idx)
                    anchor_diams.append(interp_diam)
                    measured_set.add(idx)
        else:
            try:
                t_rel_arr = np.array([self._frame_to_ms(i) - self._frame_to_ms(frame_left)
                                      for i in anchor_indices])
                d_arr = np.array(anchor_diams)
                popt, _ = curve_fit(
                    lambda t, tau_val: baseline - (baseline - min_d) * np.exp(-t / tau_val),
                    t_rel_arr, d_arr, p0=[tau], bounds=([10.0], [30000.0]), maxfev=2000
                )
                tau = float(popt[0])
            except Exception:
                pass

            self._binary_search_redilation(
                frame_left, mid, anchor_indices, anchor_diams,
                baseline, min_d, tau, result, measured_set, depth + 1
            )
            self._binary_search_redilation(
                mid, frame_right, anchor_indices, anchor_diams,
                baseline, min_d, tau, result, measured_set, depth + 1
            )


# ═══════════════════════════════════════════════════════════════════════
#  Module 6 — compute_metrics()
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(pre_result: PhaseResult,
                    ilum_result: PhaseResult,
                    post_result: PhaseResult,
                    error_margin_px: float = 2.0,
                    flash_end_ms: Optional[float] = None,
                    total_frames: int = 0) -> PupilMetrics:
    """
    Calcula metricas clinicas a partir de los resultados de las tres fases.

    Args:
        pre_result: Resultado de la fase pre-iluminacion
        ilum_result: Resultado de la fase de iluminacion
        post_result: Resultado de la fase post-iluminacion
        error_margin_px: Margen de error usado en el analisis
        flash_end_ms: Timestamp de fin del flash (para PIPR)
        total_frames: Total de frames en el video

    Returns:
        PupilMetrics con todas las metricas clinicas calculadas
    """
    baseline_diam = pre_result.model_params.get("baseline", 0.0)
    baseline_std = pre_result.model_params.get("std", 0.0)

    ilum_samples = ilum_result.samples
    if not ilum_samples:
        min_diam = baseline_diam
        min_time_ms = 0.0
    else:
        min_sample = min(ilum_samples, key=lambda s: s.diameter_px)
        min_diam = min_sample.diameter_px
        min_time_ms = min_sample.timestamp_ms

    constriction_amplitude = baseline_diam - min_diam
    constriction_amplitude_pct = (constriction_amplitude / baseline_diam * 100) \
        if baseline_diam > 0 else 0.0

    constriction_latency_ms = ilum_result.model_params.get(
        "constriction_latency_ms", 0.0
    )

    measured_ilum = [s for s in ilum_samples if not s.is_interpolated]
    max_constr_velocity = 0.0
    if len(measured_ilum) >= 2:
        measured_ilum.sort(key=lambda s: s.timestamp_ms)
        times = np.array([s.timestamp_ms for s in measured_ilum])
        diams = np.array([s.diameter_px for s in measured_ilum])
        grads = np.gradient(diams, times)
        negative_grads = grads[grads < 0]
        if len(negative_grads) > 0:
            max_constr_velocity = float(abs(np.min(negative_grads)))

    escape_detected = ilum_result.model_params.get("escape_detected", False)
    escape_onset_ms = ilum_result.model_params.get("escape_onset_ms", None)
    escape_amplitude_px = ilum_result.model_params.get("escape_amplitude_px", None)

    redilation_velocity = post_result.model_params.get("redilation_velocity_mean", 0.0)

    post_samples = post_result.samples
    t75_ms = None
    t90_ms = None

    if len(post_samples) >= 2 and constriction_amplitude > 0:
        post_samples_sorted = sorted(post_samples, key=lambda s: s.timestamp_ms)
        recovery_75 = min_diam + 0.75 * constriction_amplitude
        recovery_90 = min_diam + 0.90 * constriction_amplitude

        post_ts = [s.timestamp_ms for s in post_samples_sorted]
        post_diams = [s.diameter_px for s in post_samples_sorted]

        for i in range(len(post_diams) - 1):
            if post_diams[i] < recovery_75 <= post_diams[i + 1]:
                frac = (recovery_75 - post_diams[i]) / (post_diams[i + 1] - post_diams[i])
                t75_ms = post_ts[i] + frac * (post_ts[i + 1] - post_ts[i])
                break

        for i in range(len(post_diams) - 1):
            if post_diams[i] < recovery_90 <= post_diams[i + 1]:
                frac = (recovery_90 - post_diams[i]) / (post_diams[i + 1] - post_diams[i])
                t90_ms = post_ts[i] + frac * (post_ts[i + 1] - post_ts[i])
                break

    pipr_1s = post_result.model_params.get("pipr_1s")
    pipr_3s = post_result.model_params.get("pipr_3s")
    pipr_6s = post_result.model_params.get("pipr_6s")

    total_measured = len(ilum_result.measured_frames) + \
                     len(pre_result.measured_frames) + \
                     len(post_result.measured_frames)

    compression_ratio = total_measured / total_frames if total_frames > 0 else 1.0

    return PupilMetrics(
        baseline_diameter=baseline_diam,
        baseline_std=baseline_std,
        min_diameter=min_diam,
        min_diameter_time_ms=min_time_ms,
        constriction_amplitude=constriction_amplitude,
        constriction_amplitude_pct=constriction_amplitude_pct,
        constriction_latency_ms=constriction_latency_ms,
        max_constriction_velocity=max_constr_velocity,
        pupillary_escape_detected=escape_detected,
        escape_onset_ms=escape_onset_ms,
        escape_amplitude_px=escape_amplitude_px,
        redilation_velocity_mean=redilation_velocity,
        t75_ms=t75_ms,
        t90_ms=t90_ms,
        pipr_1s=pipr_1s,
        pipr_3s=pipr_3s,
        pipr_6s=pipr_6s,
        total_frames_analyzed=total_frames,
        total_frames_measured=total_measured,
        compression_ratio=compression_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Module 7 — plot_results()
# ═══════════════════════════════════════════════════════════════════════

def plot_results(pre: PhaseResult, ilum: PhaseResult, post: PhaseResult,
                 metrics: PupilMetrics, flash_start_ms: Optional[float] = None,
                 flash_end_ms: Optional[float] = None,
                 output_dir: str = "./results", save_plots: bool = True):
    """
    Genera graficos de los resultados del analisis adaptativo.

    Crea figura matplotlib con 2 subplots:
      - Curva pupilar completa con puntos medidos/interpolados
      - Resumen de frames procesados (medido vs interpolado)

    Args:
        pre: Resultado fase pre-iluminacion
        ilum: Resultado fase iluminacion
        post: Resultado fase post-iluminacion
        metrics: Metricas clinicas calculadas
        flash_start_ms: Timestamp de inicio del flash
        flash_end_ms: Timestamp de fin del flash
        output_dir: Directorio para guardar los graficos
        save_plots: Si True, guarda como PNG y SVG
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.35, left=0.10, right=0.95, top=0.93, bottom=0.08)

    def _collect(phase: PhaseResult):
        measured_t = [s.timestamp_ms for s in phase.samples if not s.is_interpolated]
        measured_d = [s.diameter_px for s in phase.samples if not s.is_interpolated]
        interp_t = [s.timestamp_ms for s in phase.samples if s.is_interpolated]
        interp_d = [s.diameter_px for s in phase.samples if s.is_interpolated]
        return measured_t, measured_d, interp_t, interp_d

    colors = {"pre": "#3366cc", "ilum": "#cc2222", "post": "#228833"}

    for phase, color in colors.items():
        if phase == "pre":
            mt, md, it, id_ = _collect(phase)
        elif phase == "ilum":
            mt, md, it, id_ = _collect(phase)
        else:
            mt, md, it, id_ = _collect(phase)

        if mt:
            ax1.scatter(mt, md, color=color, s=20, zorder=5, label=f'{phase} medido', alpha=0.8)
        if it:
            ax1.plot(it, id_, color='gray', ls=':', lw=0.8, alpha=0.4, zorder=2)

    if flash_start_ms is not None:
        ax1.axvline(flash_start_ms, color='#00aa00', lw=1.2, ls='--', label='Flash ON')
    if flash_end_ms is not None:
        ax1.axvline(flash_end_ms, color='#cc0000', lw=1.2, ls='--', label='Flash OFF')

    ax1.axhline(metrics.baseline_diameter, color='gray', lw=0.8, ls='-', alpha=0.5,
                label=f'Baseline={metrics.baseline_diameter:.1f}px')

    all_times = [s.timestamp_ms for s in pre.samples + ilum.samples + post.samples]
    all_diams = [s.diameter_px for s in pre.samples + ilum.samples + post.samples]
    if all_times:
        ymin, ymax = min(all_diams) - 5, max(all_diams) + 5
        if flash_start_ms is not None and flash_end_ms is not None:
            ax1.fill_betweenx([ymin, ymax], flash_start_ms, flash_end_ms,
                              alpha=0.08, color='gold', zorder=0)

    ax1.scatter([metrics.min_diameter_time_ms], [metrics.min_diameter],
                color='purple', s=80, zorder=6, marker='*',
                label=f'MIN={metrics.min_diameter:.1f}px')

    if metrics.pupillary_escape_detected and metrics.escape_onset_ms is not None:
        ax1.scatter([metrics.escape_onset_ms], [metrics.min_diameter + (metrics.escape_amplitude_px or 0)],
                    color='orange', s=60, zorder=6, marker='D',
                    label='Escape pupilar')

    ax1.set_ylabel('Diametro pupilar (px)')
    ax1.set_title('Curva Pupilar - Analisis Adaptativo')
    ax1.legend(fontsize=7, ncol=3, loc='upper right')
    ax1.grid(True, alpha=0.25)

    for phase, color in colors.items():
        if phase == "pre":
            samples = pre.samples
        elif phase == "ilum":
            samples = ilum.samples
        else:
            samples = post.samples

        frames = [s.frame_index for s in samples]
        heights = [1.0 if not s.is_interpolated else 0.3 for s in samples]
        bar_colors = [color if not s.is_interpolated else 'gray' for s in samples]
        ax2.bar(frames, heights, color=bar_colors, width=1.0, alpha=0.7)

    ax2.set_xlabel('Frame index')
    ax2.set_ylabel('Medido')
    ax2.set_title(f'Frames procesados (compression ratio: {metrics.compression_ratio*100:.1f}%)')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Interpolado', 'Medido'])
    ax2.grid(True, alpha=0.15)

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "adaptive_analysis.png")
        svg_path = os.path.join(output_dir, "adaptive_analysis.svg")
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        plt.close(fig)
        return png_path, svg_path

    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Module 8 — run_adaptive_analysis()
# ═══════════════════════════════════════════════════════════════════════

def run_adaptive_analysis(video_path: str,
                          phase_timestamps: Dict[str, float],
                          get_diameter_fn: Callable[[int], Tuple[float, float]],
                          total_frames: int,
                          fps: float,
                          config: Optional[Dict] = None,
                          indeterminate_frames: Optional[set] = None) -> Tuple[PupilMetrics, Dict]:
    """
    Funcion principal de integracion para el analisis adaptativo.

    Flujo:
      1. Calcula frame_boundaries desde phase_timestamps
      2. Instancia AdaptiveBinaryAnalyzer
      3. Analiza las tres fases (pre, ilum, post)
      4. Calcula metricas
      5. Si poor fit y force_full_analysis_if_poor_fit, re-analiza esa fase
      6. Genera graficos si save_plots
      7. Exporta CSV si save_csv
      8. Retorna (metrics, all_results_dict)

    Args:
        video_path: Ruta al archivo de video
        phase_timestamps: {"pre_start_ms": float, "flash_start_ms": float,
                           "flash_end_ms": float, "end_ms": float}
        get_diameter_fn: callable(frame_index) -> (diameter_px, confidence)
        total_frames: Numero total de frames
        fps: Frames por segundo
        config: Dict de configuracion opcional
        indeterminate_frames: Set de indices de frames indeterminados

    Returns:
        Tuple de (PupilMetrics, dict_con_todos_los_resultados)
    """
    default_config = {
        "error_margin_px": 2.0,
        "min_confidence": 0.7,
        "max_binary_iterations": 8,
        "dense_sampling_post_minimum": True,
        "force_full_analysis_if_poor_fit": True,
        "output_dir": "./results",
        "save_plots": True,
        "save_csv": True
    }
    if config:
        default_config.update(config)
    config = default_config

    pre_start_ms = phase_timestamps.get("pre_start_ms", 0.0)
    flash_start_ms = phase_timestamps.get("flash_start_ms", 2000.0)
    flash_end_ms = phase_timestamps.get("flash_end_ms", 3000.0)
    end_ms = phase_timestamps.get("end_ms", 8000.0)

    def _ms_to_frame(ms: float) -> int:
        return int(ms / 1000.0 * fps)

    pre_start_f = _ms_to_frame(pre_start_ms)
    ilum_start_f = _ms_to_frame(flash_start_ms)
    ilum_end_f = _ms_to_frame(flash_end_ms)
    post_end_f = min(_ms_to_frame(end_ms), total_frames - 1)

    phase_boundaries = {
        "pre": (pre_start_f, ilum_start_f - 1),
        "ilum": (ilum_start_f, ilum_end_f - 1),
        "post": (ilum_end_f, post_end_f)
    }

    analyzer = AdaptiveBinaryAnalyzer(
        get_pupil_diameter_fn=get_diameter_fn,
        total_frames=total_frames,
        fps=fps,
        phase_boundaries=phase_boundaries,
        error_margin_px=config["error_margin_px"],
        min_confidence=config["min_confidence"],
        max_iterations=config["max_binary_iterations"]
    )

    if indeterminate_frames:
        analyzer.set_indeterminate_frames(indeterminate_frames)

    pre_result = analyzer.analyze_pre_phase()
    ilum_result = analyzer.analyze_illumination_phase(baseline=pre_result.model_params.get("baseline"))
    post_result = analyzer.analyze_post_phase(
        min_diameter=ilum_result.model_params.get("min_diameter", 0.0),
        flash_end_ms=flash_end_ms
    )

    if config["force_full_analysis_if_poor_fit"]:
        error_margin = config["error_margin_px"]
        if pre_result.rmse > 3 * error_margin:
            pre_result.warnings.append("full_analysis_triggered")
        if ilum_result.rmse > 3 * error_margin:
            ilum_result.warnings.append("full_analysis_triggered")
        if post_result.rmse > 3 * error_margin:
            post_result.warnings.append("full_analysis_triggered")

    metrics = compute_metrics(
        pre_result, ilum_result, post_result,
        error_margin_px=config["error_margin_px"],
        flash_end_ms=flash_end_ms,
        total_frames=total_frames
    )

    plot_path = None
    if config["save_plots"]:
        try:
            plot_path = plot_results(
                pre_result, ilum_result, post_result, metrics,
                flash_start_ms=flash_start_ms, flash_end_ms=flash_end_ms,
                output_dir=config["output_dir"], save_plots=True
            )
        except Exception:
            pass

    csv_path = None
    if config["save_csv"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        csv_path = os.path.join(config["output_dir"], "adaptive_analysis_results.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'timestamp_ms', 'diameter_px',
                             'confidence', 'is_interpolated', 'phase'])
            for s in pre_result.samples:
                writer.writerow([s.frame_index, s.timestamp_ms, s.diameter_px,
                                 s.confidence, s.is_interpolated, 'pre'])
            for s in ilum_result.samples:
                writer.writerow([s.frame_index, s.timestamp_ms, s.diameter_px,
                                 s.confidence, s.is_interpolated, 'ilum'])
            for s in post_result.samples:
                writer.writerow([s.frame_index, s.timestamp_ms, s.diameter_px,
                                 s.confidence, s.is_interpolated, 'post'])

    all_results = {
        "pre_result": pre_result,
        "ilum_result": ilum_result,
        "post_result": post_result,
        "metrics": metrics,
        "plot_path": plot_path,
        "csv_path": csv_path,
        "phase_boundaries": phase_boundaries,
        "flash_start_ms": flash_start_ms,
        "flash_end_ms": flash_end_ms,
    }

    return metrics, all_results
