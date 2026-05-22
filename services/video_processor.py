"""
video_processor.py - Optimized Video Processing
===============================================
Streaming video processor for efficient frame extraction.

ALGORITHMO DE PROCESAMIENTO DE VIDEO:
=====================================

El algoritmo esta diseñado para ser eficiente en memoria y procesamiento,
evitando cargar todos los frames en memoria simultáneamente.

PASO 1: OBTENER INFO DEL VIDEO (sin cargar frames)
--------------------------------------------------
- Leer FPS, total_frames, duracion total, resolucion
- No se cargan frames en memoria

PASO 2: CALCULAR LIMITES DE FASES
---------------------------------
- Fase 1 (Basal): [0, fase1_dur]
- Fase 2 (Iluminacion): [t1_end, t1_end + fase2_dur]
- Fase 3 (Relajacion): [t2_end, t2_end + fase3_dur]

SI deteccion de flash habilitada:
- Ejecutar deteccion de flash (streaming)
- Obtener flash_on_time, flash_off_time
- Ajustar limites de fases segun tiempos de flash
- Fase 1: [0, flash_on - trim_before]
- Fase 2: [flash_on, flash_off] o [flash_on, flash_on + fase2_dur]
- Fase 3: [t2_end, t2_end + fase3_dur + trim_after]

PASO 3: CALCULAR FRAMES REQUERIDOS POR FASE
-------------------------------------------
- Fase 1: ceil(fase1_dur * min(fase1_fps, src_fps))
- Fase 2: ceil(fase2_dur * min(fase2_fps, src_fps))
- Fase 3: ceil(fase3_dur * min(fase3_fps, src_fps))

PASO 4: PROCESAR VIDEO EN STREAMING
------------------------------------
- Procesar frames uno por uno sin almacenar todos
- Para cada frame, determinar a que fase pertenece
- Mantener frame si esta cerca del timestamp objetivo para esa fase
- Usar resampling nearest-neighbor para evitar aliasing
- Detener cuando se recolecten todos los frames necesarios

PASO 5: FINALIZAR
------------------
- Almacenar solo los frames seleccionados
- Almacenar sus timestamps originales (para pupillometria)
- Calcular FPS efectivo

COMPATIBILIDAD CON PUPILOMETRIA:
================================
- Los timestamps se mantienen en segundos desde el inicio del video
- El FPS efectivo se calcula como: n_frames / duracion
- Los calculos de pupillometria usan timestamps originales
- No se alteran las posiciones relative de los frames

Autor: Pupilometer Development Team
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, NamedTuple
import math


class VideoInfo(NamedTuple):
    """Informacion basica del video."""
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


class PhaseBounds(NamedTuple):
    """Limites de una fase en segundos."""
    start: float
    end: float
    target_fps: int


class ProcessedVideo:
    """
    Contenedor para el video procesado.
    
    Attributes:
        frames: Lista de frames seleccionados
        timestamps: Timestamps en segundos de cada frame
        fps: FPS efectivo
        phase_bounds: Lista de PhaseBounds por fase
        video_info: Informacion del video original
        flash_times: Tiempos de flash detectados (si hubo)
    """
    
    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self.fps: float = 30.0
        self.phase_bounds: List[PhaseBounds] = []
        self.video_info: Optional[VideoInfo] = None
        self.flash_times: Optional[Dict[str, Any]] = None
        self._frames_by_phase: Dict[int, List[int]] = {}
    
    @property
    def frame_count(self) -> int:
        """Numero de frames."""
        return len(self.frames)
    
    @property
    def duration(self) -> float:
        """Duracion efectiva en segundos."""
        if len(self.timestamps) > 1:
            return self.timestamps[-1] - self.timestamps[0]
        return 0.0
    
    def get_frames_by_phase(self, phase: int) -> List[Tuple[int, np.ndarray, float]]:
        """
        Obtiene frames de una fase especifica.
        
        Args:
            phase: Numero de fase (1, 2, o 3)
            
        Returns:
            Lista de (indice, frame, timestamp)
        """
        if phase not in self._frames_by_phase:
            return []
        return [(i, self.frames[i], self.timestamps[i]) 
                for i in self._frames_by_phase[phase]]
    
    def _assign_frames_to_phases(self):
        """Asigna frames a sus fases basadas en timestamps."""
        self._frames_by_phase = {1: [], 2: [], 3: []}
        
        for i, ts in enumerate(self.timestamps):
            if not self.phase_bounds:
                self._frames_by_phase[1].append(i)
                continue
                
            assigned = False
            for phase_idx, bounds in enumerate(self.phase_bounds, 1):
                if bounds.start <= ts < bounds.end:
                    self._frames_by_phase[phase_idx].append(i)
                    assigned = True
                    break
            
            if not assigned and self.phase_bounds:
                last_phase = len(self.phase_bounds)
                self._frames_by_phase[last_phase].append(i)


class VideoProcessor:
    """
    Procesador de video optimizado con streaming.
    
    Procesa videos de manera eficiente en memoria, seleccionando
    frames segun la configuracion de fases y FPS objetivo.
    
    Example:
        >>> processor = VideoProcessor()
        >>> result = processor.process(
        ...     video_path='video.mp4',
        ...     phases={
        ...         'basal': {'fps': 2, 'duration': 3},
        ...         'contraccion': {'fps': 30, 'duration': 2},
        ...         'relajacion': {'fps': 10, 'duration': 7}
        ...     },
        ...     flash_config={'auto': True, 'threshold': 5.0}
        ... )
        >>> print(f"Frames: {result.frame_count}, FPS: {result.fps}")
    """
    
    def __init__(self):
        """Inicializa el procesador."""
        self._video_info: Optional[VideoInfo] = None
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Obtiene informacion del video sin cargar frames.
        
        Args:
            video_path: Ruta al video
            
        Returns:
            VideoInfo con fps, total_frames, duracion, resolucion
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
        
        cap.release()
        
        return VideoInfo(
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            width=width,
            height=height
        )
    
    def calculate_phase_bounds(
        self,
        video_duration: float,
        phases_config: Dict[str, Dict[str, Any]],
        flash_times: Optional[Dict[str, Any]] = None,
        trim_before_ms: int = 100,
        trim_after_ms: int = 100
    ) -> Tuple[List[PhaseBounds], Dict[str, float]]:
        """
        Calcula los limites de cada fase en segundos.
        
        Args:
            video_duration: Duracion total del video en segundos
            phases_config: Configuracion de fases
            flash_times: Tiempos de flash detectados (si hay)
            trim_before_ms: Ms a recortar antes del flash
            trim_after_ms: Ms a recortar despues del flash
            
        Returns:
            Tupla de (lista de PhaseBounds, limites de iluminacion)
        """
        fase1_dur = phases_config.get('basal', {}).get('duration', 3)
        fase2_dur = phases_config.get('contraccion', {}).get('duration', 2)
        fase3_dur = phases_config.get('relajacion', {}).get('duration', 7)
        
        fase1_fps = phases_config.get('basal', {}).get('fps', 2)
        fase2_fps = phases_config.get('contraccion', {}).get('fps', 30)
        fase3_fps = phases_config.get('relajacion', {}).get('fps', 10)
        
        bounds = []
        illum_start = 1.0
        illum_end = 1.2
        
        if flash_times and flash_times.get('on_ms') is not None:
            flash_on_sec = flash_times['on_ms'] / 1000.0
            flash_off_sec = (flash_times['off_ms'] / 1000.0 
                           if flash_times.get('off_ms') else None)
            
            trim_before = trim_before_ms / 1000.0
            trim_after = trim_after_ms / 1000.0
            
            t1_start = 0
            t1_end = max(0, flash_on_sec - trim_before)
            
            t2_start = flash_on_sec
            t2_end = flash_off_sec if flash_off_sec else flash_on_sec + fase2_dur
            
            t3_start = t2_end
            t3_end = min(video_duration, t3_start + fase3_dur + trim_after)
            
            illum_start = t2_start
            illum_end = t2_end
            
        else:
            required_duration = fase1_dur + fase2_dur + fase3_dur
            
            if video_duration < required_duration:
                scale = video_duration / required_duration
                fase1_dur *= scale
                fase2_dur *= scale
                fase3_dur *= scale
            
            t1_start = 0
            t1_end = t1_start + fase1_dur
            t2_start = t1_end
            t2_end = t2_start + fase2_dur
            t3_start = t2_end
            t3_end = min(video_duration, t3_start + fase3_dur)
            
            illum_start = t2_start
            illum_end = t2_end
        
        bounds.append(PhaseBounds(t1_start, t1_end, fase1_fps))
        bounds.append(PhaseBounds(t2_start, t2_end, fase2_fps))
        bounds.append(PhaseBounds(t3_start, t3_end, fase3_fps))
        
        return bounds, {'start': illum_start, 'end': illum_end}
    
    def calculate_required_frames(
        self,
        phase_bounds: List[PhaseBounds],
        src_fps: float
    ) -> List[int]:
        """
        Calcula el numero de frames requeridos por fase.
        
        Args:
            phase_bounds: Limites de cada fase
            src_fps: FPS original del video
            
        Returns:
            Lista de frames requeridos por fase
        """
        required = []
        for bounds in phase_bounds:
            duration = bounds.end - bounds.start
            effective_fps = min(bounds.target_fps, src_fps)
            frames_needed = max(1, math.ceil(duration * effective_fps))
            required.append(frames_needed)
        return required
    
    def detect_flash_streaming(
        self,
        video_path: str,
        calibration_frames: int = 5,
        change_threshold: float = 5.0,
        smoothing_window: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detecta flash en video usando streaming (sin guardar frames).
        
        Args:
            video_path: Ruta al video
            calibration_frames: Frames para calibracion
            change_threshold: Umbral de cambio de brillo
            smoothing_window: Ventana de suavizado
            progress_callback: Callback de progreso
            
        Returns:
            Diccionario con flash_times o None si no se detecta
        """
        from pupilometer.analysis.flash_detection import FlashDetector
        
        detector = FlashDetector(
            calibration_frames=calibration_frames,
            change_threshold=change_threshold,
            smoothing_window=smoothing_window
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_ms = (frame_idx / fps) * 1000 if fps > 0 else 0
            detector.process_frame(frame, timestamp_ms)
            
            frame_idx += 1
            if progress_callback and frame_idx % 50 == 0:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        
        if detector.flash_on_time is not None:
            return {
                'on_ms': detector.flash_on_time,
                'off_ms': detector.flash_off_time,
                'baseline': detector.brightness_base
            }
        
        return None
    
    def process(
        self,
        video_path: str,
        phases_config: Dict[str, Dict[str, Any]],
        flash_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        detect_iris: bool = False
    ) -> ProcessedVideo:
        """
        Procesa el video completo con streaming optimizado.
        
        Args:
            video_path: Ruta al video
            phases_config: Configuracion de fases:
                {
                    'basal': {'fps': int, 'duration': float},
                    'contraccion': {'fps': int, 'duration': float},
                    'relajacion': {'fps': int, 'duration': float}
                }
            flash_config: Configuracion de deteccion de flash:
                {
                    'auto': bool,
                    'calibration_frames': int,
                    'change_threshold': float,
                    'smoothing_window': int,
                    'trim_before_ms': int,
                    'trim_after_ms': int
                }
            progress_callback: Callback(opcion, total) para progreso
            detect_iris: Si True, incluye frames para deteccion de iris
            
        Returns:
            ProcessedVideo con frames seleccionados y metadata
        """
        result = ProcessedVideo()
        
        result.video_info = self.get_video_info(video_path)
        video_duration = result.video_info.duration
        src_fps = result.video_info.fps
        
        if progress_callback:
            progress_callback(0, 100)
        
        flash_times = None
        if flash_config and flash_config.get('auto', False):
            if progress_callback:
                progress_callback(10, 100)
            flash_times = self.detect_flash_streaming(
                video_path,
                calibration_frames=flash_config.get('calibration_frames', 5),
                change_threshold=flash_config.get('change_threshold', 5.0),
                smoothing_window=flash_config.get('smoothing_window', 4)
            )
            result.flash_times = flash_times
        
        if progress_callback:
            progress_callback(20, 100)
        
        trim_before_ms = flash_config.get('trim_before_ms', 100) if flash_config else 100
        trim_after_ms = flash_config.get('trim_after_ms', 100) if flash_config else 100
        
        phase_bounds, illum_times = self.calculate_phase_bounds(
            video_duration, phases_config, flash_times,
            trim_before_ms, trim_after_ms
        )
        result.phase_bounds = phase_bounds
        
        required_frames = self.calculate_required_frames(phase_bounds, src_fps)
        
        if progress_callback:
            progress_callback(25, 100)
        
        selected_frames = []
        selected_timestamps = []
        frames_collected_per_phase = [0, 0, 0]
        
        target_timestamps = []
        for i, bounds in enumerate(phase_bounds):
            duration = bounds.end - bounds.start
            n_frames = required_frames[i]
            effective_fps = min(bounds.target_fps, src_fps)
            
            if n_frames <= 1:
                target_timestamps.append([bounds.start + duration / 2])
            else:
                interval = duration / (n_frames - 1) if n_frames > 1 else duration
                targets = [bounds.start + j * interval for j in range(n_frames)]
                target_timestamps.append(targets)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        total_vid_frames = result.video_info.total_frames
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_sec = frame_idx / src_fps if src_fps > 0 else 0
            
            for phase_idx, bounds in enumerate(phase_bounds):
                if bounds.start <= timestamp_sec < bounds.end:
                    phase_idx_0 = phase_idx
                    
                    if frames_collected_per_phase[phase_idx_0] < required_frames[phase_idx_0]:
                        targets = target_timestamps[phase_idx_0]
                        collected = frames_collected_per_phase[phase_idx_0]
                        
                        if collected >= len(targets):
                            break
                        
                        target_ts = targets[collected]
                        
                        if len(selected_timestamps) == 0:
                            selected_frames.append(frame)
                            selected_timestamps.append(timestamp_sec)
                            frames_collected_per_phase[phase_idx_0] += 1
                        else:
                            last_ts = selected_timestamps[-1]
                            interval = 1.0 / min(bounds.target_fps, src_fps)
                            
                            if timestamp_sec - last_ts >= interval * 0.9:
                                selected_frames.append(frame)
                                selected_timestamps.append(timestamp_sec)
                                frames_collected_per_phase[phase_idx_0] += 1
                    break
            
            frame_idx += 1
            
            if progress_callback and frame_idx % 100 == 0:
                pct = 25 + (frame_idx / total_vid_frames) * 70
                progress_callback(int(pct), 100)
        
        cap.release()
        
        result.frames = selected_frames
        result.timestamps = selected_timestamps
        result._assign_frames_to_phases()
        
        if len(result.timestamps) > 1:
            eff_dur = result.timestamps[-1] - result.timestamps[0]
            result.fps = len(result.frames) / eff_dur if eff_dur > 0 else src_fps
        else:
            result.fps = src_fps
        
        if progress_callback:
            progress_callback(100, 100)
        
        return result


def fast_video_load(
    video_path: str,
    phases_config: Dict[str, Dict[str, Any]],
    flash_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Funcion de conveniencia para cargar video rapidamente.
    
    Args:
        video_path: Ruta al video
        phases_config: Configuracion de fases
        flash_config: Configuracion de flash (opcional)
        progress_callback: Callback de progreso
        
    Returns:
        Diccionario compatible con la interfaz anterior:
        {
            'frames': list,
            'timestamps': list,
            'fps': float,
            'phase_bounds': list,
            'flash_times': dict,
            'video_info': VideoInfo
        }
    """
    processor = VideoProcessor()
    result = processor.process(
        video_path, phases_config, flash_config, progress_callback
    )
    
    return {
        'frames': result.frames,
        'timestamps': result.timestamps,
        'fps': result.fps,
        'phase_bounds': result.phase_bounds,
        'flash_times': result.flash_times,
        'video_info': result.video_info,
        'illumination': {
            'start': result.phase_bounds[1].start if len(result.phase_bounds) > 1 else 1.0,
            'end': result.phase_bounds[1].end if len(result.phase_bounds) > 1 else 1.2
        }
    }


__all__ = [
    'VideoInfo',
    'PhaseBounds',
    'ProcessedVideo',
    'VideoProcessor',
    'fast_video_load'
]
