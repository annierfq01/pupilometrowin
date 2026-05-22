"""
pupillometry.py - Clinical Pupillometry Analysis
=================================================
Calculos de pupilometria clinica.

Indices implementados:
  - Dominio Espacial: Size, MIN, Amplitud, CH (%)
  - Dominio Temporal: LAT, CV, MCV, DV, T75
  - Avanzados: NPi, PIPR
  - Estimacion de frames indeterminados
"""

import numpy as np
import csv
import math
from typing import List, Optional, Dict, Tuple


def estimate_indeterminate_frames(
    frame_results: List[dict],
    indeterminate_indices: set,
    timestamps: Optional[List[float]] = None
) -> List[dict]:
    """
    Estima los diametros y posiciones de centros para frames indeterminados
    usando interpolacion分段 (piecewise) basada en frames validos cercanos.
    
    El algoritmo:
    1. Identifica secuencias de frames indeterminados
    2. Para cada secuencia, toma los frames validos anteriores y posteriores
    3. Calcula una curva de transicion gradual basada en la posicion relativa
    4. Asigna valores estimados a cada frame indeterminado de la secuencia
    
    Args:
        frame_results: Lista de diccionarios con pupil_radius, pupil_center, etc.
        indeterminate_indices: Conjunto de indices de frames indeterminados
        timestamps: Timestamps opcionales para calculos de tiempo
        
    Returns:
        Lista de diccionarios con valores estimados para frames indeterminados
        (los frames no indeterminados devuelven None en el dict 'estimated')
    """
    n = len(frame_results)
    if n == 0:
        return []
    
    # Identificar secuencias de frames indeterminados
    indeterminate_list = sorted(indeterminate_indices)
    sequences = []
    current_seq = []
    
    for idx in indeterminate_list:
        if not current_seq:
            current_seq = [idx]
        elif idx == current_seq[-1] + 1:
            current_seq.append(idx)
        else:
            sequences.append(current_seq)
            current_seq = [idx]
    if current_seq:
        sequences.append(current_seq)
    
    # Preparar resultado
    estimated_results = [None] * n
    
    # Extraer datos validos (no indeterminados)
    valid_data = []
    for i in range(n):
        if i not in indeterminate_indices:
            res = frame_results[i]
            if res and res.get('pupil_radius') is not None:
                t = timestamps[i] if timestamps else i
                valid_data.append({
                    'idx': i,
                    'time': t,
                    'pupil_radius': res.get('pupil_radius', 0),
                    'pupil_center': res.get('pupil_center', (0, 0)),
                    'iris_radius': res.get('iris_radius', 0),
                    'iris_center': res.get('iris_center', (0, 0)),
                })
    
    if len(valid_data) < 2:
        # No hay suficientes datos para interpolar
        return [None] * n
    
    # Para cada secuencia de indeterminados, calcular valores estimados
    for seq in sequences:
        seq_start = seq[0]
        seq_end = seq[-1]
        seq_len = len(seq)
        
        # Encontrar frame valido anterior mas cercano
        prev_valid = None
        for i in range(seq_start - 1, -1, -1):
            if i not in indeterminate_indices and frame_results[i] and frame_results[i].get('pupil_radius') is not None:
                prev_valid = {
                    'idx': i,
                    'time': timestamps[i] if timestamps else i,
                    'pupil_radius': frame_results[i].get('pupil_radius', 0),
                    'pupil_center': frame_results[i].get('pupil_center', (0, 0)),
                    'iris_radius': frame_results[i].get('iris_radius', 0),
                    'iris_center': frame_results[i].get('iris_center', (0, 0)),
                }
                break
        
        # Encontrar frame valido posterior mas cercano
        next_valid = None
        for i in range(seq_end + 1, n):
            if i not in indeterminate_indices and frame_results[i] and frame_results[i].get('pupil_radius') is not None:
                next_valid = {
                    'idx': i,
                    'time': timestamps[i] if timestamps else i,
                    'pupil_radius': frame_results[i].get('pupil_radius', 0),
                    'pupil_center': frame_results[i].get('pupil_center', (0, 0)),
                    'iris_radius': frame_results[i].get('iris_radius', 0),
                    'iris_center': frame_results[i].get('iris_center', (0, 0)),
                }
                break
        
        # Si no hay frames validos en alguna direccion, usar el unico disponible
        if prev_valid is None and next_valid is None:
            continue
        
        # Estimar valores para cada frame en la secuencia
        for i, frame_idx in enumerate(seq):
            # Posicion relativa dentro de la secuencia (0 a 1)
            rel_pos = (i + 1) / (seq_len + 1) if seq_len > 0 else 0.5
            
            # Interpolar diametro de pupila
            if prev_valid and next_valid:
                # Interpolar entre ambos
                pr = _interpolate_value(
                    prev_valid['pupil_radius'], next_valid['pupil_radius'],
                    prev_valid['idx'], next_valid['idx'], frame_idx
                )
                ir = _interpolate_value(
                    prev_valid['iris_radius'], next_valid['iris_radius'],
                    prev_valid['idx'], next_valid['idx'], frame_idx
                )
                cx = _interpolate_value(
                    prev_valid['pupil_center'][0], next_valid['pupil_center'][0],
                    prev_valid['idx'], next_valid['idx'], frame_idx
                )
                cy = _interpolate_value(
                    prev_valid['pupil_center'][1], next_valid['pupil_center'][1],
                    prev_valid['idx'], next_valid['idx'], frame_idx
                )
            elif prev_valid:
                # Usar valor anterior
                pr = prev_valid['pupil_radius']
                ir = prev_valid['iris_radius']
                cx = prev_valid['pupil_center'][0]
                cy = prev_valid['pupil_center'][1]
            else:
                # Usar valor posterior
                pr = next_valid['pupil_radius']
                ir = next_valid['iris_radius']
                cx = next_valid['pupil_center'][0]
                cy = next_valid['pupil_center'][1]
            
            # Crear resultado estimado
            original_result = frame_results[frame_idx] or {}
            estimated_results[frame_idx] = {
                'pupil_center': (int(cx), int(cy)),
                'pupil_radius': pr,
                'iris_center': (int(cx), int(cy)),
                'iris_radius': ir,
                'px_to_mm': original_result.get('px_to_mm'),
                'estimated': True,
                'estimation_method': 'interpolation',
                'sequence_start': seq_start,
                'sequence_end': seq_end,
                'position_in_sequence': i + 1,
                'sequence_length': seq_len,
            }
    
    return estimated_results


def _interpolate_value(v1: float, v2: float, idx1: int, idx2: int, idx: int) -> float:
    """
    Interpola linealmente un valor entre dos puntos.
    
    Args:
        v1: Valor en el punto 1
        v2: Valor en el punto 2
        idx1: Indice del punto 1
        idx2: Indice del punto 2
        idx: Indice donde interpolar
        
    Returns:
        Valor interpolado
    """
    if idx2 == idx1:
        return v1
    
    # Interpolacion lineal
    t = (idx - idx1) / (idx2 - idx1)
    return v1 + t * (v2 - v1)


def merge_estimated_results(
    frame_results: List[dict],
    estimated_results: List[dict]
) -> List[dict]:
    """
    Combina los resultados originales con los estimados.
    Los frames no estimados mantienen sus valores originales.
    
    Args:
        frame_results: Resultados originales
        estimated_results: Resultados estimados del algoritmo
        
    Returns:
        Lista combinada con valores originales y estimados
    """
    merged = []
    for i, est in enumerate(estimated_results):
        if est is not None:
            # Usar resultado estimado
            merged.append(est)
        elif i < len(frame_results):
            # Usar resultado original
            orig = frame_results[i] or {}
            merged.append({
                'pupil_center': orig.get('pupil_center', (0, 0)),
                'pupil_radius': orig.get('pupil_radius', 0),
                'iris_center': orig.get('iris_center', (0, 0)),
                'iris_radius': orig.get('iris_radius', 0),
                'px_to_mm': orig.get('px_to_mm'),
                'estimated': False,
            })
        else:
            merged.append(None)
    return merged


def compute(frame_results: list,
            fps: float,
            i0: int, i1: int,
            light_start_frac: float = 0.0,
            light_end_frac: float = 1.0,
            px_to_mm_override: Optional[float] = None,
            timestamps: Optional[list] = None,
            fases: Optional[List[dict]] = None) -> dict:
    """
    Calcula parametros pupilometricos clinicos en el rango [i0, i1].

    Args:
        frame_results: Lista de dicts con pupil_radius, iris_radius, px_to_mm
        fps: FPS efectivo del video
        i0, i1: Indices de frame inicio y fin (inclusivo)
        light_start_frac: Fraccion donde inicia la iluminacion (0-1)
        light_end_frac: Fraccion donde termina la iluminacion (0-1)
        px_to_mm_override: Escala en mm/pixel
        timestamps: Timestamps opcionales
        fases: Configuracion de fases opcional

    Returns:
        Dict con todos los indices clinicos
    """
    times, p_diam, i_diam, ratio = [], [], [], []
    unit = 'px'

    for i in range(i0, i1 + 1):
        res = frame_results[i] if i < len(frame_results) else None
        if not res:
            continue
        pr = res.get('pupil_radius')
        ir = res.get('iris_radius')
        if pr is None:
            continue

        if timestamps and i - i0 < len(timestamps):
            t = timestamps[i - i0]
        else:
            t = i / fps
        px2mm = px_to_mm_override or res.get('px_to_mm')

        if px2mm:
            unit = 'mm'
            p_diam.append(pr * 2 * px2mm)
            i_diam.append(ir * 2 * px2mm if ir else None)
        else:
            p_diam.append(float(pr * 2))
            i_diam.append(float(ir * 2) if ir else None)

        times.append(t)
        ratio.append(pr / ir * 100 if ir and ir > 0 else None)

    if len(times) < 3:
        return {}

    p_arr = np.array(p_diam, dtype=float)
    t_arr = np.array(times,  dtype=float)
    n_frames = len(t_arr)

    kernel_size = min(5, len(p_arr) if len(p_arr) % 2 == 1 else len(p_arr) - 1)
    if kernel_size >= 3:
        p_arr = np.convolve(p_arr, np.ones(kernel_size) / kernel_size, mode='same')
    p_arr = np.nan_to_num(p_arr)

    p_max  = float(np.nanmax(p_arr))
    p_min  = float(np.nanmin(p_arr))
    p_mean = float(np.nanmean(p_arr))
    amplitude = p_max - p_min
    ch = (amplitude / p_max * 100) if p_max > 0 else 0.0

    idx_max = int(np.argmax(p_arr))
    idx_min = int(np.argmin(p_arr))
    
    dp = np.diff(p_arr)
    constr_start_idx = 0
    constr_threshold = -np.std(dp) * 0.5
    for j in range(len(dp)):
        if dp[j] < constr_threshold:
            constr_start_idx = j
            break
    
    lat = float(t_arr[idx_min] - t_arr[constr_start_idx]) if constr_start_idx < idx_min else 0.0
    time_to_min = float(t_arr[idx_min] - t_arr[0])

    dt = np.gradient(t_arr)
    dp_dt = np.zeros_like(p_arr)
    for j in range(len(p_arr) - 1):
        dp_dt[j] = (p_arr[j + 1] - p_arr[j]) / dt[j] if dt[j] != 0 else 0
    dp_dt[-1] = dp_dt[-2] if len(dp_dt) > 1 else 0
    
    win = min(5, len(dp_dt) if len(dp_dt) % 2 == 1 else len(dp_dt) - 1)
    dp_s = np.convolve(dp_dt, np.ones(win) / win, mode='same') if win >= 3 else dp_dt

    c_mask = dp_s < 0
    v_constr_mean = float(np.mean(dp_s[c_mask])) if c_mask.any() else 0.0
    v_constr_max  = float(np.min(dp_s))

    d_mask = dp_s > 0
    v_dilat_mean = float(np.mean(dp_s[d_mask])) if d_mask.any() else 0.0
    v_dilat_max  = float(np.max(dp_s))

    t75 = _compute_t75(p_arr, t_arr, p_max)
    npi = _compute_npi(lat, abs(v_constr_mean), abs(v_constr_max), ch)

    light_start_idx = int(light_start_frac * (n_frames - 1))
    light_end_idx   = int(light_end_frac   * (n_frames - 1))
    pipr = _compute_pIPR(p_arr, light_start_idx, light_end_idx)

    rc = [x for x in ratio if x is not None]
    r_mean = float(np.mean(rc)) if rc else float('nan')
    r_min  = float(np.min(rc))  if rc else float('nan')
    r_max_ = float(np.max(rc))  if rc else float('nan')

    fases_data = {}
    if fases:
        for fase in fases:
            nombre = fase['nombre']
            t_inicio = fase['t_inicio']
            t_fin = fase['t_fin']
            
            idx_inicio = 0
            idx_fin = n_frames - 1
            for j in range(n_frames):
                if t_arr[j] >= t_inicio:
                    idx_inicio = j
                    break
            for j in range(n_frames - 1, -1, -1):
                if t_arr[j] <= t_fin:
                    idx_fin = j
                    break
            
            if idx_fin > idx_inicio:
                fase_t = t_arr[idx_inicio:idx_fin+1]
                fase_p = p_arr[idx_inicio:idx_fin+1]
                fase_ratio = ratio[idx_inicio:idx_fin+1]
                
                fase_size = float(np.nanmax(fase_p))
                fase_min = float(np.nanmin(fase_p))
                fase_amplitude = fase_size - fase_min
                fase_ch = (fase_amplitude / fase_size * 100) if fase_size > 0 else 0.0
                
                fase_rc = [x for x in fase_ratio if x is not None]
                fase_r_mean = float(np.mean(fase_rc)) if fase_rc else float('nan')
                fase_r_min = float(np.min(fase_rc)) if fase_rc else float('nan')
                fase_r_max = float(np.max(fase_rc)) if fase_rc else float('nan')
                
                fases_data[nombre] = {
                    't_inicio': t_inicio, 't_fin': t_fin,
                    'idx_inicio': idx_inicio, 'idx_fin': idx_fin,
                    'n_frames': idx_fin - idx_inicio + 1,
                    'size': fase_size, 'p_min': fase_min,
                    'amplitude': fase_amplitude, 'ch': fase_ch,
                    'r_mean': fase_r_mean, 'r_min': fase_r_min, 'r_max': fase_r_max,
                }

    return {
        'times': list(t_arr), 'p_diam': list(p_arr), 'i_diam': i_diam,
        'ratio': ratio, 'dp_s': list(dp_s), 'unit': unit,
        'size': p_max, 'p_min': p_min, 'amplitude': amplitude, 'ch': ch,
        'lat': lat, 'time_to_min': time_to_min,
        'cv': v_constr_mean, 'mcv': v_constr_max,
        'dv': v_dilat_mean, 'dv_max': v_dilat_max,
        't75': t75, 'npi': npi, 'pipr': pipr,
        'r_mean': r_mean, 'r_min': r_min, 'r_max': r_max_,
        'fases': fases_data,
    }


def _compute_t75(p_arr: np.ndarray, t_arr: np.ndarray, p_max: float) -> float:
    """Calcula el tiempo de redilatacion al 75% del diametro inicial."""
    if len(p_arr) < 2:
        return float('nan')

    target_75 = p_max * 0.75
    idx_min   = int(np.argmin(p_arr))

    if p_arr[idx_min] >= target_75:
        return 0.0

    for i in range(idx_min, len(p_arr) - 1):
        if p_arr[i] < target_75 <= p_arr[i + 1]:
            frac = (target_75 - p_arr[i]) / (p_arr[i + 1] - p_arr[i])
            t_cross = t_arr[i] + frac * (t_arr[i + 1] - t_arr[i])
            return float(t_cross - t_arr[0])

    return float('nan')


def _compute_npi(lat: float, cv: float, mcv: float, ch: float) -> float:
    """
    Indice Neurologico de Pupila (NPi) - aproximacion.
    
    Escala 0-5 donde:
    - Latencia baja = bueno
    - Velocidad alta = bueno
    - CH alto = bueno
    """
    lat_score  = max(0, min(1, 1 - lat / 1.0))
    cv_score   = max(0, min(1, abs(cv) / 5.0))
    mcv_score  = max(0, min(1, abs(mcv) / 10.0))
    ch_score   = max(0, min(1, ch / 50.0))
    
    npi_raw = (lat_score * 0.25 + (cv_score + mcv_score) * 0.25 + ch_score * 0.25)
    npi = npi_raw * 5
    
    return round(npi, 2)


def _compute_pIPR(p_arr: np.ndarray, light_start: int, light_end: int) -> float:
    """Post-Illumination Pupil Response."""
    if light_end >= len(p_arr) - 1:
        return float('nan')
    
    light_phase = p_arr[light_start:light_end + 1]
    light_mean = np.mean(light_phase) if len(light_phase) > 0 else p_arr[0]
    
    post_phase = p_arr[light_end:]
    post_mean = np.mean(post_phase) if len(post_phase) > 0 else p_arr[-1]
    
    if light_mean > 0:
        pipr = post_mean / light_mean
    else:
        pipr = 1.0
    
    return float(round(pipr, 3))


def format_report(data: dict) -> str:
    """Formatea los resultados como texto."""
    if not data:
        return 'Datos insuficientes para calcular pupilometria.'
    u  = data['unit']
    t0 = data['times'][0]  if data['times'] else 0
    t1 = data['times'][-1] if data['times'] else 0

    def _t75():
        v = data.get('t75', float('nan'))
        return f"{v:.3f} s" if not math.isnan(v) else "N/A"

    def _pipr():
        v = data.get('pipr', float('nan'))
        return f"{v:.3f}" if not math.isnan(v) else "N/A"

    lines = [
        "PUPILOMETRIA CLINICA",
        "=" * 50,
        f"Rango:   {t0:.3f} s  ->  {t1:.3f} s",
        f"Frames:  {len(data['times'])}     Unidad: {u}",
        "",
        "-- PARAMETROS ESPACIALES (tamano) --",
        f"  Size  (diam. max. / reposo) : {data['size']:.3f} {u}",
        f"  MIN   (diam. minimo)        : {data['p_min']:.3f} {u}",
        f"  Amplitud (Size - MIN)       : {data['amplitude']:.3f} {u}",
        f"  CH    (cambio %)            : {data['ch']:.2f} %",
        "",
        "-- PARAMETROS TEMPORALES ------------",
        f"  LAT       (latencia constr.) : {data['lat']:.3f} s",
        f"  T_min     (tiempo al minimo) : {data['time_to_min']:.3f} s",
        f"  CV        (vel. media constr): {data['cv']:+.3f} {u}/s",
        f"  MCV       (vel. max constr)  : {data['mcv']:+.3f} {u}/s",
        f"  DV        (vel. media dilat): {data['dv']:+.3f} {u}/s",
        f"  DV_max    (vel. max dilat)   : {data['dv_max']:+.3f} {u}/s",
        f"  T75       (redilat. 75 %)   : {_t75()}",
        "",
        "-- INDICES AVANZADOS ---------------",
        f"  NPi  (Ind. Neurol. Pupila)  : {data['npi']:.2f} / 5.0",
        f"  PIPR (resp. post-iluminac.)   : {_pipr()}",
        "",
        "-- RATIO PUPILA / IRIS ------------",
        f"  Media  : {data['r_mean']:.2f} %",
        f"  Minimo : {data['r_min']:.2f} %",
        f"  Maximo : {data['r_max']:.2f} %",
    ]
    
    fases_data = data.get('fases', {})
    if fases_data:
        lines.append("")
        lines.append("-- POR FASES ------------------------")
        for nombre in ['Basal', 'Contraccion', 'Relajacion']:
            if nombre in fases_data:
                f = fases_data[nombre]
                lines.extend([
                    f"  {nombre}:",
                    f"    Tiempo: {f['t_inicio']:.2f}s - {f['t_fin']:.2f}s",
                    f"    Frames: {f['n_frames']}",
                    f"    Size: {f['size']:.3f} {u}",
                    f"    MIN: {f['p_min']:.3f} {u}",
                    f"    Amplitud: {f['amplitude']:.3f} {u}",
                    f"    CH: {f['ch']:.2f} %",
                    f"    Ratio media: {f['r_mean']:.2f} %",
                ])
    
    return '\n'.join(lines)


def export_csv(data: dict, path: str, include_estimated: bool = True):
    """Exporta la serie temporal a un CSV con todos los indices.
    
    Args:
        data: Datos de pupilometria
        path: Ruta del archivo CSV
        include_estimated: Si True, incluye marca para frames estimados
    """
    if not data:
        return
    u = data['unit']

    def _fmt(v, decimals=4):
        if isinstance(v, float) and math.isnan(v):
            return 'N/A'
        return f'{v:.{decimals}f}'

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)

        w.writerow(['# PUPILOMETRIA CLINICA'])
        w.writerow([f'# Unidad: {u}'])
        w.writerow([f'# Frames: {len(data["times"])}'])
        w.writerow([f'# Rango:  {data["times"][0]:.3f} s  ->  {data["times"][-1]:.3f} s'])
        
        # Info sobre frames estimados
        if include_estimated and 'estimated_frames' in data:
            est_count = len(data['estimated_frames'])
            w.writerow([f'# Frames estimados: {est_count}'])
        
        # Info de calidad de reduccion de fps si existe
        if 'fps_reduction' in data:
            fr = data['fps_reduction']
            w.writerow([f'# FPS original: {fr.get("original_fps", "N/A")}'])
            w.writerow([f'# FPS efectivo: {fr.get("effective_fps", "N/A")}'])
            w.writerow([f'# Calidad minima达标: {fr.get("quality_met", "N/A")}'])
        
        w.writerow([])

        w.writerow(['# INDICES ESPACIALES'])
        w.writerow(['Indice', 'Valor', 'Unidad', 'Descripcion'])
        w.writerow(['Size',      _fmt(data['size']),      u,   'Diametro maximo (reposo)'])
        w.writerow(['MIN',       _fmt(data['p_min']),     u,   'Diametro minimo (pico constriccion)'])
        w.writerow(['Amplitud',  _fmt(data['amplitude']), u,   'Size - MIN'])
        w.writerow(['CH',        _fmt(data['ch'], 2),     '%', 'Cambio porcentual (Amplitud/Size*100)'])
        w.writerow([])

        w.writerow(['# INDICES TEMPORALES'])
        w.writerow(['Indice', 'Valor', 'Unidad', 'Descripcion'])
        w.writerow(['LAT',    _fmt(data['lat']),         's',       'Latencia de constriccion'])
        w.writerow(['T_min',  _fmt(data['time_to_min']), 's',       'Tiempo hasta diametro minimo'])
        w.writerow(['CV',     _fmt(data['cv']),          f'{u}/s',  'Velocidad media de constriccion'])
        w.writerow(['MCV',    _fmt(data['mcv']),         f'{u}/s',  'Velocidad maxima de constriccion'])
        w.writerow(['DV',     _fmt(data['dv']),          f'{u}/s',  'Velocidad media de dilatacion'])
        w.writerow(['DV_max', _fmt(data['dv_max']),      f'{u}/s',  'Velocidad maxima de dilatacion'])
        w.writerow(['T75',    _fmt(data['t75']),         's',       'Tiempo para recuperar 75% del diametro'])
        w.writerow([])

        w.writerow(['# INDICES AVANZADOS'])
        w.writerow(['Indice', 'Valor', 'Unidad', 'Descripcion'])
        w.writerow(['NPi',  _fmt(data['npi'],  2), '/5.0',  'Indice Neurologico de Pupila'])
        w.writerow(['PIPR', _fmt(data['pipr'], 3), 'ratio', 'Post-Illumination Pupil Response'])
        w.writerow([])

        w.writerow(['# RATIO PUPILA / IRIS'])
        w.writerow(['Indice', 'Valor', 'Unidad'])
        w.writerow(['Ratio_media',  _fmt(data['r_mean'], 2), '%'])
        w.writerow(['Ratio_minimo', _fmt(data['r_min'],  2), '%'])
        w.writerow(['Ratio_maximo', _fmt(data['r_max'],  2), '%'])
        w.writerow([])

        fases_data = data.get('fases', {})
        if fases_data:
            w.writerow(['# DATOS POR FASE'])
            w.writerow(['Fase', 'Tiempo_inicio_s', 'Tiempo_fin_s', 'Frames', 
                       'Size', 'MIN', 'Amplitud', 'CH_%', 'Ratio_media_%'])
            for nombre in ['Basal', 'Contraccion', 'Relajacion']:
                if nombre in fases_data:
                    f = fases_data[nombre]
                    w.writerow([
                        nombre,
                        _fmt(f['t_inicio'], 3),
                        _fmt(f['t_fin'], 3),
                        str(f['n_frames']),
                        _fmt(f['size']),
                        _fmt(f['p_min']),
                        _fmt(f['amplitude']),
                        _fmt(f['ch'], 2),
                        _fmt(f['r_mean'], 2),
                    ])
            w.writerow([])

        # Serie temporal con columnas adicionales si hay frames estimados
        if include_estimated and 'estimated_flags' in data:
            w.writerow(['# SERIE TEMPORAL'])
            w.writerow([f'tiempo_s',
                        f'diametro_pupila_{u}',
                        f'diametro_iris_{u}',
                        'ratio_pupila_iris_%',
                        f'velocidad_{u}_s',
                        'estimado',
                        'centro_x',
                        'centro_y'])
            for i, t in enumerate(data['times']):
                pd_ = _fmt(data['p_diam'][i]) if i < len(data['p_diam']) else ''
                id_ = (_fmt(data['i_diam'][i])
                       if i < len(data['i_diam']) and data['i_diam'][i] is not None else '')
                rt  = (_fmt(data['ratio'][i], 2)
                       if i < len(data['ratio'])  and data['ratio'][i]  is not None else '')
                dp  = _fmt(data['dp_s'][i]) if i < len(data['dp_s']) else ''
                est = data['estimated_flags'][i] if i < len(data['estimated_flags']) else ''
                cx = data['centers_x'][i] if i < len(data['centers_x']) else ''
                cy = data['centers_y'][i] if i < len(data['centers_y']) else ''
                w.writerow([f'{t:.4f}', pd_, id_, rt, dp, est, cx, cy])
        else:
            w.writerow(['# SERIE TEMPORAL'])
            w.writerow([f'tiempo_s',
                        f'diametro_pupila_{u}',
                        f'diametro_iris_{u}',
                        'ratio_pupila_iris_%',
                        f'velocidad_{u}_s'])
            for i, t in enumerate(data['times']):
                pd_ = _fmt(data['p_diam'][i]) if i < len(data['p_diam']) else ''
                id_ = (_fmt(data['i_diam'][i])
                       if i < len(data['i_diam']) and data['i_diam'][i] is not None else '')
                rt  = (_fmt(data['ratio'][i], 2)
                       if i < len(data['ratio'])  and data['ratio'][i]  is not None else '')
                dp  = _fmt(data['dp_s'][i]) if i < len(data['dp_s']) else ''
                w.writerow([f'{t:.4f}', pd_, id_, rt, dp])
