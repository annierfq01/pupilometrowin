"""
video_service.py - Servicio de Video
====================================
Maneja la carga, procesamiento y navegacion de videos.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from pupilometer.analysis import detect_flash_in_video, should_reduce_fps


class VideoService:
    """
    Servicio para el manejo de videos.
    
    Encapsula toda la logica de carga, procesamiento y navegacion
    de frames de video.
    
    Attributes:
        frames: Lista de frames del video
        timestamps: Timestamps correspondientes a cada frame
        fps: FPS efectivo del video
        current_idx: Indice del frame actual
        
    Example:
        >>> service = VideoService()
        >>> service.load_video('video.mp4', phases_config)
        >>> frame = service.get_frame(0)
        >>> service.release()
    """
    
    def __init__(self):
        """Inicializa el servicio de video."""
        self.frames: List[np.ndarray] = []
        self.frames_crop: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self.fps: float = 30.0
        self.current_idx: int = 0
        self.is_loaded: bool = False
        self.crop_rect: Optional[Tuple[int, int, int, int]] = None
        
        self._src_fps: float = 30.0
        self._total_frames: int = 0
        
    @property
    def frame_count(self) -> int:
        """Retorna el numero de frames cargados."""
        return len(self.frames)
    
    @property
    def duration(self) -> float:
        """Retorna la duracion del video en segundos."""
        if self.timestamps and len(self.timestamps) > 1:
            return self.timestamps[-1] - self.timestamps[0]
        elif self.frames:
            return len(self.frames) / self.fps
        return 0.0
    
    def load_video(
        self,
        path: str,
        phases_config: Dict[str, Any],
        flash_config: Optional[Dict[str, Any]] = None,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Carga y procesa un video con las configuraciones de fases.
        
        Args:
            path: Ruta al archivo de video
            phases_config: Configuracion de fases con keys:
                - basal_fps, basal_duration
                - contraccion_fps, contraccion_duration
                - relajacion_fps, relajacion_duration
            flash_config: Configuracion de deteccion de flash (opcional)
            crop_rect: Rectangulo de recorte (x1, y1, x2, y2) (opcional)
            progress_callback: Callback(opcion, total) para progreso
            
        Returns:
            Diccionario con informacion del video cargado:
                - success: bool
                - frame_count: int
                - fps: float
                - duration: float
                - flash_times: dict con tiempos de flash (si se detecto)
                - error: str (si hubo error)
        """
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return {'success': False, 'error': 'No se pudo abrir el video'}
            
            self._src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            
            if not all_frames:
                return {'success': False, 'error': 'Video sin frames'}
            
            flash_times = None
            if flash_config and flash_config.get('auto', False) and len(all_frames) > 0:
                try:
                    flash_result = detect_flash_in_video(
                        path,
                        calibration_frames=flash_config.get('calibration_frames', 5),
                        change_threshold=flash_config.get('change_threshold', 5.0),
                        smoothing_window=flash_config.get('smoothing_window', 4)
                    )
                    if flash_result['flash_on_time_ms'] is not None:
                        flash_times = {
                            'on_ms': flash_result['flash_on_time_ms'],
                            'off_ms': flash_result['flash_off_time_ms'],
                            'baseline': flash_result['baseline_brightness']
                        }
                except Exception:
                    pass
            
            fase1_dur = phases_config.get('basal_duration', 3)
            fase2_dur = phases_config.get('contraccion_duration', 2)
            fase3_dur = phases_config.get('relajacion_duration', 7)
            
            fase1_fps = phases_config.get('basal_fps', 2)
            fase2_fps = phases_config.get('contraccion_fps', 30)
            fase3_fps = phases_config.get('relajacion_fps', 10)
            
            t1_start = 0
            t1_end = t1_start + fase1_dur
            t2_start = t1_end
            t2_end = t2_start + fase2_dur
            t3_start = t2_end
            t3_end = t3_start + fase3_dur
            
            if flash_times and flash_times['on_ms']:
                flash_on_sec = flash_times['on_ms'] / 1000.0
                flash_off_sec = flash_times['off_ms'] / 1000.0 if flash_times['off_ms'] else None
                
                if flash_on_sec > 0:
                    t1_start = 0
                    t1_end = flash_on_sec
                    t2_start = flash_on_sec
                    t2_end = flash_off_sec if flash_off_sec else flash_on_sec + fase2_dur
                    t3_start = t2_end
                    t3_end = t3_start + fase3_dur
            
            if len(all_frames) > 0:
                sample = all_frames[len(all_frames) // 2]
                h, w = sample.shape[:2]
                quality_resolution = min(h, w)
            else:
                quality_resolution = 0
            
            quality_threshold = 480
            fps_reduction_info = []
            selected_frames = []
            selected_timestamps = []
            
            src_fps = self._src_fps
            total_frames = len(all_frames)
            
            for i in range(total_frames):
                if progress_callback and i % 100 == 0:
                    progress_callback(i, total_frames)
                
                timestamp = i / src_fps
                
                if timestamp < t1_end:
                    target_fps = fase1_fps
                elif timestamp < t2_end:
                    target_fps = fase2_fps
                else:
                    target_fps = fase3_fps
                
                should_reduce, recommended_fps = should_reduce_fps(src_fps, target_fps, quality_threshold)
                
                if should_reduce and quality_resolution < quality_threshold:
                    actual_fps = src_fps
                    if fps_reduction_info:
                        fps_reduction_info.append(f'Baja calidad ({quality_resolution}p)')
                else:
                    actual_fps = min(target_fps, src_fps)
                
                frame_duration = 1.0 / actual_fps if actual_fps > 0 else 1.0
                if len(selected_timestamps) == 0 or (timestamp - selected_timestamps[-1]) >= frame_duration * 0.95:
                    selected_frames.append(all_frames[i])
                    selected_timestamps.append(timestamp)
            
            if not selected_frames:
                return {'success': False, 'error': 'Sin frames validos'}
            
            self.frames = selected_frames
            self.timestamps = selected_timestamps
            self.current_idx = 0
            self.crop_rect = crop_rect
            
            if len(self.timestamps) > 1:
                eff_dur = self.timestamps[-1] - self.timestamps[0]
                self.fps = len(self.frames) / eff_dur if eff_dur > 0 else self._src_fps
            else:
                self.fps = self._src_fps
            
            if self.crop_rect:
                x1, y1, x2, y2 = self.crop_rect
                self.frames_crop = [f[y1:y2, x1:x2] for f in self.frames]
            else:
                self.frames_crop = []
            
            self.is_loaded = True
            
            return {
                'success': True,
                'frame_count': len(self.frames),
                'fps': self.fps,
                'duration': self.duration,
                'src_fps': self._src_fps,
                'total_frames': self._total_frames,
                'flash_times': flash_times,
                'quality_resolution': quality_resolution,
                'fps_reduction_info': fps_reduction_info
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """
        Obtiene un frame por su indice.
        
        Si hay recorte activo, retorna el frame recortado.
        
        Args:
            idx: Indice del frame
            
        Returns:
            Frame o None si el indice es invalido
        """
        if not self.is_loaded or idx < 0 or idx >= len(self.frames):
            return None
        
        if self.frames_crop:
            return self.frames_crop[idx]
        return self.frames[idx]
    
    def get_full_frame(self, idx: int) -> Optional[np.ndarray]:
        """
        Obtiene el frame completo (sin recorte).
        
        Args:
            idx: Indice del frame
            
        Returns:
            Frame completo o None
        """
        if not self.is_loaded or idx < 0 or idx >= len(self.frames):
            return None
        return self.frames[idx]
    
    def get_working_frame(self, idx: int) -> np.ndarray:
        """
        Obtiene el frame de trabajo (con o sin recorte).
        
        Alias para get_frame para compatibilidad.
        
        Args:
            idx: Indice del frame
            
        Returns:
            Frame de trabajo
        """
        return self.get_frame(idx) or self.get_full_frame(idx)
    
    def get_timestamp(self, idx: int) -> float:
        """
        Obtiene el timestamp de un frame.
        
        Args:
            idx: Indice del frame
            
        Returns:
            Timestamp en segundos
        """
        if idx < 0 or idx >= len(self.timestamps):
            return 0.0
        return self.timestamps[idx]
    
    def get_current_timestamp(self) -> float:
        """Obtiene el timestamp del frame actual."""
        return self.get_timestamp(self.current_idx)
    
    def set_current_index(self, idx: int):
        """
        Establece el indice del frame actual.
        
        Args:
            idx: Nuevo indice
        """
        if 0 <= idx < len(self.frames):
            self.current_idx = idx
    
    def set_crop_rect(self, rect: Optional[Tuple[int, int, int, int]]):
        """
        Establece el rectangulo de recorte.
        
        Args:
            rect: Tupla (x1, y1, x2, y2) o None para quitar recorte
        """
        self.crop_rect = rect
        if rect:
            x1, y1, x2, y2 = rect
            self.frames_crop = [f[y1:y2, x1:x2] for f in self.frames]
        else:
            self.frames_crop = []
    
    def release(self):
        """Libera los recursos del video."""
        self.frames = []
        self.frames_crop = []
        self.timestamps = []
        self.is_loaded = False
        self.current_idx = 0
        self.crop_rect = None
    
    def get_frame_range(
        self,
        start_idx: int,
        end_idx: int
    ) -> List[Tuple[int, np.ndarray, float]]:
        """
        Obtiene un rango de frames.
        
        Args:
            start_idx: Indice inicial
            end_idx: Indice final (exclusivo)
            
        Returns:
            Lista de tuplas (indice, frame, timestamp)
        """
        result = []
        for i in range(max(0, start_idx), min(len(self.frames), end_idx)):
            frame = self.get_frame(i)
            if frame is not None:
                result.append((i, frame, self.timestamps[i]))
        return result
