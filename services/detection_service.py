"""
detection_service.py - Servicio de Deteccion
===========================================
Contiene la logica centralizada para la deteccion de pupila e iris,
desacoplada de la interfaz grafica.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pupilometer.core import (
    AdvancedEyeDetector as AdvEyeDetector,
    DetectionMode,
    KalmanEye
)


class DetectionService:
    """
    Servicio centralizado para la deteccion de pupila e iris.
    
    Provee una interfaz unificada para los diferentes algoritmos
    de deteccion, manejando el estado y la configuracion.
    
    Attributes:
        kalman: Filtro de Kalman para suavizado
        unet_model: Modelo U-Net cargado (opcional)
        current_detector: Detector actual en uso
        
    Example:
        >>> service = DetectionService()
        >>> result = service.detect_pupil(frame)
        >>> if result['pupil_center']:
        ...     print(f"Pupila en {result['pupil_center']}")
    """
    
    PUPIL_ALGORITHMS = ['starburst', 'swirski', 'excuse', 'canny', 'threshold', 'darkcircle', 'ia']
    IRIS_ALGORITHMS = ['gradient', 'hough', 'ellipse', 'profile']
    
    IRIS_MODE_MAP = {
        'gradient': DetectionMode.GRADIENT_BASED,
        'hough': DetectionMode.CIRCULAR_HOUGH,
        'ellipse': DetectionMode.ELLIPSE_FIT,
        'profile': DetectionMode.INTENSITY_PROFILE
    }
    
    DEFAULT_IRIS_DIAMETER_MM = 11.7
    
    def __init__(self, unet_model=None):
        """
        Inicializa el servicio de deteccion.
        
        Args:
            unet_model: Modelo U-Net pre-cargado (opcional)
        """
        self.unet_model = unet_model
        self.kalman = None
        try:
            self.kalman = KalmanEye()
        except Exception:
            pass
        
        self._pupil_algorithm = 'starburst'
        self._iris_algorithm = 'gradient'
        self._use_ai = False
        
        self._escala_px_mm = None
        self._error_calibracion_mm = None
        self._distancia_regla_mm = 40.0
        self._iris_mm_var = None
        
        self.current_detector = None
        
    @property
    def pupil_algorithm(self) -> str:
        """Obtiene el algoritmo de pupila actual."""
        return self._pupil_algorithm
    
    @pupil_algorithm.setter
    def pupil_algorithm(self, value: str):
        """Establece el algoritmo de pupila."""
        if value in self.PUPIL_ALGORITHMS:
            self._pupil_algorithm = value
        else:
            raise ValueError(f'Algoritmo invalido: {value}')
    
    @property
    def iris_algorithm(self) -> str:
        """Obtiene el algoritmo de iris actual."""
        return self._iris_algorithm
    
    @iris_algorithm.setter
    def iris_algorithm(self, value: str):
        """Establece el algoritmo de iris."""
        if value in self.IRIS_ALGORITHMS:
            self._iris_algorithm = value
        else:
            raise ValueError(f'Algoritmo invalido: {value}')
    
    @property
    def use_ai(self) -> bool:
        """Indica si se usa IA para deteccion."""
        return self._use_ai
    
    @use_ai.setter
    def use_ai(self, value: bool):
        """Establece si se usa IA."""
        self._use_ai = bool(value)
    
    def set_calibration(self, escala_px_mm: float, error_mm: float = None):
        """
        Establece la calibracion de escala.
        
        Args:
            escala_px_mm: Escala en px/mm (obtenida de la regla)
            error_mm: Error estimado en mm (opcional)
        """
        self._escala_px_mm = escala_px_mm
        self._error_calibracion_mm = error_mm
    
    def set_iris_mm(self, iris_mm: float):
        """
        Establece el diametro del iris en mm para estimacion.
        
        Args:
            iris_mm: Diametro del iris en milimetros
        """
        self._iris_mm_var = iris_mm
    
    def _get_iris_mode(self) -> DetectionMode:
        """
        Obtiene el modo de deteccion de iris segun el algoritmo configurado.
        
        Returns:
            Modo de deteccion de iris
        """
        return self.IRIS_MODE_MAP.get(
            self._iris_algorithm,
            DetectionMode.GRADIENT_BASED
        )
    
    def _create_detector(self, detectar_iris: bool = False) -> AdvEyeDetector:
        """
        Crea una instancia del detector avanzado.
        
        Args:
            detectar_iris: Si True, configura para detectar iris
            
        Returns:
            Instancia de AdvancedEyeDetector
        """
        iris_mode = self._get_iris_mode() if detectar_iris else DetectionMode.GRADIENT_BASED
        
        return AdvEyeDetector(
            pupil_algorithm=self._pupil_algorithm,
            iris_mode=iris_mode,
            expected_pupil_range=(0.1, 0.7),
            expected_iris_range=(0.3, 0.9)
        )
    
    def _estimate_iris_radius(self, pupil_radius: int) -> int:
        """
        Estima el radio del iris basado en la pupila.
        
        Por defecto usa 2.5x el radio de la pupila.
        
        Args:
            pupil_radius: Radio de la pupila en pixels
            
        Returns:
            Radio estimado del iris en pixels
        """
        return int(pupil_radius * 2.5)
    
    def _calculate_px_to_mm(self, iris_radius: int) -> Optional[float]:
        """
        Calcula el factor de conversion px a mm.
        
        Args:
            iris_radius: Radio del iris en pixels
            
        Returns:
            Factor de conversion px a mm, o None si no hay referencia
        """
        if self._escala_px_mm is not None:
            return 1.0 / self._escala_px_mm
        
        if self._iris_mm_var is not None and self._iris_mm_var > 0 and iris_radius > 0:
            return self._iris_mm_var / (iris_radius * 2)
        
        return None
    
    def detect_pupil(
        self,
        frame: np.ndarray,
        prev_result: Optional[Dict[str, Any]] = None,
        detectar_iris: bool = False
    ) -> Dict[str, Any]:
        """
        Detecta pupila (y opcionalmente iris) en un frame.
        
        Args:
            frame: Frame a procesar (imagen BGR)
            prev_result: Resultado del frame anterior (para ROI)
            detectar_iris: Si True, detecta iris. Si False, solo pupila.
            
        Returns:
            Diccionario con:
                - pupil_center: Tupla (x, y) o None
                - pupil_radius: Radio en pixels o None
                - iris_center: Tupla (x, y) o None
                - iris_radius: Radio en pixels o None
                - px_to_mm: Factor de conversion o None
                
        Note:
            La IA (U-Net) solo se usa para detectar pupila, nunca para iris.
        """
        usar_ia = (self._pupil_algorithm == 'ia') or (
            self._use_ai and self.unet_model is not None
        )
        
        h, w = frame.shape[:2]
        roi_frame = frame
        offset_x, offset_y = 0, 0
        
        if usar_ia and self.unet_model is not None:
            from pupilometer.ai import segment as unet_segment
            from pupilometer.ai import extract_pupil_from_mask
            
            try:
                mask = unet_segment(self.unet_model, roi_frame)
                if mask is not None:
                    pc, pr = extract_pupil_from_mask(mask)
                    if pc and pr:
                        pc = (int(pc[0]) + offset_x, int(pc[1]) + offset_y)
                        ir = None
                        
                        if detectar_iris:
                            detector = self._create_detector(detectar_iris=True)
                            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                            iris_result = detector.detect_iris(
                                gray,
                                {'center': pc, 'diameter': pr * 2}
                            )
                            if iris_result:
                                ir = int(iris_result['radius'])
                        
                        return {
                            'pupil_center': pc,
                            'pupil_radius': pr,
                            'iris_center': pc,
                            'iris_radius': ir,
                            'px_to_mm': self._calculate_px_to_mm(ir or 0)
                        }
            except Exception:
                pass
        
        if prev_result and prev_result.get('pupil_center'):
            px, py = prev_result['pupil_center']
            pr_ = prev_result.get('pupil_radius') or 30
            margin = max(int(pr_ * 3), 60)
            x1 = max(0, px - margin)
            y1 = max(0, py - margin)
            x2 = min(w, px + margin)
            y2 = min(h, py + margin)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                roi_frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
        
        detector = self._create_detector(detectar_iris=detectar_iris)
        result = detector.detect(roi_frame)
        
        pc = pr = ic = ir = None
        
        if result.pupil_center and result.pupil_radius:
            pc = (
                int(result.pupil_center[0]) + offset_x,
                int(result.pupil_center[1]) + offset_y
            )
            pr = result.pupil_radius
        
        if detectar_iris and result.iris_radius:
            ir = result.iris_radius
        elif pr is not None:
            ir = self._estimate_iris_radius(pr)
        
        if pc is None:
            from pupilometer.core import detect_all
            pc, pr, ic, ir = detect_all(frame, False, 50, 5, 15)
        
        if pc is not None:
            ic = pc
            if ir is None and pr is not None:
                ir = self._estimate_iris_radius(pr)
            
            if self.kalman:
                try:
                    pc = self.kalman.update(pc[0], pc[1])
                    ic = pc
                except Exception:
                    pass
        
        return {
            'pupil_center': pc,
            'pupil_radius': pr,
            'iris_center': ic,
            'iris_radius': ir,
            'px_to_mm': self._calculate_px_to_mm(ir or 0)
        }
    
    def detect_iris_only(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detecta solo el iris en un frame.
        
        Si no se detecta, retorna un iris por defecto de 11.7mm
        con el mismo centro que la pupila.
        
        Args:
            frame: Frame a procesar (imagen BGR)
            
        Returns:
            Diccionario con iris_center e iris_radius, o None si falla
        """
        detector = self._create_detector(detectar_iris=True)
        result = detector.detect(frame)
        
        if result.iris_center and result.iris_radius:
            return {
                'iris_center': (
                    int(result.iris_center[0]),
                    int(result.iris_center[1])
                ),
                'iris_radius': result.iris_radius
            }
        
        return None
    
    def get_default_iris(
        self,
        pupil_center: Optional[Tuple[int, int]],
        pupil_radius: Optional[int],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Genera un iris por defecto de 11.7mm de diametro.
        
        Usa el centro de la pupila si existe, sino el centro del frame.
        
        Args:
            pupil_center: Centro de la pupila (x, y)
            pupil_radius: Radio de la pupila
            frame_shape: Forma del frame (h, w)
            
        Returns:
            Diccionario con iris_center e iris_radius
        """
        h, w = frame_shape[:2]
        
        if pupil_center:
            center = pupil_center
        else:
            center = (w // 2, h // 2)
        
        px_to_mm = None
        if self._escala_px_mm is not None:
            px_to_mm = 1.0 / self._escala_px_mm
        elif self._iris_mm_var and pupil_radius:
            try:
                iris_mm_manual = float(self._iris_mm_var)
                if iris_mm_manual > 0:
                    px_to_mm = iris_mm_manual / (pupil_radius * 2.5 * 2)
            except (ValueError, TypeError):
                pass
        
        if px_to_mm is not None:
            iris_radius_mm = self.DEFAULT_IRIS_DIAMETER_MM / 2.0
            iris_radius_px = int(iris_radius_mm / px_to_mm)
        else:
            iris_radius_px = 60
        
        return {
            'iris_center': center,
            'iris_radius': iris_radius_px
        }
    
    def reset(self):
        """Reinicia el estado del servicio."""
        if self.kalman:
            try:
                self.kalman.reset()
            except Exception:
                pass
