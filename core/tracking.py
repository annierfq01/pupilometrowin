"""
tracking.py - Kalman Filter Module
=================================
Filtro de Kalman 2D para suavizado temporal de la posicion de la pupila.
Reduce el jitter frame a frame en video, mejorando la precision
de las derivadas en pupilometria.
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class KalmanEye:
    """
    Filtro de Kalman 2D para suavizar la posicion (cx, cy) de la pupila.
    
    Estado: [x, y, vx, vy]
    Medicion: [x, y]
    """
    
    def __init__(self):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) es requerido para KalmanEye")
        
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self._initialized = False
    
    def update(self, x: int, y: int) -> tuple:
        """
        Actualiza el filtro con la medicion (x, y) y devuelve
        la posicion suavizada.
        
        Args:
            x: Coordenada x de la medicion
            y: Coordenada y de la medicion
            
        Returns:
            Tupla (x_suavizado, y_suavizado)
        """
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self._initialized:
            self.kf.statePre = np.array(
                [[np.float32(x)], [np.float32(y)],
                 [0.0], [0.0]], dtype=np.float32)
            self._initialized = True
        
        self.kf.correct(measurement)
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])
    
    def predict_only(self) -> tuple:
        """
        Devuelve la prediccion sin nueva medicion (para frames sin deteccion).
        
        Returns:
            Tupla (x_predicho, y_predicho)
        """
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])
    
    def reset(self):
        """Reinicia el filtro de Kalman."""
        self._initialized = False
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
