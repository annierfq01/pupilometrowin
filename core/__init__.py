"""
Core module - Pupilometer
=========================
Modulos fundamentales de deteccion y seguimiento.
"""

from .detection import detect_all, draw_detections
from .advanced import AdvancedEyeDetector, DetectionMode
from .darkcircle import DarkCircleDetector, detect_pupil_darkcircle, draw_darkcircle_detection
from .tracking import KalmanEye

__all__ = [
    'detect_all', 'draw_detections',
    'AdvancedEyeDetector', 'DetectionMode',
    'DarkCircleDetector', 'detect_pupil_darkcircle', 'draw_darkcircle_detection',
    'KalmanEye'
]
