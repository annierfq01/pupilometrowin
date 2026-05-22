"""
Pupilometer - Advanced Pupil and Iris Detection System
======================================================

Sistema modular para deteccion de pupila e iris con multiples algoritmos,
soporte para IA (U-Net), tracking con Kalman, y analisis de pupilometria clinica.

Modulos:
    core/       - Deteccion basica y avanzada (incluye DarkCircle)
    ai/         - U-Net y grabacion de dataset
    analysis/   - Pupillometria clinica y calibracion
    utils/      - Funciones utilitarias

Ejemplo de uso:
    >>> from pupilometer import core, ai
    >>> from pupilometer.core import detect_all, DarkCircleDetector
    >>> 
    >>> # Deteccion basica
    >>> pupil_cx, pupil_cy, pupil_r, iris_r = detect_all(image)
    >>> 
    >>> # Deteccion con DarkCircle
    >>> detector = DarkCircleDetector()
    >>> result = detector.detect(gray_image)
    >>> 
    >>> # Grabacion de dataset
    >>> from pupilometer.ai import DatasetRecorder
    >>> recorder = DatasetRecorder('dataset')
    >>> recorder.try_save(frame, pc, pr, ic, ir, crop_rect)
"""

__version__ = '1.0.0'
__author__ = 'Pupilometer Development Team'

from . import core
from . import ai
from . import analysis
from . import utils

__all__ = ['core', 'ai', 'analysis', 'utils']
