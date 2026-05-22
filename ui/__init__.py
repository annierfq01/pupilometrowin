"""
UI module - Pupilometer
=========================
Modulo de interfaz grafica.

Contiene componentes reutilizables para la interfaz:
- VideoSettingsWindow: Ventana de configuracion de video
- AnalisisVideoWindow: Ventana de ajustes de analisis de video
"""

from .main_window import main_window

from .video_settings import VideoSettingsWindow, create_video_settings_window
from .analisis_video import AnalisisVideoWindow, create_analisis_video_window

__all__ = [
    'main_window',
    'VideoSettingsWindow',
    'create_video_settings_window',
    'AnalisisVideoWindow',
    'create_analisis_video_window',
]
