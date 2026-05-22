"""
Analysis module - Pupilometer
=============================
Modulos de analisis clinico y calibracion.
"""

from .pupillometry import (
    compute, format_report, export_csv,
    estimate_indeterminate_frames, merge_estimated_results
)
from .calibration import (
    calibrar_regla_rosa_fosforescente,
    aplicar_calibracion, convertir_diametro_mm
)
from .flash_detection import (
    FlashDetector,
    detect_flash_in_video,
    trim_video_by_flash,
    should_reduce_fps
)
from .pupil_adaptive_analyzer import (
    PupilSample, PhaseResult, PupilMetrics,
    AdaptiveBinaryAnalyzer, PupilDetectionError,
    compute_metrics, plot_results, run_adaptive_analysis
)

__all__ = [
    'compute', 'format_report', 'export_csv',
    'calibrar_regla_rosa_fosforescente',
    'aplicar_calibracion', 'convertir_diametro_mm',
    'estimate_indeterminate_frames', 'merge_estimated_results',
    'FlashDetector', 'detect_flash_in_video',
    'trim_video_by_flash', 'should_reduce_fps',
    'PupilSample', 'PhaseResult', 'PupilMetrics',
    'AdaptiveBinaryAnalyzer', 'PupilDetectionError',
    'compute_metrics', 'plot_results', 'run_adaptive_analysis'
]
