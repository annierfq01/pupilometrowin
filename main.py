"""
main.py — UI principal (Tkinter) - Pupilometer Package
===================================================
Punto de entrada de la aplicacion. Usa los modulos internos:
  pupilometer.core    — vision computacional y deteccion
  pupilometer.ai     — U-Net + dataset
  pupilometer.analysis — pupillometria y calibracion
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import sys
import threading
import math
import zipfile
import json
import io

# Agregar el directorio padre al path para poder importar pupilometer
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Modulos internos del paquete pupilometer
from pupilometer.core import detect_all, draw_detections
from pupilometer.core import KalmanEye
from pupilometer.core import AdvancedEyeDetector as AdvEyeDetector
from pupilometer.core import DetectionMode
from pupilometer.analysis import compute as compute_pupillometry
from pupilometer.analysis import format_report, export_csv as export_pupillometry_csv
from pupilometer.analysis import estimate_indeterminate_frames, merge_estimated_results
from pupilometer.analysis import detect_flash_in_video, trim_video_by_flash, should_reduce_fps
from pupilometer.analysis import (
    run_adaptive_analysis, compute_metrics as compute_adaptive_metrics,
    plot_results as plot_adaptive_results, PupilMetrics as AdaptivePupilMetrics,
    PhaseResult as AdaptivePhaseResult
)
from pupilometer.ai import DatasetRecorder, review_dataset
from pupilometer.ai import segment as unet_segment, extract_pupil_from_mask, train_model as unet_train_model, create_model as unet_create_model
from pupilometer.ai import is_available as ai_available, load_model as load_ai_model

# Calibracion de regla
from pupilometer.analysis.calibration import calibrar_regla_rosa_fosforescente as calibrar_regla_rosa_fosforescente_mejorado

# UI modules
from pupilometer.ui.video_settings import create_video_settings_window
from pupilometer.ui.analisis_video import create_analisis_video_window

# Video processing
from pupilometer.services.video_processor import VideoProcessor

# Detección automática de flash
import numpy as np
from collections import deque

# Alias para compatibilidad
adv_detector = type('AdvDetector', (), {
    'AdvancedEyeDetector': AdvEyeDetector,
    'DetectionMode': DetectionMode
})()

TARGET_FPS = 30
CANVAS_W   = 640
CANVAS_H   = 480

# ── Protocolo de estimulacion (Pupilometry Recorder HTML) ─────────────────────
# Flash ON  : 1.000 s desde inicio de grabacion
# Flash OFF : 1.200 s (duracion del flash = 200 ms)
# Duracion total del video: 6.000 s
# Estos son los defaults que se cargan al iniciar la app.
# El usuario puede editarlos en el panel "Tiempo de Iluminacion".
PROTOCOL_FLASH_ON_SEC  = 1.0   # segundos
PROTOCOL_FLASH_OFF_SEC = 1.2   # segundos


# ──────────────────────────────────────────────────────────────────────────────
#  Widget: barra de rango dual
# ──────────────────────────────────────────────────────────────────────────────

class RangeSlider(tk.Canvas):
    H = 28; RAIL_H = 6; R = 9

    def __init__(self, parent, on_change=None, **kwargs):
        kwargs.setdefault('height', self.H)
        try:
            bg = parent.cget('bg')
        except Exception:
            bg = '#f0f0f0'
        kwargs.setdefault('bg', bg)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)
        self._on_change = on_change
        self._start = 0.0
        self._end   = 1.0
        self._drag  = None
        self.bind('<Configure>',      lambda e: self._draw())
        self.bind('<ButtonPress-1>',   self._on_press)
        self.bind('<B1-Motion>',       self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def set_range(self, s, e):
        self._start = max(0.0, min(1.0, s))
        self._end   = max(0.0, min(1.0, e))
        if self._start > self._end:
            self._start, self._end = self._end, self._start
        self._draw()

    def get_range(self): return self._start, self._end

    def _draw(self):
        w = self.winfo_width() or 200
        self.delete('all')
        cy = self.H // 2
        x0 = self.R + 2; x1 = w - self.R - 2
        rng = max(x1 - x0, 1)
        sx = int(x0 + self._start * rng)
        ex = int(x0 + self._end   * rng)
        self.create_rectangle(x0, cy-self.RAIL_H//2, x1, cy+self.RAIL_H//2,
                               fill='#cccccc', outline='')
        self.create_rectangle(sx, cy-self.RAIL_H//2, ex, cy+self.RAIL_H//2,
                               fill='#4a9eff', outline='')
        for xh, color in [(sx,'#1a6fcc'), (ex,'#cc3300')]:
            self.create_oval(xh-self.R, cy-self.R, xh+self.R, cy+self.R,
                              fill=color, outline='white', width=2)

    def _frac(self, mx):
        w = self.winfo_width() or 200
        x0 = self.R + 2; x1 = w - self.R - 2
        return max(0.0, min(1.0, (mx - x0) / max(x1-x0, 1)))

    def _on_press(self, e):
        w = self.winfo_width() or 200
        x0 = self.R + 2; rng = max(w - 2*self.R - 4, 1)
        sx = x0 + self._start * rng; ex = x0 + self._end * rng
        self._drag = 'start' if abs(e.x-sx) <= abs(e.x-ex) else 'end'

    def _on_drag(self, e):
        if not self._drag: return
        f = self._frac(e.x)
        if self._drag == 'start':
            self._start = min(f, self._end - 0.001)
        else:
            self._end = max(f, self._start + 0.001)
        self._draw()
        if self._on_change: self._on_change(self._start, self._end)

    def _on_release(self, e): self._drag = None


class IlluminationSlider(tk.Canvas):
    """Selector dual para inicio y fin de iluminacion con marcas visuales."""
    H = 36; RAIL_H = 8; R = 10

    def __init__(self, parent, on_change=None, **kwargs):
        kwargs.setdefault('height', self.H)
        try:
            bg = parent.cget('bg')
        except Exception:
            bg = '#f0f0f0'
        kwargs.setdefault('bg', bg)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)
        self._on_change = on_change
        self._light_on = 0.1   # Inicio iluminacion
        self._light_off = 0.9   # Fin iluminacion
        self._drag = None
        self.bind('<Configure>',      lambda e: self._draw())
        self.bind('<ButtonPress-1>',   self._on_press)
        self.bind('<B1-Motion>',       self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def set_illumination(self, on_frac, off_frac):
        self._light_on  = max(0.0, min(1.0, on_frac))
        self._light_off = max(0.0, min(1.0, off_frac))
        if self._light_on > self._light_off:
            self._light_on, self._light_off = self._light_off, self._light_on
        self._draw()

    def get_illumination(self): return self._light_on, self._light_off

    def _draw(self):
        w = self.winfo_width() or 300
        self.delete('all')
        cy = self.H // 2
        x0 = self.R + 2; x1 = w - self.R - 2
        rng = max(x1 - x0, 1)
        
        # Rail completo
        self.create_rectangle(x0, cy-self.RAIL_H//2, x1, cy+self.RAIL_H//2,
                               fill='#333333', outline='#555555', width=1)
        
        # Calcular posiciones
        on_x  = int(x0 + self._light_on  * rng)
        off_x = int(x0 + self._light_off * rng)
        
        # Zona oscura (antes de iluminacion)
        self.create_rectangle(x0, cy-self.RAIL_H//2, on_x, cy+self.RAIL_H//2,
                               fill='#222222', outline='')
        
        # Zona iluminada
        self.create_rectangle(on_x, cy-self.RAIL_H//2, off_x, cy+self.RAIL_H//2,
                               fill='#FFD700', outline='')
        
        # Zona oscura (despues de iluminacion)
        self.create_rectangle(off_x, cy-self.RAIL_H//2, x1, cy+self.RAIL_H//2,
                               fill='#222222', outline='')
        
        # Handle inicio (verde = luz ON)
        self.create_oval(on_x-self.R, cy-self.R, on_x+self.R, cy+self.R,
                           fill='#00CC00', outline='white', width=2)
        self.create_text(on_x, cy+self.R+10, text='ON', font=('Arial',7,'bold'), fill='#00CC00')
        
        # Handle fin (rojo = luz OFF)
        self.create_oval(off_x-self.R, cy-self.R, off_x+self.R, cy+self.R,
                           fill='#FF4400', outline='white', width=2)
        self.create_text(off_x, cy+self.R+10, text='OFF', font=('Arial',7,'bold'), fill='#FF4400')

    def _frac(self, mx):
        w = self.winfo_width() or 300
        x0 = self.R + 2; x1 = w - self.R - 2
        return max(0.0, min(1.0, (mx - x0) / max(x1-x0, 1)))

    def _on_press(self, e):
        w = self.winfo_width() or 300
        x0 = self.R + 2; rng = max(w - 2*self.R - 4, 1)
        on_x  = x0 + self._light_on  * rng
        off_x = x0 + self._light_off * rng
        # Click en inicio o fin
        if abs(e.x - on_x) <= abs(e.x - off_x):
            self._drag = 'on'
        else:
            self._drag = 'off'

    def _on_drag(self, e):
        if not self._drag: return
        f = self._frac(e.x)
        if self._drag == 'on':
            self._light_on = min(f, self._light_off - 0.01)
        else:
            self._light_off = max(f, self._light_on + 0.01)
        self._draw()
        if self._on_change:
            self._on_change(self._light_on, self._light_off)

    def _on_release(self, e): self._drag = None


# ──────────────────────────────────────────────────────────────────────────────
#  Aplicacion principal
# ──────────────────────────────────────────────────────────────────────────────

class AdvancedEyeDetector:

    def __init__(self, root):
        self.root = root
        self.root.title("Detector Avanzado de Pupila e Iris")
        self.root.geometry("1200x750")

        # imagen
        self.image = None; self.current_image = None
        self.pupil_center = None; self.pupil_radius = None
        self.iris_center  = None; self.iris_radius  = None
        self.pixel_to_mm  = None
        self.original_image = None

        # video
        self.is_video = False; self.video_frames = []
        self.video_frames_crop = []; self.video_fps = 30.0
        self.video_timestamps = []
        self.frame_results = []; self.current_frame_idx = 0

        # rango iluminacion — defaults del protocolo Pupilometry Recorder
        # Flash ON = 1.0 s, Flash OFF = 1.2 s (flash de 200 ms a 1 segundo)
        self._light_start_sec = PROTOCOL_FLASH_ON_SEC
        self._light_end_sec   = PROTOCOL_FLASH_OFF_SEC
        self._light_start_frac = 0.0
        self._light_end_frac   = 1.0

        # IA toggle
        self._use_ai = tk.BooleanVar(value=False)
        self._ai_enabled = False

        # Modo ajuste manual
        self._edit_mode = False
        self._edit_pupil_radius = None
        self._edit_iris_radius = None
        self._drag_circle = None
        self._drag_start = None
        self._circle_ids = {'pupil': None, 'iris': None}
        self._selected_circle = None

        # recorte
        self.crop_rect = None; self._crop_start = None
        self._crop_rect_id = None; self._crop_mode = False
        self._canvas_offset = (0, 0); self._canvas_scale = 1.0

        # Kalman (opcional, mejora suavidad en video)
        try:
            self.kalman = KalmanEye()
        except Exception:
            self.kalman = None

        # U-Net (opcional, requiere entrenamiento previo)
        self.unet_model = None
        try:
            self.unet_model = load_ai_model('unet_eye.pth')
        except Exception:
            pass

        # Dataset recorder
        self.recorder = DatasetRecorder('dataset')

        # pupilometria guardada
        self._pupillometry_data = None
        self._adaptive_results = None
        self._adaptive_metrics = None
        
        # Calibracion regla
        self._distancia_regla_mm = 40.0  # Distancia configurable entre puntos de la regla
        self._calibracion_resultado = None
        self._escala_px_mm = None
        self._error_calibracion_mm = None
        self.calibracion_info_var = tk.StringVar(value='Sin calibrar')
        self.calibracion_error_var = tk.StringVar(value='')
        
        # Analisis de iris opcional
        self._analisis_iris_habilitado = tk.BooleanVar(value=False)
        self._iris_resultado_actual = None  # Para videos: guarda el resultado de iris del frame actual
        
        # Mostrar/ocultar dibujos
        self._mostrar_dibajos = tk.BooleanVar(value=True)
        
        # Frames indeterminados (para ojos cerrados/pestañeando)
        self._indeterminate_frames = set()  # Conjunto de índices de frames indeterminados
        
        # Configuración de detección automática de flash
        self._flash_detection_auto = tk.BooleanVar(value=False)
        self._flash_calibration_frames = tk.IntVar(value=5)
        self._flash_change_threshold = tk.DoubleVar(value=5.0)
        self._flash_smoothing_window = tk.IntVar(value=4)
        self._flash_brightness_base = None
        self._flash_trim_before_ms = tk.IntVar(value=100)
        self._flash_trim_after_ms = tk.IntVar(value=100)
        # Tiempos de flash manual (en segundos desde inicio del video)
        self._flash_start_sec = tk.DoubleVar(value=1.0)
        self._flash_end_sec = tk.DoubleVar(value=1.2)
        self.iris_status_var = tk.StringVar(value='Iris: desactivado')
        self.detect_iris_frame_btn = None
        
        # Configuracion de fases del video
        # Fase 1: Basal (2 fps, 3 seg)
        # Fase 2: Iluminacion (30 fps, 2 seg)
        # Fase 3: Relajacion (10 fps, 7 seg)
        self._fase_basal_fps = tk.IntVar(value=2)
        self._fase_basal_dur = tk.IntVar(value=3)
        self._fase_contraccion_fps = tk.IntVar(value=30)
        self._fase_contraccion_dur = tk.IntVar(value=2)
        self._fase_relajacion_fps = tk.IntVar(value=10)
        self._fase_relajacion_dur = tk.IntVar(value=7)
        
        # Variables para UI
        self.iris_mm_var = tk.StringVar()
        self.scale_info_var = tk.StringVar(value='Ratio: -- | Pupila: --')
        self.threshold_var = tk.IntVar(value=50)
        self.blur_var = tk.IntVar(value=5)
        self.iris_smooth_var = tk.IntVar(value=15)
        self.line_width_var = tk.IntVar(value=2)
        self.light_start_var = tk.StringVar(value='1.0')
        self.light_end_var = tk.StringVar(value='1.2')
        self._illum_lbl_var = tk.StringVar(value='Iluminacion: 1.00s - 1.20s')

        self.supported_img = [
            ('Imagenes', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
            ('Todos', '*.*')]
        self.supported_vid = [
            ('Videos', '*.mp4 *.avi *.mov *.mkv *.wmv *.webm'),
            ('Todos', '*.*')]

        self._setup_ui()

    # ══════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════

    def _setup_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Menu completo
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu Archivo
        archivo_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=archivo_menu)
        archivo_menu.add_command(label="Nuevo", command=self.new_project, underline=0)
        archivo_menu.add_separator()
        archivo_menu.add_command(label="Importar Imagen", command=self.load_image, accelerator="Ctrl+I")
        archivo_menu.add_command(label="Importar Video", command=self.load_video, accelerator="Ctrl+V")
        archivo_menu.add_separator()
        archivo_menu.add_command(label="Abrir...", command=self.open_project, accelerator="Ctrl+O")
        archivo_menu.add_command(label="Guardar", command=self.save_project, accelerator="Ctrl+G")
        archivo_menu.add_separator()
        archivo_menu.add_command(label="Exportar CSV", command=self.export_csv, accelerator="Ctrl+Shift+C")
        archivo_menu.add_command(label="Exportar TXT", command=self.export_summary_txt, accelerator="Ctrl+T")
        archivo_menu.add_separator()
        archivo_menu.add_command(label="Salir", command=self.root.quit)

        # Menu Analisis
        analisis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analisis", menu=analisis_menu)
        analisis_menu.add_command(label="Resumen", command=self.show_summary)
        analisis_menu.add_command(label="Grafico de Evolucion", command=self.show_evolution_chart)
        analisis_menu.add_command(label="Analisis Adaptativo (busqueda binaria)", command=self.run_adaptive_analysis_ui)

        # Menu Ajustes
        ajustes_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ajustes", menu=ajustes_menu)
        ajustes_menu.add_command(label="Calibracion de Regla...", command=self._open_calibracion_window)
        ajustes_menu.add_command(label="Analisis de Video...", command=self._open_analisis_video_window)
        ajustes_menu.add_separator()
        ajustes_menu.add_checkbutton(label="Mostrar dibujos (Ctrl+H)", variable=self._mostrar_dibajos, command=self._toggle_dibujos)

        # Menu IA / Dataset
        ia_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="IA / Dataset", menu=ia_menu)
        ia_menu.add_checkbutton(label="Usar IA en deteccion", variable=self._use_ai, command=self._toggle_ai_usage)
        ia_menu.add_separator()
        ia_menu.add_command(label="Seleccionar modelo IA...", command=self._select_model)
        ia_menu.add_command(label="Guardar modelo actual...", command=self.save_model)
        ia_menu.add_separator()
        ia_menu.add_command(label="Guardar frame actual al dataset", command=self._save_current_frame_to_dataset)
        ia_menu.add_command(label="Entrenar con dataset actual", command=self._train_with_current_dataset)
        ia_menu.add_separator()
        ia_menu.add_command(label="Entrenar U-Net...", command=self.open_training_window)
        ia_menu.add_command(label="Entrenar con imagenes...", command=self.open_single_image_training)
        ia_menu.add_separator()
        ia_menu.add_command(label="Revisar dataset (OpenCV)", command=lambda: review_dataset('dataset'))
        ia_menu.add_separator()
        ia_menu.add_command(label="Grabar dataset: ON/OFF", command=self._toggle_dataset)
        ia_menu.add_command(label="Guardar al dataset (con correccion)", command=self._save_correction_to_dataset)

        # Vincular atajos de teclado
        self.root.bind('<Control-i>', lambda e: self.load_image())
        self.root.bind('<Control-v>', lambda e: self.load_video())
        self.root.bind('<Control-o>', lambda e: self.open_project())
        self.root.bind('<Control-g>', lambda e: self.save_project())
        self.root.bind('<Control-Shift-C>', lambda e: self.export_csv())
        self.root.bind('<Control-t>', lambda e: self.export_summary_txt())
        self.root.bind('<Control-r>', lambda e: self.show_summary())
        self.root.bind('<Control-k>', lambda e: self.show_evolution_chart())
        self.root.bind('<Control-h>', lambda e: self._toggle_dibujos())
        self.root.bind('<F5>', lambda e: self._actualizar_resultados())
        self.root.bind('<Key-f>', lambda e: self.full_auto_detection() if self.canvas.focus_get() == self.canvas else None)

        # Panel izquierdo con scroll
        lp = ttk.Frame(self.root, padding=4)
        lp.grid(row=0, column=0, sticky='nsew')
        lp.columnconfigure(0, weight=1); lp.rowconfigure(0, weight=1)

        ctrl_cv = tk.Canvas(lp, width=312, highlightthickness=0)
        vsb = ttk.Scrollbar(lp, orient='vertical', command=ctrl_cv.yview)
        self.sf = ttk.Frame(ctrl_cv)
        self.sf.bind('<Configure>', lambda e: ctrl_cv.configure(
            scrollregion=ctrl_cv.bbox('all')))
        ctrl_cv.create_window((0, 0), window=self.sf, anchor='nw')
        ctrl_cv.configure(yscrollcommand=vsb.set)
        ctrl_cv.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        ctrl_cv.bind_all('<MouseWheel>', lambda e: ctrl_cv.yview_scroll(
            int(-e.delta/120), 'units'))

        self._row = 0
        def R(): v=self._row; self._row+=1; return v
        def lbl(t, bold=False):
            f = ('Arial',10,'bold') if bold else ('Arial',9)
            ttk.Label(self.sf,text=t,font=f).grid(
                row=R(),column=0,sticky='w',padx=10,pady=(7,1))
        def sep():
            ttk.Separator(self.sf,orient='horizontal').grid(
                row=R(),column=0,sticky='ew',padx=8,pady=5)
        def btn(t, cmd, w=30):
            ttk.Button(self.sf,text=t,command=cmd,width=w).grid(
                row=R(),column=0,padx=10,pady=2)
        def mksl(label, var, lo, hi, default):
            ttk.Label(self.sf,text=label).grid(
                row=R(),column=0,sticky='w',padx=10)
            fr = ttk.Frame(self.sf)
            fr.grid(row=R(),column=0,padx=10,pady=2)
            lv = tk.StringVar(value=str(default))
            sc = ttk.Scale(fr,from_=lo,to=hi,variable=var,
                           orient='horizontal',length=210)
            sc.grid(row=0,column=0)
            ttk.Label(fr,textvariable=lv,width=5).grid(row=0,column=1,padx=4)
            sc.configure(command=lambda x,lv=lv: lv.set(str(int(float(x)))))

        # Titulo
        ttk.Label(self.sf, text='DETECTOR PUPILA & IRIS',
                  font=('Arial',11,'bold')).grid(row=R(),column=0,pady=8)

        sep()

        # Deteccion
        lbl('DETECCION', bold=True)
        
        alg_fr = ttk.Frame(self.sf)
        alg_fr.grid(row=R(), column=0, padx=10, pady=4, sticky='ew')
        alg_fr.columnconfigure(0, weight=1)
        alg_fr.columnconfigure(1, weight=1)
        
        ttk.Label(alg_fr, text='Algoritmo pupila:', font=('Arial', 8)).grid(row=0, column=0, sticky='w')
        self.pupil_alg_var = tk.StringVar(value='starburst')
        pupil_algs = ['starburst', 'swirski', 'excuse', 'canny', 'threshold', 'darkcircle', 'ia']
        pupil_alg_combo = ttk.Combobox(alg_fr, textvariable=self.pupil_alg_var, 
                                        values=pupil_algs, state='readonly', width=10)
        pupil_alg_combo.grid(row=1, column=0, padx=2, sticky='ew')
        
        ttk.Label(alg_fr, text='Algoritmo iris:', font=('Arial', 8)).grid(row=0, column=1, sticky='w')
        self.iris_alg_var = tk.StringVar(value='gradient')
        iris_algs = ['gradient', 'hough', 'ellipse', 'profile']
        iris_alg_combo = ttk.Combobox(alg_fr, textvariable=self.iris_alg_var,
                                       values=iris_algs, state='readonly', width=10)
        iris_alg_combo.grid(row=1, column=1, padx=2, sticky='ew')
        
        # Deteccion de iris en panel lateral
        self.iris_status_var = tk.StringVar(value='Iris: desactivado')
        ttk.Checkbutton(self.sf, text='Detectar Iris',
                        variable=self._analisis_iris_habilitado,
                        command=self._on_iris_toggle).grid(row=R(), column=0, sticky='w', padx=10)
        self.detect_iris_frame_btn = ttk.Button(self.sf, text='Detectar Iris (frame actual)',
                                                 command=self._detectar_iris_frame,
                                                 state='disabled', width=25)
        self.detect_iris_frame_btn.grid(row=R(), column=0, padx=10, pady=2)
        
        btn('Detectar (frame actual)', self.full_auto_detection)
        btn('Detectar todos los frames', self._detect_all_frames)
        btn('Resetear Vista',          self.reset_view)
        btn('Guardar Frame',           self.save_result)

        sep()

        # AJUSTE MANUAL
        lbl('AJUSTE MANUAL', bold=True)
        edit_fr = ttk.Frame(self.sf)
        edit_fr.grid(row=R(),column=0,padx=8,pady=4,sticky='ew')
        
        self.edit_mode_btn = ttk.Button(edit_fr, text='Activar Ajuste Manual',
                                        command=self._toggle_edit_mode, width=28)
        self.edit_mode_btn.grid(row=0,column=0,padx=2,pady=2)
        
        self.propagate_btn = ttk.Button(edit_fr, text='⇢ Propagar a siguientes',
                                        command=self._confirm_changes, width=20, state='disabled')
        self.propagate_btn.grid(row=1, column=0, padx=2, pady=(4, 4))
        
        # Boton Actualizar
        self.actualizar_btn = ttk.Button(edit_fr, text='Actualizar',
                                         command=self._actualizar_resultados, width=28)
        self.actualizar_btn.grid(row=2, column=0, padx=2, pady=(4, 4))
        
        self.edit_status_var = tk.StringVar(value='Activa modo y arrastra controles')
        ttk.Label(edit_fr, textvariable=self.edit_status_var,
                  font=('Arial',7), foreground='gray').grid(row=3,column=0,sticky='w',pady=(0,4))

        sep()

        # Panel de analisis de video (solo visible cuando hay video)
        self.video_panel = ttk.LabelFrame(self.sf, text='ANALISIS DE VIDEO', padding=6)
        self.video_panel.grid(row=R(),column=0,padx=8,pady=4,sticky='ew')
        self.video_panel.columnconfigure(0, weight=1)
        
        # Info de calibracion
        self.calibracion_info_var = tk.StringVar(value='Sin calibrar')
        ttk.Label(self.video_panel, textvariable=self.calibracion_info_var,
                 font=('Arial',8), foreground='green').grid(row=0,column=0,sticky='w',pady=(0,4))
        
        # Boton calcular pupilometria (solo para videos)
        self.calc_pupillometry_btn = ttk.Button(self.video_panel, 
                                               text='Calcular Pupilometria',
                                               command=self.calculate_pupillometry,
                                               state='disabled', width=20)
        self.calc_pupillometry_btn.grid(row=1,column=0,pady=4,sticky='ew')
        
        # Boton analisis adaptativo
        self.adaptive_btn = ttk.Button(self.video_panel,
                                       text='Analisis Adaptativo',
                                       command=self.run_adaptive_analysis_ui,
                                       state='disabled', width=20)
        self.adaptive_btn.grid(row=2,column=0,pady=4,sticky='ew')
        
        # Boton Indeterminado
        self.indeterminado_btn = ttk.Button(self.video_panel,
                                            text='Indeterminado (Ojo Cerrado)',
                                            command=self._toggle_indeterminate_frame,
                                            state='disabled', width=20)
        self.indeterminado_btn.grid(row=2,column=0,pady=4,sticky='ew')
        
        # Label para indicar si el frame actual es indeterminado
        self.indeterminate_label_var = tk.StringVar(value='')
        ttk.Label(self.video_panel, textvariable=self.indeterminate_label_var,
                  font=('Arial', 8, 'bold'), foreground='orange').grid(row=3,column=0,sticky='w',pady=2)

        ttk.Separator(self.video_panel,orient='horizontal').grid(
            row=2,column=0,sticky='ew',pady=4)
        ttk.Label(self.video_panel, text='Recorte region de analisis:',
                  font=('Arial',8,'bold')).grid(row=3,column=0,sticky='w')

        cbfr = ttk.Frame(self.video_panel)
        cbfr.grid(row=4,column=0,sticky='ew',pady=2)
        cbfr.columnconfigure(0,weight=1); cbfr.columnconfigure(1,weight=1)
        self.crop_btn = ttk.Button(cbfr, text='Activar recorte',
                                    command=self.toggle_crop_mode)
        self.crop_btn.grid(row=0,column=0,padx=(0,2),sticky='ew')
        ttk.Button(cbfr, text='Sin recorte',
                   command=self.reset_crop).grid(row=0,column=1,padx=(2,0),sticky='ew')

        self.crop_info_var = tk.StringVar(value='Sin recorte activo')
        ttk.Label(self.video_panel, textvariable=self.crop_info_var,
                  font=('Arial',7), foreground='gray').grid(row=5,column=0,sticky='w')

        sep()

        # Dataset status
        self.dataset_lbl_var = tk.StringVar(value='Dataset: 0 muestras')
        ttk.Label(self.sf, textvariable=self.dataset_lbl_var,
                  font=('Arial',7), foreground='green').grid(
            row=R(),column=0,padx=10,sticky='w')
         
        self._pending_changes = False
        
        sep()

        # Resultados
        lbl('RESULTADOS', bold=True)
        res_fr = ttk.Frame(self.sf)
        res_fr.grid(row=R(),column=0,padx=8,pady=4,sticky='ew')
        self.info_text = tk.Text(res_fr, height=18, width=35,
                                  wrap='word', font=('Courier',8))
        self.info_text.grid(row=0,column=0)
        isb = ttk.Scrollbar(res_fr,orient='vertical',command=self.info_text.yview)
        isb.grid(row=0,column=1,sticky='ns')
        self.info_text.configure(yscrollcommand=isb.set)

        # Panel derecho con canvas y controles de circulos
        rp = ttk.Frame(self.root, padding=4)
        rp.grid(row=0,column=1,sticky='nsew')
        rp.columnconfigure(0,weight=1)
        rp.rowconfigure(0,weight=0)  # graph row
        rp.rowconfigure(1,weight=1)  # canvas row
        rp.rowconfigure(2,weight=0)  # video bar row

        # Compact diameter graph above the main canvas
        self.graph_frame = ttk.Frame(rp)
        self.graph_frame.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        self.graph_frame.columnconfigure(0, weight=1)
        self.graph_frame.grid_remove()  # Hidden until adaptive analysis is run

        # Contenedor del canvas
        self.canvas_container = tk.Frame(rp)
        self.canvas_container.grid(row=1,column=0,sticky='nsew')
        self.canvas_container.columnconfigure(0,weight=1)
        self.canvas_container.rowconfigure(0,weight=1)

        # Canvas principal para la imagen y controles
        self.canvas = tk.Canvas(self.canvas_container, bg='#2b2b2b', cursor='crosshair',
                                 width=CANVAS_W, height=CANVAS_H)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas_container.configure(width=CANVAS_W, height=CANVAS_H)
        
        self.canvas.bind('<ButtonPress-1>',   self._canvas_press)
        self.canvas.bind('<KeyPress>',         self._canvas_key_press)
        self.canvas.bind('<B1-Motion>',        self._canvas_motion)
        self.canvas.bind('<ButtonRelease-1>',  self._canvas_release)
        self.canvas.bind('<MouseWheel>',       self._canvas_wheel)
        
        self.root.bind('<KeyPress>', self._root_key_press)
        
        self.canvas.focus_set()
        
        # Variables para control de circulos
        self._control_dragging = None
        self._control_type = None

        # Barra video
        self.video_bar = ttk.Frame(rp)
        self.video_bar.grid(row=2,column=0,sticky='ew',pady=2)
        self.video_bar.columnconfigure(1,weight=1)
        self.frame_time_lbl = ttk.Label(self.video_bar,text='0.00s / 0.00s',width=16)
        self.frame_time_lbl.grid(row=0,column=0,padx=4)
        self.video_slider = ttk.Scale(self.video_bar, from_=0, to=1,
                                       orient='horizontal',
                                       command=self._on_video_slider)
        self.video_slider.bind('<ButtonRelease-1>', lambda e: self.canvas.focus_set())
        self.video_slider.grid(row=0,column=1,sticky='ew',padx=4)
        self.frame_idx_lbl = ttk.Label(self.video_bar,text='0/0',width=10)
        self.frame_idx_lbl.grid(row=0,column=2,padx=4)
        self.video_bar.grid_remove()

        self._set_video_controls_state('disabled')
        
        # Habilitar boton indeterminado cuando hay video
        self._update_indeterminate_button()

        self.status_bar = ttk.Label(self.root, text='Listo',
                                     relief='sunken', anchor='w')
        self.status_bar.grid(row=1,column=0,columnspan=2,sticky='ew')

    # ══════════════════════════════════════════════════════════════════
    #  Menu IA / Dataset
    # ══════════════════════════════════════════════════════════════════

    def _toggle_ai_usage(self):
        self._ai_enabled = self._use_ai.get()
        state = "ACTIVADA" if self._ai_enabled else "DESACTIVADA"
        self.status_bar.config(text=f'IA para deteccion: {state}')

    def _select_model(self):
        if not ai_available():
            messagebox.showwarning('Aviso',
                'PyTorch no esta instalado.\n'
                'Instala con: pip install torch torchvision')
            return
        
        path = filedialog.askopenfilename(
            title='Seleccionar modelo IA',
            defaultextension='.pth',
            filetypes=[('Modelo PyTorch', '*.pth'), ('Todos', '*.*')])
        if not path:
            return
        
        try:
            model = load_ai_model(path)
            if model:
                self.unet_model = model
                messagebox.showinfo('Cargado', f'Modelo cargado desde:\n{path}')
                self.status_bar.config(text=f'Modelo IA: {os.path.basename(path)}')
            else:
                messagebox.showwarning('Aviso', 'No se pudo cargar el modelo.')
        except Exception as e:
            messagebox.showerror('Error', f'No se pudo cargar el modelo:\n{str(e)}')

    def save_model(self):
        if not ai_available():
            messagebox.showwarning('Aviso', 'PyTorch no esta instalado.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.pth',
            filetypes=[('Modelo PyTorch', '*.pth'), ('Todos', '*.*')])
        if not path:
            return
        try:
            if self.unet_model:
                import torch
                torch.save(self.unet_model.state_dict(), path)
                messagebox.showinfo('Guardado', f'Modelo guardado en:\n{path}')
            else:
                messagebox.showwarning('Aviso', 'No hay modelo cargado para guardar.')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def _save_current_frame_to_dataset(self):
        if self.image is None and not self.is_video:
            messagebox.showwarning('Aviso', 'Carga una imagen o video primero.')
            return
        frame = None
        if self.is_video and self.video_frames:
            frame = self.video_frames[self.current_frame_idx]
        elif self.original_image is not None:
            frame = self.original_image
        
        if frame is None:
            messagebox.showwarning('Aviso', 'No hay frame disponible.')
            return
        
        crop_offset_x, crop_offset_y = 0, 0
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            frame = frame[y1:y2, x1:x2]
            crop_offset_x, crop_offset_y = x1, y1
        
        pc = self.pupil_center
        pr = self.pupil_radius
        ic = self.iris_center
        ir = self.iris_radius
        
        if pc is None or pr is None:
            messagebox.showwarning('Aviso', 'No hay deteccion valida para guardar.')
            return
        
        pc_crop = (pc[0] - crop_offset_x, pc[1] - crop_offset_y)
        ic_crop = (ic[0] - crop_offset_x, ic[1] - crop_offset_y) if ic else None
        
        img_dir = os.path.join('dataset', 'images')
        mask_dir = os.path.join('dataset', 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        try:
            ih, iw = frame.shape[:2]
            mask = np.zeros((ih, iw), dtype=np.uint8)
            if ic_crop and ir:
                cv2.circle(mask, (int(ic_crop[0]), int(ic_crop[1])), int(ir), 128, 2)
            if pc_crop and pr:
                cv2.circle(mask, (int(pc_crop[0]), int(pc_crop[1])), int(pr), 255, -1)
            
            idx = len(os.listdir(img_dir))
            cv2.imwrite(os.path.join(img_dir, f'{idx:05d}.png'), frame)
            cv2.imwrite(os.path.join(mask_dir, f'{idx:05d}.png'), mask)
            
            self._update_dataset_label()
            messagebox.showinfo('Guardado', 
                f'Frame guardado al dataset (recortado).\n'
                f'Pupila: centro=({pc_crop[0]:.0f}, {pc_crop[1]:.0f}), radio={pr:.0f}\n'
                f'Iris: centro=({ic_crop[0] if ic_crop else 0:.0f}, {ic_crop[1] if ic_crop else 0:.0f}), radio={ir if ir else 0:.0f}')
        except Exception as e:
            messagebox.showerror('Error', f'No se pudo guardar: {str(e)}')

    def _train_with_current_dataset(self):
        if not ai_available():
            messagebox.showwarning('Aviso',
                'PyTorch no esta instalado.\n'
                'Instala con: pip install torch torchvision')
            return
        
        img_dir = os.path.join('dataset', 'images')
        mask_dir = os.path.join('dataset', 'masks')
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            messagebox.showwarning('Aviso', 'No existe el dataset. Agrega imágenes primero.')
            return
        
        img_count = len(os.listdir(img_dir))
        mask_count = len(os.listdir(mask_dir))
        
        if img_count == 0 or mask_count == 0:
            messagebox.showwarning('Aviso', f'Dataset vacío (imágenes: {img_count}, máscaras: {mask_count}).')
            return
        
        if self.unet_model is None:
            result = messagebox.askyesno('Crear Modelo', 
                'No hay modelo cargado. ¿Deseas crear uno nuevo?')
            if result:
                self.unet_model = unet_create_model()
            else:
                return
        
        win = tk.Toplevel(self.root)
        win.title("Entrenar con Dataset")
        win.geometry("420x250")
        win.grab_set()

        ttk.Label(win, text=f"Entrenar con {img_count} imágenes del dataset",
                  font=('Arial',11,'bold')).pack(pady=8)

        ttk.Label(win, text="Epochs:").pack(pady=(10,0))
        epoch_var = tk.IntVar(value=15)
        ttk.Spinbox(win, from_=1, to=100, textvariable=epoch_var,
                    width=6).pack()

        progress = ttk.Progressbar(win, length=360, maximum=100)
        progress.pack(pady=12, padx=16)
        log_lbl = ttk.Label(win, text='', font=('Courier',8))
        log_lbl.pack()

        def do_train():
            def cb(ep, total, loss):
                pct = ep / total * 100
                progress['value'] = pct
                log_lbl.config(text=f'Epoch {ep}/{total}  loss={loss:.4f}')
                win.update()

            def run():
                try:
                    model = unet_train_model(
                        img_dir, mask_dir,
                        epochs=epoch_var.get(),
                        progress_callback=cb,
                        save_path='unet_eye.pth')
                    if model:
                        self.unet_model = model
                        log_lbl.config(text='Entrenamiento completado. Modelo guardado.')
                        messagebox.showinfo('Éxito', 'Modelo entrenado y guardado como unet_eye.pth')
                    else:
                        log_lbl.config(text='Error: sin datos o torch no disponible.')
                except Exception as e:
                    log_lbl.config(text=f'Error: {e}')

            threading.Thread(target=run, daemon=True).start()

        ttk.Button(win, text="Iniciar entrenamiento",
                   command=do_train).pack(pady=6)
        ttk.Button(win, text="Cerrar",
                   command=win.destroy).pack(pady=4)

    def _save_correction_to_dataset(self):
        if self.image is None and not self.is_video:
            messagebox.showwarning('Aviso', 'Carga una imagen o video primero.')
            return
        frame = None
        if self.is_video and self.video_frames:
            frame = self.video_frames[self.current_frame_idx]
        elif self.original_image is not None:
            frame = self.original_image
        
        if frame is None:
            messagebox.showwarning('Aviso', 'No hay frame disponible.')
            return
        
        pc = self.pupil_center
        pr = self.pupil_radius
        ic = self.iris_center
        ir = self.iris_radius
        
        if pc is None or pr is None:
            messagebox.showwarning('Aviso', 'No hay deteccion valida para guardar.')
            return
        
        if self.recorder.try_save(frame, pc, pr, ic, ir):
            self._update_dataset_label()
            messagebox.showinfo('Guardado', 'Frame guardado al dataset con correccion.')
        else:
            messagebox.showwarning('Aviso', 'No se pudo guardar. Activa la grabacion primero.')

    def _toggle_dataset(self):
        recording = self.recorder.toggle()
        state = "GRABANDO" if recording else "DETENIDO"
        self.status_bar.config(text=f'Dataset: {state}')
        self._update_dataset_label()

    def _update_dataset_label(self):
        n = self.recorder.count()
        rec = " [GRABANDO]" if self.recorder.is_recording else ""
        self.dataset_lbl_var.set(f'Dataset: {n} muestras{rec}')

    def open_training_window(self):
        if not ai_available():
            messagebox.showwarning('Aviso',
                'PyTorch no esta instalado.\n'
                'Instala con: pip install torch torchvision')
            return

        win = tk.Toplevel(self.root)
        win.title("Entrenar U-Net")
        win.geometry("420x320")
        win.grab_set()

        ttk.Label(win, text="Entrenar U-Net",
                  font=('Arial',11,'bold')).pack(pady=8)

        # Directorios
        fr = ttk.Frame(win); fr.pack(padx=16, fill='x')
        ttk.Label(fr, text="Carpeta imagenes:").grid(row=0,column=0,sticky='w',pady=3)
        img_var = tk.StringVar(value='dataset/images')
        ttk.Entry(fr, textvariable=img_var, width=28).grid(row=0,column=1,padx=4)
        ttk.Button(fr, text='...', width=3,
            command=lambda: img_var.set(
                filedialog.askdirectory() or img_var.get())
        ).grid(row=0,column=2)

        ttk.Label(fr, text="Carpeta mascaras:").grid(row=1,column=0,sticky='w',pady=3)
        mask_var = tk.StringVar(value='dataset/masks')
        ttk.Entry(fr, textvariable=mask_var, width=28).grid(row=1,column=1,padx=4)
        ttk.Button(fr, text='...', width=3,
            command=lambda: mask_var.set(
                filedialog.askdirectory() or mask_var.get())
        ).grid(row=1,column=2)

        ttk.Label(fr, text="Epochs:").grid(row=2,column=0,sticky='w',pady=3)
        epoch_var = tk.IntVar(value=10)
        ttk.Spinbox(fr, from_=1, to=100, textvariable=epoch_var,
                    width=6).grid(row=2,column=1,sticky='w',padx=4)

        progress = ttk.Progressbar(win, length=360, maximum=100)
        progress.pack(pady=12, padx=16)
        log_lbl = ttk.Label(win, text='', font=('Courier',8))
        log_lbl.pack()

        def do_train():
            def cb(ep, total, loss):
                pct = ep / total * 100
                progress['value'] = pct
                log_lbl.config(text=f'Epoch {ep}/{total}  loss={loss:.4f}')
                win.update()

            def run():
                try:
                    model = unet_train_model(
                        img_var.get(), mask_var.get(),
                        epochs=epoch_var.get(),
                        progress_callback=cb,
                        save_path='unet_eye.pth')
                    if model:
                        self.unet_model = model
                        log_lbl.config(text='Entrenamiento completado. Modelo guardado.')
                    else:
                        log_lbl.config(text='Error: sin datos o torch no disponible.')
                except Exception as e:
                    log_lbl.config(text=f'Error: {e}')

            threading.Thread(target=run, daemon=True).start()

        ttk.Button(win, text='Iniciar entrenamiento',
                   command=do_train).pack(pady=6)

    def open_single_image_training(self):
        if not ai_available():
            messagebox.showwarning('Aviso',
                'PyTorch no esta instalado.\n'
                'Instala con: pip install torch torchvision')
            return

        path = filedialog.askopenfilename(filetypes=[
            ('Imagenes', '*.jpg *.jpeg *.png *.bmp'),
            ('Todos', '*.*')])
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Error', 'No se pudo cargar la imagen.')
            return

        win = tk.Toplevel(self.root)
        win.title("Entrenar con Imagen Individual")
        win.geometry("500x400")
        win.grab_set()

        ttk.Label(win, text="Entrenar U-Net con Imagen",
                  font=('Arial',11,'bold')).pack(pady=8)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        scale = min(400/iw, 300/ih, 1.0)
        dw, dh = int(iw*scale), int(ih*scale)
        tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (dw, dh))))
        img_lbl = ttk.Label(win, image=tk_img)
        img_lbl.image = tk_img
        img_lbl.pack(pady=5)

        ctrl_fr = ttk.Frame(win)
        ctrl_fr.pack(pady=10)

        ttk.Label(ctrl_fr, text="Pupila - X:").grid(row=0,column=0,sticky='w')
        px_var = tk.IntVar(value=0)
        ttk.Entry(ctrl_fr, textvariable=px_var, width=6).grid(row=0,column=1,padx=4)
        ttk.Label(ctrl_fr, text="Y:").grid(row=0,column=2)
        py_var = tk.IntVar(value=0)
        ttk.Entry(ctrl_fr, textvariable=py_var, width=6).grid(row=0,column=3,padx=4)
        ttk.Label(ctrl_fr, text="R:").grid(row=0,column=4)
        pr_var = tk.IntVar(value=20)
        ttk.Entry(ctrl_fr, textvariable=pr_var, width=6).grid(row=0,column=5,padx=4)

        ttk.Label(ctrl_fr, text="Iris - X:").grid(row=1,column=0,sticky='w',pady=(5,0))
        ix_var = tk.IntVar(value=0)
        ttk.Entry(ctrl_fr, textvariable=ix_var, width=6).grid(row=1,column=1,padx=4,pady=(5,0))
        ttk.Label(ctrl_fr, text="Y:").grid(row=1,column=2,pady=(5,0))
        iy_var = tk.IntVar(value=0)
        ttk.Entry(ctrl_fr, textvariable=iy_var, width=6).grid(row=1,column=3,padx=4,pady=(5,0))
        ttk.Label(ctrl_fr, text="R:").grid(row=1,column=4,pady=(5,0))
        ir_var = tk.IntVar(value=60)
        ttk.Entry(ctrl_fr, textvariable=ir_var, width=6).grid(row=1,column=5,padx=4,pady=(5,0))

        status_var = tk.StringVar(value='Dibuja las circunferencias sobre la imagen')
        ttk.Label(win, textvariable=status_var, font=('Arial',8)).pack(pady=5)

        cv_canvas = tk.Canvas(win, width=dw, height=dh, bg='#333333')
        cv_canvas.pack()
        cv_canvas.create_image(0, 0, anchor='nw', image=tk_img)

        pupil_id = [None]
        iris_id = [None]

        def update_preview():
            if pupil_id[0]:
                cv_canvas.delete(pupil_id[0])
            if iris_id[0]:
                cv_canvas.delete(iris_id[0])
            try:
                px, py, pr = px_var.get(), py_var.get(), pr_var.get()
                ix, iy, ir = ix_var.get(), iy_var.get(), ir_var.get()
                sc = scale
                pupil_id[0] = cv_canvas.create_oval(
                    px*sc-pr, py*sc-pr, px*sc+pr, py*sc+pr,
                    outline='red', width=2)
                iris_id[0] = cv_canvas.create_oval(
                    ix*sc-ir, iy*sc-ir, ix*sc+ir, iy*sc+ir,
                    outline='blue', width=2)
            except:
                pass

        px_var.trace_add('write', lambda *a: update_preview())
        py_var.trace_add('write', lambda *a: update_preview())
        pr_var.trace_add('write', lambda *a: update_preview())
        ix_var.trace_add('write', lambda *a: update_preview())
        iy_var.trace_add('write', lambda *a: update_preview())
        ir_var.trace_add('write', lambda *a: update_preview())

        def on_click(e):
            mx, my = e.x, e.y
            sc = scale
            ix, iy = int(mx/sc), int(my/sc)
            ix_var.set(ix)
            iy_var.set(iy)
            status_var.set(f'Click en ({ix}, {iy}). Ajusta el radio.')

        cv_canvas.bind('<Button-1>', on_click)

        def do_train_single():
            try:
                pc = (px_var.get(), py_var.get())
                pr = pr_var.get()
                ic = (ix_var.get(), iy_var.get())
                ir = ir_var.get()
                if pr <= 0 or ir <= 0 or ir <= pr:
                    messagebox.showwarning('Aviso', 'Radio invalido.')
                    return
                mask = np.zeros((ih, iw), dtype=np.uint8)
                cv2.circle(mask, ic, ir, 128, 2)
                cv2.circle(mask, pc, pr, 255, -1)
                mask_dir = os.path.join('dataset', 'masks')
                img_dir = os.path.join('dataset', 'images')
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(img_dir, exist_ok=True)
                idx = len(os.listdir(img_dir))
                cv2.imwrite(os.path.join(img_dir, f'{idx:05d}.png'), img)
                cv2.imwrite(os.path.join(mask_dir, f'{idx:05d}.png'), mask)
                self._update_dataset_label()
                status_var.set(f'Imagen {idx} guardada. Agrega mas o cierra.')
                messagebox.showinfo('Guardado', f'Imagen guardada.\nYa puedes agregar mas o cerrar.')
            except Exception as e:
                messagebox.showerror('Error', str(e))

        ttk.Button(win, text='Guardar al Dataset', command=do_train_single).pack(pady=5)

    # ══════════════════════════════════════════════════════════════════
    #  Edicion de circunferencias con teclado y rueda
    # ══════════════════════════════════════════════════════════════════

    def _root_key_press(self, e):
        if not self.canvas.winfo_exists():
            return
        self._canvas_key_press(e)
        self.root.after_idle(lambda: self.canvas.focus_set())
        return 'break'

    def _canvas_key_press(self, e):
        step = 5
        ctrl = (e.state & 0x4) != 0
        
        if e.keysym == 'a' or e.keysym == 'A':
            if self.is_video and self.video_frames:
                idx = max(0, self.current_frame_idx - 1)
                self.current_frame_idx = idx
                self.video_slider.set(idx)
                self._show_frame(idx)
            self.canvas.focus_force()
            return 'break'
        
        if e.keysym == 'd' or e.keysym == 'D':
            if self.is_video and self.video_frames:
                idx = min(len(self.video_frames) - 1, self.current_frame_idx + 1)
                self.current_frame_idx = idx
                self.video_slider.set(idx)
                self._show_frame(idx)
            self.canvas.focus_force()
            return 'break'
        
        has_detection = (self.pupil_center is not None and self.pupil_radius is not None)
        
        if e.keysym in ('Up', 'Down', 'Left', 'Right'):
            if not has_detection:
                self.canvas.focus_force()
                return 'break'
            dx = dy = 0
            if e.keysym == 'Up':
                dy = -step
            elif e.keysym == 'Down':
                dy = step
            elif e.keysym == 'Left':
                dx = -step
            elif e.keysym == 'Right':
                dx = step
            
            pc = self.pupil_center
            ic = self.iris_center
            if pc:
                self.pupil_center = (pc[0] + dx, pc[1] + dy)
            if ic:
                self.iris_center = (ic[0] + dx, ic[1] + dy)
            
            if self.is_video and self.video_frames:
                idx = self.current_frame_idx
                self._apply_single_frame_edit(idx)
                self._pending_changes = True
                self.propagate_btn.configure(state='normal')
                self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            self._update_current_view()
            return 'break'
        
        if e.keysym in ('plus', 'equal', 'asterisk', 'kp_add', '+'):
            if not has_detection:
                self.canvas.focus_force()
                return 'break'
            if ctrl:
                if self.iris_center and self.iris_radius:
                    self.iris_radius = self.iris_radius + 2
            else:
                if self.pupil_radius:
                    self.pupil_radius = max(5, self.pupil_radius + 2)
            if self.is_video and self.video_frames:
                idx = self.current_frame_idx
                self._apply_single_frame_edit(idx)
                self._pending_changes = True
                self.propagate_btn.configure(state='normal')
                self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            self._update_current_view()
            self.canvas.focus_force()
            return 'break'
        
        if e.keysym in ('minus', 'underscore', 'kp_subtract', '-', '_'):
            if not has_detection:
                self.canvas.focus_force()
                return 'break'
            if ctrl:
                if self.iris_center and self.iris_radius:
                    self.iris_radius = max(15, self.iris_radius - 2)
            else:
                if self.pupil_radius:
                    self.pupil_radius = max(5, self.pupil_radius - 2)
            if self.is_video and self.video_frames:
                idx = self.current_frame_idx
                self._apply_single_frame_edit(idx)
                self._pending_changes = True
                self.propagate_btn.configure(state='normal')
                self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            self._update_current_view()
            self.canvas.focus_force()
            return 'break'
            if ctrl:
                if self.iris_center and self.iris_radius:
                    self.iris_radius = self.iris_radius + 2
            else:
                if self.pupil_radius:
                    self.pupil_radius = max(5, self.pupil_radius + 2)
            self._update_current_view()
            if self.is_video and self.video_frames:
                idx = self.current_frame_idx
                self._apply_single_frame_edit(idx)
                self._pending_changes = True
                self.propagate_btn.configure(state='normal')
                self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            self.canvas.focus_force()
            return 'break'
        
        if e.keysym in ('minus', 'underscore', 'kp_subtract', '-', '_'):
            if not has_detection:
                self.canvas.focus_force()
                return 'break'
            if ctrl:
                if self.iris_center and self.iris_radius:
                    self.iris_radius = max(15, self.iris_radius - 2)
            else:
                if self.pupil_radius:
                    self.pupil_radius = max(5, self.pupil_radius - 2)
            self._update_current_view()
            if self.is_video and self.video_frames:
                idx = self.current_frame_idx
                self._apply_single_frame_edit(idx)
                self._pending_changes = True
                self.propagate_btn.configure(state='normal')
                self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            return 'break'
        
        if self._edit_mode and self._selected_circle is not None:
            dx = dy = 0
            dr = 0
            
            if e.keysym == 'Up':
                dy = -step
            elif e.keysym == 'Down':
                dy = step
            elif e.keysym == 'Left':
                dx = -step
            elif e.keysym == 'Right':
                dx = step
            elif e.keysym in ('plus', 'equal', 'asterisk', 'kp_add', '+'):
                dr = 2
            elif e.keysym in ('minus', 'underscore', 'kp_subtract', '-', '_'):
                dr = -2
            
            if dx != 0 or dy != 0:
                if self._selected_circle == 'pupil' and self.pupil_center:
                    self.pupil_center = (self.pupil_center[0] + dx, self.pupil_center[1] + dy)
                elif self._selected_circle == 'iris' and self.iris_center:
                    self.iris_center = (self.iris_center[0] + dx, self.iris_center[1] + dy)
                self._show_editable_view()
            elif dr != 0:
                if self._selected_circle == 'pupil' and self.pupil_center:
                    self.pupil_radius = max(5, self.pupil_radius + dr)
                elif self._selected_circle == 'iris' and self.iris_center:
                    self.iris_radius = max(10, self.iris_radius + dr)
                self._show_editable_view()
            self.canvas.focus_force()
            return 'break'

    def _canvas_wheel(self, e):
        if not self._edit_mode or self._selected_circle is None:
            return
        
        dr = 2 if e.delta > 0 else -2
        
        if self._selected_circle == 'pupil' and self.pupil_center:
            new_r = max(5, self.pupil_radius + dr)
            self.pupil_radius = new_r
        elif self._selected_circle == 'iris' and self.iris_center:
            new_r = max(10, self.iris_radius + dr)
            self.iris_radius = new_r
        
        self._show_editable_view()
        
        if self.is_video and self.video_frames:
            idx = self.current_frame_idx
            self._apply_single_frame_edit(idx)
            self._pending_changes = True
            self.propagate_btn.configure(state='normal')
            self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')

    def _control_drag(self, e):
        """Arrastra los 3 handles de control."""
        if not self._control_dragging or not self._edit_mode:
            return

        ox, oy = self._canvas_offset
        sc = self._canvas_scale
        if sc == 0:
            return

        crop_offset_x, crop_offset_y = 0, 0
        if self.crop_rect:
            crop_offset_x, crop_offset_y = self.crop_rect[0], self.crop_rect[1]

        if self._control_dragging == 'move':
            # Mover ambos circulos juntos (concentricos)
            start_x, start_y, old_cx, old_cy = self._drag_start
            new_cx = old_cx + int((e.x - start_x) / sc)
            new_cy = old_cy + int((e.y - start_y) / sc)
            self.pupil_center = (new_cx, new_cy)
            self.iris_center  = (new_cx, new_cy)
            self._show_editable_view()

        elif self._control_dragging == 'pupil_size':
            # Solo cambia el radio de la pupila; el iris no se toca
            if self.pupil_center:
                px = self.pupil_center[0] - crop_offset_x
                py = self.pupil_center[1] - crop_offset_y
                cx_c = px * sc + ox
                cy_c = py * sc + oy
                dist = math.hypot(e.x - cx_c, e.y - cy_c)
                new_r = max(5, int(dist / sc))
                # No dejar que la pupila sea mas grande que el iris
                if self.iris_radius:
                    new_r = min(new_r, self.iris_radius - 5)
                self.pupil_radius = new_r
                self._show_editable_view()

        elif self._control_dragging == 'iris_size':
            # Solo cambia el radio del iris; la pupila no se toca
            if self.iris_center:
                ix = self.iris_center[0] - crop_offset_x
                iy = self.iris_center[1] - crop_offset_y
                cx_c = ix * sc + ox
                cy_c = iy * sc + oy
                dist = math.hypot(e.x - cx_c, e.y - cy_c)
                new_r = max(15, int(dist / sc))
                # No dejar que el iris sea mas pequeno que la pupila
                if self.pupil_radius:
                    new_r = max(new_r, self.pupil_radius + 5)
                self.iris_radius = new_r
                self._show_editable_view()

    def _update_circle_controls(self):
        """
        Dibuja handles de control sobre el canvas:
          M  (verde, encima del centro)   → mueve el circulo
          P  (rojo,  a la derecha)        → cambia solo el radio de la pupila
          I  (azul,  a la izquierda)     → cambia solo el radio del iris (solo si está habilitado)
        """
        self.canvas.delete('circle_controls')

        if not self._edit_mode:
            return

        ox, oy = self._canvas_offset
        sc     = self._canvas_scale
        
        crop_offset_x, crop_offset_y = 0, 0
        if self.crop_rect:
            crop_offset_x, crop_offset_y = self.crop_rect[0], self.crop_rect[1]

        # ── Dibujar circulos (punteados) ────────────────────────────────
        # Solo dibujar iris si está habilitado
        if self._analisis_iris_habilitado.get() and self.iris_center and self.iris_radius:
            ix = self.iris_center[0] - crop_offset_x
            iy = self.iris_center[1] - crop_offset_y
            cix = int(ix * sc + ox)
            ciy = int(iy * sc + oy)
            cir = int(self.iris_radius   * sc)
            self.canvas.create_oval(cix-cir, ciy-cir, cix+cir, ciy+cir,
                                    outline='#4488ff', width=2, dash=(6,3),
                                    tags=('circle_controls',))

        if self.pupil_center and self.pupil_radius:
            px = self.pupil_center[0] - crop_offset_x
            py = self.pupil_center[1] - crop_offset_y
            cpx = int(px * sc + ox)
            cpy = int(py * sc + oy)
            cpr = int(self.pupil_radius    * sc)
            self.canvas.create_oval(cpx-cpr, cpy-cpr, cpx+cpr, cpy+cpr,
                                    outline='#ff4444', width=2, dash=(6,3),
                                    tags=('circle_controls',))

        # Necesitamos al menos pupila para dibujar handles
        if not (self.pupil_center and self.pupil_radius):
            return

        px = self.pupil_center[0] - crop_offset_x
        py = self.pupil_center[1] - crop_offset_y
        cpx = int(px * sc + ox)
        cpy = int(py * sc + oy)
        cpr = int(self.pupil_radius    * sc)

        # Calcular posición del iris solo si está habilitado
        mostrar_iris = self._analisis_iris_habilitado.get()
        if mostrar_iris and self.iris_radius:
            cir = int(self.iris_radius * sc)
            cix = cpx
            ciy = cpy
        else:
            cir = cpr + 20
            cix = cpx
            ciy = cpy

        H = 14   # semialtura del handle
        W = 18   # semiancho del handle

        def make_handle(x, y, fill, text, tag):
            self.canvas.create_rectangle(x-W, y-H, x+W, y+H,
                                         fill=fill, outline='white', width=2,
                                         tags=('circle_controls', tag))
            self.canvas.create_text(x, y, text=text, fill='white',
                                    font=('Arial', 8, 'bold'),
                                    tags=('circle_controls', tag))

        # ── Handle M: mover (encima del centro) ─────────────────────────
        move_y = cpy - cir - 28
        make_handle(cpx, move_y, '#228833', 'M  mover', 'ctrl_move')

        # ── Handle P: tamano pupila (a la derecha de la pupila) ─────────
        pupil_hx = cpx + cpr + 28
        make_handle(pupil_hx, cpy, '#cc2222', 'P  pupila', 'ctrl_pupil_size')

        # ── Lineas guia hacia los handles ───────────────────────────────
        self.canvas.create_line(cpx, cpy - cir, cpx, move_y + H,
                                fill='#228833', dash=(3,3), width=1,
                                tags=('circle_controls',))
        self.canvas.create_line(cpx + cpr, cpy, pupil_hx - W, cpy,
                                fill='#cc2222', dash=(3,3), width=1,
                                tags=('circle_controls',))

        # ── Handle I: tamano iris (solo si está habilitado) ─────────────
        if mostrar_iris:
            iris_hx = cix - cir - 28
            make_handle(iris_hx, ciy, '#2255cc', 'I  iris', 'ctrl_iris_size')
            self.canvas.create_line(cix - cir, ciy, iris_hx + W, ciy,
                                    fill='#2255cc', dash=(3,3), width=1,
                                    tags=('circle_controls',))

        # ── Bindings ────────────────────────────────────────────────────
        def on_move_press(ev):
            c = self.pupil_center or self.iris_center
            if c:
                self._control_dragging = 'move'
                self._drag_start = (ev.x, ev.y, c[0], c[1])

        def on_pupil_size_press(ev):
            if self.pupil_center:
                self._control_dragging = 'pupil_size'
                self._drag_start = (ev.x, ev.y, 0, 0)

        def on_iris_size_press(ev):
            if self.iris_center and mostrar_iris:
                self._control_dragging = 'iris_size'
                self._drag_start = (ev.x, ev.y, 0, 0)

        def on_release(ev):
            self._control_dragging = None
            self._drag_start = None
            if self.is_video and self.video_frames:
                idx = self.current_frame_idx
                self._apply_single_frame_edit(idx)
                self._pending_changes = True
                self.propagate_btn.configure(state='normal')
                self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            self._show_editable_view()

        self.canvas.tag_bind('ctrl_move',       '<ButtonPress-1>',   on_move_press)
        self.canvas.tag_bind('ctrl_pupil_size', '<ButtonPress-1>',   on_pupil_size_press)
        if mostrar_iris:
            self.canvas.tag_bind('ctrl_iris_size',  '<ButtonPress-1>',   on_iris_size_press)

        for tag in ('ctrl_move', 'ctrl_pupil_size', 'ctrl_iris_size'):
            self.canvas.tag_bind(tag, '<B1-Motion>',       self._control_drag)
            self.canvas.tag_bind(tag, '<ButtonRelease-1>', on_release)

    # ══════════════════════════════════════════════════════════════════
    #  Archivo - Guardar/Abrir proyecto .pul
    # ══════════════════════════════════════════════════════════════════

    def save_project(self):
        if not self.video_frames and self.image is None:
            messagebox.showwarning('Aviso', 'No hay datos para guardar.')
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension='.pul',
            filetypes=[('Proyecto Pupilometria', '*.pul'), ('Todos', '*.*')])
        if not path:
            return
        
        try:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Guardar metadata
                crop_rect_data = None
                if self.crop_rect:
                    crop_rect_data = {
                        'x1': self.crop_rect[0],
                        'y1': self.crop_rect[1],
                        'x2': self.crop_rect[2],
                        'y2': self.crop_rect[3]
                    }
                
                metadata = {
                    'video_fps': self.video_fps,
                    'is_video': self.is_video,
                    'light_start_sec': self._light_start_sec,
                    'light_end_sec': self._light_end_sec,
                    'iris_mm': self.iris_mm_var.get(),
                    'crop_rect': crop_rect_data,
                    'has_timestamps': bool(self.video_timestamps),
                }
                zf.writestr('metadata.json', json.dumps(metadata))
                
                if self.video_timestamps:
                    zf.writestr('timestamps.json', json.dumps(self.video_timestamps))
                
                # Guardar CSV con resultados
                csv_data = []
                if self.is_video and self.frame_results:
                    for i, res in enumerate(self.frame_results):
                        if res:
                            pc = res.get('pupil_center', (0, 0))
                            pr = res.get('pupil_radius', 0)
                            ic = res.get('iris_center', (0, 0))
                            ir = res.get('iris_radius', 0)
                            csv_data.append({
                                'frame': i,
                                'time': i / self.video_fps,
                                'pupil_cx': pc[0] if pc else 0,
                                'pupil_cy': pc[1] if pc else 0,
                                'pupil_r': pr or 0,
                                'iris_cx': ic[0] if ic else 0,
                                'iris_cy': ic[1] if ic else 0,
                                'iris_r': ir or 0,
                            })
                
                if csv_data:
                    csv_io = io.StringIO()
                    if csv_data:
                        import csv as csv_module
                        writer = csv_module.DictWriter(csv_io, fieldnames=csv_data[0].keys())
                        writer.writeheader()
                        writer.writerows(csv_data)
                    zf.writestr('results.csv', csv_io.getvalue())
                
                # Guardar video o imagen
                if self.is_video and self.video_frames:
                    for i, frame in enumerate(self.video_frames):
                        _, img_encoded = cv2.imencode('.png', frame)
                        zf.writestr(f'video/frame_{i:05d}.png', img_encoded.tobytes())
                elif self.image is not None:
                    _, img_encoded = cv2.imencode('.png', self.image)
                    zf.writestr('image.png', img_encoded.tobytes())
            
            messagebox.showinfo('Guardado', f'Proyecto guardado en:\n{path}')
            self.status_bar.config(text=f'Proyecto guardado: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Error', f'No se pudo guardar:\n{str(e)}')

    def open_project(self):
        path = filedialog.askopenfilename(
            filetypes=[('Proyecto Pupilometria', '*.pul'), ('Todos', '*.*')])
        if not path:
            return
        
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                # Cargar metadata
                metadata = json.loads(zf.read('metadata.json'))
                self.video_fps = metadata.get('video_fps', 30.0)
                self._light_start_sec = metadata.get('light_start_sec', 1.0)
                self._light_end_sec = metadata.get('light_end_sec', 1.2)
                self.iris_mm_var.set(str(metadata.get('iris_mm', '')))
                
                crop_data = metadata.get('crop_rect')
                self.crop_rect = None
                self.video_frames_crop = None
                if crop_data:
                    self.crop_rect = (crop_data['x1'], crop_data['y1'], crop_data['x2'], crop_data['y2'])
                
                # Cargar frames
                self.video_frames = []
                self.frame_results = []
                video_files = sorted([n for n in zf.namelist() if n.startswith('video/frame_')])
                
                if video_files:
                    self.is_video = True
                    for fname in video_files:
                        frame_data = zf.read(fname)
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        self.video_frames.append(frame)
                        self.frame_results.append(None)
                    
                    self.video_timestamps = []
                    if 'timestamps.json' in zf.namelist():
                        self.video_timestamps = json.loads(zf.read('timestamps.json'))
                    else:
                        self.video_timestamps = []
                    
                    if self.crop_rect:
                        x1, y1, x2, y2 = self.crop_rect
                        self.video_frames_crop = [f[y1:y2, x1:x2] for f in self.video_frames]
                    
                    self.video_slider.configure(to=len(self.video_frames) - 1)
                    self.video_bar.grid()
                    self._set_video_controls_state('normal')
                    
                    # Cargar CSV de resultados
                    if 'results.csv' in zf.namelist():
                        csv_content = zf.read('results.csv').decode('utf-8')
                        lines = csv_content.strip().split('\n')
                        if len(lines) > 1:
                            import csv as csv_module
                            reader = csv_module.DictReader(io.StringIO(csv_content))
                            for row in reader:
                                idx = int(row['frame'])
                                if idx < len(self.frame_results):
                                    self.frame_results[idx] = {
                                        'pupil_center': (int(row['pupil_cx']), int(row['pupil_cy'])),
                                        'pupil_radius': int(row['pupil_r']),
                                        'iris_center': (int(row['iris_cx']), int(row['iris_cy'])),
                                        'iris_radius': int(row['iris_r']),
                                    }
                    
                    self._show_frame(0)
                    self._update_illum_labels()
                    messagebox.showinfo('Abierto', f'Video cargado: {len(self.video_frames)} frames')
                else:
                    # Cargar imagen
                    img_data = zf.read('image.png')
                    nparr = np.frombuffer(img_data, np.uint8)
                    self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.original_image = self.image.copy()
                    self.current_image = self.image.copy()
                    self._reset_detection()
                    self.is_video = False
                    self._show_image(self.image)
                    messagebox.showinfo('Abierto', 'Imagen cargada')
                
                self._apply_illum_times()
                self.status_bar.config(text=f'Proyecto abierto: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Error', f'No se pudo abrir:\n{str(e)}')

    def new_project(self):
        self._reset_detection()
        self._reset_video_state()
        self.image = None
        self.original_image = None
        self.current_image = None
        self.video_frames = []
        self.video_frames_crop = []
        self.frame_results = []
        self.canvas.delete('all')
        self._log('Nuevo proyecto. Carga una imagen o video.')
        self.status_bar.config(text='Nuevo proyecto')

    # ══════════════════════════════════════════════════════════════════
    #  Analisis - Resumen y Grafico de Evolucion
    # ══════════════════════════════════════════════════════════════════

    def show_summary(self):
        if not self._pupillometry_data:
            messagebox.showwarning('Aviso', 'Primero calcula la pupilometria.')
            return

        report = format_report(self._pupillometry_data)

        win = tk.Toplevel(self.root)
        win.title("Resumen de Pupilometría")
        win.geometry("520x600")
        win.resizable(True, True)

        fr = ttk.Frame(win)
        fr.pack(fill='both', expand=True, padx=10, pady=(10, 4))
        text = tk.Text(fr, wrap='none', font=('Courier', 10))
        vsb  = ttk.Scrollbar(fr, orient='vertical',   command=text.yview)
        hsb  = ttk.Scrollbar(fr, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        text.grid(row=0, column=0, sticky='nsew')
        fr.rowconfigure(0, weight=1); fr.columnconfigure(0, weight=1)

        text.insert('1.0', report)
        text.configure(state='disabled')

        ttk.Button(win, text='💾 Guardar TXT',
                   command=lambda: self._save_summary_txt(report)).pack(pady=(4, 8))

    def _save_summary_txt(self, report):
        path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Texto', '*.txt'), ('Todos', '*.*')])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report)
            messagebox.showinfo('Guardado', f'Resumen guardado en:\n{path}')

    def export_summary_txt(self):
        """Exporta directamente el resumen a TXT sin abrir ventana de vista previa."""
        if not self._pupillometry_data:
            messagebox.showwarning('Aviso', 'Primero calcula la pupilometria.')
            return
        report = format_report(self._pupillometry_data)
        self._save_summary_txt(report)

    def show_evolution_chart(self):
        if not self._pupillometry_data or not self.is_video:
            messagebox.showwarning('Aviso', 'Primero calcula la pupilometria con un video.')
            return

        data = self._pupillometry_data
        u    = data['unit']

        win = tk.Toplevel(self.root)
        win.title("Evolución de Pupilometría")
        win.geometry("950x700")

        try:
            import matplotlib
            matplotlib.use('TkAgg')
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure

            fig = Figure(figsize=(9.5, 6.8), dpi=100)
            fig.subplots_adjust(hspace=0.42, left=0.09, right=0.97, top=0.93, bottom=0.08)

            times  = data['times']
            p_diam = data['p_diam']
            i_diam = data['i_diam']
            ratio  = data['ratio']
            dp_s   = data['dp_s']
            t_on   = self._light_start_sec
            t_off  = self._light_end_sec

            def shade(ax):
                """Sombrea la zona de iluminacion."""
                ymin, ymax = ax.get_ylim()
                ax.fill_betweenx([ymin, ymax], t_on, t_off,
                                 alpha=0.15, color='gold', zorder=0)
                ax.axvline(t_on,  color='#00aa00', lw=1.2, ls='--',
                           label=f'Luz ON  ({t_on:.2f}s)')
                ax.axvline(t_off, color='#cc0000', lw=1.2, ls='--',
                           label=f'Luz OFF ({t_off:.2f}s)')
                ax.set_ylim(ymin, ymax)

            # ── Subplot 1: Diámetros pupila e iris ──────────────────────
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(times, p_diam, color='#cc2222', lw=1.8,
                     label=f'Pupila ({u})')
            i_valid = [(t, d) for t, d in zip(times, i_diam) if d is not None]
            if i_valid:
                ti, di = zip(*i_valid)
                ax1.plot(ti, di, color='#2255cc', lw=1.4, ls='--',
                         label=f'Iris ({u})')
            # Marcar Size y MIN
            idx_max = int(p_diam.index(max(p_diam)))
            idx_min = int(p_diam.index(min(p_diam)))
            ax1.scatter([times[idx_max]], [p_diam[idx_max]],
                        color='orange', s=50, zorder=5,
                        label=f'Size={data["size"]:.2f}{u}')
            ax1.scatter([times[idx_min]], [p_diam[idx_min]],
                        color='purple', s=50, zorder=5,
                        label=f'MIN={data["p_min"]:.2f}{u}')
            shade(ax1)
            ax1.set_ylabel(f'Diámetro ({u})')
            ax1.set_title('Diámetros de Pupila e Iris', fontsize=10, fontweight='bold')
            ax1.legend(fontsize=7, ncol=3, loc='upper right')
            ax1.grid(True, alpha=0.25)

            # ── Subplot 2: Ratio pupila/iris ─────────────────────────────
            ax2 = fig.add_subplot(3, 1, 2)
            r_times  = [t for t, r in zip(times, ratio) if r is not None]
            r_values = [r for r in ratio if r is not None]
            if r_times:
                ax2.plot(r_times, r_values, color='#228833', lw=1.6,
                         label='Ratio Pupila/Iris (%)')
                ax2.axhline(data['r_mean'], color='#228833', lw=0.8,
                            ls=':', label=f'Media {data["r_mean"]:.1f}%')
            shade(ax2)
            ax2.set_ylabel('Ratio (%)')
            ax2.set_title('Ratio Pupila / Iris', fontsize=10, fontweight='bold')
            ax2.legend(fontsize=7, ncol=3, loc='upper right')
            ax2.grid(True, alpha=0.25)

            # ── Subplot 3: Velocidad (derivada) ──────────────────────────
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(times, dp_s, color='#8833cc', lw=1.5,
                     label=f'Velocidad ({u}/s)')
            ax3.axhline(0, color='gray', lw=0.8)
            ax3.axhline(data['cv'],  color='#cc2222', lw=0.9, ls=':',
                        label=f'CV={data["cv"]:+.2f}')
            ax3.axhline(data['dv'],  color='#2255cc', lw=0.9, ls=':',
                        label=f'DV={data["dv"]:+.2f}')
            shade(ax3)
            ax3.set_xlabel('Tiempo (s)')
            ax3.set_ylabel(f'dD/dt ({u}/s)')
            ax3.set_title('Velocidad de Cambio del Diámetro', fontsize=10, fontweight='bold')
            ax3.legend(fontsize=7, ncol=4, loc='upper right')
            ax3.grid(True, alpha=0.25)

            canvas_widget = FigureCanvasTkAgg(fig, master=win)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill='both', expand=True)

            # Toolbar de navegación + botón guardar
            bar_frame = ttk.Frame(win)
            bar_frame.pack(fill='x')
            NavigationToolbar2Tk(canvas_widget, bar_frame)

            def save_chart():
                path = filedialog.asksaveasfilename(
                    parent=win,
                    defaultextension='.png',
                    filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'),
                               ('SVG', '*.svg'), ('Todos', '*.*')])
                if path:
                    fig.savefig(path, dpi=150, bbox_inches='tight')
                    messagebox.showinfo('Guardado', f'Gráfico guardado:\n{path}', parent=win)

            ttk.Button(bar_frame, text='💾 Guardar gráfico',
                       command=save_chart).pack(side='right', padx=8, pady=4)

        except ImportError:
            messagebox.showwarning('Aviso',
                'matplotlib no está instalado.\n'
                'Instala con: pip install matplotlib')

    # ══════════════════════════════════════════════════════════════════
    #  Frames Indeterminados (Ojo Cerrado/Pestañeando)
    # ══════════════════════════════════════════════════════════════════

    def _toggle_indeterminate_frame(self):
        """Marca/desmarca el frame actual como indeterminado (ojo cerrado/pestañeando)."""
        if not self.is_video or not self.video_frames:
            return
        
        idx = self.current_frame_idx
        
        if idx in self._indeterminate_frames:
            self._indeterminate_frames.discard(idx)
            self.indeterminate_label_var.set('')
            self.indeterminado_btn.configure(text='Indeterminado (Ojo Cerrado)')
        else:
            self._indeterminate_frames.add(idx)
            self.indeterminate_label_var.set('⚠ INDETERMINADO')
            self.indeterminado_btn.configure(text='Quitar Indeterminado')
        
        self._show_frame(idx)

    def _update_indeterminate_button(self):
        """Actualiza el estado del boton de indeterminado segun si hay video."""
        if self.is_video and self.video_frames:
            self.indeterminado_btn.configure(state='normal')
            idx = self.current_frame_idx
            if idx in self._indeterminate_frames:
                self.indeterminate_label_var.set('⚠ INDETERMINADO')
                self.indeterminado_btn.configure(text='Quitar Indeterminado')
            else:
                self.indeterminate_label_var.set('')
                self.indeterminado_btn.configure(text='Indeterminado (Ojo Cerrado)')
        else:
            self.indeterminado_btn.configure(state='disabled')
            self.indeterminate_label_var.set('')

    def _is_frame_indeterminate(self, idx):
        """Retorna True si el frame esta marcado como indeterminado."""
        return idx in self._indeterminate_frames

    def _reset_indeterminate_frames(self):
        """Limpia todos los frames indeterminados."""
        self._indeterminate_frames.clear()
        self.indeterminate_label_var.set('')
        self.indeterminado_btn.configure(text='Indeterminado (Ojo Cerrado)')

    # ══════════════════════════════════════════════════════════════════
    #  Recorte
    # ══════════════════════════════════════════════════════════════════

    def open_crop_window(self):
        if not self.is_video or not self.video_frames:
            messagebox.showwarning('Aviso',
                'Primero carga un video con "Cargar Video".')
            return
        ref = self.video_frames[0]
        ih, iw = ref.shape[:2]
        win = tk.Toplevel(self.root)
        win.title('Recortar Video')
        win.resizable(False, False); win.grab_set()
        max_w, max_h = 900, 620
        scale = min(max_w/iw, max_h/ih, 1.0)
        dw, dh = int(iw*scale), int(ih*scale)
        rgb  = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        tk_bg = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb,(dw,dh))))
        ttk.Label(win, text='Arrastra para seleccionar la region. Confirmar para aplicar.',
                  font=('Arial',9)).pack(pady=(8,2),padx=10)
        cv = tk.Canvas(win, width=dw, height=dh, cursor='tcross')
        cv.pack(padx=10, pady=4)
        cv.create_image(0,0,anchor='nw',image=tk_bg); cv.image=tk_bg
        rect_id=[None]; drag_start=[None]; coords=[None]
        if self.crop_rect:
            cx1,cy1,cx2,cy2 = self.crop_rect
            dx1,dy1,dx2,dy2 = [int(v*scale) for v in [cx1,cy1,cx2,cy2]]
            coords[0]=(dx1,dy1,dx2,dy2)
            rect_id[0]=cv.create_rectangle(dx1,dy1,dx2,dy2,
                outline='#00e000',width=2,dash=(6,3))
            lbl_sz = ttk.Label(win, text=f'{cx2-cx1}x{cy2-cy1} px')
        else:
            lbl_sz = ttk.Label(win, text='Sin seleccion')
        lbl_sz.pack()
        def on_press(e):
            drag_start[0]=(e.x,e.y)
            if rect_id[0]: cv.delete(rect_id[0])
            rect_id[0]=cv.create_rectangle(e.x,e.y,e.x,e.y,
                outline='#00e000',width=2,dash=(6,3))
        def on_drag(e):
            if not drag_start[0]: return
            x0,y0=drag_start[0]; cv.coords(rect_id[0],x0,y0,e.x,e.y)
            lbl_sz.configure(text=f'{int(abs(e.x-x0)/scale)}x{int(abs(e.y-y0)/scale)} px')
        def on_release(e):
            if not drag_start[0]: return
            x0,y0=drag_start[0]
            coords[0]=(min(x0,e.x),min(y0,e.y),max(x0,e.x),max(y0,e.y))
            drag_start[0]=None
        cv.bind('<ButtonPress-1>',on_press)
        cv.bind('<B1-Motion>',on_drag)
        cv.bind('<ButtonRelease-1>',on_release)
        btn_fr=ttk.Frame(win); btn_fr.pack(pady=8)
        def confirm():
            if not coords[0]:
                messagebox.showwarning('Aviso','Dibuja una region primero.',parent=win); return
            dx1,dy1,dx2,dy2=coords[0]
            ix1=max(0,int(dx1/scale)); iy1=max(0,int(dy1/scale))
            ix2=min(iw,int(dx2/scale)); iy2=min(ih,int(dy2/scale))
            if (ix2-ix1)<10 or (iy2-iy1)<10:
                messagebox.showwarning('Aviso','Seleccion demasiado pequeña.',parent=win); return
            self.crop_rect=(ix1,iy1,ix2,iy2)
            self.video_frames_crop=[f[iy1:iy2,ix1:ix2] for f in self.video_frames]
            self.frame_results=[None]*len(self.video_frames)
            try: self.crop_info_var.set(f'{ix2-ix1}x{iy2-iy1} px ({ix1},{iy1})-({ix2},{iy2})')
            except: pass
            self.status_bar.config(text=f'Recorte: {ix2-ix1}x{iy2-iy1} px')
            win.destroy(); self._show_frame(self.current_frame_idx)
        def full_reset():
            self.crop_rect=None; self.video_frames_crop=[]
            self.frame_results=[None]*len(self.video_frames)
            try: self.crop_info_var.set('Sin recorte activo')
            except: pass
            win.destroy(); self._show_frame(self.current_frame_idx)
        ttk.Button(btn_fr,text='Confirmar recorte',command=confirm,width=20).grid(row=0,column=0,padx=4)
        ttk.Button(btn_fr,text='Cancelar',command=win.destroy,width=12).grid(row=0,column=1,padx=4)
        ttk.Button(btn_fr,text='Quitar recorte',command=full_reset,width=14).grid(row=0,column=2,padx=4)

    def toggle_crop_mode(self):
        self._crop_mode = not self._crop_mode
        if self._crop_mode:
            self.canvas.configure(cursor='tcross')
            self.crop_btn.configure(text='Cancelar recorte')
            self.status_bar.config(text='MODO RECORTE — arrastra sobre el video.')
        else:
            self.canvas.configure(cursor='crosshair')
            self.crop_btn.configure(text='Activar recorte')
            self.status_bar.config(text='Modo recorte cancelado.')
            if self._crop_rect_id:
                self.canvas.delete(self._crop_rect_id); self._crop_rect_id=None

    def reset_crop(self):
        self.crop_rect=None; self._crop_mode=False
        self.video_frames_crop=[]; self.frame_results=[None]*len(self.video_frames)
        if self._crop_rect_id:
            self.canvas.delete(self._crop_rect_id); self._crop_rect_id=None
        try:
            self.crop_btn.configure(text='Activar recorte')
            self.crop_info_var.set('Sin recorte activo')
        except: pass
        if self.is_video and self.video_frames: self._show_frame(self.current_frame_idx)
        elif self.image is not None: self._show_image(self.image)
        self.status_bar.config(text='Recorte eliminado.')

    def _canvas_press(self, e):
        if self._edit_mode:
            # Verificar si se hizo click en alguno de los 3 handles
            items = self.canvas.find_overlapping(e.x-22, e.y-22, e.x+22, e.y+22)
            control_found = False
            for item in items:
                tags = self.canvas.gettags(item)
                if 'ctrl_move' in tags:
                    c = self.pupil_center or self.iris_center
                    if c:
                        self._control_dragging = 'move'
                        self._drag_start = (e.x, e.y, c[0], c[1])
                        control_found = True
                    break
                elif 'ctrl_pupil_size' in tags:
                    if self.pupil_center:
                        self._control_dragging = 'pupil_size'
                        self._drag_start = (e.x, e.y, 0, 0)
                        control_found = True
                    break
                elif 'ctrl_iris_size' in tags:
                    if self.iris_center:
                        self._control_dragging = 'iris_size'
                        self._drag_start = (e.x, e.y, 0, 0)
                        control_found = True
                    break
            if not control_found:
                self._edit_press(e)
        elif self._crop_mode:
            self._crop_press(e)

    def _canvas_motion(self, e):
        if self._edit_mode:
            if self._control_dragging:
                self._control_drag(e)
            elif self._drag_start is not None:
                self._edit_drag(e)
        elif self._crop_mode:
            self._crop_drag(e)

    def _canvas_release(self, e):
        if self._edit_mode:
            if self._control_dragging:
                self._control_dragging = None
                self._drag_start = None
                if self.is_video and self.video_frames:
                    idx = self.current_frame_idx
                    self._apply_single_frame_edit(idx)
                    self._pending_changes = True
                    self.propagate_btn.configure(state='normal')
                    self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
                self._update_current_view()
            elif self._drag_start is not None:
                self._edit_release(e)
        elif self._crop_mode:
            self._crop_release(e)
    
    def _crop_press(self, e):
        self._crop_start=(e.x,e.y)
        if self._crop_rect_id: self.canvas.delete(self._crop_rect_id)
        self._crop_rect_id=self.canvas.create_rectangle(
            e.x,e.y,e.x,e.y,outline='#00ff00',width=2,dash=(6,3))

    def _crop_drag(self, e):
        if not self._crop_mode or not self._crop_start: return
        x0,y0=self._crop_start; self.canvas.coords(self._crop_rect_id,x0,y0,e.x,e.y)
    def _crop_release(self, e):
        if not self._crop_mode or not self._crop_start: return
        x0c,y0c=self._crop_start; self._crop_start=None
        self._crop_mode=False; self.canvas.configure(cursor='crosshair')
        ox,oy=self._canvas_offset; sc=self._canvas_scale
        if sc==0: return
        lx=min(x0c,e.x); rx=max(x0c,e.x)
        ty=min(y0c,e.y); by=max(y0c,e.y)
        ix1=int((lx-ox)/sc); iy1=int((ty-oy)/sc)
        ix2=int((rx-ox)/sc); iy2=int((by-oy)/sc)
        ref=self.video_frames[0] if self.is_video and self.video_frames else self.image
        if ref is None: return
        ih,iw=ref.shape[:2]
        ix1=max(0,min(ix1,iw-1)); iy1=max(0,min(iy1,ih-1))
        ix2=max(0,min(ix2,iw));   iy2=max(0,min(iy2,ih))
        if (ix2-ix1)<20 or (iy2-iy1)<20:
            self.status_bar.config(text='Recorte demasiado pequeño.')
            if self._crop_rect_id:
                self.canvas.delete(self._crop_rect_id); self._crop_rect_id=None
            return
        self.crop_rect=(ix1,iy1,ix2,iy2)
        if self.is_video and self.video_frames:
            self.video_frames_crop=[f[iy1:iy2,ix1:ix2] for f in self.video_frames]
            self.frame_results=[None]*len(self.video_frames)
            self._show_frame(self.current_frame_idx)
        try: self.crop_info_var.set(f'{ix2-ix1}x{iy2-iy1} px')
        except: pass
        self.status_bar.config(text=f'Recorte: ({ix1},{iy1})->({ix2},{iy2})')

    def _working_frame(self, idx):
        if self.video_frames_crop: return self.video_frames_crop[idx]
        return self.video_frames[idx]

    # ══════════════════════════════════════════════════════════════════
    #  Carga
    # ══════════════════════════════════════════════════════════════════

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=self.supported_img)
        if not path: return
        try:
            img = cv2.imread(path)
            if img is None:
                from PIL import Image as PILImg
                img=cv2.cvtColor(np.array(PILImg.open(path)),cv2.COLOR_RGB2BGR)
            if img is None: messagebox.showerror('Error','No se pudo cargar.'); return
            self._reset_video_state()
            self.image=img
            self.original_image=img.copy()
            self.current_image=img.copy()
            self._reset_detection(); self._show_image(img)
            h,w=img.shape[:2]; sz=os.path.getsize(path)/1024
            self._log(f'IMAGEN CARGADA\n{"="*34}\n'
                      f'Archivo: {os.path.basename(path)}\n'
                      f'Dimensiones: {w}x{h} px\nTamaño: {sz:.1f} KB\n\n'
                      f'Pulsa Detectar para analizar.')
            self.status_bar.config(text=f'Imagen: {os.path.basename(path)}')
        except Exception as e: messagebox.showerror('Error',str(e))

    def _show_video_settings_window(self, path, after_accept_callback):
        """Muestra la ventana de ajustes de video antes de procesar."""
        fase_vars = {
            'basal_fps': self._fase_basal_fps,
            'basal_dur': self._fase_basal_dur,
            'contraccion_fps': self._fase_contraccion_fps,
            'contraccion_dur': self._fase_contraccion_dur,
            'relajacion_fps': self._fase_relajacion_fps,
            'relajacion_dur': self._fase_relajacion_dur,
        }
        flash_vars = {
            'auto': self._flash_detection_auto,
            'calibration_frames': self._flash_calibration_frames,
            'change_threshold': self._flash_change_threshold,
            'smoothing_window': self._flash_smoothing_window,
            'flash_start_sec': self._flash_start_sec,
            'flash_end_sec': self._flash_end_sec,
        }
        create_video_settings_window(
            parent=self.root,
            video_path=path,
            fase_vars=fase_vars,
            flash_vars=flash_vars,
            on_accept_callback=after_accept_callback,
            status_bar=self.status_bar
        )

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=self.supported_vid)
        if not path: return
        self._reset_detection(); self._reset_video_state()
        self._reset_indeterminate_frames()
        self.status_bar.config(text='Cargando video...'); self.root.update()
        
        self._show_video_settings_window(path, self._process_video_after_settings)

    def _process_video_after_settings(self, path):
        """
        Procesa el video después de que el usuario acepta los ajustes.
        
        Utiliza procesamiento en streaming para evitar problemas de memoria
        con videos grandes.
        """
        try:
            self.status_bar.config(text='Procesando video...')
            self.root.update()
            
            phases_config = {
                'basal': {
                    'fps': self._fase_basal_fps.get(),
                    'duration': self._fase_basal_dur.get()
                },
                'contraccion': {
                    'fps': self._fase_contraccion_fps.get(),
                    'duration': self._fase_contraccion_dur.get()
                },
                'relajacion': {
                    'fps': self._fase_relajacion_fps.get(),
                    'duration': self._fase_relajacion_dur.get()
                }
            }
            
            flash_config = None
            if self._flash_detection_auto.get():
                flash_config = {
                    'auto': True,
                    'calibration_frames': self._flash_calibration_frames.get(),
                    'change_threshold': self._flash_change_threshold.get(),
                    'smoothing_window': self._flash_smoothing_window.get(),
                    'trim_before_ms': self._flash_trim_before_ms.get(),
                    'trim_after_ms': self._flash_trim_after_ms.get()
                }
            
            def progress_callback(current, total):
                if current % 100 == 0:
                    pct = int(current / total * 100) if total > 0 else 0
                    self.status_bar.config(text=f'Procesando video... {pct}%')
                    self.root.update()
            
            processor = VideoProcessor()
            
            video_result = processor.process(
                video_path=path,
                phases_config=phases_config,
                flash_config=flash_config,
                progress_callback=progress_callback
            )
            
            if not video_result.frames:
                messagebox.showerror('Error', 'No se encontraron frames validos.')
                return
            
            self.video_frames = video_result.frames
            self.video_timestamps = video_result.timestamps
            self.video_fps = video_result.fps
            self.frame_results = [None] * len(self.video_frames)
            self.is_video = True
            
            video_info = video_result.video_info
            flash_times = video_result.flash_times
            
            dur = video_result.duration if video_result.duration > 0 else 0
            
            if len(video_result.phase_bounds) >= 3:
                fase2_bounds = video_result.phase_bounds[1]
                self._light_start_sec = fase2_bounds.start
                self._light_end_sec = fase2_bounds.end
            else:
                self._light_start_sec = 1.0
                self._light_end_sec = 1.2

            # Override with manual flash times if set
            manual_start = self._flash_start_sec.get()
            manual_end = self._flash_end_sec.get()
            if manual_start > 0 or manual_end > 0:
                if manual_start > 0:
                    self._light_start_sec = manual_start
                if manual_end > 0:
                    self._light_end_sec = manual_end

            self._light_start_frac = self._light_start_sec / dur if dur > 0 else 0.0
            self._light_end_frac = self._light_end_sec / dur if dur > 0 else 1.0
            
            self._update_illum_labels()
            
            self.video_slider.configure(to=len(self.video_frames) - 1)
            self.video_slider.set(0)
            self.video_bar.grid()
            self._set_video_controls_state('normal')
            
            if self.kalman:
                try:
                    self.kalman.reset()
                except Exception:
                    pass
            
            self._show_frame(0)
            
            flash_info = ""
            if flash_times and flash_times.get('on_ms'):
                flash_on = flash_times['on_ms'] / 1000.0
                flash_off = flash_times.get('off_ms', 0) / 1000.0 if flash_times.get('off_ms') else None
                flash_info = f"Flash: {flash_on:.2f}s"
                if flash_off:
                    flash_info += f" - {flash_off:.2f}s"
                flash_info += "\n"
                
                self._log(f'FLASH DETECTADO\n{"="*30}\n'
                         f'Inicio: {flash_on:.3f}s\n'
                         f'Fin: {flash_off:.3f}s\n'
                         f'Brillo base: {flash_times.get("baseline", 0):.1f}\n\n')
            
            quality_info = f"Calidad: {min(video_info.height, video_info.width)}p"
            
            fase1_fps = phases_config['basal']['fps']
            fase2_fps = phases_config['contraccion']['fps']
            fase3_fps = phases_config['relajacion']['fps']
            fase1_dur = phases_config['basal']['duration']
            fase2_dur = phases_config['contraccion']['duration']
            fase3_dur = phases_config['relajacion']['duration']
            
            self._log(f'VIDEO CARGADO\n{"="*34}\n'
                      f'Archivo: {os.path.basename(path)}\n'
                      f'FPS orig: {video_info.fps:.1f} | FPS efec: {self.video_fps:.1f}\n'
                      f'Frames orig: {video_info.total_frames} | Frames sel: {len(self.video_frames)}\n'
                      f'Fase 1 (Basal): {fase1_fps} fps, {fase1_dur}s\n'
                      f'Fase 2 (Iluminac): {fase2_fps} fps, {fase2_dur}s\n'
                      f'Fase 3 (Relajac): {fase3_fps} fps, {fase3_dur}s\n'
                      f'Duracion: {dur:.2f} s\n'
                      f'{quality_info}\n'
                      f'{flash_info}'
                      f'Mueve la barra para navegar.')
            
            self.status_bar.config(
                text=f'{os.path.basename(path)} | {len(self.video_frames)} frames | {self.video_fps:.1f} fps'
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror('Error', f'Error al procesar video:\n{str(e)}')

    # ══════════════════════════════════════════════════════════════════
    #  Deteccion
    # ══════════════════════════════════════════════════════════════════

    def _detect_frame(self, frame, prev_result=None, detectar_iris=False):
        """
        Detecta pupila (y opcionalmente iris) en un frame.
        
        Args:
            frame: Frame a procesar
            prev_result: Resultado del frame anterior (para ROI)
            detectar_iris: Si True, detecta iris. Si False, solo pupila.
        
        NOTA: La IA (U-Net) solo se usa para detectar pupila, nunca para iris ni para calibración de regla.
        """
        pupil_alg = self.pupil_alg_var.get()
        
        # IA (U-Net) para pupila si está seleccionada o habilitada
        usar_ia_para_pupila = (pupil_alg == 'ia') or (self._ai_enabled and self.unet_model is not None)
        
        # Solo crear detector si se necesita iris
        iris_mode = None
        if detectar_iris:
            iris_alg = self.iris_alg_var.get()
            iris_mode_map = {
                'gradient': adv_detector.DetectionMode.GRADIENT_BASED,
                'hough': adv_detector.DetectionMode.CIRCULAR_HOUGH,
                'ellipse': adv_detector.DetectionMode.ELLIPSE_FIT,
                'profile': adv_detector.DetectionMode.INTENSITY_PROFILE
            }
            iris_mode = iris_mode_map.get(iris_alg, adv_detector.DetectionMode.GRADIENT_BASED)
        
        detector = adv_detector.AdvancedEyeDetector(
            pupil_algorithm=pupil_alg,
            iris_mode=iris_mode if iris_mode else adv_detector.DetectionMode.GRADIENT_BASED,
            expected_pupil_range=(0.1, 0.7),
            expected_iris_range=(0.3, 0.9)
        )
        
        h, w = frame.shape[:2]
        roi_frame = frame
        offset_x, offset_y = 0, 0
        
        # IA para pupila si está habilitada
        if usar_ia_para_pupila:
            try:
                mask = unet_segment(self.unet_model, roi_frame)
                if mask is not None:
                    pc, pr = extract_pupil_from_mask(mask)
                    if pc and pr:
                        # IA detectó pupila exitosamente
                        pc = (int(pc[0]) + offset_x, int(pc[1]) + offset_y)
                        # Usar detección clásica para iris si se pidió
                        ir = None
                        if detectar_iris:
                            iris_result = detector.detect_iris(
                                cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY),
                                {'center': pc, 'diameter': pr * 2}
                            )
                            if iris_result:
                                ir = int(iris_result['radius'])
                        return {'pupil_center': pc, 'pupil_radius': pr,
                                'iris_center': pc, 'iris_radius': ir,
                                'px_to_mm': None}
            except Exception:
                # Si falla la IA, continuar con detección clásica
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
        
        result = detector.detect(roi_frame)
        
        pc = pr = ic = ir = None
        
        # Solo detectar pupila (NO IA para iris ni regla)
        if result.pupil_center and result.pupil_radius:
            pc = (int(result.pupil_center[0]) + offset_x, int(result.pupil_center[1]) + offset_y)
            pr = result.pupil_radius
        
        # Si se pide iris, detectarlo. Si no, estimar como 2.5x la pupila
        if detectar_iris and result.iris_radius:
            ir = result.iris_radius
        elif pr is not None:
            # Estimar iris como 2.5x la pupila si no se detecta explícitamente
            ir = int(pr * 2.5)
        
        if pc is None:
            pc, pr, ic, ir = detect_all(frame, False, 50, 5, 15)
        
        if pc is not None:
            ic = pc
            
            if ir is None and pr is not None:
                ir = int(pr * 2.5)
        
        if pc is not None:
            try:
                pc = self.kalman.update(pc[0], pc[1])
                ic = pc
            except Exception:
                pass

        # Calcular px_to_mm: usar calibración de regla si existe, o estimación de iris
        px_to_mm = None
        if self._escala_px_mm is not None:
            # Usar calibración de regla
            px_to_mm = 1.0 / self._escala_px_mm
        else:
            # Usar estimación manual de iris
            try:
                iris_mm = float(self.iris_mm_var.get())
                if iris_mm > 0 and ir:
                    px_to_mm = iris_mm / (ir * 2)
            except ValueError:
                pass

        if self.recorder.is_recording:
            if self.recorder.try_save(frame, pc, pr, ic, ir):
                self._update_dataset_label()

        return {'pupil_center': pc, 'pupil_radius': pr,
                'iris_center':  ic, 'iris_radius':  ir,
                'px_to_mm':     px_to_mm}

    def _detect_all_frames(self):
        """Detecta pupila/iris en todos los frames del video."""
        if not self.is_video or not self.video_frames:
            messagebox.showwarning('Aviso', 'Carga un video primero.')
            return
        
        n = len(self.video_frames)
        total = n
        self.status_bar.config(text=f'Detectando en {n} frames...')
        self.root.update()
        
        crop_offset_x, crop_offset_y = 0, 0
        if self.crop_rect:
            crop_offset_x, crop_offset_y = self.crop_rect[0], self.crop_rect[1]
        
        detectar_iris = self._analisis_iris_habilitado.get()
        
        for i in range(n):
            if i % 10 == 0:
                pct = i / n * 100
                self.status_bar.config(text=f'Detectando... {i}/{n} ({pct:.0f}%)')
                self.root.update()
            
            prev = self._get_prev_result(i) if i > 0 else None
            work = self._working_frame(i)
            
            result = self._detect_frame(work, prev, detectar_iris=detectar_iris)
            self.frame_results[i] = result
            
            if prev is None and result.get('pupil_center'):
                pass
        
        self._show_frame(self.current_frame_idx)
        self.status_bar.config(text=f'Deteccion completada en {n} frames.')
        messagebox.showinfo('Completado', f'Deteccion completada en {n} frames.')

    def full_auto_detection(self):
        # Modo video: detectar en frame actual
        if self.is_video:
            idx = self.current_frame_idx
            self.status_bar.config(text='Detectando...')
            self.root.update()
            
            work = self._working_frame(idx)
            crop_offset_x, crop_offset_y = 0, 0
            if self.crop_rect:
                crop_offset_x, crop_offset_y = self.crop_rect[0], self.crop_rect[1]
            
            # Detectar la regla en el frame de video
            regla_resultado = self._detectar_regla_en_frame(work)
            
            if regla_resultado:
                # Se detectó la regla
                self._calibracion_resultado = regla_resultado
                self._escala_px_mm = regla_resultado['escala_px_mm']
                self._error_calibracion_mm = regla_resultado['error_estimado_mm']
                self.pixel_to_mm = 1.0 / regla_resultado['escala_px_mm']
                
                # Actualizar info de calibración
                info = f"Escala: {regla_resultado['escala_px_mm']:.4f} px/mm"
                info += f" | Error: ±{regla_resultado['error_estimado_mm']:.3f} mm"
                self.calibracion_info_var.set(info)
                self.calibracion_error_var.set(f"Calidad: {regla_resultado['calidad']*100:.1f}%")
            
            # Detectar iris SOLO si está habilitado
            detectar_iris = self._analisis_iris_habilitado.get()
            result = self._detect_frame(work, detectar_iris=detectar_iris)
            self.frame_results[idx] = result
            
            # Actualizar variables de instancia
            pc = result['pupil_center']
            pr = result['pupil_radius']
            ic = result['iris_center']
            ir = result['iris_radius']
            
            if pc:
                pc = (pc[0] + crop_offset_x, pc[1] + crop_offset_y)
            if ic:
                ic = (ic[0] + crop_offset_x, ic[1] + crop_offset_y)
            
            self.pupil_center = pc
            self.pupil_radius = pr
            self.iris_center = ic
            self.iris_radius = ir
            
            # Si se detectó iris, guardarlo
            if detectar_iris and ir:
                self._iris_resultado_actual = {
                    'iris_center': ic,
                    'iris_radius': ir
                }
            
            # Mostrar el frame con las detecciones
            self._show_frame(idx)
            
            regla_msg = " | Regla detectada" if regla_resultado else ""
            self.status_bar.config(text=f'Deteccion completada.{regla_msg}')
            return
        
        # Modo imagen
        if self.image is None:
            messagebox.showwarning('Aviso', 'Primero carga una imagen.')
            return
        self.status_bar.config(text='Detectando...')
        self.root.update()
        
        work = self.image
        crop_offset_x, crop_offset_y = 0, 0
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            work = self.image[y1:y2, x1:x2]
            crop_offset_x, crop_offset_y = x1, y1
        
        # Detectar la regla en el frame
        regla_resultado = self._detectar_regla_en_frame(work)
        imagen_con_regla = None
        
        if regla_resultado:
            # Se detectó la regla
            self._calibracion_resultado = regla_resultado
            self._escala_px_mm = regla_resultado['escala_px_mm']
            self._error_calibracion_mm = regla_resultado['error_estimado_mm']
            self.pixel_to_mm = 1.0 / regla_resultado['escala_px_mm']
            imagen_con_regla = regla_resultado['imagen_procesada']
            
            # Actualizar info de calibración
            info = f"Escala: {regla_resultado['escala_px_mm']:.4f} px/mm"
            info += f" | Error: ±{regla_resultado['error_estimado_mm']:.3f} mm"
            self.calibracion_info_var.set(info)
        
        # Detectar iris SOLO si está habilitado
        detectar_iris = self._analisis_iris_habilitado.get()
        result = self._detect_frame(work, detectar_iris=detectar_iris)
        pc = result['pupil_center']
        ic = result['iris_center']
        pr = result['pupil_radius']
        ir = result['iris_radius']
        
        # Si NO está habilitado el análisis de iris, no mostrar iris
        if not detectar_iris:
            ic = None
            ir = None
        
        if pc:
            pc = (pc[0] + crop_offset_x, pc[1] + crop_offset_y)
        if ic:
            ic = (ic[0] + crop_offset_x, ic[1] + crop_offset_y)
        
        self.pupil_center = pc
        self.pupil_radius = pr
        self.iris_center = ic
        self.iris_radius = ir
        
        # Si se detectó iris, guardarlo
        if detectar_iris and ir:
            self._iris_resultado_actual = {
                'iris_center': ic,
                'iris_radius': ir
            }
        
        line_width = self.line_width_var.get()
        
        # Combinar: dibujar regla detectada + pupila/iris
        if imagen_con_regla is not None:
            annotated = imagen_con_regla.copy()
        else:
            annotated = self.image.copy()
        
        # Dibujar pupila e iris sobre la imagen
        annotated = draw_detections(annotated, pc, pr, ic, ir, line_width)
        
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 1)
        
        # Agregar información de escala calibrada a la imagen
        if self._escala_px_mm is not None and pc and pr:
            px_to_mm = 1.0 / self._escala_px_mm
            pupil_diam_mm = pr * 2 * px_to_mm
            info_text = f"Pupila: {pupil_diam_mm:.2f} mm (diam.)"
            if ir and detectar_iris:
                iris_diam_mm = ir * 2 * px_to_mm
                info_text += f" | Iris: {iris_diam_mm:.2f} mm"
            if self._error_calibracion_mm:
                info_text += f" | Error: +/-{self._error_calibracion_mm:.3f} mm"
            cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.current_image = annotated
        self._show_image(annotated)
        self._update_info_from_result(result)
        
        regla_msg = " | Regla detectada" if regla_resultado else ""
        self.status_bar.config(text=f'Deteccion completada.{regla_msg}')

    # ══════════════════════════════════════════════════════════════════
    #  Video navigation
    # ══════════════════════════════════════════════════════════════════

    def _on_video_slider(self, val):
        if not self.is_video or not self.video_frames: 
            return
        
        idx = max(0, min(int(float(val)), len(self.video_frames) - 1))
        self.current_frame_idx = idx
        self._show_frame(idx)
        self._update_indeterminate_button()
        self.canvas.focus_set()

    def _show_frame(self, idx):
        work = self._working_frame(idx)
        result = self.frame_results[idx]
        if result is None:
            prev = self._get_prev_result(idx)
            # NO detectar iris automáticamente en videos
            result = self._detect_frame(work, prev, detectar_iris=False)
            self.frame_results[idx] = result
        
        base = self.video_frames[idx]
        pc = result.get('pupil_center')
        pr = result.get('pupil_radius')
        
        # Para videos: solo mostrar iris si está habilitado Y se detectó en este frame
        ic = None
        ir = None
        
        if self._analisis_iris_habilitado.get():
            # Si es el frame actual Y hay resultado de iris guardado, mostrarlo
            if idx == self.current_frame_idx and self._iris_resultado_actual is not None:
                ic = self._iris_resultado_actual.get('iris_center')
                ir = self._iris_resultado_actual.get('iris_radius')
            elif idx == self.current_frame_idx:
                # Estimar iris para el frame actual
                if pr is not None:
                    ir = int(pr * 2.5)
                    ic = pc
        # Si NO está habilitado el análisis de iris, NO mostrar iris
        
        ox, oy = 0, 0
        if self.crop_rect:
            ox, oy = self.crop_rect[0], self.crop_rect[1]
        
        pc_display = pc
        ic_display = ic
        if pc and ox > 0:
            pc_display = (pc[0] + ox, pc[1] + oy)
        if ic and ox > 0:
            ic_display = (ic[0] + ox, ic[1] + oy)
        
        line_width = self.line_width_var.get()
        
        # Ocultar pupila si el frame es indeterminado, pero mantener la regla
        if self._is_frame_indeterminate(idx):
            # Para frames indeterminados, dibujar solo el iris si está habilitado
            annotated = draw_detections(base, None, None, ic_display, ir, line_width)
            # Agregar marca visual de indeterminado
            cv2.putText(annotated, 'INDETERMINADO', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            annotated = draw_detections(base, pc_display, pr, ic_display, ir, line_width)
        
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 1)
        
        # Agregar información de escala calibrada a la imagen del video
        if self._escala_px_mm is not None and pc and pr:
            px_to_mm = 1.0 / self._escala_px_mm
            pupil_diam_mm = pr * 2 * px_to_mm
            info_text = f"Pupila: {pupil_diam_mm:.2f} mm (diam.)"
            if ir and self._analisis_iris_habilitado.get():
                iris_diam_mm = ir * 2 * px_to_mm
                info_text += f" | Iris: {iris_diam_mm:.2f} mm"
            if self._error_calibracion_mm:
                info_text += f" | Error: +/-{self._error_calibracion_mm:.3f} mm"
            cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.current_image = annotated
        self._show_image(annotated)
        
        self.pupil_center = pc_display
        self.pupil_radius = pr
        self.iris_center = ic_display
        self.iris_radius = ir
        n = len(self.video_frames)
        
        if self.video_timestamps and idx < len(self.video_timestamps):
            t = self.video_timestamps[idx]
            dur = self.video_timestamps[-1] if self.video_timestamps else 0
        else:
            t = idx / self.video_fps
            dur = (n - 1) / self.video_fps
        
        self.frame_time_lbl.config(text=f'{t:.2f}s / {dur:.2f}s')
        self.frame_idx_lbl.config(text=f'{idx+1}/{n}')
        self._update_info_from_result(result, t)
        self.canvas.focus_set()

    def _get_prev_result(self, idx):
        for i in range(idx-1,max(idx-10,-1),-1):
            if i>=0 and self.frame_results[i] is not None:
                return self.frame_results[i]
        return None

    # ══════════════════════════════════════════════════════════════════
    #  Selector de iluminacion
    # ══════════════════════════════════════════════════════════════════

    def _on_illum_change(self, on_frac, off_frac):
        self._light_start_frac = on_frac
        self._light_end_frac = off_frac
        if self.is_video and self.video_frames:
            if self.video_timestamps:
                dur = self.video_timestamps[-1] - self.video_timestamps[0]
            else:
                dur = len(self.video_frames) / self.video_fps
            t_on = on_frac * dur
            t_off = off_frac * dur
            self._illum_lbl_var.set(f'Iluminacion: {t_on:.2f}s - {t_off:.2f}s')

    def _apply_illum_times(self):
        try:
            t_start = float(self.light_start_var.get())
            t_end = float(self.light_end_var.get())
            if t_end <= t_start:
                messagebox.showwarning('Aviso', 'El tiempo de fin debe ser mayor al de inicio.')
                return
            self._light_start_sec = t_start
            self._light_end_sec = t_end
            if self.is_video and self.video_frames:
                dur = len(self.video_frames) / self.video_fps
                self._light_start_frac = min(t_start / dur, 1.0)
                self._light_end_frac = min(t_end / dur, 1.0)
            self._illum_lbl_var.set(f'Iluminación: {t_start:.2f}s - {t_end:.2f}s')
        except ValueError:
            messagebox.showwarning('Aviso', 'Tiempo invalido. Usa numeros.')

    def _reset_illum_to_protocol(self):
        """Restaura los tiempos de iluminacion al protocolo por defecto (flash 1.0s-1.2s)."""
        self.light_start_var.set(str(PROTOCOL_FLASH_ON_SEC))
        self.light_end_var.set(str(PROTOCOL_FLASH_OFF_SEC))
        self._apply_illum_times()
        self.status_bar.config(
            text=f'Tiempos restaurados al protocolo: '
                 f'ON={PROTOCOL_FLASH_ON_SEC:.1f}s  OFF={PROTOCOL_FLASH_OFF_SEC:.1f}s'
        )

    def _update_illum_labels(self):
        self.light_start_var.set(str(self._light_start_sec))
        self.light_end_var.set(str(self._light_end_sec))
        self._illum_lbl_var.set(
            f'Iluminación: {self._light_start_sec:.2f}s - {self._light_end_sec:.2f}s')

    # ══════════════════════════════════════════════════════════════════
    #  Ajuste manual de circulos
    # ══════════════════════════════════════════════════════════════════

    def _select_for_edit(self, circle_type):
        """Selecciona que circulo editar: 'pupil' o 'iris'"""
        if not self._edit_mode:
            self._toggle_edit_mode()
        
        if circle_type == 'pupil':
            if self.pupil_center is None:
                self.status_bar.config(text='Primero detecta la pupila')
                return
            self._selected_circle = 'pupil'
            self.edit_status_var.set(f'Editando: Pupila')
        else:
            if self.iris_center is None:
                self.status_bar.config(text='Primero detecta el iris')
                return
            self._selected_circle = 'iris'
            self.edit_status_var.set(f'Editando: Iris')
        
        self._update_circle_controls()

    def _toggle_edit_mode(self):
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            self.canvas.configure(cursor='hand1')
            self.canvas.focus_set()
            self.edit_mode_btn.configure(text='Desactivar Ajuste')
            self._selected_circle = None
            self.edit_status_var.set('Usa los controles M=move R=resize')
            self._update_circle_controls()
        else:
            self.canvas.configure(cursor='crosshair')
            self.edit_mode_btn.configure(text='Activar Ajuste Manual')
            self.edit_status_var.set('Modo normal (arrastra para recorte)')
            self._selected_circle = None
            self.canvas.delete('circle_controls')
            self._update_current_view()
        self._pending_changes = False
        self.propagate_btn.configure(state='disabled')

    def _apply_single_frame_edit(self, idx):
        if not self.is_video or not self.video_frames:
            return
        
        pc_full = self.pupil_center
        pr = self.pupil_radius
        ir = self.iris_radius

        if pc_full is None or pr is None:
            return

        if self.crop_rect:
            ox, oy = self.crop_rect[0], self.crop_rect[1]
            pc_store = (pc_full[0] - ox, pc_full[1] - oy)
        else:
            pc_store = pc_full

        ic_store = pc_store

        self.frame_results[idx] = {
            'pupil_center': pc_store,
            'pupil_radius': pr,
            'iris_center': ic_store,
            'iris_radius': ir,
            'px_to_mm': (self.frame_results[idx] or {}).get('px_to_mm'),
        }

    def _confirm_changes(self):
        if not self._pending_changes:
            return
        if self.is_video and self.video_frames:
            self._apply_edit_to_following_frames()
            self._pending_changes = False
            self.propagate_btn.configure(state='disabled')
            self.edit_status_var.set('Cambios propagados a todos los frames.')
            self.status_bar.config(text='Cambios propagados a los frames siguientes.')

    def _update_current_view(self):
        if self.is_video and self.video_frames:
            self._show_frame(self.current_frame_idx)
        elif self.image is not None:
            line_width = self.line_width_var.get()
            annotated = draw_detections(self.image, self.pupil_center,
                                        self.pupil_radius, self.iris_center, self.iris_radius, line_width)
            self.current_image = annotated
            self._show_image(annotated)

    def _show_editable_view(self):
        """Muestra los circulos editables sobre la imagen"""
        img = self.current_image if self.current_image is not None else self.image
        if img is None:
            return
        
        self._show_image(img)
        if self._edit_mode:
            self._update_circle_controls()

    def _get_canvas_to_image_coords(self, cx, cy):
        """Convierte coordenadas del canvas a coordenadas de imagen."""
        ox, oy = self._canvas_offset
        sc = self._canvas_scale
        if sc == 0:
            return cx, cy
        ix = int((cx - ox) / sc)
        iy = int((cy - oy) / sc)
        
        if self.crop_rect:
            crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_rect
            ix = ix + crop_x1
            iy = iy + crop_y1
        
        return ix, iy

    def _edit_press(self, e):
        """Al hacer click, seleccionar el circulo mas cercano"""
        if self._selected_circle is None:
            ix, iy = self._get_canvas_to_image_coords(e.x, e.y)
            self._select_circle_at(ix, iy)
        
        if self._selected_circle:
            # Store (canvas_x, canvas_y, img_cx, img_cy) for _control_drag compatibility
            if self._selected_circle == 'pupil' and self.pupil_center:
                self._drag_start = (e.x, e.y, self.pupil_center[0], self.pupil_center[1])
            elif self._selected_circle == 'iris' and self.iris_center:
                self._drag_start = (e.x, e.y, self.iris_center[0], self.iris_center[1])
            else:
                self._drag_start = (e.x, e.y, 0, 0)
            self._show_editable_view()

    def _edit_drag(self, e):
        """Mover ambos circulos arrastrando (son concentricos)"""
        if self._drag_start is None or self._selected_circle is None:
            return
        
        sc = self._canvas_scale
        if sc == 0:
            return

        start_x, start_y, old_cx, old_cy = self._drag_start
        img_dx = int((e.x - start_x) / sc)
        img_dy = int((e.y - start_y) / sc)
        new_cx = old_cx + img_dx
        new_cy = old_cy + img_dy

        # Mover ambos circulos (concentricos)
        self.pupil_center = (new_cx, new_cy)
        self.iris_center  = (new_cx, new_cy)

        self._show_editable_view()

    def _edit_release(self, e):
        """Al soltar, aplicar cambios al frame actual y habilitar propagacion."""
        self._drag_start = None
        self._control_dragging = None
        
        if self.is_video and self.video_frames:
            idx = self.current_frame_idx
            self._apply_single_frame_edit(idx)
            self._pending_changes = True
            self.propagate_btn.configure(state='normal')
            self.edit_status_var.set('Cambios en este frame. Presiona Propagar para siguientes.')
            self.status_bar.config(text=f'{self._selected_circle.capitalize()} ajustada. Propagar si deseas.')
        else:
            self._update_current_view()
        
        if self._selected_circle and not self.is_video:
            self.status_bar.config(text=f'{self._selected_circle.capitalize()} ajustada')

    def _select_circle_at(self, ix, iy):
        """Selecciona el circulo mas cercano al punto dado"""
        dist_pupil = float('inf')
        dist_iris = float('inf')
        
        if self.pupil_center and self.pupil_radius:
            dist_pupil = math.hypot(ix - self.pupil_center[0], iy - self.pupil_center[1])
        if self.iris_center and self.iris_radius:
            dist_iris = math.hypot(ix - self.iris_center[0], iy - self.iris_center[1])
        
        if dist_pupil <= dist_iris and dist_pupil < 50:
            self._selected_circle = 'pupil'
        elif dist_iris < 50:
            self._selected_circle = 'iris'
        else:
            self._selected_circle = None
        
        return self._selected_circle

    def _apply_edit_to_following_frames(self):
        """
        Tras un ajuste manual en video, escribe los valores corregidos
        en el frame actual y los propaga a todos los frames siguientes.

        Lo que se propaga:
          - pupil_center  (posicion, en coords de imagen completa o crop segun corresponda)
          - pupil_radius  (radio de la pupila)
          - iris_radius   (radio del iris)
          - iris_center   (igual al pupil_center, son concentricos)

        Los frames ANTERIORES al actual no se modifican.
        """
        if not self.is_video or not self.video_frames:
            return

        idx   = self.current_frame_idx
        total = len(self.video_frames)

        # Coordenadas que el usuario ve son las del frame completo.
        # frame_results almacena coordenadas en espacio de CROP (si hay crop).
        # Hay que convertir de vuelta al espacio del crop antes de guardar.
        pc_full = self.pupil_center   # (cx, cy) en frame completo
        pr      = self.pupil_radius
        ir      = self.iris_radius

        if pc_full is None or pr is None:
            return

        if self.crop_rect:
            ox, oy = self.crop_rect[0], self.crop_rect[1]
            pc_store = (pc_full[0] - ox, pc_full[1] - oy)
        else:
            pc_store = pc_full

        ic_store = pc_store   # concentricos

        new_result_base = {
            'pupil_center': pc_store,
            'pupil_radius': pr,
            'iris_center':  ic_store,
            'iris_radius':  ir,
            'px_to_mm':     (self.frame_results[idx] or {}).get('px_to_mm'),
        }

        # Aplicar al frame actual y a todos los siguientes
        frames_updated = 0
        for i in range(idx, total):
            existing = self.frame_results[i]
            if existing is not None:
                # Actualizar posicion y radios, conservar px_to_mm si existe
                existing['pupil_center'] = pc_store
                existing['pupil_radius'] = pr
                existing['iris_center']  = ic_store
                existing['iris_radius']  = ir
            else:
                # Frame sin detectar aun: guardar resultado base
                self.frame_results[i] = dict(new_result_base)
            frames_updated += 1

        n_following = total - idx - 1
        self.status_bar.config(
            text=f'Ajuste aplicado al frame {idx+1} y {n_following} frames siguientes.'
        )
        """Muestra los circulos editables sobre la imagen"""
        img = self.current_image if self.current_image is not None else self.image
        if img is None:
            return
        
        self._show_image(img)
        # Redibujar handles sobre la imagen actualizada
        self._update_circle_controls()

    # ══════════════════════════════════════════════════════════════════
    #  Pupilometria
    # ══════════════════════════════════════════════════════════════════

    def calculate_pupillometry(self):
        if not self.is_video or not self.video_frames:
            messagebox.showwarning('Aviso','Carga un video primero.'); return
        n=len(self.video_frames)
        
        # Obtener timestamps o crear estimation
        if self.video_timestamps:
            t0 = self.video_timestamps[0]
            t1 = self.video_timestamps[-1]
            timestamps_all = self.video_timestamps
        else:
            fps = self.video_fps
            t0 = 0
            t1 = n / fps
            timestamps_all = [i / fps for i in range(n)]
        
        # Configuracion de fases
        fase1_dur=self._fase_basal_dur.get()
        fase2_dur=self._fase_contraccion_dur.get()
        fase3_dur=self._fase_relajacion_dur.get()
        
        # Calcular tiempo de inicio de cada fase
        t1_start = t0
        t1_end = t1_start + fase1_dur
        t2_start = t1_end
        t2_end = t2_start + fase2_dur
        t3_start = t2_end
        t3_end = t3_start + fase3_dur
        
        # Definir las fases para el calculo
        fases = [
            {'nombre': 'Basal', 't_inicio': t1_start, 't_fin': t1_end},
            {'nombre': 'Contraccion', 't_inicio': t2_start, 't_fin': t2_end},
            {'nombre': 'Relajacion', 't_inicio': t3_start, 't_fin': t3_end},
        ]
        
        # Calcular rango total
        t_start = self._light_start_sec
        t_end = self._light_end_sec
        
        # Encontrar indices de frames
        i0 = 0
        i1 = n - 1
        for i, ts in enumerate(timestamps_all):
            if ts >= t_start:
                i0 = i
                break
        for i in range(n - 1, -1, -1):
            if timestamps_all[i] <= t_end:
                i1 = i
                break
        
        if i1 - i0 < 2:
            messagebox.showwarning('Aviso', 'Rango demasiado corto.')
            return
        
        timestamps_slice = timestamps_all[i0:i1+1]
        
        for i in range(i0,i1+1):
            if self.frame_results[i] is None:
                prev=self._get_prev_result(i)
                self.frame_results[i]=self._detect_frame(
                    self._working_frame(i),prev)
            if i%20==0:
                pct=(i-i0)/max(i1-i0,1)*100
                self.status_bar.config(text=f'Procesando... {pct:.0f}%')
                self.root.update()
        
        # Estimar frames indeterminados antes del calculo
        indeterminate_in_range = {
            i for i in self._indeterminate_frames if i0 <= i <= i1
        }
        
        estimated_results = None
        if indeterminate_in_range:
            self.status_bar.config(text='Estimando frames indeterminados...')
            self.root.update()
            estimated_results = estimate_indeterminate_frames(
                self.frame_results,
                indeterminate_in_range,
                timestamps_all
            )
        
        # Usar resultados combinados o originales
        if estimated_results:
            # Combinar resultados originales con estimados
            merged_results = []
            for i in range(n):
                if estimated_results[i] is not None:
                    merged_results.append(estimated_results[i])
                elif self.frame_results[i] is not None:
                    merged_results.append(self.frame_results[i])
                else:
                    merged_results.append(None)
        else:
            merged_results = self.frame_results
        
        fps = self.video_fps
        frac_start = t_start / (t1 - t0) if (t1 - t0) > 0 else 0.0
        frac_end = t_end / (t1 - t0) if (t1 - t0) > 0 else 1.0
        
        data=compute_pupillometry(merged_results,fps,i0,i1,
                                 frac_start, frac_end, timestamps_slice, fases)
        if not data:
            messagebox.showwarning('Aviso','Datos insuficientes.'); return
        
        # Agregar informacion de frames estimados al resultado
        if indeterminate_in_range:
            data['estimated_frames'] = list(indeterminate_in_range)
            data['estimated_flags'] = []
            data['centers_x'] = []
            data['centers_y'] = []
            for i in range(len(merged_results)):
                if merged_results[i]:
                    data['estimated_flags'].append(
                        'SI' if merged_results[i].get('estimated', False) else ''
                    )
                    pc = merged_results[i].get('pupil_center', (0, 0))
                    data['centers_x'].append(str(int(pc[0])) if pc else '')
                    data['centers_y'].append(str(int(pc[1])) if pc else '')
                else:
                    data['estimated_flags'].append('')
                    data['centers_x'].append('')
                    data['centers_y'].append('')
        
        # Agregar info de reduccion de fps si aplica
        if hasattr(self, '_fps_reduction_info'):
            data['fps_reduction'] = self._fps_reduction_info
        
        self._pupillometry_data=data
        self._log(format_report(data))
        
        est_msg = f' ({len(indeterminate_in_range)} frames estimados)' if indeterminate_in_range else ''
        self.status_bar.config(text=f'Pupilometria calculada.{est_msg}')

    def run_adaptive_analysis_ui(self):
        """Ejecuta el analisis adaptativo de busqueda binaria y muestra el grafico."""
        if not self.is_video or not self.video_frames:
            messagebox.showwarning('Aviso', 'Carga un video primero.')
            return

        n = len(self.video_frames)
        fps = self.video_fps

        if self.video_timestamps:
            t0 = self.video_timestamps[0]
            t1 = self.video_timestamps[-1]
            timestamps_all = self.video_timestamps
        else:
            t0 = 0
            t1 = n / fps
            timestamps_all = [i / fps for i in range(n)]

        fase1_dur = self._fase_basal_dur.get()
        fase2_dur = self._fase_contraccion_dur.get()
        fase3_dur = self._fase_relajacion_dur.get()

        t1_start = t0
        t1_end = t1_start + fase1_dur
        t2_start = t1_end
        t2_end = t2_start + fase2_dur
        t3_start = t2_end
        t3_end = t3_start + fase3_dur

        pre_start_ms = t1_start * 1000
        flash_start_ms = t2_start * 1000
        flash_end_ms = t3_start * 1000
        end_ms = t3_end * 1000

        px_to_mm = self.pixel_to_mm if hasattr(self, 'pixel_to_mm') and self.pixel_to_mm else None

        def get_diameter_fn(frame_idx: int):
            """Callback que obtiene el diametro pupilar de un frame."""
            if frame_idx < 0 or frame_idx >= n:
                return (0.0, 0.0)

            if frame_idx in self._indeterminate_frames:
                return (0.0, 0.0)

            if self.frame_results[frame_idx] is None:
                prev = self._get_prev_result(frame_idx)
                self.frame_results[frame_idx] = self._detect_frame(
                    self._working_frame(frame_idx), prev
                )

            res = self.frame_results[frame_idx]
            if res is None or res.get('pupil_radius') is None:
                return (0.0, 0.0)

            pupil_radius = res['pupil_radius']
            diameter_px = pupil_radius * 2
            confidence = 0.9
            if res.get('estimated', False):
                confidence = 0.5
            return (diameter_px, confidence)

        total_frames = n

        self.status_bar.config(text='Ejecutando analisis adaptativo...')
        self.root.update()

        try:
            config = {
                "error_margin_px": 2.0,
                "min_confidence": 0.7,
                "max_binary_iterations": 8,
                "force_full_analysis_if_poor_fit": True,
                "output_dir": "./results",
                "save_plots": False,
                "save_csv": False
            }

            phase_timestamps = {
                "pre_start_ms": pre_start_ms,
                "flash_start_ms": flash_start_ms,
                "flash_end_ms": flash_end_ms,
                "end_ms": end_ms
            }

            metrics, all_results = run_adaptive_analysis(
                video_path="",
                phase_timestamps=phase_timestamps,
                get_diameter_fn=get_diameter_fn,
                total_frames=total_frames,
                fps=fps,
                config=config,
                indeterminate_frames=self._indeterminate_frames.copy()
            )

            self._adaptive_results = all_results
            self._adaptive_metrics = metrics

            self._update_diameter_graph(all_results, flash_start_ms, flash_end_ms)

            pre = all_results["pre_result"]
            ilum = all_results["ilum_result"]
            post = all_results["post_result"]

            report_lines = [
                "ANALISIS ADAPTATIVO (BUSQUEDA BINARIA)",
                "=" * 50,
                f"Basal: {metrics.baseline_diameter:.2f} px (+/- {metrics.baseline_std:.2f})",
                f"Minimo: {metrics.min_diameter:.2f} px @ {metrics.min_diameter_time_ms:.0f} ms",
                f"Amplitud: {metrics.constriction_amplitude:.2f} px ({metrics.constriction_amplitude_pct:.1f}%)",
                f"Latencia: {metrics.constriction_latency_ms:.0f} ms",
                f"Vel. max contraccion: {metrics.max_constriction_velocity:.4f} px/ms",
                "",
                "-- REDILATACION --",
                f"T75: {metrics.t75_ms:.0f} ms" if metrics.t75_ms else "T75: N/A",
                f"T90: {metrics.t90_ms:.0f} ms" if metrics.t90_ms else "T90: N/A",
                "",
                "-- PIPR --",
                f"PIPR 1s: {metrics.pipr_1s:.2f} px" if metrics.pipr_1s else "PIPR 1s: N/A",
                f"PIPR 3s: {metrics.pipr_3s:.2f} px" if metrics.pipr_3s else "PIPR 3s: N/A",
                f"PIPR 6s: {metrics.pipr_6s:.2f} px" if metrics.pipr_6s else "PIPR 6s: N/A",
                "",
                "-- EFICIENCIA --",
                f"Frames medidos: {metrics.total_frames_measured}/{metrics.total_frames_analyzed}",
                f"Compression: {metrics.compression_ratio*100:.1f}%",
                "",
                "-- FASES --",
                f"Pre: {len(pre.measured_frames)} medidos, {len(pre.interpolated_frames)} interpolados",
                f"Ilum: {len(ilum.measured_frames)} medidos, {len(ilum.interpolated_frames)} interpolados",
                f"Post: {len(post.measured_frames)} medidos, {len(post.interpolated_frames)} interpolados",
            ]

            if pre.warnings:
                report_lines.append(f"\nAdvertencias pre: {', '.join(pre.warnings)}")
            if ilum.warnings:
                report_lines.append(f"Advertencias ilum: {', '.join(ilum.warnings)}")
            if post.warnings:
                report_lines.append(f"Advertencias post: {', '.join(post.warnings)}")

            self._log('\n'.join(report_lines))
            self.status_bar.config(text='Analisis adaptativo completado.')

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror('Error', f'Error en analisis adaptativo:\n{str(e)}')
            self.status_bar.config(text='Error en analisis adaptativo.')

    def _update_diameter_graph(self, all_results, flash_start_ms=None, flash_end_ms=None):
        """Actualiza el grafico compacto de diametro pupilar sobre el canvas."""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            pre = all_results["pre_result"]
            ilum = all_results["ilum_result"]
            post = all_results["post_result"]

            fig = Figure(figsize=(10, 1.2), dpi=100)
            fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.25)
            ax = fig.add_subplot(1, 1, 1)

            colors = {"pre": "#3366cc", "ilum": "#cc2222", "post": "#228833"}

            for phase_name, color in colors.items():
                if phase_name == "pre":
                    samples = pre.samples
                elif phase_name == "ilum":
                    samples = ilum.samples
                else:
                    samples = post.samples

                if not samples:
                    continue

                measured_t = [s.timestamp_ms / 1000 for s in samples if not s.is_interpolated]
                measured_d = [s.diameter_px for s in samples if not s.is_interpolated]
                interp_t = [s.timestamp_ms / 1000 for s in samples if s.is_interpolated]
                interp_d = [s.diameter_px for s in samples if s.is_interpolated]

                if measured_t:
                    ax.scatter(measured_t, measured_d, color=color, s=8, zorder=5, alpha=0.8)
                if interp_t:
                    ax.plot(interp_t, interp_d, color='gray', ls=':', lw=0.5, alpha=0.3, zorder=2)

            if flash_start_ms is not None:
                ax.axvline(flash_start_ms / 1000, color='#00aa00', lw=1, ls='--', alpha=0.7)
            if flash_end_ms is not None:
                ax.axvline(flash_end_ms / 1000, color='#cc0000', lw=1, ls='--', alpha=0.7)

            metrics = all_results["metrics"]
            ax.axhline(metrics.baseline_diameter, color='gray', lw=0.6, ls='-', alpha=0.4)

            ax.scatter([metrics.min_diameter_time_ms / 1000], [metrics.min_diameter],
                       color='purple', s=30, zorder=6, marker='*')

            all_times = [s.timestamp_ms / 1000 for s in pre.samples + ilum.samples + post.samples]
            all_diams = [s.diameter_px for s in pre.samples + ilum.samples + post.samples]
            if all_times:
                y_min = min(all_diams) - 3
                y_max = max(all_diams) + 3
                if flash_start_ms is not None and flash_end_ms is not None:
                    ax.fill_betweenx([y_min, y_max],
                                     flash_start_ms / 1000, flash_end_ms / 1000,
                                     alpha=0.06, color='gold', zorder=0)
                ax.set_ylim(y_min, y_max)

            ax.set_ylabel('Diam (px)', fontsize=7)
            ax.set_xlabel('Tiempo (s)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

            for child in self.graph_frame.winfo_children():
                child.destroy()

            canvas_widget = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().grid(row=0, column=0, sticky='ew')

            self.graph_frame.grid(row=0, column=0, sticky='ew', pady=(0, 2))

        except ImportError:
            pass
        except Exception:
            pass

    def export_csv(self):
        if not self._pupillometry_data:
            messagebox.showwarning('Aviso','Primero calcula la pupilometria.'); return
        path=filedialog.asksaveasfilename(
            defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if not path: return
        try:
            export_pupillometry_csv(self._pupillometry_data,path)
            messagebox.showinfo('Exportado',f'CSV guardado en:\n{path}')
        except Exception as e: messagebox.showerror('Error',str(e))

    # ══════════════════════════════════════════════════════════════════
    #  Helpers UI
    # ══════════════════════════════════════════════════════════════════

    def _estimate_pupil_mm(self):
        try:
            iris_mm = float(self.iris_mm_var.get())
            if iris_mm <= 0:
                self.scale_info_var.set('Diam. iris invalido')
                return
            if self.pupil_radius and self.iris_radius and self.iris_radius > 0:
                iris_diam_px = self.iris_radius * 2
                pupil_diam_px = self.pupil_radius * 2
                px_to_mm = iris_mm / iris_diam_px
                pupil_mm = pupil_diam_px * px_to_mm
                self.pixel_to_mm = px_to_mm
                self.scale_info_var.set(f'Ratio: {self.pupil_radius/self.iris_radius*100:.1f}% | Pupila: {pupil_mm:.2f} mm')
                self._show_estimated_result(iris_mm, pupil_mm, px_to_mm)
            else:
                self.scale_info_var.set(f'Iris: {iris_mm:.2f} mm | Detecta primero')
        except ValueError:
            self.scale_info_var.set('Error en valor')

    def _show_estimated_result(self, iris_mm, pupil_mm, px_to_mm):
        """Muestra el resultado estimado en el panel de resultados."""
        info = f"ESCALA ESTIMADA\n{'='*34}\n"
        info += f"Iris: {iris_mm:.2f} mm (diam. real)\n"
        info += f"Pupila: {pupil_mm:.2f} mm (estimado)\n"
        info += f"Ratio: {self.pupil_radius/self.iris_radius*100:.1f}%\n"
        info += f"Pixel->mm: {px_to_mm:.6f}\n"
        info += f"\n{'='*34}\n"
        info += f"(Usa 'Calcular Pupilometria' para analisis completo)"
        self._log(info)

    def _update_info_from_result(self, result, t=None):
        if not result: return
        pc=result.get('pupil_center'); pr=result.get('pupil_radius')
        ic=result.get('iris_center');  ir=result.get('iris_radius')
        px2mm=result.get('px_to_mm')
        info=''
        if t is not None: info+=f't = {t:.3f} s\n{"="*34}\n\n'
        if pc and pr:
            info+=f'PUPILA\n  Centro: {pc}\n  Radio: {pr} px | Diam: {pr*2} px\n'
            if px2mm: info+=f'  Diam: {pr*2*px2mm:.3f} mm\n'
        else: info+='PUPILA: no detectada\n'
        info+='\n'
        if ic and ir and self._analisis_iris_habilitado.get():
            info+=f'IRIS\n  Radio: {ir} px | Diam: {ir*2} px\n'
            if px2mm: info+=f'  Diam: {ir*2*px2mm:.3f} mm\n'
        elif not self._analisis_iris_habilitado.get():
            info+='IRIS: no analisis activado\n'
        else:
            info+='IRIS: no detectado\n'
        if pr and ir and ir>0 and self._analisis_iris_habilitado.get(): 
            ratio = pr/ir
            info+=f'\nRatio pupila/iris: {ratio*100:.1f}%\n'
        if self._escala_px_mm:
            info+=f'\nEscala calibrada: {1.0/self._escala_px_mm:.5f} mm/px\n'
            info+=f'Error estimado: +/-{self._error_calibracion_mm:.4f} mm\n'
        elif px2mm:
            info+=f'\nEscala: {px2mm:.5f} mm/px\n'
        if self.crop_rect:
            x1,y1,x2,y2=self.crop_rect
            info+=f'\nRecorte: ({x1},{y1})->({x2},{y2})\n'
        self._log(info)

    def _log(self, text):
        self.info_text.delete('1.0','end'); self.info_text.insert('1.0',text)

    # ══════════════════════════════════════════════════════════════════
    #  Ventanas de Ajustes
    # ══════════════════════════════════════════════════════════════════

    def _open_calibracion_window(self):
        """Abre ventana de ajustes de calibracion de regla."""
        win = tk.Toplevel(self.root)
        win.title("Calibracion de Regla")
        win.geometry("400x250")
        win.resizable(False, False)
        win.grab_set()

        main_fr = ttk.Frame(win, padding=20)
        main_fr.pack(fill='both', expand=True)

        ttk.Label(main_fr, text='Distancia entre puntos de la regla (mm):',
                  font=('Arial', 10)).pack(anchor='w', pady=(0, 10))

        dist_fr = ttk.Frame(main_fr)
        dist_fr.pack(fill='x', pady=(0, 15))
        
        self.distancia_regla_var = tk.StringVar(value=str(self._distancia_regla_mm))
        ttk.Entry(dist_fr, textvariable=self.distancia_regla_var, width=10).pack(side='left', padx=(0, 10))
        ttk.Label(dist_fr, text='mm').pack(side='left')

        ttk.Label(main_fr, text='La regla se detecta automaticamente de la imagen cargada.',
                  font=('Arial', 8), foreground='gray').pack(anchor='w', pady=(0, 5))
        ttk.Label(main_fr, text='Estado actual:', font=('Arial', 9, 'bold')).pack(anchor='w', pady=(10, 5))
        ttk.Label(main_fr, textvariable=self.calibracion_info_var,
                  font=('Arial', 9), foreground='green').pack(anchor='w')

        btn_fr = ttk.Frame(main_fr)
        btn_fr.pack(side='bottom', fill='x', pady=(20, 0))
        btn_fr.columnconfigure(0, weight=1)
        btn_fr.columnconfigure(1, weight=1)
        ttk.Button(btn_fr, text='Cancelar', command=win.destroy).grid(row=0, column=0, padx=(0, 10), sticky='ew')
        ttk.Button(btn_fr, text='Aceptar', command=lambda: self._aplicar_ajustes_calibracion(win)).grid(row=0, column=1, sticky='ew')
    
    def _aplicar_ajustes_calibracion(self, win):
        """Aplica los ajustes de calibracion y cierra la ventana."""
        try:
            distancia_mm = float(self.distancia_regla_var.get())
            if distancia_mm > 0:
                self._distancia_regla_mm = distancia_mm
        except ValueError:
            pass
        win.destroy()

    def _open_analisis_video_window(self):
        """Abre ventana de ajustes de analisis de video."""
        fase_vars = {
            'basal_fps': self._fase_basal_fps,
            'basal_dur': self._fase_basal_dur,
            'contraccion_fps': self._fase_contraccion_fps,
            'contraccion_dur': self._fase_contraccion_dur,
            'relajacion_fps': self._fase_relajacion_fps,
            'relajacion_dur': self._fase_relajacion_dur,
        }
        flash_vars = {
            'auto': self._flash_detection_auto,
            'calibration_frames': self._flash_calibration_frames,
            'change_threshold': self._flash_change_threshold,
            'smoothing_window': self._flash_smoothing_window,
        }
        create_analisis_video_window(
            parent=self.root,
            fase_vars=fase_vars,
            flash_vars=flash_vars,
            on_accept_callback=self._aplicar_ajustes_video
        )

    def _aplicar_ajustes_video(self, win):
        """Aplica los ajustes y cierra la ventana."""
        self._apply_illum_times()
        win.destroy()
        messagebox.showinfo('Info', 'Los cambios en fases se aplicaran al cargar un nuevo video.')

    def _actualizar_resultados(self):
        """Actualiza los resultados mostrados en base a los ajustes manuales."""
        if self.is_video and self.video_frames:
            self._show_frame(self.current_frame_idx)
        elif self.image is not None:
            # Solo actualizar la visualización con los valores actuales, no volver a detectar
            self._update_current_view()
        self.status_bar.config(text='Resultados actualizados.')
    
    def _update_current_view(self):
        """Actualiza la vista actual con los valores de pupila/iris actuales."""
        if self.image is not None:
            line_width = self.line_width_var.get()
            annotated = self.image.copy()
            
            # Obtener los valores actuales de pupila e iris
            pc = self.pupil_center
            pr = self.pupil_radius
            ic = self.iris_center
            ir = self.iris_radius
            
            # Dibujar regla si se detectó
            if self._calibracion_resultado:
                imagen_con_regla = self._calibracion_resultado.get('imagen_procesada')
                if imagen_con_regla is not None:
                    annotated = imagen_con_regla.copy()
            
            # Dibujar pupila e iris
            annotated = draw_detections(annotated, pc, pr, ic, ir, line_width)
            
            # Agregar información de escala
            if self._escala_px_mm is not None and pc and pr:
                px_to_mm = 1.0 / self._escala_px_mm
                pupil_diam_mm = pr * 2 * px_to_mm
                info_text = f"Pupila: {pupil_diam_mm:.2f} mm (diam.)"
                if ir and self._analisis_iris_habilitado.get():
                    iris_diam_mm = ir * 2 * px_to_mm
                    info_text += f" | Iris: {iris_diam_mm:.2f} mm"
                if self._error_calibracion_mm:
                    info_text += f" | Error: +/-{self._error_calibracion_mm:.3f} mm"
                cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            self.current_image = annotated
            self._show_image(annotated)
            
            # Actualizar panel de resultados
            result = {
                'pupil_center': pc,
                'pupil_radius': pr,
                'iris_center': ic if self._analisis_iris_habilitado.get() else None,
                'iris_radius': ir if self._analisis_iris_habilitado.get() else None,
                'px_to_mm': 1.0 / self._escala_px_mm if self._escala_px_mm else None
            }
            self._update_info_from_result(result)

    def _toggle_dibujos(self):
        """Muestra/oculta los dibujos sobre la imagen."""
        current = self._mostrar_dibajos.get()
        self._mostrar_dibajos.set(not current)
        
        if self.is_video and self.video_frames:
            self._show_frame(self.current_frame_idx)
        elif self.image is not None:
            if self._mostrar_dibajos.get():
                # Mostrar con dibujos
                self._show_image(self.current_image if self.current_image is not None else self.image)
            else:
                # Mostrar imagen original sin dibujos
                img_original = self.original_image if hasattr(self, 'original_image') and self.original_image is not None else self.image
                self._show_image(img_original)
        
        estado = "mostrados" if self._mostrar_dibajos.get() else "ocultos"
        self.status_bar.config(text=f'Dibujos {estado} (Ctrl+H)')


    def _show_image(self, img):
        cw=self.canvas.winfo_width() or CANVAS_W
        ch=self.canvas.winfo_height() or CANVAS_H
        
        display_img = img
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            display_img = img[y1:y2, x1:x2]
        
        h, w = display_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(display_img, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self._canvas_offset = (ox, oy)
        self._canvas_scale = scale
        self.canvas.delete('all')
        self.canvas.create_image(ox, oy, anchor='nw', image=tk_img)
        self.canvas.image = tk_img
        if self._edit_mode:
            self._update_circle_controls()
        else:
            self.canvas.delete('circle_controls')

    # ══════════════════════════════════════════════════════════════════
    #  Analisis de Iris Opcional
    # ══════════════════════════════════════════════════════════════════

    def _on_iris_toggle(self):
        """Cuando se togglea el checkbox de analisis de iris."""
        if self._analisis_iris_habilitado.get():
            self.iris_status_var.set('Iris: activado')
            self.detect_iris_frame_btn.configure(state='normal')
            # Intentar detectar iris automaticamente al activar
            if self.image is not None or (self.is_video and self.video_frames):
                self._detectar_iris_frame()
        else:
            self.iris_status_var.set('Iris: desactivado')
            self.detect_iris_frame_btn.configure(state='disabled')
            # Limpiar resultado de iris
            self._iris_resultado_actual = None

    def _detectar_iris_frame(self):
        """Detecta el iris solo en el frame actual (para videos o imagenes)."""
        if self.image is None and not self.is_video:
            messagebox.showwarning('Aviso', 'Carga una imagen o video primero.')
            return
        
        frame = None
        crop_offset_x, crop_offset_y = 0, 0
        
        if self.is_video and self.video_frames:
            frame = self._working_frame(self.current_frame_idx)
            if self.crop_rect:
                crop_offset_x, crop_offset_y = self.crop_rect[0], self.crop_rect[1]
        elif self.image is not None:
            if self.crop_rect:
                x1, y1, x2, y2 = self.crop_rect
                frame = self.image[y1:y2, x1:x2]
                crop_offset_x, crop_offset_y = x1, y1
            else:
                frame = self.image
        
        if frame is None:
            messagebox.showwarning('Aviso', 'No hay frame disponible.')
            return
        
        # Usar el detector avanzado para iris (sin IA, solo visión por computadora)
        iris_alg = self.iris_alg_var.get()
        iris_mode_map = {
            'gradient': adv_detector.DetectionMode.GRADIENT_BASED,
            'hough': adv_detector.DetectionMode.CIRCULAR_HOUGH,
            'ellipse': adv_detector.DetectionMode.ELLIPSE_FIT,
            'profile': adv_detector.DetectionMode.INTENSITY_PROFILE
        }
        iris_mode = iris_mode_map.get(iris_alg, adv_detector.DetectionMode.GRADIENT_BASED)
        
        detector = adv_detector.AdvancedEyeDetector(
            pupil_algorithm='starburst',  # No importa para iris
            iris_mode=iris_mode,
            expected_pupil_range=(0.1, 0.7),
            expected_iris_range=(0.3, 0.9)
        )
        
        result = detector.detect(frame)
        
        if result.iris_center and result.iris_radius:
            # Ajustar por offset de crop si existe
            self._iris_resultado_actual = {
                'iris_center': (
                    int(result.iris_center[0]) + crop_offset_x,
                    int(result.iris_center[1]) + crop_offset_y
                ),
                'iris_radius': result.iris_radius
            }
            
            # Usar el iris detectado como referencia para escala si no hay calibración
            if self.pupil_center and self.pupil_radius:
                pc = self.pupil_center
                ic = self._iris_resultado_actual['iris_center']
                ir = self._iris_resultado_actual['iris_radius']
                
                # Calcular distancia Euclidiana entre pupila e iris
                dist = math.sqrt((pc[0] - ic[0])**2 + (pc[1] - ic[1])**2)
                
                # Si están muy separados, ajustar el iris para que sea concéntrico
                if dist > 10:
                    ic = pc  # Hacer concéntrico
                    self._iris_resultado_actual['iris_center'] = ic
                
                messagebox.showinfo('Iris Detectado',
                    f'Iris detectado en frame actual.\n'
                    f'Centro: {ic}\n'
                    f'Radio: {ir:.1f} px\n\n'
                    f'Nota: El iris solo se detecta en este frame.')
            else:
                messagebox.showinfo('Iris Detectado',
                    f'Iris detectado.\n'
                    f'Centro: {self._iris_resultado_actual["iris_center"]}\n'
                    f'Radio: {self._iris_resultado_actual["iris_radius"]:.1f} px\n\n'
                    f'Nota: El iris solo se detecta en este frame.')
            
            self.status_bar.config(text='Iris detectado en frame actual')
        else:
            # No se pudo detectar el iris - usar valor por defecto
            messagebox.showwarning('Aviso', 
                'No se pudo detectar el iris automáticamente.\n'
                'Se establecerá un valor por defecto de 11.7mm de diámetro.\n'
                'Por favor ajústelo manualmente si es necesario.')
            
            # Calcular radio en pixels a partir de 11.7mm
            iris_diametro_mm = 11.7
            iris_radio_mm = iris_diametro_mm / 2.0  # 5.85mm
            
            # Obtener factor de conversión mm->px
            px_to_mm = None
            if self._escala_px_mm is not None:
                px_to_mm = 1.0 / self._escala_px_mm
            else:
                try:
                    iris_mm_manual = float(self.iris_mm_var.get())
                    if iris_mm_manual > 0 and self.pupil_radius:
                        # Estimar px_to_mm basado en iris manual si hay pupila
                        px_to_mm = iris_mm_manual / (self.pupil_radius * 2.5 * 2)
                except ValueError:
                    pass
            
            if px_to_mm is not None:
                iris_radio_px = int(iris_radio_mm / px_to_mm)
            else:
                # Valor por defecto si no hay calibración
                iris_radio_px = 60
            
            # Usar centro de la pupila si existe, sino usar centro del frame
            if self.pupil_center:
                centro_iris = self.pupil_center
            else:
                h_frame, w_frame = frame.shape[:2]
                centro_iris = (w_frame // 2, h_frame // 2)
            
            # Ajustar por offset de crop
            centro_iris_full = (centro_iris[0] + crop_offset_x, centro_iris[1] + crop_offset_y)
            
            self._iris_resultado_actual = {
                'iris_center': centro_iris_full,
                'iris_radius': iris_radio_px
            }
            
            self.iris_center = centro_iris_full
            self.iris_radius = iris_radio_px
            
            self.status_bar.config(text='Iris establecido con valor por defecto (11.7mm). Ajuste manualmente si es necesario.')

    # ══════════════════════════════════════════════════════════════════
    #  Calibracion de Regla
    # ══════════════════════════════════════════════════════════════════

    def _detectar_regla_en_frame(self, frame):
        """
        Detecta la regla rosa fosforescente en un frame.
        Retorna el resultado de calibración si se detecta, None si no.
        """
        try:
            resultado = calibrar_regla_rosa_fosforescente_mejorado(frame, self._distancia_regla_mm)
            if resultado['exito']:
                return resultado
        except Exception:
            pass
        return None

    def _calibrar_desde_imagen(self):
        """Carga una imagen de la regla y ejecuta la calibracion."""
        # Obtener distancia configurable
        try:
            distancia_mm = float(self.distancia_regla_var.get())
            if distancia_mm <= 0:
                messagebox.showwarning('Aviso', 'La distancia debe ser mayor a 0.')
                return
            self._distancia_regla_mm = distancia_mm
        except ValueError:
            messagebox.showwarning('Aviso', 'Distancia inválida. Usa un número.')
            return
        
        path = filedialog.askopenfilename(
            title='Seleccionar imagen de calibración de regla',
            filetypes=[('Imágenes', '*.jpg *.jpeg *.png *.bmp *.tiff')]
        )
        
        if not path:
            return
        
        self.status_bar.config(text='Calibrando... Por favor espere.')
        self.root.update()
        
        try:
            resultado = calibrar_regla_rosa_fosforescente_mejorado(path, self._distancia_regla_mm)
            
            if resultado['exito']:
                self._calibracion_resultado = resultado
                self._escala_px_mm = resultado['escala_px_mm']
                self._error_calibracion_mm = resultado['error_estimado_mm']
                
                # Mostrar info de calibración
                info = f"Escala: {resultado['escala_px_mm']:.4f} px/mm\n"
                info += f"Distancia px: {resultado['distancia_px']:.1f}\n"
                info += f"Calidad: {resultado['calidad']*100:.1f}%"
                self.calibracion_info_var.set(info)
                
                # Mostrar error de calibración
                error_info = f"Error estimado: ±{resultado['error_estimado_mm']:.3f} mm"
                self.calibracion_error_var.set(error_info)
                
                # Usar la escala de la calibración para pixel_to_mm
                self.pixel_to_mm = 1.0 / resultado['escala_px_mm']  # Convertir a mm por pixel
                
                # Mostrar imagen procesada
                if 'imagen_procesada' in resultado:
                    self._show_image(resultado['imagen_procesada'])
                
                self.status_bar.config(text=f'Calibración exitosa: {resultado["escala_px_mm"]:.4f} px/mm')
                
                # Mostrar detalles en el panel de resultados
                detalles = (
                    f"CALIBRACION DE REGLA\n"
                    f"{'='*40}\n"
                    f"Punto 1: ({resultado['punto1'][0]}, {resultado['punto1'][1]})\n"
                    f"Punto 2: ({resultado['punto2'][0]}, {resultado['punto2'][1]})\n"
                    f"Distancia px: {resultado['distancia_px']:.2f}\n"
                    f"Distancia mm: {resultado['distancia_mm']:.1f}\n"
                    f"Escala: {resultado['escala_px_mm']:.4f} px/mm\n"
                    f"ERROR ESTIMADO: ±{resultado['error_estimado_mm']:.4f} mm\n"
                    f"Ángulo: {resultado['angulo']:.1f}°\n"
                    f"Desviación Y: {resultado['desviacion_vertical']:.2f} px\n"
                    f"Calidad: {resultado['calidad']*100:.1f}%\n"
                    f"Área fondo: {resultado['area_fondo']:.0f} px²\n"
                    f"Porcentaje fondo: {resultado['porcentaje_fondo']:.1f}%"
                )
                self._log(detalles)
                
            else:
                self.calibracion_info_var.set('Error en calibración')
                self.calibracion_error_var.set(resultado.get('error', 'Error desconocido'))
                self.status_bar.config(text='Error en calibración')
                messagebox.showerror('Error de Calibración',
                    f"No se pudo calibrar:\n{resultado.get('error', 'Error desconocido')}")
                    
        except Exception as e:
            self.calibracion_info_var.set('Error en calibración')
            self.calibracion_error_var.set(str(e))
            self.status_bar.config(text='Error en calibración')
            messagebox.showerror('Error', f'Error en calibración:\n{str(e)}')

    def _reset_detection(self):
        self.pupil_center=None; self.pupil_radius=None
        self.iris_center=None;  self.iris_radius=None; self.pixel_to_mm=None
        self._iris_resultado_actual = None

    def _reset_video_state(self):
        self.is_video=False; self.video_frames=[]; self.video_frames_crop=[]
        self.frame_results=[]; self.current_frame_idx=0; self.crop_rect=None
        self.video_timestamps=[]
        self._pupillometry_data=None
        self._adaptive_results=None
        self._adaptive_metrics=None
        try:
            self.graph_frame.grid_remove()
        except: pass
        try:
            self.video_bar.grid_remove(); self._set_video_controls_state('disabled')
        except: pass

    def _set_video_controls_state(self, state):
        try:
            for child in self.video_panel.winfo_children():
                if isinstance(child,(ttk.Button,ttk.Entry)):
                    try: child.configure(state=state)
                    except: pass
                if isinstance(child,ttk.Frame):
                    for sub in child.winfo_children():
                        if isinstance(sub,(ttk.Button,ttk.Entry)):
                            try: sub.configure(state=state)
                            except: pass
        except: pass

    def reset_view(self):
        if self.is_video and self.video_frames: self._show_frame(self.current_frame_idx)
        elif self.image is not None: self._show_image(self.image)
        self.status_bar.config(text='Vista reseteada.')

    def save_result(self):
        img=self.current_image
        if img is None: return
        path=filedialog.asksaveasfilename(
            defaultextension='.jpg',filetypes=[('JPEG','*.jpg'),('PNG','*.png')])
        if path:
            cv2.imwrite(path,img)
            messagebox.showinfo('Guardado',f'Imagen guardada:\n{path}')


# ──────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    AdvancedEyeDetector(root)
    root.mainloop()

if __name__ == '__main__':
    main()