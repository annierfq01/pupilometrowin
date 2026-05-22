"""
video_settings.py - Video Settings Window
==========================================
Ventana de configuracion de video para ajustar fases,
parametros de deteccion de flash y otros ajustes previos
al procesamiento de video.

Incluye deteccion manual de flash con visor frame a frame.
"""

import tkinter as tk
from tkinter import ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk


class ManualFlashDetectionWindow:
    """
    Ventana de deteccion manual de flash con visor frame a frame.
    
    Permite al usuario navegar el video frame por frame y marcar
    el inicio y fin del flash visualmente.
    """

    def __init__(self, parent, video_path, flash_start_var, flash_end_var):
        self.parent = parent
        self.video_path = video_path
        self.flash_start_var = flash_start_var
        self.flash_end_var = flash_end_var
        self.window = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0.0
        self.current_frame_idx = 0
        self.mark_start_frame = None
        self.mark_end_frame = None
        self.photo = None
        self._create_window()

    def _create_window(self):
        """Crea la ventana de deteccion manual de flash."""
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

        w = 800
        h = 620
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Deteccion Manual de Flash - {os.path.basename(self.video_path)}")
        self.window.geometry(f"{w}x{h}")
        self.window.resizable(True, True)
        self.window.transient(self.parent)
        self.window.grab_set()

        main_fr = ttk.Frame(self.window, padding=8)
        main_fr.pack(fill='both', expand=True)
        main_fr.columnconfigure(0, weight=1)
        main_fr.rowconfigure(0, weight=1)

        video_fr = ttk.LabelFrame(main_fr, text='Visor de Video', padding=4)
        video_fr.grid(row=0, column=0, sticky='nsew', pady=(0, 6))
        video_fr.columnconfigure(0, weight=1)
        video_fr.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(video_fr, background='#2b2b2b', anchor='center')
        self.video_label.grid(row=0, column=0, sticky='nsew')
        self._show_frame(self.current_frame_idx)

        info_fr = ttk.Frame(main_fr)
        info_fr.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        info_fr.columnconfigure(1, weight=1)

        ttk.Label(info_fr, text='Frame:').grid(row=0, column=0, padx=(0, 4))
        self.frame_label = ttk.Label(info_fr, text=f'{self.current_frame_idx + 1} / {self.total_frames}')
        self.frame_label.grid(row=0, column=1, sticky='w')

        ttk.Label(info_fr, text='Tiempo:').grid(row=0, column=2, padx=(10, 4))
        t_cur = self.current_frame_idx / self.fps if self.fps > 0 else 0
        self.time_label = ttk.Label(info_fr, text=f'{t_cur:.3f} s')
        self.time_label.grid(row=0, column=3, sticky='w')

        nav_fr = ttk.Frame(main_fr)
        nav_fr.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        nav_fr.columnconfigure(4, weight=1)

        ttk.Button(nav_fr, text='|<', width=3, command=self._go_start).grid(row=0, column=0, padx=(0, 2))
        ttk.Button(nav_fr, text='<<', width=3, command=self._go_prev_10).grid(row=0, column=1, padx=(0, 2))
        ttk.Button(nav_fr, text='<', width=3, command=self._go_prev).grid(row=0, column=2, padx=(0, 2))
        ttk.Button(nav_fr, text='>', width=3, command=self._go_next).grid(row=0, column=3, padx=(0, 2))
        ttk.Button(nav_fr, text='>>', width=3, command=self._go_next_10).grid(row=0, column=4, padx=(0, 2))
        ttk.Button(nav_fr, text='>|', width=3, command=self._go_end).grid(row=0, column=5, padx=(0, 10))

        self.slider_var = tk.IntVar(value=self.current_frame_idx)
        slider = ttk.Scale(nav_fr, from_=0, to=max(0, self.total_frames - 1),
                           variable=self.slider_var, orient='horizontal',
                           command=self._on_slider)
        slider.grid(row=0, column=6, sticky='ew', padx=(0, 4))

        markers_fr = ttk.LabelFrame(main_fr, text='Marcadores de Flash', padding=6)
        markers_fr.grid(row=3, column=0, sticky='ew', pady=(0, 6))
        markers_fr.columnconfigure(1, weight=1)
        markers_fr.columnconfigure(3, weight=1)

        self.mark_start_btn = ttk.Button(markers_fr, text='Marcar Inicio Flash',
                                         command=self._mark_start, width=20)
        self.mark_start_btn.grid(row=0, column=0, padx=(0, 10))

        self.start_info_var = tk.StringVar(value='No marcado')
        ttk.Label(markers_fr, textvariable=self.start_info_var,
                  font=('Arial', 8, 'bold'), foreground='#00aa00').grid(row=0, column=1, sticky='w')

        self.mark_end_btn = ttk.Button(markers_fr, text='Marcar Fin Flash',
                                       command=self._mark_end, width=20)
        self.mark_end_btn.grid(row=0, column=2, padx=(10, 10))

        self.end_info_var = tk.StringVar(value='No marcado')
        ttk.Label(markers_fr, textvariable=self.end_info_var,
                  font=('Arial', 8, 'bold'), foreground='#cc0000').grid(row=0, column=3, sticky='w')

        btn_fr = ttk.Frame(main_fr)
        btn_fr.grid(row=4, column=0, sticky='ew')
        btn_fr.columnconfigure(0, weight=1)
        btn_fr.columnconfigure(1, weight=1)

        ttk.Button(btn_fr, text='Cancelar', command=self._on_cancel).grid(
            row=0, column=0, padx=(0, 4), sticky='ew')
        ttk.Button(btn_fr, text='Aceptar', command=self._on_accept).grid(
            row=0, column=1, padx=(4, 0), sticky='ew')

        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.window.bind('<Left>', lambda e: self._go_prev())
        self.window.bind('<Right>', lambda e: self._go_next())
        self.window.bind('<Up>', lambda e: self._go_prev_10())
        self.window.bind('<Down>', lambda e: self._go_next_10())
        self.window.bind('<Home>', lambda e: self._go_start())
        self.window.bind('<End>', lambda e: self._go_end())

    def _show_frame(self, idx):
        """Muestra el frame en el indice dado."""
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame_idx = idx
        self.slider_var.set(idx)

        t_cur = idx / self.fps if self.fps > 0 else 0
        self.frame_label.config(text=f'{idx + 1} / {self.total_frames}')
        self.time_label.config(text=f'{t_cur:.3f} s')

        lw = self.video_label.winfo_width() or 700
        lh = self.video_label.winfo_height() or 400
        if lw < 100:
            lw = 700
        if lh < 100:
            lh = 400

        h, w = frame.shape[:2]
        scale = min(lw / w, lh / h)
        nw = int(w * scale)
        nh = int(h * scale)
        if nw > 0 and nh > 0:
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        if self.mark_start_frame is not None:
            start_sec = self.mark_start_frame / self.fps
            cur_sec = idx / self.fps
            if cur_sec >= start_sec:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (nw - 1, nh - 1), (0, 200, 0), 3)
                cv2.putText(overlay, f'INICIO FLASH: {start_sec:.3f}s',
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                if self.mark_end_frame is not None and cur_sec >= self.mark_end_frame / self.fps:
                    cv2.rectangle(overlay, (0, 0), (nw - 1, nh - 1), (0, 0, 200), 3)
                    cv2.putText(overlay, f'FIN FLASH: {self.mark_end_frame / self.fps:.3f}s',
                                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
                frame = overlay

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.photo)

    def _go_prev(self):
        self._show_frame(self.current_frame_idx - 1)

    def _go_next(self):
        self._show_frame(self.current_frame_idx + 1)

    def _go_prev_10(self):
        self._show_frame(self.current_frame_idx - 10)

    def _go_next_10(self):
        self._show_frame(self.current_frame_idx + 10)

    def _go_start(self):
        self._show_frame(0)

    def _go_end(self):
        self._show_frame(self.total_frames - 1)

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self.current_frame_idx:
            self._show_frame(idx)

    def _mark_start(self):
        """Marca el frame actual como inicio del flash."""
        self.mark_start_frame = self.current_frame_idx
        t = self.current_frame_idx / self.fps if self.fps > 0 else 0
        self.start_info_var.set(f'Frame {self.current_frame_idx + 1} = {t:.3f}s')
        self._show_frame(self.current_frame_idx)

    def _mark_end(self):
        """Marca el frame actual como fin del flash."""
        self.mark_end_frame = self.current_frame_idx
        t = self.current_frame_idx / self.fps if self.fps > 0 else 0
        self.end_info_var.set(f'Frame {self.current_frame_idx + 1} = {t:.3f}s')
        self._show_frame(self.current_frame_idx)

    def _on_accept(self):
        """Acepta los marcadores y actualiza las variables de flash."""
        if self.mark_start_frame is not None:
            t_start = self.mark_start_frame / self.fps if self.fps > 0 else 0
            self.flash_start_var.set(t_start)
        if self.mark_end_frame is not None:
            t_end = self.mark_end_frame / self.fps if self.fps > 0 else 0
            self.flash_end_var.set(t_end)
        self._cleanup()
        self.window.destroy()

    def _on_cancel(self):
        """Cancela sin guardar."""
        self._cleanup()
        self.window.destroy()

    def _cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoSettingsWindow:
    """
    Ventana de configuracion de video.
    
    Permite al usuario configurar:
    - FPS y duracion de cada fase (Basal, Iluminacion, Relajacion)
    - Parametros de deteccion automatica de flash
    - Deteccion manual de flash con visor frame a frame
    - Tiempos de inicio y fin de flash (manual o automatico)
    """

    def __init__(self, parent, video_path, 
                 fase_basal_fps, fase_basal_dur,
                 fase_contraccion_fps, fase_contraccion_dur,
                 fase_relajacion_fps, fase_relajacion_dur,
                 flash_detection_auto, flash_calibration_frames,
                 flash_change_threshold, flash_smoothing_window,
                 flash_start_sec, flash_end_sec,
                 on_accept_callback, status_bar=None):
        self.parent = parent
        self.video_path = video_path
        self.fase_basal_fps = fase_basal_fps
        self.fase_basal_dur = fase_basal_dur
        self.fase_contraccion_fps = fase_contraccion_fps
        self.fase_contraccion_dur = fase_contraccion_dur
        self.fase_relajacion_fps = fase_relajacion_fps
        self.fase_relajacion_dur = fase_relajacion_dur
        self.flash_detection_auto = flash_detection_auto
        self.flash_calibration_frames = flash_calibration_frames
        self.flash_change_threshold = flash_change_threshold
        self.flash_smoothing_window = flash_smoothing_window
        self.flash_start_sec = flash_start_sec
        self.flash_end_sec = flash_end_sec
        self.on_accept_callback = on_accept_callback
        self.status_bar = status_bar
        self.window = None
        
        self._create_window()

    def _create_window(self):
        """Crea y configura la ventana de ajustes."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Ajustes de Video")
        self.window.geometry("450x620")
        self.window.resizable(False, False)
        self.window.grab_set()
        
        container = ttk.Frame(self.window)
        container.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        main_fr = ttk.Frame(scrollable_frame, padding=15)
        main_fr.pack(fill='both', expand=True)
        
        self._add_video_info(main_fr)
        self._add_phase_settings(main_fr)
        self._add_flash_manual_detection(main_fr)
        self._add_flash_detection_settings(main_fr)
        self._add_buttons(main_fr)

    def _add_video_info(self, parent):
        """Agrega informacion del video."""
        ttk.Label(parent, text=f'Video: {os.path.basename(self.video_path)}',
                  font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=5)

    def _add_phase_settings(self, parent):
        """Agrega configuracion de fases del video."""
        ttk.Label(parent, text='Fases del Video:',
                  font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))
        
        fase1_fr = ttk.LabelFrame(parent, text='1. Basal (Pre-estimulo)', padding=8)
        fase1_fr.pack(fill='x', pady=(0, 6))
        self._create_phase_row(fase1_fr, self.fase_basal_fps, self.fase_basal_dur)
        
        fase2_fr = ttk.LabelFrame(parent, text='2. Iluminacion (Estimulo)', padding=8)
        fase2_fr.pack(fill='x', pady=(0, 6))
        self._create_phase_row(fase2_fr, self.fase_contraccion_fps, self.fase_contraccion_dur)
        
        fase3_fr = ttk.LabelFrame(parent, text='3. Relajacion (Post-estimulo)', padding=8)
        fase3_fr.pack(fill='x', pady=(0, 10))
        self._create_phase_row(fase3_fr, self.fase_relajacion_fps, self.fase_relajacion_dur)

    def _create_phase_row(self, parent, fps_var, dur_var):
        """Crea una fila de configuracion de fase."""
        ttk.Label(parent, text='FPS:').grid(row=0, column=0, sticky='w', padx=(5, 2))
        ttk.Entry(parent, textvariable=fps_var, width=6).grid(row=0, column=1, sticky='w', padx=(0, 10))
        ttk.Label(parent, text='Dur (s):').grid(row=0, column=2, sticky='w', padx=(0, 2))
        ttk.Entry(parent, textvariable=dur_var, width=6).grid(row=0, column=3, sticky='w', padx=(0, 5))

    def _add_flash_manual_detection(self, parent):
        """Agrega seccion de deteccion manual de flash."""
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=5)
        
        manual_fr = ttk.LabelFrame(parent, text='Deteccion Manual de Flash', padding=8)
        manual_fr.pack(fill='x', pady=(0, 6))
        
        ttk.Label(manual_fr, text='Usa el visor frame a frame para marcar el inicio y fin del flash:',
                  font=('Arial', 8)).pack(anchor='w', pady=(0, 6))
        
        btn_fr = ttk.Frame(manual_fr)
        btn_fr.pack(fill='x', pady=(0, 6))
        
        ttk.Button(btn_fr, text='Abrir Deteccion Manual',
                   command=self._open_manual_detection).pack(side='left')
        
        times_fr = ttk.LabelFrame(manual_fr, text='Tiempos de Flash (manual)', padding=6)
        times_fr.pack(fill='x')
        
        ttk.Label(times_fr, text='Inicio flash (s):').grid(row=0, column=0, sticky='w', padx=(5, 4))
        ttk.Entry(times_fr, textvariable=self.flash_start_sec, width=8).grid(row=0, column=1, sticky='w', padx=(0, 10))
        
        ttk.Label(times_fr, text='Fin flash (s):').grid(row=0, column=2, sticky='w', padx=(5, 4))
        ttk.Entry(times_fr, textvariable=self.flash_end_sec, width=8).grid(row=0, column=3, sticky='w')

    def _open_manual_detection(self):
        """Abre la ventana de deteccion manual de flash."""
        ManualFlashDetectionWindow(
            parent=self.window,
            video_path=self.video_path,
            flash_start_var=self.flash_start_sec,
            flash_end_var=self.flash_end_sec
        )

    def _add_flash_detection_settings(self, parent):
        """Agrega configuracion de deteccion de flash."""
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=5)
        
        ttk.Label(parent, text='Deteccion Automatica de Flash:',
                  font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        
        ttk.Checkbutton(parent, text='Habilitar deteccion automatica de flash',
                        variable=self.flash_detection_auto).pack(anchor='w', pady=(0, 8))
        
        flash_params_fr = ttk.LabelFrame(parent, text='Parametros de Deteccion', padding=8)
        flash_params_fr.pack(fill='x', pady=(0, 6))
        
        self._create_flash_param_row(flash_params_fr, 'Frames calibracion:',
                                     self.flash_calibration_frames)
        self._create_flash_param_row(flash_params_fr, 'Umbral cambio brillo:',
                                     self.flash_change_threshold)
        self._create_flash_param_row(flash_params_fr, 'Ventana suavizado:',
                                     self.flash_smoothing_window)

    def _create_flash_param_row(self, parent, label_text, var):
        """Crea una fila de parametro de deteccion de flash."""
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label_text).pack(side='left', padx=(0, 5))
        ttk.Entry(row, textvariable=var, width=6).pack(side='left')

    def _add_buttons(self, parent):
        """Agrega botones de accion."""
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(side='bottom', fill='x', padx=15, pady=10)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text='Cancelar', command=self._on_cancel).grid(
            row=0, column=0, padx=(0, 5), sticky='ew')
        ttk.Button(btn_frame, text='Aceptar', command=self._on_accept).grid(
            row=0, column=1, padx=(5, 0), sticky='ew')

    def _on_cancel(self):
        """Maneja el evento de cancelacion."""
        self.window.destroy()
        if self.status_bar:
            self.status_bar.config(text='Carga de video cancelada')

    def _on_accept(self):
        """Maneja el evento de aceptacion."""
        self.window.destroy()
        if self.on_accept_callback:
            self.on_accept_callback(self.video_path)


def create_video_settings_window(parent, video_path, 
                                  fase_vars, flash_vars,
                                  on_accept_callback, status_bar=None):
    """
    Factory function para crear una ventana de ajustes de video.
    
    Parametros
    ----------
    parent : tk.Tk or tk.Toplevel
        Ventana padre
    video_path : str
        Ruta del video
    fase_vars : dict
        Diccionario con variables de fases:
        {
            'basal_fps', 'basal_dur',
            'contraccion_fps', 'contraccion_dur',
            'relajacion_fps', 'relajacion_dur'
        }
    flash_vars : dict
        Diccionario con variables de flash:
        {
            'auto', 'calibration_frames', 'change_threshold',
            'smoothing_window', 'flash_start_sec', 'flash_end_sec'
        }
    on_accept_callback : callable
        Funcion a llamar al aceptar
    status_bar : tk.Label, optional
        Barra de estado
        
    Retorna
    -------
    VideoSettingsWindow
        Instancia de la ventana
    """
    return VideoSettingsWindow(
        parent=parent,
        video_path=video_path,
        fase_basal_fps=fase_vars['basal_fps'],
        fase_basal_dur=fase_vars['basal_dur'],
        fase_contraccion_fps=fase_vars['contraccion_fps'],
        fase_contraccion_dur=fase_vars['contraccion_dur'],
        fase_relajacion_fps=fase_vars['relajacion_fps'],
        fase_relajacion_dur=fase_vars['relajacion_dur'],
        flash_detection_auto=flash_vars['auto'],
        flash_calibration_frames=flash_vars['calibration_frames'],
        flash_change_threshold=flash_vars['change_threshold'],
        flash_smoothing_window=flash_vars['smoothing_window'],
        flash_start_sec=flash_vars['flash_start_sec'],
        flash_end_sec=flash_vars['flash_end_sec'],
        on_accept_callback=on_accept_callback,
        status_bar=status_bar
    )
