"""
analisis_video.py - Video Analysis Settings Window
==================================================
Ventana de configuracion para ajustes de analisis de video,
incluyendo fases, deteccion de flash y otros parametros.

Autor: Pupilometer Development Team
"""

import tkinter as tk
from tkinter import ttk, messagebox


class AnalisisVideoWindow:
    """
    Ventana de ajustes de analisis de video.
    
    Permite al usuario configurar los mismos parametros que
    VideoSettingsWindow pero orientada al analisis de video
    ya cargado. Los cambios se aplicaran al cargar un nuevo video.
    
    Parametros
    ----------
    parent : tk.Tk or tk.Toplevel
        Ventana padre
    fase_basal_fps : tk.IntVar
        Variable para FPS de fase basal
    fase_basal_dur : tk.IntVar
        Variable para duracion de fase basal
    fase_contraccion_fps : tk.IntVar
        Variable para FPS de fase de iluminacion
    fase_contraccion_dur : tk.IntVar
        Variable para duracion de fase de iluminacion
    fase_relajacion_fps : tk.IntVar
        Variable para FPS de fase de relajacion
    fase_relajacion_dur : tk.IntVar
        Variable para duracion de fase de relajacion
    flash_detection_auto : tk.BooleanVar
        Variable para habilitar deteccion automatica de flash
    flash_calibration_frames : tk.IntVar
        Frames para calibracion de deteccion de flash
    flash_change_threshold : tk.DoubleVar
        Umbral de cambio de brillo para deteccion de flash
    flash_smoothing_window : tk.IntVar
        Ventana de suavizado para deteccion de flash
    on_accept_callback : callable
        Funcion a llamar cuando el usuario acepta los ajustes
    """

    def __init__(self, parent,
                 fase_basal_fps, fase_basal_dur,
                 fase_contraccion_fps, fase_contraccion_dur,
                 fase_relajacion_fps, fase_relajacion_dur,
                 flash_detection_auto, flash_calibration_frames,
                 flash_change_threshold, flash_smoothing_window,
                 on_accept_callback=None):
        self.parent = parent
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
        self.on_accept_callback = on_accept_callback
        self.window = None
        
        self._create_window()

    def _create_window(self):
        """Crea y configura la ventana de ajustes."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Analisis de Video - Ajustes")
        self.window.geometry("450x500")
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
        
        self._add_phase_settings(main_fr)
        self._add_flash_detection_settings(main_fr)
        self._add_info_note(main_fr)
        self._add_buttons(main_fr)

    def _add_phase_settings(self, parent):
        """Agrega configuracion de fases del video."""
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
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

    def _add_flash_detection_settings(self, parent):
        """Agrega configuracion de deteccion de flash."""
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
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

    def _add_info_note(self, parent):
        """Agrega nota informativa."""
        ttk.Label(parent, text='Los cambios se aplicaran al cargar un nuevo video.',
                  font=('Arial', 8), foreground='gray').pack(anchor='w', pady=(5, 10))

    def _add_buttons(self, parent):
        """Agrega botones de accion."""
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(side='bottom', fill='x', padx=15, pady=10)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text='Cancelar', command=self.window.destroy).grid(
            row=0, column=0, padx=(0, 5), sticky='ew')
        ttk.Button(btn_frame, text='Aceptar', 
                   command=lambda: self._on_accept()).grid(
            row=0, column=1, padx=(5, 0), sticky='ew')

    def _on_accept(self):
        """Maneja el evento de aceptacion."""
        if self.on_accept_callback:
            self.on_accept_callback(self.window)
        else:
            self.window.destroy()
            messagebox.showinfo('Info', 
                'Los cambios en fases se aplicaran al cargar un nuevo video.')


def create_analisis_video_window(parent,
                                  fase_vars, flash_vars,
                                  on_accept_callback=None):
    """
    Factory function para crear una ventana de ajustes de analisis.
    
    Parametros
    ----------
    parent : tk.Tk or tk.Toplevel
        Ventana padre
    fase_vars : dict
        Diccionario con variables de fases
    flash_vars : dict
        Diccionario con variables de flash
    on_accept_callback : callable, optional
        Funcion a llamar al aceptar
        
    Retorna
    -------
    AnalisisVideoWindow
        Instancia de la ventana
    """
    return AnalisisVideoWindow(
        parent=parent,
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
        on_accept_callback=on_accept_callback
    )
