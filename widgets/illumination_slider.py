"""
illumination_slider.py - Widget de Slider de Iluminacion
========================================================
Widget personalizado de Tkinter para seleccionar el periodo de iluminacion
con marcas visuales de ON/OFF.
"""

import tkinter as tk


class IlluminationSlider(tk.Canvas):
    """
    Slider dual para seleccionar inicio y fin de iluminacion con marcas visuales.
    
    Muestra tres zonas: oscura (antes), iluminada (entre), oscura (despues).
    Handles verdes para ON, rojos para OFF.
    
    Attributes:
        H: Altura del canvas en pixels
        RAIL_H: Altura de la barra
        R: Radio de los handles
        
    Args:
        parent: Widget padre de Tkinter
        on_change: Callback opcional que recibe (on_frac, off_frac)
        **kwargs: Argumentos adicionales para tk.Canvas
    """
    
    H = 36
    RAIL_H = 8
    R = 10

    def __init__(self, parent, on_change=None, **kwargs):
        """
        Inicializa el IlluminationSlider.
        
        Args:
            parent: Widget padre
            on_change: Funcion callback(on_frac, off_frac) llamada al cambiar
            **kwargs: Argumentos para tk.Canvas
        """
        kwargs.setdefault('height', self.H)
        try:
            bg = parent.cget('bg')
        except Exception:
            bg = '#f0f0f0'
        kwargs.setdefault('bg', bg)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)
        
        self._on_change = on_change
        self._light_on = 0.1
        self._light_off = 0.9
        self._drag = None
        
        self.bind('<Configure>', lambda e: self._draw())
        self.bind('<ButtonPress-1>', self._on_press)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def set_illumination(self, on_frac, off_frac):
        """
        Establece los valores de iluminacion.
        
        Args:
            on_frac: Fraccion donde inicia la iluminacion (0.0 - 1.0)
            off_frac: Fraccion donde termina la iluminacion (0.0 - 1.0)
        """
        self._light_on = max(0.0, min(1.0, on_frac))
        self._light_off = max(0.0, min(1.0, off_frac))
        if self._light_on > self._light_off:
            self._light_on, self._light_off = self._light_off, self._light_on
        self._draw()

    def get_illumination(self):
        """
        Obtiene los valores de iluminacion actuales.
        
        Returns:
            Tupla (on_frac, off_frac) con valores entre 0.0 y 1.0
        """
        return self._light_on, self._light_off

    def _draw(self):
        """Dibuja el slider con sus tres zonas y handles."""
        w = self.winfo_width() or 300
        self.delete('all')
        cy = self.H // 2
        x0 = self.R + 2
        x1 = w - self.R - 2
        rng = max(x1 - x0, 1)
        
        # Rail completo con borde
        self.create_rectangle(x0, cy - self.RAIL_H // 2, x1, cy + self.RAIL_H // 2,
                               fill='#333333', outline='#555555', width=1)
        
        # Calcular posiciones
        on_x = int(x0 + self._light_on * rng)
        off_x = int(x0 + self._light_off * rng)
        
        # Zona oscura (antes de iluminacion)
        self.create_rectangle(x0, cy - self.RAIL_H // 2, on_x, cy + self.RAIL_H // 2,
                               fill='#222222', outline='')
        
        # Zona iluminada (dorado)
        self.create_rectangle(on_x, cy - self.RAIL_H // 2, off_x, cy + self.RAIL_H // 2,
                               fill='#FFD700', outline='')
        
        # Zona oscura (despues de iluminacion)
        self.create_rectangle(off_x, cy - self.RAIL_H // 2, x1, cy + self.RAIL_H // 2,
                               fill='#222222', outline='')
        
        # Handle inicio (verde = luz ON)
        self.create_oval(on_x - self.R, cy - self.R, on_x + self.R, cy + self.R,
                           fill='#00CC00', outline='white', width=2)
        self.create_text(on_x, cy + self.R + 10, text='ON', font=('Arial', 7, 'bold'), fill='#00CC00')
        
        # Handle fin (rojo = luz OFF)
        self.create_oval(off_x - self.R, cy - self.R, off_x + self.R, cy + self.R,
                           fill='#FF4400', outline='white', width=2)
        self.create_text(off_x, cy + self.R + 10, text='OFF', font=('Arial', 7, 'bold'), fill='#FF4400')

    def _frac(self, mx):
        """
        Convierte coordenada X a valor fraccional.
        
        Args:
            mx: Posicion X en pixels
            
        Returns:
            Valor entre 0.0 y 1.0
        """
        w = self.winfo_width() or 300
        x0 = self.R + 2
        x1 = w - self.R - 2
        return max(0.0, min(1.0, (mx - x0) / max(x1 - x0, 1)))

    def _on_press(self, e):
        """
        Maneja el evento de presionar el mouse.
        
        Determina cual handle se esta arrastrando.
        """
        w = self.winfo_width() or 300
        x0 = self.R + 2
        rng = max(w - 2 * self.R - 4, 1)
        on_x = x0 + self._light_on * rng
        off_x = x0 + self._light_off * rng
        
        if abs(e.x - on_x) <= abs(e.x - off_x):
            self._drag = 'on'
        else:
            self._drag = 'off'

    def _on_drag(self, e):
        """
        Maneja el evento de arrastrar.
        
        Actualiza el valor del handle seleccionado.
        """
        if not self._drag:
            return
        f = self._frac(e.x)
        if self._drag == 'on':
            self._light_on = min(f, self._light_off - 0.01)
        else:
            self._light_off = max(f, self._light_on + 0.01)
        self._draw()
        if self._on_change:
            self._on_change(self._light_on, self._light_off)

    def _on_release(self, e):
        """Maneja el evento de soltar el mouse."""
        self._drag = None
