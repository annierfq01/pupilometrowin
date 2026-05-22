"""
range_slider.py - Widget de Slider de Rango Dual
================================================
Widget personalizado de Tkinter para seleccionar un rango con dos handles.
"""

import tkinter as tk


class RangeSlider(tk.Canvas):
    """
    Slider de rango dual que permite seleccionar un intervalo entre 0.0 y 1.0.
    
    Attributes:
        H: Altura del canvas en pixels
        RAIL_H: Altura de la barra del slider
        R: Radio de los handles
        
    Args:
        parent: Widget padre de Tkinter
        on_change: Callback opcional que recibe (start, end) cuando cambia el rango
        **kwargs: Argumentos adicionales para tk.Canvas
    """
    
    H = 28
    RAIL_H = 6
    R = 9

    def __init__(self, parent, on_change=None, **kwargs):
        """
        Inicializa el RangeSlider.
        
        Args:
            parent: Widget padre
            on_change: Funcion callback(start, end) llamada al cambiar
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
        self._start = 0.0
        self._end = 1.0
        self._drag = None
        
        self.bind('<Configure>', lambda e: self._draw())
        self.bind('<ButtonPress-1>', self._on_press)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def set_range(self, s, e):
        """
        Establece el rango del slider.
        
        Args:
            s: Valor inicial (0.0 - 1.0)
            e: Valor final (0.0 - 1.0)
        """
        self._start = max(0.0, min(1.0, s))
        self._end = max(0.0, min(1.0, e))
        if self._start > self._end:
            self._start, self._end = self._end, self._start
        self._draw()

    def get_range(self):
        """
        Obtiene el rango actual.
        
        Returns:
            Tupla (start, end) con valores entre 0.0 y 1.0
        """
        return self._start, self._end

    def _draw(self):
        """Dibuja el slider con sus handles."""
        w = self.winfo_width() or 200
        self.delete('all')
        cy = self.H // 2
        x0 = self.R + 2
        x1 = w - self.R - 2
        rng = max(x1 - x0, 1)
        sx = int(x0 + self._start * rng)
        ex = int(x0 + self._end * rng)
        
        # Barra de fondo
        self.create_rectangle(x0, cy - self.RAIL_H // 2, x1, cy + self.RAIL_H // 2,
                               fill='#cccccc', outline='')
        
        # Barra seleccionada
        self.create_rectangle(sx, cy - self.RAIL_H // 2, ex, cy + self.RAIL_H // 2,
                               fill='#4a9eff', outline='')
        
        # Handle inicio (azul oscuro)
        self.create_oval(sx - self.R, cy - self.R, sx + self.R, cy + self.R,
                           fill='#1a6fcc', outline='white', width=2)
        
        # Handle fin (rojo)
        self.create_oval(ex - self.R, cy - self.R, ex + self.R, cy + self.R,
                           fill='#cc3300', outline='white', width=2)

    def _frac(self, mx):
        """
        Convierte coordenada X a valor fraccional.
        
        Args:
            mx: Posicion X en pixels
            
        Returns:
            Valor entre 0.0 y 1.0
        """
        w = self.winfo_width() or 200
        x0 = self.R + 2
        x1 = w - self.R - 2
        return max(0.0, min(1.0, (mx - x0) / max(x1 - x0, 1)))

    def _on_press(self, e):
        """
        Maneja el evento de presionar el mouse.
        
        Determina cual handle se esta arrastrando.
        """
        w = self.winfo_width() or 200
        x0 = self.R + 2
        rng = max(w - 2 * self.R - 4, 1)
        sx = x0 + self._start * rng
        ex = x0 + self._end * rng
        self._drag = 'start' if abs(e.x - sx) <= abs(e.x - ex) else 'end'

    def _on_drag(self, e):
        """
        Maneja el evento de arrastrar.
        
        Actualiza el valor del handle seleccionado.
        """
        if not self._drag:
            return
        f = self._frac(e.x)
        if self._drag == 'start':
            self._start = min(f, self._end - 0.001)
        else:
            self._end = max(f, self._start + 0.001)
        self._draw()
        if self._on_change:
            self._on_change(self._start, self._end)

    def _on_release(self, e):
        """Maneja el evento de soltar el mouse."""
        self._drag = None
