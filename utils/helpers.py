"""
helpers.py - Utility Functions
==============================
Funciones utilitarias para el modulo pupilometer.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> Tuple[np.ndarray, float]:
    """
    Redimensiona una imagen para que quepa dentro de los limites dados.
    
    Args:
        image: Imagen a redimensionar
        max_width: Ancho maximo
        max_height: Alto maximo
        
    Returns:
        Tupla (imagen_redimensionada, escala)
    """
    h, w = image.shape[:2]
    
    if w <= max_width and h <= max_height:
        return image, 1.0
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    return resized, scale


def create_thumbnail(image: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Crea un thumbnail de la imagen.
    
    Args:
        image: Imagen original
        size: Tamano del thumbnail (cuadrado)
        
    Returns:
        Thumbnail redimensionado
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def image_to_display(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR a RGB para mostrar en Tkinter.
    
    Args:
        image: Imagen en BGR
        
    Returns:
        Imagen en RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_coordinates(x: float, y: float, 
                          offset_x: int = 0, offset_y: int = 0,
                          scale: float = 1.0) -> Tuple[float, float]:
    """
    Normaliza coordenadas aplicando offset y escala.
    
    Args:
        x, y: Coordenadas originales
        offset_x, offset_y: Offset a aplicar
        scale: Escala a aplicar
        
    Returns:
        Tupla de coordenadas normalizadas
    """
    return (x - offset_x) / scale, (y - offset_y) / scale


def denormalize_coordinates(x: float, y: float,
                            offset_x: int = 0, offset_y: int = 0,
                            scale: float = 1.0) -> Tuple[int, int]:
    """
    Convierte coordenadas normalizadas a pixeles.
    
    Args:
        x, y: Coordenadas normalizadas
        offset_x, offset_y: Offset a aplicar
        scale: Escala a aplicar
        
    Returns:
        Tupla de coordenadas en pixeles
    """
    return int(x * scale + offset_x), int(y * scale + offset_y)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Imagen en escala de grises
        clip_limit: Limite de contraste
        tile_grid_size: Tamano de tile
        
    Returns:
        Imagen con CLAHE aplicado
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def safe_crop(image: np.ndarray, x1: int, y1: int, 
              x2: int, y2: int) -> np.ndarray:
    """
    Recorta una imagen con verificacion de limites.
    
    Args:
        image: Imagen a recortar
        x1, y1, x2, y2: Coordenadas del rectangulo de recorte
        
    Returns:
        Imagen recortada
    """
    h, w = image.shape[:2]
    
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    
    return image[y1:y2, x1:x2]


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Asegura que la imagen este en escala de grises.
    
    Args:
        image: Imagen BGR o en escala de grises
        
    Returns:
        Imagen en escala de grises
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def circularity_score(contour) -> float:
    """
    Calcula el score de circularidad de un contorno.
    
    Args:
        contour: Contorno de OpenCV
        
    Returns:
        Score de circularidad (0-1, 1 = circulo perfecto)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0.0
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(1.0, circularity)


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4) -> np.ndarray:
    """
    Superpone una mascara sobre la imagen.
    
    Args:
        image: Imagen original
        mask: Mascara binaria
        alpha: Transparencia de la mascara
        
    Returns:
        Imagen con mascara superpuesta
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    overlay = image.copy()
    overlay[mask > 127] = (0, 0, 255)
    
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
