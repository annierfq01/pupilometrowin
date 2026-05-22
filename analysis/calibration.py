"""
calibration.py - Rule Calibration Module
=========================================
Modulo de calibracion usando regla rosa fosforescente.
Detecta los dos puntos negros en el fondo rosa y calcula la escala px/mm.
"""

import cv2
import numpy as np
import math
from typing import Dict, Optional, Tuple


def detectar_region_fondo_mas_grande(img_hsv, rangos_color):
    """
    Detecta la region mas grande del color especificado.
    
    Args:
        img_hsv: Imagen en espacio HSV
        rangos_color: Lista de tuplas (bajo, alto) para el color
        
    Returns:
        Tupla (mascara_fondo, contorno_mas_grande, area_max)
    """
    mascara_total = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    
    for bajo, alto in rangos_color:
        mascara = cv2.inRange(img_hsv, bajo, alto)
        mascara_total = cv2.bitwise_or(mascara_total, mascara)
    
    kernel = np.ones((5, 5), np.uint8)
    mascara_total = cv2.morphologyEx(mascara_total, cv2.MORPH_OPEN, kernel)
    mascara_total = cv2.morphologyEx(mascara_total, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return None, None, 0
    
    contorno_mas_grande = max(contornos, key=cv2.contourArea)
    area_max = cv2.contourArea(contorno_mas_grande)
    
    mascara_fondo = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mascara_fondo, [contorno_mas_grande], -1, 255, -1)
    
    return mascara_fondo, contorno_mas_grande, area_max


def detectar_puntos_negros_en_region(mascara_fondo, img_gris, img_hsv):
    """
    Detecta los puntos negros mas grandes dentro de la region del fondo.
    
    Args:
        mascara_fondo: Mascara de la region del fondo
        img_gris: Imagen en escala de grises
        img_hsv: Imagen en HSV (opcional, para verificacion adicional)
        
    Returns:
        Lista de puntos detectados ordenados por area
    """
    _, mascara_negro = cv2.threshold(img_gris, 80, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((2, 2), np.uint8)
    mascara_negro = cv2.morphologyEx(mascara_negro, cv2.MORPH_OPEN, kernel)
    mascara_negro = cv2.morphologyEx(mascara_negro, cv2.MORPH_CLOSE, kernel)
    
    mascara_puntos = cv2.bitwise_and(mascara_negro, mascara_fondo)
    
    contornos, _ = cv2.findContours(mascara_puntos, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    puntos = []
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        
        if area > 15:
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                perimetro = cv2.arcLength(contorno, True)
                if perimetro > 0:
                    circularidad = 4 * np.pi * area / (perimetro * perimetro)
                    
                    puntos.append({
                        'centro': (cx, cy),
                        'area': area,
                        'circularidad': circularidad,
                        'contorno': contorno
                    })
    
    puntos.sort(key=lambda p: p['area'], reverse=True)
    
    return puntos[:2]


def calibrar_regla_rosa_fosforescente(ruta_o_imagen, distancia_real_mm=40.0):
    """
    Calibra la escala usando una imagen de regla rosa fosforescente.
    
    Busca la region mas grande de fondo rosa y luego los puntos negros mas grandes.
    
    Args:
        ruta_o_imagen: Ruta a la imagen o una imagen ya cargada (numpy array BGR)
        distancia_real_mm: Distancia real entre los dos puntos en milimetros
        
    Returns:
        Dict con resultados de calibracion o {'exito': False, 'error': mensaje}
    """
    if isinstance(ruta_o_imagen, np.ndarray):
        img = ruta_o_imagen.copy()
    else:
        img = cv2.imread(ruta_o_imagen)
        if img is None:
            return {'exito': False, 'error': 'No se pudo cargar la imagen'}
    
    img_original = img.copy()
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    rangos_rosa = [
        (np.array([140, 80, 80]), np.array([170, 255, 255])),
        (np.array([145, 100, 100]), np.array([165, 255, 255])),
        (np.array([150, 120, 100]), np.array([165, 255, 255])),
        (np.array([140, 50, 100]), np.array([170, 200, 255])),
    ]
    
    mascara_fondo, contorno_fondo, area_fondo = detectar_region_fondo_mas_grande(img_hsv, rangos_rosa)
    
    if mascara_fondo is None:
        return {'exito': False, 'error': 'No se detecto fondo rosa en la imagen'}
    
    area_total = img.shape[0] * img.shape[1]
    porcentaje_fondo = (area_fondo / area_total) * 100
    
    puntos = detectar_puntos_negros_en_region(mascara_fondo, img_gris, img_hsv)
    
    if len(puntos) < 2:
        return {'exito': False, 'error': f'Solo se detectaron {len(puntos)} puntos negros (se necesitan 2)'}
    
    puntos.sort(key=lambda p: p['centro'][0])
    
    x1, y1 = puntos[0]['centro']
    x2, y2 = puntos[1]['centro']
    
    distancia_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    escala_px_mm = distancia_px / distancia_real_mm
    
    desviacion_y = abs(y1 - y2)
    angulo = math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    error_deteccion_px = 0.5
    error_estimado_mm = error_deteccion_px / escala_px_mm
    
    calidad = 1.0
    calidad -= min(desviacion_y / distancia_px, 0.2)
    calidad -= min(abs(angulo) / 30, 0.2)
    calidad -= max(0, (30 - porcentaje_fondo) / 30) * 0.3
    calidad = max(0, min(1, calidad))
    
    img_procesada = img_original.copy()
    
    cv2.drawContours(img_procesada, [contorno_fondo], -1, (255, 100, 0), 3)
    
    for i, punto in enumerate(puntos):
        cx, cy = punto['centro']
        cv2.circle(img_procesada, (cx, cy), 12, (0, 255, 0), 3)
        cv2.circle(img_procesada, (cx, cy), 3, (0, 0, 255), -1)
        
        if i == 0:
            cv2.putText(img_procesada, "PUNTO 1 (0mm)", (cx-40, cy-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img_procesada, f"PUNTO 2 ({distancia_real_mm:.0f}mm)", (cx-45, cy-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.line(img_procesada, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    texto_dist = f"DISTANCIA: {distancia_px:.1f} px = {distancia_real_mm:.1f} mm"
    cv2.putText(img_procesada, texto_dist, ((x1+x2)//2 - 100, (y1+y2)//2 - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return {
        'exito': True,
        'escala_px_mm': escala_px_mm,
        'distancia_px': distancia_px,
        'distancia_mm': distancia_real_mm,
        'error_estimado_mm': error_estimado_mm,
        'punto1': (x1, y1),
        'punto2': (x2, y2),
        'area_punto1': puntos[0]['area'],
        'area_punto2': puntos[1]['area'],
        'angulo': angulo,
        'desviacion_vertical': desviacion_y,
        'calidad': calidad,
        'area_fondo': area_fondo,
        'porcentaje_fondo': porcentaje_fondo,
        'contorno_fondo': contorno_fondo,
        'imagen_procesada': img_procesada
    }


def aplicar_calibracion(radio_px: float, escala_px_mm: float) -> float:
    """
    Convierte un radio en pixeles a milimetros.
    
    Args:
        radio_px: Radio en pixeles
        escala_px_mm: Escala en px/mm
        
    Returns:
        Radio en milimetros
    """
    return radio_px / escala_px_mm


def convertir_diametro_mm(diametro_px: float, escala_px_mm: float) -> float:
    """
    Convierte un diametro en pixeles a milimetros.
    
    Args:
        diametro_px: Diametro en pixeles
        escala_px_mm: Escala en px/mm
        
    Returns:
        Diametro en milimetros
    """
    return diametro_px / escala_px_mm
