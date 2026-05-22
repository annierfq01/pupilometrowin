"""
detection.py - Core Detection Module
====================================
Algoritmo de deteccion de pupila e iris CONCENTRICOS.

Estrategia simplificada y robusta:
1. La pupila es el circulo oscuro MAS GRANDE de la imagen
2. El iris es el circulo concentrico alrededor de la pupila
3. Se usan umbrales bajos para encontrar zonas muy oscuras
4. Se manejan reflejos de luz
"""

import cv2
import numpy as np
import math


def _find_largest_dark_circle(gray, threshold):
    """
    Encuentra el circulo oscuro mas grande de la imagen.
    Este debe ser la pupila.
    Retorna (cx, cy, r) o None.
    """
    h, w = gray.shape
    
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    best = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cx, cy = ellipse[0]
            axes = ellipse[1]
            rx, ry = axes[0], axes[1]
            r = (rx + ry) / 4.0
        else:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy = float(cx), float(cy)
        
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity < 0.4:
            continue
        
        mask_circle = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask_circle, (int(cx), int(cy)), int(r), 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask_circle)[0]
        
        score = area * (1 - mean_intensity / 255)
        
        if score > best_score:
            best_score = score
            best = (cx, cy, r, mean_intensity, circularity)
    
    if best is None:
        return None, None, None
    
    cx, cy, r, mean_int, circ = best
    return int(cx), int(cy), int(r)


def _refine_pupil_edge(gray, cx, cy, approx_r):
    """
    Refina el borde de la pupila buscando el radio exacto
    donde la intensidad cambia significativamente.
    """
    h, w = gray.shape
    
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    
    r_min = max(5, int(approx_r * 0.5))
    r_max = min(int(approx_r * 1.5), cx, cy, w - cx, h - cy)
    
    if r_max <= r_min:
        return approx_r
    
    best_r = approx_r
    best_score = 0
    
    for r in range(r_min, r_max, 2):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        
        mean_val = cv2.mean(gray, mask=mask)[0]
        
        if mean_val < 80:
            score = (80 - mean_val) * r
            if score > best_score:
                best_score = score
                best_r = r
    
    return best_r if best_r > 0 else approx_r


def _find_iris_edge(gray, cx, cy, pupil_r):
    """
    Encuentra el borde exterior del iris buscando maximo contraste.
    """
    h, w = gray.shape
    
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    
    r_min = int(pupil_r * 1.2)
    r_max = min(int(pupil_r * 4), cx, cy, w - cx, h - cy)
    
    if r_max <= r_min:
        return None
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    best_r = None
    best_contrast = 0
    
    for r in range(r_min, r_max, 3):
        inner_points = []
        outer_points = []
        
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            
            xi = int(cx + (r - 3) * math.cos(rad))
            yi = int(cy + (r - 3) * math.sin(rad))
            if 0 <= xi < w and 0 <= yi < h:
                inner_points.append(blurred[yi, xi])
            
            xo = int(cx + (r + 3) * math.cos(rad))
            yo = int(cy + (r + 3) * math.sin(rad))
            if 0 <= xo < w and 0 <= yo < h:
                outer_points.append(blurred[yo, xo])
        
        if inner_points and outer_points:
            contrast = abs(np.mean(outer_points) - np.mean(inner_points))
            if contrast > best_contrast:
                best_contrast = contrast
                best_r = r
    
    if best_r is None or best_contrast < 10:
        return int(pupil_r * 2.5)
    
    refine_min = max(r_min, best_r - 5)
    refine_max = min(r_max, best_r + 5)
    
    best_refined = best_r
    best_refined_contrast = best_contrast
    
    for r in range(refine_min, refine_max):
        inner_points = []
        outer_points = []
        
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            
            xi = int(cx + (r - 2) * math.cos(rad))
            yi = int(cy + (r - 2) * math.sin(rad))
            if 0 <= xi < w and 0 <= yi < h:
                inner_points.append(blurred[yi, xi])
            
            xo = int(cx + (r + 2) * math.cos(rad))
            yo = int(cy + (r + 2) * math.sin(rad))
            if 0 <= xo < w and 0 <= yo < h:
                outer_points.append(blurred[yo, xo])
        
        if inner_points and outer_points:
            contrast = abs(np.mean(outer_points) - np.mean(inner_points))
            if contrast > best_refined_contrast:
                best_refined_contrast = contrast
                best_refined = r
    
    return best_refined


def _find_approx_center(gray):
    """
    Encuentra una aproximacion del centro de la pupila
    buscando la region mas oscura de la imagen.
    """
    h, w = gray.shape
    
    result = cv2.minMaxLoc(gray)
    min_val = result[0]
    min_loc = result[2]
    
    if min_val > 80:
        return w // 2, h // 2
    
    scale = 4
    small_h, small_w = h // scale, w // scale
    small = cv2.resize(gray, (small_w, small_h))
    
    blur_small = cv2.blur(small, (15, 15))
    min_loc_small = cv2.minMaxLoc(blur_small)[2]
    
    approx_cx = min_loc_small[0] * scale
    approx_cy = min_loc_small[1] * scale
    
    return approx_cx, approx_cy


def detect_all(image, dark_eyes=False, threshold=50, blur_size=5, smooth=15):
    """
    Deteccion de pupila e iris.
    
    Pasos:
    1. Encontrar el centro aproximado (zona mas oscura)
    2. Encontrar el circulo oscuro mas grande = pupila
    3. Encontrar el borde del iris = borde de maximo contraste
    4. Asegurar concentricidad
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    approx_cx, approx_cy = _find_approx_center(blurred)
    
    pupil_cx, pupil_cy, pupil_r = _find_largest_dark_circle(blurred, 60)
    
    if pupil_r is None or pupil_r < 5:
        pupil_cx, pupil_cy = approx_cx, approx_cy
        pupil_r = _refine_pupil_edge(blurred, pupil_cx, pupil_cy, 30)
    
    if pupil_r < 5:
        return None, None, None, None
    
    pupil_center = (pupil_cx, pupil_cy)
    pupil_radius = pupil_r
    
    pupil_radius = _refine_pupil_edge(blurred, pupil_cx, pupil_cy, pupil_r)
    
    iris_r = _find_iris_edge(blurred, pupil_cx, pupil_cy, pupil_radius)
    
    if iris_r is None or iris_r <= pupil_radius:
        iris_r = int(pupil_radius * 2.5)
    
    ratio = pupil_radius / iris_r
    if ratio < 0.15:
        pupil_radius = int(iris_r * 0.15)
    elif ratio > 0.5:
        pupil_radius = int(iris_r * 0.5)
    
    iris_center = pupil_center
    iris_radius = iris_r
    
    return pupil_center, pupil_radius, iris_center, iris_radius


def draw_detections(image, pupil_center, pupil_radius, iris_center, iris_radius, line_width=2):
    """Dibuja las detecciones en la imagen."""
    out = image.copy()
    
    if iris_center and iris_radius and iris_radius > 0:
        cv2.circle(out, iris_center, iris_radius, (255, 0, 0), line_width)
    
    if pupil_center and pupil_radius and pupil_radius > 0:
        cv2.circle(out, pupil_center, pupil_radius, (0, 0, 255), line_width)
        cv2.circle(out, pupil_center, max(2, line_width), (0, 255, 0), -1)
    
    return out
