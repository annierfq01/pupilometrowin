"""
DarkCircle Detector - Pupilometer
=================================
Algoritmo de deteccion de pupila relajado que busca circulos oscuros
con alta tolerancia a iluminaciones y formas imperfectas.

Busca elementos que puedan ser la pupila considerando:
1. Zonas oscuras (con tolerancia a puntos iluminados)
2. Formas redondas o elipticas (tolerancia baja de circularidad)
3. Prioriza el mas grande y mas cercano al centro
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List


class DarkCircleDetector:
    """
    Detector de pupila relajado. Busca candidatos oscuros/redondos
    y selecciona el mas grande y centrado.
    """
    
    def __init__(self, 
                 min_radius: int = 5,
                 max_radius_ratio: float = 0.45,
                 darkness_threshold: int = 100,
                 circularity_threshold: float = 0.15,
                 center_bias_weight: float = 0.4,
                 size_weight: float = 0.4,
                 darkness_weight: float = 0.2):
        """
        Args:
            min_radius: Radio minimo esperado para la pupila (pixeles)
            max_radius_ratio: Ratio maximo del radio respecto a la imagen
            darkness_threshold: Umbral de oscuridad (mayor = mas permisivo)
            circularity_threshold: Minima circularidad aceptada (0.0-1.0)
            center_bias_weight: Peso para el sesgo hacia el centro
            size_weight: Peso para el tamano del circulo
            darkness_weight: Peso para la oscuridad
        """
        self.min_radius = min_radius
        self.max_radius_ratio = max_radius_ratio
        self.darkness_threshold = darkness_threshold
        self.circularity_threshold = circularity_threshold
        self.center_bias_weight = center_bias_weight
        self.size_weight = size_weight
        self.darkness_weight = darkness_weight
    
    def detect(self, gray: np.ndarray) -> Optional[Dict]:
        """
        Detecta la pupila con alta tolerancia a iluminaciones.
        """
        h, w = gray.shape
        center_x, center_y = w / 2, h / 2
        max_dist_to_center = np.sqrt(center_x**2 + center_y**2)
        
        max_radius = int(min(w, h) * self.max_radius_ratio)
        if max_radius < self.min_radius:
            max_radius = min(w, h) // 2
        
        candidates = self._find_all_candidates(
            gray, center_x, center_y, 
            max_radius, max_dist_to_center
        )
        
        if not candidates:
            return self._fallback_detection(gray, center_x, center_y, max_radius)
        
        best = self._select_best_candidate(candidates, max_radius)
        
        refined = self._refine_detection(gray, best)
        
        return {
            'center': refined['center'],
            'radius': refined['radius'],
            'diameter': refined['radius'] * 2,
            'confidence': refined['confidence'],
            'mean_intensity': refined.get('mean_intensity', 50),
            'method': 'darkcircle'
        }
    
    def _find_all_candidates(self, gray: np.ndarray, center_x: float, center_y: float,
                            max_radius: float, max_dist: float) -> List[Dict]:
        """
        Encuentra todos los candidatos posibles usando multiples tecnicas
        para ser tolerante a diferentes condiciones de iluminacion.
        """
        all_candidates = []
        
        thresholds = [self.darkness_threshold, 80, 60, 120]
        
        for thresh in thresholds:
            candidates = self._find_candidates_with_threshold(
                gray, thresh, center_x, center_y, max_radius, max_dist
            )
            all_candidates.extend(candidates)
        
        candidates_by_dark_region = self._find_by_dark_regions(
            gray, center_x, center_y, max_radius, max_dist
        )
        all_candidates.extend(candidates_by_dark_region)
        
        seen = set()
        unique = []
        for c in all_candidates:
            key = (int(c['center'][0]/10), int(c['center'][1]/10), int(c['radius']/5))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique
    
    def _find_candidates_with_threshold(self, gray: np.ndarray, threshold: int,
                                         center_x: float, center_y: float,
                                         max_radius: float, max_dist: float) -> List[Dict]:
        """
        Encuentra candidatos usando un umbral especifico.
        """
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        h, w = gray.shape
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = np.pi * self.min_radius ** 2 * 0.3
            if area < min_area:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            if circularity < self.circularity_threshold:
                continue
            
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (ma, MA), _ = ellipse
                    radius = (ma + MA) / 4.0
                except:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    cx, cy = float(cx), float(cy)
            else:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cx, cy = float(cx), float(cy)
            
            if radius < self.min_radius or radius > max_radius:
                continue
            
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            mask_circle = np.zeros_like(gray)
            cv2.circle(mask_circle, (int(cx), int(cy)), int(radius), 255, -1)
            
            mean_intensity = cv2.mean(gray, mask=mask_circle)[0]
            
            roi = gray[max(0,int(cy-radius)):min(h,int(cy+radius)),
                       max(0,int(cx-radius)):min(w,int(cx+radius))]
            if roi.size > 0:
                darkness_ratio = np.sum(roi < threshold) / roi.size
            else:
                darkness_ratio = 0.5
            
            area_score = min(1.0, area / (np.pi * max_radius ** 2))
            center_score = 1.0 - (dist_to_center / max_dist)
            dark_score = darkness_ratio * 0.5 + (1.0 - mean_intensity/255) * 0.5
            
            total_score = (
                center_score * self.center_bias_weight +
                area_score * self.size_weight +
                dark_score * self.darkness_weight
            )
            
            candidates.append({
                'center': (float(cx), float(cy)),
                'radius': float(radius),
                'area': float(area),
                'circularity': float(circularity),
                'mean_intensity': float(mean_intensity),
                'darkness_ratio': float(darkness_ratio),
                'center_score': float(center_score),
                'total_score': float(total_score),
                'threshold_used': threshold
            })
        
        return candidates
    
    def _find_by_dark_regions(self, gray: np.ndarray, center_x: float, center_y: float,
                              max_radius: float, max_dist: float) -> List[Dict]:
        """
        Encuentra regiones oscuras usando blob detection o busqueda de componentes.
        """
        blurred = cv2.blur(gray, (5, 5))
        
        _, binary = cv2.threshold(blurred, self.darkness_threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        candidates = []
        h, w = gray.shape
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            if area < 50:
                continue
            
            min_area = np.pi * self.min_radius ** 2 * 0.3
            if area < min_area:
                continue
            
            radius = np.sqrt(area / np.pi)
            
            if radius < self.min_radius or radius > max_radius:
                continue
            
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            mask_circle = np.zeros_like(gray)
            cv2.circle(mask_circle, (int(cx), int(cy)), int(radius), 255, -1)
            mean_int = cv2.mean(gray, mask=mask_circle)[0]
            
            area_score = min(1.0, area / (np.pi * max_radius ** 2))
            center_score = 1.0 - (dist_to_center / max_dist)
            dark_score = 1.0 - mean_int / 255
            
            total_score = (
                center_score * self.center_bias_weight +
                area_score * self.size_weight +
                dark_score * self.darkness_weight
            )
            
            candidates.append({
                'center': (float(cx), float(cy)),
                'radius': float(radius),
                'area': float(area),
                'circularity': 0.5,
                'mean_intensity': float(mean_int),
                'darkness_ratio': 1.0 - mean_int / 255,
                'center_score': float(center_score),
                'total_score': float(total_score),
                'threshold_used': 'connected_components'
            })
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[Dict], max_radius: float) -> Dict:
        """
        Selecciona el mejor candidato: mas grande y mas centrado.
        """
        if not candidates:
            return None
        
        for c in candidates:
            c['final_score'] = (
                c['total_score'] +
                c['area'] / (np.pi * max_radius ** 2) * 0.3 +
                (1.0 - c['mean_intensity'] / 255) * 0.2
            )
        
        best = max(candidates, key=lambda x: x['final_score'])
        
        return best
    
    def _refine_detection(self, gray: np.ndarray, candidate: Dict) -> Dict:
        """
        Refina la deteccion buscando el borde real de la pupila.
        """
        cx, cy = candidate['center']
        r = candidate['radius']
        
        h, w = gray.shape
        
        search_min = max(self.min_radius, int(r * 0.3))
        search_max = min(int(r * 1.8), w // 2, h // 2)
        
        if search_max <= search_min:
            return {
                'center': candidate['center'],
                'radius': r,
                'confidence': min(1.0, candidate['total_score']),
                'mean_intensity': candidate['mean_intensity']
            }
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        best_radius = r
        best_score = 0
        
        for radius in range(search_min, search_max, 2):
            angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
            inner_vals = []
            outer_vals = []
            
            for angle in angles:
                xi = int(cx + (radius - 3) * np.cos(angle))
                yi = int(cy + (radius - 3) * np.sin(angle))
                xo = int(cx + (radius + 3) * np.cos(angle))
                yo = int(cy + (radius + 3) * np.sin(angle))
                
                if 0 <= xi < w and 0 <= yi < h:
                    inner_vals.append(blurred[yi, xi])
                if 0 <= xo < w and 0 <= yo < h:
                    outer_vals.append(blurred[yo, xo])
            
            if inner_vals and outer_vals:
                inner_mean = np.mean(inner_vals)
                outer_mean = np.mean(outer_vals)
                
                if inner_mean < outer_mean:
                    gradient = outer_mean - inner_mean
                    score = gradient * (1.0 - abs(inner_mean) / 255)
                    if score > best_score:
                        best_score = score
                        best_radius = radius
        
        mask_final = np.zeros_like(gray)
        cv2.circle(mask_final, (int(cx), int(cy)), int(best_radius), 255, -1)
        mean_int = cv2.mean(gray, mask=mask_final)[0]
        
        confidence = min(1.0, candidate['total_score'])
        if best_score > 5:
            confidence = min(1.0, confidence * 1.1)
        
        return {
            'center': candidate['center'],
            'radius': float(best_radius),
            'confidence': float(confidence),
            'mean_intensity': float(mean_int)
        }
    
    def _fallback_detection(self, gray: np.ndarray, center_x: float, center_y: float,
                           max_radius: float) -> Optional[Dict]:
        """
        Fallback: busca la region mas oscura cercana al centro.
        """
        blurred = cv2.blur(gray, (15, 15))
        
        min_loc = cv2.minMaxLoc(blurred)[2]
        if min_loc is None:
            return None
        
        approx_cx, approx_cy = min_loc
        
        h, w = gray.shape
        
        for search_radius in [max_radius, max_radius*0.7, max_radius*0.5]:
            x1 = max(0, int(approx_cx - search_radius))
            y1 = max(0, int(approx_cy - search_radius))
            x2 = min(w, int(approx_cx + search_radius))
            y2 = min(h, int(approx_cy + search_radius))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            _, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                
                if area > 50:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        local_cx = M["m10"] / M["m00"]
                        local_cy = M["m01"] / M["m00"]
                        
                        cx = x1 + local_cx
                        cy = y1 + local_cy
                        
                        radius = np.sqrt(area / np.pi)
                        
                        mask = np.zeros_like(gray)
                        cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
                        mean_int = cv2.mean(gray, mask=mask)[0]
                        
                        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        
                        return {
                            'center': (float(cx), float(cy)),
                            'radius': float(radius),
                            'confidence': 0.4,
                            'mean_intensity': float(mean_int),
                            'method': 'darkcircle_fallback'
                        }
        
        radius = min(max_radius * 0.3, 50)
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(approx_cx), int(approx_cy)), int(radius), 255, -1)
        mean_int = cv2.mean(gray, mask=mask)[0]
        
        return {
            'center': (float(approx_cx), float(approx_cy)),
            'radius': float(radius),
            'confidence': 0.3,
            'mean_intensity': float(mean_int),
            'method': 'darkcircle_center_fallback'
        }


def detect_pupil_darkcircle(image: np.ndarray, **kwargs) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """
    Funcion de conveniencia para detectar pupila usando DarkCircle.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    detector = DarkCircleDetector(**kwargs)
    result = detector.detect(gray)
    
    if result:
        return result['center'], result['radius']
    return None, None


def draw_darkcircle_detection(image: np.ndarray, result: Dict, 
                               line_width: int = 2) -> np.ndarray:
    """
    Dibuja la deteccion de DarkCircle en la imagen.
    """
    out = image.copy()
    
    if result and result.get('radius'):
        cx, cy = result['center']
        r = result['radius']
        
        color = (0, 165, 255)
        cv2.circle(out, (int(cx), int(cy)), int(r), color, line_width)
        cv2.circle(out, (int(cx), int(cy)), max(2, line_width), (0, 255, 0), -1)
        
        cv2.putText(out, f"DARKCIRCLE r={r:.0f}", 
                   (int(cx) - 40, int(cy) - int(r) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return out
