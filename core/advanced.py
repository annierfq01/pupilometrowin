"""
advanced.py - Advanced Detection Algorithms
==========================================
Algoritmos avanzados de deteccion de pupila e iris:
- starburst
- swirski
- excuse
- canny
- threshold
- darkcircle (NUEVO: circulo oscuro mas grande y centrado)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum


class DetectionMode(Enum):
    """Modos de deteccion de iris."""
    GRADIENT_BASED = "gradient"
    CIRCULAR_HOUGH = "hough"
    ELLIPSE_FIT = "ellipse"
    INTENSITY_PROFILE = "profile"


@dataclass
class DetectionResult:
    """Resultado de deteccion."""
    pupil_center: Optional[Tuple[float, float]]
    pupil_diameter: Optional[float]
    pupil_radius: Optional[float]
    pupil_confidence: float
    iris_center: Optional[Tuple[float, float]]
    iris_diameter: Optional[float]
    iris_radius: Optional[float]
    iris_confidence: float
    is_blink: bool
    image_quality: float
    method: str = "unknown"


class AdvancedEyeDetector:
    """
    Detector de ojo con multiples algoritmos de deteccion de pupila.
    """
    
    def __init__(self, 
                 pupil_algorithm: str = "starburst",
                 iris_mode: DetectionMode = DetectionMode.GRADIENT_BASED,
                 expected_pupil_range: Tuple[float, float] = (0.1, 0.7),
                 expected_iris_range: Tuple[float, float] = (0.3, 0.9)):
        """
        Args:
            pupil_algorithm: Algoritmo a usar ('starburst', 'swirski', 'excuse', 'canny', 'threshold', 'darkcircle')
            iris_mode: Modo de deteccion de iris
            expected_pupil_range: Rango esperado de tamano de pupila (relativo a imagen)
            expected_iris_range: Rango esperado de tamano de iris (relativo a imagen)
        """
        self.pupil_algorithm = pupil_algorithm
        self.iris_mode = iris_mode
        self.expected_pupil_range = expected_pupil_range
        self.expected_iris_range = expected_iris_range
    
    def assess_image_quality(self, gray_roi: np.ndarray) -> float:
        """Evalua la calidad de la imagen basada en contraste y nitidez."""
        if gray_roi.size == 0:
            return 0.0
        
        contrast = np.std(gray_roi) / 128.0
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000.0
        sharpness = min(sharpness, 1.0)
        mean_brightness = np.mean(gray_roi) / 255.0
        uniformity = 1.0 - abs(mean_brightness - 0.5) * 2
        
        quality = (contrast * 0.4 + sharpness * 0.4 + uniformity * 0.2)
        return min(max(quality, 0.0), 1.0)
    
    def detect_pupil(self, gray: np.ndarray) -> Optional[Dict]:
        """Detecta la pupila usando el algoritmo configurado."""
        h, w = gray.shape
        
        if self.assess_image_quality(gray) < 0.1:
            return None
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        algorithms = {
            "starburst": self._starburst_pupil,
            "swirski": self._swirski_pupil,
            "excuse": self._excuse_pupil,
            "canny": self._canny_pupil,
            "threshold": self._threshold_pupil,
            "darkcircle": self._darkcircle_pupil,
        }
        
        alg_func = algorithms.get(self.pupil_algorithm, self._starburst_pupil)
        return alg_func(blurred, w, h)
    
    def _darkcircle_pupil(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """
        Algoritmo DarkCircle RELAJADO - busca circulos oscuros con alta tolerancia.
        
        Busca elementos que puedan ser la pupila con:
        1. Multiple umbralizacion para tolerate iluminaciones
        2. Baja circularidad (formas elipticas aceptadas)
        3. Prioriza el mas grande y centrado
        """
        center_x, center_y = w / 2, h / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        all_candidates = []
        
        thresholds = [100, 80, 60, 120]
        
        for thresh in thresholds:
            candidates = self._find_darkcircle_candidates(
                gray, thresh, center_x, center_y, max_dist
            )
            all_candidates.extend(candidates)
        
        if not all_candidates:
            return self._darkcircle_fallback(gray, w, h)
        
        for c in all_candidates:
            c['final_score'] = (
                c['confidence'] +
                c['area'] / (np.pi * (max(w, h) * 0.4) ** 2) * 0.3
            )
        
        best = max(all_candidates, key=lambda x: x['final_score'])
        
        return {
            'center': best['center'],
            'axes': best.get('axes', (best['diameter'], best['diameter'])),
            'angle': 0,
            'diameter': best['diameter'],
            'area': best['area'],
            'confidence': min(1.0, best['confidence']),
            'contour': best.get('contour')
        }
    
    def _find_darkcircle_candidates(self, gray: np.ndarray, threshold: int,
                                    center_x: float, center_y: float,
                                    max_dist: float) -> List[Dict]:
        """
        Encuentra candidatos oscuros usando un umbral especifico.
        """
        _, dark_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            if circularity < 0.15:
                continue
            
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    center, axes, angle = ellipse
                except:
                    center, _ = cv2.minEnclosingCircle(cnt)
                    axes = (0, 0)
                    angle = 0
            else:
                center, _ = cv2.minEnclosingCircle(cnt)
                axes = (0, 0)
                angle = 0
            
            cx, cy = center
            diameter = np.sqrt(axes[0] * axes[1]) if max(axes) > 0 else np.sqrt(4 * area / np.pi)
            
            rel_diameter = diameter / max(len(gray[0]), len(gray))
            if not (0.05 <= rel_diameter <= 0.5):
                continue
            
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            mask_circle = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask_circle, (int(cx), int(cy)), int(diameter/2), 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask_circle)[0]
            
            center_score = 1.0 - (dist_to_center / max_dist)
            size_score = rel_diameter
            darkness_score = 1.0 - (mean_intensity / 255.0)
            
            total_score = (center_score * 0.4 + size_score * 0.4 + darkness_score * 0.2)
            
            candidates.append({
                'center': center,
                'axes': axes,
                'angle': angle,
                'diameter': diameter,
                'area': area,
                'confidence': total_score,
                'contour': cnt,
                'mean_intensity': mean_intensity
            })
        
        return candidates
    
    def _darkcircle_fallback(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """
        Fallback para DarkCircle: usa connected components y busca region oscura
        """
        blurred = cv2.blur(gray, (7, 7))
        
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        best_candidate = None
        best_score = -1
        center_x, center_y = w / 2, h / 2
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:
                continue
            
            cx, cy = centroids[i]
            
            diameter = np.sqrt(area / np.pi)
            
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            mask_circle = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask_circle, (int(cx), int(cy)), int(diameter), 255, -1)
            mean_int = cv2.mean(gray, mask=mask_circle)[0]
            
            score = (1.0 - dist / max(w, h) / 2) * area / (np.pi * (min(w, h) * 0.4) ** 2)
            
            if score > best_score:
                best_score = score
                
                best_candidate = {
                    'center': (float(cx), float(cy)),
                    'axes': (diameter, diameter),
                    'angle': 0,
                    'diameter': diameter,
                    'area': area,
                    'confidence': min(1.0, score),
                    'contour': None
                }
        
        if best_candidate is None:
            min_loc = cv2.minMaxLoc(blurred)[2]
            if min_loc:
                cx, cy = min_loc
                diameter = min(w, h) * 0.15
                best_candidate = {
                    'center': (float(cx), float(cy)),
                    'axes': (diameter, diameter),
                    'angle': 0,
                    'diameter': diameter,
                    'area': np.pi * diameter ** 2,
                    'confidence': 0.3,
                    'contour': None
                }
        
        return best_candidate
    
    def _starburst_pupil(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """Algoritmo Starburst para deteccion de pupila."""
        _, dark_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, dark_kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_thresh, connectivity=8)
        
        best_component = None
        best_score = -1
        center_x, center_y = w // 2, h // 2
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            min_expected = self.expected_pupil_range[0] * w * h
            max_expected = self.expected_pupil_range[1] * w * h
            
            if area < min_expected or area > max_expected:
                continue
            
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            score = area / (1 + dist_to_center / 10)
            
            if score > best_score:
                best_score = score
                best_component = i
        
        if best_component is None:
            return None
        
        pupil_mask = (labels == best_component).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        pupil_contour = max(contours, key=cv2.contourArea)
        if len(pupil_contour) < 5:
            return None
        
        ellipse = cv2.fitEllipse(pupil_contour)
        center, axes, angle = ellipse
        area = cv2.contourArea(pupil_contour)
        
        perimeter = cv2.arcLength(pupil_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        diameter = np.sqrt(4 * area / np.pi)
        rel_diameter = diameter / max(w, h)
        
        if not (self.expected_pupil_range[0] <= rel_diameter <= self.expected_pupil_range[1]):
            return None
        
        return {
            'center': center,
            'axes': axes,
            'angle': angle,
            'diameter': diameter,
            'area': area,
            'confidence': circularity,
            'contour': pupil_contour
        }
    
    def _swirski_pupil(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """Algoritmo Swirski para deteccion de pupila."""
        inverted = 255 - gray
        _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ellipse = None
        best_score = -1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            if len(cnt) < 5:
                continue
            
            try:
                ellipse = cv2.fitEllipse(cnt)
                center, axes, angle = ellipse
                
                if min(axes) < 5 or max(axes) > max(w, h):
                    continue
                
                diameter = np.sqrt(axes[0] * axes[1])
                rel_diameter = diameter / max(w, h)
                
                if not (self.expected_pupil_range[0] <= rel_diameter <= self.expected_pupil_range[1]):
                    continue
                
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                dist_to_center = np.sqrt((center[0] - w/2)**2 + (center[1] - h/2)**2)
                score = circularity * 0.7 + (1 - dist_to_center/max(w,h)) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_ellipse = {
                        'center': center,
                        'axes': axes,
                        'angle': angle,
                        'diameter': diameter,
                        'area': area,
                        'confidence': circularity,
                        'contour': cnt
                    }
            except cv2.error:
                continue
        
        return best_ellipse
    
    def _excuse_pupil(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """Algoritmo EXCUSE para deteccion de pupila con multiples umbrales."""
        thresholds = np.linspace(20, 100, 10)
        candidates = []
        
        for thresh_val in thresholds:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                
                if len(cnt) < 5:
                    continue
                
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    center, axes, angle = ellipse
                    
                    if min(axes) < 5 or max(axes) > max(w, h):
                        continue
                    
                    diameter = np.sqrt(axes[0] * axes[1])
                    rel_diameter = diameter / max(w, h)
                    
                    if not (self.expected_pupil_range[0] <= rel_diameter <= self.expected_pupil_range[1]):
                        continue
                    
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    
                    dist_to_center = np.sqrt((center[0] - w/2)**2 + (center[1] - h/2)**2)
                    score = circularity * 0.6 + (1 - dist_to_center/max(w,h)) * 0.4
                    
                    candidates.append({
                        'center': center,
                        'axes': axes,
                        'angle': angle,
                        'diameter': diameter,
                        'area': area,
                        'confidence': circularity,
                        'contour': cnt,
                        'score': score
                    })
                except cv2.error:
                    continue
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda x: x['score'])
        del best['score']
        return best
    
    def _canny_pupil(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """Algoritmo Canny para deteccion de pupila."""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        all_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                all_points.extend(cnt.reshape(-1, 2))
        
        if len(all_points) < 20:
            return None
        
        all_points = np.array(all_points)
        
        try:
            ellipse = cv2.fitEllipse(all_points.reshape(-1, 1, 2))
            center, axes, angle = ellipse
            
            if min(axes) < 5 or max(axes) > max(w, h):
                return None
            
            diameter = np.sqrt(axes[0] * axes[1])
            rel_diameter = diameter / max(w, h)
            
            if not (self.expected_pupil_range[0] <= rel_diameter <= self.expected_pupil_range[1]):
                return None
            
            area = np.pi * (axes[0]/2) * (axes[1]/2)
            
            return {
                'center': center,
                'axes': axes,
                'angle': angle,
                'diameter': diameter,
                'area': area,
                'confidence': 0.6,
                'contour': None
            }
        except cv2.error:
            return None
    
    def _threshold_pupil(self, gray: np.ndarray, w: int, h: int) -> Optional[Dict]:
        """Algoritmo de umbral adaptativo para deteccion de pupila."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center_x, center_y = w // 2, h // 2
        best_contour = None
        best_score = -1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            perimeter = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            score = circ * 0.5 + (1 - dist/max(w,h)) * 0.5
            
            if score > best_score:
                best_score = score
                best_contour = cnt
        
        if best_contour is None or len(best_contour) < 5:
            return None
        
        ellipse = cv2.fitEllipse(best_contour)
        center, axes, angle = ellipse
        area = cv2.contourArea(best_contour)
        diameter = np.sqrt(4 * area / np.pi)
        
        rel_diameter = diameter / max(w, h)
        if not (self.expected_pupil_range[0] <= rel_diameter <= self.expected_pupil_range[1]):
            return None
        
        return {
            'center': center,
            'axes': axes,
            'angle': angle,
            'diameter': diameter,
            'area': area,
            'confidence': best_score,
            'contour': best_contour
        }
    
    def detect_iris(self, gray: np.ndarray, pupil_data: Optional[Dict]) -> Optional[Dict]:
        """Detecta el iris usando el modo configurado."""
        if self.iris_mode == DetectionMode.GRADIENT_BASED:
            return self._gradient_iris(gray, pupil_data)
        elif self.iris_mode == DetectionMode.CIRCULAR_HOUGH:
            return self._hough_iris(gray, pupil_data)
        elif self.iris_mode == DetectionMode.INTENSITY_PROFILE:
            return self._profile_iris(gray, pupil_data)
        else:
            return self._ellipse_iris(gray, pupil_data)
    
    def _gradient_iris(self, gray: np.ndarray, pupil_data: Optional[Dict]) -> Optional[Dict]:
        """Deteccion de iris por gradiente."""
        h, w = gray.shape
        
        if pupil_data:
            seed_x, seed_y = int(pupil_data['center'][0]), int(pupil_data['center'][1])
            max_radius = int(pupil_data['diameter'] * 1.5)
        else:
            seed_x, seed_y = w // 2, h // 2
            max_radius = min(w, h) // 2
        
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        border_points = []
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            
            samples = []
            for r in range(5, max_radius):
                x = int(seed_x + r * dx)
                y = int(seed_y + r * dy)
                
                if 0 <= x < w and 0 <= y < h:
                    samples.append((r, gray[y, x]))
            
            if len(samples) < 10:
                continue
            
            intensities = np.array([s[1] for s in samples])
            gradients = np.gradient(intensities)
            
            if len(gradients) > 5:
                valid_range = slice(5, min(len(gradients)-2, len(gradients)))
                max_grad_idx = np.argmax(np.abs(gradients[valid_range])) + 5
                
                r_iris = samples[max_grad_idx][0]
                x_iris = int(seed_x + r_iris * dx)
                y_iris = int(seed_y + r_iris * dy)
                border_points.append([x_iris, y_iris])
        
        if len(border_points) < 8:
            return None
        
        border_points = np.array(border_points)
        
        A = np.column_stack([2*border_points[:, 0], 2*border_points[:, 1], 
                            np.ones(len(border_points))])
        b = border_points[:, 0]**2 + border_points[:, 1]**2
        
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            center_x, center_y = sol[0], sol[1]
            radius = np.sqrt(sol[2] + center_x**2 + center_y**2)
            
            rel_radius = radius / max(w, h)
            if not (self.expected_iris_range[0]/2 <= rel_radius <= self.expected_iris_range[1]/2):
                return None
            
            distances = np.sqrt((border_points[:, 0] - center_x)**2 + 
                              (border_points[:, 1] - center_y)**2)
            std_distance = np.std(distances)
            confidence = max(0, 1 - std_distance / radius)
            
            return {
                'center': (float(center_x), float(center_y)),
                'radius': float(radius),
                'diameter': float(radius * 2),
                'confidence': float(confidence),
                'border_points': border_points.tolist()
            }
        except np.linalg.LinAlgError:
            return None
    
    def _hough_iris(self, gray: np.ndarray, pupil_data: Optional[Dict]) -> Optional[Dict]:
        """Deteccion de iris usando Hough Circle Transform."""
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        if pupil_data:
            min_r = int(pupil_data['diameter'] / 2 * 1.2)
            max_r = int(pupil_data['diameter'] / 2 * 2.0)
        else:
            min_r = int(min(gray.shape) * 0.15)
            max_r = int(min(gray.shape) * 0.45)
        
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30,
                                   minRadius=min_r, maxRadius=max_r)
        
        if circles is None:
            return None
        
        best_circle = None
        best_score = -1
        h, w = gray.shape
        
        for circle in circles[0, :]:
            cx, cy, r = circle
            
            dist_to_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            score = (1 - dist_to_center/max(w,h)) * 0.5 + 0.5
            
            if pupil_data:
                pupil_dist = np.sqrt((cx - pupil_data['center'][0])**2 + 
                                    (cy - pupil_data['center'][1])**2)
                if pupil_dist < r * 0.3:
                    score += 0.3
            
            if score > best_score:
                best_score = score
                best_circle = circle
        
        if best_circle is None:
            return None
        
        cx, cy, r = best_circle
        
        return {
            'center': (float(cx), float(cy)),
            'radius': float(r),
            'diameter': float(r * 2),
            'confidence': float(best_score),
            'border_points': None
        }
    
    def _profile_iris(self, gray: np.ndarray, pupil_data: Optional[Dict]) -> Optional[Dict]:
        """Deteccion de iris por perfil de intensidad."""
        h, w = gray.shape
        
        if pupil_data:
            cx, cy = int(pupil_data['center'][0]), int(pupil_data['center'][1])
        else:
            cx, cy = w // 2, h // 2
        
        max_r = min(w, h) // 2 - 5
        profile = []
        
        for r in range(5, max_r):
            angles = np.linspace(0, 2*np.pi, 60, endpoint=False)
            intensities = []
            
            for angle in angles:
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                
                if 0 <= x < w and 0 <= y < h:
                    intensities.append(gray[y, x])
            
            if intensities:
                profile.append((r, np.mean(intensities), np.std(intensities)))
        
        if len(profile) < 10:
            return None
        
        profile_array = np.array([p[1] for p in profile])
        first_deriv = np.gradient(profile_array)
        
        search_start = int(len(profile) * 0.2)
        search_end = int(len(profile) * 0.8)
        
        if search_start >= search_end:
            return None
        
        iris_idx = search_start + np.argmin(np.abs(first_deriv[search_start:search_end]))
        iris_radius = profile[iris_idx][0]
        
        return {
            'center': (float(cx), float(cy)),
            'radius': float(iris_radius),
            'diameter': float(iris_radius * 2),
            'confidence': 0.7,
            'border_points': None
        }
    
    def _ellipse_iris(self, gray: np.ndarray, pupil_data: Optional[Dict]) -> Optional[Dict]:
        """Deteccion de iris por ajuste de elipse."""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        
        _, edge_thresh = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge_thresh = cv2.morphologyEx(edge_thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edge_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        h, w = gray.shape
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            if len(cnt) < 5:
                continue
            
            try:
                ellipse = cv2.fitEllipse(cnt)
                center, axes, angle = ellipse
                
                diameter = np.sqrt(axes[0] * axes[1])
                rel_diameter = diameter / max(w, h)
                
                if not (self.expected_iris_range[0] <= rel_diameter <= self.expected_iris_range[1]):
                    continue
                
                score = 0.5
                if pupil_data:
                    dist_to_pupil = np.sqrt((center[0] - pupil_data['center'][0])**2 +
                                           (center[1] - pupil_data['center'][1])**2)
                    if dist_to_pupil < diameter * 0.2:
                        score += 0.3
                    if diameter > pupil_data['diameter'] * 1.5:
                        score += 0.2
                
                candidates.append({
                    'center': center,
                    'axes': axes,
                    'angle': angle,
                    'diameter': diameter,
                    'confidence': score
                })
            except cv2.error:
                continue
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda x: x['confidence'])
        return {
            'center': best['center'],
            'radius': best['diameter'] / 2,
            'diameter': best['diameter'],
            'confidence': best['confidence'],
            'border_points': None
        }
    
    def is_blink(self, gray_roi: np.ndarray, pupil_data: Optional[Dict]) -> bool:
        """Determina si hay un parpadeo en el frame actual."""
        if gray_roi.size == 0:
            return True
        
        if pupil_data is None:
            std_dev = np.std(gray_roi)
            return std_dev < 15
        
        axes = pupil_data['axes']
        aspect_ratio = min(axes) / max(axes) if max(axes) > 0 else 0
        
        if aspect_ratio < 0.3:
            return True
        
        return False
    
    def detect(self, image) -> DetectionResult:
        """Ejecuta la deteccion completa de pupila e iris."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        quality = self.assess_image_quality(gray)
        
        pupil_data = self.detect_pupil(gray)
        
        blink = self.is_blink(gray, pupil_data)
        
        iris_data = None
        if not blink and pupil_data:
            iris_data = self.detect_iris(gray, pupil_data)
        
        return DetectionResult(
            pupil_center=pupil_data['center'] if pupil_data else None,
            pupil_diameter=pupil_data['diameter'] if pupil_data else None,
            pupil_radius=int(pupil_data['diameter'] / 2) if pupil_data and pupil_data.get('diameter') else None,
            pupil_confidence=pupil_data['confidence'] if pupil_data else 0.0,
            iris_center=iris_data['center'] if iris_data else None,
            iris_diameter=iris_data['diameter'] if iris_data else None,
            iris_radius=int(iris_data['radius']) if iris_data and iris_data.get('radius') else None,
            iris_confidence=iris_data['confidence'] if iris_data else 0.0,
            is_blink=blink,
            image_quality=quality,
            method=self.pupil_algorithm
        )


def draw_detections(image, pupil_center, pupil_radius, iris_center, iris_radius, line_width=2):
    """Dibuja las detecciones en la imagen."""
    out = image.copy()
    
    if iris_center and iris_radius and iris_radius > 0:
        cv2.circle(out, (int(iris_center[0]), int(iris_center[1])), int(iris_radius), (255, 0, 0), line_width)
    
    if pupil_center and pupil_radius and pupil_radius > 0:
        cv2.circle(out, (int(pupil_center[0]), int(pupil_center[1])), int(pupil_radius), (0, 0, 255), line_width)
        cv2.circle(out, (int(pupil_center[0]), int(pupil_center[1])), max(2, line_width), (0, 255, 0), -1)
    
    return out
