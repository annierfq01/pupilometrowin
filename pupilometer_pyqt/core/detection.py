"""
Core detection algorithms for pupil and iris detection
"""

import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of pupil/iris detection"""
    pupil_center: Optional[Tuple[float, float]] = None
    pupil_radius: Optional[float] = None
    iris_center: Optional[Tuple[float, float]] = None
    iris_radius: Optional[float] = None
    confidence: float = 0.0
    method: str = ""


class DetectionMode(Enum):
    """Available detection algorithms"""
    STARBURST = "starburst"
    SWIRSKI = "swirski"
    EXCUSE = "excuse"
    CANNY = "canny"
    THRESHOLD = "threshold"
    DARKCIRCLE = "darkcircle"
    IA = "ia"


class PupilDetector:
    """Base class for pupil detection"""
    
    def __init__(self):
        self.mode = DetectionMode.THRESHOLD
        
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect pupil in image"""
        raise NotImplementedError
        
    def detect_threshold(self, image: np.ndarray, 
                        threshold: int = 50) -> DetectionResult:
        """Simple threshold-based detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return DetectionResult(confidence=0.0, method="threshold")
        
        # Find largest contour (likely the pupil)
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        
        return DetectionResult(
            pupil_center=(float(x), float(y)),
            pupil_radius=float(radius),
            confidence=0.8,
            method="threshold"
        )
    
    def detect_canny(self, image: np.ndarray,
                    threshold1: int = 50,
                    threshold2: int = 150) -> DetectionResult:
        """Canny edge detection-based method"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return DetectionResult(confidence=0.0, method="canny")
        
        # Find best circular contour
        best_circle = None
        best_score = 0
        
        for contour in contours:
            if len(contour) < 5:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 5:
                continue
            
            # Check circularity
            area = cv2.contourArea(contour)
            circle_area = np.pi * radius ** 2
            score = area / circle_area if circle_area > 0 else 0
            
            if score > best_score and 0.5 < score < 1.0:
                best_score = score
                best_circle = ((x, y), radius)
        
        if best_circle:
            (x, y), radius = best_circle
            return DetectionResult(
                pupil_center=(float(x), float(y)),
                pupil_radius=float(radius),
                confidence=best_score,
                method="canny"
            )
        
        return DetectionResult(confidence=0.0, method="canny")
    
    def detect_darkcircle(self, image: np.ndarray) -> DetectionResult:
        """Detect dark circles (pupil region)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Morphological operations to enhance dark regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Detect circles using Hough
        circles = cv2.HoughCircles(
            opening,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return largest circle
            x, y, r = circles[0][np.argmax(circles[0][:, 2])]
            return DetectionResult(
                pupil_center=(float(x), float(y)),
                pupil_radius=float(r),
                confidence=0.85,
                method="darkcircle"
            )
        
        return DetectionResult(confidence=0.0, method="darkcircle")


class IrisDetector:
    """Iris detection"""
    
    def detect_gradient(self, image: np.ndarray) -> DetectionResult:
        """Iris detection using gradient"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Find local maxima
        circles = cv2.HoughCircles(
            np.uint8(gradient),
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=200
        )
        
        if circles is not None:
            x, y, r = circles[0][0]
            return DetectionResult(
                iris_center=(float(x), float(y)),
                iris_radius=float(r),
                confidence=0.75,
                method="gradient"
            )
        
        return DetectionResult(confidence=0.0, method="gradient")


def draw_detections(image: np.ndarray, result: DetectionResult,
                   draw_pupil: bool = True,
                   draw_iris: bool = True) -> np.ndarray:
    """Draw detection results on image"""
    output = image.copy()
    
    if draw_pupil and result.pupil_center and result.pupil_radius:
        cv2.circle(output, 
                  tuple(map(int, result.pupil_center)),
                  int(result.pupil_radius),
                  (0, 255, 0), 2)
        cv2.circle(output,
                  tuple(map(int, result.pupil_center)),
                  3, (0, 255, 0), -1)
    
    if draw_iris and result.iris_center and result.iris_radius:
        cv2.circle(output,
                  tuple(map(int, result.iris_center)),
                  int(result.iris_radius),
                  (255, 0, 0), 2)
    
    return output
