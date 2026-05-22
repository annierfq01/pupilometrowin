"""
Calibration and measurement utilities
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Calibration data"""
    pixels_per_mm: float
    reference_points: List[Tuple[int, int]]
    reference_distance_mm: float
    confidence: float


class RulerCalibration:
    """Handle ruler calibration for measurements"""
    
    def __init__(self):
        self.calibration = None
        self.color_range = None
        self.reference_points = []
    
    def set_color_range(self, color_bgr: Tuple[int, int, int],
                       tolerance: int = 30) -> None:
        """
        Set color range for ruler detection
        color_bgr: BGR color tuple
        tolerance: +/- range for each channel
        """
        lower = np.array([
            max(0, color_bgr[0] - tolerance),
            max(0, color_bgr[1] - tolerance),
            max(0, color_bgr[2] - tolerance)
        ])
        upper = np.array([
            min(255, color_bgr[0] + tolerance),
            min(255, color_bgr[1] + tolerance),
            min(255, color_bgr[2] + tolerance)
        ])
        self.color_range = (lower, upper)
    
    def detect_ruler_automatic(self, image: np.ndarray,
                              color_bgr: Tuple[int, int, int] = (0, 192, 255)
                              ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Automatically detect ruler (fluorescent pink/magenta)
        Returns mask and detected points
        """
        self.set_color_range(color_bgr, tolerance=30)
        lower, upper = self.color_range
        
        mask = cv2.inRange(image, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
        
        return mask, sorted(points)
    
    def calibrate_from_points(self, point1: Tuple[int, int],
                             point2: Tuple[int, int],
                             distance_mm: float) -> CalibrationResult:
        """
        Calibrate using two reference points
        point1, point2: pixel coordinates
        distance_mm: real-world distance in millimeters
        """
        # Calculate pixel distance
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        pixel_distance = np.sqrt(dx**2 + dy**2)
        
        if pixel_distance == 0:
            return CalibrationResult(0, [point1, point2], distance_mm, 0.0)
        
        pixels_per_mm = pixel_distance / distance_mm
        
        result = CalibrationResult(
            pixels_per_mm=pixels_per_mm,
            reference_points=[point1, point2],
            reference_distance_mm=distance_mm,
            confidence=0.95
        )
        
        self.calibration = result
        return result
    
    def pixel_to_mm(self, pixels: float) -> float:
        """Convert pixels to millimeters"""
        if not self.calibration or self.calibration.pixels_per_mm == 0:
            return 0.0
        return pixels / self.calibration.pixels_per_mm
    
    def mm_to_pixel(self, mm: float) -> float:
        """Convert millimeters to pixels"""
        if not self.calibration or self.calibration.pixels_per_mm == 0:
            return 0.0
        return mm * self.calibration.pixels_per_mm
    
    def draw_calibration(self, image: np.ndarray) -> np.ndarray:
        """Draw calibration reference on image"""
        if not self.calibration:
            return image
        
        output = image.copy()
        p1, p2 = self.calibration.reference_points
        
        # Draw line between points
        cv2.line(output, p1, p2, (0, 255, 255), 2)
        
        # Draw circles at points
        cv2.circle(output, p1, 5, (0, 255, 0), -1)
        cv2.circle(output, p2, 5, (0, 255, 0), -1)
        
        # Draw text
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        text = f"{self.calibration.reference_distance_mm:.1f}mm"
        cv2.putText(output, text, (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return output
