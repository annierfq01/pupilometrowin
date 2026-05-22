"""
Pupillometry analysis and computation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Phase(Enum):
    """Video phases"""
    BASAL = "basal"
    ILLUMINATION = "illumination"
    RELAXATION = "relaxation"


@dataclass
class PupilMeasurement:
    """Single pupil measurement"""
    timestamp: float
    diameter_mm: float
    area_mm2: float
    velocity_mms: Optional[float] = None
    confidence: float = 0.0


@dataclass
class PupilometryResult:
    """Complete pupillometry analysis"""
    phase: Phase
    measurements: List[PupilMeasurement]
    mean_diameter: float
    min_diameter: float
    max_diameter: float
    mean_velocity: float
    latency: Optional[float] = None  # Time to start constriction
    max_constriction_velocity: float = 0.0
    redilation_velocity: float = 0.0


class PupillometryAnalyzer:
    """Analyze pupillometry data"""
    
    def __init__(self, pixels_per_mm: float = 1.0):
        self.pixels_per_mm = pixels_per_mm
    
    def compute_diameter_mm(self, radius_pixels: float) -> float:
        """Convert radius in pixels to diameter in millimeters"""
        diameter_pixels = radius_pixels * 2
        return diameter_pixels / self.pixels_per_mm
    
    def compute_area_mm2(self, radius_pixels: float) -> float:
        """Compute pupil area in mm²"""
        diameter_mm = self.compute_diameter_mm(radius_pixels)
        radius_mm = diameter_mm / 2
        return np.pi * radius_mm ** 2
    
    def compute_velocity(self, measurements: List[PupilMeasurement]) -> List[float]:
        """Compute velocity between consecutive measurements"""
        velocities = []
        for i in range(1, len(measurements)):
            dt = measurements[i].timestamp - measurements[i-1].timestamp
            if dt > 0:
                d_diameter = measurements[i].diameter_mm - measurements[i-1].diameter_mm
                velocity = d_diameter / dt  # mm/s
                velocities.append(velocity)
            else:
                velocities.append(0.0)
        return [0.0] + velocities  # First measurement has no velocity
    
    def analyze_phase(self, measurements: List[PupilMeasurement],
                     phase: Phase) -> PupilometryResult:
        """Analyze measurements for a specific phase"""
        
        if not measurements:
            return PupilometryResult(
                phase=phase,
                measurements=[],
                mean_diameter=0.0,
                min_diameter=0.0,
                max_diameter=0.0,
                mean_velocity=0.0
            )
        
        diameters = [m.diameter_mm for m in measurements]
        
        # Compute velocity
        velocities = self.compute_velocity(measurements)
        for i, m in enumerate(measurements):
            m.velocity_mms = velocities[i]
        
        mean_velocity = np.mean([v for v in velocities if v != 0]) if any(velocities) else 0.0
        
        return PupilometryResult(
            phase=phase,
            measurements=measurements,
            mean_diameter=np.mean(diameters),
            min_diameter=np.min(diameters),
            max_diameter=np.max(diameters),
            mean_velocity=mean_velocity,
            max_constriction_velocity=np.min(velocities) if velocities else 0.0,
            redilation_velocity=np.max(velocities) if velocities else 0.0
        )
    
    def compute_latency(self, basal_measurements: List[PupilMeasurement],
                       illumination_measurements: List[PupilMeasurement],
                       threshold_mm: float = 0.2) -> Optional[float]:
        """
        Compute latency (time to start constriction)
        threshold_mm: diameter reduction threshold to detect constriction start
        """
        if not basal_measurements or not illumination_measurements:
            return None
        
        basal_mean = np.mean([m.diameter_mm for m in basal_measurements])
        
        for m in illumination_measurements:
            if m.diameter_mm < (basal_mean - threshold_mm):
                return m.timestamp - illumination_measurements[0].timestamp
        
        return None
    
    def estimate_missing_frames(self, measurements: List[PupilMeasurement]) -> List[bool]:
        """
        Estimate which frames are missing/invalid
        Returns boolean list (True = estimated)
        """
        if len(measurements) < 2:
            return [False] * len(measurements)
        
        estimated = [False] * len(measurements)
        diameters = [m.diameter_mm for m in measurements]
        
        # Simple heuristic: detect outliers
        mean = np.mean(diameters)
        std = np.std(diameters)
        
        for i, d in enumerate(diameters):
            if abs(d - mean) > 3 * std:
                estimated[i] = True
        
        return estimated
    
    def format_report(self, results: Dict[Phase, PupilometryResult]) -> str:
        """Format analysis results as text report"""
        report = "=" * 60 + "\n"
        report += "PUPILLOMETRY ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for phase, result in results.items():
            report += f"PHASE: {phase.value.upper()}\n"
            report += "-" * 40 + "\n"
            report += f"Mean Diameter: {result.mean_diameter:.2f} mm\n"
            report += f"Min Diameter:  {result.min_diameter:.2f} mm\n"
            report += f"Max Diameter:  {result.max_diameter:.2f} mm\n"
            report += f"Mean Velocity: {result.mean_velocity:.2f} mm/s\n"
            
            if result.latency:
                report += f"Latency:       {result.latency:.3f} s\n"
            
            report += f"Max Constriction: {result.max_constriction_velocity:.2f} mm/s\n"
            report += f"Redilation:       {result.redilation_velocity:.2f} mm/s\n"
            report += "\n"
        
        return report
