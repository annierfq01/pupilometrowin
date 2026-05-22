"""
Flash Detection Module - Pupilometer
=====================================
Automatic flash/illumination detection for video analysis.
Detects light ON/OFF events using brightness analysis.
"""

import cv2
import numpy as np
from collections import deque


class FlashDetector:
    """
    Detects illumination changes in video frames using brightness analysis.
    States: normal, high (light ON), low (light OFF)
    """
    
    def __init__(self, calibration_frames=5, change_threshold=5.0, 
                 smoothing_window=4, history_max=300):
        """
        Initialize flash detector.
        
        Args:
            calibration_frames: Number of initial frames for baseline calibration
            change_threshold: Minimum brightness difference for event detection
            smoothing_window: Number of frames for smoothing
            history_max: Maximum number of brightness values to keep in history
        """
        self.calibration_frames = calibration_frames
        self.change_threshold = change_threshold
        self.smoothing_window = smoothing_window
        self.history_max = history_max
        
        self.reset()
    
    def reset(self):
        """Reset detector state for new video."""
        self.calibrating = True
        self.frames_cal = []
        self.brightness_base = None
        
        self.smoothing = deque(maxlen=self.smoothing_window)
        self.history = deque(maxlen=self.history_max)
        
        self.current_state = "normal"
        self.event_log = deque(maxlen=50)
        
        self.flash_on_time = None
        self.flash_off_time = None
        
        self.calibration_progress = 0.0
    
    def _calculate_brightness(self, frame):
        """Calculate frame brightness using HSV value channel mean."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))
    
    def process_frame(self, frame, timestamp_ms=None):
        """
        Process a single frame and detect illumination changes.
        
        Args:
            frame: BGR image frame
            timestamp_ms: Frame timestamp in milliseconds (optional)
            
        Returns:
            dict with keys:
                - state: current state ('normal', 'high', 'low')
                - brightness: current smoothed brightness value
                - delta: difference from baseline
                - is_calibrating: whether still in calibration phase
                - calibration_progress: calibration progress (0-1)
                - event: detected event type or None
        """
        result = {
            'state': self.current_state,
            'brightness': None,
            'delta': 0.0,
            'is_calibrating': self.calibrating,
            'calibration_progress': self.calibration_progress,
            'event': None
        }
        
        b_raw = self._calculate_brightness(frame)
        
        if self.calibrating:
            self.frames_cal.append(b_raw)
            self.calibration_progress = len(self.frames_cal) / self.calibration_frames
            
            result['brightness'] = b_raw
            result['calibration_progress'] = self.calibration_progress
            
            if len(self.frames_cal) >= self.calibration_frames:
                self.brightness_base = float(np.mean(self.frames_cal))
                self.calibrating = False
                self.current_state = "normal"
                result['state'] = "normal"
        else:
            self.smoothing.append(b_raw)
            b = float(np.mean(self.smoothing))
            self.history.append(b)
            
            delta = b - self.brightness_base
            result['brightness'] = b
            result['delta'] = delta
            
            new_state = self.current_state
            event = None
            
            if self.current_state == "normal":
                if delta >= self.change_threshold:
                    new_state = "high"
                    event = ("ON", timestamp_ms)
                elif delta <= -self.change_threshold:
                    new_state = "low"
                    event = ("OFF", timestamp_ms)
            
            elif self.current_state == "high":
                if delta < self.change_threshold:
                    if delta <= -self.change_threshold:
                        new_state = "low"
                        event = ("OFF", timestamp_ms)
                        if self.flash_off_time is None and timestamp_ms is not None:
                            self.flash_off_time = timestamp_ms
                    else:
                        new_state = "normal"
                        event = ("NORM", timestamp_ms)
            
            elif self.current_state == "low":
                if delta > -self.change_threshold:
                    if delta >= self.change_threshold:
                        new_state = "high"
                        event = ("ON", timestamp_ms)
                    else:
                        new_state = "normal"
                        event = ("NORM", timestamp_ms)
            
            if event:
                self.event_log.append(event)
                result['event'] = event[0]
                
                if event[0] == "ON" and self.flash_on_time is None and timestamp_ms is not None:
                    self.flash_on_time = timestamp_ms
                elif event[0] == "OFF" and self.flash_off_time is None and timestamp_ms is not None:
                    self.flash_off_time = timestamp_ms
            
            self.current_state = new_state
            result['state'] = new_state
        
        return result


def detect_flash_in_video(video_path, calibration_frames=5, change_threshold=5.0,
                         smoothing_window=4, progress_callback=None):
    """
    Process entire video and detect flash events.
    
    Args:
        video_path: Path to video file
        calibration_frames: Number of frames for calibration
        change_threshold: Brightness change threshold
        smoothing_window: Smoothing window size
        progress_callback: Optional callback(current_frame, total_frames)
        
    Returns:
        dict with keys:
            - detector: FlashDetector instance with final state
            - flash_on_time_ms: Flash ON timestamp in ms (first detection)
            - flash_off_time_ms: Flash OFF timestamp in ms (first detection)
            - events: List of (event_type, timestamp_ms)
            - brightness_history: List of brightness values
            - total_frames: Total frames processed
    """
    detector = FlashDetector(
        calibration_frames=calibration_frames,
        change_threshold=change_threshold,
        smoothing_window=smoothing_window
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    events = []
    brightness_history = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = (frame_idx / fps) * 1000 if fps > 0 else 0
        
        result = detector.process_frame(frame, timestamp_ms)
        
        if result['event']:
            events.append((result['event'], timestamp_ms))
        
        if result['brightness'] is not None:
            brightness_history.append(result['brightness'])
        
        frame_idx += 1
        
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx, total_frames)
    
    cap.release()
    
    return {
        'detector': detector,
        'flash_on_time_ms': detector.flash_on_time,
        'flash_off_time_ms': detector.flash_off_time,
        'events': events,
        'brightness_history': brightness_history,
        'total_frames': frame_idx,
        'baseline_brightness': detector.brightness_base
    }


def trim_video_by_flash(video_path, output_path, flash_on_ms, flash_off_ms,
                       trim_before_ms=100, trim_after_ms=100):
    """
    Trim video based on flash detection times.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        flash_on_ms: Flash ON time in milliseconds
        flash_off_ms: Flash OFF time in milliseconds
        trim_before_ms: Extra time to keep before flash (phase 1)
        trim_after_ms: Extra time to keep after flash (phase 3)
        
    Returns:
        dict with trim information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    start_time_ms = max(0, flash_on_ms - trim_before_ms)
    end_time_ms = flash_off_ms + trim_after_ms if flash_off_ms else flash_on_ms + trim_after_ms
    
    start_frame = int((start_time_ms / 1000) * fps)
    end_frame = int((end_time_ms / 1000) * fps)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = start_frame + frames_written
        if current_frame > end_frame:
            break
        
        out.write(frame)
        frames_written += 1
    
    cap.release()
    out.release()
    
    return {
        'start_time_ms': start_time_ms,
        'end_time_ms': end_time_ms,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'frames_written': frames_written,
        'output_path': output_path
    }


def should_reduce_fps(original_fps, target_fps, quality_threshold=480):
    """
    Determine if FPS should be reduced based on video quality.
    
    Args:
        original_fps: Original video FPS
        target_fps: Target analysis FPS
        quality_threshold: Minimum resolution threshold (pixels)
        
    Returns:
        tuple: (should_reduce, recommended_fps)
    """
    if target_fps >= original_fps:
        return False, original_fps
    
    recommended = min(original_fps, target_fps)
    
    return True, recommended


__all__ = [
    'FlashDetector',
    'detect_flash_in_video',
    'trim_video_by_flash',
    'should_reduce_fps'
]
