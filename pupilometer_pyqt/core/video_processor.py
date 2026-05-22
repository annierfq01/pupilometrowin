"""
Video processing and frame management
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class VideoInfo:
    """Video metadata"""
    filename: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float  # seconds
    

class VideoProcessor:
    """Handle video loading and processing"""
    
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.cap = None
        self.info = None
        self.frames = []
        self.current_frame_idx = 0
        
    def open(self) -> bool:
        """Open video file"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.info = VideoInfo(
            filename=Path(self.video_path).name,
            fps=self.cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            duration=0
        )
        
        if self.info.fps > 0:
            self.info.duration = self.info.frame_count / self.info.fps
        
        return True
    
    def close(self):
        """Close video file"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read specific frame"""
        if not self.cap:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame_idx = frame_idx
            return frame
        return None
    
    def read_all_frames(self) -> List[np.ndarray]:
        """Load all frames into memory"""
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        
        self.frames = frames
        return frames
    
    def read_frames_range(self, start_idx: int, end_idx: int) -> List[np.ndarray]:
        """Read frames in range"""
        frames = []
        for i in range(start_idx, min(end_idx, self.info.frame_count)):
            frame = self.read_frame(i)
            if frame is not None:
                frames.append(frame)
        return frames
    
    def get_frame_timestamp(self, frame_idx: int) -> float:
        """Get timestamp of frame in seconds"""
        if self.info and self.info.fps > 0:
            return frame_idx / self.info.fps
        return 0.0
    
    def detect_illumination_changes(self, threshold: float = 10.0) -> List[Tuple[int, str]]:
        """
        Detect illumination changes (flash detection)
        Returns list of (frame_index, change_type) tuples
        change_type: "on" or "off"
        """
        if not self.frames:
            self.read_all_frames()
        
        changes = []
        brightness_history = []
        
        for idx, frame in enumerate(self.frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_history.append(brightness)
        
        # Detect significant changes
        for i in range(1, len(brightness_history)):
            change = brightness_history[i] - brightness_history[i-1]
            if abs(change) > threshold:
                change_type = "on" if change > 0 else "off"
                changes.append((i, change_type))
        
        return changes
    
    def reduce_frames(self, frame_indices: List[int]) -> List[np.ndarray]:
        """
        Get reduced frame set
        frame_indices: indices of frames to extract
        """
        reduced = []
        for idx in frame_indices:
            if idx < self.info.frame_count:
                frame = self.read_frame(idx)
                if frame is not None:
                    reduced.append(frame)
        return reduced
    
    def crop_roi(self, frame: np.ndarray, x: int, y: int, 
                width: int, height: int) -> np.ndarray:
        """Crop region of interest from frame"""
        return frame[y:y+height, x:x+width]
    
    def __del__(self):
        self.close()
