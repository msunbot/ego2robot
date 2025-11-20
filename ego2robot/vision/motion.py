"""
Motion analysis for clip quality scoring.
"""
import numpy as np
import cv2

class MotionScorer:
    def __init__(self, config: dict):
        self.config = config
        
    def score_clip(self, frames: np.ndarray) -> float:
        """
        Compute motion score for a clip.
        Returns value between 0 and 1.
        """
        if len(frames) < 2:
            return 0.0
        
        # Downsample for speed
        frames_small = [cv2.resize(f, (160, 90)) for f in frames[::2]]
    
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_small]
        
        # Compute frame-to-frame differences
        diffs = []
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
            diffs.append(diff.mean())
        
        # Normalize
        motion_score = np.mean(diffs) / 255.0
        return motion_score