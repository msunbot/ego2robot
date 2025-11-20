"""
Extract clips from long videos.
"""
import cv2
import numpy as np
from typing import List, Dict
import tempfile
import os

class ClipExtractor:
    def __init__(self, config: dict):
        self.config = config
        
    def extract_clips(self, video_bytes: bytes, metadata: dict) -> List[Dict]:
        """
        Extract clips from a video.
        Returns list of clips with metadata.
        """
        # Write video bytes to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
            
        try:
            clips = self._process_video(temp_path, metadata)
        finally:
            os.unlink(temp_path)
            
        return clips
    
    def _process_video(self, video_path: str, metadata: dict) -> List[Dict]:
        """Process video file and extract clips."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video")
            return []
                
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        target_duration = self.config['clips']['target_duration']
        stride = self.config['clips']['stride']
        
        clips = []
        start_time = 0
        
        while start_time + target_duration <= duration:
            clip_data = self._extract_single_clip(
                cap, 
                start_time, 
                target_duration, 
                fps
            )
            
            if clip_data is not None:
                clips.append({
                    'frames': clip_data,
                    'start_time': start_time,
                    'duration': target_duration,
                    'source_metadata': metadata
                })
            
            start_time += stride
        
        cap.release()
        return clips
    
    def _extract_single_clip(self, cap, start_time, duration, fps):
        """Extract single clip with downsampling."""
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        target_fps = self.config['processing']['target_fps']
        frame_skip = max(1, int(fps / target_fps))
        
        num_frames = int(duration * fps)
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only keep every Nth frame
            if i % frame_skip == 0:
                # Downsample resolution immediately
                h, w = self.config['processing']['target_resolution']
                frame_small = cv2.resize(frame, (w, h))
                frames.append(frame_small)
        
        if len(frames) < 10:  # Need at least 10 frames
            return None
            
        return np.array(frames, dtype=np.uint8)