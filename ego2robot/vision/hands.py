"""Hand detection and tracking."""
import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, config):
        self.config = config
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_clip(self, frames):
        """Process clip and return hand info."""
        hand_data = []
        
        # Sample every 3rd frame for speed
        for frame in frames[::3]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_data.append(True)
            else:
                hand_data.append(False)
        
        visibility = sum(hand_data) / len(hand_data) if hand_data else 0.0
        
        return {
            'visibility_score': visibility,
            'frames_with_hands': sum(hand_data),
            'total_frames': len(hand_data)
        }