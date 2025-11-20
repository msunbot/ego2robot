"""Extract hand trajectories from clips."""
import mediapipe as mp
import cv2
import numpy as np

class HandTracker:
    def __init__(self, config):
        self.config = config
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def track_hands(self, frames):
        """
        Track hands across frames.
        Returns: list of hand data per frame
        """
        hand_tracks = []
        
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get wrist position (landmark 0) as proxy for hand position
                h, w = frame.shape[:2]
                wrist = hand_landmarks.landmark[0]
                
                hand_data = {
                    'x': wrist.x * w,
                    'y': wrist.y * h,
                    'visible': True,
                    'bbox': self._get_bbox(hand_landmarks, h, w)
                }
            else:
                # Hand not visible
                hand_data = {
                    'x': -1,
                    'y': -1,
                    'visible': False,
                    'bbox': [0, 0, 0, 0]
                }
            
            hand_tracks.append(hand_data)
        
        return hand_tracks
    
    def _get_bbox(self, hand_landmarks, h, w):
        """Get bounding box for hand."""
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        return [
            min(x_coords),
            min(y_coords),
            max(x_coords),
            max(y_coords)
        ]
    
    def compute_hand_motion(self, hand_tracks):
        """
        Compute 2D hand motion vectors (pseudo-actions).
        Returns: array of (delta_x, delta_y) normalized to [-1, 1]
        """
        actions = []
        
        for i in range(len(hand_tracks) - 1):
            curr = hand_tracks[i]
            next_hand = hand_tracks[i + 1]
            
            if curr['visible'] and next_hand['visible']:
                # Compute motion
                delta_x = next_hand['x'] - curr['x']
                delta_y = next_hand['y'] - curr['y']
                
                # Normalize by image dimensions (360 x 640)
                delta_x_norm = delta_x / 640.0
                delta_y_norm = delta_y / 360.0
                
                # Clip to [-1, 1]
                delta_x_norm = np.clip(delta_x_norm, -1.0, 1.0)
                delta_y_norm = np.clip(delta_y_norm, -1.0, 1.0)
            else:
                # Hand not visible - zero action
                delta_x_norm = 0.0
                delta_y_norm = 0.0
            
            actions.append([delta_x_norm, delta_y_norm])
        
        # Add last action (repeat previous)
        if actions:
            actions.append(actions[-1])
        else:
            actions.append([0.0, 0.0])
        
        return np.array(actions, dtype=np.float32)