"""Quality filtering for clips."""
from ego2robot.vision.motion import MotionScorer
from ego2robot.vision.hands import HandDetector

class QualityFilter:
    def __init__(self, config):
        self.config = config
        self.motion_scorer = MotionScorer(config)
        self.hand_detector = HandDetector(config)
        
    def filter_clips(self, clips):
        """Filter clips by quality scores."""
        min_motion = self.config['clips']['min_motion_score']
        min_hands = self.config['clips']['min_hand_visibility']
        
        filtered = []
        
        for clip in clips:
            frames = clip['frames']
            
            # Score motion
            motion = self.motion_scorer.score_clip(frames)
            
            # Score hands
            hand_info = self.hand_detector.process_clip(frames)
            hand_vis = hand_info['visibility_score']
            
            # Apply filters
            if motion >= min_motion and hand_vis >= min_hands:
                clip['quality_scores'] = {
                    'motion': motion,
                    'hand_visibility': hand_vis
                }
                filtered.append(clip)
        
        return filtered