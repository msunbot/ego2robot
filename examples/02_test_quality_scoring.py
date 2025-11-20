"""
Test motion and hand detection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from ego2robot.data.sampler import EgocentricSampler
from ego2robot.data.clips import ClipExtractor
from ego2robot.vision.motion import MotionScorer
from ego2robot.vision.hands import HandDetector

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Get one video
sampler = EgocentricSampler(config)
video = next(sampler.filter_videos())

# Extract clips
extractor = ClipExtractor(config)
clips = extractor.extract_clips(video['video_bytes'], video['metadata'])

print(f"Extracted {len(clips)} clips")

# Score clips
motion_scorer = MotionScorer(config)
hand_detector = HandDetector(config)

for i, clip in enumerate(clips[:3]):  # Test first 3
    frames = clip['frames']
    
    # Motion score
    motion = motion_scorer.score_clip(frames)
    print(f"\nClip {i}:")
    print(f"  Motion score: {motion:.3f}")
    
    # Hand detection
    hand_data = hand_detector.process_clip(frames)
    print(f"  Hand visibility: {hand_data['visibility_score']:.3f}")
    print(f"  Hand motion: {hand_data['motion_score']:.1f}px")
    
    # Filter decision
    min_motion = config['clips']['min_motion_score']
    min_hands = config['clips']['min_hand_visibility']
    
    keep = (motion >= min_motion and 
            hand_data['visibility_score'] >= min_hands)
    print(f"  Keep clip: {keep}")

print("\n✓ Quality scoring test complete!")