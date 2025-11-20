# examples/day2_test_clips.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from ego2robot.data.sampler import EgocentricSampler
from ego2robot.data.clips import ClipExtractor

with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# ONLY 1 VIDEO
config['data']['max_videos'] = 1

sampler = EgocentricSampler(config)
extractor = ClipExtractor(config)

print("Extracting clips from 1 video...")

for i, video in enumerate(sampler.filter_videos()):
    print(f"\nProcessing video {i}...")
    print(f"  Duration: {video['metadata'].get('duration_sec', 0):.1f}s")
    
    clips = extractor.extract_clips(video['video_bytes'], video['metadata'])
    
    print(f"  Extracted {len(clips)} clips")
    if clips:
        print(f"  First clip shape: {clips[0]['frames'].shape}")
        print(f"  Memory per clip: ~{clips[0]['frames'].nbytes / 1e6:.1f} MB")

print(f"\n✓ Clips working!")