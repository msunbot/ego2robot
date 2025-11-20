"""Test quality filtering."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from ego2robot.data.sampler import EgocentricSampler
from ego2robot.data.clips import ClipExtractor
from ego2robot.data.quality import QualityFilter

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

sampler = EgocentricSampler(config)
extractor = ClipExtractor(config)
quality_filter = QualityFilter(config)

print("Testing quality filtering...")
all_clips = []

for i, video in enumerate(sampler.filter_videos()):
    print(f"\nVideo {i}:")
    clips = extractor.extract_clips(video['video_bytes'], video['metadata'])
    print(f"  Extracted: {len(clips)} clips")
    all_clips.extend(clips)

print(f"\nTotal clips before filtering: {len(all_clips)}")

# Filter
filtered_clips = quality_filter.filter_clips(all_clips)

print(f"Total clips after filtering: {len(filtered_clips)}")

# Show scores
for i, clip in enumerate(filtered_clips[:5]):
    scores = clip['quality_scores']
    print(f"\nClip {i}:")
    print(f"  Motion: {scores['motion']:.3f}")
    print(f"  Hand visibility: {scores['hand_visibility']:.3f}")

print("\n✓ Quality filtering works!")