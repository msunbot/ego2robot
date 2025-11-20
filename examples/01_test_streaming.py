"""
Test the streaming and clip extraction pipeline.
"""
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ego2robot.data.sampler import EgocentricSampler
from ego2robot.data.clips import ClipExtractor

from datasets import get_dataset_config_names

configs = get_dataset_config_names("builddotai/Egocentric-10K")
print(configs)

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Test sampling
print("Testing video sampling...")
sampler = EgocentricSampler(config)
sampler.load_dataset()

# Get first 3 videos
videos = []
for i, video in enumerate(sampler.filter_videos()):
    print(f"Video {i}: Factory {video['metadata']['factory_id']}, "
          f"Duration {video['metadata']['duration']}s")
    videos.append(video)
    if i >= 2:
        break

# Test clip extraction
print("\nTesting clip extraction...")
extractor = ClipExtractor(config)

for i, video in enumerate(videos):
    clips = extractor.extract_clips(
        video['video_bytes'],
        video['metadata']
    )
    print(f"Video {i}: Extracted {len(clips)} clips")
    
    if clips:
        print(f"  First clip: {clips[0]['frames'].shape}, "
              f"starts at {clips[0]['start_time']}s")

print("\n✓ Pipeline test complete!")