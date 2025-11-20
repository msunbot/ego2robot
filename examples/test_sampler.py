import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from ego2robot.data.sampler import EgocentricSampler

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Test
sampler = EgocentricSampler(config)
print("Sampling videos...")

for i, video in enumerate(sampler.filter_videos()):
    meta = video['metadata']
    print(f"\nVideo {i}:")
    print(f"  Factory: {meta.get('factory_id', 'unknown')}")
    print(f"  Duration: {meta.get('duration', 0):.1f}s")
    print(f"  Size: {len(video['video_bytes']) / 1e6:.1f}MB")

print("\n✓ Sampler works!")