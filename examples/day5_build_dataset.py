# examples/day5_build_dataset.py (IMPROVED VERSION)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from ego2robot.data.sampler import EgocentricSampler
from ego2robot.data.clips import ClipExtractor
from ego2robot.data.quality import QualityFilter
from ego2robot.data.storage import ClipStorage

with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

config['data']['max_videos'] = 1

print("Building dataset from multiple tar files...")

sampler = EgocentricSampler(config)
extractor = ClipExtractor(config)
quality_filter = QualityFilter(config)
storage = ClipStorage(config)

all_filtered_clips = []

# Process different part files
part_files = [0, 1, 2, 3, 4]  # part00 through part04

for part_idx in part_files:
    print(f"\n{'='*60}")
    print(f"PROCESSING PART {part_idx}")
    print(f"{'='*60}")
    
    # Modify sampler to use specific part
    # (For now, this will still use part00 - we'll fix sampler next)
    
    for video in sampler.filter_videos():
        print(f"\nProcessing video from part{part_idx:02d}...")
        print(f"  Duration: {video['metadata'].get('duration_sec', 0):.1f}s")
        
        clips = extractor.extract_clips(video['video_bytes'], video['metadata'])
        print(f"  → Extracted {len(clips)} clips")
        
        filtered = quality_filter.filter_clips(clips)
        print(f"  → Kept {len(filtered)} after filtering")
        
        all_filtered_clips.extend(filtered)
        
        del video, clips, filtered
        break  # Only process first video from this part

print(f"\n{'='*60}")
print(f"TOTAL: {len(all_filtered_clips)} curated clips")
print(f"{'='*60}")

storage.save_clips(all_filtered_clips)