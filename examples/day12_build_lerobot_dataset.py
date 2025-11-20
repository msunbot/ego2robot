"""Build LeRobot dataset from curated clips."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import numpy as np
from tqdm import tqdm
from ego2robot.vision.hand_tracker import HandTracker
from ego2robot.export.lerobot_builder import LeRobotEpisodeBuilder

# Create export directory
os.makedirs('ego2robot/export', exist_ok=True)
open('ego2robot/export/__init__.py', 'w').close()

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Load manifest
with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("="*60)
print("BUILDING LEROBOT DATASET")
print("="*60)
print(f"Total clips: {len(manifest)}")

# Select 50 best clips (highest quality scores)
manifest_sorted = sorted(
    manifest,
    key=lambda x: x['quality_scores']['motion'] + x['quality_scores']['hand_visibility'],
    reverse=True
)
selected_clips = manifest_sorted[:50]

print(f"Selected top 50 clips for LeRobot export")

# Initialize
tracker = HandTracker(config)
builder = LeRobotEpisodeBuilder(config)

# Process each clip
clips_data = []

for i, clip_meta in enumerate(tqdm(selected_clips, desc="Processing clips")):
    # Load frames
    frames = np.load(clip_meta['frames_path'])
    
    # Track hands
    hand_tracks = tracker.track_hands(frames)
    
    # Compute actions
    actions = tracker.compute_hand_motion(hand_tracks)
    
    # Package
    clips_data.append({
        'frames': frames,
        'hand_tracks': hand_tracks,
        'actions': actions,
        'metadata': {
            'clip_id': clip_meta['clip_id'],
            'skill_cluster_id': clip_meta['skill_cluster_id'],
            'skill_cluster_name': clip_meta['skill_cluster_name'],
            'zero_shot_label': clip_meta['zero_shot_labels']['top_label'],
            'quality_scores': clip_meta['quality_scores']
        }
    })

# Build episodes
dataset_path = builder.build_episodes(clips_data)

print("\n" + "="*60)
print("LEROBOT DATASET COMPLETE")
print("="*60)
print(f"Location: {dataset_path}")
print(f"Episodes: 50")
print(f"Total frames: ~{50 * 36} (~1800)")