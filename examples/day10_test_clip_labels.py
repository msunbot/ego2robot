"""Add CLIP labels to all clips."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import json
from tqdm import tqdm
from ego2robot.vision.clip_text import CLIPLabeler

with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("Adding CLIP labels to all clips...")

labeler = CLIPLabeler(config)
updated_manifest = []

for clip_meta in tqdm(manifest, desc="Labeling clips"):
    frames = np.load(clip_meta['frames_path'])
    labels = labeler.label_clip(frames)
    
    clip_meta['zero_shot_labels'] = labels
    updated_manifest.append(clip_meta)

# Save updated manifest
with open('data/ego2robot_dataset/clips_manifest.json', 'w') as f:
    json.dump(updated_manifest, f, indent=2)

print("\n✓ All clips labeled!")
print("✓ Manifest updated with zero-shot labels")