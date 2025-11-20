"""Test VideoMAE embeddings."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import json
from ego2robot.vision.videomae import VideoMAEEmbedder

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Load manifest
with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("Testing VideoMAE embeddings...")
print(f"Total clips: {len(manifest)}")

# Initialize embedder
embedder = VideoMAEEmbedder(config)

# Test on first 3 clips
for i in range(3):
    clip_meta = manifest[i]
    
    # Load frames
    frames = np.load(clip_meta['frames_path'])
    print(f"\nClip {i}:")
    print(f"  Shape: {frames.shape}")
    
    # Extract embedding
    embedding = embedder.embed_clip(frames)
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding mean: {embedding.mean():.3f}")
    print(f"  Embedding std: {embedding.std():.3f}")

print("\n✓ VideoMAE embeddings working!")