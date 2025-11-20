"""Extract embeddings for all clips."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import json
from tqdm import tqdm
from ego2robot.vision.videomae import VideoMAEEmbedder

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Load manifest
with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("="*60)
print("EXTRACTING EMBEDDINGS FOR ALL CLIPS")
print("="*60)
print(f"Total clips: {len(manifest)}")

# Initialize embedder
embedder = VideoMAEEmbedder(config)

# Extract embeddings
embeddings = []
updated_manifest = []

for i, clip_meta in enumerate(tqdm(manifest, desc="Extracting embeddings")):
    # Load frames
    frames = np.load(clip_meta['frames_path'])
    
    # Extract embedding
    embedding = embedder.embed_clip(frames)
    embeddings.append(embedding)
    
    # Update metadata
    clip_meta['embedding_path'] = clip_meta['frames_path'].replace('.npy', '_embedding.npy')
    np.save(clip_meta['embedding_path'], embedding)
    
    updated_manifest.append(clip_meta)

# Save embeddings as single file too
embeddings_array = np.stack(embeddings)
np.save('data/ego2robot_dataset/all_embeddings.npy', embeddings_array)

# Update manifest
with open('data/ego2robot_dataset/clips_manifest.json', 'w') as f:
    json.dump(updated_manifest, f, indent=2)

print("\n" + "="*60)
print("EMBEDDINGS EXTRACTED")
print("="*60)
print(f"Embeddings shape: {embeddings_array.shape}")
print(f"Saved to: data/ego2robot_dataset/all_embeddings.npy")
print(f"Updated manifest with embedding paths")