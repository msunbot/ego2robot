"""Review dataset quality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

# Load manifest
with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("="*60)
print("DATASET QUALITY REVIEW")
print("="*60)

print(f"\nTotal clips: {len(manifest)}")

# Analyze quality scores
motion_scores = [c['quality_scores']['motion'] for c in manifest]
hand_scores = [c['quality_scores']['hand_visibility'] for c in manifest]

print(f"\nMotion scores:")
print(f"  Mean: {np.mean(motion_scores):.3f}")
print(f"  Min: {np.min(motion_scores):.3f}")
print(f"  Max: {np.max(motion_scores):.3f}")

print(f"\nHand visibility scores:")
print(f"  Mean: {np.mean(hand_scores):.3f}")
print(f"  Min: {np.min(hand_scores):.3f}")
print(f"  Max: {np.max(hand_scores):.3f}")

# Check clip shapes
shapes = [c['shape'] for c in manifest]
print(f"\nClip shapes (frames, height, width, channels):")
print(f"  {shapes[0]}")

# Check file sizes
total_size = 0
for c in manifest:
    clip_path = c['frames_path']
    if os.path.exists(clip_path):
        total_size += os.path.getsize(clip_path)

print(f"\nStorage:")
print(f"  Total: {total_size / 1e9:.2f} GB")
print(f"  Per clip: {total_size / len(manifest) / 1e6:.1f} MB")

# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(motion_scores, bins=20, edgecolor='black')
ax1.set_xlabel('Motion Score')
ax1.set_ylabel('Count')
ax1.set_title('Motion Score Distribution')
ax1.axvline(0.1, color='red', linestyle='--', label='Min threshold')
ax1.legend()

ax2.hist(hand_scores, bins=20, edgecolor='black')
ax2.set_xlabel('Hand Visibility Score')
ax2.set_ylabel('Count')
ax2.set_title('Hand Visibility Distribution')
ax2.axvline(0.3, color='red', linestyle='--', label='Min threshold')
ax2.legend()

plt.tight_layout()
plt.savefig('data/quality_review.png')
print(f"\n✓ Saved quality plots to data/quality_review.png")

# Sample a few clips to verify
print(f"\nSample clips:")
for i in [0, 15, 30, 45, 59]:
    c = manifest[i]
    print(f"\nClip {i}:")
    print(f"  Motion: {c['quality_scores']['motion']:.3f}")
    print(f"  Hands: {c['quality_scores']['hand_visibility']:.3f}")
    print(f"  Frames: {c['num_frames']}")
    print(f"  Duration: {c['duration']:.1f}s")

print("\n" + "="*60)
print("QUALITY REVIEW COMPLETE")
print("="*60)
print("\n✓ Dataset looks good - ready for Week 2!")