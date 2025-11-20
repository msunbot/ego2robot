"""Visualize episodes with matplotlib."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

dataset_path = Path("data/lerobot_dataset")

# Load episode
ep_idx = 0
ep = np.load(dataset_path / "data" / f"episode_{ep_idx:06d}.npz")

print(f"Visualizing episode {ep_idx}...")
print(f"Frames: {len(ep['frame_index'])}")

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Show 8 frames
frame_indices = np.linspace(0, len(ep['frame_index'])-1, 8, dtype=int)

for i, frame_idx in enumerate(frame_indices):
    ax = axes[i]
    
    # Show image
    img = ep['observation.images.top'][frame_idx]
    ax.imshow(img)
    
    # Draw hand bbox
    bbox = ep['observation.state'][frame_idx]
    if bbox[2] > 0:  # Hand visible
        x_min, y_min, x_max, y_max = bbox
        x_min *= 640
        y_min *= 360
        x_max *= 640
        y_max *= 360
        
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Show action
    action = ep['action'][frame_idx]
    ax.set_title(f"Frame {frame_idx}\nAction: [{action[0]:.3f}, {action[1]:.3f}]", 
                fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'data/episode_{ep_idx}_visualization.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: data/episode_{ep_idx}_visualization.png")

plt.show()