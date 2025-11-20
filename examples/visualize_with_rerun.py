"""3D hand trajectory visualization with Rerun."""
import rerun as rr
import numpy as np
from pathlib import Path

# Initialize Rerun
rr.init("ego2robot_trajectories", spawn=True)

# Load episode
dataset_path = Path("data/lerobot_dataset")
ep = np.load(dataset_path / "data" / "episode_000000.npz")

print("Visualizing episode with Rerun...")

# Log images
for i, img in enumerate(ep['observation.images.top']):
    rr.set_time_sequence("frame", i)
    rr.log("camera/image", rr.Image(img))
    
    # Log hand position in 3D (project 2D bbox to 3D)
    bbox = ep['observation.state'][i]
    if bbox[2] > 0:
        # Use bbox center as 3D point (x, y, depth=1.0)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        point_3d = [center_x, center_y, 1.0]
        rr.log("hand/trajectory", rr.Points3D([point_3d], radii=0.02))
        
        # Log action as arrow
        action = ep['action'][i]
        rr.log("hand/action", rr.Arrows3D(
            origins=[point_3d],
            vectors=[[action[0], action[1], 0]],
        ))

print("✓ Rerun visualization complete!")
print("View in browser: http://localhost:9876")