"""Build LeRobot v3 episodes."""
import numpy as np
import json
import os
from pathlib import Path
import shutil

class LeRobotEpisodeBuilder:
    def __init__(self, config, output_dir="data/lerobot_dataset"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create structure
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "meta").mkdir(exist_ok=True)
        
    def build_episodes(self, clips_data):
        """
        Build LeRobot episodes from processed clips.
        clips_data: list of dicts with frames, hand_tracks, actions, metadata
        """
        print(f"Building {len(clips_data)} episodes...")
        
        episodes_metadata = []
        
        for ep_idx, clip in enumerate(clips_data):
            episode_data = self._build_episode(ep_idx, clip)
            episodes_metadata.append(episode_data)
            
            if (ep_idx + 1) % 10 == 0:
                print(f"  Built {ep_idx + 1}/{len(clips_data)} episodes")
        
        # Create info.json
        self._create_info_json(episodes_metadata)
        
        print(f"\n✓ Built {len(clips_data)} episodes")
        print(f"✓ Dataset saved to: {self.output_dir}")
        
        return str(self.output_dir)
    
    def _build_episode(self, ep_idx, clip):
        """Build single episode."""
        frames = clip['frames']
        hand_tracks = clip['hand_tracks']
        actions = clip['actions']
        metadata = clip['metadata']
        
        num_frames = len(frames)
        
        # Create episode dict
        episode = {
            'observation.images.top': [],
            'observation.state': [],
            'action': [],
            'episode_index': [],
            'frame_index': [],
            'timestamp': [],
            'next.done': [],
            'index': [],
        }
        
        for frame_idx in range(num_frames):
            # Observation: image
            episode['observation.images.top'].append(frames[frame_idx])
            
            # Observation: state (hand bbox normalized to [0, 1])
            hand = hand_tracks[frame_idx]
            if hand['visible']:
                bbox_norm = [
                    hand['bbox'][0] / 640.0,  # x_min
                    hand['bbox'][1] / 360.0,  # y_min
                    hand['bbox'][2] / 640.0,  # x_max
                    hand['bbox'][3] / 360.0,  # y_max
                ]
            else:
                bbox_norm = [0.0, 0.0, 0.0, 0.0]
            
            episode['observation.state'].append(bbox_norm)
            
            # Action: 2D hand motion
            episode['action'].append(actions[frame_idx])
            
            # Metadata
            episode['episode_index'].append(ep_idx)
            episode['frame_index'].append(frame_idx)
            episode['timestamp'].append(frame_idx / 6.0)  # 6 fps
            episode['next.done'].append(frame_idx == num_frames - 1)
            episode['index'].append(ep_idx * 1000 + frame_idx)  # Global index
        
        # Save as numpy arrays
        episode_file = self.output_dir / "data" / f"episode_{ep_idx:06d}.npz"
        
        np.savez_compressed(
            episode_file,
            **{k: np.array(v) for k, v in episode.items()}
        )
        
        return {
            'episode_index': ep_idx,
            'num_frames': num_frames,
            'clip_metadata': metadata
        }
    
    def _create_info_json(self, episodes_metadata):
        """Create LeRobot info.json metadata."""
        total_frames = sum(ep['num_frames'] for ep in episodes_metadata)
        
        info = {
            "codebase_version": "v2.0",
            "robot_type": "egocentric_human",
            "total_episodes": len(episodes_metadata),
            "total_frames": total_frames,
            "total_tasks": 1,
            "fps": 6,
            "splits": {
                "train": f"0:{len(episodes_metadata)}"
            },
            "data_path": "data/episode_{episode_index:06d}.npz",
            "features": {
                "observation.images.top": {
                    "dtype": "image",
                    "shape": [360, 640, 3],
                    "names": ["height", "width", "channel"]
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [4],
                    "names": ["hand_bbox_x_min", "hand_bbox_y_min", 
                             "hand_bbox_x_max", "hand_bbox_y_max"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["hand_delta_x", "hand_delta_y"]
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1]
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": [1]
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1]
                }
            }
        }
        
        info_path = self.output_dir / "meta" / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Created info.json with {len(episodes_metadata)} episodes")