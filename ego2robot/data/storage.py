# ego2robot/data/storage.py
"""Save clips to disk."""
import numpy as np
import json
import os

class ClipStorage:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output']['local_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_clips(self, clips):
        """Save clips as numpy arrays."""
        manifest = []
        
        for i, clip in enumerate(clips):
            clip_id = f"clip_{i:04d}"
            
            # Save frames
            frames_path = os.path.join(self.output_dir, f"{clip_id}.npy")
            np.save(frames_path, clip['frames'])
            
            # Create metadata - handle both 'metadata' and source metadata
            source_meta = clip.get('metadata', {})
            if not source_meta:
                # Try alternative key name
                source_meta = {
                    'factory_id': 'unknown',
                    'worker_id': 'unknown'
                }
            
            metadata = {
                'clip_id': clip_id,
                'start_time': clip.get('start_time', 0),
                'duration': clip.get('duration', 0),
                'source_metadata': source_meta,
                'quality_scores': clip.get('quality_scores', {}),
                'frames_path': frames_path,
                'num_frames': len(clip['frames']),
                'shape': list(clip['frames'].shape)
            }
            
            manifest.append(metadata)
        
        # Save manifest
        manifest_path = os.path.join(self.output_dir, 'clips_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Saved {len(clips)} clips to {self.output_dir}")
        print(f"✓ Manifest: {manifest_path}")
        
        return manifest_path