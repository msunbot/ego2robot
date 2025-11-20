"""VideoMAE embedding extraction."""
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel
from PIL import Image

class VideoMAEEmbedder:
    def __init__(self, config):
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading VideoMAE model on {self.device}...")
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def embed_clip(self, frames):
        """
        Extract embedding for a clip.
        frames: numpy array (T, H, W, 3)
        returns: embedding vector (768,)
        """
        # VideoMAE expects 16 frames
        num_frames = len(frames)
        if num_frames < 16:
            # Repeat frames if too short
            frames = np.tile(frames, (16 // num_frames + 1, 1, 1, 1))[:16]
        else:
            # Sample 16 frames evenly
            indices = np.linspace(0, num_frames - 1, 16, dtype=int)
            frames = frames[indices]
        
        # Convert to PIL Images
        images = [Image.fromarray(f) for f in frames]
        
        # Process
        inputs = self.processor(images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()