"""CLIP zero-shot action labeling."""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class CLIPLabeler:
    def __init__(self, config):
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Action prompts
        self.action_prompts = [
            "picking up an object",
            "placing an object down",
            "assembling components together",
            "tightening a screw",
            "inspecting quality of a part",
            "packaging items into a box",
            "moving materials between locations",
            "wiping or cleaning a surface",
            "measuring or marking",
            "operating machinery"
        ]
        
    def label_clip(self, frames):
        """
        Get zero-shot action labels for clip.
        frames: numpy array (T, H, W, 3)
        returns: dict with top labels and scores
        """
        # Sample middle frame
        mid_idx = len(frames) // 2
        frame = frames[mid_idx]
        
        # Convert to PIL
        image = Image.fromarray(frame)
        
        # Process
        inputs = self.processor(
            text=self.action_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get similarities
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Get top 2 labels
        top_indices = np.argsort(probs)[-2:][::-1]
        
        return {
            'top_label': self.action_prompts[top_indices[0]],
            'top_confidence': float(probs[top_indices[0]]),
            'second_label': self.action_prompts[top_indices[1]],
            'second_confidence': float(probs[top_indices[1]]),
            'all_scores': {prompt: float(score) for prompt, score in zip(self.action_prompts, probs)}
        }