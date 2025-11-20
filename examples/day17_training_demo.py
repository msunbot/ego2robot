"""Train action prediction model on ego2robot dataset."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# Simple CNN for action prediction
class ActionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Predict (delta_x, delta_y)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Dataset
class Ego2RobotDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.episodes = sorted((self.dataset_path / "data").glob("episode_*.npz"))
        
        # Load all data
        self.images = []
        self.actions = []
        
        for ep_file in self.episodes:
            ep = np.load(ep_file)
            self.images.append(ep['observation.images.top'])
            self.actions.append(ep['action'])
        
        self.images = np.concatenate(self.images)
        self.actions = np.concatenate(self.actions)
        
        print(f"Loaded {len(self.images)} frames from {len(self.episodes)} episodes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        action = torch.from_numpy(self.actions[idx])
        
        return img, action

# Train
print("="*60)
print("TRAINING ACTION PREDICTOR")
print("="*60)

dataset = Ego2RobotDataset("data/lerobot_dataset")

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ActionPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train loop
epochs = 10
for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    for images, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, actions)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, actions in val_loader:
            images = images.to(device)
            actions = actions.to(device)
            pred = model(images)
            loss = criterion(pred, actions)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

print("\n✓ Training complete!")
print(f"Final validation loss: {val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'data/action_predictor.pth')
print(f"✓ Model saved: data/action_predictor.pth")