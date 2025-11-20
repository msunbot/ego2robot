"""Upload dataset to Hugging Face Hub."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo
from pathlib import Path

print("="*60)
print("UPLOADING TO HUGGING FACE HUB")
print("="*60)

# Configuration
USERNAME = input("Enter your HF username: ").strip()
DATASET_NAME = "ego2robot-factory-episodes"
REPO_ID = f"{USERNAME}/{DATASET_NAME}"

print(f"\nRepo ID: {REPO_ID}")
print(f"Creating repository...")

# Create repo
api = HfApi()

try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False
    )
    print(f"✓ Created repo: {REPO_ID}")
except Exception as e:
    print(f"⚠️  Repo might already exist: {e}")

# Upload files
dataset_path = Path("data/lerobot_dataset")

print(f"\nUploading dataset from {dataset_path}...")
print(f"This may take 5-10 minutes...")

api.upload_folder(
    folder_path=str(dataset_path),
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Initial upload: 50 factory manipulation episodes"
)

print(f"\n" + "="*60)
print(f"✓ UPLOAD COMPLETE!")
print(f"="*60)
print(f"\nDataset URL: https://huggingface.co/datasets/{REPO_ID}")
print(f"\nTo load:")
print(f"  from datasets import load_dataset")
print(f"  ds = load_dataset('{REPO_ID}')")