"""Cluster clips into skills."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import json
import matplotlib.pyplot as plt
from ego2robot.skills.cluster import SkillClusterer

# Create skills directory
os.makedirs('ego2robot/skills', exist_ok=True)
open('ego2robot/skills/__init__.py', 'w').close()

with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("="*60)
print("SKILL CLUSTERING")
print("="*60)

# Load embeddings
embeddings = np.load('data/ego2robot_dataset/all_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")

# Cluster
clusterer = SkillClusterer(config, n_clusters=10)
cluster_ids = clusterer.fit(embeddings)

# Add to manifest
for i, clip_meta in enumerate(manifest):
    clip_meta['skill_cluster_id'] = int(cluster_ids[i])

# Save
with open('data/ego2robot_dataset/clips_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

# Compute t-SNE
tsne_coords = clusterer.compute_tsne(embeddings)

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    tsne_coords[:, 0],
    tsne_coords[:, 1],
    c=cluster_ids,
    cmap='tab10',
    s=100,
    alpha=0.6,
    edgecolors='black'
)
plt.colorbar(scatter, label='Skill Cluster')
plt.title('Skill Clusters (t-SNE Visualization)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(alpha=0.3)
plt.savefig('data/skill_clusters.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to data/skill_clusters.png")

# Print cluster distribution
print("\nCluster distribution:")
unique, counts = np.unique(cluster_ids, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} clips")

# Show sample labels per cluster
print("\nSample labels per cluster:")
for cluster_id in range(min(5, max(cluster_ids)+1)):
    clips_in_cluster = [m for m, cid in zip(manifest, cluster_ids) if cid == cluster_id]
    if clips_in_cluster:
        print(f"\nCluster {cluster_id}:")
        for clip in clips_in_cluster[:3]:
            label = clip['zero_shot_labels']['top_label']
            conf = clip['zero_shot_labels']['top_confidence']
            print(f"  - {label} ({conf:.2f})")

print("\n" + "="*60)
print("CLUSTERING COMPLETE")
print("="*60)