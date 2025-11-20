"""Analyze and name clusters."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import Counter

# Load manifest
with open('data/ego2robot_dataset/clips_manifest.json') as f:
    manifest = json.load(f)

print("="*60)
print("CLUSTER ANALYSIS")
print("="*60)

# Analyze each cluster
cluster_names = {}

for cluster_id in range(10):
    clips_in_cluster = [c for c in manifest if c['skill_cluster_id'] == cluster_id]
    
    if not clips_in_cluster:
        continue
    
    # Get all top labels
    labels = [c['zero_shot_labels']['top_label'] for c in clips_in_cluster]
    label_counts = Counter(labels)
    
    # Get dominant label
    dominant_label = label_counts.most_common(1)[0][0]
    
    # Create short name
    name_map = {
        'inspecting quality of a part': 'Quality Inspection',
        'operating machinery': 'Machine Operation',
        'assembling components together': 'Assembly',
        'tightening a screw': 'Fastening',
        'picking up an object': 'Picking',
        'placing an object down': 'Placing',
        'packaging items into a box': 'Packaging',
        'moving materials between locations': 'Material Handling',
        'wiping or cleaning a surface': 'Cleaning',
        'measuring or marking': 'Measurement'
    }
    
    cluster_name = name_map.get(dominant_label, dominant_label)
    cluster_names[cluster_id] = cluster_name
    
    print(f"\nCluster {cluster_id}: {cluster_name}")
    print(f"  Size: {len(clips_in_cluster)} clips")
    print(f"  Label distribution:")
    for label, count in label_counts.most_common():
        short_label = name_map.get(label, label)
        print(f"    - {short_label}: {count}")
    
    # Show quality scores
    motion_scores = [c['quality_scores']['motion'] for c in clips_in_cluster]
    hand_scores = [c['quality_scores']['hand_visibility'] for c in clips_in_cluster]
    print(f"  Avg motion: {sum(motion_scores)/len(motion_scores):.3f}")
    print(f"  Avg hand vis: {sum(hand_scores)/len(hand_scores):.3f}")

# Save cluster names to manifest
for clip in manifest:
    cluster_id = clip['skill_cluster_id']
    clip['skill_cluster_name'] = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

with open('data/ego2robot_dataset/clips_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print("\n" + "="*60)
print("✓ Cluster names added to manifest")
print("="*60)

# Summary
print("\nCluster Summary:")
for cluster_id, name in sorted(cluster_names.items()):
    count = len([c for c in manifest if c['skill_cluster_id'] == cluster_id])
    print(f"  {cluster_id}: {name} ({count} clips)")