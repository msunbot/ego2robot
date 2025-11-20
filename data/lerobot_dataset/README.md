---
license: apache-2.0
task_categories:
- robotics
- video-classification
tags:
- robot-learning
- egocentric-video
- manufacturing
- lerobot
pretty_name: Ego2Robot Factory Episodes
size_categories:
- n<1K
---

# Ego2Robot: Factory Manipulation Episodes

## Dataset Description

50 curated episodes of factory worker manipulation tasks, converted from egocentric video into LeRobot-compatible format for robot learning research.

### Key Features

- **50 episodes** (~1,800 frames total)
- **Real factory work** from 85 manufacturing facilities
- **10 skill clusters** discovered via unsupervised learning
- **LeRobot v3.0 format** with observations + pseudo-actions
- **Rich annotations**: VideoMAE embeddings, CLIP labels, quality scores

### Data Structure

Each episode contains:
- **Observations**:
  - `observation.images.top`: RGB frames (360x640, 6fps)
  - `observation.state`: Hand bounding box [x_min, y_min, x_max, y_max]
- **Actions**: 2D hand motion vectors `[delta_x, delta_y]` (pseudo-actions for representation learning)
- **Metadata**: Skill cluster ID, zero-shot action label, quality scores

### Skill Distribution

- Quality Inspection: 50% (25 episodes)
- Assembly: 17% (9 episodes)
- Fastening: 17% (8 episodes)
- Machine Operation: 8% (4 episodes)
- Mixed: 8% (4 episodes)

### Intended Use

**Primary**: Representation learning and pretraining for vision-language-action (VLA) models

- Pretrain visual encoders on diverse manipulation tasks
- Learn spatial reasoning from egocentric perspective
- Discover manipulation primitives via clustering
- Domain adaptation for manufacturing robotics

**NOT intended for**: Direct robot policy learning (actions are pseudo-actions from human hand motion, not robot joint commands)

### Data Collection

- **Source**: BuildAI Egocentric-10K dataset
- **Processing pipeline**:
  1. Quality filtering (motion + hand visibility)
  2. VideoMAE embeddings (768-dim)
  3. CLIP zero-shot labeling
  4. K-means clustering (10 skills)
  5. Hand tracking (MediaPipe)
  6. LeRobot format conversion

### Citation
```bibtex
@dataset{ego2robot2025,
  title={Ego2Robot: Factory Manipulation Episodes for Robot Learning},
  author={Michelle Sun},
  year={2025},
  url={https://huggingface.co/datasets/msunbot1/ego2robot-factory-episodes}
}
```

### License

Apache 2.0 (inherits from Egocentric-10K source dataset)

### Contact

For questions or collaboration: x.com/michellelsun
