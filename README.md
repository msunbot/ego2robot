# Ego2Robot: Egocentric Factory Episodes for Robot Foundation Models
# Ego2Robot 🤖

**Transform egocentric factory video into robot-ready training data**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/msunbot1/ego2robot-factory-episodes)

Ego2Robot is an open-source pipeline that converts egocentric human demonstrations into LeRobot-compatible datasets for robot foundation model training.

## ✨ Features

- 🏭 **Real manufacturing data** from 10,000 hours of factory work
- 🔍 **Intelligent curation** with motion + hand visibility filtering
- 🧠 **Unsupervised skill discovery** via VideoMAE embeddings + clustering
- 🤖 **LeRobot v3 format** with observations + pseudo-actions
- 📊 **Rich annotations** including zero-shot labels and quality scores
- 🚀 **Reusable pipeline** for any egocentric video dataset

## 🎯 Quick Start

### Installation
```bash
git clone https://github.com/msunbot/ego2robot.git
cd ego2robot
pip install -r requirements.txt
```

### Usage
```python
from ego2robot.data.sampler import EgocentricSampler
from ego2robot.data.clips import ClipExtractor

# Load and process video
sampler = EgocentricSampler(config)
extractor = ClipExtractor(config)

for video in sampler.filter_videos():
    clips = extractor.extract_clips(video['video_bytes'], video['metadata'])
    # Process clips...
```

### Load Pre-built Dataset
```python
from datasets import load_dataset

ds = load_dataset("msunbot1/ego2robot-factory-episodes")

for episode in ds:
    images = episode['observation.images.top']
    actions = episode['action']
    # Your code here
```

## 📊 Dataset

**50 curated episodes** of factory manipulation tasks:

- **Quality Inspection:** 50% (25 episodes)
- **Assembly:** 17% (9 episodes) 
- **Fastening:** 17% (8 episodes)
- **Machine Operation:** 8% (4 episodes)
- **Mixed:** 8% (4 episodes)

**Format:** LeRobot v3 with:
- Observations: RGB (360x640@6fps) + hand bounding boxes
- Actions: 2D hand motion vectors (pseudo-actions)
- Metadata: Skill clusters, quality scores, zero-shot labels

[→ View Dataset on Hugging Face](https://huggingface.co/datasets/msunbot1/ego2robot-factory-episodes)

## 🏗️ Architecture
```
┌─────────────────────────────────────────┐
│     Egocentric-10K (10,000 hours)        │
└─────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  Quality Filtering    │
         │  - Motion scoring     │
         │  - Hand detection     │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Feature Extraction    │
         │  - VideoMAE (768-dim) │
         │  - CLIP labels        │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  Skill Clustering     │
         │  - K-means (k=10)     │
         │  - t-SNE viz          │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │   LeRobot Export      │
         │  - Hand tracking      │
         │  - Pseudo-actions     │
         └──────────────────────┘
                    ↓
      50 Robot-Ready Episodes
```

## 📁 Project Structure
```
ego2robot/
├── data/
│   ├── sampler.py          # Stream videos from HF
│   ├── clips.py            # Extract 6s clips
│   ├── quality.py          # Motion + hand filtering
│   └── storage.py          # Save curated clips
├── vision/
│   ├── motion.py           # Motion scoring
│   ├── hands.py            # Hand detection
│   ├── videomae.py         # Video embeddings
│   ├── clip_text.py        # Zero-shot labeling
│   └── hand_tracker.py     # Trajectory extraction
├── skills/
│   └── cluster.py          # K-means clustering
├── export/
│   └── lerobot_builder.py  # LeRobot format
└── examples/
    ├── day5_build_dataset.py        # Full pipeline
    ├── day12_build_lerobot_dataset.py
    └── day17_training_demo.py       # Validation
```

## 🚀 Pipeline Steps

### 1. Curate Clips (Week 1)
```bash
python examples/day5_build_dataset.py
```

Outputs: 50-100 high-quality clips in `data/ego2robot_dataset/`

### 2. Extract Features (Week 2)
```bash
python examples/day9_extract_all_embeddings.py
python examples/day10_add_all_labels.py
python examples/day11_cluster_skills.py
```

Outputs: Embeddings, labels, and cluster IDs

### 3. Export to LeRobot (Week 3)
```bash
python examples/day12_build_lerobot_dataset.py
```

Outputs: 50 episodes in `data/lerobot_dataset/`

### 4. Upload to HF Hub
```bash
python examples/day14_upload_to_hf.py
```

## 📈 Results

### Quality Metrics
- Motion score: 0.168 avg (>0.15 threshold)
- Hand visibility: 0.421 avg (>0.30 threshold)
- Cluster separation: Clear in t-SNE visualization
- Training demo: Converged MSE loss

### Discovered Skills
10 fine-grained clusters mapping to 5 high-level actions:
1. **Quality Inspection** (6 variants) - 30 clips
2. **Assembly** (2 variants) - 10 clips
3. **Fastening** - 10 clips
4. **Machine Operation** - 5 clips
5. **Mixed** - 5 clips

[→ View t-SNE Visualization](data/skill_clusters.png)

## 🎓 Use Cases

### For Researchers
- **VLA pretraining:** Diverse visual data for models like π₀
- **Representation learning:** Learn manipulation primitives
- **Skill discovery:** Study unsupervised clustering approaches
- **Domain adaptation:** Manufacturing → other domains

### For Companies
- **Custom datasets:** Process your factory video
- **Robot training:** Fine-tune policies on domain-specific data
- **Quality control:** Automated task recognition

## 🤝 Contributing

We welcome contributions! Areas of interest:

- [ ] Additional domains (warehouses, kitchens, etc.)
- [ ] Depth estimation integration
- [ ] Improved action generation (3D trajectories)
- [ ] Evaluation benchmarks
- [ ] Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use this dataset or code, please cite:
```bibtex
@software{ego2robot2025,
  author = {Michelle Sun},
  title = {Ego2Robot: Egocentric Factory Episodes for Robot Learning},
  year = {2025},
  url = {https://github.com/msunbot/ego2robot}
}
```

## 📄 License

- **Code:** MIT License
- **Dataset:** Apache 2.0 (inherits from Egocentric-10K)

## 🙏 Acknowledgments

- [BuildAI](https://huggingface.co/datasets/builddotai/Egocentric-10K) for Egocentric-10K dataset
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for format standards
- [Physical Intelligence](https://www.physicalintelligence.company/) for π₀ inspiration
- Open-source community for VideoMAE, CLIP, MediaPipe

## 📬 Contact

**Michelle Sun**
- LinkedIn: linkedin.com/in/sunmichelle
- Twitter: @michellelsun
- Email: michelle@aetherone.xyz

**Interested in:**
- Collaborations on Physical AI data & ecosystem
- Advisory & angel investing opportunities in robotics/AI

## 🔗 Links

- 📊 [Dataset on Hugging Face](https://huggingface.co/datasets/msunbot1/ego2robot-factory-episodes)
- 📝 [Blog Post](michellebuilds.substack.com)
- 📈 [Project Roadmap](ROADMAP.md)

---

*Built with ❤️ for the robotics community*
