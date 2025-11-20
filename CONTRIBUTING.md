# Contributing to Ego2Robot

Thank you for your interest in contributing to Ego2Robot! This project aims to democratize robot learning data by making it easy to convert egocentric video into robot-ready formats.

## 🎯 Ways to Contribute

### 1. New Datasets
Port additional egocentric datasets to LeRobot format:
- Ego4D (kitchen tasks)
- EPIC-KITCHENS (cooking)
- Assembly101 (IKEA furniture assembly)
- Your own egocentric videos

### 2. Improved Features
- **3D understanding:** Integrate depth estimation (DepthAnything, MiDaS)
- **Better actions:** Improve motion retargeting (IK, trajectory smoothing)
- **More modalities:** Audio, IMU data, force sensors
- **Quality metrics:** Automated dataset quality scoring

### 3. Documentation
- Tutorials for specific use cases
- Video walkthroughs
- API documentation improvements
- Translation to other languages

### 4. Bug Fixes
- Report issues with clear reproduction steps
- Fix bugs and submit PRs
- Improve error handling

### 5. Integration
- LeRobot v3 format improvements
- Physical Intelligence π₀ integration examples
- World Labs Marble integration
- ROS/Isaac Sim connectors

## 🚀 Getting Started

### Development Setup
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/ego2robot.git
cd ego2robot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Run Tests
```bash
# Run basic pipeline test
python examples/day5_build_dataset.py

# Validate LeRobot format
ego2robot validate data/lerobot_dataset
```

## 📝 Contribution Process

### 1. Open an Issue
Before starting work, open an issue describing:
- What you want to add/fix
- Why it's valuable
- Your proposed approach

This helps avoid duplicate work and ensures alignment with project goals.

### 2. Fork & Branch
```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ego2robot.git
cd ego2robot
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Write clean, documented code
- Add type hints where appropriate
- Follow existing code style
- Update relevant documentation

### 4. Test Locally
```bash
# Test your changes
python examples/test_your_feature.py

# Ensure existing examples still work
python examples/day5_build_dataset.py
```

### 5. Submit Pull Request
- Push to your fork
- Open PR with clear description
- Reference related issues
- Be responsive to feedback

## 💻 Code Style

### Python Style
```python
# Good: Clear, documented, typed
def extract_clips(
    video_bytes: bytes,
    metadata: dict,
    target_duration: float = 6.0
) -> List[Dict[str, Any]]:
    """
    Extract clips from video.
    
    Args:
        video_bytes: Raw video file bytes
        metadata: Video metadata dict
        target_duration: Clip length in seconds
        
    Returns:
        List of clip dicts with frames and metadata
    """
    # Implementation...
```

### Documentation Style
- Use clear, concise language
- Include code examples
- Explain *why*, not just *what*
- Add type hints and docstrings

### Commit Messages
```bash
# Good commit messages:
git commit -m "Add depth estimation support with DepthAnything"
git commit -m "Fix hand tracking timeout on long videos"
git commit -m "Update README with new installation instructions"

# Not great:
git commit -m "fixed stuff"
git commit -m "update"
```

## 🎓 Priority Areas

We especially welcome contributions in these areas:

### High Priority
1. **Dataset diversity:** Port Ego4D, EPIC-KITCHENS, Assembly101
2. **Depth estimation:** Integrate DepthAnything for 3D understanding
3. **Evaluation:** Create benchmark for VLA pretraining quality
4. **Documentation:** Video tutorials, Colab notebooks

### Medium Priority
1. **Performance:** Optimize video processing speed
2. **Quality control:** Automated clip quality scoring
3. **Visualization:** Better inspection tools (Rerun integration)
4. **Testing:** Unit tests, integration tests

### Ideas Welcome
- Novel feature extraction methods
- Alternative action representations
- Multi-modal data (audio, force, etc.)
- Domain-specific adaptations

## 📋 Pull Request Checklist

Before submitting, ensure:

- [ ] Code runs without errors
- [ ] New features are documented
- [ ] Examples are updated if relevant
- [ ] README is updated if needed
- [ ] Type hints added for new functions
- [ ] Docstrings added for new classes/functions
- [ ] No large binary files (>.npy, .npz, .pth)
- [ ] Git history is clean (squash WIP commits)

## 🤝 Code of Conduct

### Be Respectful
- Welcome newcomers
- Be patient with questions
- Provide constructive feedback
- Celebrate contributions

### Be Professional
- Focus on technical merits
- Avoid personal attacks
- Keep discussions on-topic
- Respect differing viewpoints

### Be Collaborative
- Share knowledge openly
- Credit others' work
- Help others succeed
- Build together

## 📞 Questions?

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/ego2robot/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/ego2robot/discussions)
- **Email:** your@email.com
- **Twitter:** [@your_handle](https://twitter.com/your_handle)

## 🏆 Recognition

Contributors are credited in:
- README.md (Contributors section)
- Release notes for their features
- Dataset cards (if they contributed data)

Significant contributions may result in co-authorship on any related publications.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License (code) and Apache 2.0 License (datasets), matching the project's licenses.

---

## Example Contributions

### Adding a New Dataset
```python
# ego2robot/data/ego4d_sampler.py
from ego2robot.data.sampler import EgocentricSampler

class Ego4DSampler(EgocentricSampler):
    """Sampler for Ego4D dataset."""
    
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "ego4d/ego4d"  # HF dataset
    
    def filter_videos(self):
        # Custom filtering for Ego4D format
        # ...
```

### Adding Depth Estimation
```python
# ego2robot/vision/depth.py
import torch
from transformers import pipeline

class DepthEstimator:
    """Estimate depth from RGB images."""
    
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small"):
        self.pipe = pipeline("depth-estimation", model=model_name)
    
    def estimate_depth(self, image):
        """
        Estimate depth map from image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Depth map as numpy array
        """
        depth = self.pipe(image)
        return depth["depth"]
```

### Improving Documentation
```markdown
<!-- docs/tutorials/custom_dataset.md -->
# Tutorial: Adding Your Own Dataset

This guide shows how to adapt Ego2Robot for your custom egocentric videos.

## Step 1: Prepare Your Videos
...

## Step 2: Create Custom Sampler
...

## Step 3: Test Pipeline
...
```

---

**Thank you for helping make robot learning more accessible!** 🤖❤️

Every contribution, no matter how small, moves the field forward.