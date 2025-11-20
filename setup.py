from setuptools import setup, find_packages

setup(
    name="ego2robot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        'console_scripts': [
            'ego2robot=ego2robot.cli:cli',
        ],
    },
    author="Michelle Sun",
    description="Convert egocentric video to robot training datasets",
    python_requires='>=3.8',
)