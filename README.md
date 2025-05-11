# CNN Backbone Object Detection Framework

This project implements a modular object detection framework with various CNN backbones. It supports multiple detection heads (SSD, YOLO), datasets (COCO, Pascal VOC), and provides tools for training, evaluation, and inference.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [GitHub Repository Setup](#github-repository-setup)

## Features

- **Modular Architecture**: Easily swap between different CNN backbones and detection heads
- **Multiple Backbones**: Support for ResNet and MobileNet architectures
- **Detection Heads**: Implementations of SSD and YOLO detection approaches
- **Dataset Support**: Handlers for COCO and Pascal VOC datasets
- **Training Pipeline**: Complete training workflow with checkpointing and evaluation
- **Visualization Tools**: Utilities for visualizing detections and training metrics

## Project Structure

```
├── backbone/
│   ├── __init__.py
│   ├── resnet.py       # ResNet backbone implementation
│   └── mobilenet.py    # MobileNet backbone implementation
├── configs/
│   ├── train_ssd_resnet.yaml    # Config for SSD with ResNet50
│   └── train_yolo_mobilenet.yaml # Config for YOLO with MobileNetV2
├── detection/
│   ├── __init__.py
│   ├── ssd.py          # Single Shot Detector implementation
│   └── yolo.py         # YOLO-style detection head
├── dataset/
│   ├── __init__.py
│   ├── coco.py         # COCO dataset handler
│   └── pascal_voc.py   # Pascal VOC dataset handler
├── utils/
│   ├── __init__.py
│   ├── metrics.py      # Evaluation metrics (mAP, precision, recall)
│   └── visualization.py # Visualization utilities
├── train.py            # Training pipeline
├── train_voc_only.py   # Training pipeline for VOC dataset only
├── evaluate.py         # Evaluation script
├── demo.py             # Demo script for inference
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Method 1: Using the batch file

Run the provided batch file to install all required dependencies:

```bash
install_dependencies.bat
```

### Method 2: Using pip directly

Install the dependencies manually using pip:

```bash
pip install -r requirements.txt
```

Or install the core dependencies individually:

```bash
pip install torch>=1.7.0 torchvision>=0.8.0 numpy>=1.19.0 pillow>=8.0.0 matplotlib>=3.3.0
```

## Dataset Setup

### Downloading Datasets

We provide a convenient script to download both COCO and Pascal VOC datasets:

```bash
python download_datasets.py
```

This script will:
- Install `kagglehub` if not already installed
- Download the COCO 2017 dataset from Kaggle
- Download the Pascal VOC 2012 dataset from Kaggle
- Extract and organize the datasets in the correct structure

### Dataset Structure

After running the download script, the datasets will be organized as follows:

```
data/
├── coco/
│   ├── annotations/
│   ├── train2017/
│   ├── val2017/
│   └── ...
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── ImageSets/
        ├── JPEGImages/
        └── ...
```

### Verifying Dataset Structure

You can verify the dataset structure by running:

```bash
python test_dataset_paths.py
```

If the Pascal VOC dataset is not properly set up, you can run:

```bash
python setup_voc_dataset.py
```

## Training

### Full Training (COCO and VOC)

To train a model with both datasets, use the `train.py` script with a configuration file:

```bash
python train.py --config configs/train_ssd_resnet.yaml --data-dir data/VOCdevkit --batch-size 16 --epochs 100 --output-dir outputs/ssd_resnet
```

Options:
- `--config`: Path to the configuration file (YAML)
- `--data-dir`: Path to the dataset directory
- `--batch-size`: Batch size for training
- `--epochs`: Number of epochs to train for
- `--lr`: Learning rate
- `--device`: Device to train on (cuda or cpu)
- `--output-dir`: Directory to save outputs
- `--resume`: Path to checkpoint to resume from

### VOC-Only Training

If you only want to train with the Pascal VOC dataset (useful if you have issues with pycocotools), use:

```bash
python train_voc_only.py
```

The training configuration can be modified in `config_voc_only.py`.

## Evaluation

To evaluate a trained model, use the `evaluate.py` script:

```bash
python evaluate.py --config configs/train_ssd_resnet.yaml --checkpoint outputs/ssd_resnet/best_model.pth --data-dir data/VOCdevkit --output-dir evaluation_results --visualize
```

Options:
- `--config`: Path to the configuration file (YAML)
- `--checkpoint`: Path to the model checkpoint
- `--data-dir`: Path to the dataset directory
- `--batch-size`: Batch size for evaluation
- `--device`: Device to evaluate on (cuda or cpu)
- `--output-dir`: Directory to save evaluation results
- `--split`: Dataset split to evaluate on (test or val)
- `--year`: VOC dataset year
- `--visualize`: Visualize detections on sample images
- `--num-vis-samples`: Number of samples to visualize

## Inference

To run inference on images or videos, use the `demo.py` script:

```bash
# Run on a single image
python demo.py --config configs/train_ssd_resnet.yaml --checkpoint outputs/ssd_resnet/best_model.pth --input path/to/image.jpg --output-dir demo_results

# Run on a directory of images
python demo.py --config configs/train_ssd_resnet.yaml --checkpoint outputs/ssd_resnet/best_model.pth --input path/to/images/ --output-dir demo_results

# Run on a video
python demo.py --config configs/train_ssd_resnet.yaml --checkpoint outputs/ssd_resnet/best_model.pth --input path/to/video.mp4 --output-dir demo_results --save-video

# Run on webcam
python demo.py --config configs/train_ssd_resnet.yaml --checkpoint outputs/ssd_resnet/best_model.pth --webcam --webcam-id 0
```

Options:
- `--config`: Path to the configuration file (YAML)
- `--checkpoint`: Path to the model checkpoint
- `--input`: Path to input image, directory of images, or video
- `--output-dir`: Directory to save results
- `--device`: Device to run inference on (cuda or cpu)
- `--score-threshold`: Score threshold for detections
- `--save-video`: Save video if input is a video file
- `--webcam`: Use webcam as input
- `--webcam-id`: Webcam device ID

## Troubleshooting

### PyCocoTools Installation Issues

The `pycocotools` package is a dependency for working with the COCO dataset in this project. However, it requires Microsoft Visual C++ Build Tools to compile its C extensions on Windows, which can cause installation failures if these build tools are not properly installed.

The error typically looks like this:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Option 1: Install Microsoft Visual C++ Build Tools

If you want to use the COCO dataset, you'll need to install the Microsoft Visual C++ Build Tools:

1. Download the installer from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer and select "Desktop development with C++"
3. After installation, try installing pycocotools again:
   ```bash
   pip install pycocotools
   ```

#### Option 2: Use Pascal VOC Dataset Only

This project has been configured to work with the Pascal VOC dataset without requiring pycocotools. To use this option:

1. Run the fix script to patch the dataset module:
   ```bash
   python fix_pycocotools.py
   ```
2. Train using the VOC-only script:
   ```bash
   python train_voc_only.py
   ```

### Dataset Issues

If you encounter dataset-related errors, make sure the Pascal VOC dataset is properly set up by running:

```bash
python setup_voc_dataset.py
```

or for a more robust setup:

```bash
python setup_voc_dataset_robust.py
```

## GitHub Repository Setup

To upload this project to GitHub, follow these steps:

### 1. Create a New GitHub Repository

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the '+' icon in the top-right corner and select 'New repository'
3. Enter a repository name (e.g., "cnn-backbone-object-detection")
4. Add a description (optional)
5. Choose public or private visibility
6. Do not initialize the repository with a README, .gitignore, or license
7. Click 'Create repository'

### 2. Initialize Git in Your Local Project

Open a command prompt in your project directory and run:

```bash
git init
git add .
git commit -m "Initial commit"
```

### 3. Connect to GitHub Repository

Connect your local repository to the GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your GitHub username and the name of your repository.

### 4. Verify Upload

Go to your GitHub repository page to verify that all files have been uploaded successfully.

### 5. Additional GitHub Features to Consider

- **Issues**: Enable issues to track bugs, enhancements, and other requests
- **Wiki**: Set up a wiki for more detailed documentation
- **GitHub Actions**: Create workflows for continuous integration and testing
- **Releases**: Create releases for stable versions of your project

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The COCO dataset team for providing the dataset
- The Pascal VOC dataset team for providing the dataset
- The PyTorch team for the deep learning framework