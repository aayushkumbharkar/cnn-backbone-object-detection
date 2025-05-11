# CNN Backbone Training Guide

## Setup and Training with Pascal VOC Dataset

This guide will help you set up the required dependencies and run the training script for object detection using the Pascal VOC dataset.

## Prerequisites

- Python 3.6 or higher
- pip package manager

## Installation

The project requires several dependencies including PyTorch, torchvision, and other libraries. You can install them using one of the following methods:

### Method 1: Using the batch file

Run the provided batch file to install all required dependencies:

```
install_dependencies.bat
```

### Method 2: Using pip directly

Install the dependencies manually using pip:

```
pip install -r requirements.txt
```

Or install the core dependencies individually:

```
pip install torch>=1.7.0 torchvision>=0.8.0 numpy>=1.19.0 pillow>=8.0.0 matplotlib>=3.3.0
```

## Dataset

The Pascal VOC dataset should be structured as follows:

```
data/VOCdevkit/
├── VOC2012/
│   ├── Annotations/
│   │   └── [xml annotation files]
│   ├── ImageSets/
│   │   └── Main/
│   │       ├── train.txt
│   │       ├── val.txt
│   │       └── ...
│   ├── JPEGImages/
│   │   └── [image files]
│   └── ...
└── ...
```

You can verify the dataset structure by running:

```
python test_dataset_paths.py
```

If the dataset is not properly set up, you can run:

```
python setup_voc_dataset.py
```

## Training

After installing the dependencies and setting up the dataset, you can start training with:

```
python train_voc_only.py
```

The training configuration can be modified in `config_voc_only.py`.

## Troubleshooting

### Missing Dependencies

If you encounter a "ModuleNotFoundError" like:

```
ModuleNotFoundError: No module named 'torchvision'
```

Run the installation commands again to ensure all dependencies are properly installed.

### Dataset Issues

If you encounter dataset-related errors, make sure the Pascal VOC dataset is properly set up by running:

```
python setup_voc_dataset.py
```

or

```
python setup_voc_dataset_robust.py
```

## Output

Training outputs will be saved to the following directories (as configured in `config_voc_only.py`):

- Model checkpoints: `checkpoints/voc_only/`
- Training logs: `logs/voc_only/`
- Other outputs: `output/voc_only/`