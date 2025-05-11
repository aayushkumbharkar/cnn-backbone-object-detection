# Dataset Download and Usage Guide

## Overview

This guide explains how to download and use the COCO 2017 and Pascal VOC 2012 datasets with the CNN Backbone project. These datasets are essential for training and evaluating object detection models.

## Prerequisites

Before downloading the datasets, ensure you have:

1. Installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Sufficient disk space (approximately 20GB for both datasets)

## Downloading the Datasets

We provide a convenient script to download both datasets using `kagglehub`:

```bash
python download_datasets.py
```

This script will:
- Install `kagglehub` if not already installed
- Download the COCO 2017 dataset from Kaggle
- Download the Pascal VOC 2012 dataset from Kaggle
- Extract and organize the datasets in the correct structure

## Dataset Structure

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

## Using the Datasets

You can use the datasets with the provided dataset handlers:

```python
# For COCO dataset
from dataset.coco import COCODataset

train_dataset = COCODataset(
    root_dir='data/coco',
    ann_file='data/coco/annotations/instances_train2017.json',
    is_train=True
)

# For Pascal VOC dataset
from dataset.pascal_voc import PascalVOCDataset

train_dataset = PascalVOCDataset(
    root_dir='data/VOCdevkit',
    split='train',
    year='2012',
    is_train=True
)
```

## Example Usage

We provide an example script that demonstrates how to load and visualize samples from both datasets:

```bash
python examples/dataset_usage.py
```

This script will:
1. Load both datasets
2. Display dataset statistics
3. Visualize sample images with bounding boxes

## Troubleshooting

If you encounter issues with the download:

1. Ensure you have a stable internet connection
2. Check that you have sufficient disk space
3. If `kagglehub` authentication fails, you may need to set up Kaggle API credentials

For more detailed information about the datasets, refer to the README in the `dataset` directory.