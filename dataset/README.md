# Dataset Handling in CNN Backbone

## Available Datasets

This project supports two popular object detection datasets:

1. **COCO (Common Objects in Context)**: A large-scale object detection, segmentation, and captioning dataset with 80 object categories.
2. **Pascal VOC (Visual Object Classes)**: A dataset for object detection, segmentation, and classification with 20 object categories.

## Dataset Structure

After downloading the datasets using the provided script, they will be organized as follows:

### COCO Dataset
```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017/
│   └── [image files]
├── val2017/
│   └── [image files]
└── ...
```

### Pascal VOC Dataset
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

## Downloading Datasets

We provide a script to download and set up both datasets. The script uses `kagglehub` to download the datasets from Kaggle.

```bash
python download_datasets.py
```

This script will:
1. Install `kagglehub` if not already installed
2. Download the COCO 2017 dataset
3. Download the Pascal VOC 2012 dataset
4. Extract and organize the datasets in the correct structure

## Using the Datasets in Your Code

### COCO Dataset

```python
from dataset.coco import COCODataset

# For training
train_dataset = COCODataset(
    root_dir='data/coco',
    ann_file='data/coco/annotations/instances_train2017.json',
    is_train=True
)

# For validation
val_dataset = COCODataset(
    root_dir='data/coco',
    ann_file='data/coco/annotations/instances_val2017.json',
    is_train=False
)
```

### Pascal VOC Dataset

```python
from dataset.pascal_voc import PascalVOCDataset

# For training
train_dataset = PascalVOCDataset(
    root_dir='data/VOCdevkit',
    split='train',
    year='2012',
    is_train=True
)

# For validation
val_dataset = PascalVOCDataset(
    root_dir='data/VOCdevkit',
    split='val',
    year='2012',
    is_train=False
)
```

## Dataset Classes

### COCO Dataset

The COCO dataset contains 80 object categories plus a background class.

### Pascal VOC Dataset

The Pascal VOC dataset contains 20 object categories plus a background class:

- background
- aeroplane
- bicycle
- bird
- boat
- bottle
- bus
- car
- cat
- chair
- cow
- diningtable
- dog
- horse
- motorbike
- person
- pottedplant
- sheep
- sofa
- train
- tvmonitor