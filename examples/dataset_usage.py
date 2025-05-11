import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Import dataset classes
from dataset.coco import COCODataset
from dataset.pascal_voc import PascalVOCDataset

def visualize_sample(img, target, dataset_name, class_names):
    """Visualize a sample from the dataset with bounding boxes.
    
    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W)
        target (dict): Target dictionary with 'boxes' and 'labels' keys
        dataset_name (str): Name of the dataset
        class_names (list): List of class names
    """
    # Convert image from tensor to numpy array
    img = img.permute(1, 2, 0).numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    
    # Get boxes and labels
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    
    # Plot each box
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label]
        ax.text(x1, y1, class_name, bbox=dict(facecolor='yellow', alpha=0.5))
    
    ax.set_title(f'Sample from {dataset_name} dataset')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Check if datasets are downloaded
    coco_path = os.path.join('data', 'coco')
    voc_path = os.path.join('data', 'VOCdevkit')
    
    if not os.path.exists(coco_path) or not os.path.exists(voc_path):
        print("Datasets not found. Please run download_datasets.py first.")
        print("Command: python download_datasets.py")
        return
    
    # Example 1: Load COCO dataset
    print("Loading COCO dataset...")
    try:
        coco_train = COCODataset(
            root_dir=coco_path,
            ann_file=os.path.join(coco_path, 'annotations', 'instances_train2017.json'),
            is_train=True
        )
        
        coco_val = COCODataset(
            root_dir=coco_path,
            ann_file=os.path.join(coco_path, 'annotations', 'instances_val2017.json'),
            is_train=False
        )
        
        print(f"COCO train dataset size: {len(coco_train)}")
        print(f"COCO validation dataset size: {len(coco_val)}")
        print(f"Number of classes: {coco_train.num_classes}")
        
        # Create data loaders
        coco_train_loader = DataLoader(
            coco_train,
            batch_size=2,
            shuffle=True,
            collate_fn=coco_train.collate_fn
        )
        
        # Get a sample
        sample_img, sample_target = next(iter(coco_train))
        
        # Visualize sample
        visualize_sample(
            sample_img,
            sample_target,
            'COCO',
            coco_train.get_category_names()
        )
        
    except Exception as e:
        print(f"Error loading COCO dataset: {e}")
    
    # Example 2: Load Pascal VOC dataset
    print("\nLoading Pascal VOC dataset...")
    try:
        voc_train = PascalVOCDataset(
            root_dir=voc_path,
            split='train',
            year='2012',
            is_train=True
        )
        
        voc_val = PascalVOCDataset(
            root_dir=voc_path,
            split='val',
            year='2012',
            is_train=False
        )
        
        print(f"Pascal VOC train dataset size: {len(voc_train)}")
        print(f"Pascal VOC validation dataset size: {len(voc_val)}")
        print(f"Number of classes: {voc_train.num_classes}")
        
        # Create data loaders
        voc_train_loader = DataLoader(
            voc_train,
            batch_size=2,
            shuffle=True,
            collate_fn=voc_train.collate_fn
        )
        
        # Get a sample
        sample_img, sample_target = next(iter(voc_train))
        
        # Visualize sample
        visualize_sample(
            sample_img,
            sample_target,
            'Pascal VOC',
            voc_train.get_class_names()
        )
        
    except Exception as e:
        print(f"Error loading Pascal VOC dataset: {e}")

if __name__ == "__main__":
    main()