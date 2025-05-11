import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import albumentations as A

class COCODataset(Dataset):
    """COCO dataset for object detection.
    
    This class handles loading and preprocessing of the COCO dataset for training
    object detection models. It supports data augmentation and transforms the
    annotations to the format required by the detection models.
    """
    
    def __init__(self, root_dir, ann_file, transform=None, target_transform=None, is_train=True):
        """Initialize the COCO dataset.
        
        Args:
            root_dir (str): Root directory of the COCO dataset
            ann_file (str): Path to the annotation file
            transform (callable, optional): Optional transform to be applied on the image
            target_transform (callable, optional): Optional transform to be applied on the target
            is_train (bool): Whether this is for training or validation
        """
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories.sort(key=lambda x: x['id'])
        
        # Create category to index mapping
        self.category_to_idx = {}
        for i, cat in enumerate(self.categories):
            self.category_to_idx[cat['id']] = i + 1  # +1 because 0 is background
        
        # Default transforms if none provided
        if self.transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
            else:
                self.transform = A.Compose([
                    A.Resize(height=512, width=512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.ids)
    
    def __getitem__(self, idx):
        """Get an item from the dataset.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            tuple: (image, target) where target is a dictionary containing the annotations
        """
        # Get image ID
        img_id = self.ids[idx]
        
        # Get image info
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and categories
        boxes = []
        category_ids = []
        
        for ann in anns:
            # Skip annotations with no area or no segmentation
            if ann['area'] <= 0 or not ann['segmentation']:
                continue
            
            # Get bounding box
            x, y, w, h = ann['bbox']
            boxes.append([x, y, w, h])
            
            # Get category ID and map to index
            cat_id = ann['category_id']
            category_ids.append(self.category_to_idx[cat_id])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        category_ids = np.array(category_ids, dtype=np.int64)
        
        # Apply transforms
        if self.transform and len(boxes) > 0:
            transformed = self.transform(image=img, bboxes=boxes, category_ids=category_ids)
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            category_ids = np.array(transformed['category_ids'], dtype=np.int64)
        elif self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        
        # Prepare target
        target = {}
        target['boxes'] = torch.from_numpy(boxes)
        target['labels'] = torch.from_numpy(category_ids)
        target['image_id'] = torch.tensor([img_id])
        
        # Convert COCO format (x, y, w, h) to (x1, y1, x2, y2)
        if len(boxes) > 0:
            target['boxes'][:, 2] = target['boxes'][:, 0] + target['boxes'][:, 2]
            target['boxes'][:, 3] = target['boxes'][:, 1] + target['boxes'][:, 3]
        
        # Apply target transform if provided
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_img_info(self, idx):
        """Get image info for the given index.
        
        Args:
            idx (int): Index of the image
            
        Returns:
            dict: Dictionary containing image information
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info
    
    @property
    def num_classes(self):
        """Return the number of classes in the dataset."""
        return len(self.categories) + 1  # +1 for background class
    
    def get_category_names(self):
        """Get the names of all categories.
        
        Returns:
            list: List of category names
        """
        return ['background'] + [cat['name'] for cat in self.categories]
    
    def collate_fn(self, batch):
        """Custom collate function for batching.
        
        Args:
            batch (list): List of (image, target) tuples
            
        Returns:
            tuple: (images, targets) where images is a tensor of shape (B, C, H, W)
                  and targets is a list of dictionaries
        """
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, targets