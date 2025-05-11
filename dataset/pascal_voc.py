import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
import xml.etree.ElementTree as ET

class PascalVOCDataset(Dataset):
    """Pascal VOC dataset for object detection.
    
    This class handles loading and preprocessing of the Pascal VOC dataset for training
    object detection models. It supports data augmentation and transforms the
    annotations to the format required by the detection models.
    """
    
    def __init__(self, root_dir, split='train', year='2012', transform=None, target_transform=None, is_train=True):
        """Initialize the Pascal VOC dataset.
        
        Args:
            root_dir (str): Root directory of the VOC dataset
            split (str): 'train', 'val', or 'test'
            year (str): Dataset year ('2007' or '2012')
            transform (callable, optional): Optional transform to be applied on the image
            target_transform (callable, optional): Optional transform to be applied on the target
            is_train (bool): Whether this is for training or validation
        """
        self.root_dir = root_dir
        self.split = split
        self.year = year
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        
        # Define class names
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get image IDs
        self.image_dir = os.path.join(root_dir, 'VOC' + year, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'VOC' + year, 'Annotations')
        
        # Load image IDs from the split file
        split_file = os.path.join(
            root_dir, 'VOC' + year, 'ImageSets', 'Main', split + '.txt'
        )
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
        
        # Default transforms if none provided
        if self.transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            else:
                self.transform = A.Compose([
                    A.Resize(height=512, width=512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
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
        
        # Load image
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Load annotation
        ann_path = os.path.join(self.annotation_dir, img_id + '.xml')
        boxes, labels = self._parse_voc_xml(ann_path)
        
        # Apply transforms
        if self.transform and len(boxes) > 0:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['class_labels'], dtype=np.int64)
        elif self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        
        # Prepare target
        target = {}
        target['boxes'] = torch.from_numpy(boxes).float() if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.from_numpy(labels).long() if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        
        # Apply target transform if provided
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def _parse_voc_xml(self, xml_path):
        """Parse Pascal VOC XML annotation file.
        
        Args:
            xml_path (str): Path to the XML annotation file
            
        Returns:
            tuple: (boxes, labels) where boxes is a list of [x1, y1, x2, y2] coordinates
                  and labels is a list of class indices
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Get class name and convert to index
            class_name = obj.find('name').text
            class_idx = self.class_to_idx[class_name]
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Add to lists
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_idx)
        
        return boxes, labels
    
    @property
    def num_classes(self):
        """Return the number of classes in the dataset."""
        return len(self.classes)
    
    def get_class_names(self):
        """Get the names of all classes.
        
        Returns:
            list: List of class names
        """
        return self.classes
    
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