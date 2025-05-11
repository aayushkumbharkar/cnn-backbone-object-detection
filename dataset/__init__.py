# Dataset module initialization

# Import PascalVOCDataset first as it doesn't require pycocotools
from .pascal_voc import PascalVOCDataset

# Try to import COCODataset, but handle the case where pycocotools is not available
try:
    from .coco import COCODataset
    __all__ = ['COCODataset', 'PascalVOCDataset']
except ImportError:
    # If pycocotools is not available, only expose PascalVOCDataset
    print("Warning: COCODataset could not be imported due to missing dependencies.")
    print("To use COCODataset, install pycocotools: pip install pycocotools")
    print("You can still train with Pascal VOC dataset using train_voc_only.py")
    __all__ = ['PascalVOCDataset']
