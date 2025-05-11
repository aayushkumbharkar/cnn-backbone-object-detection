import os
import kagglehub
import shutil
import zipfile
import sys

def download_and_setup_datasets():
    """Download and set up COCO 2017 and Pascal VOC 2012 datasets.
    
    This function downloads the datasets from Kaggle using kagglehub,
    extracts them to the appropriate locations, and organizes them to work
    with the existing dataset handlers.
    """
    print("Starting dataset download and setup...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download COCO 2017 dataset
    print("\nDownloading COCO 2017 dataset...")
    try:
        coco_path = kagglehub.dataset_download("sabahesaraki/coco-2017")
        print(f"COCO dataset downloaded to: {coco_path}")
        
        # Set up COCO dataset structure
        coco_dir = os.path.join(data_dir, 'coco')
        os.makedirs(coco_dir, exist_ok=True)
        
        # Extract and organize COCO files
        print("Extracting and organizing COCO dataset...")
        for root, dirs, files in os.walk(coco_path):
            for file in files:
                if file.endswith('.zip'):
                    zip_path = os.path.join(root, file)
                    print(f"Extracting {zip_path} to {coco_dir}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(coco_dir)
        
        print("COCO dataset setup complete!")
    except Exception as e:
        print(f"Error downloading or setting up COCO dataset: {e}")
    
    # Download Pascal VOC 2012 dataset
    print("\nDownloading Pascal VOC 2012 dataset...")
    try:
        voc_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")
        print(f"Pascal VOC dataset downloaded to: {voc_path}")
        
        # Set up VOC dataset structure
        voc_dir = os.path.join(data_dir, 'VOCdevkit')
        os.makedirs(voc_dir, exist_ok=True)
        
        # Extract and organize VOC files
        print("Extracting and organizing Pascal VOC dataset...")
        for root, dirs, files in os.walk(voc_path):
            for file in files:
                if file.endswith('.zip') or file.endswith('.tar'):
                    archive_path = os.path.join(root, file)
                    print(f"Extracting {archive_path} to {voc_dir}...")
                    if file.endswith('.zip'):
                        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                            zip_ref.extractall(voc_dir)
                    elif file.endswith('.tar'):
                        import tarfile
                        with tarfile.open(archive_path) as tar_ref:
                            tar_ref.extractall(voc_dir)
        
        print("Pascal VOC dataset setup complete!")
    except Exception as e:
        print(f"Error downloading or setting up Pascal VOC dataset: {e}")
    
    print("\nDataset download and setup complete!")
    print("\nUsage examples:")
    print("\nFor COCO dataset:")
    print("from dataset.coco import COCODataset")
    print("train_dataset = COCODataset(root_dir='data/coco', ann_file='data/coco/annotations/instances_train2017.json', is_train=True)")
    print("val_dataset = COCODataset(root_dir='data/coco', ann_file='data/coco/annotations/instances_val2017.json', is_train=False)")
    
    print("\nFor Pascal VOC dataset:")
    print("from dataset.pascal_voc import PascalVOCDataset")
    print("train_dataset = PascalVOCDataset(root_dir='data/VOCdevkit', split='train', year='2012', is_train=True)")
    print("val_dataset = PascalVOCDataset(root_dir='data/VOCdevkit', split='val', year='2012', is_train=False)")

if __name__ == "__main__":
    # Check if kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        print("kagglehub installed successfully!")
    
    download_and_setup_datasets()