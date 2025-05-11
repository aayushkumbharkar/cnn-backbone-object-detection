import os
import shutil
import zipfile
import tarfile
import glob
import sys

def setup_voc_dataset():
    """Copy and extract the Pascal VOC dataset from the cache to the project directory."""
    # Define paths
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets', 
                           'gopalbhattrai', 'pascal-voc-2012-dataset', 'versions', '1')
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    voc_dir = os.path.join(project_dir, 'VOCdevkit')
    voc2012_dir = os.path.join(voc_dir, 'VOC2012')
    
    print(f"Setting up Pascal VOC dataset...")
    print(f"Source: {cache_dir}")
    print(f"Destination: {voc_dir}")
    
    # Create destination directories if they don't exist
    os.makedirs(voc_dir, exist_ok=True)
    os.makedirs(voc2012_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(cache_dir):
        print(f"Error: Source directory {cache_dir} does not exist.")
        print("Attempting to download the dataset using download_datasets.py")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "download_datasets.py"])
            if not os.path.exists(cache_dir):
                print("Download attempt failed. Please run download_datasets.py manually.")
                return False
        except Exception as e:
            print(f"Error running download_datasets.py: {e}")
            print("Please run download_datasets.py manually.")
            return False
    
    try:
        # Look for archive files in the cache directory
        archive_files = []
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith('.zip') or file.endswith('.tar'):
                    archive_files.append(os.path.join(root, file))
        
        if not archive_files:
            # If no archives found, try to copy the directory structure
            print("No archive files found. Attempting to copy directory structure...")
            for item in os.listdir(cache_dir):
                source_item = os.path.join(cache_dir, item)
                dest_item = os.path.join(voc_dir, item)
                
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_item, dest_item)
        else:
            # Extract archive files
            for archive_path in archive_files:
                print(f"Extracting {archive_path} to {voc_dir}...")
                if archive_path.endswith('.zip'):
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(voc_dir)
                elif archive_path.endswith('.tar'):
                    with tarfile.open(archive_path) as tar_ref:
                        tar_ref.extractall(voc_dir)
        
        # Check if VOC2012 directory exists, if not create it and move files
        if not os.path.exists(voc2012_dir) or not os.listdir(voc2012_dir):
            print("Creating VOC2012 directory structure...")
            
            # Create required subdirectories
            for subdir in ['Annotations', 'ImageSets/Main', 'JPEGImages']:
                os.makedirs(os.path.join(voc2012_dir, subdir), exist_ok=True)
            
            # Look for annotation files and move them
            annotation_files = glob.glob(os.path.join(voc_dir, '**', '*.xml'), recursive=True)
            if annotation_files:
                for file in annotation_files:
                    dest_file = os.path.join(voc2012_dir, 'Annotations', os.path.basename(file))
                    shutil.copy2(file, dest_file)
                print(f"Copied {len(annotation_files)} annotation files to {os.path.join(voc2012_dir, 'Annotations')}")
            
            # Look for image files and move them
            image_files = glob.glob(os.path.join(voc_dir, '**', '*.jpg'), recursive=True)
            if image_files:
                for file in image_files:
                    dest_file = os.path.join(voc2012_dir, 'JPEGImages', os.path.basename(file))
                    shutil.copy2(file, dest_file)
                print(f"Copied {len(image_files)} image files to {os.path.join(voc2012_dir, 'JPEGImages')}")
            
            # Create train.txt and val.txt files if they don't exist
            main_dir = os.path.join(voc2012_dir, 'ImageSets', 'Main')
            image_basenames = [os.path.splitext(os.path.basename(file))[0] for file in image_files]
            
            # Split images 80/20 for train/val
            train_count = int(len(image_basenames) * 0.8)
            train_images = image_basenames[:train_count]
            val_images = image_basenames[train_count:]
            
            with open(os.path.join(main_dir, 'train.txt'), 'w') as f:
                f.write('\n'.join(train_images))
            
            with open(os.path.join(main_dir, 'val.txt'), 'w') as f:
                f.write('\n'.join(val_images))
            
            print(f"Created train.txt with {len(train_images)} images and val.txt with {len(val_images)} images")
        
        print("Pascal VOC dataset setup complete!")
        return True
    
    except Exception as e:
        print(f"Error setting up Pascal VOC dataset: {e}")
        return False

def check_voc_dataset():
    """Check if the Pascal VOC dataset is properly set up."""
    voc_path = os.path.join('data', 'VOCdevkit')
    voc2012_path = os.path.join(voc_path, 'VOC2012')
    
    if not os.path.exists(voc_path):
        print(f"Pascal VOC dataset directory not found at: {os.path.abspath(voc_path)}")
        return False
    
    if not os.path.exists(voc2012_path):
        print(f"VOC2012 directory not found at: {os.path.abspath(voc2012_path)}")
        return False
    
    # Check key subdirectories
    required_dirs = ['Annotations', 'ImageSets', 'JPEGImages']
    for subdir in required_dirs:
        path = os.path.join(voc2012_path, subdir)
        if not os.path.exists(path):
            print(f"{subdir} directory not found at: {os.path.abspath(path)}")
            return False
    
    # Check ImageSets/Main directory and split files
    main_path = os.path.join(voc2012_path, 'ImageSets', 'Main')
    if not os.path.exists(main_path):
        print(f"Main directory not found at: {os.path.abspath(main_path)}")
        return False
    
    for split_file in ['train.txt', 'val.txt']:
        if not os.path.exists(os.path.join(main_path, split_file)):
            print(f"{split_file} not found at: {os.path.abspath(os.path.join(main_path, split_file))}")
            return False
    
    # Check if there are actually files in the directories
    if len(os.listdir(os.path.join(voc2012_path, 'Annotations'))) == 0:
        print("No annotation files found in Annotations directory")
        return False
        
    if len(os.listdir(os.path.join(voc2012_path, 'JPEGImages'))) == 0:
        print("No image files found in JPEGImages directory")
        return False
    
    print("Pascal VOC dataset is properly set up!")
    return True

if __name__ == "__main__":
    print("Pascal VOC Dataset Setup")
    print("=======================")
    
    # Check if dataset is already set up
    if check_voc_dataset():
        print("\nPascal VOC dataset is already properly set up.")
    else:
        print("\nSetting up Pascal VOC dataset...")
        if setup_voc_dataset():
            # Verify setup was successful
            print("\nVerifying Pascal VOC dataset setup:")
            check_voc_dataset()
    
    print("\nUsage example:")
    print("from dataset.pascal_voc import PascalVOCDataset")
    print("train_dataset = PascalVOCDataset(root_dir='data/VOCdevkit', split='train', year='2012', is_train=True)")
    print("val_dataset = PascalVOCDataset(root_dir='data/VOCdevkit', split='val', year='2012', is_train=False)")