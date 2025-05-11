import os
import shutil
import zipfile
import tarfile
import glob
import sys
import subprocess
import time

def download_voc_dataset():
    """Download the Pascal VOC dataset using kagglehub."""
    print("Downloading Pascal VOC 2012 dataset...")
    try:
        # Try to import kagglehub
        try:
            import kagglehub
        except ImportError:
            print("kagglehub is not installed. Installing it now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub
            print("kagglehub installed successfully!")
        
        # Download the dataset
        voc_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")
        print(f"Pascal VOC dataset downloaded to: {voc_path}")
        return voc_path
    except Exception as e:
        print(f"Error downloading Pascal VOC dataset: {e}")
        return None

def setup_voc_dataset():
    """Set up the Pascal VOC dataset with proper directory structure."""
    # Define paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, 'data')
    voc_dir = os.path.join(data_dir, 'VOCdevkit')
    voc2012_dir = os.path.join(voc_dir, 'VOC2012')
    
    print(f"Setting up Pascal VOC dataset...")
    print(f"Destination: {voc_dir}")
    
    # Create destination directories if they don't exist
    os.makedirs(voc_dir, exist_ok=True)
    os.makedirs(voc2012_dir, exist_ok=True)
    
    # Try to find the dataset in the Kaggle cache
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets', 
                           'gopalbhattrai', 'pascal-voc-2012-dataset')
    
    # Look for the dataset in all version directories
    dataset_found = False
    source_dir = None
    
    if os.path.exists(cache_dir):
        for version_dir in os.listdir(cache_dir):
            if version_dir.startswith('version'):
                version_path = os.path.join(cache_dir, version_dir)
                if os.path.isdir(version_path):
                    for root, dirs, files in os.walk(version_path):
                        for file in files:
                            if file.endswith('.zip') or file.endswith('.tar'):
                                source_dir = version_path
                                dataset_found = True
                                break
                        if dataset_found:
                            break
                if dataset_found:
                    break
    
    # If dataset not found in cache, try to download it
    if not dataset_found:
        print("Dataset not found in cache. Attempting to download...")
        download_path = download_voc_dataset()
        if download_path:
            source_dir = download_path
            dataset_found = True
    
    if not dataset_found:
        print("Failed to find or download the dataset. Please run download_datasets.py manually.")
        return False
    
    print(f"Source: {source_dir}")
    
    try:
        # Look for archive files in the source directory
        archive_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.zip') or file.endswith('.tar'):
                    archive_files.append(os.path.join(root, file))
        
        if archive_files:
            # Extract archive files
            for archive_path in archive_files:
                print(f"Extracting {archive_path} to {voc_dir}...")
                if archive_path.endswith('.zip'):
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(voc_dir)
                elif archive_path.endswith('.tar'):
                    with tarfile.open(archive_path) as tar_ref:
                        tar_ref.extractall(voc_dir)
        else:
            # If no archives found, try to copy the directory structure
            print("No archive files found. Attempting to copy directory structure...")
            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                dest_item = os.path.join(voc_dir, item)
                
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_item, dest_item)
        
        # Create VOC2012 directory structure if it doesn't exist or is empty
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
    annotations_dir = os.path.join(voc2012_path, 'Annotations')
    if os.path.exists(annotations_dir) and len(os.listdir(annotations_dir)) == 0:
        print("No annotation files found in Annotations directory")
        return False
        
    images_dir = os.path.join(voc2012_path, 'JPEGImages')
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) == 0:
        print("No image files found in JPEGImages directory")
        return False
    
    print("Pascal VOC dataset is properly set up!")
    return True

def create_dummy_dataset():
    """Create a minimal dummy dataset for testing purposes."""
    print("Creating a minimal dummy VOC dataset for testing...")
    
    voc_path = os.path.join('data', 'VOCdevkit')
    voc2012_path = os.path.join(voc_path, 'VOC2012')
    
    # Create required directories
    for subdir in ['Annotations', 'ImageSets/Main', 'JPEGImages']:
        os.makedirs(os.path.join(voc2012_path, subdir), exist_ok=True)
    
    # Create dummy annotation file
    annotation_content = """<annotation>
    <folder>VOC2012</folder>
    <filename>dummy.jpg</filename>
    <source>
        <database>The VOC2012 Database</database>
        <annotation>PASCAL VOC2012</annotation>
    </source>
    <size>
        <width>500</width>
        <height>375</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>300</xmax>
            <ymax>300</ymax>
        </bndbox>
    </object>
</annotation>"""
    
    with open(os.path.join(voc2012_path, 'Annotations', 'dummy.xml'), 'w') as f:
        f.write(annotation_content)
    
    # Create dummy image file (1x1 pixel black image)
    dummy_image = bytes([0, 0, 0] * 375 * 500)  # RGB black image data
    with open(os.path.join(voc2012_path, 'JPEGImages', 'dummy.jpg'), 'wb') as f:
        # Write a minimal JPEG header and data
        # This is a very simplified version and won't be a valid JPEG
        # but will serve as a placeholder
        f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
        f.write(dummy_image[:1000])  # Just write some bytes
    
    # Create train.txt and val.txt
    with open(os.path.join(voc2012_path, 'ImageSets', 'Main', 'train.txt'), 'w') as f:
        f.write('dummy')
    
    with open(os.path.join(voc2012_path, 'ImageSets', 'Main', 'val.txt'), 'w') as f:
        f.write('dummy')
    
    print("Dummy dataset created successfully!")
    return True

if __name__ == "__main__":
    print("Pascal VOC Dataset Setup (Robust Version)")
    print("=======================================")
    
    # Check if dataset is already set up
    if check_voc_dataset():
        print("\nPascal VOC dataset is already properly set up.")
    else:
        print("\nSetting up Pascal VOC dataset...")
        success = setup_voc_dataset()
        
        # Verify setup was successful
        print("\nVerifying Pascal VOC dataset setup:")
        if not check_voc_dataset() and not success:
            print("\nFailed to set up the dataset from downloads. Creating a minimal dummy dataset for testing...")
            create_dummy_dataset()
            print("\nVerifying dummy dataset:")
            check_voc_dataset()
    
    print("\nUsage example:")
    print("from dataset.pascal_voc import PascalVOCDataset")
    print("train_dataset = PascalVOCDataset(root_dir='data/VOCdevkit', split='train', year='2012', is_train=True)")
    print("val_dataset = PascalVOCDataset(root_dir='data/VOCdevkit', split='val', year='2012', is_train=False)")