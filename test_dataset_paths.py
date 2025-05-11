import os

def create_voc_structure():
    """Create a minimal VOC2012 directory structure with dummy files."""
    # Define paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    voc_dir = os.path.join(project_dir, 'data', 'VOCdevkit')
    voc2012_dir = os.path.join(voc_dir, 'VOC2012')
    
    print(f"Creating VOC2012 directory structure at: {voc2012_dir}")
    
    # Create required directories
    os.makedirs(voc_dir, exist_ok=True)
    os.makedirs(voc2012_dir, exist_ok=True)
    
    for subdir in ['Annotations', os.path.join('ImageSets', 'Main'), 'JPEGImages']:
        os.makedirs(os.path.join(voc2012_dir, subdir), exist_ok=True)
    
    # Create dummy files
    # Create train.txt and val.txt
    with open(os.path.join(voc2012_dir, 'ImageSets', 'Main', 'train.txt'), 'w') as f:
        f.write('dummy')
    
    with open(os.path.join(voc2012_dir, 'ImageSets', 'Main', 'val.txt'), 'w') as f:
        f.write('dummy')
    
    # Create a dummy annotation file
    with open(os.path.join(voc2012_dir, 'Annotations', 'dummy.xml'), 'w') as f:
        f.write('<annotation>\n  <filename>dummy.jpg</filename>\n  <object>\n    <name>person</name>\n  </object>\n</annotation>')
    
    # Create a dummy image file
    with open(os.path.join(voc2012_dir, 'JPEGImages', 'dummy.jpg'), 'w') as f:
        f.write('dummy image content')
    
    print("VOC2012 directory structure created successfully!")

def check_dataset_paths():
    """Check if the dataset paths exist and print their structure."""
    # Define expected paths
    coco_path = os.path.join('data', 'coco')
    voc_path = os.path.join('data', 'VOCdevkit')
    
    # Check COCO dataset
    print("Checking COCO dataset paths:")
    if os.path.exists(coco_path):
        print(f"✓ COCO dataset directory exists at: {os.path.abspath(coco_path)}")
        # Check key subdirectories
        for subdir in ['annotations', 'train2017', 'val2017']:
            path = os.path.join(coco_path, subdir)
            if os.path.exists(path):
                print(f"  ✓ {subdir} directory exists")
                if subdir == 'annotations':
                    for ann_file in ['instances_train2017.json', 'instances_val2017.json']:
                        if os.path.exists(os.path.join(path, ann_file)):
                            print(f"    ✓ {ann_file} exists")
                        else:
                            print(f"    ✗ {ann_file} not found")
            else:
                print(f"  ✗ {subdir} directory not found")
    else:
        print(f"✗ COCO dataset directory not found at: {os.path.abspath(coco_path)}")
        print("  Run 'python download_datasets.py' to download the dataset")
    
    # Check Pascal VOC dataset
    print("\nChecking Pascal VOC dataset paths:")
    if not os.path.exists(voc_path):
        print(f"✗ Pascal VOC dataset directory not found at: {os.path.abspath(voc_path)}")
        print("  Creating minimal VOC dataset structure...")
        create_voc_structure()
    
    # Re-check after potential creation
    if os.path.exists(voc_path):
        print(f"✓ Pascal VOC dataset directory exists at: {os.path.abspath(voc_path)}")
        # Check VOC2012 directory
        voc2012_path = os.path.join(voc_path, 'VOC2012')
        if not os.path.exists(voc2012_path):
            print(f"  ✗ VOC2012 directory not found")
            print("  Creating VOC2012 directory structure...")
            create_voc_structure()
        
        # Re-check VOC2012 after potential creation
        if os.path.exists(voc2012_path):
            print(f"  ✓ VOC2012 directory exists")
            # Check key subdirectories
            for subdir in ['Annotations', 'ImageSets', 'JPEGImages']:
                path = os.path.join(voc2012_path, subdir)
                if os.path.exists(path):
                    print(f"    ✓ {subdir} directory exists")
                    if subdir == 'ImageSets':
                        main_path = os.path.join(path, 'Main')
                        if os.path.exists(main_path):
                            print(f"      ✓ Main directory exists")
                            for split_file in ['train.txt', 'val.txt']:
                                if os.path.exists(os.path.join(main_path, split_file)):
                                    print(f"        ✓ {split_file} exists")
                                else:
                                    print(f"        ✗ {split_file} not found")
                                    # Create missing split file
                                    with open(os.path.join(main_path, split_file), 'w') as f:
                                        f.write('dummy')
                                    print(f"        ✓ {split_file} created")
                        else:
                            print(f"      ✗ Main directory not found")
                            # Create Main directory
                            os.makedirs(main_path, exist_ok=True)
                            print(f"      ✓ Main directory created")
                            # Create split files
                            for split_file in ['train.txt', 'val.txt']:
                                with open(os.path.join(main_path, split_file), 'w') as f:
                                    f.write('dummy')
                                print(f"        ✓ {split_file} created")
                else:
                    print(f"    ✗ {subdir} directory not found")
                    # Create missing subdirectory
                    os.makedirs(path, exist_ok=True)
                    print(f"    ✓ {subdir} directory created")
        else:
            print(f"  ✗ VOC2012 directory not found even after creation attempt")
    else:
        print(f"✗ Pascal VOC dataset directory not found at: {os.path.abspath(voc_path)} even after creation attempt")
        print("  Run 'python download_datasets.py' to download the dataset")

if __name__ == "__main__":
    print("Dataset Path Checker")
    print("====================")
    check_dataset_paths()
    print("\nIf datasets are missing, run: python download_datasets.py")