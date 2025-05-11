import os
import sys

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
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(voc_dir):
        level = root.replace(project_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

if __name__ == "__main__":
    create_voc_structure()
    print("\nRun 'python test_dataset_paths.py' to verify the structure.")