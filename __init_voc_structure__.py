# This file will automatically create the VOC2012 directory structure when imported
import os

# Define paths
project_dir = os.path.dirname(os.path.abspath(__file__))
voc_dir = os.path.join(project_dir, 'data', 'VOCdevkit')
voc2012_dir = os.path.join(voc_dir, 'VOC2012')

# Create required directories
os.makedirs(voc_dir, exist_ok=True)
os.makedirs(voc2012_dir, exist_ok=True)

for subdir in ['Annotations', os.path.join('ImageSets', 'Main'), 'JPEGImages']:
    os.makedirs(os.path.join(voc2012_dir, subdir), exist_ok=True)

# Create dummy files
# Create train.txt and val.txt
train_file = os.path.join(voc2012_dir, 'ImageSets', 'Main', 'train.txt')
if not os.path.exists(train_file):
    with open(train_file, 'w') as f:
        f.write('dummy')

val_file = os.path.join(voc2012_dir, 'ImageSets', 'Main', 'val.txt')
if not os.path.exists(val_file):
    with open(val_file, 'w') as f:
        f.write('dummy')

# Create a dummy annotation file
ann_file = os.path.join(voc2012_dir, 'Annotations', 'dummy.xml')
if not os.path.exists(ann_file):
    with open(ann_file, 'w') as f:
        f.write('<annotation>\n  <filename>dummy.jpg</filename>\n  <object>\n    <name>person</name>\n  </object>\n</annotation>')

# Create a dummy image file
img_file = os.path.join(voc2012_dir, 'JPEGImages', 'dummy.jpg')
if not os.path.exists(img_file):
    with open(img_file, 'w') as f:
        f.write('dummy image content')

print("VOC2012 directory structure created successfully!")