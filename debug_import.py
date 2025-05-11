import os
import sys

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print(f"Added to Python path: {project_root}")
print(f"Python path: {sys.path}")

# Try to import the dataset module
try:
    from dataset.pascal_voc import PascalVOCDataset
    print("Successfully imported PascalVOCDataset")
except ImportError as e:
    print(f"Error importing PascalVOCDataset: {e}")
    
    # Try alternative import approaches
    print("\nTrying alternative import approaches:")
    
    # Check if dataset directory exists
    dataset_dir = os.path.join(project_root, 'dataset')
    print(f"Dataset directory exists: {os.path.exists(dataset_dir)}")
    
    # List files in dataset directory
    if os.path.exists(dataset_dir):
        print(f"Files in dataset directory: {os.listdir(dataset_dir)}")
    
    # Try direct import
    try:
        import dataset
        print(f"Successfully imported dataset module: {dataset.__file__}")
        print(f"Dataset module contains: {dir(dataset)}")
    except ImportError as e2:
        print(f"Error importing dataset module: {e2}")