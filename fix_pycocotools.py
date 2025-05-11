import subprocess
import sys
import os
import shutil

def install_pycocotools():
    """Attempt to install pycocotools package"""
    print("Attempting to install pycocotools...")
    try:
        # Check if Visual C++ Build Tools are installed
        print("Checking for Microsoft Visual C++ Build Tools...")
        try:
            # Try to compile a simple C extension to test if build tools are available
            from setuptools import Extension, setup
            from setuptools.command.build_ext import build_ext
            test_ext = Extension("test_build", sources=[])
            setup(name="test_build", ext_modules=[test_ext], script_args=["build_ext"])
            print("Microsoft Visual C++ Build Tools are installed.")
        except Exception:
            print("Microsoft Visual C++ Build Tools are not installed or not properly configured.")
            print("\nTo install pycocotools, you need Microsoft Visual C++ Build Tools.")
            print("Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            return False
            
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools"])
        print("pycocotools installed successfully!")
        return True
    except Exception as e:
        print(f"Error installing pycocotools: {e}")
        print("\nYou can try installing it manually with:")
        print("pip install pycocotools")
        return False

def patch_dataset_module():
    """Patch the dataset module to work without pycocotools"""
    print("Patching dataset module to work without pycocotools...")
    
    # Backup the original __init__.py file
    init_path = os.path.join('dataset', '__init__.py')
    backup_path = os.path.join('dataset', '__init__.py.bak')
    
    if not os.path.exists(backup_path) and os.path.exists(init_path):
        shutil.copy2(init_path, backup_path)
        print(f"Backed up original {init_path} to {backup_path}")
    
    # Update the __init__.py file to handle missing pycocotools gracefully
    with open(init_path, 'w') as f:
        f.write("""# Dataset module initialization

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
""")
    
    print("Dataset module patched successfully!")
    return True

def main():
    print("CNN Backbone - Fix Missing pycocotools")
    print("====================================")
    
    # Check if pycocotools is already installed
    try:
        import pycocotools
        print(f"pycocotools is already installed (version: {pycocotools.__version__})")
        print("You should now be able to run all training scripts without errors.")
    except ImportError:
        print("pycocotools is not installed. This is required for the COCO dataset.")
        print("However, you can still train with the Pascal VOC dataset without it.")
        
        # Patch the dataset module to work without pycocotools
        patch_dataset_module()
        
        # Ask user if they want to install pycocotools
        response = input("\nWould you like to try installing pycocotools now? (y/n): ")
        if response.lower() == 'y':
            if install_pycocotools():
                print("\nYou should now be able to run all training scripts without errors.")
            else:
                print("\nFailed to install pycocotools, but you can still train with the VOC dataset.")
                print("Use train_voc_only.py which has been configured to work without pycocotools.")
        else:
            print("\nYou can still train with the VOC dataset by using train_voc_only.py")
            print("The code has been updated to handle missing pycocotools for VOC-only training.")
    
    print("\nTo train with Pascal VOC dataset:")
    print("python train_voc_only.py")

if __name__ == "__main__":
    main()