import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        print("\nYou can try installing them manually with:")
        print("pip install -r requirements.txt")
        return False

def check_dependencies():
    """Check if critical dependencies are installed"""
    missing_deps = []
    
    # Check for torch and torchvision
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        missing_deps.append("torch and torchvision")
    
    # Check for albumentations
    try:
        import albumentations
        print(f"Albumentations version: {albumentations.__version__}")
    except ImportError:
        missing_deps.append("albumentations")
    
    # Check for pycocotools (optional for VOC-only training)
    try:
        import pycocotools
        print(f"Pycocotools version: {pycocotools.__version__}")
        has_pycocotools = True
    except ImportError:
        print("Warning: pycocotools not found. COCO dataset functionality will be disabled.")
        print("This is OK if you're only using Pascal VOC dataset.")
        has_pycocotools = False
    
    if missing_deps:
        print(f"\nMissing critical dependencies: {', '.join(missing_deps)}")
        print("Please install them before proceeding.")
        return False
    
    return True

def main():
    print("CNN Backbone Environment Setup")
    print("=============================")
    
    # Check if dependencies are already installed
    print("\nChecking dependencies...")
    if check_dependencies():
        print("\nAll critical dependencies are installed!")
    else:
        # Ask user if they want to install dependencies
        response = input("\nWould you like to install the missing dependencies now? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
            # Check again after installation
            print("\nVerifying installation...")
            check_dependencies()
    
    print("\nSetup complete!")
    print("\nTo train with Pascal VOC dataset only:")
    print("python train_voc_only.py")

if __name__ == "__main__":
    main()