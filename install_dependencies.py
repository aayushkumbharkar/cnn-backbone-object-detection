import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    print("Installing required dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("Error: requirements.txt not found!")
        return False
    
    try:
        # Install dependencies using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nAll dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError installing dependencies: {e}")
        return False

if __name__ == "__main__":
    print("CNN Backbone - Dependency Installer")
    print("==================================")
    
    success = install_dependencies()
    
    if success:
        print("\nYou can now run the training script with: python train_voc_only.py")
    else:
        print("\nFailed to install dependencies. Please try installing them manually:")
        print("pip install -r requirements.txt")