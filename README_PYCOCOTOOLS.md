# PyCocoTools Installation Guide and Workaround

## Issue Overview

The `pycocotools` package is a dependency for working with the COCO dataset in this project. However, it requires Microsoft Visual C++ Build Tools to compile its C extensions on Windows, which can cause installation failures if these build tools are not properly installed.

The error typically looks like this:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

## Solutions

### Option 1: Install Microsoft Visual C++ Build Tools

If you want to use the COCO dataset, you'll need to install the Microsoft Visual C++ Build Tools:

1. Download the installer from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer and select "Desktop development with C++"
3. After installation, try installing pycocotools again:
   ```
   pip install pycocotools
   ```

### Option 2: Use Pascal VOC Dataset Only

This project has been configured to work with the Pascal VOC dataset without requiring pycocotools. To use this option:

1. Run the fix script to patch the dataset module:
   ```
   python fix_pycocotools.py
   ```
2. Train using the VOC-only script:
   ```
   python train_voc_only.py
   ```

## How the Workaround Works

The `fix_pycocotools.py` script performs the following actions:

1. Checks if pycocotools is already installed
2. If not installed, it patches the dataset module to work without pycocotools
3. Offers to attempt installation of pycocotools
4. Provides instructions for using the VOC-only training option

The patch modifies the dataset module's `__init__.py` to gracefully handle the missing pycocotools dependency, allowing the Pascal VOC dataset to be used independently.

## Troubleshooting

If you encounter issues with the workaround:

1. Make sure you're using the `train_voc_only.py` script, not the regular `train.py`
2. Verify that the Pascal VOC dataset is properly set up by running `setup_voc_dataset.py`
3. Check that the dataset module has been properly patched by examining `dataset/__init__.py`

For additional help, please open an issue on the project repository.