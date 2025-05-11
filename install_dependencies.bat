@echo off
echo CNN Backbone - Dependency Installer
echo ==================================

echo Installing required dependencies...

pip install torch>=1.7.0 torchvision>=0.8.0 numpy>=1.19.0,<1.21.0 pillow>=8.0.0 matplotlib>=3.3.0 tqdm>=4.50.0 pyyaml>=5.3.0 pycocotools>=2.0.2 opencv-python>=4.4.0 albumentations>=0.5.2,<1.0.0 kagglehub>=0.2.0 pandas>=1.1.0 seaborn>=0.11.0 scikit-learn>=0.24.0

echo.
echo Dependencies installation completed.
echo You can now run the training script with: python train_voc_only.py
echo.
pause