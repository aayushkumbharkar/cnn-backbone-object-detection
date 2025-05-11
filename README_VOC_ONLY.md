# Pascal VOC Object Detection Training Guide

## Overview
This guide provides instructions for setting up and training an object detection model using the Pascal VOC 2012 dataset. Since the COCO dataset could not be downloaded due to permission issues, this guide focuses exclusively on using the Pascal VOC dataset.

## Project Structure

The project contains the following key files:

- `setup_voc_dataset.py` - Script to copy and set up the Pascal VOC dataset
- `test_dataset_paths.py` - Script to verify dataset structure
- `config_voc_only.py` - Configuration for training with Pascal VOC
- `train_voc_only.py` - Training script for Pascal VOC dataset
- `inference_voc.py` - Script for running inference with trained model

## Setup Instructions

### 1. Dataset Setup
The Pascal VOC dataset has been downloaded to your cache directory. To set it up properly for training:

```bash
python setup_voc_dataset.py
```

This script will:
- Copy the Pascal VOC dataset from your cache directory to the project's data directory
- Verify that all necessary files and directories are present
- Provide usage examples for the dataset

### 2. Verify Dataset Structure
You can verify the dataset structure using:

```bash
python test_dataset_paths.py
```

This will check if the Pascal VOC dataset is properly set up in the expected directory structure.

## Training the Model

### Configuration
The training configuration is defined in `config_voc_only.py`. You can modify this file to adjust:
- Batch size
- Learning rate
- Number of epochs
- Model backbone
- Output directories

### Start Training
To start training the object detection model using the Pascal VOC dataset:

```bash
python train_voc_only.py
```

The training script will:
1. Check if the Pascal VOC dataset is properly set up
2. Create the necessary output directories
3. Load the dataset and create data loaders
4. Initialize the Faster R-CNN model with a ResNet-50 backbone
5. Train the model for the specified number of epochs
6. Save checkpoints and the best model based on validation loss

## Model Checkpoints
Model checkpoints will be saved to the `checkpoints/voc_only` directory. The best model (based on validation loss) will be saved as `best_model.pth`.

## Using the Trained Model

After training, you can use the trained model for inference on new images using the provided inference script:

```bash
python inference_voc.py
```

The inference script will:
1. Load the best model from the checkpoints directory
2. Prompt you to enter a path to an image for detection
3. If no path is provided, it will attempt to use a sample image from the VOC dataset
4. Perform object detection on the image
5. Display and save the visualization with bounding boxes and class labels

The detection results will be saved to `output/detections/detection_result.png`.

## Improving Performance
To improve model performance, you can:
- Adjust the learning rate and training schedule in `config_voc_only.py`
- Try different backbones (ResNet-18, ResNet-34, ResNet-101)
- Apply additional data augmentation techniques
- Increase the number of training epochs
- Experiment with different batch sizes

## Getting COCO Dataset
If you need to use the COCO dataset in the future, you'll need to:
1. Log in to Kaggle and accept the dataset terms
2. Set up your Kaggle API credentials
3. Run the download_datasets.py script again