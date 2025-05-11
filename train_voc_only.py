import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import datetime
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also add the dataset directory explicitly
dataset_dir = os.path.join(project_root, 'dataset')
if os.path.exists(dataset_dir) and dataset_dir not in sys.path:
    sys.path.insert(0, dataset_dir)

# Import configuration
from config_voc_only import config

# Check if dataset exists
def check_voc_dataset():
    """Check if the Pascal VOC dataset is properly set up."""
    voc_path = os.path.join('data', 'VOCdevkit')
    voc2012_path = os.path.join(voc_path, 'VOC2012')
    
    if not os.path.exists(voc_path):
        print(f"Pascal VOC dataset directory not found at: {os.path.abspath(voc_path)}")
        return False
    
    if not os.path.exists(voc2012_path):
        print(f"VOC2012 directory not found at: {os.path.abspath(voc2012_path)}")
        return False
    
    # Check key subdirectories
    required_dirs = ['Annotations', 'ImageSets', 'JPEGImages']
    for subdir in required_dirs:
        path = os.path.join(voc2012_path, subdir)
        if not os.path.exists(path):
            print(f"{subdir} directory not found at: {os.path.abspath(path)}")
            return False
    
    return True

# Import dataset class
try:
    # First try direct import
    from dataset.pascal_voc import PascalVOCDataset
except ImportError as e:
    print(f"Error importing PascalVOCDataset: {e}")
    print("Attempting alternative import methods...")
    
    try:
        # Try importing from the dataset module directly
        import dataset
        from dataset.pascal_voc import PascalVOCDataset
        print("Successfully imported PascalVOCDataset using alternative method")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("Make sure the dataset module is in your path.")
        print(f"Current Python path: {sys.path}")
        print(f"Dataset directory exists: {os.path.exists(os.path.join(project_root, 'dataset'))}")
        if os.path.exists(os.path.join(project_root, 'dataset')):
            print(f"Files in dataset directory: {os.listdir(os.path.join(project_root, 'dataset'))}")
        sys.exit(1)

# Create model
def get_model(num_classes):
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    start_time = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Batch [{i}/{len(data_loader)}], Loss: {losses.item():.4f}, Time: {elapsed:.2f}s")
            start_time = time.time()
    
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
    
    return total_loss / len(data_loader)

# Main training loop
def main():
    print("Pascal VOC Object Detection Training")
    print("==================================")
    
    # Check if dataset exists
    if not check_voc_dataset():
        print("\nError: Pascal VOC dataset is not properly set up.")
        print("Please run setup_voc_dataset.py to set up the dataset.")
        return
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading Pascal VOC dataset...")
    train_dataset = PascalVOCDataset(
        root_dir=config['dataset']['root_dir'],
        split=config['dataset']['train_split'],
        year=config['dataset']['year'],
        is_train=True
    )
    
    val_dataset = PascalVOCDataset(
        root_dir=config['dataset']['root_dir'],
        split=config['dataset']['val_split'],
        year=config['dataset']['year'],
        is_train=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config['dataset']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=config['dataset']['num_workers']
    )
    
    # Create model
    print("\nCreating model...")
    model = get_model(config['model']['num_classes'] + 1)  # +1 for background class
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    if config['training']['lr_scheduler'] == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_gamma']
        )
    elif config['training']['lr_scheduler'] == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate model
        if (epoch + 1) % config['evaluation']['eval_frequency'] == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model checkpoints saved to: {os.path.abspath(config['paths']['checkpoint_dir'])}")

if __name__ == "__main__":
    main()