import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import project modules
from backbone.resnet import ResNet50Backbone
from backbone.mobilenet import MobileNetV2Backbone
from detection.ssd import SSDDetector
from detection.yolo import YOLODetector
from dataset.pascal_voc import PascalVOCDataset
from utils.metrics import calculate_map

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='configs/train_ssd_resnet.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/VOCdevkit',
                        help='Path to VOC dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config, num_classes):
    """Create model based on configuration.
    
    Args:
        config (dict): Model configuration
        num_classes (int): Number of classes in the dataset
        
    Returns:
        nn.Module: The detection model
    """
    # Create backbone
    backbone_type = config['backbone']['type']
    backbone_args = config['backbone'].get('args', {})
    
    if backbone_type == 'resnet50':
        backbone = ResNet50Backbone(**backbone_args)
    elif backbone_type == 'mobilenetv2':
        backbone = MobileNetV2Backbone(**backbone_args)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    # Create detector
    detector_type = config['detector']['type']
    detector_args = config['detector'].get('args', {})
    
    if detector_type == 'ssd':
        model = SSDDetector(backbone, num_classes, **detector_args)
    elif detector_type == 'yolo':
        model = YOLODetector(backbone, num_classes, **detector_args)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch.
    
    Args:
        model (nn.Module): The detection model
        dataloader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion (callable): Loss function
        device (torch.device): Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for images, targets in progress_bar:
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        optimizer.zero_grad()
        
        # Different handling based on detector type
        if isinstance(model, SSDDetector):
            # SSD forward pass
            class_preds, loc_preds, default_boxes = model(images)
            loss = criterion(class_preds, loc_preds, targets, default_boxes)
        elif isinstance(model, YOLODetector):
            # YOLO forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)
        else:
            raise ValueError("Unsupported model type")
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate model on validation set.
    
    Args:
        model (nn.Module): The detection model
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to validate on
        
    Returns:
        float: mAP score
    """
    model.eval()
    all_detections = []
    all_ground_truth = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            
            # Get predictions
            if isinstance(model, SSDDetector):
                # SSD predictions
                class_preds, loc_preds, default_boxes = model(images)
                detections = model.detect_objects(class_preds, loc_preds, default_boxes)
            elif isinstance(model, YOLODetector):
                # YOLO predictions
                predictions = model(images)
                input_size = images.shape[2:4]  # (H, W)
                detections = model.detect_objects(predictions, input_size)
            else:
                raise ValueError("Unsupported model type")
            
            # Convert detections to format expected by mAP calculation
            for i, detection in enumerate(detections):
                # Get ground truth for this image
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                # Format ground truth
                gt_formatted = [
                    [label, *box]
                    for label, box in zip(gt_labels, gt_boxes)
                ]
                
                all_detections.append(detection)
                all_ground_truth.append(gt_formatted)
    
    # Calculate mAP
    num_classes = len(dataloader.dataset.get_class_names())
    mAP = calculate_map(all_detections, all_ground_truth, num_classes)
    
    return mAP

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = PascalVOCDataset(
        root_dir=args.data_dir,
        split='train',
        year='2012',
        is_train=True
    )
    
    val_dataset = PascalVOCDataset(
        root_dir=args.data_dir,
        split='val',
        year='2012',
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    num_classes = train_dataset.num_classes
    model = create_model(config, num_classes)
    model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # Create loss function based on detector type
    if isinstance(model, SSDDetector):
        from detection.ssd import SSDLoss
        criterion = SSDLoss(num_classes)
    elif isinstance(model, YOLODetector):
        from detection.yolo import YOLOLoss
        criterion = YOLOLoss(model.anchors, num_classes, device)
    else:
        raise ValueError("Unsupported model type")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_map = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint.get('best_map', 0)
            print(f"Loaded checkpoint from epoch {start_epoch-1}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    train_losses = []
    val_maps = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_map = validate(model, val_loader, device)
        val_maps.append(val_map)
        
        print(f"Train Loss: {train_loss:.4f}, Validation mAP: {val_map:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_map)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_map': val_map,
            'best_map': best_map
        }, checkpoint_path)
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_map': val_map,
                'best_map': best_map
            }, best_model_path)
            print(f"Saved best model with mAP: {best_map:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_maps)
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    print(f"\nTraining completed. Best mAP: {best_map:.4f}")

if __name__ == "__main__":
    main()