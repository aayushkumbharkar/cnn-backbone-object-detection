import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Import dataset classes
from dataset.coco import COCODataset
from dataset.pascal_voc import PascalVOCDataset

# Import model (assuming a simple model for demonstration)
from detection.ssd import SSD
from backbone.resnet import ResNet50

def train_model_example(dataset_type='pascal_voc', num_epochs=1):
    """Example of training a model with the downloaded datasets.
    
    Args:
        dataset_type (str): Type of dataset to use ('coco' or 'pascal_voc')
        num_epochs (int): Number of epochs to train for
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset based on type
    if dataset_type.lower() == 'coco':
        # Check if COCO dataset exists
        coco_path = os.path.join('data', 'coco')
        if not os.path.exists(coco_path):
            print("COCO dataset not found. Please run download_datasets.py first.")
            return
        
        print("Loading COCO dataset...")
        train_dataset = COCODataset(
            root_dir=coco_path,
            ann_file=os.path.join(coco_path, 'annotations', 'instances_train2017.json'),
            is_train=True
        )
        num_classes = train_dataset.num_classes
        print(f"Number of classes: {num_classes}")
        
    elif dataset_type.lower() == 'pascal_voc':
        # Check if Pascal VOC dataset exists
        voc_path = os.path.join('data', 'VOCdevkit')
        if not os.path.exists(voc_path):
            print("Pascal VOC dataset not found. Please run download_datasets.py first.")
            return
        
        print("Loading Pascal VOC dataset...")
        train_dataset = PascalVOCDataset(
            root_dir=voc_path,
            split='train',
            year='2012',
            is_train=True
        )
        num_classes = train_dataset.num_classes
        print(f"Number of classes: {num_classes}")
    
    else:
        print(f"Unknown dataset type: {dataset_type}")
        print("Supported types: 'coco', 'pascal_voc'")
        return
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0  # Set to higher value if using GPU
    )
    
    # Initialize model (this is just a placeholder, adjust according to your actual model)
    try:
        backbone = ResNet50(pretrained=True)
        model = SSD(backbone=backbone, num_classes=num_classes)
        model = model.to(device)
        
        # Define optimizer
        optimizer = Adam(model.parameters(), lr=0.001)
        
        # Training loop
        print(f"\nStarting training for {num_epochs} epochs...")
        model.train()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0
            
            # Use tqdm for progress bar
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                # Move data to device
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
        
        print("Training complete!")
        
    except Exception as e:
        print(f"Error during training: {e}")

def main():
    print("CNN Backbone - Dataset Training Example")
    print("=====================================")
    
    # Check if datasets exist
    if not os.path.exists(os.path.join('data', 'coco')) and not os.path.exists(os.path.join('data', 'VOCdevkit')):
        print("Datasets not found. Please run download_datasets.py first.")
        print("Command: python download_datasets.py")
        return
    
    # Ask user which dataset to use
    print("\nAvailable datasets:")
    print("1. COCO")
    print("2. Pascal VOC")
    
    try:
        choice = input("\nSelect dataset (1 or 2): ")
        if choice == '1':
            train_model_example('coco')
        elif choice == '2':
            train_model_example('pascal_voc')
        else:
            print("Invalid choice. Please enter 1 or 2.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

if __name__ == "__main__":
    main()