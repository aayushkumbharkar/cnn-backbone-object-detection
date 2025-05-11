import os
import argparse
import torch
import numpy as np
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
from utils.metrics import calculate_map, calculate_precision_recall_curve
from utils.visualization import visualize_predictions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/VOCdevkit',
                        help='Path to VOC dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to evaluate on (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate on (test or val)')
    parser.add_argument('--year', type=str, default='2012',
                        help='VOC dataset year')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize detections on sample images')
    parser.add_argument('--num-vis-samples', type=int, default=5,
                        help='Number of samples to visualize')
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

def evaluate(model, dataloader, device, class_names, output_dir=None, visualize=False, num_vis_samples=5):
    """Evaluate model on dataset.
    
    Args:
        model (nn.Module): The detection model
        dataloader (DataLoader): Evaluation data loader
        device (torch.device): Device to evaluate on
        class_names (list): List of class names
        output_dir (str, optional): Directory to save results
        visualize (bool): Whether to visualize detections
        num_vis_samples (int): Number of samples to visualize
        
    Returns:
        dict: Evaluation results
    """
    model.eval()
    all_detections = []
    all_ground_truth = []
    
    # For visualization
    vis_images = []
    vis_detections = []
    vis_ground_truth = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
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
            for j, detection in enumerate(detections):
                # Get ground truth for this image
                gt_boxes = targets[j]['boxes'].cpu().numpy()
                gt_labels = targets[j]['labels'].cpu().numpy()
                
                # Format ground truth
                gt_formatted = [
                    [label, *box]
                    for label, box in zip(gt_labels, gt_boxes)
                ]
                
                all_detections.append(detection)
                all_ground_truth.append(gt_formatted)
                
                # Save some samples for visualization
                if visualize and len(vis_images) < num_vis_samples and i % (len(dataloader) // num_vis_samples) == 0:
                    vis_images.append(images[j].cpu())
                    vis_detections.append(detection)
                    vis_ground_truth.append(gt_formatted)
    
    # Calculate mAP
    num_classes = len(class_names)
    mAP = calculate_map(all_detections, all_ground_truth, num_classes)
    
    # Calculate per-class AP
    per_class_ap = calculate_map(all_detections, all_ground_truth, num_classes, return_per_class=True)
    
    # Calculate precision-recall curve
    precisions, recalls, average_precision = calculate_precision_recall_curve(
        all_detections, all_ground_truth, num_classes
    )
    
    # Visualize results if requested
    if visualize and output_dir:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        for i, (image, detection, gt) in enumerate(zip(vis_images, vis_detections, vis_ground_truth)):
            # Visualize detections
            fig, _ = visualize_predictions(model, image, device, class_names=class_names)
            fig.savefig(os.path.join(output_dir, 'visualizations', f'detection_{i}.png'))
            plt.close(fig)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        for i in range(1, num_classes):  # Skip background class
            plt.plot(recalls[i], precisions[i], label=f'{class_names[i]} (AP={per_class_ap[i]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
    
    # Prepare results
    results = {
        'mAP': mAP,
        'per_class_ap': {class_names[i]: per_class_ap[i] for i in range(num_classes)},
        'precisions': precisions,
        'recalls': recalls,
        'average_precision': average_precision
    }
    
    return results

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = PascalVOCDataset(
        root_dir=args.data_dir,
        split=args.split,
        year=args.year,
        is_train=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )
    
    # Get class names
    class_names = dataset.get_class_names()
    
    # Create model
    num_classes = dataset.num_classes
    model = create_model(config, num_classes)
    model.to(device)
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"No checkpoint found at {args.checkpoint}")
    
    # Evaluate model
    results = evaluate(
        model, dataloader, device, class_names, 
        output_dir=args.output_dir, 
        visualize=args.visualize,
        num_vis_samples=args.num_vis_samples
    )
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"mAP: {results['mAP']:.4f}")
    print("\nPer-class AP:")
    for class_name, ap in results['per_class_ap'].items():
        if class_name != 'background':  # Skip background class
            print(f"{class_name}: {ap:.4f}")
    
    # Save results
    if args.output_dir:
        import json
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'mAP': float(results['mAP']),
            'per_class_ap': {k: float(v) for k, v in results['per_class_ap'].items()}
        }
        
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()