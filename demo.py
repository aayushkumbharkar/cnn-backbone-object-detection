import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import yaml
from tqdm import tqdm

# Import project modules
from backbone.resnet import ResNet50Backbone
from backbone.mobilenet import MobileNetV2Backbone
from detection.ssd import SSDDetector
from detection.yolo import YOLODetector
from dataset.pascal_voc import PascalVOCDataset
from utils.visualization import visualize_predictions, draw_boxes_on_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Demo object detection model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output-dir', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for detections')
    parser.add_argument('--data-dir', type=str, default='data/VOCdevkit',
                        help='Path to VOC dataset (for class names)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save video if input is a video file')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam as input')
    parser.add_argument('--webcam-id', type=int, default=0,
                        help='Webcam device ID')
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

def process_image(model, image_path, device, class_names, score_threshold=0.5):
    """Process a single image and return visualization.
    
    Args:
        model (nn.Module): The detection model
        image_path (str): Path to the image
        device (torch.device): Device to run inference on
        class_names (list): List of class names
        score_threshold (float): Score threshold for detections
        
    Returns:
        tuple: (figure, detections, original_image)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Run prediction and visualization
    fig, detections = visualize_predictions(
        model, image, device, class_names=class_names, score_threshold=score_threshold
    )
    
    return fig, detections, image

def process_video(model, video_path, output_path, device, class_names, score_threshold=0.5):
    """Process a video and save the output with detections.
    
    Args:
        model (nn.Module): The detection model
        video_path (str): Path to the video
        output_path (str): Path to save the output video
        device (torch.device): Device to run inference on
        class_names (list): List of class names
        score_threshold (float): Score threshold for detections
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Apply transform and move to device
            image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Run inference
            if isinstance(model, SSDDetector):
                # SSD predictions
                class_preds, loc_preds, default_boxes = model(image_tensor)
                detections = model.detect_objects(class_preds, loc_preds, default_boxes)
            elif isinstance(model, YOLODetector):
                # YOLO predictions
                predictions = model(image_tensor)
                input_size = image_tensor.shape[2:4]  # (H, W)
                detections = model.detect_objects(predictions, input_size)
            else:
                raise ValueError("Unsupported model type")
            
            # Draw detections on frame
            frame_with_boxes = draw_boxes_on_image(frame, detections[0], class_names, score_threshold)
            
            # Write frame to output video
            out.write(frame_with_boxes)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

def process_webcam(model, device, class_names, score_threshold=0.5, webcam_id=0):
    """Run object detection on webcam feed.
    
    Args:
        model (nn.Module): The detection model
        device (torch.device): Device to run inference on
        class_names (list): List of class names
        score_threshold (float): Score threshold for detections
        webcam_id (int): Webcam device ID
    """
    # Open webcam
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_id}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    print("Press 'q' to quit")
    
    with torch.no_grad():
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Apply transform and move to device
            image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Run inference
            if isinstance(model, SSDDetector):
                # SSD predictions
                class_preds, loc_preds, default_boxes = model(image_tensor)
                detections = model.detect_objects(class_preds, loc_preds, default_boxes)
            elif isinstance(model, YOLODetector):
                # YOLO predictions
                predictions = model(image_tensor)
                input_size = image_tensor.shape[2:4]  # (H, W)
                detections = model.detect_objects(predictions, input_size)
            else:
                raise ValueError("Unsupported model type")
            
            # Draw detections on frame
            frame_with_boxes = draw_boxes_on_image(frame, detections[0], class_names, score_threshold)
            
            # Display the frame
            cv2.imshow('Object Detection', frame_with_boxes)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main demo function."""
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
    
    # Get class names from dataset
    try:
        dataset = PascalVOCDataset(
            root_dir=args.data_dir,
            split='val',
            year='2012',
            is_train=False
        )
        class_names = dataset.get_class_names()
        num_classes = dataset.num_classes
    except Exception as e:
        print(f"Warning: Could not load dataset for class names: {e}")
        print("Using default class names")
        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        num_classes = len(class_names)
    
    # Create model
    model = create_model(config, num_classes)
    model.to(device)
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"No checkpoint found at {args.checkpoint}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Process input based on type
    if args.webcam:
        # Process webcam feed
        process_webcam(model, device, class_names, args.score_threshold, args.webcam_id)
    elif os.path.isdir(args.input):
        # Process directory of images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(args.input) 
                      if os.path.isfile(os.path.join(args.input, f)) and 
                      f.lower().endswith(image_extensions)]
        
        print(f"Found {len(image_files)} images in {args.input}")
        
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(args.input, image_file)
            fig, _, _ = process_image(model, image_path, device, class_names, args.score_threshold)
            
            # Save figure
            output_path = os.path.join(args.output_dir, f"detection_{os.path.basename(image_file)}")
            fig.savefig(output_path)
            plt.close(fig)
        
        print(f"Results saved to {args.output_dir}")
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        if args.save_video:
            output_path = os.path.join(args.output_dir, f"detection_{os.path.basename(args.input)}")
            process_video(model, args.input, output_path, device, class_names, args.score_threshold)
        else:
            print("Use --save-video flag to save the processed video")
    else:
        # Process single image
        fig, detections, _ = process_image(model, args.input, device, class_names, args.score_threshold)
        
        # Print detections
        print("\nDetections:")
        for det in detections[0]:
            class_id, confidence, x1, y1, x2, y2 = det
            class_name = class_names[int(class_id)]
            print(f"{class_name}: {confidence:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # Save figure
        output_path = os.path.join(args.output_dir, f"detection_{os.path.basename(args.input)}")
        fig.savefig(output_path)
        plt.close(fig)
        
        print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()