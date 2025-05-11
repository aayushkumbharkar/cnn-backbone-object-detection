import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import transforms

# Import configuration
from config_voc_only import config

# Pascal VOC class names
VOC_CLASSES = [
    'background',  # Always include background as class 0
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]

# Create model
def get_model(num_classes):
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Load trained model
def load_model(checkpoint_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint file {checkpoint_path} not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    return model, device

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return image, transform(image)

# Perform inference
def detect_objects(model, image_tensor, device, threshold=0.5):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model([image_tensor])
    
    # Extract predictions
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Filter by threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    return boxes, scores, labels

# Visualize results
def visualize_detection(image, boxes, scores, labels, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Generate random colors for each class
    colors = np.random.rand(len(class_names), 3)
    
    for box, score, label in zip(boxes, scores, labels):
        # Convert box coordinates
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                               edgecolor=colors[label], facecolor='none')
        ax.add_patch(rect)
        
        # Add label and score
        class_name = class_names[label]
        ax.text(x1, y1-10, f"{class_name}: {score:.2f}", 
               color=colors[label], fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs('output/detections', exist_ok=True)
    plt.savefig('output/detections/detection_result.png')
    print(f"Visualization saved to output/detections/detection_result.png")
    
    plt.show()

def main():
    print("Pascal VOC Object Detection Inference")
    print("===================================")
    
    # Path to model checkpoint
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_voc_only.py")
        return
    
    # Load model
    model, device = load_model(checkpoint_path, len(VOC_CLASSES))
    
    # Get image path from user
    image_path = input("\nEnter path to image for detection (or press Enter to use a sample image): ")
    
    # Use a sample image if no path provided
    if not image_path:
        # Check if VOC dataset is available for sample images
        voc_path = os.path.join('data', 'VOCdevkit', 'VOC2012', 'JPEGImages')
        if os.path.exists(voc_path):
            # Get a random image from VOC dataset
            import random
            image_files = [f for f in os.listdir(voc_path) if f.endswith('.jpg')]
            if image_files:
                image_path = os.path.join(voc_path, random.choice(image_files))
                print(f"Using sample image: {image_path}")
            else:
                print("No sample images found in VOC dataset.")
                return
        else:
            print("VOC dataset not found. Please provide an image path.")
            return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Preprocess image
    image, image_tensor = preprocess_image(image_path)
    
    # Perform detection
    print("\nPerforming object detection...")
    boxes, scores, labels = detect_objects(model, image_tensor, device)
    
    # Print results
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_name = VOC_CLASSES[label]
        print(f"  {i+1}. {class_name}: {score:.4f}, Box: {box}")
    
    # Visualize results
    print("\nVisualizing detection results...")
    visualize_detection(image, boxes, scores, labels, VOC_CLASSES)

if __name__ == "__main__":
    main()