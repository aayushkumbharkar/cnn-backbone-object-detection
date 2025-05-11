import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
import cv2

def visualize_detections(image, detections, class_names, score_threshold=0.5, figsize=(12, 12)):
    """Visualize object detections on an image.
    
    Args:
        image (PIL.Image or numpy.ndarray): The input image
        detections (list): List of detections, each is [class_id, confidence, x1, y1, x2, y2]
        class_names (list): List of class names
        score_threshold (float): Threshold for showing detections
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure with detections
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert torch tensor to numpy array if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize if image is normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Define colors for each class
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    # Draw each detection
    for detection in detections:
        class_id, confidence, x1, y1, x2, y2 = detection
        
        # Skip low confidence detections
        if confidence < score_threshold:
            continue
        
        # Get class name and color
        class_name = class_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]
        
        # Create rectangle patch
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        
        # Add rectangle to the image
        ax.add_patch(rect)
        
        # Add label
        label = f'{class_name}: {confidence:.2f}'
        plt.text(x1, y1, label, fontsize=10, bbox=dict(facecolor=color, alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def visualize_predictions(model, image, device, transform=None, class_names=None, score_threshold=0.5, figsize=(12, 12)):
    """Run model prediction on an image and visualize the results.
    
    Args:
        model: The object detection model
        image (PIL.Image or numpy.ndarray): The input image
        device (torch.device): Device to run inference on
        transform (callable, optional): Transform to apply to the image
        class_names (list, optional): List of class names
        score_threshold (float): Threshold for showing detections
        figsize (tuple): Figure size
        
    Returns:
        tuple: (figure, detections)
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Make a copy of the original image for visualization
    orig_image = image.copy()
    
    # Apply transform if provided
    if transform is not None:
        image_tensor = transform(image)
    else:
        # Basic transform: convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    with torch.no_grad():
        if hasattr(model, 'detect_objects'):
            # For SSD and YOLO detectors
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'forward'):
                # Get features from backbone
                features = model.backbone(image_tensor)
                
                # Forward pass through detector
                if hasattr(model, 'forward'):
                    predictions = model(image_tensor)
                    
                    # Get detections
                    if isinstance(predictions, tuple) and len(predictions) == 3:
                        # SSD format: (class_preds, loc_preds, default_boxes)
                        class_preds, loc_preds, default_boxes = predictions
                        detections = model.detect_objects(class_preds, loc_preds, default_boxes)
                    elif isinstance(predictions, list):
                        # YOLO format: list of detection outputs at different scales
                        input_size = image_tensor.shape[2:4]  # (H, W)
                        detections = model.detect_objects(predictions, input_size)
                    else:
                        raise ValueError("Unsupported prediction format")
            else:
                # Direct detection
                detections = model.detect_objects(image_tensor)
        else:
            # Generic model with different output format
            outputs = model(image_tensor)
            detections = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            
            # Convert to common format
            detections = [
                [label, score, *box]
                for label, score, box in zip(labels, scores, detections)
                if score >= score_threshold
            ]
    
    # Get class names if not provided
    if class_names is None:
        if hasattr(model, 'num_classes'):
            class_names = [f'Class {i}' for i in range(model.num_classes)]
        else:
            class_names = [f'Class {i}' for i in range(100)]  # Arbitrary large number
    
    # Visualize detections
    fig = visualize_detections(orig_image, detections[0] if isinstance(detections, list) and len(detections) > 0 else detections, 
                              class_names, score_threshold, figsize)
    
    return fig, detections

def draw_boxes_on_image(image, detections, class_names, score_threshold=0.5):
    """Draw bounding boxes directly on an image.
    
    Args:
        image (numpy.ndarray): The input image (BGR format for OpenCV)
        detections (list): List of detections, each is [class_id, confidence, x1, y1, x2, y2]
        class_names (list): List of class names
        score_threshold (float): Threshold for showing detections
        
    Returns:
        numpy.ndarray: Image with drawn bounding boxes
    """
    # Make a copy of the image
    image_with_boxes = image.copy()
    
    # Define colors for each class (BGR format for OpenCV)
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    colors = (colors[:, :3] * 255).astype(np.uint8)[:, ::-1]  # Convert to BGR
    
    # Draw each detection
    for detection in detections:
        class_id, confidence, x1, y1, x2, y2 = detection
        
        # Skip low confidence detections
        if confidence < score_threshold:
            continue
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class name and color
        class_name = class_names[int(class_id)]
        color = colors[int(class_id) % len(colors)].tolist()
        
        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f'{class_name}: {confidence:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_with_boxes, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_with_boxes