import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class YOLODetector(nn.Module):
    """YOLO-style detector implementation.
    
    This class implements a YOLO-style detection head that can be attached to various
    CNN backbones for object detection. It follows a simplified approach inspired by
    the YOLO (You Only Look Once) family of detectors.
    """
    
    def __init__(self, backbone, num_classes, anchors=None, num_anchors=3):
        """Initialize the YOLO detector.
        
        Args:
            backbone: CNN backbone for feature extraction
            num_classes (int): Number of object classes (excluding background)
            anchors (list): List of anchor box dimensions (w, h) for each detection scale
            num_anchors (int): Number of anchor boxes per grid cell
        """
        super(YOLODetector, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Default anchors if not provided (these should be tuned for your dataset)
        if anchors is None:
            self.anchors = [
                # Small objects anchors (for high-resolution feature map)
                [(10, 13), (16, 30), (33, 23)],
                # Medium objects anchors (for medium-resolution feature map)
                [(30, 61), (62, 45), (59, 119)],
                # Large objects anchors (for low-resolution feature map)
                [(116, 90), (156, 198), (373, 326)]
            ]
        else:
            self.anchors = anchors
        
        # Get feature channels from backbone
        self.feature_channels = backbone.get_feature_channels()
        
        # Feature map names from backbone to use for detection
        # We'll use 3 scales for detection (small, medium, large objects)
        if hasattr(backbone, 'feature_extraction_points'):
            # For MobileNet
            self.feature_maps = ["layer3", "layer4", "layer5"]
        else:
            # For ResNet
            self.feature_maps = ["layer2", "layer3", "layer4"]
        
        # Create detection heads for each feature map
        self.detection_heads = nn.ModuleDict()
        
        for i, feature_map in enumerate(self.feature_maps):
            # Get number of input channels
            in_channels = self.feature_channels[feature_map]
            
            # Create detection head
            # Each head outputs [num_anchors * (5 + num_classes)] values per grid cell
            # 5 = objectness score + 4 box coordinates (tx, ty, tw, th)
            self.detection_heads[feature_map] = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
                nn.BatchNorm2d(in_channels//2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
                nn.BatchNorm2d(in_channels//2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)
            )
            
            # Initialize weights of the final layer
            self._init_weights(self.detection_heads[feature_map][-1])
    
    def _init_weights(self, layer):
        """Initialize the weights of the final convolutional layer."""
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward pass through the YOLO detector.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            list: List of detection outputs at different scales
                 Each element has shape (B, num_anchors*(5+num_classes), H, W)
        """
        # Get feature maps from backbone
        feature_maps = self.backbone(x)
        
        # Apply detection heads to selected feature maps
        detections = []
        
        for i, feature_map in enumerate(self.feature_maps):
            feat = feature_maps[feature_map]
            detection = self.detection_heads[feature_map](feat)
            detections.append(detection)
        
        return detections
    
    def transform_predictions(self, predictions, input_size):
        """Transform raw predictions to bounding box parameters.
        
        Args:
            predictions (list): List of raw predictions from forward pass
            input_size (tuple): Input image size (H, W)
            
        Returns:
            list: List of transformed predictions
                 Each element has shape (B, H*W*num_anchors, 5+num_classes)
        """
        batch_size = predictions[0].size(0)
        transformed = []
        
        for i, pred in enumerate(predictions):
            # Get grid size
            _, _, grid_h, grid_w = pred.shape
            
            # Reshape prediction to [B, num_anchors, 5+num_classes, H, W]
            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
            # Permute to [B, num_anchors, H, W, 5+num_classes]
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Get anchors for this scale
            anchors = torch.tensor(self.anchors[i], device=pred.device)
            
            # Create grid offsets
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_h, device=pred.device),
                                           torch.arange(grid_w, device=pred.device),
                                           indexing='ij')
            grid_y = grid_y.view(1, 1, grid_h, grid_w, 1).float()
            grid_x = grid_x.view(1, 1, grid_h, grid_w, 1).float()
            grid = torch.cat([grid_x, grid_y], dim=-1)
            
            # Transform predictions
            # Box center coordinates
            pred[..., 0:2] = torch.sigmoid(pred[..., 0:2]) + grid
            # Box width and height
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors.view(1, self.num_anchors, 1, 1, 2)
            # Objectness score
            pred[..., 4] = torch.sigmoid(pred[..., 4])
            # Class probabilities
            pred[..., 5:] = torch.sigmoid(pred[..., 5:])
            
            # Scale to input size
            stride_h = input_size[0] / grid_h
            stride_w = input_size[1] / grid_w
            pred[..., 0] *= stride_w
            pred[..., 1] *= stride_h
            pred[..., 2] *= stride_w
            pred[..., 3] *= stride_h
            
            # Reshape to [B, H*W*num_anchors, 5+num_classes]
            pred = pred.view(batch_size, -1, 5 + self.num_classes)
            transformed.append(pred)
        
        return transformed
    
    def detect_objects(self, predictions, input_size, conf_threshold=0.5, nms_threshold=0.45):
        """Detect objects from model predictions.
        
        Args:
            predictions (list): List of raw predictions from forward pass
            input_size (tuple): Input image size (H, W)
            conf_threshold (float): Confidence threshold for object detection
            nms_threshold (float): IoU threshold for non-maximum suppression
            
        Returns:
            list: List of detections for each image
                 Each detection is [class_id, confidence, x1, y1, x2, y2]
        """
        # Transform predictions to bounding box parameters
        transformed = self.transform_predictions(predictions, input_size)
        
        # Concatenate predictions from all scales
        batch_size = transformed[0].size(0)
        all_predictions = torch.cat(transformed, dim=1)
        
        # Lists to store detections for each image
        all_detections = []
        
        for b in range(batch_size):
            # Get predictions for this image
            pred = all_predictions[b]
            
            # Filter by objectness score
            mask = pred[:, 4] > conf_threshold
            pred = pred[mask]
            
            if pred.size(0) == 0:
                all_detections.append([])
                continue
            
            # Get class with highest probability
            class_scores, class_ids = torch.max(pred[:, 5:], dim=1)
            
            # Calculate confidence = objectness * class_probability
            confidence = pred[:, 4] * class_scores
            
            # Filter by confidence threshold
            mask = confidence > conf_threshold
            if not mask.any():
                all_detections.append([])
                continue
            
            # Get filtered boxes, scores, classes
            filtered_boxes = pred[mask, :4]
            filtered_conf = confidence[mask]
            filtered_class_ids = class_ids[mask]
            
            # Convert center-form to corner-form (cx, cy, w, h) -> (x1, y1, x2, y2)
            boxes = torch.zeros_like(filtered_boxes)
            boxes[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1
            boxes[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1
            boxes[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2
            boxes[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2
            
            # Apply non-maximum suppression for each class
            detections = []
            unique_classes = filtered_class_ids.unique()
            
            for cls in unique_classes:
                cls_mask = filtered_class_ids == cls
                cls_boxes = boxes[cls_mask]
                cls_conf = filtered_conf[cls_mask]
                
                # Apply NMS
                keep_indices = self._nms(cls_boxes, cls_conf, nms_threshold)
                
                # Add to detections
                for idx in keep_indices:
                    box = cls_boxes[idx]
                    detections.append([cls.item(), cls_conf[idx].item(), *box.tolist()])
            
            all_detections.append(detections)
        
        return all_detections
    
    def _nms(self, boxes, scores, threshold):
        """Apply non-maximum suppression.
        
        Args:
            boxes (torch.Tensor): Boxes in corner-form (x1, y1, x2, y2)
            scores (torch.Tensor): Confidence scores
            threshold (float): IoU threshold
            
        Returns:
            list: Indices of boxes to keep
        """
        # Sort by score
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            # Pick the box with highest score
            i = order[0].item()
            keep.append(i)
            
            # If only one box left, break
            if order.numel() == 1:
                break
            
            # Get IoU with remaining boxes
            curr_box = boxes[i].unsqueeze(0)
            other_boxes = boxes[order[1:]]
            
            # Calculate IoU
            ious = self._box_iou(curr_box, other_boxes)
            
            # Filter boxes with IoU > threshold
            mask = ious < threshold
            order = order[1:][mask]
        
        return keep
    
    def _box_iou(self, box1, box2):
        """Calculate IoU between boxes.
        
        Args:
            box1 (torch.Tensor): First box (x1, y1, x2, y2)
            box2 (torch.Tensor): Second box (x1, y1, x2, y2)
            
        Returns:
            torch.Tensor: IoU values
        """
        # Calculate intersection area
        x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
        y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
        x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
        y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
        
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        
        intersection = w * h
        
        # Calculate union area
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        area1 = area1.unsqueeze(1)
        area2 = area2.unsqueeze(0)
        
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        return iou.squeeze(0)