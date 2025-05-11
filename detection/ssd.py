import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSDDetector(nn.Module):
    """Single Shot Detector (SSD) implementation.
    
    This class implements the SSD detection head that can be attached to various
    CNN backbones for object detection. It follows the approach from the paper
    "SSD: Single Shot MultiBox Detector" by Liu et al.
    """
    
    def __init__(self, backbone, num_classes, aspect_ratios=None, min_scale=0.2, max_scale=0.9):
        """Initialize the SSD detector.
        
        Args:
            backbone: CNN backbone for feature extraction
            num_classes (int): Number of object classes (including background)
            aspect_ratios (list): List of aspect ratios for default boxes at each feature map
            min_scale (float): Minimum scale of default boxes
            max_scale (float): Maximum scale of default boxes
        """
        super(SSDDetector, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Default aspect ratios if not provided
        if aspect_ratios is None:
            self.aspect_ratios = [
                [1.0, 2.0, 0.5],  # For layer1
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  # For layer2
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  # For layer3
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  # For layer4
                [1.0, 2.0, 0.5],  # For extra layer 1
                [1.0, 2.0, 0.5]   # For extra layer 2
            ]
        else:
            self.aspect_ratios = aspect_ratios
        
        # Get feature channels from backbone
        self.feature_channels = backbone.get_feature_channels()
        
        # Feature map names from backbone and additional layers
        if hasattr(backbone, 'feature_extraction_points'):
            # For MobileNet
            self.feature_maps = list(backbone.feature_extraction_points.keys())
        else:
            # For ResNet
            self.feature_maps = ["layer2", "layer3", "layer4"]
        
        # Add extra layers to the backbone for multi-scale detection
        self.extra_layers = self._create_extra_layers()
        self.feature_maps.extend([f"extra_{i+1}" for i in range(len(self.extra_layers))])
        
        # Calculate scales for default boxes
        self.scales = self._compute_scales(min_scale, max_scale, len(self.feature_maps))
        
        # Create classification and regression heads for each feature map
        self.classification_heads = nn.ModuleDict()
        self.regression_heads = nn.ModuleDict()
        
        for i, feature_map in enumerate(self.feature_maps):
            # Number of default boxes per location
            num_defaults = len(self.aspect_ratios[i]) + 1  # +1 for scale=sqrt(s_k * s_(k+1))
            
            # Get number of input channels
            if feature_map.startswith("layer"):
                in_channels = self.feature_channels[feature_map]
            else:
                # For extra layers
                idx = int(feature_map.split("_")[1]) - 1
                in_channels = self.extra_layers[idx][0].out_channels
            
            # Create classification head (num_defaults * num_classes outputs per location)
            self.classification_heads[feature_map] = nn.Conv2d(
                in_channels, num_defaults * num_classes, kernel_size=3, padding=1
            )
            
            # Create regression head (num_defaults * 4 outputs per location for box coordinates)
            self.regression_heads[feature_map] = nn.Conv2d(
                in_channels, num_defaults * 4, kernel_size=3, padding=1
            )
            
            # Initialize weights
            self._init_weights(self.classification_heads[feature_map])
            self._init_weights(self.regression_heads[feature_map])
    
    def _create_extra_layers(self):
        """Create extra convolutional layers after the backbone for detection at multiple scales."""
        extra_layers = nn.ModuleList()
        
        # Get the last layer's output channels from backbone
        if hasattr(self.backbone, 'feature_extraction_points'):
            # For MobileNet
            in_channels = self.feature_channels["layer5"]
        else:
            # For ResNet
            in_channels = self.feature_channels["layer4"]
        
        # Add two extra layers with decreasing spatial dimensions
        # First extra layer
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        extra_layers.append(layer1)
        
        # Second extra layer
        layer2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        extra_layers.append(layer2)
        
        return extra_layers
    
    def _compute_scales(self, min_scale, max_scale, num_layers):
        """Compute scales for default boxes based on feature map level."""
        scales = []
        for k in range(num_layers):
            scale = min_scale + (max_scale - min_scale) * k / (num_layers - 1)
            scales.append(scale)
        return scales
    
    def _init_weights(self, layer):
        """Initialize the weights of convolutional layers."""
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the SSD detector.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            tuple: (class_preds, loc_preds, default_boxes)
                class_preds: Class predictions for each default box
                loc_preds: Bounding box regression predictions for each default box
                default_boxes: Default box coordinates (cx, cy, w, h) normalized to [0,1]
        """
        # Get feature maps from backbone
        feature_maps = self.backbone(x)
        
        # Process extra layers
        if hasattr(self.backbone, 'feature_extraction_points'):
            # For MobileNet
            prev_feature = feature_maps["layer5"]
        else:
            # For ResNet
            prev_feature = feature_maps["layer4"]
        
        # Add results from extra layers to feature maps
        for i, layer in enumerate(self.extra_layers):
            extra_feature = layer(prev_feature)
            feature_maps[f"extra_{i+1}"] = extra_feature
            prev_feature = extra_feature
        
        # Apply classification and regression heads to each feature map
        class_preds = []
        loc_preds = []
        
        # Get input image size
        _, _, input_h, input_w = x.shape
        default_boxes = []
        
        for i, feature_map in enumerate(self.feature_maps):
            feat = feature_maps[feature_map]
            
            # Apply classification head
            class_pred = self.classification_heads[feature_map](feat)
            # Reshape to (batch_size, height, width, num_defaults, num_classes)
            batch_size, _, height, width = class_pred.shape
            class_pred = class_pred.permute(0, 2, 3, 1).contiguous()
            class_pred = class_pred.view(batch_size, height, width, -1, self.num_classes)
            class_preds.append(class_pred.reshape(batch_size, -1, self.num_classes))
            
            # Apply regression head
            loc_pred = self.regression_heads[feature_map](feat)
            # Reshape to (batch_size, height, width, num_defaults, 4)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(batch_size, height, width, -1, 4)
            loc_preds.append(loc_pred.reshape(batch_size, -1, 4))
            
            # Generate default boxes for this feature map
            scale = self.scales[i]
            boxes_for_layer = self._create_default_boxes(
                width, height, input_w, input_h, scale, self.aspect_ratios[i]
            )
            default_boxes.append(boxes_for_layer)
        
        # Concatenate predictions from all feature maps
        class_preds = torch.cat(class_preds, dim=1)
        loc_preds = torch.cat(loc_preds, dim=1)
        default_boxes = torch.cat(default_boxes, dim=0)
        
        # Repeat default boxes for batch size
        default_boxes = default_boxes.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return class_preds, loc_preds, default_boxes
    
    def _create_default_boxes(self, width, height, input_w, input_h, scale, aspect_ratios):
        """Create default boxes for a feature map.
        
        Args:
            width (int): Width of the feature map
            height (int): Height of the feature map
            input_w (int): Width of the input image
            input_h (int): Height of the input image
            scale (float): Scale for this feature map's default boxes
            aspect_ratios (list): Aspect ratios for default boxes
            
        Returns:
            torch.Tensor: Default boxes of shape (height*width*num_defaults, 4)
                          where 4 corresponds to (cx, cy, w, h) normalized to [0,1]
        """
        device = next(self.parameters()).device
        
        # Add a default box with scale = sqrt(scale * next_scale)
        if len(self.scales) > 1:
            next_scale = self.scales[min(len(self.scales)-1, self.feature_maps.index(f"layer{i+1}") + 1)]
            aspect_ratios = aspect_ratios + [1.0]
            scales = [scale] * len(aspect_ratios)
            scales[-1] = math.sqrt(scale * next_scale)
        else:
            scales = [scale] * len(aspect_ratios)
            scales.append(1.0)  # Add a default box with scale=1.0
            aspect_ratios = aspect_ratios + [1.0]
        
        num_defaults = len(aspect_ratios)
        
        # Create grid of box centers
        step_x = 1.0 / width
        step_y = 1.0 / height
        
        cx = torch.arange(0, width, device=device) * step_x + step_x / 2
        cy = torch.arange(0, height, device=device) * step_y + step_y / 2
        
        cx, cy = torch.meshgrid(cx, cy, indexing='ij')
        cx = cx.reshape(-1, 1)
        cy = cy.reshape(-1, 1)
        
        # Create default boxes
        default_boxes = []
        for i, ar in enumerate(aspect_ratios):
            scale_i = scales[i]
            
            # Calculate width and height of default box
            if ar == 1.0:
                w = h = scale_i
            else:
                w = scale_i * math.sqrt(ar)
                h = scale_i / math.sqrt(ar)
            
            # Normalize to [0, 1]
            w_norm = w
            h_norm = h
            
            # Stack box parameters
            box = torch.cat([cx, cy, w_norm * torch.ones_like(cx), h_norm * torch.ones_like(cx)], dim=1)
            default_boxes.append(box)
        
        # Concatenate all default boxes
        default_boxes = torch.cat(default_boxes, dim=0)
        
        # Clip to [0, 1]
        default_boxes = torch.clamp(default_boxes, 0, 1)
        
        return default_boxes
    
    def detect_objects(self, class_preds, loc_preds, default_boxes, confidence_threshold=0.5, nms_threshold=0.45):
        """Detect objects from model predictions.
        
        Args:
            class_preds (torch.Tensor): Class predictions from forward pass
            loc_preds (torch.Tensor): Location predictions from forward pass
            default_boxes (torch.Tensor): Default boxes from forward pass
            confidence_threshold (float): Minimum confidence for a box to be considered
            nms_threshold (float): IoU threshold for non-maximum suppression
            
        Returns:
            list: List of detections for each image, each detection is [class_id, confidence, x1, y1, x2, y2]
        """
        batch_size = class_preds.size(0)
        num_classes = class_preds.size(2)
        
        # Apply softmax to class predictions
        class_preds = F.softmax(class_preds, dim=2)
        
        # Lists to store detections for each image
        all_detections = []
        
        for i in range(batch_size):
            # Convert center-form to corner-form (cx, cy, w, h) -> (x1, y1, x2, y2)
            boxes = self._center_to_corner(default_boxes[i], loc_preds[i])
            
            # Store detections for this image
            detections = []
            
            # Skip background class (class 0)
            for c in range(1, num_classes):
                # Get confidence scores for this class
                conf = class_preds[i, :, c]
                
                # Filter by confidence threshold
                mask = conf > confidence_threshold
                if not mask.any():
                    continue
                
                # Get filtered boxes, scores
                filtered_boxes = boxes[mask]
                filtered_conf = conf[mask]
                
                # Apply non-maximum suppression
                keep_indices = self._nms(filtered_boxes, filtered_conf, nms_threshold)
                
                # Add to detections
                for idx in keep_indices:
                    box = filtered_boxes[idx]
                    detections.append([c, filtered_conf[idx].item(), *box.tolist()])
            
            all_detections.append(detections)
        
        return all_detections
    
    def _center_to_corner(self, default_boxes, loc_preds):
        """Convert center-form boxes to corner-form.
        
        Args:
            default_boxes (torch.Tensor): Default boxes in center-form (cx, cy, w, h)
            loc_preds (torch.Tensor): Predicted offsets (tx, ty, tw, th)
            
        Returns:
            torch.Tensor: Boxes in corner-form (x1, y1, x2, y2)
        """
        # Extract components
        cx = default_boxes[:, 0]
        cy = default_boxes[:, 1]
        w = default_boxes[:, 2]
        h = default_boxes[:, 3]
        
        # Extract predicted offsets
        tx = loc_preds[:, 0]
        ty = loc_preds[:, 1]
        tw = loc_preds[:, 2]
        th = loc_preds[:, 3]
        
        # Apply offsets to center coordinates
        cx = cx + tx * 0.1 * w
        cy = cy + ty * 0.1 * h
        w = w * torch.exp(tw * 0.2)
        h = h * torch.exp(th * 0.2)
        
        # Convert to corner form
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Stack and return
        return torch.stack([x1, y1, x2, y2], dim=1)
    
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