import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 backbone for object detection.
    
    This class wraps the pre-trained MobileNetV2 model from torchvision and allows
    extracting features from different layers for use in object detection.
    """
    
    def __init__(self, pretrained=True, trainable_layers=3):
        """Initialize the MobileNetV2 backbone.
        
        Args:
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
            trainable_layers (int): Number of layers to fine-tune, counting from the last layer.
                                    0 means all layers are frozen, 6 means all layers are trainable.
        """
        super(MobileNetV2Backbone, self).__init__()
        
        # Load pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract feature layers
        self.features = mobilenet.features
        
        # Define the feature extraction points
        # MobileNetV2 has 19 inverted residual blocks organized in 7 stages
        self.feature_extraction_points = {
            "layer1": 1,   # After first bottleneck, 24 channels, 1/4 resolution
            "layer2": 3,   # After third bottleneck, 32 channels, 1/8 resolution
            "layer3": 6,   # After sixth bottleneck, 64 channels, 1/16 resolution
            "layer4": 13,  # After 13th bottleneck, 96 channels, 1/16 resolution
            "layer5": 18   # After last bottleneck, 320 channels, 1/32 resolution
        }
        
        # Freeze layers based on trainable_layers parameter
        num_layers = len(self.features)
        for i, layer in enumerate(self.features):
            if i < num_layers - trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """Forward pass through the backbone, returning features at different scales.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            dict: Dictionary of feature maps at different scales
        """
        features = {}
        
        # Pass through each layer and save features at extraction points
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Save features at specified extraction points
            for name, idx in self.feature_extraction_points.items():
                if i == idx:
                    features[name] = x
        
        return features
    
    def get_feature_channels(self):
        """Get the number of channels in each feature map.
        
        Returns:
            dict: Dictionary mapping layer names to channel counts
        """
        return {
            "layer1": 24,    # After first bottleneck
            "layer2": 32,    # After third bottleneck
            "layer3": 64,    # After sixth bottleneck
            "layer4": 96,    # After 13th bottleneck
            "layer5": 320    # After last bottleneck
        }
    
    def get_feature_shapes(self, input_shape):
        """Calculate the output shape of each feature map given the input shape.
        
        Args:
            input_shape (tuple): Input shape (H, W)
            
        Returns:
            dict: Dictionary mapping layer names to output shapes (H, W)
        """
        h, w = input_shape
        return {
            "layer1": (h // 4, w // 4),      # 1/4 resolution
            "layer2": (h // 8, w // 8),      # 1/8 resolution
            "layer3": (h // 16, w // 16),    # 1/16 resolution
            "layer4": (h // 16, w // 16),    # 1/16 resolution
            "layer5": (h // 32, w // 32)     # 1/32 resolution
        }