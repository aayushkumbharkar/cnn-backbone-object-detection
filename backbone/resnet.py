import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    """ResNet50 backbone for object detection.
    
    This class wraps the pre-trained ResNet50 model from torchvision and allows
    extracting features from different layers for use in object detection.
    """
    
    def __init__(self, pretrained=True, trainable_layers=3):
        """Initialize the ResNet50 backbone.
        
        Args:
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
            trainable_layers (int): Number of layers to fine-tune, counting from the last layer.
                                    0 means all layers are frozen, 5 means all layers are trainable.
        """
        super(ResNet50Backbone, self).__init__()
        
        # Load pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 1/4 resolution, 256 channels
        self.layer2 = resnet.layer2  # 1/8 resolution, 512 channels
        self.layer3 = resnet.layer3  # 1/16 resolution, 1024 channels
        self.layer4 = resnet.layer4  # 1/32 resolution, 2048 channels
        
        # Freeze layers based on trainable_layers parameter
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(layers):
            if i < len(layers) - trainable_layers:
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
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features["layer1"] = x  # 1/4 resolution
        
        x = self.layer2(x)
        features["layer2"] = x  # 1/8 resolution
        
        x = self.layer3(x)
        features["layer3"] = x  # 1/16 resolution
        
        x = self.layer4(x)
        features["layer4"] = x  # 1/32 resolution
        
        return features
    
    def get_feature_channels(self):
        """Get the number of channels in each feature map.
        
        Returns:
            dict: Dictionary mapping layer names to channel counts
        """
        return {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048
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
            "layer1": (h // 4, w // 4),
            "layer2": (h // 8, w // 8),
            "layer3": (h // 16, w // 16),
            "layer4": (h // 32, w // 32)
        }