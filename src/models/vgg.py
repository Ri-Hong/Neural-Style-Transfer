"""
VGG19 Feature Extractor for Neural Style Transfer

This module provides a modified VGG19 network that extracts features from specific layers
for Neural Style Transfer. The VGG19 network is a deep convolutional neural network 
pre-trained on ImageNet, which has learned to recognize various image features.

Key Concepts:
- VGG19: A deep CNN with 19 layers (16 conv layers + 3 fully connected)
- Feature Extraction: We use intermediate layers to capture:
  * Content features (from layer conv4_2)
  * Style features (from layers conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
- Pre-trained weights: We use weights from ImageNet training, no further training needed

Usage:
    extractor = VGG19FeatureExtractor()
    content_features, style_features = extractor(image)

Note:
- Input images should be in range [0, 1]
- Images are automatically normalized using ImageNet statistics
- All parameters are frozen (requires_grad=False) as we only use it for extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms

class VGG19FeatureExtractor(nn.Module):
    """VGG19 feature extractor for Neural Style Transfer.
    
    Extracts features from specific layers of a pre-trained VGG19 network
    for computing content and style losses.
    """
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        # Load pre-trained VGG19 model with the latest weights
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).eval()
        
        # Content layer - conv4_2 (22nd layer in network)
        # This is the 2nd conv layer in Block 4
        # Block 4 structure: conv4_1 → conv4_2 → conv4_3 → conv4_4 → pooling
        self.content_layer = '22'
        
        # Style layers - first conv layer from each block
        # Each block processes features at different scales:
        self.style_layers = {
            '0': 'conv1_1',   # Block 1: Basic features (edges, colors)
            '5': 'conv2_1',   # Block 2: Simple textures (after pool1)
            '10': 'conv3_1',  # Block 3: Complex patterns (after pool2)
            '19': 'conv4_1',  # Block 4: Object parts (after pool3)
            '28': 'conv5_1'   # Block 5: Large structures (after pool4)
        }
        
        # Create feature extractors for content and style
        self.features = vgg19.features
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for VGG input
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
        
    def forward(self, x):
        """Extract features from the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
                            with values in range [0, 1]
        
        Returns:
            tuple: (content_features, style_features)
                - content_features: Features from content layer
                - style_features: Dictionary of features from style layers
        """
        # Normalize input
        x = self.normalize(x)
        
        # Store features
        features = {}
        style_features = {}
        
        # Extract features layer by layer
        for name, layer in self.features.named_children():
            x = layer(x)
            
            # Save content features
            if name == self.content_layer:
                features['content'] = x
            
            # Save style features
            if name in self.style_layers:
                style_features[self.style_layers[name]] = x
        
        return features['content'], style_features
    
    def preprocess(self, image_tensor):
        """Preprocess image tensor for VGG input.
        
        Args:
            image_tensor (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
                                       with values in range [0, 1]
        
        Returns:
            torch.Tensor: Preprocessed tensor ready for VGG
        """
        return self.normalize(image_tensor)
