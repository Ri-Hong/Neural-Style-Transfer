"""
Neural Style Transfer Implementation

This module implements the core Neural Style Transfer (NST) algorithm as described in the paper
"A Neural Algorithm of Artistic Style" by Gatys et al. and the Medium article by Aldo Ferlatti.

How Neural Style Transfer Works:
1. Content Image: The image we want to stylize
2. Style Image: The image whose style we want to capture
3. Generated Image: Initially a copy of content image, gradually updated to match both:
   - Content features from the content image
   - Style features from the style image

The Process:
1. Extract features from both images using VGG19
2. Calculate two losses:
   - Content Loss: How different are the content features?
   - Style Loss: How different are the Gram matrices (style features)?
3. Update the generated image to minimize both losses
   - Use L-BFGS optimizer for efficient image optimization
   - Clamp values to ensure valid image range [0, 1]

Key Parameters:
- content_weight: How much to prioritize content preservation (default: 1)
- style_weight: How much to prioritize style matching (default: 1e5)
- num_steps: Number of optimization steps (default: 500)

Usage:
    nst = NeuralStyleTransfer(device='cpu')  # or 'cuda' for GPU
    output = nst.transfer_style(
        content_img,
        style_img,
        num_steps=500,
        content_weight=1,
        style_weight=1e5
    )

Note:
- All images should be torch tensors in range [0, 1]
- Tensor shape: (batch_size, channels, height, width)
- Higher style_weight means stronger stylization
- More steps generally means better results but longer processing time
"""

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.vgg import VGG19FeatureExtractor
from src.utils.image_utils import gram_matrix

class NeuralStyleTransfer:
    """Neural Style Transfer implementation.
    
    Implements the Neural Style Transfer algorithm using an optimization-based
    approach with VGG19 features.
    """
    def __init__(self, device='cpu'):
        """Initialize the Neural Style Transfer model.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        self.feature_extractor = VGG19FeatureExtractor().to(device)
        
    def content_loss(self, gen_features, content_features):
        """Calculate content loss between generated and content image features.
        
        Args:
            gen_features (torch.Tensor): Features from generated image
            content_features (torch.Tensor): Features from content image
        
        Returns:
            torch.Tensor: Content loss
        """
        return torch.mean((gen_features - content_features) ** 2)
    
    def style_loss(self, gen_features, style_features):
        """Calculate style loss between generated and style image features.
        
        Args:
            gen_features (dict): Features from generated image
            style_features (dict): Features from style image
        
        Returns:
            torch.Tensor: Style loss
        """
        style_loss = 0
        for layer in gen_features:
            gen_gram = gram_matrix(gen_features[layer])
            style_gram = gram_matrix(style_features[layer])
            style_loss += torch.mean((gen_gram - style_gram) ** 2)
        return style_loss
    
    def transfer_style(self, content_img, style_img, num_steps=500,
                      content_weight=1, style_weight=1e5,
                      log_progress=True):
        """Perform style transfer.
        
        Args:
            content_img (torch.Tensor): Content image tensor
            style_img (torch.Tensor): Style image tensor
            num_steps (int): Number of optimization steps
            content_weight (float): Weight for content loss
            style_weight (float): Weight for style loss
            log_progress (bool): Whether to show progress bar
        
        Returns:
            torch.Tensor: Generated image tensor
        """
        # Move images to device
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        
        # Initialize generated image with content image
        gen_img = content_img.clone().requires_grad_(True)
        
        # Extract content and style features
        content_features, _ = self.feature_extractor(content_img)
        _, style_features = self.feature_extractor(style_img)
        
        # Setup optimizer
        optimizer = optim.LBFGS([gen_img])
        
        # Progress bar
        pbar = tqdm(range(num_steps)) if log_progress else range(num_steps)
        
        # Optimization loop
        step = 0
        while step < num_steps:
            def closure():
                nonlocal step
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Clamp values to valid image range [0, 1]
                with torch.no_grad():
                    gen_img.clamp_(0, 1)
                
                # Get features of generated image
                gen_content_features, gen_style_features = self.feature_extractor(gen_img)
                
                # Calculate losses
                c_loss = content_weight * self.content_loss(gen_content_features, content_features)
                s_loss = style_weight * self.style_loss(gen_style_features, style_features)
                # Calculate total loss and detach for LBFGS
                total_loss = c_loss + s_loss
                loss_value = total_loss.detach().item()
                
                # Backward pass
                total_loss.backward()
                
                # Update progress bar
                if log_progress:
                    pbar.set_postfix({
                        'content_loss': f'{c_loss.item():.2f}',
                        'style_loss': f'{s_loss.item():.2f}'
                    })
                    pbar.update(1)
                
                step += 1
                return loss_value
            
            optimizer.step(closure)
        
        # Final clamp to ensure valid image
        with torch.no_grad():
            gen_img.clamp_(0, 1)
        
        return gen_img
