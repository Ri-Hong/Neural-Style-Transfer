"""
Image Utilities for Neural Style Transfer

This module provides essential image processing functions for Neural Style Transfer.
It handles loading, saving, displaying, and transforming images, as well as computing
the Gram matrix which is crucial for style transfer.

Key Functions:
1. Image Loading/Saving:
   - load_image: Loads and preprocesses images for the network
   - save_image: Saves the generated image to disk

2. Visualization:
   - show_images: Displays content, style, and generated images side by side

3. Style Transfer Math:
   - gram_matrix: Computes the Gram matrix for style representation
     * The Gram matrix captures feature correlations in the image
     * It's used to represent the "style" of an image
     * Size of matrix is (channels × channels)

Usage:
    # Load images
    content_img = load_image('content.jpg', max_size=512)
    style_img = load_image('style.jpg', max_size=512)
    
    # Save result
    save_image(output_tensor, 'output.jpg')
    
    # Display results
    show_images(content_img, style_img, output_img)
    
    # Compute Gram matrix
    gram = gram_matrix(feature_maps)

Note:
- All image tensors are in range [0, 1]
- Images are converted to RGB format
- Tensors have shape (batch_size, channels, height, width)
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, max_size=None):
    """Load and preprocess an image.
    
    Args:
        image_path (str): Path to the image file
        max_size (int, optional): Maximum size of the larger dimension.
                                If None, original size is kept.
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, height, width)
                     with values in range [0, 1]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize if needed
    if max_size:
        # Get original size
        size = image.size
        # Get scaling factor
        scaling_factor = max_size / max(size)
        # Calculate new size maintaining aspect ratio
        size = tuple([int(dim * scaling_factor) for dim in size])
        # Resize image
        image = image.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts to tensor and scales to [0, 1]
    ])
    
    # Add batch dimension
    image = transform(image).unsqueeze(0)
    
    return image

def save_image(tensor, output_path):
    """Save a tensor as an image.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (1, 3, height, width)
                             with values in range [0, 1]
        output_path (str): Path where to save the image
    """
    # Remove batch dimension and convert to PIL Image
    image = tensor.squeeze(0).cpu()
    image = transforms.ToPILImage()(image)
    
    # Save image
    image.save(output_path)

def show_images(content_img, style_img, output_img):
    """Display content, style, and output images side by side.
    
    Args:
        content_img (torch.Tensor): Content image tensor
        style_img (torch.Tensor): Style image tensor
        output_img (torch.Tensor): Output image tensor
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Helper function to convert tensor to displayable format
    def tensor_to_display(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    # Display images
    ax1.imshow(tensor_to_display(content_img))
    ax1.set_title('Content Image')
    ax1.axis('off')
    
    ax2.imshow(tensor_to_display(style_img))
    ax2.set_title('Style Image')
    ax2.axis('off')
    
    ax3.imshow(tensor_to_display(output_img))
    ax3.set_title('Output Image')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def gram_matrix(tensor):
    """Calculate Gram Matrix of a given tensor.
    
    The Gram Matrix is a matrix of dot products between vectorized feature maps,
    used to capture style information in neural style transfer.
    
    Args:
        tensor (torch.Tensor): Feature tensor of shape (batch_size, channels, height, width)
    
    Returns:
        torch.Tensor: Gram matrix of shape (batch_size, channels, channels)
    """
    batch_size, channels, height, width = tensor.size()
    
    # Reshape tensor to 2D matrix
    features = tensor.view(batch_size, channels, height * width)
    
    # Compute Gram matrix
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by total elements
    gram = gram / (channels * height * width)
    
    return gram
