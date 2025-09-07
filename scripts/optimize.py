"""
Neural Style Transfer Command Line Tool

This script provides a command-line interface for running Neural Style Transfer on images.
It's the main entry point for using the NST implementation to stylize images.

Features:
1. Command Line Arguments:
   --content: Path to the content image
   --style: Path to the style image
   --output: Where to save the result (default: output.png)
   --content-weight: How much to preserve content (default: 1)
   --style-weight: How much to apply style (default: 1e5)
   --steps: Number of optimization steps (default: 500)
   --max-size: Maximum image dimension (default: 512)
   --use-cuda: Whether to use GPU acceleration

Example Usage:
    # Basic usage
    python optimize.py --content photo.jpg --style painting.jpg

    # Advanced usage
    python optimize.py --content photo.jpg --style painting.jpg \\
                      --output result.png --steps 1000 \\
                      --content-weight 1.5 --style-weight 2e5 \\
                      --max-size 1024 --use-cuda

The script will:
1. Load and preprocess both images
2. Run the style transfer optimization
3. Save the result and display a comparison
4. Show progress bar with loss values

Note:
- Larger --max-size means better quality but slower processing
- More --steps generally gives better results but takes longer
- Adjust weights to control content/style balance
- Use --use-cuda if you have a GPU for faster processing
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.style_transfer import NeuralStyleTransfer
from src.utils.image_utils import load_image, save_image, show_images, set_use_custom_cuda

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True,
                      help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                      help='Path to style image')
    parser.add_argument('--output', type=str, default='output.png',
                      help='Path to output image')
    parser.add_argument('--content-weight', type=float, default=1,
                      help='Weight for content loss')
    parser.add_argument('--style-weight', type=float, default=1e5,
                      help='Weight for style loss')
    parser.add_argument('--steps', type=int, default=500,
                      help='Number of optimization steps')
    parser.add_argument('--max-size', type=int, default=512,
                      help='Maximum size of the larger image dimension')
    parser.add_argument('--use-custom-cuda', action='store_true',
                      help='Use custom CUDA implementation for Gram matrix computation')
    args = parser.parse_args()

    # Automatically detect best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set whether to use custom CUDA implementation
    set_use_custom_cuda(args.use_custom_cuda)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load images
    content_img = load_image(args.content, max_size=args.max_size)
    style_img = load_image(args.style, max_size=args.max_size)
    
    print(f'Content image size: {tuple(content_img.shape[2:])}')
    print(f'Style image size: {tuple(style_img.shape[2:])}')

    # Initialize style transfer model
    model = NeuralStyleTransfer(device=device)

    # Perform style transfer
    print('Starting style transfer...')
    output_img = model.transfer_style(
        content_img,
        style_img,
        num_steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight
    )

    # Save result
    save_image(output_img, args.output)
    print(f'Output saved to: {args.output}')

    # Display results
    show_images(content_img, style_img, output_img)

if __name__ == '__main__':
    main()
