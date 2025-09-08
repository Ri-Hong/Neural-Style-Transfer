# Neural Style Transfer

Implementation of Neural Style Transfer based on [Aldo Ferlatti's article](https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d), using PyTorch with two implementations for Gram matrix computation: one using PyTorch's native `torch.bmm` and another using a custom CUDA kernel for optimization.

## Features

- Optimization-based Neural Style Transfer using VGG19
- PyTorch implementation using native `torch.bmm` for Gram matrix computation
- Alternative implementation with custom CUDA kernel for optimized Gram matrix calculation

## Example Output

Here's an example of applying Van Gogh's "Starry Night" style to a landscape image:

| Content Image                              | Style Image                                 | Output                                                  |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------------- |
| ![Content](/examples/images/landscape.jpg) | ![Style](/examples/images/starry_night.jpg) | ![Output](/examples/outputs/landscape_starry_night.png) |

## Setup

1. Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Neural-Style-Transfer/
├── examples/        # Example images and outputs
│   ├── images/     # Input images for style transfer
│   └── outputs/    # Generated style transfer results
├── scripts/
│   └── optimize.py # Main entry point - CLI tool for style transfer
├── src/
│   ├── cuda_ops/   # Custom CUDA kernel for Gram matrix computation
│   ├── models/     # VGG19 feature extractor
│   ├── utils/      # Image and matrix utilities
│   └── style_transfer.py  # Main NST implementation
```

## Usage

The main entry point is `optimize.py`, which provides a command-line interface for style transfer:

### Basic Usage

```bash
python scripts/optimize.py --content path/to/content.jpg --style path/to/style.jpg
```

### Advanced Options

```bash
python scripts/optimize.py \
    --content path/to/content.jpg \
    --style path/to/style.jpg \
    --output result.png \
    --content-weight 1.5 \
    --style-weight 2e5 \
    --steps 1000 \
    --max-size 1024 \
    --use-custom-cuda  # Enable custom CUDA implementation
```

### Available Options

- `--content`: Path to the content image
- `--style`: Path to the style image
- `--output`: Where to save the result (default: output.png)
- `--content-weight`: How much to preserve content (default: 1)
- `--style-weight`: How much to apply style (default: 1e5)
- `--steps`: Number of optimization steps (default: 500)
- `--max-size`: Maximum image dimension (default: 512)
- `--use-custom-cuda`: Use our optimized CUDA implementation for Gram matrix computation

```

## Development

### Core Implementation

- VGG19 feature extraction
- Content and style loss computation
- Basic optimization loop
- PyTorch native Gram matrix computation using `torch.bmm`

### CUDA Optimization

The custom CUDA implementation (`gram_kernel.cu`) provides an optimized Gram matrix computation:

- **Efficient Memory Access**
  - Uses shared memory for tiled matrix operations
  - Processes input in blocks to maximize GPU utilization
  - Supports batch processing for multiple images

- **Performance Optimizations**
  - Thread coarsening: each thread processes multiple matrix columns
  - Shared memory tiling to reduce global memory access
  - Efficient parallel reduction for matrix multiplication
  - Supports up to 32 columns per thread for better workload distribution

- **Technical Details**
  - Input shape: (Batch, Channels, Height, Width)
  - Output: (Batch, Channels, Channels) Gram matrices
  - Uses 256-thread blocks for optimal occupancy
  - Automatically handles padding and boundary conditions
```
