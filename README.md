# Neural Style Transfer

Implementation of Neural Style Transfer based on [Aldo Ferlatti's article](https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d), using PyTorch with a two-phase development approach: CPU implementation followed by CUDA optimization.

## Features

- Optimization-based Neural Style Transfer using VGG19
- CPU implementation with vectorization and caching optimizations
- CUDA acceleration with custom Gram matrix kernel (Phase 2)
- FastAPI service for image processing
- React frontend for easy experimentation

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
3
```

3. (Optional) Install CUDA toolkit for Phase 2 development:

- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Follow installation instructions for your OS

## Project Structure

```
style-transfer/
├── configs/          # Configuration files
├── data/            # Example images (gitignored)
├── src/
│   ├── models/      # VGG19 feature extractor
│   └── utils/       # Image and Gram matrix utilities
│   └── style_transfer.py  # Main NST implementation
├── cuda_ops/        # CUDA optimizations (Phase 2)
├── api/             # FastAPI service
├── web/             # React frontend
├── tests/           # Unit tests
└── scripts/         # Optimization and benchmark scripts
```

## Usage

### Phase 1 (CPU)

1. Style Transfer:

```bash
python scripts/optimize.py --content path/to/content.jpg --style path/to/style.jpg
```

2. Run API Service:

```bash
uvicorn api.app:app --reload
```

### Phase 2 (CUDA)

1. Build CUDA extensions:

```bash
cd cuda_ops && python setup.py install
```

2. Run with GPU acceleration:

```bash
python scripts/optimize.py --content path/to/content.jpg --style path/to/style.jpg --use-cuda
```

## Development

### Phase 1 (CPU Implementation)

- VGG19 feature extraction
- Content and style loss computation
- Basic optimization loop
- CPU-based optimizations
- FastAPI service

### Phase 2 (CUDA Optimization)

- Custom CUDA Gram matrix kernel
- GPU acceleration
- Batch processing
- Performance optimizations
- React frontend
