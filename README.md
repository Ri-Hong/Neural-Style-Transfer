# Neural Style Transfer

Real-time neural style transfer implementation using PyTorch with CUDA optimization.

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install CUDA toolkit if you plan to develop custom CUDA kernels:

- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Follow installation instructions for your OS

## Project Structure

```
style-transfer/
├── configs/          # Training/inference configs
├── data/            # Dataset storage (gitignored)
├── src/
│   ├── models/      # Neural network architectures
│   └── losses/      # Loss functions and VGG network
├── cuda_ops/        # Custom CUDA kernels
├── api/             # FastAPI service
├── web/             # React frontend
├── tests/           # Unit tests
└── scripts/         # Utility scripts
```

## Development

- Training: TBD
- Inference: TBD
- API Service: TBD
- Web Demo: TBD

## License

MIT
