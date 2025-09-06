# 🎨 Neural Style Transfer Project (with PyTorch + CUDA + Deployment)

## 1. Project Overview
We are building a **real-time neural style transfer system** that applies the style of a painting (e.g., Van Gogh’s *Starry Night*) to arbitrary images or webcam input.  
The project highlights:
- **Deep Learning (PyTorch):** Feed-forward transformer network for fast stylization.
- **GPU Acceleration (CUDA):** Training/inference on GPU. Includes at least one **custom CUDA kernel** for optimization.
- **MLOps Practices:** Experiment tracking, unit testing, Dockerization, and CI/CD.
- **Deployment:** Exposed as a **FastAPI** service, optionally with a simple **React** frontend for demos.

Recruiter-facing pitch:
> *“Built a neural style transfer app that runs in real time. Used PyTorch with CUDA acceleration for training and inference. Implemented a custom CUDA kernel to optimize Gram matrix computation, cutting style loss evaluation by ~1.6×. Packaged into a Dockerized FastAPI service with a live web demo.”*

---

## 2. Goals and Deliverables
- ✅ Train a feed-forward style transfer model for at least **one style image**.
- ✅ Create an inference pipeline for arbitrary input images.
- ✅ Implement and benchmark **one custom CUDA kernel** (Gram matrix or Instance Norm).
- ✅ Deploy the model as a GPU-powered API using FastAPI.
- ✅ Build a small web demo (React or static HTML/JS) to showcase results.
- ✅ Provide documentation, training scripts, and example outputs.
- ✅ Include MLOps components: experiment tracking, CI/CD, and Docker.

---

## 3. Architecture Overview

### 3.1 Training Workflow
1. **Inputs:**
   - **Content images:** Subset of COCO dataset (~10k images resized to 256–512px).
   - **Style image:** A single painting (e.g., *Starry Night*).
2. **Transformer Network:** A CNN with residual blocks and upsampling layers that learns to apply the style.
3. **Loss Network (VGG-19):**
   - **Content Loss:** Match high-level features between output and content image.
   - **Style Loss:** Match Gram matrices between output and style image.
   - **Total Variation Loss:** Regularizer to encourage smoothness.
4. **Training Loop:**
   - Forward pass content image through Transformer → stylized output.
   - Compute content/style/TV losses via VGG.
   - Backpropagate to update Transformer network weights.
5. **Output:** A trained Transformer model that can stylize arbitrary images in one forward pass.

### 3.2 Inference Workflow
1. Load trained Transformer model.
2. Accept image input (bytes, file, or webcam).
3. Run forward pass → stylized output.
4. Return processed image as result.

### 3.3 Deployment Workflow
1. Package model and code into Docker image (CUDA-enabled base).
2. Serve model with FastAPI (`/stylize` endpoint).
3. Add monitoring: latency (p50, p95), throughput, GPU utilization.
4. Optional React frontend for uploading photos or live webcam demo.

---

## 4. CUDA Usage

### 4.1 Default CUDA (via PyTorch)
- PyTorch automatically runs tensor operations (conv, matmul, ReLU, backprop) on GPU using **CUDA kernels**.
- `.to("cuda")` ensures data and model live on GPU.

### 4.2 Custom CUDA Extension
We will implement at least one custom CUDA kernel:
- **Target candidate: Gram Matrix computation**
  - Input: feature map tensor `(B, C, H, W)`.
  - Output: Gram matrix `(B, C, C)` measuring feature correlations.
  - Optimized with **shared memory tiling** and optional half precision.
- **Alternative candidate: Instance Normalization**
  - Fuse mean/variance computation + normalization + scale/shift into one CUDA kernel.

This demonstrates GPU engineering skills beyond PyTorch’s abstractions.

---

## 5. Technical Components

### 5.1 Model Architecture (Transformer Network)
- Encoder: Convolution layers with stride for downsampling.
- Residual Blocks: 5–6 blocks with Instance Norm + ReLU.
- Decoder: Upsampling (nearest neighbor + conv) layers.
- Output: Clamp to pixel range `[0, 255]`.

### 5.2 Loss Functions
- **Content Loss:**  
  \( L_{content} = \| \phi_{l}(y) - \phi_{l}(x) \|^2 \)  
  where \( \phi_{l} \) is VGG features at layer \( l \).
- **Style Loss:**  
  \( L_{style} = \sum_l \| G(\phi_{l}(y)) - G(\phi_{l}(s)) \|^2 \)  
  where \( G(\cdot) \) is the Gram matrix.
- **Total Variation Loss:** Smoothness regularization.

### 5.3 Training Details
- Optimizer: Adam (lr = 1e-3, weight decay = 1e-4).
- Batch size: 8–16 (with AMP for memory efficiency).
- Training epochs: 4–6 at 512px resolution.
- Loss weights:  
  - Content: 1.0  
  - Style: 10.0 (tune 5–50)  
  - TV: 1e-6  

### 5.4 Performance Optimizations
- Mixed Precision Training (AMP).
- Channels-last tensor format.
- `torch.backends.cudnn.benchmark = True`.
- `torch.compile()` (PyTorch 2.x).
- Custom CUDA kernel for Gram matrix.

---

## 6. MLOps & Engineering Requirements

### 6.1 Experiment Tracking
- Use **Weights & Biases (W&B)** or **MLflow**.
- Log: training/validation loss curves, style previews, best model checkpoints.

### 6.2 Repository Layout

style-transfer/
configs/          # training/inference configs (YAML/Hydra)
data/             # (gitignored)
src/
models/transformer.py
losses/vgg.py
losses/style.py
train.py
infer.py
cuda_ops/
gram.cu
gram.cpp
setup.py
api/app.py        # FastAPI service
web/              # React demo
tests/            # pytest unit tests
scripts/
export_onnx.py
benchmark.py
Dockerfile
README.md

### 6.3 Testing
- Unit tests:
  - Shape checks for model output.
  - Custom CUDA kernel parity vs PyTorch reference.
- Integration tests:
  - Stylize a small fixed image and compare MD5 checksum.
- CI/CD (GitHub Actions):
  - Run tests (CPU only).
  - Lint with black/ruff.
  - Build Docker image.

### 6.4 Deployment
- Base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
- Expose FastAPI on port 8000 (`uvicorn api.app:app`).
- Endpoints:
  - `POST /stylize` → returns stylized image.
  - `GET /styles` → list available styles.
  - `GET /healthz` → health check.
- Metrics: Prometheus client (QPS, latency, GPU memory utilization).
- Deploy target: Render, AWS EC2 G-instance, or GCP A2.

---

## 7. Project Timeline (Suggested)

- **Day 1–2:** Set up repo, baseline transformer + losses, run small training (256px).  
- **Day 3:** Enable AMP, cudnn benchmark, profiling.  
- **Day 4:** Implement custom CUDA Gram kernel + tests.  
- **Day 5:** Run solid training at 512px, save model.  
- **Day 6:** Package inference pipeline + export TorchScript/ONNX.  
- **Day 7:** Build FastAPI service.  
- **Day 8:** Create frontend demo (React).  
- **Day 9:** Dockerize + deploy to cloud.  
- **Day 10:** Polish README, record demo video, add metrics.

---

## 8. Success Criteria
- ✅ Model trains in <1 hour on a single GPU (per style).
- ✅ Inference latency < 25 ms per 512px image (real-time capable).
- ✅ Custom CUDA kernel runs faster than PyTorch baseline.
- ✅ API returns stylized images via REST endpoint.
- ✅ Demo page (upload/webcam) works and is visually impressive.
- ✅ Repo includes CI, Docker, and docs for reproducibility.

---

## 9. Future Extensions (Optional)
- Multi-style models (Conditional Instance Norm).
- Zero-shot style transfer with AdaIN.
- Video style transfer with temporal consistency.
- Deploy with NVIDIA Triton Inference Server.
- Optimize inference with TensorRT.

---

## 10. Résumé-Ready Highlights
- Built real-time neural style transfer with PyTorch + CUDA.
- Wrote custom CUDA kernel for Gram matrix, achieving ~1.6× faster training steps.
- Deployed GPU-powered FastAPI service with React demo.
- Integrated MLOps practices: experiment tracking, CI/CD, Docker, monitoring.
