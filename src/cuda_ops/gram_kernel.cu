#include <torch/extension.h>

// CUDA declarations
void gram_matrix_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output);

// CUDA kernel for computing Gram matrix
__global__ void gram_matrix_kernel(
    const float* input,  // Input tensor (B, C, H*W)
    float* output,       // Output Gram matrix (B, C, C)
    const int batch_size,
    const int channels,
    const int hw_size  // H*W
) {
    // Get thread indices
    const int b = blockIdx.x;   // batch index
    const int i = blockIdx.y;   // row index
    const int j = threadIdx.x;  // column index

    if (i >= channels || j >= channels) return;

    // Calculate offset for current batch
    const int batch_offset = b * channels * channels;
    const int input_batch_offset = b * channels * hw_size;

    // Compute dot product between i-th and j-th channels
    float sum = 0.0f;
    for (int k = 0; k < hw_size; k++) {
        sum += input[input_batch_offset + i * hw_size + k] *
               input[input_batch_offset + j * hw_size + k];
    }

    // Normalize by total elements
    sum /= (channels * hw_size);

    // Write result to output
    output[batch_offset + i * channels + j] = sum;
}

// C++ implementation of forward pass
void gram_matrix_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output) {
    // Get tensor dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto hw_size = height * width;

    // Reshape input to (B, C, H*W)
    auto input_reshaped = input.view({batch_size, channels, hw_size});

    // Launch kernel
    const dim3 blocks(batch_size, channels);
    const int threads = channels;

    gram_matrix_kernel<<<blocks, threads>>>(
        input_reshaped.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        hw_size);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gram_matrix_cuda_forward", &gram_matrix_cuda_forward, "Gram matrix forward (CUDA)");
}
