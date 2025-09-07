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
    // Shared memory for tile of input features
    extern __shared__ float shared_mem[];

    // Get thread indices
    const int b = blockIdx.x;    // batch index
    const int i = blockIdx.y;    // row index
    const int tx = threadIdx.x;  // thread index within block

    if (i >= channels) return;

    // Calculate offsets
    const int batch_offset = b * channels * channels;
    const int input_batch_offset = b * channels * hw_size;

    // Each thread accumulates partial dot products
    float sum[32] = {0.0f};  // Support up to 32 columns per thread
    const int cols_per_thread = (channels + blockDim.x - 1) / blockDim.x;

    // Process input in tiles
    const int TILE_SIZE = blockDim.x;
    for (int tile = 0; tile < hw_size; tile += TILE_SIZE) {
        // Load tile into shared memory
        if (tile + tx < hw_size) {
            shared_mem[tx] = input[input_batch_offset + i * hw_size + tile + tx];
        } else {
            shared_mem[tx] = 0.0f;
        }
        __syncthreads();

        // Each thread processes multiple columns
        for (int j = 0; j < cols_per_thread; j++) {
            const int col = tx * cols_per_thread + j;
            if (col < channels) {
                // Load column data
                float col_val = 0.0f;
                if (tile + tx < hw_size) {
                    col_val = input[input_batch_offset + col * hw_size + tile + tx];
                }
                // Multiply with row data from shared memory
                sum[j] += shared_mem[tx] * col_val;
            }
        }
        __syncthreads();
    }

    // Write results to output
    for (int j = 0; j < cols_per_thread; j++) {
        const int col = tx * cols_per_thread + j;
        if (col < channels) {
            output[batch_offset + i * channels + col] = sum[j] / (channels * hw_size);
        }
    }
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

    // Configure kernel launch parameters
    const int BLOCK_SIZE = 256;  // Use larger thread blocks
    const dim3 blocks(batch_size, channels);
    const int shared_mem_size = BLOCK_SIZE * sizeof(float);

    // Launch kernel
    gram_matrix_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
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
