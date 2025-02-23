#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void fused_attention_kernel(half* output, const half* q, const half* k, const half* v,
                                       int batch, int seq_len, int d_k, half scale) {
    __shared__ half s_q[TILE_SIZE][TILE_SIZE];
    __shared__ half s_k[TILE_SIZE][TILE_SIZE];
    __shared__ half s_scores[TILE_SIZE][TILE_SIZE];

    int b_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = threadIdx.x;

    half sum = 0.0f;
    for (int tile = 0; tile < (d_k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < seq_len && (tile * TILE_SIZE + col) < d_k) {
            s_q[threadIdx.y][col] = q[b_idx * seq_len * d_k + row * d_k + tile * TILE_SIZE + col];
        } else {
            s_q[threadIdx.y][col] = 0.0f;
        }
        if (col < seq_len && (tile * TILE_SIZE + threadIdx.y) < d_k) {
            s_k[threadIdx.y][col] = k[b_idx * seq_len * d_k + col * d_k + tile * TILE_SIZE + threadIdx.y];
        } else {
            s_k[threadIdx.y][col] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += s_q[threadIdx.y][i] * s_k[i][col];
        }
        __syncthreads();
    }

    if (row < seq_len && col < seq_len) {
        sum *= scale;
        half exp_sum = expf(sum);
        s_scores[threadIdx.y][col] = exp_sum / (exp_sum + 1.0f); // Aproximação do softmax
    }
    __syncthreads();

    if (row < seq_len && col < seq_len) {
        half attn = s_scores[threadIdx.y][col];
        for (int i = 0; i < d_k; i++) {
            output[b_idx * seq_len * d_k + row * d_k + i] += attn * v[b_idx * seq_len * d_k + col * d_k + i];
        }
    }
}

torch::Tensor fused_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale) {
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be FP16");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be FP16");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be FP16");

    int batch = q.size(0);
    int seq_len = q.size(1);
    int d_k = q.size(2);
    auto output = torch::zeros_like(q);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((seq_len + TILE_SIZE - 1) / TILE_SIZE, batch, 1);

    fused_attention_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(q.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(k.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(v.data_ptr<torch::Half>()),
        batch, seq_len, d_k, __float2half(scale)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_cuda", &fused_attention_cuda, "Fused Attention CUDA");
}
