#include <cuda_runtime.h>

__global__ void kernel(const float* in, float* out, int N) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void solve(const float* input, float* output, int N) {
    float *d_in = nullptr;
    float *d_out = nullptr;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemset(d_out, 0, sizeof(float));
    cudaMemcpy(d_in, input, N * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<blocks, threads>>>(d_in, d_out, N);

    cudaMemcpy(output, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}