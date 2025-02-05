#include <iostream>
#include <cuda_runtime.h>

__global__ void linearAlgebraKernel(float *x, float *y, float a, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    float *h_x = new float[N];
    float *h_y = new float[N];
    float *d_x, *d_y;
    float a = 2.0f;

    for (int i = 0; i < N; i++) {
        h_x[i] = i + 1.0f;
        h_y[i] = i + 2.0f;
    }

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    linearAlgebraKernel<<<numBlocks, blockSize>>>(d_x, d_y, a, N);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
