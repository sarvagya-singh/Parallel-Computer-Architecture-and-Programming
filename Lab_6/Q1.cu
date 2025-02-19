#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(float* N, float* M, float* P, int width, int mask_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width) {
        float result = 0.0f;
        int half_mask = mask_width / 2;

        for (int m = -half_mask; m <= half_mask; ++m) {
            int input_idx = idx + m;

            if (input_idx >= 0 && input_idx < width) {
                result += N[input_idx] * M[m + half_mask];
            }
        }
        P[idx] = result;
    }
}

void convolution_1d(float* N, float* M, float* P, int width, int mask_width) {
    float *d_N, *d_M, *d_P;

    int size_N = width * sizeof(float);
    int size_M = mask_width * sizeof(float);
    int size_P = width * sizeof(float);

    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_P, size_P);

    cudaMemcpy(d_N, N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, size_M, cudaMemcpyHostToDevice);

    int blockSize = 256; 
    int numBlocks = (width + blockSize - 1) / blockSize;  

    convolution_1d_kernel<<<numBlocks, blockSize>>>(d_N, d_M, d_P, width, mask_width);

    cudaMemcpy(P, d_P, size_P, cudaMemcpyDeviceToHost);

    cudaFree(d_N);       
    cudaFree(d_M);
    cudaFree(d_P);
}

int main() {
    int width = 8;
    int mask_width = 3;

    float N[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    
    float M[] = {0.25f, 0.5f, 0.25f};  

    float P[width];

    convolution_1d(N, M, P, width, mask_width);

    printf("Convolution Result: ");
    for (int i = 0; i < width; ++i) {
        printf("%f ", P[i]);
    }
    printf("\n");

    return 0;
}
