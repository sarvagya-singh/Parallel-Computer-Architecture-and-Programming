#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024       
#define FILTER_SIZE 5 
#define BLOCK_SIZE 256

__constant__ int d_Filter[FILTER_SIZE];

__global__ void conv1D(int *d_Input, int *d_Output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = FILTER_SIZE / 2;
    int sum = 0;

    if (idx < n) {
        for (int i = -radius; i <= radius; i++) {
            int index = idx + i;
            if (index >= 0 && index < n) {
                sum += d_Input[index] * d_Filter[i + radius];
            }
        }
        d_Output[idx] = sum;
    }
}

int main() {
    int size = N * sizeof(int);
    int h_Input[N], h_Output[N], h_Filter[FILTER_SIZE] = {1, 2, 3, 2, 1};
    int *d_Input, *d_Output;

    for (int i = 0; i < N; i++) {
        h_Input[i] = rand() % 10;
    }

    cudaMalloc(&d_Input, size);
    cudaMalloc(&d_Output, size);

    cudaMemcpy(d_Input, h_Input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_SIZE * sizeof(int)); 

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv1D<<<gridSize, BLOCK_SIZE>>>(d_Input, d_Output, N);

    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < 20; i++) { 
        printf("%d ", h_Input[i]);
    }
    printf("\n\nOutput:\n");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_Output[i]);
    }
    printf("\n");

    cudaFree(d_Input);
    cudaFree(d_Output);

    return 0;
}