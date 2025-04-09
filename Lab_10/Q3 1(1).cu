#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  
#define BLOCK_SIZE 256  

__global__ void inclusiveScan(int *d_Input, int *d_Output, int n) {
    __shared__ int temp[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        temp[threadIdx.x] = d_Input[idx];
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp_val = 0;
        if (threadIdx.x >= stride) {
            temp_val = temp[threadIdx.x] + temp[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            temp[threadIdx.x] = temp_val;
        }
        __syncthreads();
    }

    if (idx < n) {
        d_Output[idx] = temp[threadIdx.x];
    }
}

int main() {
    int size = N * sizeof(int);
    int h_Input[N], h_Output[N];
    int *d_Input, *d_Output;

    for (int i = 0; i < N; i++) {
        h_Input[i] = i + 1; 
    }

    cudaMalloc(&d_Input, size);
    cudaMalloc(&d_Output, size);

    cudaMemcpy(d_Input, h_Input, size, cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    inclusiveScan<<<gridSize, BLOCK_SIZE>>>(d_Input, d_Output, N);

    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < 20; i++) { 
        printf("%d ", h_Input[i]);
    }
    printf("\n\nOutput (Inclusive Scan):\n");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_Output[i]);
    }
    printf("\n");

    cudaFree(d_Input);
    cudaFree(d_Output);

    return 0;
}
