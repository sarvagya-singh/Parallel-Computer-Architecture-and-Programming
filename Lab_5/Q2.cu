#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256

__global__ void addVectors(int* A, int* B, int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int N = 1024;
    int h_A[N], h_B[N], h_C[N];
    int *d_A, *d_B, *d_C;

    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 32; i < 256; i *= 2) {
        int blocksPerGrid = (N + i - 1) / i;
        printf("Launching kernel with %d blocks, %d threads per block\n", blocksPerGrid, i);
        addVectors<<<blocksPerGrid, i>>>(d_A, d_B, d_C, N);
    }

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResult of adding vectors:\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
