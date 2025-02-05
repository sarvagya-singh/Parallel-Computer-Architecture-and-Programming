#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10  

__global__ void addBlockSizeN(int* A, int* B, int* C) {
    int idx = threadIdx.x;  
    if (idx < N) {  
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void addNThreads(int* A, int* B, int* C) {
    int idx = blockIdx.x; 
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int h_A[N], h_B[N], h_C[N]; 
    int* d_A, *d_B, *d_C;        

    for (int i = 0; i < N; i++) {
        h_A[i] = i;  
        h_B[i] = i * 2;  
    }

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    addBlockSizeN<<<1, N>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result of adding vectors (Block size as N):\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    for (int i = 0; i < N; i++) {
        h_C[i] = 0;
    }

    addNThreads<<<N, 1>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResult of adding vectors (N threads):\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
