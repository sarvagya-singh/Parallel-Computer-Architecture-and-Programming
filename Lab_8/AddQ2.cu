#include <stdio.h>
#include <cuda_runtime.h>

__device__ int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

__device__ int sum_of_digits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

__global__ void processMatrix(int *A, int *B, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 

    if (idx < N && idy < N) {
        if (idx == idy) {
            B[idx * N + idy] = 0;
        } else if (idx < idy) {
            B[idx * N + idy] = factorial(A[idx * N + idy]);
        } else {
            B[idx * N + idy] = sum_of_digits(A[idx * N + idy]);
        }
    }
}

int main() {
    int N;

    printf("Enter the size of the matrix (N x N): ");
    scanf("%d", &N);

    int *h_A = (int *)malloc(N * N * sizeof(int));
    int *h_B = (int *)malloc(N * N * sizeof(int));

    printf("Enter the elements of the matrix A (%d x %d):\n", N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &h_A[i * N + j]);
        }
    }

    int *d_A, *d_B;

    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // Grid size

    processMatrix<<<gridDim, blockDim>>>(d_A, d_B, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(h_B, d_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}