#include <stdio.h>
#include <stdlib.h>

#define M 4
#define N 4
#define K 4

__global__ void multiplyMatricesRowWise(int *A, int *B, int *C) {
    int row = threadIdx.x;
    if (row < M) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = 0;
            for (int k = 0; k < K; k++) {
                C[row * N + col] += A[row * K + k] * B[k * N + col];
            }
        }
    }
}

__global__ void multiplyMatricesColumnWise(int *A, int *B, int *C) {
    int col = threadIdx.x;
    if (col < N) {
        for (int row = 0; row < M; row++) {
            C[row * N + col] = 0;
            for (int k = 0; k < K; k++) {
                C[row * N + col] += A[row * K + k] * B[k * N + col];
            }
        }
    }
}

__global__ void multiplyMatricesElementWise(int *A, int *B, int *C) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = 0;
        for (int k = 0; k < K; k++) {
            C[row * N + col] += A[row * K + k] * B[k * N + col];
        }
    }
}

void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    A = (int*)malloc(M * K * sizeof(int));
    B = (int*)malloc(K * N * sizeof(int));
    C = (int*)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * K; i++) {
        A[i] = rand() % 10;
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = rand() % 10;
    }

    printf("Matrix A:\n");
    printMatrix(A, M, K);
    printf("Matrix B:\n");
    printMatrix(B, K, N);

    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    multiplyMatricesRowWise<<<1, M>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix (Row-wise multiplication):\n");
    printMatrix(C, M, N);

    multiplyMatricesColumnWise<<<1, N>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix (Column-wise multiplication):\n");
    printMatrix(C, M, N);

    multiplyMatricesElementWise<<<M, N>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix (Element-wise multiplication):\n");
    printMatrix(C, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
