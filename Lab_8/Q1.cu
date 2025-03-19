#include <stdio.h>
#include <stdlib.h>

#define M 4
#define N 4

__global__ void addMatricesRowWise(int *A, int *B, int *C) {
    int row = threadIdx.x;
    if (row < M) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

__global__ void addMatricesColumnWise(int *A, int *B, int *C) {
    int col = threadIdx.x;
    if (col < N) {
        for (int row = 0; row < M; row++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

__global__ void addMatricesElementWise(int *A, int *B, int *C) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
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

    A = (int*)malloc(M * N * sizeof(int));
    B = (int*)malloc(M * N * sizeof(int));
    C = (int*)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    printf("Matrix A:\n");
    printMatrix(A, M, N);
    printf("Matrix B:\n");
    printMatrix(B, M, N);

    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    addMatricesRowWise<<<1, M>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix (Row-wise addition):\n");
    printMatrix(C, M, N);

    addMatricesColumnWise<<<1, N>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix (Column-wise addition):\n");
    printMatrix(C, M, N);

    addMatricesElementWise<<<M, N>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix (Element-wise addition):\n");
    printMatrix(C, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
