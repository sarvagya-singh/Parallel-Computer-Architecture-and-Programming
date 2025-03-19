#include <stdio.h>
#include <stdlib.h>

#define M 4
#define N 4

__global__ void modifyMatrix(int *A, int *B) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < M && col < N) {
        if (A[row * N + col] % 2 == 0) {
            int rowSum = 0;
            for (int i = 0; i < N; i++) {
                rowSum += A[row * N + i];
            }
            B[row * N + col] = rowSum;
        } else {
            int colSum = 0;
            for (int i = 0; i < M; i++) {
                colSum += A[i * N + col];
            }
            B[row * N + col] = colSum;
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
    int *A, *B;
    int *d_A, *d_B;

    A = (int*)malloc(M * N * sizeof(int));
    B = (int*)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 10;
    }

    printf("Matrix A:\n");
    printMatrix(A, M, N);

    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    modifyMatrix<<<M, N>>>(d_A, d_B);
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nMatrix B:\n");
    printMatrix(B, M, N);

    cudaFree(d_A);
    cudaFree(d_B);

    free(A);
    free(B);

    return 0;
}
