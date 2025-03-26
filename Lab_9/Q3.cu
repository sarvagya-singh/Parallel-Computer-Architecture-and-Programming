#include <stdio.h>
#include <cuda.h>

#define M 4
#define N 4

__global__ void modify_non_border(int *matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
        matrix[row * cols + col] = ~matrix[row * cols + col] & 0b111;
    }
}

void print_binary(int num) {
    for (int i = 2; i >= 0; i--) {
        printf("%d", (num >> i) & 1);
    }
}

int main() {
    int h_matrix[M][N] = {
        {1, 2, 3, 4},
        {5, 5, 8, 8},
        {9, 4, 10, 12},
        {13, 14, 15, 16}
    };

    int *d_matrix;
    cudaMalloc((void**)&d_matrix, M * N * sizeof(int));
    cudaMemcpy(d_matrix, h_matrix, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks(M / 2, N / 2);
    modify_non_border<<<numBlocks, threadsPerBlock>>>(d_matrix, M, N);

    cudaMemcpy(h_matrix, d_matrix, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Modified Matrix (3-bit Binary):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i > 0 && i < M - 1 && j > 0 && j < N - 1) {
                print_binary(h_matrix[i][j]);
            } else {
                printf("%3d", h_matrix[i][j]);  
            }
            printf(" ");
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    return 0;
}
