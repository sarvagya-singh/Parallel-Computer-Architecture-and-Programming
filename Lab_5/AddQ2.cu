#include <iostream>
#include <cuda_runtime.h>

__global__ void selectionSortKernel(int *matrix, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        for (int i = 0; i < cols - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < cols; j++) {
                if (matrix[row * cols + j] < matrix[row * cols + minIdx]) {
                    minIdx = j;
                }
            }

            if (minIdx != i) {
                int temp = matrix[row * cols + i];
                matrix[row * cols + i] = matrix[row * cols + minIdx];
                matrix[row * cols + minIdx] = temp;
            }
        }
    }
}

int main() {
    int rows = 4;
    int cols = 5;
    size_t size = rows * cols * sizeof(int);

    int h_matrix[4][5] = {
        {12, 11, 13, 5, 6},
        {5, 9, 3, 7, 1},
        {10, 8, 2, 4, 6},
        {4, 14, 9, 7, 13}
    };

    int *d_matrix;
    cudaMalloc((void **)&d_matrix, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    int blockSize = cols;
    int numBlocks = rows; 

    selectionSortKernel<<<numBlocks, blockSize>>>(d_matrix, rows, cols);
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    printf("Sorted matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_matrix[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    return 0;
}
