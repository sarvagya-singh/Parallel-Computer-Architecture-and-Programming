#include <iostream>
#include <cuda_runtime.h>

__global__ void oddEvenSortKernel(int *arr, int n, int phase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n - 1) {
        int evenPhase = (phase % 2 == 0);
        int oddPhase = (phase % 2 != 0);

        if (oddPhase && idx % 2 == 1) {
            if (arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }

        if (evenPhase && idx % 2 == 0) {
            if (arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
    }
}

void oddEvenTranspositionSort(int *arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);

    cudaMalloc((void **)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int phase = 0; phase < n; phase++) {
        oddEvenSortKernel<<<numBlocks, blockSize>>>(d_arr, n, phase);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[] = {12, 34, 54, 2, 3, 21, 3, 56};
    int n = sizeof(arr) / sizeof(arr[0]);

    oddEvenTranspositionSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
