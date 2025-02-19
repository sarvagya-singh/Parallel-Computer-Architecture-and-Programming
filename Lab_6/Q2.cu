#include <stdio.h>
#include <cuda_runtime.h>

__global__ void find_min_kernel(float* arr, int* min_idx, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < end - start) {
        int global_idx = start + idx;
        
        if (arr[global_idx] < arr[*min_idx]) {
            atomicMin(min_idx, global_idx); 
        }
    }
}

void parallel_selection_sort(float* arr, int n) {
    float *d_arr;
    int *d_min_idx;
    int *min_idx = (int*)malloc(sizeof(int));
    
    cudaMalloc((void**)&d_arr, n * sizeof(float));
    cudaMalloc((void**)&d_min_idx, sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(float), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < n - 1; i++) {
        *min_idx = i;
        cudaMemcpy(d_min_idx, min_idx, sizeof(int), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (n - i) / blockSize + 1;
        find_min_kernel<<<numBlocks, blockSize>>>(d_arr, d_min_idx, i, n);
        
        cudaMemcpy(min_idx, d_min_idx, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (i != *min_idx) {
            float temp = arr[i];
            arr[i] = arr[*min_idx];
            arr[*min_idx] = temp;
        }
        
        cudaMemcpy(d_arr, arr, n * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaFree(d_arr);
    cudaFree(d_min_idx);
    free(min_idx);
}

int main() {
    int n = 10;
    float arr[] = {64.0f, 25.0f, 12.0f, 22.0f, 11.0f, 5.0f, 34.0f, 20.0f, 42.0f, 36.0f};
    
    printf("Original Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", arr[i]);
    }
    printf("\n");
    
    parallel_selection_sort(arr, n);

    printf("Sorted Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", arr[i]);
    }
    printf("\n");

    return 0;
}
