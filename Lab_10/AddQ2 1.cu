#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 1024 
#define MASK_WIDTH 5
#define BLOCK_SIZE 256 

__constant__ int d_Mask[MASK_WIDTH]; 

__global__ void tiledConv1D(int *d_Input, int *d_Output, int width) {
    __shared__ int sharedMem[BLOCK_SIZE + MASK_WIDTH - 1];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int radius = MASK_WIDTH / 2;
    int sharedIdx = tid + radius;
    
    if (idx < width) {
        sharedMem[sharedIdx] = d_Input[idx];
    }
    
    if (tid < radius) {
        sharedMem[sharedIdx - radius] = (idx >= radius) ? d_Input[idx - radius] : 0;
        sharedMem[sharedIdx + BLOCK_SIZE] = (idx + BLOCK_SIZE < width) ? d_Input[idx + BLOCK_SIZE] : 0;
    }
    
    __syncthreads();
    
    if (idx < width) {
        int sum = 0;
        for (int i = 0; i < MASK_WIDTH; i++) {
            sum += sharedMem[sharedIdx - radius + i] * d_Mask[i];
        }
        d_Output[idx] = sum;
    }
}

int main() {
    int size = WIDTH * sizeof(int);
    int h_Input[WIDTH], h_Output[WIDTH], h_Mask[MASK_WIDTH] = {1, 2, 3, 2, 1};
    int *d_Input, *d_Output;

    for (int i = 0; i < WIDTH; i++) {
        h_Input[i] = rand() % 10;
    }

    cudaMalloc(&d_Input, size);
    cudaMalloc(&d_Output, size);

    cudaMemcpy(d_Input, h_Input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Mask, h_Mask, MASK_WIDTH * sizeof(int)); 

    int gridSize = (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tiledConv1D<<<gridSize, BLOCK_SIZE>>>(d_Input, d_Output, WIDTH);

    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < 20; i++) { 
        printf("%d ", h_Input[i]);
    }
    printf("\n\nOutput:\n");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_Output[i]);
    }
    printf("\n");

    cudaFree(d_Input);
    cudaFree(d_Output);

    return 0;
}
