#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void calculateSine(float *input, float *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = sinf(input[idx]);
    }
}

int main() {
    int size = 5;
    float h_input[] = {0.0f, 0.5f, 1.0f, 1.5708f, 3.1416f};
    float h_output[size];

    float *d_input, *d_output;
    
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    calculateSine<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
 
    printf("Input angles (in radians):\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", h_input[i]);
    }
    printf("\nSine values:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
