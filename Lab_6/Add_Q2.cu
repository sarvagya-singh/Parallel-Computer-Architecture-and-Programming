#include <stdio.h>
#include <cuda_runtime.h>

__global__ void onesComplementKernel(int* input, int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = ~input[idx];  
    }
}

int main() {
    int N = 10;  
    int* h_input = (int*)malloc(N * sizeof(int));
    int* h_output = (int*)malloc(N * sizeof(int));
    int* d_input;
    int* d_output;

    h_input[0] = 0b1010;  
    h_input[1] = 0b1111; 
    h_input[2] = 0b1100;  
    h_input[3] = 0b0111; 
    h_input[4] = 0b1001;  
    h_input[5] = 0b0011;
    h_input[6] = 0b1101;  
    h_input[7] = 0b0001;  
    h_input[8] = 0b1011;  
    h_input[9] = 0b0100; 

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    onesComplementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Input binary numbers and their one's complement:\n");
    for (int i = 0; i < N; i++) {
        int bit_mask = (1 << 4) - 1;  
        printf("Input: %d (0b", h_input[i]);
        for (int bit = 3; bit >= 0; bit--) {  
            printf("%d", (h_input[i] >> bit) & 1);  
        }
        printf(") -> One's complement: %u (0b", h_output[i] & bit_mask);  
        for (int bit = 3; bit >= 0; bit--) {  
            printf("%d", ((h_output[i] & bit_mask) >> bit) & 1);
        }
        printf(")\n");

    }

    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}
