#include <stdio.h>
#include <cuda_runtime.h>

__global__ void intToOctalKernel(int* input, int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) {
        int num = input[idx];
        int octal = 0, place = 1;
        
        while (num > 0) {
            octal += (num % 8) * place;
            num /= 8;
            place *= 10;
        }

        output[idx] = octal; 
    }
}

int main() {
    int N = 10; 
    int* h_input = (int*)malloc(N * sizeof(int));
    int* h_output = (int*)malloc(N * sizeof(int));
    int* d_input;
    int* d_output;

    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1; 
    }

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

    intToOctalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Input integers and their corresponding octal values:\n");
    for (int i = 0; i < N; i++) {
        printf("%d -> %d\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}
