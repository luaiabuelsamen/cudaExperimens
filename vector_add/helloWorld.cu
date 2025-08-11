#include <iostream>

__global__ void hello_from_gpu() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch 1 block with 5 threads
    hello_from_gpu<<<1, 5>>>();
    cudaDeviceSynchronize(); // Wait for GPU to finish
    return 0;
}

