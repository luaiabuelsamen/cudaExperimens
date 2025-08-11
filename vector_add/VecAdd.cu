#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const int* a, const int* b, int* c, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < n){
        c[idx] = a[idx] + b[idx];   
    }
}

extern "C" int* vector_add(const int* h_a, const int* h_b, int n){
    size_t bytes = sizeof(int) * n;
    int *a_d, *b_d, *c_d, *h_c;
    h_c = (int*)malloc(bytes);
    cudaMalloc((void**)&a_d, bytes);
    cudaMalloc((void**)&b_d, bytes);
    cudaMalloc((void**)&c_d, bytes);

    cudaMemcpy(a_d, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, h_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = n * (+ threadsPerBlock - 1)/ threadsPerBlock;
    vector_add_kernel<<<blocks, threadsPerBlock>>>(a_d, b_d, c_d, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, c_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(h_c);

    return h_c;
}

// int main(){
//     int N = 10000;
//     std::vector<int> a_h(N), b_h(N), c_h(N);
//     for(int i = 0; i < N; i++){
//         a_h[i] = i;
//         b_h[i] = i;
//     }

//     size_t bytes = sizeof(int) * N;
//     int *a_d, *b_d, *c_d;
//     cudaMalloc((void**)&a_d, bytes);
//     cudaMalloc((void**)&b_d, bytes);
//     cudaMalloc((void**)&c_d, bytes);

//     cudaMemcpy(a_d, a_h.data(), bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(b_d, b_h.data(), bytes, cudaMemcpyHostToDevice);

//     int threadsPerBlock = 256;
//     int blocks = (N + threadsPerBlock - 1)/ threadsPerBlock;
//     vector_add_kernel<<<blocks, threadsPerBlock>>>(a_d, b_d, c_d, N);
//     cudaDeviceSynchronize();

//     cudaMemcpy(c_h.data(), c_d, bytes, cudaMemcpyDeviceToHost);
//     cudaFree(a_d);
//     cudaFree(b_d);
//     cudaFree(c_d);

//     for(int i = 0; i < N; i++){
//         std::cout << c_h[i] << std::endl;
//     }
//     return 0;
// }