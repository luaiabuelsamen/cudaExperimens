#include <iostream>
#include <vector>

extern "C" void vector_add_nc(const int* h_a, const int* h_b, int* h_c, int n){
    size_t bytes = sizeof(int) * n;

    for(int i = 0; i < n; i++){
        h_c[i] = h_a[i] + h_b[i];
    }
}

int main(){
    int N = 10000;
    std::vector<int> a_h(N), b_h(N), c_h(N);
    for(int i = 0; i < N; i++){
        a_h[i] = i;
        b_h[i] = i;
    }

    vector_add_nc(a_h.data(), b_h.data(), c_h.data(), N);

    for(int i = 0; i < N; i++){
        std::cout << c_h[i] << std::endl;
    }
    return 0;
}