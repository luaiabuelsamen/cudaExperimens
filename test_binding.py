import vector_add
import numpy as np
import time

N = 500000000
a = np.arange(N, dtype=np.int32)
b = np.arange(N, dtype=np.int32)

# Time CUDA
start = time.time()
c_cuda = vector_add.vector_add_cuda(a, b)
cuda_time = time.time() - start

# Time CPU
start = time.time()
c_cpu = vector_add.vector_add_cpu(a, b)
cpu_time = time.time() - start

print(f"CUDA time: {cuda_time:.6f} s")
print(f"CPU time: {cpu_time:.6f} s")