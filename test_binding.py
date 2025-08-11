import vector_add
import numpy as np

N = 10
a = np.arange(N, dtype=np.int32)
b = np.arange(N, dtype=np.int32)
c = np.zeros(N, dtype=np.int32)

vector_add(a, b, c, N)

print(c)