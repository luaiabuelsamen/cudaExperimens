import vector_add
import numpy as np

N = 10
a = np.arange(N, dtype=np.int32)
b = np.arange(N, dtype=np.int32)
c = vector_add.vector_add(a, b)
print(c)