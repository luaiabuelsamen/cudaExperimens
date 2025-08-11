import ctypes
import numpy as np
import os

_lib_path = os.path.join(os.path.dirname(__file__), "libvector_add.so")
_lib = ctypes.cdll.LoadLibrary(_lib_path)

_lib.vector_add.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
]
_lib.vector_add.restype = None

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.dtype == np.int32 and b.dtype == np.int32
    assert a.shape == b.shape

    c = _lib.vector_add(a, b, a.size)
    return c
