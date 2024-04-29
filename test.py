import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./lib.so')

# Define the argument types for the add_floats function
lib.add_floats.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

# Create example data
size = 1024  # must be a multiple of 4 for this example
a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
result = np.zeros(size, dtype=np.float32)

# Call the function
lib.add_floats(a, b, result, size)

print("Result:", result)
