import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./lib.so')

# Helper function to set up function argument types
def setup_function(lib_func, arg_types):
    lib_func.argtypes = arg_types
    lib_func.restype = None

# Define common argument types for SIMD functions
simd_argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

# Set up each function
setup_function(lib.add_floats, simd_argtypes)
setup_function(lib.subtract_floats, simd_argtypes)
setup_function(lib.multiply_floats, simd_argtypes)
setup_function(lib.divide_floats, simd_argtypes)

# Create example data
size = 1024  # must be a multiple of 4 or 8 depending on the SIMD width used in the compiled library
a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
result = np.zeros(size, dtype=np.float32)

# Test addition
lib.add_floats(a, b, result, size)
print("Addition Result:", result)

# Test subtraction
lib.subtract_floats(a, b, result, size)
print("Subtraction Result:", result)

# Test multiplication
lib.multiply_floats(a, b, result, size)
print("Multiplication Result:", result)

# Test division
lib.divide_floats(a, b, result, size)
print("Division Result:", result)

# Additional setup for matrix multiplication if included in your C library
if hasattr(lib, 'matrix_multiply_simd'):
    # Assuming matrix_multiply_simd uses matrices and not flattened arrays
    N = 32  # Define a manageable size, ensure N*N == size for simplicity in this example
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    # Update argument types specifically for matrix multiplication
    lib.matrix_multiply_simd.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_int
    ]

    # Test matrix multiplication
    lib.matrix_multiply_simd(A, B, C, N)
    print("Matrix Multiplication Result:\n", C)

