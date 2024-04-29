# SIMD Arithmetic Library

This library provides high-performance arithmetic operations on arrays of floating-point numbers using SIMD (Single Instruction, Multiple Data) instructions. The current implementation utilizes SSE (Streaming SIMD Extensions) to perform operations such as addition of float arrays.

## Features

- **Vector Addition**: Adds two arrays of floats using SIMD instructions for accelerated computation.
- **High Performance**: Utilizes SSE instructions to process data in parallel, significantly speeding up large-scale computations compared to scalar operations.

## Requirements

- GCC Compiler
- Linux Operating System (for the provided build instructions)

## Building the Library

This project uses a Makefile for building the shared library. To build the library, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/narain1/simd-python
   cd simd-python
   ```

2. Run the build command to build
   ```bash
   ninja
   ```


