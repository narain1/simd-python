#include <immintrin.h>
#include <stdio.h>

// Define whether to use AVX2 or SSE based on compiler flags
#if defined(USE_AVX2) && !defined(USE_SSE)
#define SIMD_WIDTH 8
#else
#define SIMD_WIDTH 4
#endif

void add_floats(float* a, float* b, float* result, int n) {
    int i;
#if defined(USE_AVX2) && !defined(USE_SSE)
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
#else
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m128 a_vec = _mm_loadu_ps(&a[i]);
        __m128 b_vec = _mm_loadu_ps(&b[i]);
        __m128 result_vec = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps(&result[i], result_vec);
    }
#endif
}

void subtract_floats(float* a, float* b, float* result, int n) {
    int i;
#if defined(USE_AVX2) && !defined(USE_SSE)
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
#else
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m128 a_vec = _mm_loadu_ps(&a[i]);
        __m128 b_vec = _mm_loadu_ps(&b[i]);
        __m128 result_vec = _mm_sub_ps(a_vec, b_vec);
        _mm_storeu_ps(&result[i], result_vec);
    }
#endif
}

void multiply_floats(float* a, float* b, float* result, int n) {
    int i;
#if defined(USE_AVX2) && !defined(USE_SSE)
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
#else
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m128 a_vec = _mm_loadu_ps(&a[i]);
        __m128 b_vec = _mm_loadu_ps(&b[i]);
        __m128 result_vec = _mm_mul_ps(a_vec, b_vec);
        _mm_storeu_ps(&result[i], result_vec);
    }
#endif
}


void divide_floats(float* a, float* b, float* result, int n) {
    int i;
#if defined(USE_AVX2) && !defined(USE_SSE)
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_div_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
#else
    for (i = 0; i < n; i += SIMD_WIDTH) {
        __m128 a_vec = _mm_loadu_ps(&a[i]);
        __m128 b_vec = _mm_loadu_ps(&b[i]);
        __m128 result_vec = _mm_div_ps(a_vec, b_vec);
        _mm_storeu_ps(&result[i], result_vec);
    }
#endif
}
