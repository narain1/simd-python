#include <immintrin.h>
#include <pmmintrin.h>
#include <stdio.h>

// Define whether to use AVX2 or SSE based on compiler flags
#if defined(USE_AVX2) && !defined(USE_SSE)
    #define SIMD_WIDTH 8
    #define SIMD_TYPE "AVX2"
#elif !defined(USE_AVX2) && defined(USE_SSE)
    #define SIMD_WIDTH 4
    #define SIMD_TYPE "SSE"
#else
    #define SIMD_WIDTH 1
    #define SIMD_TYPE "No SIMD"
#endif


void print_simd_info() {
    printf("Compiled with %s support.\n", SIMD_TYPE);
}

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

void matrix_multiply_simd(float* A, float* B, float* C, int N) {
    int i, j, k;

    // Choose SIMD width based on available instruction set
    #if defined(USE_AVX2) && !defined(USE_SSE)
    const int width = 8;
    __m256 sum_vec, a_vec, b_vec;
    #else
    const int width = 4;
    __m128 sum_vec, a_vec, b_vec;
    #endif

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            #if defined(USE_AVX2) && !defined(USE_SSE)
            sum_vec = _mm256_setzero_ps();
            #else
            sum_vec = _mm_setzero_ps();
            #endif

            for (k = 0; k < N; k += width) {
                #if defined(USE_AVX2) && !defined(USE_SSE)
                a_vec = _mm256_loadu_ps(&A[i * N + k]);
                b_vec = _mm256_loadu_ps(&B[k * N + j]);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, b_vec));
                #else
                a_vec = _mm_loadu_ps(&A[i * N + k]);
                b_vec = _mm_loadu_ps(&B[k * N + j]);
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(a_vec, b_vec));
                #endif
            }

            // Sum the elements of sum_vec and store the result in C[i*N+j]
            #if defined(USE_AVX2) && !defined(USE_SSE)
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            C[i * N + j] = temp[0] + temp[6];  // Manually sum the remaining elements
            #else
            sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
            C[i * N + j] = _mm_cvtss_f32(sum_vec);
            #endif
        }
    }
}


int main() {
  print_simd_info();
  return 0;
}
