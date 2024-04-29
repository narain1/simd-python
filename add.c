#include <xmmintrin.h>
#include <stdio.h>

void add_floats(float* a, float* b, float* result, int n) {
  printf("function is called\n");
    for (int i = 0; i < n; i += 4) {
        __m128 a_vec = _mm_loadu_ps(&a[i]);
        __m128 b_vec = _mm_loadu_ps(&b[i]);
        __m128 result_vec = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps(&result[i], result_vec);
    }
}
