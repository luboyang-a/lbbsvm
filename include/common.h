#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
#include <climits>
#include <immintrin.h>
#define N 200000000
#include <algorithm>
#define TAU 1e-12
#define INF 1e20
#define gmax(x, y) (((x) > (y)) ? (x) : (y))
#define gmin(x, y) (((x) < (y)) ? (x) : (y))

inline double dot_avx2(const double* a, const double* b, int dim) {
    __m256d sum = _mm256_setzero_pd();
    int k = 0;
    for(; k + 4 <= dim; k += 4) {
        __m256d va = _mm256_loadu_pd(a + k);
        __m256d vb = _mm256_loadu_pd(b + k);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, sum);
    double res = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for(; k < dim; k++)
        res += a[k] * b[k];
    return res;
}

enum ktype {
    LINEAR,
    RBF
};

#endif