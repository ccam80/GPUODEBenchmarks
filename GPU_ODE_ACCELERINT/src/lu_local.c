// Unblocked LU factorise and solve on the LAPACK getrf/getrs interface: column major, 1-based ipiv.

#include <complex.h>
#include <math.h>
#include "lu_local.h"

// LAPACK pivots complex matrices on |re| + |im|.
#define ABS_R(x) fabs(x)
#define ABS_C(x) (fabs(creal(x)) + fabs(cimag(x)))

#define DEFINE_GETRF(NAME, TYPE, ABS, ZERO)                                   \
void NAME(const int* m, const int* n, TYPE* a, const int* lda,                \
          int* ipiv, int* info)                                               \
{                                                                             \
    const int M = *m, N = *n, LDA = *lda;                                     \
    const int mn = M < N ? M : N;                                             \
    *info = 0;                                                                \
    for (int j = 0; j < mn; ++j) {                                            \
        int piv = j;                                                          \
        double best = ABS(a[j + j * LDA]);                                    \
        for (int i = j + 1; i < M; ++i) {                                     \
            const double v = ABS(a[i + j * LDA]);                             \
            if (v > best) { best = v; piv = i; }                              \
        }                                                                     \
        ipiv[j] = piv + 1;                                                    \
        if (best == 0.0) {                                                    \
            if (*info == 0) { *info = j + 1; }                                \
            continue;                                                         \
        }                                                                     \
        if (piv != j) {                                                       \
            for (int k = 0; k < N; ++k) {                                      \
                TYPE t = a[j + k * LDA];                                      \
                a[j + k * LDA] = a[piv + k * LDA];                            \
                a[piv + k * LDA] = t;                                         \
            }                                                                 \
        }                                                                     \
        const TYPE pivot = a[j + j * LDA];                                    \
        for (int i = j + 1; i < M; ++i) {                                     \
            a[i + j * LDA] /= pivot;                                          \
        }                                                                     \
        for (int k = j + 1; k < N; ++k) {                                      \
            const TYPE ujk = a[j + k * LDA];                                  \
            if (ujk == ZERO) { continue; }                                    \
            for (int i = j + 1; i < M; ++i) {                                  \
                a[i + k * LDA] -= a[i + j * LDA] * ujk;                       \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

#define DEFINE_GETRS(NAME, TYPE)                                              \
void NAME(const char* trans, const int* n, const int* nrhs, TYPE* a,          \
          const int* lda, int* ipiv, TYPE* b, const int* ldb, int* info)      \
{                                                                             \
    const int N = *n, NRHS = *nrhs, LDA = *lda, LDB = *ldb;                   \
    *info = 0;                                                                \
    if (*trans != 'N' && *trans != 'n') { *info = -1; return; }               \
    for (int r = 0; r < NRHS; ++r) {                                          \
        TYPE* x = b + r * LDB;                                                \
        for (int j = 0; j < N; ++j) {                                          \
            const int p = ipiv[j] - 1;                                        \
            if (p != j) { TYPE t = x[j]; x[j] = x[p]; x[p] = t; }             \
        }                                                                     \
        for (int j = 0; j < N; ++j) {                                          \
            for (int i = j + 1; i < N; ++i) {                                  \
                x[i] -= a[i + j * LDA] * x[j];                                \
            }                                                                 \
        }                                                                     \
        for (int j = N - 1; j >= 0; --j) {                                     \
            x[j] /= a[j + j * LDA];                                           \
            for (int i = 0; i < j; ++i) {                                      \
                x[i] -= a[i + j * LDA] * x[j];                                \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GETRF(sgetrf_, float, ABS_R, 0.0f)
DEFINE_GETRF(dgetrf_, double, ABS_R, 0.0)
DEFINE_GETRF(cgetrf_, float complex, ABS_C, 0.0f)
DEFINE_GETRF(zgetrf_, double complex, ABS_C, 0.0)

DEFINE_GETRS(sgetrs_, float)
DEFINE_GETRS(dgetrs_, double)
DEFINE_GETRS(cgetrs_, float complex)
DEFINE_GETRS(zgetrs_, double complex)
