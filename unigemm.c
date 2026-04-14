#include "gemm_benchmark.h"
#include "unigemm_920f.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * cblas_sgemm: C = alpha * op(A) * op(B) + beta * C
 *
 * Multi-threaded implementation using OpenMP (when available).
 * The outer loop over i (rows of C) is parallelized.
 */

/* Thread-local variable for number of threads */
#ifdef _OPENMP
static _Thread_local int blas_num_threads = 1;
#else
static int blas_num_threads = 1;
#endif

void cblas_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                 const BLASINT k, const float alpha, const float *a, const BLASINT lda,
                 const float *b, const BLASINT ldb, const float beta, float *c,
                 const BLASINT ldc) {

    /* For float, conjugate transpose is the same as transpose */
    int tA = (transA == CblasNoTrans || transA == CblasConjNoTrans) ? 0 : 1;
    int tB = (transB == CblasNoTrans || transB == CblasConjNoTrans) ? 0 : 1;

    /* Set number of threads for this region */
    int num_threads = (blas_num_threads > 0) ? blas_num_threads : 1;

#ifdef _OPENMP
    /* Multi-threaded version with OpenMP */
    if (num_threads > 1 && m > 4) {  /* Only use multiple threads for larger matrices */
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (BLASINT i = 0; i < m; i++) {
            for (BLASINT j = 0; j < n; j++) {
                /* Compute (op(A) * op(B))[i][j] */
                float sum = 0.0f;
                for (BLASINT p = 0; p < k; p++) {
                    float a_val, b_val;

                    if (order == CblasRowMajor) {
                        /* Row-major: element [r][c] at offset r * ld + c */
                        a_val = tA ? a[p * lda + i] : a[i * lda + p];
                        b_val = tB ? b[j * ldb + p] : b[p * ldb + j];
                    } else {
                        /* Col-major: element [r][c] at offset c * ld + r */
                        a_val = tA ? a[i * lda + p] : a[p * lda + i];
                        b_val = tB ? b[p * ldb + j] : b[j * ldb + p];
                    }

                    sum += a_val * b_val;
                }

                /* C[i][j] = alpha * sum + beta * C[i][j] */
                float c_val;
                if (order == CblasRowMajor) {
                    c_val = c[i * ldc + j];
                } else {
                    c_val = c[j * ldc + i];
                }

                c_val = alpha * sum + beta * c_val;

                if (order == CblasRowMajor) {
                    c[i * ldc + j] = c_val;
                } else {
                    c[j * ldc + i] = c_val;
                }
            }
        }
    } else
#endif
    {
        /* Single-threaded fallback for small matrices or when OpenMP is not available */
        for (BLASINT i = 0; i < m; i++) {
            for (BLASINT j = 0; j < n; j++) {
                /* Compute (op(A) * op(B))[i][j] */
                float sum = 0.0f;
                for (BLASINT p = 0; p < k; p++) {
                    float a_val, b_val;

                    if (order == CblasRowMajor) {
                        /* Row-major: element [r][c] at offset r * ld + c */
                        a_val = tA ? a[p * lda + i] : a[i * lda + p];
                        b_val = tB ? b[j * ldb + p] : b[p * ldb + j];
                    } else {
                        /* Col-major: element [r][c] at offset c * ld + r */
                        a_val = tA ? a[i * lda + p] : a[p * lda + i];
                        b_val = tB ? b[p * ldb + j] : b[j * ldb + p];
                    }

                    sum += a_val * b_val;
                }

                /* C[i][j] = alpha * sum + beta * C[i][j] */
                float c_val;
                if (order == CblasRowMajor) {
                    c_val = c[i * ldc + j];
                } else {
                    c_val = c[j * ldc + i];
                }

                c_val = alpha * sum + beta * c_val;

                if (order == CblasRowMajor) {
                    c[i * ldc + j] = c_val;
                } else {
                    c[j * ldc + i] = c_val;
                }
            }
        }
    }
}

/* Set the number of threads for BLAS operations */
void BlasSetNumThreadsLocal(int num_threads) {
    blas_num_threads = num_threads;
}
