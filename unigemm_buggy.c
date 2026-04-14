#include "gemm_benchmark.h"
#include "unigemm_920f.h"

/*
 * INTENTIONALLY BUGGY cblas_sgemm implementation for testing fuzz_test
 *
 * Contains several deliberate bugs:
 * 1. Off-by-one error in matrix indexing
 * 2. Wrong handling of ConjTrans (treats as NoTrans)
 * 3. Occasional wrong alpha value (should be alpha, sometimes uses -alpha)
 * 4. Missing beta scaling in some cases
 */

void cblas_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                 const BLASINT k, const float alpha, const float *a, const BLASINT lda,
                 const float *b, const BLASINT ldb, const float beta, float *c,
                 const BLASINT ldc) {

    /* BUG: Wrong handling of ConjTrans - treat as NoTrans */
    int tA = (transA == CblasNoTrans || transA == CblasConjNoTrans || transA == CblasConjTrans) ? 0 : 1;
    int tB = (transB == CblasNoTrans || transB == CblasConjNoTrans || transB == CblasConjTrans) ? 0 : 1;

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

            /* BUG: Wrong alpha when alpha is negative and i+j is even */
            float effective_alpha = alpha;
            if (alpha < 0 && ((i + j) % 2 == 0)) {
                effective_alpha = -alpha;  // Wrong: double negative
            }

            c_val = effective_alpha * sum + beta * c_val;

            if (order == CblasRowMajor) {
                c[i * ldc + j] = c_val;
            } else {
                c[j * ldc + i] = c_val;
            }
        }
    }
}

/* Set the number of threads for BLAS operations
 * Note: Buggy implementation is also single-threaded, so this is a no-op.
 */
void BlasSetNumThreadsLocal(int num_threads) {
    /* No-op for this implementation */
    (void)num_threads;
}
