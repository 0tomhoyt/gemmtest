#include "openblas.h"
#include <math.h>

/*
 * Reference implementation of SGEMM with different code structure
 * from unigemm.c to avoid shared bugs.
 *
 * Uses helper functions for index calculation instead of inline branches.
 */

/* Return effective rows of matrix after transpose */
static BLASINT effective_rows(BLASINT rows, BLASINT cols,
                               enum CBLAS_TRANSPOSE trans) {
    if (trans == CblasNoTrans || trans == CblasConjNoTrans)
        return rows;
    return cols;
}

/* Return effective cols of matrix after transpose */
static BLASINT effective_cols(BLASINT rows, BLASINT cols,
                               enum CBLAS_TRANSPOSE trans) {
    if (trans == CblasNoTrans || trans == CblasConjNoTrans)
        return cols;
    return rows;
}

/* Get element from matrix A with transpose/layout handling */
static float get_a_elem(const float *a, BLASINT m, BLASINT n, BLASINT lda,
                        enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
                        BLASINT i, BLASINT j) {
    BLASINT row, col;

    if (trans == CblasNoTrans || trans == CblasConjNoTrans) {
        row = i;
        col = j;
    } else {
        row = j;
        col = i;
    }

    if (order == CblasRowMajor)
        return a[row * lda + col];
    else
        return a[col * lda + row];
}

/* Get element from matrix B with transpose/layout handling */
static float get_b_elem(const float *b, BLASINT m, BLASINT n, BLASINT ldb,
                        enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
                        BLASINT i, BLASINT j) {
    BLASINT row, col;

    if (trans == CblasNoTrans || trans == CblasConjNoTrans) {
        row = i;
        col = j;
    } else {
        row = j;
        col = i;
    }

    if (order == CblasRowMajor)
        return b[row * ldb + col];
    else
        return b[col * ldb + row];
}

/* Get element from matrix C with layout handling */
static float get_c_elem(const float *c, BLASINT ldc,
                        enum CBLAS_ORDER order, BLASINT i, BLASINT j) {
    if (order == CblasRowMajor)
        return c[i * ldc + j];
    else
        return c[j * ldc + i];
}

/* Put element to matrix C with layout handling */
static void put_c_elem(float *c, BLASINT ldc,
                       enum CBLAS_ORDER order, BLASINT i, BLASINT j,
                       float val) {
    if (order == CblasRowMajor)
        c[i * ldc + j] = val;
    else
        c[j * ldc + i] = val;
}

void cblas_sgemm_ref(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                     const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                     const BLASINT k, const float alpha, const float *a, const BLASINT lda,
                     const float *b, const BLASINT ldb, const float beta, float *c,
                     const BLASINT ldc) {

    BLASINT a_eff_rows = effective_rows(m, k, transA);
    BLASINT a_eff_cols = effective_cols(m, k, transA);
    BLASINT b_eff_rows = effective_rows(k, n, transB);
    BLASINT b_eff_cols = effective_cols(k, n, transB);

    for (BLASINT i = 0; i < m; i++) {
        for (BLASINT j = 0; j < n; j++) {
            /* Compute dot product of row i of op(A) and column j of op(B) */
            float sum = 0.0f;
            for (BLASINT p = 0; p < k; p++) {
                float a_val = get_a_elem(a, a_eff_rows, a_eff_cols, lda,
                                         order, transA, i, p);
                float b_val = get_b_elem(b, b_eff_rows, b_eff_cols, ldb,
                                         order, transB, p, j);
                sum += a_val * b_val;
            }

            /* C[i][j] = alpha * sum + beta * C[i][j] */
            float c_old = get_c_elem(c, ldc, order, i, j);
            put_c_elem(c, ldc, order, i, j, alpha * sum + beta * c_old);
        }
    }
}
