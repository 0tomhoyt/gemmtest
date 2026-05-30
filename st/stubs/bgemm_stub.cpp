#include "bgemm_stub.h"
#include "unigemm_920f.h"
#include "openblas.h"
#include <cstring>
#include <cstdlib>

/**
 * bfloat16_t 到 float 的转换
 */
static inline float bfloat16_to_float(bfloat16_t bf16) {
#if defined(__BF16_TYPE__)
    return static_cast<float>(bf16);
#else
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
#endif
}

/**
 * float 到 bfloat16_t 的转换
 */
static inline bfloat16_t float_to_bfloat16(float f) {
#if defined(__BF16_TYPE__)
    return static_cast<bfloat16_t>(f);
#else
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return static_cast<bfloat16_t>(bits >> 16);
#endif
}

void cblas_bgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                 const BLASINT k, const bfloat16_t alpha, const bfloat16_t *a,
                 const BLASINT lda, const bfloat16_t *b, const BLASINT ldb,
                 const bfloat16_t beta, bfloat16_t *c, const BLASINT ldc) {

    float alpha_f = bfloat16_to_float(alpha);
    float beta_f = bfloat16_to_float(beta);

    BLASINT a_rows, a_cols, b_rows, b_cols;

    if (transA == CblasNoTrans || transA == CblasConjNoTrans) {
        a_rows = (order == CblasRowMajor) ? m : k;
        a_cols = (order == CblasRowMajor) ? k : m;
    } else {
        a_rows = (order == CblasRowMajor) ? k : m;
        a_cols = (order == CblasRowMajor) ? m : k;
    }

    if (transB == CblasNoTrans || transB == CblasConjNoTrans) {
        b_rows = (order == CblasRowMajor) ? k : n;
        b_cols = (order == CblasRowMajor) ? n : k;
    } else {
        b_rows = (order == CblasRowMajor) ? n : k;
        b_cols = (order == CblasRowMajor) ? k : n;
    }

    BLASINT a_size = a_rows * a_cols;
    BLASINT b_size = b_rows * b_cols;
    BLASINT c_size = ldc * ((order == CblasRowMajor) ? m : n);

    float *a_float = static_cast<float*>(std::malloc(a_size * sizeof(float)));
    float *b_float = static_cast<float*>(std::malloc(b_size * sizeof(float)));
    float *c_float = static_cast<float*>(std::malloc(c_size * sizeof(float)));

    if (!a_float || !b_float || !c_float) {
        std::free(a_float);
        std::free(b_float);
        std::free(c_float);
        return;
    }

    for (BLASINT i = 0; i < a_size; i++)
        a_float[i] = bfloat16_to_float(a[i]);
    for (BLASINT i = 0; i < b_size; i++)
        b_float[i] = bfloat16_to_float(b[i]);
    for (BLASINT i = 0; i < c_size; i++)
        c_float[i] = bfloat16_to_float(c[i]);

    cblas_sgemm(order, transA, transB, m, n, k, alpha_f, a_float, lda,
                b_float, ldb, beta_f, c_float, ldc);

    for (BLASINT i = 0; i < c_size; i++)
        c[i] = float_to_bfloat16(c_float[i]);

    std::free(a_float);
    std::free(b_float);
    std::free(c_float);
}
