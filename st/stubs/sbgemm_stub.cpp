#include "sbgemm_stub.h"
#include "unigemm_920f.h"
#include "openblas.h"
#include <cstring>
#include <cstdlib>

/**
 * bfloat16 到 float 的转换
 */
#if defined(__BF16_TYPE__)
static inline float bfloat16_to_float(bfloat16_t bf16) {
    return static_cast<float>(bf16);
}
#else
static inline float bfloat16_to_float(bfloat16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}
#endif

/**
 * SBGEMM 存根实现
 * SBGEMM 存根实现
 *
 * 标准 BLAS SBGEMM 接口：
 * - alpha, beta: float 类型
 * - a, b: bfloat16_t* 类型 (BF16 输入矩阵)
 * - c: float* 类型 (单精度输入/输出矩阵)
 *
 * 实现步骤：
 * 1. 将 BF16 A、B 矩阵转换为单精度
 * 2. 调用 cblas_sgemm 进行计算
 * 3. 结果直接输出到单精度 C 矩阵（无需转换）
 */
void cblas_sbgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                  const BLASINT k, const float alpha, const bfloat16_t *a,
                  const BLASINT lda, const bfloat16_t *b, const BLASINT ldb,
                  const float beta, float *c, const BLASINT ldc) {
    /* alpha 和 beta 已经是 float 类型，无需转换 */

    /* 计算矩阵大小 */
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

    /* 分配临时单精度缓冲区（用于 A 和 B） */
    BLASINT a_size = a_rows * a_cols;
    BLASINT b_size = b_rows * b_cols;

    float *a_float = static_cast<float*>(std::malloc(a_size * sizeof(float)));
    float *b_float = static_cast<float*>(std::malloc(b_size * sizeof(float)));

    if (!a_float || !b_float) {
        std::free(a_float);
        std::free(b_float);
        return;
    }

    /* 转换输入矩阵 A 和 B 从 BF16 到单精度 */
    for (BLASINT i = 0; i < a_size; i++) {
        a_float[i] = bfloat16_to_float(a[i]);
    }
    for (BLASINT i = 0; i < b_size; i++) {
        b_float[i] = bfloat16_to_float(b[i]);
    }

    /* 调用 SGEMM 进行计算 */
    cblas_sgemm(order, transA, transB, m, n, k, alpha, a_float, lda,
                b_float, ldb, beta, c, ldc);

    /* 清理临时缓冲区 */
    std::free(a_float);
    std::free(b_float);
}
