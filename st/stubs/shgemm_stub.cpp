#include "shgemm_stub.h"
#include "unigemm_920f.h"
#include "openblas.h"
#include <cstring>
#include <cstdlib>

/**
 * float16_t 到 float 的转换辅助函数
 */
static inline float float16_to_float(const float16_t *f16) {
#if defined(__FLT16_MAX__)
    /* 使用 _Float16 类型，编译器自动转换 */
    return static_cast<float>(*f16);
#elif defined(__FP16_IETF__)
    /* 使用 __fp16 类型，编译器自动转换 */
    return static_cast<float>(*f16);
#else
    /* 回退：将 uint16 视为 IEEE 754 half precision 格式 */
    uint16_t bits = *reinterpret_cast<const uint16_t*>(f16);
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exponent = (bits >> 10) & 0x1f;
    uint32_t mantissa = bits & 0x3ff;

    uint32_t f32_bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero */
            f32_bits = sign << 31;
        } else {
            /* Subnormal number - convert to normalized float */
            /* 简化处理：直接视为 0 */
            f32_bits = sign << 31;
        }
    } else if (exponent == 31) {
        /* Infinity or NaN */
        f32_bits = (sign << 31) | 0x7f800000 | (mantissa << 13);
    } else {
        /* Normalized number */
        f32_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return *reinterpret_cast<float*>(&f32_bits);
#endif
}

/**
 * SHGEMM 存根实现
 *
 * 标准 BLAS SHGEMM 接口：
 * - alpha, beta: float 类型
 * - a, b: float16_t* 类型 (半精度输入矩阵)
 * - c: float* 类型 (单精度输入/输出矩阵)
 *
 * 实现步骤：
 * 1. 将半精度 A、B 矩阵转换为单精度
 * 2. 调用 cblas_sgemm 进行计算
 * 3. 结果直接输出到单精度 C 矩阵（无需转换）
 */
void cblas_shgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                  const BLASINT k, const float alpha, const float16_t *a,
                  const BLASINT lda, const float16_t *b, const BLASINT ldb,
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
        /* 内存分配失败，清理并返回 */
        std::free(a_float);
        std::free(b_float);
        return;
    }

    /* 转换输入矩阵 A 和 B 从半精度到单精度 */
    for (BLASINT i = 0; i < a_size; i++) {
        a_float[i] = float16_to_float(&a[i]);
    }
    for (BLASINT i = 0; i < b_size; i++) {
        b_float[i] = float16_to_float(&b[i]);
    }

    /* 调用 SGEMM 进行计算 */
    /* C 矩阵已经是 float* 类型，可以直接使用 */
    cblas_sgemm(order, transA, transB, m, n, k, alpha, a_float, lda,
                b_float, ldb, beta, c, ldc);

    /* 清理临时缓冲区 */
    std::free(a_float);
    std::free(b_float);
}
