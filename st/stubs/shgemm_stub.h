#ifndef SHGEMM_STUB_H
#define SHGEMM_STUB_H

#include "unigemm_920f.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SHGEMM 存根实现
 *
 * 由于 BLAS 库没有提供 SHGEMM 实现，此存根函数将：
 * 1. 接受半精度 (float16_t) A、B 矩阵输入
 * 2. 转换为单精度进行计算（使用 cblas_sgemm）
 * 3. 将结果输出到单精度 C 矩阵
 *
 * 注意：这是标准 BLAS SHGEMM 接口：
 * - alpha, beta: float 类型
 * - a, b: float16_t* 类型 (半精度输入矩阵)
 * - c: float* 类型 (单精度输出矩阵)
 *
 * @param order 矩阵存储顺序 (RowMajor/ColMajor)
 * @param transA A 矩阵转置模式
 * @param transB B 矩阵转置模式
 * @param m, n, k 矩阵维度
 * @param alpha 标量乘数 (单精度)
 * @param a A 矩阵指针 (半精度)
 * @param lda A 矩阵主维度
 * @param b B 矩阵指针 (半精度)
 * @param ldb B 矩阵主维度
 * @param beta 标量乘数 (单精度)
 * @param c C 矩阵指针 (单精度，输入/输出)
 * @param ldc C 矩阵主维度
 */
void cblas_shgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                  const BLASINT k, const float alpha, const float16_t *a,
                  const BLASINT lda, const float16_t *b, const BLASINT ldb,
                  const float beta, float *c, const BLASINT ldc);

#ifdef __cplusplus
}
#endif

#endif /* SHGEMM_STUB_H */
