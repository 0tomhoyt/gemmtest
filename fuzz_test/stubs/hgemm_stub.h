#ifndef HGEMM_STUB_H
#define HGEMM_STUB_H

#include "unigemm_920f.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * HGEMM 存根实现
 *
 * 全 FP16 接口：
 * - alpha, beta: float16_t
 * - a, b: float16_t*
 * - c: float16_t*
 *
 * 实现步骤：
 * 1. FP16 alpha/beta/A/B/C → float
 * 2. 调用 cblas_sgemm
 * 3. float C → FP16 C
 */
void cblas_hgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                 const BLASINT k, const float16_t alpha, const float16_t *a,
                 const BLASINT lda, const float16_t *b, const BLASINT ldb,
                 const float16_t beta, float16_t *c, const BLASINT ldc);

#ifdef __cplusplus
}
#endif

#endif /* HGEMM_STUB_H */
