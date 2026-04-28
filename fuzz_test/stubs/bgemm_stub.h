#ifndef BGEMM_STUB_H
#define BGEMM_STUB_H

#include "unigemm_920f.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * BGEMM 存根实现
 *
 * 全 BF16 接口：
 * - alpha, beta: bfloat16_t
 * - a, b: bfloat16_t*
 * - c: bfloat16_t*
 *
 * 实现步骤：
 * 1. BF16 alpha/beta/A/B/C → float
 * 2. 调用 cblas_sgemm
 * 3. float C → BF16 C
 */
void cblas_bgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                 const BLASINT k, const bfloat16_t alpha, const bfloat16_t *a,
                 const BLASINT lda, const bfloat16_t *b, const BLASINT ldb,
                 const bfloat16_t beta, bfloat16_t *c, const BLASINT ldc);

#ifdef __cplusplus
}
#endif

#endif /* BGEMM_STUB_H */
