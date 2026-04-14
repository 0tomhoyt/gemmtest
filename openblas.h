#ifndef OPENBLAS_H
#define OPENBLAS_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"

#ifdef __cplusplus
extern "C" {
#endif

void cblas_sgemm_ref(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                     const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                     const BLASINT k, const float alpha, const float *a, const BLASINT lda,
                     const float *b, const BLASINT ldb, const float beta, float *c,
                     const BLASINT ldc);

void BlasSetNumThreadsLocal(int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* OPENBLAS_H */
