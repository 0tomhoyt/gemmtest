#ifndef UNIGEMM_H
#define UNIGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int BLASINT;

typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans   = 111,
    CblasTrans     = 112,
    CblasConjTrans = 113,
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

void cblas_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
                 const BLASINT m, const BLASINT n, const BLASINT k, const float alpha, const float *a,
                 const BLASINT lda, const float *b, const BLASINT ldb, const float beta, float *c, const BLASINT ldc);

void BlasSetNumThreadsLocal(int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* UNIGEMM_H */