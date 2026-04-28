#ifndef FUZZ_TEST_FAILURE_H
#define FUZZ_TEST_FAILURE_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"

/* Forward declaration - defined in fuzz_test_worker.h */
enum class PrecisionType;

/* Failure log structure */
struct FailureInfo {
    /* Stage identification */
    int stage_num;
    PrecisionType precision;
    const char *dim_label;
    const char *blas_label;

    /* Test parameters */
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transA;
    enum CBLAS_TRANSPOSE transB;
    BLASINT m, n, k;
    float alpha, beta;
    BLASINT lda, ldb, ldc;
    int num_threads;
};

#endif /* FUZZ_TEST_FAILURE_H */
