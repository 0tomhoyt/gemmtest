#ifndef FUZZ_TEST_FAILURE_H
#define FUZZ_TEST_FAILURE_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"

constexpr int MAX_MISMATCHES = 20;

/* Single mismatch record */
struct MismatchRecord {
    BLASINT i;
    BLASINT j;
    float impl_val;
    float ref_val;
    float rel_error;
};

/* Failure log structure */
struct FailureInfo {
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transA;
    enum CBLAS_TRANSPOSE transB;
    BLASINT m, n, k;
    float alpha, beta;
    BLASINT lda, ldb, ldc;
    int num_threads;           /* Number of threads for this BLAS call */
    BLASINT fail_i, fail_j;
    float impl_val;
    float ref_val;
    float rel_error;

    /* Detailed mismatch info */
    int num_mismatches;
    MismatchRecord mismatches[MAX_MISMATCHES];
};

#endif /* FUZZ_TEST_FAILURE_H */
