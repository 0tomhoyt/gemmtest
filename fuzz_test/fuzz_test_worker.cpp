#include "fuzz_test_worker.h"
#include "fuzz_test_random.cpp"
#include "fuzz_test_compare.cpp"
#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "../openblas.h"

/* Thread worker function implementation */
void thread_worker(ThreadArg* targ) {
    RandomGenerator rng(targ->rand_seed);
    float* a_buf = targ->buffers->a_ptr();
    float* b_buf = targ->buffers->b_ptr();
    float* c_impl_buf = targ->buffers->c_impl_ptr();
    float* c_ref_buf = targ->buffers->c_ref_ptr();
    size_t max_buf_size = targ->buffers->max_size;

    for (int iter = 0; iter < targ->iterations; iter++) {
        /* Generate random parameters */
        enum CBLAS_ORDER order = rng.random_order();
        enum CBLAS_TRANSPOSE transA = rng.random_transpose();
        enum CBLAS_TRANSPOSE transB = rng.random_transpose();

        BLASINT m = rng.random_dim();
        BLASINT n = rng.random_dim();
        BLASINT k = rng.random_dim();

        float alpha = rng.random_alpha_beta();
        float beta = rng.random_alpha_beta();

        /* Generate random thread count */
        int num_threads = rng.random_num_threads();

        /* Calculate minimum leading dimensions */
        BLASINT min_lda, min_ldb, min_ldc;

        if (transA == CblasNoTrans || transA == CblasConjNoTrans) {
            min_lda = (order == CblasRowMajor) ? k : m;
        } else {
            min_lda = (order == CblasRowMajor) ? m : k;
        }

        if (transB == CblasNoTrans || transB == CblasConjNoTrans) {
            min_ldb = (order == CblasRowMajor) ? n : k;
        } else {
            min_ldb = (order == CblasRowMajor) ? k : n;
        }

        min_ldc = (order == CblasRowMajor) ? n : m;

        /* Add random padding */
        BLASINT lda = min_lda + rng.random_int<BLASINT>(0, 7);
        BLASINT ldb = min_ldb + rng.random_int<BLASINT>(0, 7);
        BLASINT ldc = min_ldc + rng.random_int<BLASINT>(0, 7);

        /* Clamp to maximum */
        if (lda > MAX_LD) lda = MAX_LD;
        if (ldb > MAX_LD) ldb = MAX_LD;
        if (ldc > MAX_LD) ldc = MAX_LD;

        /* Handle edge cases */
        if (m < 0) m = 0;
        if (n < 0) n = 0;
        if (k < 0) k = 0;
        if (lda < 1) lda = 1;
        if (ldb < 1) ldb = 1;
        if (ldc < 1) ldc = 1;

        /* Fill matrices with random values */
        BLASINT a_size = lda * ((transA == CblasNoTrans || transA == CblasConjNoTrans) ?
                                ((order == CblasRowMajor) ? m : k) :
                                ((order == CblasRowMajor) ? k : m));
        if (static_cast<size_t>(a_size) > max_buf_size) a_size = static_cast<BLASINT>(max_buf_size);

        BLASINT b_size = ldb * ((transB == CblasNoTrans || transB == CblasConjNoTrans) ?
                                ((order == CblasRowMajor) ? k : n) :
                                ((order == CblasRowMajor) ? n : k));
        if (static_cast<size_t>(b_size) > max_buf_size) b_size = static_cast<BLASINT>(max_buf_size);

        BLASINT c_size = ldc * ((order == CblasRowMajor) ? m : n);
        if (static_cast<size_t>(c_size) > max_buf_size) c_size = static_cast<BLASINT>(max_buf_size);

        for (BLASINT i = 0; i < a_size; i++) {
            a_buf[i] = rng.random_float(-10.0f, 10.0f);
        }
        for (BLASINT i = 0; i < b_size; i++) {
            b_buf[i] = rng.random_float(-10.0f, 10.0f);
        }

        /* Initialize C matrices */
        for (BLASINT i = 0; i < c_size; i++) {
            float val = rng.random_float(-10.0f, 10.0f);
            c_impl_buf[i] = val;
            c_ref_buf[i] = val;
        }

        /* Set number of threads for this BLAS call */
        BlasSetNumThreadsLocal(num_threads);

        /* Run implementation */
        cblas_sgemm(order, transA, transB, m, n, k, alpha, a_buf, lda,
                    b_buf, ldb, beta, c_impl_buf, ldc);

        /* Set number of threads for reference (same as implementation) */
        BlasSetNumThreadsLocal(num_threads);

        /* Run reference */
        cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                        b_buf, ldb, beta, c_ref_buf, ldc);

        /* Compare results */
        FailureInfo fail_info{};
        fail_info.order = order;
        fail_info.transA = transA;
        fail_info.transB = transB;
        fail_info.m = m;
        fail_info.n = n;
        fail_info.k = k;
        fail_info.alpha = alpha;
        fail_info.beta = beta;
        fail_info.lda = lda;
        fail_info.ldb = ldb;
        fail_info.ldc = ldc;
        fail_info.num_threads = num_threads;

        bool passed = compare_matrices(c_impl_buf, c_ref_buf, order, m, n, ldc,
                                       1e-2f, &fail_info);

        total_tests.fetch_add(1, std::memory_order_relaxed);

        if (passed) {
            passed_tests.fetch_add(1, std::memory_order_relaxed);
        } else {
            failed_tests.fetch_add(1, std::memory_order_relaxed);

            /* Log failure */
            std::lock_guard<std::mutex> lock(fail_mutex);
            if (failure_count < MAX_FAIL_LOGS) {
                failures[failure_count++] = fail_info;
            }
        }
    }
}
