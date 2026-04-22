#include "fuzz_test_worker.h"
#include "fuzz_test_random.h"
#include "fuzz_test_compare.h"
#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "../openblas.h"
#include "test_util.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>

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

        /* Generate three dimensions with controlled category distribution */
        BLASINT m, n, k;
        if (targ->dim_range > 0) {
            rng.random_three_dims_fixed(m, n, k, targ->dim_range);
        } else {
            rng.random_three_dims(m, n, k);
        }

        float alpha = rng.random_alpha_beta();
        float beta = rng.random_alpha_beta();

        /* Determine BLAS thread count
         * If blas_threads == 0, use random value (2-50, weighted)
         * Otherwise use the fixed value from ThreadArg
         */
        int num_threads = targ->blas_threads;
        if (num_threads == 0) {
            num_threads = rng.random_blas_threads();
        }

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
        BLASINT lda = min_lda;
        BLASINT ldb = min_ldb;
        BLASINT ldc = min_ldc;

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

        /* Initialize matrices using InitMatrix */
        InitMatrix(a_buf, a_size, iter * 3);
        InitMatrix(b_buf, b_size, iter * 3 + 1);

        /* Initialize C matrices (same seed for both impl and ref) */
        InitMatrix(c_impl_buf, c_size, iter * 3 + 2);
        memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

        /* Set number of threads for this BLAS call */
        BlasSetNumThreadsLocal(num_threads);

        /* Run implementation and reference based on precision type */
        bool passed = false;
        FailureInfo fail_info{};
        fail_info.order = order;
        fail_info.transA = transA;
        fail_info.transB = transB;
        fail_info.m = m;
        fail_info.n = n;
        fail_info.k = k;
        fail_info.lda = lda;
        fail_info.ldb = ldb;
        fail_info.ldc = ldc;
        fail_info.num_threads = num_threads;

        if (targ->precision == PrecisionType::SGEMM) {
            /* SGEMM 路径 - 原有实现 */
            fail_info.alpha = alpha;
            fail_info.beta = beta;

            cblas_sgemm(order, transA, transB, m, n, k, alpha, a_buf, lda,
                        b_buf, ldb, beta, c_impl_buf, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = compare_matrices(c_impl_buf, c_ref_buf, order, m, n, ldc,
                                      SGEMM_TOLERANCE, &fail_info);
        } else if (targ->precision == PrecisionType::SHGEMM) {
            /* SHGEMM 路径 - 使用标准 BLAS SHGEMM 接口
             * alpha, beta: float 类型
             * a, b: float16_t* 类型
             * c: float* 类型
             */

            /* 分配半精度缓冲区（仅用于 A 和 B） */
            float16_t* a_half = static_cast<float16_t*>(std::malloc(a_size * sizeof(float16_t)));
            float16_t* b_half = static_cast<float16_t*>(std::malloc(b_size * sizeof(float16_t)));

            if (!a_half || !b_half) {
                /* 内存分配失败，清理并跳过此测试 */
                std::free(a_half);
                std::free(b_half);
                continue;
            }

            /* 将单精度 A、B 矩阵转换为半精度 */
            for (BLASINT i = 0; i < a_size; i++) a_half[i] = static_cast<float16_t>(a_buf[i]);
            for (BLASINT i = 0; i < b_size; i++) b_half[i] = static_cast<float16_t>(b_buf[i]);

            /* 记录原始值到 failure info */
            fail_info.alpha = alpha;
            fail_info.beta = beta;

            /* 运行 SHGEMM 实现 (存根函数)
             * alpha, beta 已经是 float，直接传递
             * a_half, b_half 是 float16_t*
             * c_impl_buf 是 float*，直接作为输出
             */
            cblas_shgemm(order, transA, transB, m, n, k, alpha, a_half, lda,
                        b_half, ldb, beta, c_impl_buf, ldc);

            /* 运行参考实现：将半精度转单精度后调用 SGEMM reference */
            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            /* 比较结果
             * c_impl_buf 和 c_ref_buf 都是 float* 类型，直接比较
             */
            passed = compare_matrices(c_impl_buf, c_ref_buf, order, m, n, ldc,
                                      SHGEMM_TOLERANCE, &fail_info);

            /* 清理半精度缓冲区 */
            std::free(a_half);
            std::free(b_half);
        } else {
            /* SBGEMM 路径 - 使用标准 BLAS SBGEMM 接口
             * alpha, beta: float 类型
             * a, b: bfloat16_t* 类型
             * c: float* 类型
             */

            /* 分配 BF16 缓冲区（仅用于 A 和 B） */
            bfloat16_t* a_bf16 = static_cast<bfloat16_t*>(std::malloc(a_size * sizeof(bfloat16_t)));
            bfloat16_t* b_bf16 = static_cast<bfloat16_t*>(std::malloc(b_size * sizeof(bfloat16_t)));

            if (!a_bf16 || !b_bf16) {
                std::free(a_bf16);
                std::free(b_bf16);
                continue;
            }

            /* 将单精度 A、B 矩阵转换为 BF16 */
            /* BF16 转换：截断 float 低 16 位，保留高 16 位 */
            for (BLASINT i = 0; i < a_size; i++) {
                uint32_t bits;
                std::memcpy(&bits, &a_buf[i], sizeof(float));
                a_bf16[i] = static_cast<bfloat16_t>(bits >> 16);
            }
            for (BLASINT i = 0; i < b_size; i++) {
                uint32_t bits;
                std::memcpy(&bits, &b_buf[i], sizeof(float));
                b_bf16[i] = static_cast<bfloat16_t>(bits >> 16);
            }

            /* 记录原始值到 failure info */
            fail_info.alpha = alpha;
            fail_info.beta = beta;

            /* 运行 SBGEMM 实现 (存根函数) */
            cblas_sbgemm(order, transA, transB, m, n, k, alpha, a_bf16, lda,
                        b_bf16, ldb, beta, c_impl_buf, ldc);

            /* 运行参考实现 */
            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            /* 比较结果 */
            passed = compare_matrices(c_impl_buf, c_ref_buf, order, m, n, ldc,
                                      SBGEMM_TOLERANCE, &fail_info);

            /* 清理 BF16 缓冲区 */
            std::free(a_bf16);
            std::free(b_bf16);
        }

        total_tests.fetch_add(1, std::memory_order_relaxed);
        completed_tests.fetch_add(1, std::memory_order_relaxed);

        /* Increment the appropriate size category counter */
        int max_dim = m;
        if (n > max_dim) max_dim = n;
        if (k > max_dim) max_dim = k;

        if (max_dim <= 128) {
            completed_small.fetch_add(1, std::memory_order_relaxed);
        } else if (max_dim <= 512) {
            completed_medium.fetch_add(1, std::memory_order_relaxed);
        } else {
            completed_large.fetch_add(1, std::memory_order_relaxed);
        }

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
