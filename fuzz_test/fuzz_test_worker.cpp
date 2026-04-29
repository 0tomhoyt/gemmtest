#include "fuzz_test_worker.h"
#include "fuzz_test_random.h"
#include "fuzz_test_report.h"
#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "../openblas.h"
#include "test_util.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>

/* Thread worker function implementation */
void thread_worker(ThreadArg *targ) {
    RandomGenerator rng(targ->rand_seed);
    float *a_buf = targ->buffers->a_ptr();
    float *b_buf = targ->buffers->b_ptr();
    float *c_impl_buf = targ->buffers->c_impl_ptr();
    float *c_ref_buf = targ->buffers->c_ref_ptr();
    float16_t *a_half = targ->buffers->a_half_ptr();
    float16_t *b_half = targ->buffers->b_half_ptr();
    float16_t *c_half = targ->buffers->c_half_ptr();
    bfloat16_t *a_bf16 = targ->buffers->a_bf16_ptr();
    bfloat16_t *b_bf16 = targ->buffers->b_bf16_ptr();
    bfloat16_t *c_bf16 = targ->buffers->c_bf16_ptr();
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

        /* Set number of threads for this BLAS call */
        BlasSetNumThreadsLocal(num_threads);

        /* Run implementation and reference based on precision type */
        bool passed = false;

        if (targ->precision == PrecisionType::SGEMM) {
            /* SGEMM 路径
             * 1. InitMatrix 生成 [0,1] float A,B 矩阵
             * 2. InitMatrix 生成 float C 矩阵，复制到 C_ref
             * 3. impl 和 ref 都使用 a_buf/b_buf
             */
            InitMatrix(a_buf, a_size, iter * 3);
            InitMatrix(b_buf, b_size, iter * 3 + 1);
            InitMatrix(c_impl_buf, c_size, iter * 3 + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

            cblas_sgemm(order, transA, transB, m, n, k, alpha, a_buf, lda,
                        b_buf, ldb, beta, c_impl_buf, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = CheckMatrixResult(c_ref_buf, c_impl_buf, m, n, ldc,
                                       SGEMM_TOLERANCE, false,
                                       order == CblasRowMajor);
        } else if (targ->precision == PrecisionType::SHGEMM) {
            /* SHGEMM 路径
             * 1. InitMatrix 直接生成 [0,1] FP16 A,B 矩阵到 a_half/b_half
             * 2. InitMatrix 生成 float C 矩阵，复制到 C_ref
             * 3. static_cast FP16 → float 扩展到 a_buf/b_buf（给 ref 使用）
             * 4. impl 用 a_half/b_half，ref 用 a_buf/b_buf
             */

            /* 1. Init FP16 A,B */
            InitMatrix(a_half, a_size, iter * 3);
            InitMatrix(b_half, b_size, iter * 3 + 1);

            /* 2. Init float C, copy to C_ref */
            InitMatrix(c_impl_buf, c_size, iter * 3 + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

            /* 3. FP16 → float（static_cast 扩展，给 ref 使用） */
            for (BLASINT i = 0; i < a_size; i++)
                a_buf[i] = static_cast<float>(a_half[i]);
            for (BLASINT i = 0; i < b_size; i++)
                b_buf[i] = static_cast<float>(b_half[i]);

            /* 4. impl: stub 内部将 FP16→float→cblas_sgemm */
            cblas_shgemm(order, transA, transB, m, n, k, alpha, a_half, lda,
                        b_half, ldb, beta, c_impl_buf, ldc);

            /* ref: 用扩展后的 float 数据 */
            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = CheckMatrixResult(c_ref_buf, c_impl_buf, m, n, ldc,
                                       SHGEMM_TOLERANCE, false,
                                       order == CblasRowMajor);
        } else if (targ->precision == PrecisionType::HGEMM) {
            /* HGEMM 路径 — 全 FP16 (alpha/beta/A/B/C 都是 float16_t)
             * 1. InitMatrix → a_half, b_half (FP16 A,B)
             * 2. FP16 → float → a_buf, b_buf（给 ref）
             * 3. InitMatrix → c_impl_buf (float C) → memcpy → c_ref_buf
             * 4. float C → FP16 → c_half（给 impl 初始 C）
             * 5. float alpha/beta → FP16
             * 6. impl: cblas_hgemm(fp16_alpha, a_half, b_half, fp16_beta, c_half)
             * 7. ref:  cblas_sgemm_ref(alpha, a_buf, b_buf, beta, c_ref_buf)
             * 8. FP16 c_half → float → c_impl_buf
             * 9. compare(c_impl_buf, c_ref_buf)
             */

            InitMatrix(a_half, a_size, iter * 3);
            InitMatrix(b_half, b_size, iter * 3 + 1);

            for (BLASINT i = 0; i < a_size; i++)
                a_buf[i] = static_cast<float>(a_half[i]);
            for (BLASINT i = 0; i < b_size; i++)
                b_buf[i] = static_cast<float>(b_half[i]);

            InitMatrix(c_impl_buf, c_size, iter * 3 + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));
            for (BLASINT i = 0; i < c_size; i++)
                c_half[i] = static_cast<float16_t>(c_impl_buf[i]);

            float16_t alpha_half = static_cast<float16_t>(alpha);
            float16_t beta_half = static_cast<float16_t>(beta);

            cblas_hgemm(order, transA, transB, m, n, k, alpha_half, a_half, lda,
                        b_half, ldb, beta_half, c_half, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = CheckMatrixResult(c_ref_buf, c_half, m, n, ldc,
                                       HGEMM_TOLERANCE, false,
                                       order == CblasRowMajor);
        } else if (targ->precision == PrecisionType::BGEMM) {
            /* BGEMM 路径 — 全 BF16 (alpha/beta/A/B/C 都是 bfloat16_t)
             * 同 HGEMM 但使用 BF16 类型
             */

            /* 用 InitMatrix 生成 [0,1] float 数据 */
            InitMatrix(a_buf, a_size, iter * 3);
            InitMatrix(b_buf, b_size, iter * 3 + 1);

            /* float → BF16（截断低 16 位） */
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

            /* BF16 → float（扩展回 float，给 ref 使用） */
            for (BLASINT i = 0; i < a_size; i++) {
                uint32_t bits = static_cast<uint32_t>(a_bf16[i]) << 16;
                std::memcpy(&a_buf[i], &bits, sizeof(float));
            }
            for (BLASINT i = 0; i < b_size; i++) {
                uint32_t bits = static_cast<uint32_t>(b_bf16[i]) << 16;
                std::memcpy(&b_buf[i], &bits, sizeof(float));
            }

            InitMatrix(c_impl_buf, c_size, iter * 3 + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));
            for (BLASINT i = 0; i < c_size; i++) {
                uint32_t bits;
                std::memcpy(&bits, &c_impl_buf[i], sizeof(float));
                c_bf16[i] = static_cast<bfloat16_t>(bits >> 16);
            }

            /* float → BF16（位操作，兼容 bfloat16_t = uint16_t 的情况） */
            uint32_t alpha_bits, beta_bits;
            std::memcpy(&alpha_bits, &alpha, sizeof(float));
            std::memcpy(&beta_bits, &beta, sizeof(float));
            bfloat16_t alpha_bf16 = static_cast<bfloat16_t>(alpha_bits >> 16);
            bfloat16_t beta_bf16 = static_cast<bfloat16_t>(beta_bits >> 16);

            cblas_bgemm(order, transA, transB, m, n, k, alpha_bf16, a_bf16, lda,
                        b_bf16, ldb, beta_bf16, c_bf16, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            for (BLASINT i = 0; i < c_size; i++) {
                uint32_t bits = static_cast<uint32_t>(c_bf16[i]) << 16;
                std::memcpy(&c_impl_buf[i], &bits, sizeof(float));
            }

            passed = CheckMatrixResult(c_ref_buf, c_impl_buf, m, n, ldc,
                                       BGEMM_TOLERANCE, false,
                                       order == CblasRowMajor);
        } else {
            /* SBGEMM 路径
             * 1. InitMatrix 直接生成 [0,1] BF16 A,B 矩阵到 a_bf16/b_bf16
             * 2. InitMatrix 生成 float C 矩阵，复制到 C_ref
             * 3. static_cast BF16 → float 扩展到 a_buf/b_buf（给 ref 使用）
             * 4. impl 用 a_bf16/b_bf16，ref 用 a_buf/b_buf
             */

            /* 1. Init BF16 A,B */
            InitMatrix(a_bf16, a_size, iter * 3);
            InitMatrix(b_bf16, b_size, iter * 3 + 1);

            /* 2. Init float C, copy to C_ref */
            InitMatrix(c_impl_buf, c_size, iter * 3 + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

            /* 3. BF16 → float（static_cast 扩展，给 ref 使用） */
            for (BLASINT i = 0; i < a_size; i++)
                a_buf[i] = static_cast<float>(a_bf16[i]);
            for (BLASINT i = 0; i < b_size; i++)
                b_buf[i] = static_cast<float>(b_bf16[i]);

            /* 4. impl: stub 内部将 BF16→float→cblas_sgemm */
            cblas_sbgemm(order, transA, transB, m, n, k, alpha, a_bf16, lda,
                        b_bf16, ldb, beta, c_impl_buf, ldc);

            /* ref: 用扩展后的 float 数据 */
            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = CheckMatrixResult(c_ref_buf, c_impl_buf, m, n, ldc,
                                       SBGEMM_TOLERANCE, false,
                                       order == CblasRowMajor);
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

            /* Increment per-stage failure counter */
            int sn = targ->stage_num;
            if (sn >= 1 && sn < MAX_STAGES) {
                stage_fail_count[sn].fetch_add(1, std::memory_order_relaxed);
            }

            /* Print failure parameters to stderr (avoids mixing with progress bar) */
            std::fprintf(stderr, "  FAIL [%s] %s transA=%s transB=%s M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n",
                        precision_name(targ->precision),
                        order_name(order),
                        trans_name(transA), trans_name(transB),
                        (int)m, (int)n, (int)k,
                        (int)lda, (int)ldb, (int)ldc);
        }
    }
}
