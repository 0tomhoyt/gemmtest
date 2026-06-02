#include "st_worker.h"
#include "st_common.h"
#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "../openblas.h"
#include "test_util.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <random>

/* 根据 GEMM 参数计算各 buffer 需要的精确大小 */
static void compute_buffer_sizes(const TestParams& params,
                                 BLASINT& a_size, BLASINT& b_size, BLASINT& c_size) {
    enum CBLAS_ORDER order = params.order;
    enum CBLAS_TRANSPOSE transA = params.transA;
    enum CBLAS_TRANSPOSE transB = params.transB;
    BLASINT m = params.M, n = params.N, k = params.K;

    BLASINT lda, ldb, ldc;
    if (transA == CblasNoTrans || transA == CblasConjNoTrans)
        lda = (order == CblasRowMajor) ? k : m;
    else
        lda = (order == CblasRowMajor) ? m : k;

    if (transB == CblasNoTrans || transB == CblasConjNoTrans)
        ldb = (order == CblasRowMajor) ? n : k;
    else
        ldb = (order == CblasRowMajor) ? k : n;

    ldc = (order == CblasRowMajor) ? n : m;

    if (lda > MAX_LD) lda = MAX_LD;
    if (ldb > MAX_LD) ldb = MAX_LD;
    if (ldc > MAX_LD) ldc = MAX_LD;
    if (lda < 1) lda = 1;
    if (ldb < 1) ldb = 1;
    if (ldc < 1) ldc = 1;

    a_size = lda * ((transA == CblasNoTrans || transA == CblasConjNoTrans) ?
                    ((order == CblasRowMajor) ? m : k) :
                    ((order == CblasRowMajor) ? k : m));
    b_size = ldb * ((transB == CblasNoTrans || transB == CblasConjNoTrans) ?
                    ((order == CblasRowMajor) ? k : n) :
                    ((order == CblasRowMajor) ? n : k));
    c_size = ldc * ((order == CblasRowMajor) ? m : n);

    BLASINT max_buf = static_cast<BLASINT>(MAX_LD) * MAX_DIM;
    if (a_size > max_buf) a_size = max_buf;
    if (b_size > max_buf) b_size = max_buf;
    if (c_size > max_buf) c_size = max_buf;
}

/* Thread worker function implementation */
void thread_worker(ThreadArg *targ) {
    RandomGenerator rng(targ->rand_seed);
    std::random_device rd;
    int seed_offset = rd() % 64;

    for (int iter = 0; iter < targ->iterations; iter++) {
        /* Get test parameters from generator */
        auto params_opt = targ->generator();
        if (!params_opt.has_value()) {
            break;  /* No more test cases (CSV mode exhausted) */
        }
        const auto& params = params_opt.value();

        /* 计算本次迭代的精确 buffer 大小 */
        BLASINT a_size, b_size, c_size;
        compute_buffer_sizes(params, a_size, b_size, c_size);

        /* 本次迭代的 buffer — 作用域结束自动释放 */
        ThreadBuffers bufs;
        if (!bufs.allocate_for_sizes(a_size, b_size, c_size, params.precision)) {
            std::fprintf(stderr, "Error: buffer allocation failed\n");
            break;
        }

        /* 提取局部指针别名 */
        float *a_buf = bufs.a_ptr();
        float *b_buf = bufs.b_ptr();
        float *c_impl_buf = bufs.c_impl_ptr();
        float *c_ref_buf = bufs.c_ref_ptr();
        float16_t *a_half = bufs.a_half_ptr();
        float16_t *b_half = bufs.b_half_ptr();
        float16_t *c_half = bufs.c_half_ptr();
        bfloat16_t *a_bf16 = bufs.a_bf16_ptr();
        bfloat16_t *b_bf16 = bufs.b_bf16_ptr();
        bfloat16_t *c_bf16 = bufs.c_bf16_ptr();

        enum CBLAS_ORDER order = params.order;
        enum CBLAS_TRANSPOSE transA = params.transA;
        enum CBLAS_TRANSPOSE transB = params.transB;
        BLASINT m = params.M, n = params.N, k = params.K;
        float alpha = params.alpha;
        float beta = params.beta;

        /* Determine BLAS thread count */
        int num_threads = targ->blas_threads;
        if (num_threads == 0) {
            num_threads = rng.random_blas_threads();
        }

        /* Calculate leading dimensions (same logic as compute_buffer_sizes) */
        BLASINT lda, ldb, ldc;
        if (transA == CblasNoTrans || transA == CblasConjNoTrans)
            lda = (order == CblasRowMajor) ? k : m;
        else
            lda = (order == CblasRowMajor) ? m : k;
        if (transB == CblasNoTrans || transB == CblasConjNoTrans)
            ldb = (order == CblasRowMajor) ? n : k;
        else
            ldb = (order == CblasRowMajor) ? k : n;
        ldc = (order == CblasRowMajor) ? n : m;

        if (lda > MAX_LD) lda = MAX_LD;
        if (ldb > MAX_LD) ldb = MAX_LD;
        if (ldc > MAX_LD) ldc = MAX_LD;
        if (m < 0) m = 0;
        if (n < 0) n = 0;
        if (k < 0) k = 0;
        if (lda < 1) lda = 1;
        if (ldb < 1) ldb = 1;
        if (ldc < 1) ldc = 1;

        /* Set crash context so signal handler can print shape info */
        CrashContext crash_ctx = {
            targ->thread_id, targ->stage_num, params.precision,
            m, n, k, transA, transB, order, alpha, beta,
            lda, ldb, ldc
        };
        g_crash_ctx = &crash_ctx;

        BlasSetNumThreadsLocal(num_threads);

        bool passed = false;

        if (params.precision == PrecisionType::SGEMM) {
            InitMatrix(a_buf, a_size, iter * 3 + seed_offset);
            InitMatrix(b_buf, b_size, iter * 3 + seed_offset + 1);
            InitMatrix(c_impl_buf, c_size, iter * 3 + seed_offset + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

            cblas_sgemm(order, transA, transB, m, n, k, alpha, a_buf, lda,
                        b_buf, ldb, beta, c_impl_buf, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = MatrixCompare(c_ref_buf, c_impl_buf, m, n, ldc,
                                       SGEMM_TOLERANCE, true,
                                       order == CblasRowMajor);
        } else if (params.precision == PrecisionType::SHGEMM) {
            InitMatrix(a_half, a_size, iter * 3 + seed_offset);
            InitMatrix(b_half, b_size, iter * 3 + seed_offset + 1);

            InitMatrix(c_impl_buf, c_size, iter * 3 + seed_offset + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

            for (BLASINT i = 0; i < a_size; i++)
                a_buf[i] = static_cast<float>(a_half[i]);
            for (BLASINT i = 0; i < b_size; i++)
                b_buf[i] = static_cast<float>(b_half[i]);

            cblas_shgemm(order, transA, transB, m, n, k, alpha, a_half, lda,
                        b_half, ldb, beta, c_impl_buf, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = MatrixCompare(c_ref_buf, c_impl_buf, m, n, ldc,
                                       SHGEMM_TOLERANCE, true,
                                       order == CblasRowMajor);
        } else if (params.precision == PrecisionType::HGEMM) {
            InitMatrix(a_half, a_size, iter * 3 + seed_offset);
            InitMatrix(b_half, b_size, iter * 3 + seed_offset + 1);

            for (BLASINT i = 0; i < a_size; i++)
                a_buf[i] = static_cast<float>(a_half[i]);
            for (BLASINT i = 0; i < b_size; i++)
                b_buf[i] = static_cast<float>(b_half[i]);

            InitMatrix(c_impl_buf, c_size, iter * 3 + seed_offset + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));
            for (BLASINT i = 0; i < c_size; i++)
                c_half[i] = static_cast<float16_t>(c_impl_buf[i]);

            float16_t alpha_half = static_cast<float16_t>(alpha);
            float16_t beta_half = static_cast<float16_t>(beta);

            cblas_hgemm(order, transA, transB, m, n, k, alpha_half, a_half, lda,
                        b_half, ldb, beta_half, c_half, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = MatrixCompare(c_ref_buf, c_half, m, n, ldc,
                                       HGEMM_TOLERANCE, true,
                                       order == CblasRowMajor);
        } else if (params.precision == PrecisionType::BGEMM) {
            InitMatrix(a_buf, a_size, iter * 3 + seed_offset);
            InitMatrix(b_buf, b_size, iter * 3 + seed_offset + 1);

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

            for (BLASINT i = 0; i < a_size; i++) {
                uint32_t bits = static_cast<uint32_t>(a_bf16[i]) << 16;
                std::memcpy(&a_buf[i], &bits, sizeof(float));
            }
            for (BLASINT i = 0; i < b_size; i++) {
                uint32_t bits = static_cast<uint32_t>(b_bf16[i]) << 16;
                std::memcpy(&b_buf[i], &bits, sizeof(float));
            }

            InitMatrix(c_impl_buf, c_size, iter * 3 + seed_offset + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));
            for (BLASINT i = 0; i < c_size; i++) {
                uint32_t bits;
                std::memcpy(&bits, &c_impl_buf[i], sizeof(float));
                c_bf16[i] = static_cast<bfloat16_t>(bits >> 16);
            }

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

            passed = MatrixCompare(c_ref_buf, c_impl_buf, m, n, ldc,
                                       BGEMM_TOLERANCE, true,
                                       order == CblasRowMajor);
        } else {
            /* SBGEMM */
            InitMatrix(a_bf16, a_size, iter * 3 + seed_offset);
            InitMatrix(b_bf16, b_size, iter * 3 + seed_offset + 1);

            InitMatrix(c_impl_buf, c_size, iter * 3 + seed_offset + 2);
            memcpy(c_ref_buf, c_impl_buf, c_size * sizeof(float));

            for (BLASINT i = 0; i < a_size; i++)
                a_buf[i] = static_cast<float>(a_bf16[i]);
            for (BLASINT i = 0; i < b_size; i++)
                b_buf[i] = static_cast<float>(b_bf16[i]);

            cblas_sbgemm(order, transA, transB, m, n, k, alpha, a_bf16, lda,
                        b_bf16, ldb, beta, c_impl_buf, ldc);

            cblas_sgemm_ref(order, transA, transB, m, n, k, alpha, a_buf, lda,
                            b_buf, ldb, beta, c_ref_buf, ldc);

            passed = MatrixCompare(c_ref_buf, c_impl_buf, m, n, ldc,
                                       SBGEMM_TOLERANCE, true,
                                       order == CblasRowMajor);
        }

        g_crash_ctx = nullptr;

        total_tests.fetch_add(1, std::memory_order_relaxed);
        completed_tests.fetch_add(1, std::memory_order_relaxed);

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

            int sn = targ->stage_num;
            if (sn >= 1 && sn < MAX_STAGES) {
                stage_fail_count[sn].fetch_add(1, std::memory_order_relaxed);
            }

            {
                std::lock_guard<std::mutex> lock(console_mutex);
                std::fprintf(stderr, "\r%80s\r", "");
                std::fprintf(stderr, "  ERROR: [%s] %s transA=%s transB=%s M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n",
                            precision_name(params.precision),
                            order_name(order),
                            trans_name(transA), trans_name(transB),
                            (int)m, (int)n, (int)k,
                            (int)lda, (int)ldb, (int)ldc);
            }
        }
    } // bufs 析构 → 释放本次迭代的所有 buffer
}
