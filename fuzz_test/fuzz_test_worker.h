#ifndef FUZZ_TEST_WORKER_H
#define FUZZ_TEST_WORKER_H

#include "fuzz_test_config.h"
#include "fuzz_test_buffer.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

/* Synchronizes progress bar (stdout) with error output (stderr) */
extern std::mutex console_mutex;

/* Atomic counters for statistics */
extern std::atomic<int> total_tests;
extern std::atomic<int> passed_tests;
extern std::atomic<int> failed_tests;
extern std::atomic<int> completed_tests;  /* For progress tracking */

/* Completed tests by size category */
extern std::atomic<int> completed_small;
extern std::atomic<int> completed_medium;
extern std::atomic<int> completed_large;

/* ============================================================
 * 精度类型枚举 (Precision Type Enumeration)
 * ============================================================ */

enum class PrecisionType {
    SGEMM,  /* 单精度 (float) */
    SHGEMM, /* FP16 A/B + float alpha/beta/C */
    SBGEMM, /* BF16 A/B + float alpha/beta/C */
    HGEMM,  /* 全 FP16 (float16_t alpha/beta/A/B/C) */
    BGEMM   /* 全 BF16 (bfloat16_t alpha/beta/A/B/C) */
};

/* Per-stage failure counts (indexed by stage_num, 1-30) */
constexpr int MAX_STAGES = 32;
extern std::atomic<int> stage_fail_count[];

/* Thread worker function arguments */
struct ThreadArg {
    int thread_id;
    int iterations;
    unsigned int rand_seed;
    int blas_threads = 1;   /* Fixed BLAS thread count for this worker */
    int dim_range = 0;       /* 0=随机类别, 128/512/1024=固定维度范围 */
    PrecisionType precision = PrecisionType::SGEMM;  /* 测试精度类型 */
    int stage_num = 0;       /* Stage number for failure reporting */
    const char *dim_label = "";    /* "Small" / "Medium" / "Large" */
    const char *blas_label = "";   /* "single thread" / "multi thread" */
    std::unique_ptr<ThreadBuffers> buffers;
};

/* Thread worker function declaration */
void thread_worker(ThreadArg *targ);

#endif /* FUZZ_TEST_WORKER_H */
