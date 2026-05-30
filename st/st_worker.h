#ifndef ST_WORKER_H
#define ST_WORKER_H

#include "st_common.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <functional>
#include <optional>

/* Synchronizes progress bar (stdout) with error output (stderr) */
extern std::mutex console_mutex;

/* Atomic counters for statistics */
extern std::atomic<int> total_tests;
extern std::atomic<int> passed_tests;
extern std::atomic<int> failed_tests;
extern std::atomic<int> completed_tests;

/* Completed tests by size category */
extern std::atomic<int> completed_small;
extern std::atomic<int> completed_medium;
extern std::atomic<int> completed_large;

/* Per-stage failure counts */
constexpr int MAX_STAGES = 32;
extern std::atomic<int> stage_fail_count[];

/* ============================================================
 * TestParams: GEMM parameters for a single test case
 * ============================================================ */

struct TestParams {
    PrecisionType precision;
    BLASINT M, N, K;
    CBLAS_TRANSPOSE transA, transB;
    CBLAS_ORDER order;
    float alpha, beta;
};

/* ParamsGenerator: callback that produces TestParams
 * Returns std::nullopt when no more test cases (signals worker to stop)
 */
using ParamsGenerator = std::function<std::optional<TestParams>()>;

/* ============================================================
 * ThreadArg: per-worker arguments
 * ============================================================ */

struct ThreadArg {
    int thread_id;
    int iterations;
    unsigned int rand_seed;
    int blas_threads = 1;
    int stage_num = 0;
    const char *dim_label = "";
    const char *blas_label = "";
    PrecisionType precision = PrecisionType::SGEMM;
    ParamsGenerator generator;
};

/* Thread worker function declaration */
void thread_worker(ThreadArg *targ);

#endif /* ST_WORKER_H */
