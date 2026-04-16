#ifndef FUZZ_TEST_WORKER_H
#define FUZZ_TEST_WORKER_H

#include "fuzz_test_config.h"
#include "fuzz_test_failure.h"
#include "fuzz_test_buffer.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

/* Atomic counters for statistics */
extern std::atomic<int> total_tests;
extern std::atomic<int> passed_tests;
extern std::atomic<int> failed_tests;
extern std::atomic<int> completed_tests;  /* For progress tracking */

/* Completed tests by size category */
extern std::atomic<int> completed_small;
extern std::atomic<int> completed_medium;
extern std::atomic<int> completed_large;

/* Mutex for failure log */
extern std::mutex fail_mutex;

/* Failure log array */
extern FailureInfo failures[];
extern int failure_count;

/* Thread worker function arguments */
struct ThreadArg {
    int thread_id;
    int iterations;
    unsigned int rand_seed;
    int blas_threads = 1;   /* Fixed BLAS thread count for this worker */
    int dim_range = 0;       /* 0=随机类别, 128/512/1024=固定维度范围 */
    std::unique_ptr<ThreadBuffers> buffers;
};

/* Thread worker function declaration */
void thread_worker(ThreadArg* targ);

#endif /* FUZZ_TEST_WORKER_H */
