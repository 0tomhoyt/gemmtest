#include "fuzz_test_worker.h"
#include "fuzz_test_config.h"

/* Atomic counters for statistics */
std::atomic<int> total_tests{0};
std::atomic<int> passed_tests{0};
std::atomic<int> failed_tests{0};
std::atomic<int> completed_tests{0};  /* For progress tracking */

/* Mutex for failure log */
std::mutex fail_mutex;

/* Failure log array */
FailureInfo failures[MAX_FAIL_LOGS];
int failure_count = 0;
