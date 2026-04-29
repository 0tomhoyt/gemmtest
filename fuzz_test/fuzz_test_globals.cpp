#include "fuzz_test_worker.h"
#include "fuzz_test_config.h"

/* Atomic counters for statistics */
std::atomic<int> total_tests{0};
std::atomic<int> passed_tests{0};
std::atomic<int> failed_tests{0};
std::atomic<int> completed_tests{0};  /* For progress tracking */

/* Completed tests by size category */
std::atomic<int> completed_small{0};
std::atomic<int> completed_medium{0};
std::atomic<int> completed_large{0};

/* Per-stage failure counts */
std::atomic<int> stage_fail_count[MAX_STAGES];
