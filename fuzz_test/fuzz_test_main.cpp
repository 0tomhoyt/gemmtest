#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <array>

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include "fuzz_test_worker.h"
#include "fuzz_test_report.cpp"

/* 运行单阶段测试 */
static int run_test_stage(int num_threads, int blas_threads, int total_iterations, unsigned int base_seed) {
    /* Calculate iterations per thread (discard remainder) */
    int iterations_per_worker = total_iterations / num_threads;
    int actual_iterations = iterations_per_worker * num_threads;

    std::cout << "  Workers: " << num_threads << "\n";
    std::cout << "  BLAS threads per worker: " << blas_threads << "\n";
    std::cout << "  Total threads: " << (num_threads * blas_threads) << "\n";
    std::cout << "  Requested iterations: " << total_iterations << "\n";
    std::cout << "  Actual iterations: " << actual_iterations << "\n";
    std::cout << "  Iterations per worker: " << iterations_per_worker << "\n\n";

    /* Initialize thread data */
    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::thread> threads;

    /* Create threads and allocate buffers */
    for (int i = 0; i < num_threads; i++) {
        targs[i].thread_id = i;
        targs[i].iterations = iterations_per_worker;
        targs[i].rand_seed = base_seed + i * 7919;  /* Use prime number for better distribution */
        targs[i].blas_threads = blas_threads;  /* Fixed BLAS thread count for this stage */

        /* Allocate buffers for this thread */
        targs[i].buffers = alloc_thread_buffers(MAX_DIM, MAX_LD);
        if (!targs[i].buffers) {
            std::cerr << "Error: Failed to allocate buffers for thread " << i << "\n";
            return 1;
        }

        threads.emplace_back(thread_worker, &targs[i]);
    }

    /* Wait for threads to complete */
    for (auto& t : threads) {
        t.join();
    }

    return 0;
}

int main(int argc, char *argv[]) {
    int num_threads = 4;
    int total_iterations = 100;

    /* 检测CPU核心数，作为 MAX_WORKERS 默认值 */
    int max_workers = std::thread::hardware_concurrency();

    /* 检查 UNIGEMM_MAX_WORKERS 环境变量（最高优先级）*/
    const char* env_workers = std::getenv("UNIGEMM_MAX_WORKERS");
    if (env_workers != nullptr) {
        int env_val = std::atoi(env_workers);
        if (env_val > 0 && env_val <= 1024) {  /* 安全上限检查 */
            max_workers = env_val;
        }
    }

#ifdef MAX_WORKERS
    max_workers = MAX_WORKERS;  /* 编译时配置 */
#endif
    if (max_workers < 1) max_workers = 4;  /* 回退值 */

    /* BLAS线程模式配置 */
#ifndef BLAS_THREADS_MODES
#define BLAS_THREADS_MODES {1, 8}
#endif
    std::vector<int> blas_threads_modes = BLAS_THREADS_MODES;

    /* 手动配置标志 */
    bool manual_config = false;
    int manual_blas_threads = -1;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--thread") == 0 && i + 1 < argc) {
            num_threads = std::atoi(argv[i + 1]);
            manual_config = true;  /* 手动配置时跳过多阶段测试 */
            i++;
        } else if (strcmp(argv[i], "--iteration") == 0 && i + 1 < argc) {
            total_iterations = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--blas-threads") == 0 && i + 1 < argc) {
            manual_blas_threads = std::atoi(argv[i + 1]);
            manual_config = true;  /* 手动配置时跳过多阶段测试 */
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [--thread <threads>] [--blas-threads <threads>] [--iteration <total_iterations>]\n";
            std::cout << "  --thread <threads>       Number of worker threads (default: auto-calculated)\n";
            std::cout << "  --blas-threads <threads> BLAS threads per worker (default: auto-calculated)\n";
            std::cout << "  --iteration <total>      Total iterations across all threads (default: 100)\n";
            std::cout << "\nNote: Without --thread, runs multi-stage test with BLAS_THREADS_MODES from config.\n";
            return 0;
        }
    }

    if (total_iterations < 1) total_iterations = 1;

    /* Initialize RNG with time */
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    if (manual_config) {
        /* 单阶段测试：手动配置 */
        if (num_threads < 1) num_threads = 1;
        int blas_threads = (manual_blas_threads > 0) ? manual_blas_threads : 1;

        std::cout << "Starting fuzz test (manual configuration):\n";
        std::cout << "========================================\n";

        if (run_test_stage(num_threads, blas_threads, total_iterations, seed) != 0) {
            return 1;
        }
    } else {
        /* 多阶段测试：自动配置 */
        std::cout << "Starting fuzz test (multi-stage):\n";
        std::cout << "  Workers: " << max_workers << "\n";
        std::cout << "  Test stages: " << blas_threads_modes.size() << "\n\n";

        int stage = 1;
        for (int blas_threads : blas_threads_modes) {
            int workers = max_workers;
            if (workers < 1) workers = 1;

            std::cout << "========================================\n";
            std::cout << "Stage " << stage << "/" << blas_threads_modes.size() << ": BLAS threads = " << blas_threads << "\n";
            std::cout << "========================================\n";

            if (run_test_stage(workers, blas_threads, total_iterations, seed) != 0) {
                return 1;
            }

            stage++;
            std::cout << "\n";
        }
    }

    /* Get statistics */
    int total = total_tests.load(std::memory_order_relaxed);
    int passed = passed_tests.load(std::memory_order_relaxed);
    int failed = failed_tests.load(std::memory_order_relaxed);

    /* Print failures first (if any) */
    if (failed > 0) {
        std::cout << "\nFirst " << failure_count << " failures:\n";
        for (int i = 0; i < failure_count; i++) {
            print_failure(failures[i]);
        }
    }

    /* Print results last */
    std::cout << "\n========================================\n";
    std::cout << "Final Results:\n";
    std::cout << "  Total:   " << total << "\n";
    std::cout << "  Passed:  " << passed << "\n";
    std::cout << "  Failed:  " << failed << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  Error rate: " << (total > 0 ? (100.0 * failed / total) : 0.0) << "%\n";
    std::cout << "========================================\n";

    return (failed > 0) ? 1 : 0;
}
