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
    /* Calculate iterations per thread - distribute remainder to first workers */
    int iterations_per_worker = total_iterations / num_threads;
    int remainder = total_iterations % num_threads;

    std::cout << "  ├─ Workers: " << num_threads << "\n";
    std::cout << "  ├─ BLAS threads/worker: " << blas_threads << " (total: " << (num_threads * blas_threads) << " threads)\n";
    std::cout << "  └─ Iterations: " << total_iterations << "\n";

    /* Initialize thread data */
    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::thread> threads;

    /* Create threads and allocate buffers */
    for (int i = 0; i < num_threads; i++) {
        targs[i].thread_id = i;
        /* First 'remainder' workers get one extra iteration */
        targs[i].iterations = iterations_per_worker + (i < remainder ? 1 : 0);
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

    auto start_time = std::chrono::steady_clock::now();

    if (manual_config) {
        /* 单阶段测试：手动配置 */
        if (num_threads < 1) num_threads = 1;
        int blas_threads = (manual_blas_threads > 0) ? manual_blas_threads : 1;

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "  UniGEMM Fuzz Test - Manual Configuration\n";
        std::cout << std::string(70, '=') << "\n\n";

        if (run_test_stage(num_threads, blas_threads, total_iterations, seed) != 0) {
            return 1;
        }
    } else {
        /* 多阶段测试：自动配置 */
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "  UniGEMM Fuzz Test - Multi-Stage Auto Configuration\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  Configuration: Workers=" << max_workers << " | Stages=" << blas_threads_modes.size() << " (BLAS: [";
        for (size_t i = 0; i < blas_threads_modes.size(); i++) {
            std::cout << blas_threads_modes[i];
            if (i < blas_threads_modes.size() - 1) std::cout << ",";
        }
        std::cout << "])\n";
        std::cout << std::string(70, '=') << "\n\n";

        int stage = 1;
        for (int blas_threads : blas_threads_modes) {
            int workers = max_workers;
            if (workers < 1) workers = 1;

            auto stage_start = std::chrono::steady_clock::now();

            std::cout << "┌─ Stage " << stage << "/" << blas_threads_modes.size() << " (BLAS threads=" << blas_threads << ")\n";
            if (run_test_stage(workers, blas_threads, total_iterations, seed) != 0) {
                return 1;
            }

            auto stage_end = std::chrono::steady_clock::now();
            auto stage_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
            std::cout << "  └─ Completed in " << stage_duration.count() << " ms\n\n";

            stage++;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    /* Get statistics */
    int total = total_tests.load(std::memory_order_relaxed);
    int passed = passed_tests.load(std::memory_order_relaxed);
    int failed = failed_tests.load(std::memory_order_relaxed);

    /* Print failures first (if any) */
    if (failed > 0) {
        std::cout << "\n[ Failure Details (first " << failure_count << ") ]\n";
        std::cout << std::string(70, '-') << "\n";
        for (int i = 0; i < failure_count; i++) {
            print_failure(failures[i]);
        }
        std::cout << std::string(70, '-') << "\n";
    }

    /* Print results last */
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  Final Results\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "  Total Tests:    " << std::setw(10) << total << "  |  Passed:  " << std::setw(10) << passed << "  |  Failed:  " << std::setw(10) << failed << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  Error Rate:     " << std::setw(9) << (total > 0 ? (100.0 * failed / total) : 0.0) << "%  |  Time:    " << std::setw(10) << total_duration.count() << " ms\n";
    std::cout << std::string(70, '=') << "\n";

    return (failed > 0) ? 1 : 0;
}
