#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <array>
#include <atomic>
#include <mutex>

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include "fuzz_test_worker.h"
#include "fuzz_test_report.cpp"

/* Progress bar state */
static std::atomic<bool> progress_running{false};
static std::thread progress_thread;
static int progress_target = 0;

/* Progress bar display function */
static void progress_bar_func() {
    const int bar_width = 15;
    while (progress_running.load()) {
        int current = completed_tests.load(std::memory_order_relaxed);
        int small = completed_small.load(std::memory_order_relaxed);
        int medium = completed_medium.load(std::memory_order_relaxed);
        int large = completed_large.load(std::memory_order_relaxed);

        if (progress_target > 0) {
            float total_ratio = static_cast<float>(current) / progress_target;
            if (total_ratio > 1.0f) total_ratio = 1.0f;
            int total_filled = static_cast<int>(total_ratio * bar_width);

            /* Build single-line progress display */
            std::cout << "\r  ";
            std::cout << "S[" << std::setw(3) << small << "] ";
            std::cout << "M[" << std::setw(3) << medium << "] ";
            std::cout << "L[" << std::setw(3) << large << "] ";
            std::cout << " [";
            for (int i = 0; i < bar_width; i++) {
                if (i < total_filled) std::cout << "=";
                else if (i == total_filled) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::setw(3) << static_cast<int>(total_ratio * 100) << "% ("
                      << std::setw(4) << current << "/" << progress_target << ")" << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/* Start progress bar */
static void start_progress(int target) {
    progress_target = target;
    completed_tests.store(0, std::memory_order_relaxed);
    progress_running.store(true);
    progress_thread = std::thread(progress_bar_func);
}

/* Stop progress bar */
static void stop_progress() {
    progress_running.store(false);
    if (progress_thread.joinable()) {
        progress_thread.join();
    }
    /* Clear the progress bar line */
    std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
}

/* 运行单阶段测试 */
static int run_test_stage(int num_threads, int blas_threads, int dim_range,
                          int total_iterations, unsigned int base_seed) {
    /* Calculate iterations per thread - distribute remainder to first workers */
    int iterations_per_worker = total_iterations / num_threads;
    int remainder = total_iterations % num_threads;

    std::cout << "  ├─ Workers: " << num_threads << "\n";
    if (dim_range > 0) {
        std::cout << "  ├─ Dim range: 1-" << dim_range << "\n";
    }
    if (blas_threads == 0) {
        std::cout << "  ├─ BLAS threads/worker: random 2-" << MAX_BLAS_THREADS << " (weighted)\n";
    } else {
        std::cout << "  ├─ BLAS threads/worker: " << blas_threads << " (total: " << (num_threads * blas_threads) << " threads)\n";
    }
    std::cout << "  └─ Iterations: " << total_iterations << "\n";

    /* Start progress bar */
    start_progress(total_iterations);

    /* Initialize thread data */
    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::thread> threads;

    /* Create threads and allocate buffers */
    for (int i = 0; i < num_threads; i++) {
        targs[i].thread_id = i;
        /* First 'remainder' workers get one extra iteration */
        targs[i].iterations = iterations_per_worker + (i < remainder ? 1 : 0);
        targs[i].rand_seed = base_seed + i * 7919;  /* Use prime number for better distribution */
        targs[i].blas_threads = blas_threads;  /* Fixed BLAS thread count for this worker */
        targs[i].dim_range = dim_range;         /* Fixed dimension range for this stage */

        /* Allocate buffers for this thread */
        targs[i].buffers = alloc_thread_buffers(MAX_DIM, MAX_LD);
        if (!targs[i].buffers) {
            std::cerr << "Error: Failed to allocate buffers for thread " << i << "\n";
            stop_progress();
            return 1;
        }

        threads.emplace_back(thread_worker, &targs[i]);
    }

    /* Wait for threads to complete */
    for (auto& t : threads) {
        t.join();
    }

    /* Stop progress bar */
    stop_progress();

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
            std::cout << "Usage: " << argv[0] << " [--thread <threads>] [--iteration <total_iterations>]\n";
            std::cout << "  --thread <threads>       Number of worker threads (default: auto-calculated)\n";
            std::cout << "  --iteration <total>      Total iterations across all threads (default: 100)\n";
            std::cout << "\nNote: Without --thread, runs six-stage test (3 dims x 2 BLAS modes):\n";
            std::cout << "  Small (1-128):   40%% of iterations, split single/multi BLAS\n";
            std::cout << "  Medium (1-512):  40%% of iterations, split single/multi BLAS\n";
            std::cout << "  Large (1-1024):  20%% of iterations, split single/multi BLAS\n";
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

        if (run_test_stage(num_threads, blas_threads, 0, total_iterations, seed) != 0) {
            return 1;
        }
    } else {
        /* 六阶段测试：3 维度 × 2 BLAS 线程模式 */
        /* 迭代分配：small 40%, medium 40%, large 20% */
        int small_total  = total_iterations * 40 / 100;
        int medium_total = total_iterations * 40 / 100;
        int large_total  = total_iterations * 20 / 100;
        int remainder = total_iterations - small_total - medium_total - large_total;
        small_total += remainder;  /* 余数给 small 阶段 */

        /* 每种维度内 single/multi 各半 */
        int small_single  = small_total / 2;
        int small_multi   = small_total - small_single;
        int medium_single = medium_total / 2;
        int medium_multi  = medium_total - medium_single;
        int large_single  = large_total / 2;
        int large_multi   = large_total - large_single;

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "  UniGEMM Fuzz Test - Six-Stage Auto Configuration\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  Workers=" << max_workers << " | Total Iterations=" << total_iterations << "\n";
        std::cout << "  Small: " << small_total << " | Medium: " << medium_total << " | Large: " << large_total << "\n";
        std::cout << std::string(70, '=') << "\n\n";

        struct StageConfig {
            int stage_num;
            const char* dim_label;
            int dim_range;
            int blas_threads;  /* 1=single, 0=multi */
            int iters;
        };

        StageConfig stages[] = {
            {1, "Small (1-128)",   128,  1, small_single},
            {2, "Small (1-128)",   128,  0, small_multi},
            {3, "Medium (1-512)",  512,  1, medium_single},
            {4, "Medium (1-512)",  512,  0, medium_multi},
            {5, "Large (1-1024)",  1024, 1, large_single},
            {6, "Large (1-1024)",  1024, 0, large_multi},
        };

        for (const auto& s : stages) {
            if (s.iters <= 0) continue;

            auto stage_start = std::chrono::steady_clock::now();

            const char* blas_label = (s.blas_threads == 1) ? "BLAS=1 (single)" : "BLAS=random 2-50 (multi)";
            std::cout << "┌─ Stage " << s.stage_num << "/6 [" << s.dim_label << "] " << blas_label << "\n";

            if (run_test_stage(max_workers, s.blas_threads, s.dim_range, s.iters, seed) != 0) {
                return 1;
            }

            auto stage_end = std::chrono::steady_clock::now();
            auto stage_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
            std::cout << "  └─ Completed in " << stage_duration.count() << " ms\n\n";
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
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "  Failure Details (first " << failure_count << ")\n";
        std::cout << std::string(70, '-') << "\n";
        for (int i = 0; i < failure_count; i++) {
            print_failure(failures[i]);
        }
        std::cout << std::string(70, '-') << "\n";
    }

    /* Print results last */
    std::cout << "\n" << std::string(70, '=') << "\n";

    if (failed == 0) {
        /* All tests passed - show success message */
        std::cout << "  ✓ All Success! " << std::string(50, '=') << "\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  Total Tests:     " << std::setw(10) << total << "  |  Time:    " << std::setw(10) << total_duration.count() << " ms\n";
        std::cout << std::string(70, '=') << "\n";
    } else {
        /* Some tests failed - show detailed stats */
        std::cout << "  Final Results\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  Total Tests:    " << std::setw(10) << total << "  |  Passed:  " << std::setw(10) << passed << "  |  Failed:  " << std::setw(10) << failed << "\n";
        std::cout << std::setprecision(4);
        std::cout << "  Error Rate:     " << std::setw(9) << (total > 0 ? (100.0 * failed / total) : 0.0) << "%  |  Time:    " << std::setw(10) << total_duration.count() << " ms\n";
        std::cout << std::string(70, '=') << "\n";
    }

    return (failed > 0) ? 1 : 0;
}
