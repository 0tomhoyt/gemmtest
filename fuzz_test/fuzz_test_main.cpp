#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdlib>

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include "fuzz_test_worker.h"
#include "fuzz_test_report.cpp"

int main(int argc, char *argv[]) {
    int num_threads = 4;
    int total_iterations = 100;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--thread") == 0 && i + 1 < argc) {
            num_threads = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--iteration") == 0 && i + 1 < argc) {
            total_iterations = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [--thread <threads>] [--iteration <total_iterations>]\n";
            std::cout << "  --thread <threads>     Number of threads (default: 4)\n";
            std::cout << "  --iteration <total>    Total iterations across all threads (default: 100)\n";
            return 0;
        }
    }

    if (num_threads < 1) num_threads = 1;
    if (total_iterations < 1) total_iterations = 1;

    /* Calculate iterations per thread (last thread handles remainder) */
    int base_iterations = total_iterations / num_threads;
    int remainder = total_iterations % num_threads;

    std::cout << "Starting fuzz test:\n";
    std::cout << "  Threads: " << num_threads << "\n";
    std::cout << "  Total iterations: " << total_iterations << "\n";
    std::cout << "  Iterations per thread: " << base_iterations;
    if (remainder > 0) {
        std::cout << " (last thread: " << (base_iterations + remainder) << ")";
    }
    std::cout << "\n\n";

    /* Initialize thread data */
    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::thread> threads;

    /* Initialize RNG with time */
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    /* Create threads and allocate buffers */
    for (int i = 0; i < num_threads; i++) {
        targs[i].thread_id = i;
        /* Last thread handles remainder iterations */
        targs[i].iterations = base_iterations + (i == num_threads - 1 ? remainder : 0);
        targs[i].rand_seed = seed + i * 7919;  /* Use prime number for better distribution */

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
    std::cout << "Results:\n";
    std::cout << "  Total:   " << total << "\n";
    std::cout << "  Passed:  " << passed << "\n";
    std::cout << "  Failed:  " << failed << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  Error rate: " << (total > 0 ? (100.0 * failed / total) : 0.0) << "%\n";
    std::cout << "========================================\n";

    return (failed > 0) ? 1 : 0;
}
