#ifndef GEMM_BENCHMARK_H
#define GEMM_BENCHMARK_H

#ifdef __cplusplus
extern "C" {
#endif

#include "unigemm_920f.h"

/* Benchmark configuration structure */
typedef struct {
    BLASINT m, n, k;
    float alpha, beta;
    int num_iterations;
    int warmup_iterations;
} GemmBenchmarkConfig;

/* Benchmark result structure */
typedef struct {
    double gflops;           /* GFLOPS achieved */
    double time_ms;          /* Time in milliseconds */
    int iterations;          /* Number of iterations performed */
} GemmBenchmarkResult;

/**
 * Run a SGEMM benchmark
 *
 * @param config Benchmark configuration
 * @param result Output parameter for benchmark results
 * @return 0 on success, non-zero on error
 */
int gemm_benchmark(const GemmBenchmarkConfig *config, GemmBenchmarkResult *result);

/**
 * Run multiple SGEMM benchmarks with different configurations
 *
 * @param configs Array of benchmark configurations
 * @param num_configs Number of configurations
 * @param results Array to store results (must be pre-allocated)
 * @return Number of successful benchmarks
 */
int gemm_benchmark_multiple(const GemmBenchmarkConfig *configs,
                              int num_configs,
                              GemmBenchmarkResult *results);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_BENCHMARK_H */
