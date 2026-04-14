#ifndef FUZZ_TEST_CONFIG_H
#define FUZZ_TEST_CONFIG_H

/* ============================================================
 * 维度测试范围配置 (Dimension Test Range Configuration)
 * ============================================================ */

/* 维度范围定义 */
constexpr int DIM_RANGE_SMALL = 64;      // 小维度范围
constexpr int DIM_RANGE_MEDIUM = 128;    // 中维度范围
constexpr int DIM_RANGE_LARGE = 512;     // 大维度范围

/* 维度分布概率 (总和建议为 100) */
constexpr int DIM_PROB_SMALL = 10;       // 0-64 范围概率 (%)
constexpr int DIM_PROB_MEDIUM = 40;      // 0-512 范围概率 (%)
constexpr int DIM_PROB_LARGE = 50;       // 0-1024 范围概率 (%)

/* 计算最大需要的内存 (加上 padding) */
constexpr int MAX_DIM = DIM_RANGE_LARGE;
constexpr int MAX_LD = DIM_RANGE_LARGE + 7;  // 最大维度 + 最大 padding

constexpr int MAX_FAIL_LOGS = 20;

/* 缓冲区对齐要求 (字节) */
constexpr int BUFFER_ALIGNMENT = 64;

/* ============================================================
 * 线程配置 (Threading Configuration)
 * ============================================================ */

/* 最大使用的CPU核数 (不设置则运行时自动检测) */
// #define MAX_CORES 100

/* 要测试的BLAS线程模式列表 (线程计数数组)
 * 每个模式将作为独立的测试阶段运行，其中：
 *   num_workers = MAX_CORES / blas_threads
 */
#ifndef BLAS_THREADS_MODES
#define BLAS_THREADS_MODES {1, 8}
#endif

#endif /* FUZZ_TEST_CONFIG_H */
