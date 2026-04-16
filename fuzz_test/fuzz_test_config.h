#ifndef FUZZ_TEST_CONFIG_H
#define FUZZ_TEST_CONFIG_H

/* ============================================================
 * 维度测试范围配置 (Dimension Test Range Configuration)
 * ============================================================ */

/* 维度范围定义 */
constexpr int DIM_RANGE_SMALL = 128;      // 小维度范围
constexpr int DIM_RANGE_MEDIUM = 512;    // 中维度范围
constexpr int DIM_RANGE_LARGE = 1024;     // 大维度范围

/* 维度分布概率 (总和建议为 100) */
constexpr int DIM_PROB_SMALL = 40;       // 0-128 范围概率 (%)
constexpr int DIM_PROB_MEDIUM = 40;      // 0-512 范围概率 (%)
constexpr int DIM_PROB_LARGE = 20;       // 0-1024 范围概率 (%)

/* 编译时检查：确保概率总和为100 */
static_assert(DIM_PROB_SMALL + DIM_PROB_MEDIUM + DIM_PROB_LARGE == 100,
              "DIM_PROB_SMALL + DIM_PROB_MEDIUM + DIM_PROB_LARGE must equal 100");

/* 计算最大需要的内存 (加上 padding) */
constexpr int MAX_DIM = DIM_RANGE_LARGE;
constexpr int MAX_LD = DIM_RANGE_LARGE + 7;  // 最大维度 + 最大 padding

constexpr int MAX_FAIL_LOGS = 20;

/* 缓冲区对齐要求 (字节) */
constexpr int BUFFER_ALIGNMENT = 64;

/* ============================================================
 * 线程配置 (Threading Configuration)
 * ============================================================ */

/* 最大并行 worker 线程数 (不设置则运行时自动检测 CPU 核数)
 * 每个测试阶段的 std::thread 数量 = MAX_WORKERS
 */
#ifdef USE_HBM
#define MAX_WORKERS 32  /* HBM 模式默认 32 worker */
#else
#define MAX_WORKERS 8  /* 非 HBM 模式，8 worker */
#endif

/* BLAS 线程数范围配置 (用于多线程阶段)
 * 多线程阶段将使用 2-MAX_BLAS_THREADS 范围内的随机值
 * 线程数越大，随机到的概率越低 (线性衰减)
 */
#define MAX_BLAS_THREADS 50

#endif /* FUZZ_TEST_CONFIG_H */
