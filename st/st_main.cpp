#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "st_common.h"
#include "st_worker.h"
#include "st_csv_cases.h"

/* ============================================================
 * 全局计数器 — 多线程安全，worker 线程写入，主线程读取
 * ============================================================ */

std::mutex console_mutex;               // 保护 stdout 输出不交错

std::atomic<int> total_tests{0};
std::atomic<int> passed_tests{0};
std::atomic<int> failed_tests{0};
std::atomic<int> completed_tests{0};    // 进度条用：已完成的测试总数

// 按尺寸分类的完成计数，进度条分区显示
std::atomic<int> completed_small{0};
std::atomic<int> completed_medium{0};
std::atomic<int> completed_large{0};

std::atomic<int> stage_fail_count[MAX_STAGES]; // 每个 stage 的失败次数，用于汇总报告

/* ============================================================
 * 辅助函数：精度类型解析
 * ============================================================ */

// 精度枚举 → 短名称字符串，用于日志输出
static inline const char* precision_short_name(PrecisionType p) {
    switch (p) {
        case PrecisionType::SGEMM:  return "SGEMM";
        case PrecisionType::SHGEMM: return "SHGEMM";
        case PrecisionType::SBGEMM: return "SBGEMM";
        case PrecisionType::HGEMM:  return "HGEMM";
        case PrecisionType::BGEMM:  return "BGEMM";
    }
    return "?";
}

// 解析命令行精度参数字符串，不区分大小写
static inline bool parse_precision(const char* s, PrecisionType& out) {
    if (strcasecmp(s, "sgemm") == 0)  { out = PrecisionType::SGEMM;  return true; }
    if (strcasecmp(s, "shgemm") == 0) { out = PrecisionType::SHGEMM; return true; }
    if (strcasecmp(s, "sbgemm") == 0) { out = PrecisionType::SBGEMM; return true; }
    if (strcasecmp(s, "hgemm") == 0)  { out = PrecisionType::HGEMM;  return true; }
    if (strcasecmp(s, "bgemm") == 0)  { out = PrecisionType::BGEMM;  return true; }
    return false;
}

/* ============================================================
 * 进度条 — 独立线程运行，每 100ms 刷新一次终端
 * ============================================================ */

static std::atomic<bool> progress_running{false};
static std::thread progress_thread;
static int progress_target = 0;

// 进度条线程函数：并行于 worker 线程运行，读取原子计数器更新显示
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

            {
                std::lock_guard<std::mutex> lock(console_mutex);
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
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// 启动进度条线程
static void start_progress(int target) {
    progress_target = target;
    completed_tests.store(0, std::memory_order_relaxed);
    progress_running.store(true);
    progress_thread = std::thread(progress_bar_func);
}

// 停止进度条线程并清除进度条行
static void stop_progress() {
    progress_running.store(false);
    if (progress_thread.joinable()) progress_thread.join();
    std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
}


/* ============================================================
 * 单个测试阶段：启动多线程并行执行 GEMM 测试
 * ============================================================ */

// 运行一个 stage：将 total_iterations 次测试分配给 num_threads 个 worker 并行执行
// 每个 worker 内部还可能用多线程 BLAS（由 blas_threads 控制）→ 两层并行
static int run_test_stage(int num_threads, int blas_threads, int dim_range,
                          int total_iterations, unsigned int base_seed,
                          PrecisionType precision = PrecisionType::SGEMM,
                          int stage_num = 0,
                          const char *dim_label = "",
                          const char *blas_label = "") {
    // 将迭代次数均分给各 worker，余数分配给前几个
    int iterations_per_worker = total_iterations / num_threads;
    int remainder = total_iterations % num_threads;

    std::cout << "  ├─ Workers: " << num_threads << "\n";
    if (dim_range > 0) {
        std::cout << "  ├─ Dim range: 1-" << dim_range << "\n";
    }
    if (blas_threads == 0) {
        std::cout << "  ├─ Threads/worker: random 2-" << MAX_BLAS_THREADS << " (weighted)\n";
    } else {
        std::cout << "  ├─ Threads/worker: " << blas_threads << " (total: " << (num_threads * blas_threads) << ")\n";
    }
    std::cout << "  └─ Iterations: " << total_iterations << "\n";

    start_progress(total_iterations);

    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::thread> threads;

    // 并行启动所有 worker 线程
    for (int i = 0; i < num_threads; i++) {
        targs[i].thread_id = i;
        targs[i].iterations = iterations_per_worker + (i < remainder ? 1 : 0);
        targs[i].rand_seed = base_seed + i * 7919;  // 每个 worker 独立种子偏移
        targs[i].blas_threads = blas_threads;
        targs[i].precision = precision;
    targs[i].stage_num = stage_num;
        targs[i].dim_label = dim_label;
        targs[i].blas_label = blas_label;

        // 每个 worker 有独立的 RNG，不用 thread_local，确保跨 stage 时 seed 生效
        unsigned int rng_seed = base_seed + i * 7919;
        targs[i].generator = [rng_seed, dim_range, precision]() -> std::optional<TestParams> {
            RandomGenerator rng(rng_seed);
            TestParams p;
            p.precision = precision;
            p.order = rng.random_order();
            p.transA = rng.random_transpose();
            p.transB = rng.random_transpose();
            if (dim_range > 0)
                rng.random_three_dims_fixed(p.M, p.N, p.K, dim_range);
            else
                rng.random_three_dims(p.M, p.N, p.K);
            p.alpha = rng.random_alpha_beta();
            p.beta = rng.random_alpha_beta();
            return p;
        };

        threads.emplace_back(thread_worker, &targs[i]);
    }

    // 等待所有 worker 线程完成
    for (auto& t : threads) t.join();
    stop_progress();

    return 0;
}

/* ============================================================
 * CSV 模式：运行编译期嵌入的测试用例
 * ============================================================ */

static std::atomic<int> csv_next{0}; // 原子索引，多线程安全地分配 CSV 用例

static int run_csv_mode(int num_threads, int blas_threads, int seed) {
    csv_next.store(0);

    // 所有 worker 共享同一个生成器，通过原子 fetch_add 无锁分配用例
    auto generator = []() -> std::optional<TestParams> {
        int idx = csv_next.fetch_add(1);
        if (idx >= g_csv_case_count) return std::nullopt;
        return g_csv_cases[idx];
    };

    int total_cases = g_csv_case_count;
    int iterations_per_worker = total_cases / num_threads;
    int remainder = total_cases % num_threads;

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  System Test - CSV Mode\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "  Cases: " << total_cases << "\n";
    std::cout << "  Workers: " << num_threads << "\n";
    std::cout << "  Seed: " << seed << "\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::vector<ThreadArg> targs(num_threads);
    std::vector<std::thread> threads;

    // 并行启动 worker，每个 worker 分配一段连续的 CSV 用例索引
    for (int i = 0; i < num_threads; i++) {
        targs[i].thread_id = i;
        targs[i].iterations = iterations_per_worker + (i < remainder ? 1 : 0);
        targs[i].rand_seed = seed + i * 7919;
        targs[i].blas_threads = blas_threads;
        targs[i].stage_num = 0;
        targs[i].generator = generator;

        threads.emplace_back(thread_worker, &targs[i]);
    }

    for (auto& t : threads) t.join();

    return 0;
}

/* ============================================================
 * 随机 30 阶段模式：5 精度 × 3 尺寸 × 2 BLAS 线程模式
 * ============================================================ */

static int run_random_30stage(int max_workers, int total_iterations,
                               unsigned int seed, const PrecisionType* single_prec) {
    constexpr int DIM_PROB_TOTAL = DIM_PROB_SMALL + DIM_PROB_MEDIUM + DIM_PROB_LARGE;

    struct StageConfig {
        int stage_num;
        const char *dim_label;
        const char *precision_label;
        int dim_range;
        int blas_threads;
        int iters;
        PrecisionType precision;
    };

    // 全部精度定义，每种精度占 6 个 stage（3 尺寸 × 2 BLAS 模式）
    struct PrecEntry { const char* label; PrecisionType prec; int offset; };
    constexpr PrecEntry ALL_PRECS[] = {
        {"SGEMM",  PrecisionType::SGEMM,  0},
        {"SHGEMM", PrecisionType::SHGEMM, 6},
        {"SBGEMM", PrecisionType::SBGEMM, 12},
        {"HGEMM",  PrecisionType::HGEMM,  18},
        {"BGEMM",  PrecisionType::BGEMM,  24},
    };

    std::vector<StageConfig> stages;

    // 为一种精度生成 6 个 stage：Small/Medium/Large 各分 single-thread 和 multi-thread
    auto add_precision_stages = [&](const char* label, PrecisionType prec, int stage_offset) {
        // 按概率权重分配迭代次数到三个尺寸区间
        int s = total_iterations * DIM_PROB_SMALL  / DIM_PROB_TOTAL;
        int m = total_iterations * DIM_PROB_MEDIUM / DIM_PROB_TOTAL;
        int l = total_iterations - s - m;
        // 每个尺寸再均分给 single-thread 和 multi-thread 两种 BLAS 模式
        int s1 = s/2, s2 = s - s1;
        int m1 = m/2, m2 = m - m1;
        int l1 = l/2, l2 = l - l1;
        stages.push_back({stage_offset+1, "Small",  label, DIM_RANGE_SMALL,  1, s1, prec});
        stages.push_back({stage_offset+2, "Small",  label, DIM_RANGE_SMALL,  0, s2, prec});
        stages.push_back({stage_offset+3, "Medium", label, DIM_RANGE_MEDIUM, 1, m1, prec});
        stages.push_back({stage_offset+4, "Medium", label, DIM_RANGE_MEDIUM, 0, m2, prec});
        stages.push_back({stage_offset+5, "Large",  label, DIM_RANGE_LARGE,  1, l1, prec});
        stages.push_back({stage_offset+6, "Large",  label, DIM_RANGE_LARGE,  0, l2, prec});
    };

    // 根据是否指定了单精度过滤，生成全部或部分 stage
    for (const auto& pc : ALL_PRECS) {
        if (!single_prec || *single_prec == pc.prec)
            add_precision_stages(pc.label, pc.prec, pc.offset);
    }

    int total_stage_iters = 0;
    for (const auto& s : stages) total_stage_iters += s.iters;

    // 打印测试头部信息
    std::cout << "\n" << std::string(70, '=') << "\n";
    if (single_prec) {
        std::cout << "  System Test - " << precision_short_name(*single_prec) << " Random Mode\n";
    } else {
        std::cout << "  System Test - Thirty-Stage Random Mode\n";
    }
    std::cout << std::string(70, '-') << "\n";
    std::cout << "  Workers=" << max_workers << " | Iterations/precision=" << total_iterations
              << " | Stages=" << stages.size() << " | Total=" << total_stage_iters << "\n";
    std::cout << std::string(70, '=') << "\n\n";

    // 依次执行每个 stage（每个 stage 内部是多线程并行的）
    for (const auto& s : stages) {
        if (s.iters <= 0) continue;

        auto stage_start = std::chrono::steady_clock::now();

        const char *blas_label = (s.blas_threads == 1) ? "single thread" : "multi thread";
        std::cout << "┌─ Stage " << s.stage_num << "/30 " << s.dim_label << " "
                  << s.precision_label << " " << blas_label << "\n";

        if (run_test_stage(max_workers, s.blas_threads, s.dim_range, s.iters, seed, s.precision,
                           s.stage_num, s.dim_label, blas_label) != 0) {
            return 1;
        }

        auto stage_end = std::chrono::steady_clock::now();
        auto stage_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
        std::cout << "  └─ Completed in " << stage_duration.count() << " ms\n\n";
    }

    return 0;
}

/* ============================================================
 * 结果报告（所有模式共用）
 * ============================================================ */

// 打印每个 stage 的失败次数汇总
static void print_failure_summary() {
    static const char *stage_meta[][3] = {
        {"", "", ""},
        {"SGEMM",  "Small",  "single"}, {"SGEMM",  "Small",  "multi"},
        {"SGEMM",  "Medium", "single"}, {"SGEMM",  "Medium", "multi"},
        {"SGEMM",  "Large",  "single"}, {"SGEMM",  "Large",  "multi"},
        {"SHGEMM", "Small",  "single"}, {"SHGEMM", "Small",  "multi"},
        {"SHGEMM", "Medium", "single"}, {"SHGEMM", "Medium", "multi"},
        {"SHGEMM", "Large",  "single"}, {"SHGEMM", "Large",  "multi"},
        {"SBGEMM", "Small",  "single"}, {"SBGEMM", "Small",  "multi"},
        {"SBGEMM", "Medium", "single"}, {"SBGEMM", "Medium", "multi"},
        {"SBGEMM", "Large",  "single"}, {"SBGEMM", "Large",  "multi"},
        {"HGEMM",  "Small",  "single"}, {"HGEMM",  "Small",  "multi"},
        {"HGEMM",  "Medium", "single"}, {"HGEMM",  "Medium", "multi"},
        {"HGEMM",  "Large",  "single"}, {"HGEMM",  "Large",  "multi"},
        {"BGEMM",  "Small",  "single"}, {"BGEMM",  "Small",  "multi"},
        {"BGEMM",  "Medium", "single"}, {"BGEMM",  "Medium", "multi"},
        {"BGEMM",  "Large",  "single"}, {"BGEMM",  "Large",  "multi"},
    };
    std::cout << "  Stage Failure Summary:\n";
    for (int sn = 1; sn <= 30; sn++) {
        int fc = stage_fail_count[sn].load(std::memory_order_relaxed);
        if (fc == 0) continue;
        const char *blas_label = (strcmp(stage_meta[sn][2], "single") == 0)
                                 ? "single thread" : "multi thread";
        std::cout << "    Stage " << std::setw(2) << sn << "/30  "
                  << std::setw(6) << stage_meta[sn][0] << "  "
                  << std::setw(6) << stage_meta[sn][1] << "  "
                  << std::setw(13) << blas_label << ": "
                  << std::setw(5) << fc << " failures\n";
    }
}

// 打印最终结果，返回 exit code
static int print_results(long long duration_ms) {
    int total = total_tests.load(std::memory_order_relaxed);
    int passed = passed_tests.load(std::memory_order_relaxed);
    int failed = failed_tests.load(std::memory_order_relaxed);

    if (failed > 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        print_failure_summary();
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    if (failed == 0) {
        std::cout << "  ✓ All Success! " << std::string(50, '=') << "\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  Total Tests:     " << std::setw(10) << total
                  << "  |  Time:    " << std::setw(10) << duration_ms << " ms\n";
    } else {
        std::cout << "  Final Results\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  Total Tests:    " << std::setw(10) << total
                  << "  |  Passed:  " << std::setw(10) << passed
                  << "  |  Failed:  " << std::setw(10) << failed << "\n";
        std::cout << std::setprecision(4);
        std::cout << "  Error Rate:     " << std::setw(9) << (total > 0 ? (100.0 * failed / total) : 0.0)
                  << "%  |  Time:    " << std::setw(10) << duration_ms << " ms\n";
    }
    std::cout << std::string(70, '=') << "\n";
    return (failed > 0) ? 1 : 0;
}

/* ============================================================
 * 各模式入口函数：自行解析参数并执行
 * ============================================================ */

// 从 argv 中查找 --key value
static const char* find_option(int argc, char *argv[], const char *key) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return nullptr;
}

// csv 模式
static void cmd_csv(int argc, char *argv[]) {
    int seed = 42;
    int workers = 32;
    if (auto v = find_option(argc, argv, "--seed"))    seed = std::atoi(v);
    if (auto v = find_option(argc, argv, "--workers")) workers = std::atoi(v);

    auto start = std::chrono::steady_clock::now();
    run_csv_mode(workers, 1, seed);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    exit(print_results(ms));
}

// random 模式
static void cmd_random(int argc, char *argv[]) {
    int iterations = 100;
    int workers = 32;
    PrecisionType single_prec;
    bool has_single = false;

    if (auto v = find_option(argc, argv, "--iteration")) iterations = std::atoi(v);
    if (auto v = find_option(argc, argv, "--workers"))   workers = std::atoi(v);
    if (auto v = find_option(argc, argv, "--precision")) {
        if (!parse_precision(v, single_prec)) {
            std::cerr << "Error: Unknown precision '" << v << "'\n";
            exit(1);
        }
        has_single = true;
    }

    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    auto start = std::chrono::steady_clock::now();
    run_random_30stage(workers, iterations, seed,
                       has_single ? &single_prec : nullptr);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    exit(print_results(ms));
}

// full 模式（csv + random）
static void cmd_full(int argc, char *argv[]) {
    int iterations = 100;
    int csv_seed = 42;
    int workers = 32;
    PrecisionType single_prec;
    bool has_single = false;

    if (auto v = find_option(argc, argv, "--iteration")) iterations = std::atoi(v);
    if (auto v = find_option(argc, argv, "--seed"))      csv_seed = std::atoi(v);
    if (auto v = find_option(argc, argv, "--workers"))   workers = std::atoi(v);
    if (auto v = find_option(argc, argv, "--precision")) {
        if (!parse_precision(v, single_prec)) {
            std::cerr << "Error: Unknown precision '" << v << "'\n";
            exit(1);
        }
        has_single = true;
    }

    auto start = std::chrono::steady_clock::now();
    run_csv_mode(workers, 1, csv_seed);

    unsigned int seed = static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count());
    run_random_30stage(workers, iterations, seed,
                       has_single ? &single_prec : nullptr);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    exit(print_results(ms));
}

// 打印帮助信息
static void cmd_help(int, char *argv[]) {
    std::cout << "Usage: " << argv[0] << " <mode> [OPTIONS]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  csv        Run embedded CSV test cases\n";
    std::cout << "  random     Run random 30-stage test\n";
    std::cout << "  full       Run both csv + random (default)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --precision <name>  Single precision: sgemm, shgemm, sbgemm, hgemm, bgemm\n";
    std::cout << "  --iteration <N>     Iterations per precision (default: 100)\n";
    std::cout << "  --seed <N>          Seed for CSV matrix generation (default: 42)\n";
    std::cout << "  --workers <N>       Max worker threads (default: 32)\n";
    exit(0);
}

/* ============================================================
 * 函数表派发
 * ============================================================ */

static const std::unordered_map<std::string, void(*)(int, char*[])> g_functionTable = {
    {"csv",       cmd_csv},
    {"--csv",     cmd_csv},
    {"random",    cmd_random},
    {"--random",  cmd_random},
    {"full",      cmd_full},
    {"--full",    cmd_full},
    {"help",      cmd_help},
    {"-h",        cmd_help},
    {"--help",    cmd_help},
};

int main(int argc, char *argv[]) {
    const char *mode = (argc > 1) ? argv[1] : "full";

    auto it = g_functionTable.find(mode);
    if (it == g_functionTable.end()) {
        std::cerr << "Error: Unknown mode '" << mode << "'\n";
        std::cerr << "  Options: csv, random, full\n";
        return 1;
    }

    it->second(argc, argv);
    return 0;
}
