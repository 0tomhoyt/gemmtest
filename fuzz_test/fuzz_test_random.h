#ifndef FUZZ_TEST_RANDOM_H
#define FUZZ_TEST_RANDOM_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include <random>
#include <array>
#include <algorithm>
#include <cstdio>
#include <mutex>

extern std::mutex console_mutex;
#include <cmath>

/* Random parameter generator class */
class RandomGenerator {
public:
    explicit RandomGenerator(unsigned int seed) : rng_(seed) {}

    /* 随机选择一个测试类别，返回该类别的最大维度范围 */
    int get_test_category() {
        std::uniform_int_distribution<int> dist_0_99(0, 99);
        int r = dist_0_99(rng_);

        if (r < DIM_PROB_SMALL) {
            return DIM_RANGE_SMALL;  // 1-128
        }
        if (r < DIM_PROB_SMALL + DIM_PROB_MEDIUM) {
            return DIM_RANGE_MEDIUM;  // 1-512
        }
        return DIM_RANGE_LARGE;  // 1-1024
    }

    /* 根据指定类别生成三个维度 (m, n, k)
     * 确保 max(m,n,k) 的分布符合配置的概率
     */
    void random_three_dims(BLASINT& m, BLASINT& n, BLASINT& k) {
        int category = get_test_category();

        if (category == DIM_RANGE_SMALL) {
            /* Small: 所有维度都在 1-128 */
            std::uniform_int_distribution<BLASINT> dist(1, DIM_RANGE_SMALL);
            m = dist(rng_);
            n = dist(rng_);
            k = dist(rng_);
        } else if (category == DIM_RANGE_MEDIUM) {
            /* Medium: 所有维度都在 1-512，至少一个 > 128 */
            std::uniform_int_distribution<BLASINT> dist_small(1, DIM_RANGE_SMALL);
            std::uniform_int_distribution<BLASINT> dist_medium(DIM_RANGE_SMALL + 1, DIM_RANGE_MEDIUM);
            m = dist_medium(rng_);
            n = dist_medium(rng_);
            k = dist_medium(rng_);
            /* 30% 概率让某个维度变小，增加混合情况 */
            std::uniform_int_distribution<int> dist_0_99(0, 99);
            if (dist_0_99(rng_) < 30) {
                int choice = dist_0_99(rng_) % 3;
                if (choice == 0) m = dist_small(rng_);
                else if (choice == 1) n = dist_small(rng_);
                else k = dist_small(rng_);
            }
        } else {
            /* Large: 所有维度都在 1-1024，至少一个 > 512 */
            std::uniform_int_distribution<BLASINT> dist_medium(1, DIM_RANGE_MEDIUM);
            std::uniform_int_distribution<BLASINT> dist_large(DIM_RANGE_MEDIUM + 1, DIM_RANGE_LARGE);
            m = dist_large(rng_);
            n = dist_large(rng_);
            k = dist_large(rng_);
            /* 30% 概率让某个维度变小，增加混合情况 */
            std::uniform_int_distribution<int> dist_0_99(0, 99);
            if (dist_0_99(rng_) < 30) {
                int choice = dist_0_99(rng_) % 3;
                if (choice == 0) m = dist_medium(rng_);
                else if (choice == 1) n = dist_medium(rng_);
                else k = dist_medium(rng_);
            }
        }
    }

    /* Random float for alpha/beta */
    float random_alpha_beta() {
        std::uniform_int_distribution<int> dist_0_99(0, 99);
        int r = dist_0_99(rng_);

        /* 70% probability: pick from special values */
        if (r < 70) {
            std::uniform_int_distribution<int> dist_0_7(0, special_values_.size() - 1);
            return special_values_[dist_0_7(rng_)];
        }

        /* 30% probability: random from [-10, 10] */
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        return dist(rng_);
    }

    /* Random CBLAS_ORDER */
    enum CBLAS_ORDER random_order() {
        std::uniform_int_distribution<int> dist(0, 1);
        return (dist(rng_) == 0) ? CblasRowMajor : CblasColMajor;
    }

    /* Random CBLAS_TRANSPOSE */
    enum CBLAS_TRANSPOSE random_transpose() {
        std::uniform_int_distribution<int> dist(0, 1);
        return (dist(rng_) == 0) ? CblasNoTrans : CblasTrans;
    }

    /* Weighted random number of threads for multi-threaded BLAS operations
     * 范围: 2-MAX_BLAS_THREADS 线程 (配置于 fuzz_test_config.h)
     * 分布策略: 线性衰减，线程数越大概率越低
     *
     * P(thread = t) ∝ (MAX_THREAD - t + 1)
     * 即:
     *   - 2 threads:           权重 MAX-1 (最高)
     *   - MAX_BLAS_THREADS/2:  权重 MAX/2
     *   - MAX_BLAS_THREADS:    权重 1     (最低)
     */
    int random_blas_threads() {
        constexpr int MIN_THREAD = 2;
        constexpr int MAX_THREAD = MAX_BLAS_THREADS;
        constexpr int TOTAL_WEIGHT = (MAX_THREAD - MIN_THREAD + 1) * (MAX_THREAD - MIN_THREAD + 2) / 2;

        std::uniform_int_distribution<int> dist(1, TOTAL_WEIGHT);
        int r = dist(rng_);

        /* 找到对应的线程数 */
        int remaining = r;
        for (int t = MIN_THREAD; t <= MAX_THREAD; t++) {
            int weight = MAX_THREAD - t + 1;
            if (remaining <= weight) {
                return t;
            }
            remaining -= weight;
        }

        return MAX_THREAD;  /* Fallback */
    }

    /* Generate three dimensions with fixed range (all in [1, range]) */
    void random_three_dims_fixed(BLASINT& m, BLASINT& n, BLASINT& k, int range) {
        std::uniform_int_distribution<BLASINT> dist(1, range);
        m = dist(rng_);
        n = dist(rng_);
        k = dist(rng_);
    }

    /* Generic random integer in range [min, max] */
    template<typename T>
    T random_int(T min, T max) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng_);
    }

    /* Generic random float in range [min, max) */
    float random_float(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }

private:
    std::mt19937 rng_;

public:
    /* Expose engine for external distributions (e.g. uint16_t random) */
    std::mt19937& get_engine() { return rng_; }
private:

    /* Special values for alpha/beta with higher probability */
    static constexpr std::array<float, 8> special_values_ = {
        0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 0.25f, -2.0f
    };
};

/* Matrix comparison utility
 * Compares ref vs test matrices with relative tolerance.
 * Returns true if all elements match within eps, false otherwise.
 * verbose=true: prints up to 20 mismatches with position, values, and relative error.
 */
template <typename T1, typename T2>
bool MatrixCompare(const T1 *ref, const T2 *test, int rows, int cols,
                   int ldc, double eps, bool verbose, bool rowMajor) {
    int mismatch_count = 0;
    const int max_print = 20;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = rowMajor ? (i * ldc + j) : (j * ldc + i);
            double val_ref  = static_cast<double>(ref[idx]);
            double val_test = static_cast<double>(test[idx]);

            double max_val = std::max({std::fabs(val_ref), std::fabs(val_test), 1.0});
            double diff = std::fabs(val_ref - val_test);

            if (diff >= eps * max_val) {
                mismatch_count++;
                if (verbose && mismatch_count <= max_print) {
                    double rel_err = diff / max_val;
                    std::lock_guard<std::mutex> lock(console_mutex);
                    std::printf("  [%d,%d] ref=%.8g, test=%.8g, rel_err=%.8g\n",
                                i, j, val_ref, val_test, rel_err);
                }
            }
        }
    }

    if (verbose && mismatch_count > max_print) {
        std::lock_guard<std::mutex> lock(console_mutex);
        std::printf("  ... %d more mismatches (total %d)\n",
                     mismatch_count - max_print, mismatch_count);
    }

    return mismatch_count == 0;
}

#endif /* FUZZ_TEST_RANDOM_H */
