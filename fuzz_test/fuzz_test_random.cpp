#ifndef FUZZ_TEST_RANDOM_H
#define FUZZ_TEST_RANDOM_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include <random>
#include <array>
#include <algorithm>

/* Random parameter generator class */
class RandomGenerator {
public:
    explicit RandomGenerator(unsigned int seed) : rng_(seed) {}

    /* Weighted random for dimensions
     *
     * 分布策略:
     *   - DIM_PROB_SMALL (10%): 从特殊小维度列表中选择 (1-64)
     *   - DIM_PROB_MEDIUM (40%): 随机选择 1-512
     *   - DIM_PROB_LARGE (50%): 随机选择 1-1024
     */
    BLASINT random_dim() {
        std::uniform_int_distribution<int> dist_0_99(0, 99);
        int r = dist_0_99(rng_);

        if (r < DIM_PROB_SMALL) {
            std::uniform_int_distribution<int> dist_0_10(0, special_small_dims_.size() - 1);
            return special_small_dims_[dist_0_10(rng_)];
        }

        if (r < DIM_PROB_SMALL + DIM_PROB_MEDIUM) {
            std::uniform_int_distribution<BLASINT> dist(1, DIM_RANGE_MEDIUM);
            return dist(rng_);
        }

        std::uniform_int_distribution<BLASINT> dist(1, DIM_RANGE_LARGE);
        return dist(rng_);
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

    /* Random number of threads for BLAS operations
     * 分布策略:
     *   - 40%: 1 (single-threaded)
     *   - 30%: 2 (dual-threaded)
     *   - 15%: 4 (quad-threaded)
     *   - 10%: 8 (common for modern CPUs)
     *   - 5%:  1 (default/single-threaded)
     */
    int random_num_threads() {
        std::uniform_int_distribution<int> dist_0_99(0, 99);
        int r = dist_0_99(rng_);

        if (r < 40) return 1;       // 40%
        if (r < 70) return 2;       // 30%
        if (r < 85) return 4;       // 15%
        if (r < 95) return 8;       // 10%
        return 1;                   // 5% (default)
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

    /* Special values for alpha/beta with higher probability */
    static constexpr std::array<float, 8> special_values_ = {
        0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 0.25f, -2.0f
    };

    /* Special small dimensions for edge case testing (1-64) */
    static constexpr std::array<BLASINT, 11> special_small_dims_ = {
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63
    };
};

#endif /* FUZZ_TEST_RANDOM_H */
