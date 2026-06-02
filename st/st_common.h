#ifndef ST_COMMON_H
#define ST_COMMON_H

/* ============================================================
 * Merged header: config + buffer + random + report
 * ============================================================ */

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include <cstdlib>
#include <memory>
#include <cassert>
#include <variant>
#include <random>
#include <array>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <cmath>
#include <functional>
#include <optional>
#include <vector>
#include <atomic>

#ifdef __linux__
#include "ref_test_util.h"
#else
/* macOS fallback: AllocateMemory/FreeMemory wrappers using posix_memalign */
#include <cstdlib>
template<typename T>
inline T* AllocateMemory(size_t count, size_t align = 64, bool /*useHBM*/ = false) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, align, count * sizeof(T)) != 0) return nullptr;
    return static_cast<T*>(ptr);
}
template<typename T>
inline void FreeMemory(size_t /*count*/, T *ptr, bool /*useHBM*/ = false) {
    std::free(ptr);
}
#endif

/* ============================================================
 * Config: Dimension ranges, tolerances, thread limits
 * ============================================================ */

constexpr int DIM_RANGE_SMALL = 128;
constexpr int DIM_RANGE_MEDIUM = 512;
constexpr int DIM_RANGE_LARGE = 1024;

constexpr int DIM_PROB_SMALL = 40;
constexpr int DIM_PROB_MEDIUM = 40;
constexpr int DIM_PROB_LARGE = 20;

static_assert(DIM_PROB_SMALL + DIM_PROB_MEDIUM + DIM_PROB_LARGE == 100,
              "DIM_PROB_SMALL + DIM_PROB_MEDIUM + DIM_PROB_LARGE must equal 100");

constexpr int MAX_DIM = DIM_RANGE_LARGE;
constexpr int MAX_LD = DIM_RANGE_LARGE + 7;

constexpr int MAX_FAIL_LOGS = 20;
constexpr int BUFFER_ALIGNMENT = 64;

#define MAX_WORKERS 32

#define MAX_BLAS_THREADS 50

constexpr float SGEMM_TOLERANCE  = 1e-3f;
constexpr float SHGEMM_TOLERANCE = 1e-3f;
constexpr float SBGEMM_TOLERANCE = 1e-3f;
constexpr float HGEMM_TOLERANCE  = 5e-2f;
constexpr float BGEMM_TOLERANCE  = 5e-2f;

/* ============================================================
 * Buffer: BufferPrecision enum, ThreadBuffers struct
 * ============================================================ */

enum class BufferPrecision {
    SGEMM, SHGEMM, SBGEMM, HGEMM, BGEMM
};

enum class PrecisionType {
    SGEMM,
    SHGEMM,
    SBGEMM,
    HGEMM,
    BGEMM
};

/* Runtime HBM toggle (set by --no-hbm flag, defaults to true on Linux with HBM) */
extern bool use_hbm;

struct ThreadBuffers {
    float *a_buf;
    float *b_buf;
    float *c_impl_buf;
    float *c_ref_buf;
    float16_t *a_half;
    float16_t *b_half;
    float16_t *c_half;
    bfloat16_t *a_bf16;
    bfloat16_t *b_bf16;
    bfloat16_t *c_bf16;
    size_t max_size;        // 旧接口用（统一大小）
    size_t float_alloc_size; // allocate_for_sizes: float buffer 分配大小
    size_t half_alloc_size;  // allocate_for_sizes: half buffer 分配大小
    size_t bf16_alloc_size;  // allocate_for_sizes: bf16 buffer 分配大小
    bool has_half_ab;
    bool has_half_c;
    bool has_bf16_ab;
    bool has_bf16_c;

    ThreadBuffers() : a_buf(nullptr), b_buf(nullptr),
                      c_impl_buf(nullptr), c_ref_buf(nullptr),
                      a_half(nullptr), b_half(nullptr), c_half(nullptr),
                      a_bf16(nullptr), b_bf16(nullptr), c_bf16(nullptr),
                      max_size(0), float_alloc_size(0), half_alloc_size(0), bf16_alloc_size(0),
                      has_half_ab(false), has_half_c(false),
                      has_bf16_ab(false), has_bf16_c(false) {}

    bool allocate_for_precision(BLASINT max_dim, BLASINT max_ld, BufferPrecision precision) {
        max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

        if (use_hbm) {
            a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            b_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            c_impl_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            c_ref_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);

            if (precision == BufferPrecision::SHGEMM || precision == BufferPrecision::HGEMM) {
                a_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
                b_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
                has_half_ab = true;
            }
            if (precision == BufferPrecision::HGEMM) {
                c_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
                has_half_c = true;
            }
            if (precision == BufferPrecision::SBGEMM || precision == BufferPrecision::BGEMM) {
                a_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
                b_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
                has_bf16_ab = true;
            }
            if (precision == BufferPrecision::BGEMM) {
                c_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
                has_bf16_c = true;
            }
        } else {
            size_t float_bytes = max_size * sizeof(float);
            size_t half_bytes = max_size * sizeof(float16_t);
            size_t bf16_bytes = max_size * sizeof(bfloat16_t);

            if (posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, float_bytes) != 0) a_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&b_buf), BUFFER_ALIGNMENT, float_bytes) != 0) b_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_impl_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_impl_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_ref_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_ref_buf = nullptr;

            if (precision == BufferPrecision::SHGEMM || precision == BufferPrecision::HGEMM) {
                if (posix_memalign(reinterpret_cast<void**>(&a_half), BUFFER_ALIGNMENT, half_bytes) != 0) a_half = nullptr;
                if (posix_memalign(reinterpret_cast<void**>(&b_half), BUFFER_ALIGNMENT, half_bytes) != 0) b_half = nullptr;
                has_half_ab = true;
            }
            if (precision == BufferPrecision::HGEMM) {
                if (posix_memalign(reinterpret_cast<void**>(&c_half), BUFFER_ALIGNMENT, half_bytes) != 0) c_half = nullptr;
                has_half_c = true;
            }
            if (precision == BufferPrecision::SBGEMM || precision == BufferPrecision::BGEMM) {
                if (posix_memalign(reinterpret_cast<void**>(&a_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) a_bf16 = nullptr;
                if (posix_memalign(reinterpret_cast<void**>(&b_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) b_bf16 = nullptr;
                has_bf16_ab = true;
            }
            if (precision == BufferPrecision::BGEMM) {
                if (posix_memalign(reinterpret_cast<void**>(&c_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) c_bf16 = nullptr;
                has_bf16_c = true;
            }
        }

        if (!a_buf || !b_buf || !c_impl_buf || !c_ref_buf) return false;
        if (has_half_ab && (!a_half || !b_half)) return false;
        if (has_half_c && !c_half) return false;
        if (has_bf16_ab && (!a_bf16 || !b_bf16)) return false;
        if (has_bf16_c && !c_bf16) return false;

        assert((reinterpret_cast<uintptr_t>(a_buf) % BUFFER_ALIGNMENT == 0));
        assert((reinterpret_cast<uintptr_t>(b_buf) % BUFFER_ALIGNMENT == 0));
        assert((reinterpret_cast<uintptr_t>(c_impl_buf) % BUFFER_ALIGNMENT == 0));
        assert((reinterpret_cast<uintptr_t>(c_ref_buf) % BUFFER_ALIGNMENT == 0));

        return true;
    }

    bool allocate(BLASINT max_dim, BLASINT max_ld) {
        bool result = allocate_for_precision(max_dim, max_ld, BufferPrecision::BGEMM);
        if (result) {
            has_half_ab = true;
            has_half_c = true;
            has_bf16_ab = true;
            has_bf16_c = true;
        }
        return result;
    }

    bool allocate_all(BLASINT max_dim, BLASINT max_ld) {
        max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

        if (use_hbm) {
            a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            b_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            c_impl_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            c_ref_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
            a_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
            b_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
            c_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
            a_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
            b_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
            c_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
        } else {
            size_t float_bytes = max_size * sizeof(float);
            size_t half_bytes = max_size * sizeof(float16_t);
            size_t bf16_bytes = max_size * sizeof(bfloat16_t);

            if (posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, float_bytes) != 0) a_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&b_buf), BUFFER_ALIGNMENT, float_bytes) != 0) b_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_impl_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_impl_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_ref_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_ref_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&a_half), BUFFER_ALIGNMENT, half_bytes) != 0) a_half = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&b_half), BUFFER_ALIGNMENT, half_bytes) != 0) b_half = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_half), BUFFER_ALIGNMENT, half_bytes) != 0) c_half = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&a_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) a_bf16 = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&b_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) b_bf16 = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) c_bf16 = nullptr;
        }

        has_half_ab = true;
        has_half_c = true;
        has_bf16_ab = true;
        has_bf16_c = true;

        if (!a_buf || !b_buf || !c_impl_buf || !c_ref_buf) return false;
        if (!a_half || !b_half || !c_half) return false;
        if (!a_bf16 || !b_bf16 || !c_bf16) return false;

        return true;
    }

    // 按精确大小分配 buffer，用于 per-iteration 分配
    bool allocate_for_sizes(BLASINT a_size, BLASINT b_size, BLASINT c_size, PrecisionType prec) {
        float_alloc_size = static_cast<size_t>(a_size);
        size_t b_sz = static_cast<size_t>(b_size);
        size_t c_sz = static_cast<size_t>(c_size);

        if (use_hbm) {
            // SGEMM: 4 个 float buffer
            // SHGEMM/SBGEMM/HGEMM/BGEMM: 2 个 float buffer (c_impl, c_ref)
            a_buf = AllocateMemory<float>(float_alloc_size, BUFFER_ALIGNMENT, true);
            b_buf = AllocateMemory<float>(b_sz, BUFFER_ALIGNMENT, true);
            c_impl_buf = AllocateMemory<float>(c_sz, BUFFER_ALIGNMENT, true);
            c_ref_buf = AllocateMemory<float>(c_sz, BUFFER_ALIGNMENT, true);

            if (prec == PrecisionType::SHGEMM || prec == PrecisionType::HGEMM) {
                half_alloc_size = float_alloc_size > b_sz ? float_alloc_size : b_sz;
                a_half = AllocateMemory<float16_t>(float_alloc_size, BUFFER_ALIGNMENT, true);
                b_half = AllocateMemory<float16_t>(b_sz, BUFFER_ALIGNMENT, true);
                has_half_ab = true;
            }
            if (prec == PrecisionType::HGEMM) {
                half_alloc_size = c_sz > half_alloc_size ? c_sz : half_alloc_size;
                c_half = AllocateMemory<float16_t>(c_sz, BUFFER_ALIGNMENT, true);
                has_half_c = true;
            }
            if (prec == PrecisionType::SBGEMM || prec == PrecisionType::BGEMM) {
                bf16_alloc_size = float_alloc_size > b_sz ? float_alloc_size : b_sz;
                a_bf16 = AllocateMemory<bfloat16_t>(float_alloc_size, BUFFER_ALIGNMENT, true);
                b_bf16 = AllocateMemory<bfloat16_t>(b_sz, BUFFER_ALIGNMENT, true);
                has_bf16_ab = true;
            }
            if (prec == PrecisionType::BGEMM) {
                bf16_alloc_size = c_sz > bf16_alloc_size ? c_sz : bf16_alloc_size;
                c_bf16 = AllocateMemory<bfloat16_t>(c_sz, BUFFER_ALIGNMENT, true);
                has_bf16_c = true;
            }
        } else {
            size_t float_bytes_a = float_alloc_size * sizeof(float);
            size_t float_bytes_b = b_sz * sizeof(float);
            size_t float_bytes_c = c_sz * sizeof(float);

            if (posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, float_bytes_a) != 0) a_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&b_buf), BUFFER_ALIGNMENT, float_bytes_b) != 0) b_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_impl_buf), BUFFER_ALIGNMENT, float_bytes_c) != 0) c_impl_buf = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&c_ref_buf), BUFFER_ALIGNMENT, float_bytes_c) != 0) c_ref_buf = nullptr;

            if (prec == PrecisionType::SHGEMM || prec == PrecisionType::HGEMM) {
                half_alloc_size = float_alloc_size > b_sz ? float_alloc_size : b_sz;
                if (posix_memalign(reinterpret_cast<void**>(&a_half), BUFFER_ALIGNMENT, float_bytes_a) != 0) a_half = nullptr;
                if (posix_memalign(reinterpret_cast<void**>(&b_half), BUFFER_ALIGNMENT, float_bytes_b) != 0) b_half = nullptr;
                has_half_ab = true;
            }
            if (prec == PrecisionType::HGEMM) {
                half_alloc_size = c_sz > half_alloc_size ? c_sz : half_alloc_size;
                if (posix_memalign(reinterpret_cast<void**>(&c_half), BUFFER_ALIGNMENT, float_bytes_c) != 0) c_half = nullptr;
                has_half_c = true;
            }
            if (prec == PrecisionType::SBGEMM || prec == PrecisionType::BGEMM) {
                bf16_alloc_size = float_alloc_size > b_sz ? float_alloc_size : b_sz;
                size_t bf16_bytes_a = float_alloc_size * sizeof(bfloat16_t);
                size_t bf16_bytes_b = b_sz * sizeof(bfloat16_t);
                if (posix_memalign(reinterpret_cast<void**>(&a_bf16), BUFFER_ALIGNMENT, bf16_bytes_a) != 0) a_bf16 = nullptr;
                if (posix_memalign(reinterpret_cast<void**>(&b_bf16), BUFFER_ALIGNMENT, bf16_bytes_b) != 0) b_bf16 = nullptr;
                has_bf16_ab = true;
            }
            if (prec == PrecisionType::BGEMM) {
                bf16_alloc_size = c_sz > bf16_alloc_size ? c_sz : bf16_alloc_size;
                size_t bf16_bytes_c = c_sz * sizeof(bfloat16_t);
                if (posix_memalign(reinterpret_cast<void**>(&c_bf16), BUFFER_ALIGNMENT, bf16_bytes_c) != 0) c_bf16 = nullptr;
                has_bf16_c = true;
            }
        }

        if (!a_buf || !b_buf || !c_impl_buf || !c_ref_buf) return false;
        if (has_half_ab && (!a_half || !b_half)) return false;
        if (has_half_c && !c_half) return false;
        if (has_bf16_ab && (!a_bf16 || !b_bf16)) return false;
        if (has_bf16_c && !c_bf16) return false;

        return true;
    }

    ~ThreadBuffers() {
        // 两种分配模式：max_size > 0（旧接口，统一大小）或 float_alloc_size > 0（新接口，精确大小）
        bool is_sized = (float_alloc_size > 0);
        if (max_size == 0 && !is_sized) return;

        if (is_sized) {
            FreeMemory<float>(float_alloc_size, a_buf, use_hbm);
            FreeMemory<float>(float_alloc_size, b_buf, use_hbm);
            FreeMemory<float>(float_alloc_size, c_impl_buf, use_hbm);
            FreeMemory<float>(float_alloc_size, c_ref_buf, use_hbm);
            if (has_half_ab) { FreeMemory<float16_t>(half_alloc_size, a_half, use_hbm); FreeMemory<float16_t>(half_alloc_size, b_half, use_hbm); }
            if (has_half_c) FreeMemory<float16_t>(half_alloc_size, c_half, use_hbm);
            if (has_bf16_ab) { FreeMemory<bfloat16_t>(bf16_alloc_size, a_bf16, use_hbm); FreeMemory<bfloat16_t>(bf16_alloc_size, b_bf16, use_hbm); }
            if (has_bf16_c) FreeMemory<bfloat16_t>(bf16_alloc_size, c_bf16, use_hbm);
        } else {
            FreeMemory<float>(max_size, a_buf, use_hbm);
            FreeMemory<float>(max_size, b_buf, use_hbm);
            FreeMemory<float>(max_size, c_impl_buf, use_hbm);
            FreeMemory<float>(max_size, c_ref_buf, use_hbm);
            if (has_half_ab) { FreeMemory<float16_t>(max_size, a_half, use_hbm); FreeMemory<float16_t>(max_size, b_half, use_hbm); }
            if (has_half_c) FreeMemory<float16_t>(max_size, c_half, use_hbm);
            if (has_bf16_ab) { FreeMemory<bfloat16_t>(max_size, a_bf16, use_hbm); FreeMemory<bfloat16_t>(max_size, b_bf16, use_hbm); }
            if (has_bf16_c) FreeMemory<bfloat16_t>(max_size, c_bf16, use_hbm);
        }
    }

    ThreadBuffers(const ThreadBuffers&) = delete;
    ThreadBuffers& operator=(const ThreadBuffers&) = delete;

    float *a_ptr() { return a_buf; }
    float *b_ptr() { return b_buf; }
    float *c_impl_ptr() { return c_impl_buf; }
    float *c_ref_ptr() { return c_ref_buf; }
    float16_t *a_half_ptr() { return a_half; }
    float16_t *b_half_ptr() { return b_half; }
    float16_t *c_half_ptr() { return c_half; }
    bfloat16_t *a_bf16_ptr() { return a_bf16; }
    bfloat16_t *b_bf16_ptr() { return b_bf16; }
    bfloat16_t *c_bf16_ptr() { return c_bf16; }
};

/* ============================================================
 * Random: RandomGenerator class
 * ============================================================ */

extern std::mutex console_mutex;

class RandomGenerator {
public:
    explicit RandomGenerator(unsigned int seed) : rng_(seed) {}

    int get_test_category() {
        std::uniform_int_distribution<int> dist_0_99(0, 99);
        int r = dist_0_99(rng_);
        if (r < DIM_PROB_SMALL) return DIM_RANGE_SMALL;
        if (r < DIM_PROB_SMALL + DIM_PROB_MEDIUM) return DIM_RANGE_MEDIUM;
        return DIM_RANGE_LARGE;
    }

    void random_three_dims(BLASINT& m, BLASINT& n, BLASINT& k) {
        int category = get_test_category();
        if (category == DIM_RANGE_SMALL) {
            std::uniform_int_distribution<BLASINT> dist(1, DIM_RANGE_SMALL);
            m = dist(rng_); n = dist(rng_); k = dist(rng_);
        } else if (category == DIM_RANGE_MEDIUM) {
            std::uniform_int_distribution<BLASINT> dist_small(1, DIM_RANGE_SMALL);
            std::uniform_int_distribution<BLASINT> dist_medium(DIM_RANGE_SMALL + 1, DIM_RANGE_MEDIUM);
            m = dist_medium(rng_); n = dist_medium(rng_); k = dist_medium(rng_);
            std::uniform_int_distribution<int> dist_0_99(0, 99);
            if (dist_0_99(rng_) < 30) {
                int choice = dist_0_99(rng_) % 3;
                if (choice == 0) m = dist_small(rng_);
                else if (choice == 1) n = dist_small(rng_);
                else k = dist_small(rng_);
            }
        } else {
            std::uniform_int_distribution<BLASINT> dist_medium(1, DIM_RANGE_MEDIUM);
            std::uniform_int_distribution<BLASINT> dist_large(DIM_RANGE_MEDIUM + 1, DIM_RANGE_LARGE);
            m = dist_large(rng_); n = dist_large(rng_); k = dist_large(rng_);
            std::uniform_int_distribution<int> dist_0_99(0, 99);
            if (dist_0_99(rng_) < 30) {
                int choice = dist_0_99(rng_) % 3;
                if (choice == 0) m = dist_medium(rng_);
                else if (choice == 1) n = dist_medium(rng_);
                else k = dist_medium(rng_);
            }
        }
    }

    float random_alpha_beta() {
        std::uniform_int_distribution<int> dist_0_99(0, 99);
        int r = dist_0_99(rng_);
        if (r < 70) {
            std::uniform_int_distribution<int> dist_0_7(0, special_values_.size() - 1);
            return special_values_[dist_0_7(rng_)];
        }
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        return dist(rng_);
    }

    enum CBLAS_ORDER random_order() {
        std::uniform_int_distribution<int> dist(0, 1);
        return (dist(rng_) == 0) ? CblasRowMajor : CblasColMajor;
    }

    enum CBLAS_TRANSPOSE random_transpose() {
        std::uniform_int_distribution<int> dist(0, 1);
        return (dist(rng_) == 0) ? CblasNoTrans : CblasTrans;
    }

    int random_blas_threads() {
        constexpr int MIN_THREAD = 2;
        constexpr int MAX_THREAD = MAX_BLAS_THREADS;
        constexpr int TOTAL_WEIGHT = (MAX_THREAD - MIN_THREAD + 1) * (MAX_THREAD - MIN_THREAD + 2) / 2;
        std::uniform_int_distribution<int> dist(1, TOTAL_WEIGHT);
        int r = dist(rng_);
        int remaining = r;
        for (int t = MIN_THREAD; t <= MAX_THREAD; t++) {
            int weight = MAX_THREAD - t + 1;
            if (remaining <= weight) return t;
            remaining -= weight;
        }
        return MAX_THREAD;
    }

    void random_three_dims_fixed(BLASINT& m, BLASINT& n, BLASINT& k, int range) {
        std::uniform_int_distribution<BLASINT> dist(1, range);
        m = dist(rng_); n = dist(rng_); k = dist(rng_);
    }

    template<typename T>
    T random_int(T min, T max) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng_);
    }

    float random_float(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }

    std::mt19937& get_engine() { return rng_; }

private:
    std::mt19937 rng_;
    static constexpr std::array<float, 8> special_values_ = {
        0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 0.25f, -2.0f
    };
};

/* ============================================================
 * Report: Name helpers
 * ============================================================ */

inline const char *trans_name(enum CBLAS_TRANSPOSE trans) {
    switch (trans) {
        case CblasNoTrans: return "N";
        case CblasTrans: return "T";
        default: return "?";
    }
}

inline const char *order_name(enum CBLAS_ORDER order) {
    switch (order) {
        case CblasRowMajor: return "R";
        case CblasColMajor: return "C";
        default: return "?";
    }
}

inline const char *precision_name(PrecisionType p) {
    switch (p) {
        case PrecisionType::SGEMM:  return "SGEMM";
        case PrecisionType::SHGEMM: return "SHGEMM";
        case PrecisionType::SBGEMM: return "SBGEMM";
        case PrecisionType::HGEMM:  return "HGEMM";
        case PrecisionType::BGEMM:  return "BGEMM";
    }
    return "?";
}

/* ============================================================
 * MatrixCompare: Reference vs test comparison
 * ============================================================ */

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

/* ============================================================
 * Crash handler: print shape on SIGSEGV/SIGBUS
 * ============================================================ */

#include <csignal>
#include <unistd.h>

struct CrashContext {
    int thread_id;
    int stage_num;
    PrecisionType precision;
    BLASINT M, N, K;
    CBLAS_TRANSPOSE transA, transB;
    CBLAS_ORDER order;
    float alpha, beta;
    BLASINT lda, ldb, ldc;
};

static thread_local CrashContext* g_crash_ctx = nullptr;

static void crash_signal_handler(int sig) {
    CrashContext* ctx = g_crash_ctx;
    if (ctx) {
        char buf[512];
        int len = snprintf(buf, sizeof(buf),
            "\n\n=== CRASH: signal %d ===\n"
            "  Thread: %d | Stage: %d\n"
            "  Precision: %s\n"
            "  M=%d N=%d K=%d\n"
            "  transA=%s transB=%s order=%s\n"
            "  alpha=%g beta=%g\n"
            "  lda=%d ldb=%d ldc=%d\n"
            "=== Re-raising signal for core dump ===\n\n",
            sig,
            ctx->thread_id, ctx->stage_num,
            precision_name(ctx->precision),
            (int)ctx->M, (int)ctx->N, (int)ctx->K,
            trans_name(ctx->transA), trans_name(ctx->transB), order_name(ctx->order),
            ctx->alpha, ctx->beta,
            (int)ctx->lda, (int)ctx->ldb, (int)ctx->ldc);
        write(STDERR_FILENO, buf, len);
    }
    // Re-raise with default handler to get core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

inline void install_crash_handler() {
    struct sigaction sa;
    sa.sa_handler = crash_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
}

#endif /* ST_COMMON_H */
