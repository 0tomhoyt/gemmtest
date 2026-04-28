#ifndef FUZZ_TEST_BUFFER_H
#define FUZZ_TEST_BUFFER_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include <cstdlib>
#include <memory>
#include <cassert>
#include <variant>

#ifdef USE_HBM
#include "test_util.h"
#endif

/* Precision allocation flags (subset of PrecisionType) */
/* Used for buffer allocation decision to avoid circular dependency with fuzz_test_worker.h */
enum class BufferPrecision {
    SGEMM,   /* Only float buffers needed */
    SHGEMM,  /* float + FP16 A,B */
    SBGEMM,  /* float + BF16 A,B */
    HGEMM,   /* float + FP16 A,B,C */
    BGEMM    /* float + BF16 A,B,C */
};

/* 原有的 ThreadBuffers 保持不变，用于向后兼容 */
/* Buffer for matrix data (allocated per thread with 64-byte alignment) */
struct ThreadBuffers {
    float *a_buf;
    float *b_buf;
    float *c_impl_buf;
    float *c_ref_buf;
    float16_t *a_half;   // SHGEMM/HGEMM: impl 使用，与 a_buf 同步
    float16_t *b_half;
    float16_t *c_half;   // HGEMM: impl C 输出
    bfloat16_t *a_bf16;  // SBGEMM/BGEMM: impl 使用，与 a_buf 同步
    bfloat16_t *b_bf16;
    bfloat16_t *c_bf16;  // BGEMM: impl C 输出
    size_t max_size;
    bool has_half_ab;    // 是否分配了 a_half, b_half
    bool has_half_c;     // 是否分配了 c_half
    bool has_bf16_ab;    // 是否分配了 a_bf16, b_bf16
    bool has_bf16_c;     // 是否分配了 c_bf16

    ThreadBuffers() : a_buf(nullptr), b_buf(nullptr),
                      c_impl_buf(nullptr), c_ref_buf(nullptr),
                      a_half(nullptr), b_half(nullptr), c_half(nullptr),
                      a_bf16(nullptr), b_bf16(nullptr), c_bf16(nullptr),
                      max_size(0),
                      has_half_ab(false), has_half_c(false),
                      has_bf16_ab(false), has_bf16_c(false) {}

    /* Allocate buffers for a specific precision type */
    bool allocate_for_precision(BLASINT max_dim, BLASINT max_ld, BufferPrecision precision) {
        max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

#ifdef USE_HBM
        /* Float buffers are always needed for reference implementation */
        a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        b_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        c_impl_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        c_ref_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);

        /* Allocate half-precision buffers based on precision type */
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
#else
        /* Use posix_memalign for cross-platform compatibility */
        size_t float_bytes = max_size * sizeof(float);
        size_t half_bytes = max_size * sizeof(float16_t);
        size_t bf16_bytes = max_size * sizeof(bfloat16_t);

        /* Float buffers are always needed for reference implementation */
        if (posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, float_bytes) != 0) a_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&b_buf), BUFFER_ALIGNMENT, float_bytes) != 0) b_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&c_impl_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_impl_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&c_ref_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_ref_buf = nullptr;

        /* Allocate half-precision buffers based on precision type */
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
#endif

        /* Verify required buffers are allocated */
        if (!a_buf || !b_buf || !c_impl_buf || !c_ref_buf) {
            return false;
        }
        if (has_half_ab && (!a_half || !b_half)) {
            return false;
        }
        if (has_half_c && !c_half) {
            return false;
        }
        if (has_bf16_ab && (!a_bf16 || !b_bf16)) {
            return false;
        }
        if (has_bf16_c && !c_bf16) {
            return false;
        }

        /* Verify 64-byte alignment */
        assert((reinterpret_cast<uintptr_t>(a_buf) % BUFFER_ALIGNMENT == 0) && "a_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(b_buf) % BUFFER_ALIGNMENT == 0) && "b_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(c_impl_buf) % BUFFER_ALIGNMENT == 0) && "c_impl_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(c_ref_buf) % BUFFER_ALIGNMENT == 0) && "c_ref_buf not 64-byte aligned");
        if (has_half_ab) {
            assert((reinterpret_cast<uintptr_t>(a_half) % BUFFER_ALIGNMENT == 0) && "a_half not 64-byte aligned");
            assert((reinterpret_cast<uintptr_t>(b_half) % BUFFER_ALIGNMENT == 0) && "b_half not 64-byte aligned");
        }
        if (has_half_c) {
            assert((reinterpret_cast<uintptr_t>(c_half) % BUFFER_ALIGNMENT == 0) && "c_half not 64-byte aligned");
        }
        if (has_bf16_ab) {
            assert((reinterpret_cast<uintptr_t>(a_bf16) % BUFFER_ALIGNMENT == 0) && "a_bf16 not 64-byte aligned");
            assert((reinterpret_cast<uintptr_t>(b_bf16) % BUFFER_ALIGNMENT == 0) && "b_bf16 not 64-byte aligned");
        }
        if (has_bf16_c) {
            assert((reinterpret_cast<uintptr_t>(c_bf16) % BUFFER_ALIGNMENT == 0) && "c_bf16 not 64-byte aligned");
        }

        return true;
    }

    /* Legacy method for backward compatibility - allocates all buffers */
    bool allocate(BLASINT max_dim, BLASINT max_ld) {
        /* Temporarily use SGEMM to trigger all allocations for backward compatibility */
        bool result = allocate_for_precision(max_dim, max_ld, BufferPrecision::BGEMM);
        if (result) {
            has_half_ab = true;
            has_half_c = true;
            has_bf16_ab = true;
            has_bf16_c = true;
        }
        return result;
    }

    ~ThreadBuffers() {
        if (max_size == 0) return;

#ifdef USE_HBM
        /* Use test_util.h FreeMemory for HBM */
        FreeMemory<float>(max_size, a_buf, true);
        FreeMemory<float>(max_size, b_buf, true);
        FreeMemory<float>(max_size, c_impl_buf, true);
        FreeMemory<float>(max_size, c_ref_buf, true);
        if (has_half_ab) {
            FreeMemory<float16_t>(max_size, a_half, true);
            FreeMemory<float16_t>(max_size, b_half, true);
        }
        if (has_half_c) {
            FreeMemory<float16_t>(max_size, c_half, true);
        }
        if (has_bf16_ab) {
            FreeMemory<bfloat16_t>(max_size, a_bf16, true);
            FreeMemory<bfloat16_t>(max_size, b_bf16, true);
        }
        if (has_bf16_c) {
            FreeMemory<bfloat16_t>(max_size, c_bf16, true);
        }
#else
        std::free(a_buf);
        std::free(b_buf);
        std::free(c_impl_buf);
        std::free(c_ref_buf);
        if (has_half_ab) {
            std::free(a_half);
            std::free(b_half);
        }
        if (has_half_c) {
            std::free(c_half);
        }
        if (has_bf16_ab) {
            std::free(a_bf16);
            std::free(b_bf16);
        }
        if (has_bf16_c) {
            std::free(c_bf16);
        }
#endif
    }

    /* Disable copy */
    ThreadBuffers(const ThreadBuffers&) = delete;
    ThreadBuffers& operator=(const ThreadBuffers&) = delete;

    /* Get raw pointers (for C interface compatibility) */
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

    const float *a_ptr() const { return a_buf; }
    const float *b_ptr() const { return b_buf; }
    const float *c_impl_ptr() const { return c_impl_buf; }
    const float *c_ref_ptr() const { return c_ref_buf; }
    const float16_t *a_half_ptr() const { return a_half; }
    const float16_t *b_half_ptr() const { return b_half; }
    const float16_t *c_half_ptr() const { return c_half; }
    const bfloat16_t *a_bf16_ptr() const { return a_bf16; }
    const bfloat16_t *b_bf16_ptr() const { return b_bf16; }
    const bfloat16_t *c_bf16_ptr() const { return c_bf16; }
};

/* Allocate thread buffers - returns unique_ptr for automatic cleanup */
inline std::unique_ptr<ThreadBuffers> alloc_thread_buffers(BLASINT max_dim, BLASINT max_ld) {
    auto bufs = std::make_unique<ThreadBuffers>();
    /* Legacy method: allocate all buffers for backward compatibility */
    if (!bufs->allocate(max_dim, max_ld)) {
        return nullptr;
    }
    return bufs;
}

/* Allocate thread buffers for a specific precision - returns unique_ptr */
inline std::unique_ptr<ThreadBuffers> alloc_thread_buffers(BLASINT max_dim, BLASINT max_ld, BufferPrecision precision) {
    auto bufs = std::make_unique<ThreadBuffers>();
    if (!bufs->allocate_for_precision(max_dim, max_ld, precision)) {
        return nullptr;
    }
    return bufs;
}

/* No need for explicit free - unique_ptr handles it automatically */

#endif /* FUZZ_TEST_BUFFER_H */
