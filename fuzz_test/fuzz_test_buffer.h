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

/* 原有的 ThreadBuffers 保持不变，用于向后兼容 */
/* Buffer for matrix data (allocated per thread with 64-byte alignment) */
struct ThreadBuffers {
    float* a_buf;
    float* b_buf;
    float* c_impl_buf;
    float* c_ref_buf;
    float16_t* a_half;   // SHGEMM: impl 使用，与 a_buf 同步
    float16_t* b_half;
    bfloat16_t* a_bf16;  // SBGEMM: impl 使用，与 a_buf 同步
    bfloat16_t* b_bf16;
    size_t max_size;

    ThreadBuffers() : a_buf(nullptr), b_buf(nullptr),
                      c_impl_buf(nullptr), c_ref_buf(nullptr),
                      a_half(nullptr), b_half(nullptr),
                      a_bf16(nullptr), b_bf16(nullptr), max_size(0) {}

    /* Allocate buffers with specified size and 64-byte alignment */
    bool allocate(BLASINT max_dim, BLASINT max_ld) {
        max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

#ifdef USE_HBM
        /* Use test_util.h AllocateMemory for HBM with 64-byte alignment */
        a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        b_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        c_impl_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        c_ref_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        a_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
        b_half = AllocateMemory<float16_t>(max_size, BUFFER_ALIGNMENT, true);
        a_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
        b_bf16 = AllocateMemory<bfloat16_t>(max_size, BUFFER_ALIGNMENT, true);
#else
        /* Use posix_memalign for cross-platform compatibility */
        size_t float_bytes = max_size * sizeof(float);
        size_t half_bytes = max_size * sizeof(float16_t);
        size_t bf16_bytes = max_size * sizeof(bfloat16_t);
        if (posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, float_bytes) != 0) a_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&b_buf), BUFFER_ALIGNMENT, float_bytes) != 0) b_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&c_impl_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_impl_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&c_ref_buf), BUFFER_ALIGNMENT, float_bytes) != 0) c_ref_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&a_half), BUFFER_ALIGNMENT, half_bytes) != 0) a_half = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&b_half), BUFFER_ALIGNMENT, half_bytes) != 0) b_half = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&a_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) a_bf16 = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&b_bf16), BUFFER_ALIGNMENT, bf16_bytes) != 0) b_bf16 = nullptr;
#endif

        if (!a_buf || !b_buf || !c_impl_buf || !c_ref_buf ||
            !a_half || !b_half || !a_bf16 || !b_bf16) {
            return false;
        }

        /* Verify 64-byte alignment */
        assert((reinterpret_cast<uintptr_t>(a_buf) % BUFFER_ALIGNMENT == 0) && "a_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(b_buf) % BUFFER_ALIGNMENT == 0) && "b_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(c_impl_buf) % BUFFER_ALIGNMENT == 0) && "c_impl_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(c_ref_buf) % BUFFER_ALIGNMENT == 0) && "c_ref_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(a_half) % BUFFER_ALIGNMENT == 0) && "a_half not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(b_half) % BUFFER_ALIGNMENT == 0) && "b_half not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(a_bf16) % BUFFER_ALIGNMENT == 0) && "a_bf16 not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(b_bf16) % BUFFER_ALIGNMENT == 0) && "b_bf16 not 64-byte aligned");

        return true;
    }

    ~ThreadBuffers() {
        if (max_size == 0) return;

#ifdef USE_HBM
        /* Use test_util.h FreeMemory for HBM */
        FreeMemory<float>(max_size, a_buf, true);
        FreeMemory<float>(max_size, b_buf, true);
        FreeMemory<float>(max_size, c_impl_buf, true);
        FreeMemory<float>(max_size, c_ref_buf, true);
        FreeMemory<float16_t>(max_size, a_half, true);
        FreeMemory<float16_t>(max_size, b_half, true);
        FreeMemory<bfloat16_t>(max_size, a_bf16, true);
        FreeMemory<bfloat16_t>(max_size, b_bf16, true);
#else
        std::free(a_buf);
        std::free(b_buf);
        std::free(c_impl_buf);
        std::free(c_ref_buf);
        std::free(a_half);
        std::free(b_half);
        std::free(a_bf16);
        std::free(b_bf16);
#endif
    }

    /* Disable copy */
    ThreadBuffers(const ThreadBuffers&) = delete;
    ThreadBuffers& operator=(const ThreadBuffers&) = delete;

    /* Get raw pointers (for C interface compatibility) */
    float* a_ptr() { return a_buf; }
    float* b_ptr() { return b_buf; }
    float* c_impl_ptr() { return c_impl_buf; }
    float* c_ref_ptr() { return c_ref_buf; }
    float16_t* a_half_ptr() { return a_half; }
    float16_t* b_half_ptr() { return b_half; }
    bfloat16_t* a_bf16_ptr() { return a_bf16; }
    bfloat16_t* b_bf16_ptr() { return b_bf16; }

    const float* a_ptr() const { return a_buf; }
    const float* b_ptr() const { return b_buf; }
    const float* c_impl_ptr() const { return c_impl_buf; }
    const float* c_ref_ptr() const { return c_ref_buf; }
    const float16_t* a_half_ptr() const { return a_half; }
    const float16_t* b_half_ptr() const { return b_half; }
    const bfloat16_t* a_bf16_ptr() const { return a_bf16; }
    const bfloat16_t* b_bf16_ptr() const { return b_bf16; }
};

/* Allocate thread buffers - returns unique_ptr for automatic cleanup */
inline std::unique_ptr<ThreadBuffers> alloc_thread_buffers(BLASINT max_dim, BLASINT max_ld) {
    auto bufs = std::make_unique<ThreadBuffers>();
    if (!bufs->allocate(max_dim, max_ld)) {
        return nullptr;
    }
    return bufs;
}

/* No need for explicit free - unique_ptr handles it automatically */

#endif /* FUZZ_TEST_BUFFER_H */
