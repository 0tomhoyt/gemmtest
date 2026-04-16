#ifndef FUZZ_TEST_BUFFER_H
#define FUZZ_TEST_BUFFER_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_config.h"
#include <cstdlib>
#include <memory>
#include <cassert>

#ifdef USE_HBM
#include "test_util.h"
#endif

/* Buffer for matrix data (allocated per thread with 64-byte alignment) */
struct ThreadBuffers {
    float* a_buf;
    float* b_buf;
    float* c_impl_buf;
    float* c_ref_buf;
    size_t max_size;

    ThreadBuffers() : a_buf(nullptr), b_buf(nullptr),
                      c_impl_buf(nullptr), c_ref_buf(nullptr), max_size(0) {}

    /* Allocate buffers with specified size and 64-byte alignment */
    bool allocate(BLASINT max_dim, BLASINT max_ld) {
        max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

#ifdef USE_HBM
        /* Use test_util.h AllocateMemory for HBM with 64-byte alignment */
        a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        b_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        c_impl_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
        c_ref_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
#else
        /* Use posix_memalign for cross-platform compatibility */
        size_t byte_size = max_size * sizeof(float);
        if (posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, byte_size) != 0) a_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&b_buf), BUFFER_ALIGNMENT, byte_size) != 0) b_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&c_impl_buf), BUFFER_ALIGNMENT, byte_size) != 0) c_impl_buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&c_ref_buf), BUFFER_ALIGNMENT, byte_size) != 0) c_ref_buf = nullptr;
#endif

        if (!a_buf || !b_buf || !c_impl_buf || !c_ref_buf) {
            return false;
        }

        /* Verify 64-byte alignment */
        assert((reinterpret_cast<uintptr_t>(a_buf) % BUFFER_ALIGNMENT == 0) && "a_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(b_buf) % BUFFER_ALIGNMENT == 0) && "b_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(c_impl_buf) % BUFFER_ALIGNMENT == 0) && "c_impl_buf not 64-byte aligned");
        assert((reinterpret_cast<uintptr_t>(c_ref_buf) % BUFFER_ALIGNMENT == 0) && "c_ref_buf not 64-byte aligned");

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
#else
        std::free(a_buf);
        std::free(b_buf);
        std::free(c_impl_buf);
        std::free(c_ref_buf);
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

    const float* a_ptr() const { return a_buf; }
    const float* b_ptr() const { return b_buf; }
    const float* c_impl_ptr() const { return c_impl_buf; }
    const float* c_ref_ptr() const { return c_ref_buf; }
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
