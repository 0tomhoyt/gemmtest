#ifndef FUZZ_TEST_COMPARE_H
#define FUZZ_TEST_COMPARE_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_failure.h"
#include "fuzz_test_config.h"
#include <cmath>
#include <cstring>
#include <algorithm>

/* Compare two matrices with relative tolerance
 * Returns true if matrices match within tolerance, false otherwise
 * If info is non-NULL and mismatch found, fills it with first mismatch details
 */
inline bool compare_matrices(const float *c_impl, const float *c_ref,
                             enum CBLAS_ORDER order, BLASINT m, BLASINT n,
                             BLASINT ldc, float tolerance,
                             FailureInfo *info) {
    bool has_mismatch = false;

    /* Initialize mismatch tracking */
    if (info != nullptr) {
        info->num_mismatches = 0;
    }

    for (BLASINT i = 0; i < m; i++) {
        for (BLASINT j = 0; j < n; j++) {
            float val_impl, val_ref;

            if (order == CblasRowMajor) {
                val_impl = c_impl[i * ldc + j];
                val_ref = c_ref[i * ldc + j];
            } else {
                val_impl = c_impl[j * ldc + i];
                val_ref = c_ref[j * ldc + i];
            }

            float max_val = std::max({std::fabs(val_impl), std::fabs(val_ref), 1.0f});
            float diff = std::fabs(val_impl - val_ref);
            float rel_error = diff / max_val;

            if (diff >= tolerance * max_val) {
                /* Found mismatch */
                has_mismatch = true;

                if (info != nullptr) {
                    /* Record first mismatch for backward compatibility */
                    if (info->num_mismatches == 0) {
                        info->fail_i = i;
                        info->fail_j = j;
                        info->impl_val = val_impl;
                        info->ref_val = val_ref;
                        info->rel_error = rel_error;
                    }

                    /* Collect up to MAX_MISMATCHES records */
                    if (info->num_mismatches < MAX_MISMATCHES) {
                        info->mismatches[info->num_mismatches].i = i;
                        info->mismatches[info->num_mismatches].j = j;
                        info->mismatches[info->num_mismatches].impl_val = val_impl;
                        info->mismatches[info->num_mismatches].ref_val = val_ref;
                        info->mismatches[info->num_mismatches].rel_error = rel_error;
                        info->num_mismatches++;
                    }
                }
            }
        }
    }

    return !has_mismatch;
}

#endif /* FUZZ_TEST_COMPARE_H */
