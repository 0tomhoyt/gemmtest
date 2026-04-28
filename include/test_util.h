#ifndef FUZZ_TEST_UTIL_H
#define FUZZ_TEST_UTIL_H

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>

/* InitMatrix: initialize matrix with random values in [0, 1] using rand()
 * Template definition must be in header for implicit instantiation.
 */
template<typename T>
void InitMatrix(T *data, size_t count, unsigned int seed = 0) {
    std::srand(seed);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(std::rand() / static_cast<double>(RAND_MAX));
    }
}

/* Explicit instantiation for float (avoids duplicate symbol in multiple TUs) */
extern template void InitMatrix<float>(float *, size_t, unsigned int);

/* CheckMatrixResult: compare ref vs test matrices with relative tolerance
 * T1 = reference type, T2 = test type (may differ, e.g. float vs float16_t)
 * Returns true if all elements match within eps, false otherwise.
 * verbose=true: prints up to 20 mismatches with position, values, and relative error.
 */
template <typename T1, typename T2>
bool CheckMatrixResult(const T1 *ref, const T2 *test, int rows, int cols,
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
                    std::printf("  [%d,%d] ref=%.8g, test=%.8g, rel_err=%.8g\n",
                                i, j, val_ref, val_test, rel_err);
                }
            }
        }
    }

    if (verbose && mismatch_count > max_print) {
        std::printf("  ... %d more mismatches (total %d)\n",
                     mismatch_count - max_print, mismatch_count);
    }

    return mismatch_count == 0;
}

#endif /* FUZZ_TEST_UTIL_H */
