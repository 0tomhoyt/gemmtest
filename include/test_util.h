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

#endif /* FUZZ_TEST_UTIL_H */
