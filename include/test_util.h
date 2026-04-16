#ifndef FUZZ_TEST_UTIL_H
#define FUZZ_TEST_UTIL_H

#include <cstdlib>

/* InitMatrix: initialize matrix with random values in [0, 1] using rand() */
template<typename T>
void InitMatrix(T *data, size_t count, unsigned int seed = 0) {
    std::srand(seed);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(std::rand() / static_cast<double>(RAND_MAX));
    }
}

#endif /* FUZZ_TEST_UTIL_H */
