#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <cstdio>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include "gemm_benchmark.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

//time
clock_t Now();
double ElapsedTime(clock_t start, clock_t end);

//cpu
double GetCurrentFreq();
double GetCpuPeakFp16();
double GetCpuPeakFp32();
double GetCpuPeakFp64();
float GetTolerance();

//memory
constexpr size_t HBM_2M_SIZE = 512ULL;
constexpr size_t HBM_ALIGNED_SIZE = (1 << 21);
constexpr size_t DEFAULT_ALIGN = 64;

template<typename T>
T* AllocateMemory(size_t count, size_t align = 64, bool useHBM = false);
template<typename T>
void FreeMemory(size_t count, T *ptr, bool useHBM = false);

//matrix
template<typename T>
void PrintMatrix(const char *name, const T *data, int rows, int cols, int ldc, bool rowMajor);

template<typename T>
void InitMatrix(T *data, size_t count, unsigned int seed = 0);

template<typename T1, typename T2>
bool CheckMatrixResult(const T1 *ref, const T2 *test, int rows, int cols, int ldc,
                    double eps, bool verbose, bool rowMajor);

static inline int ReadIntEnvParam(const char *env);
int GetNumThreads();

std::string GetGenerateTestName(const ParamCombination &param, int mode);

#endif