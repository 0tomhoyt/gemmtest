#ifndef FUZZ_TEST_REPORT_H
#define FUZZ_TEST_REPORT_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_worker.h"
#include <iostream>
#include <iomanip>

/* Get transpose name string */
inline const char *trans_name(enum CBLAS_TRANSPOSE trans) {
    switch (trans) {
        case CblasNoTrans: return "N";
        case CblasTrans: return "T";
        default: return "?";
    }
}

/* Get order name string */
inline const char *order_name(enum CBLAS_ORDER order) {
    switch (order) {
        case CblasRowMajor: return "R";
        case CblasColMajor: return "C";
        default: return "?";
    }
}

/* Get precision name string */
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

#endif /* FUZZ_TEST_REPORT_H */
