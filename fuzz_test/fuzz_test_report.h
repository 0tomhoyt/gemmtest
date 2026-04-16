#ifndef FUZZ_TEST_REPORT_H
#define FUZZ_TEST_REPORT_H

#include "gemm_benchmark.h"
#include "unigemm_920f.h"
#include "fuzz_test_failure.h"
#include <iostream>
#include <iomanip>

/* Get transpose name string */
inline const char* trans_name(enum CBLAS_TRANSPOSE trans) {
    switch (trans) {
        case CblasNoTrans: return "N";
        case CblasTrans: return "T";
        default: return "?";
    }
}

/* Get order name string */
inline const char* order_name(enum CBLAS_ORDER order) {
    switch (order) {
        case CblasRowMajor: return "R";
        case CblasColMajor: return "C";
        default: return "?";
    }
}

/* Print failure details */
inline void print_failure(const FailureInfo& info) {
    std::cout << "┌─ Test Failure\n";
    std::cout << "│  Parameters:\n";
    std::cout << "│    " << order_name(info.order) << " | "
              << "transA=" << trans_name(info.transA) << ", "
              << "transB=" << trans_name(info.transB) << " | "
              << "dims=[" << info.m << "," << info.n << "," << info.k << "]\n";
    std::cout << std::setprecision(6);
    std::cout << "│    α=" << info.alpha << ", β=" << info.beta << " | "
              << "lda=" << info.lda << ", ldb=" << info.ldb << ", ldc=" << info.ldc << " | "
              << "threads=" << info.num_threads << "\n";

    std::cout << "│  Mismatches (" << info.num_mismatches << " shown):\n";
    for (int idx = 0; idx < info.num_mismatches; idx++) {
        const MismatchRecord& m = info.mismatches[idx];
        std::cout << std::setprecision(8);
        std::cout << "│    [" << m.i << "," << m.j << "] "
                  << "impl=" << m.impl_val << ", ref=" << m.ref_val << ", "
                  << "rel_err=" << m.rel_error << "\n";
    }
    std::cout << "└─────────────────────────────────────────────────────────\n";
}

#endif /* FUZZ_TEST_REPORT_H */
