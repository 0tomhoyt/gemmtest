#pragma once

#include "st_common.h"
#include "st_worker.h"

/* Embedded CSV test cases (generated from test_cases.csv) */
static const TestParams g_csv_cases[] = {
    { PrecisionType::SGEMM,  128, 128, 128, CblasNoTrans, CblasNoTrans, CblasRowMajor, 1.0f, 0.0f },
    { PrecisionType::SGEMM,   64,  64,  64, CblasTrans,   CblasNoTrans, CblasColMajor, 1.0f, 1.0f },
    { PrecisionType::SHGEMM, 100, 100, 100, CblasNoTrans, CblasNoTrans, CblasRowMajor, 1.0f, 0.0f },
    { PrecisionType::SBGEMM,  50,  50,  50, CblasNoTrans, CblasTrans,   CblasRowMajor, 1.5f, 0.5f },
    { PrecisionType::HGEMM,   32,  32,  32, CblasNoTrans, CblasNoTrans, CblasColMajor, 1.0f, 0.0f },
    { PrecisionType::BGEMM,   20,  20,  20, CblasTrans,   CblasTrans,   CblasRowMajor, 1.0f, 1.0f },
};
static constexpr int g_csv_case_count = sizeof(g_csv_cases) / sizeof(g_csv_cases[0]);
