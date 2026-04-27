#ifndef UNIGEMM_H
#define UNIGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int BLASINT;

typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans   = 111,
    CblasTrans     = 112,
    CblasConjTrans = 113,
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

void cblas_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
                 const BLASINT m, const BLASINT n, const BLASINT k, const float alpha, const float *a,
                 const BLASINT lda, const float *b, const BLASINT ldb, const float beta, float *c, const BLASINT ldc);

/* ============================================================
 * 半精度浮点类型定义 (Half-Precision Floating Point Types)
 * ============================================================ */

/* 半精度浮点类型定义 */
#if defined(__FLT16_MAX__)
typedef _Float16 float16_t;
#elif defined(__FP16_IETF__)
typedef __fp16 float16_t;
#else
typedef uint16_t float16_t;  /* 回退到 uint16，视为 IEEE 754 half precision 格式 */
#endif

/* SHGEMM 函数声明 */
/* 注意：alpha 和 beta 是 float，a 和 b 是 fp16，c 是 float (标准 BLAS 接口) */
void cblas_shgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                  const BLASINT k, const float alpha, const float16_t *a,
                  const BLASINT lda, const float16_t *b, const BLASINT ldb,
                  const float beta, float *c, const BLASINT ldc);

/* ============================================================
 * BF16 浮点类型定义 (Bfloat16 Floating Point Types)
 * ============================================================ */

#if defined(__BF16_TYPE__)
typedef __bf16 bfloat16_t;
#else
#include <stdint.h>
typedef uint16_t bfloat16_t;  /* 回退到 uint16，需用位操作转换 */
#endif

/* SBGEMM 函数声明 */
/* 注意：alpha 和 beta 是 float，a 和 b 是 bfloat16，c 是 float (标准 BLAS 接口) */
void cblas_sbgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB, const BLASINT m, const BLASINT n,
                  const BLASINT k, const float alpha, const bfloat16_t *a,
                  const BLASINT lda, const bfloat16_t *b, const BLASINT ldb,
                  const float beta, float *c, const BLASINT ldc);

void BlasSetNumThreadsLocal(int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* UNIGEMM_H */