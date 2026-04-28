# UniGEMM 模糊测试工具

## 项目概述

这是一个多线程随机测试工具，用于全面测试 `cblas_sgemm`/`cblas_shgemm`/`cblas_sbgemm`/`cblas_hgemm`/`cblas_bgemm` 函数实现的正确性。通过独立编写的参考实现进行正确性比对，支持五种精度类型：FP32 (SGEMM)、FP16 (SHGEMM/HGEMM) 和 BF16 (SBGEMM/BGEMM)。

## 运行方式

### 三十阶段自动测试（默认）

```bash
# 使用默认参数运行
./out/fuzz_test
# 将自动检测 CPU 核心数，并运行三十个测试阶段

# 自定义总迭代次数（每个精度运行指定次数）
./out/fuzz_test --iteration 100
```

**三十阶段测试模式**：
- 5 种精度 × 3 种维度范围 × 2 种 BLAS 线程模式 = 30 个阶段
- **精度**：SGEMM (FP32)、SHGEMM (FP16 A/B, FP32 alpha/beta/C)、SBGEMM (BF16 A/B, FP32 alpha/beta/C)、HGEMM (全 FP16)、BGEMM (全 BF16)
- **维度**：Small (1-128)、Medium (1-512)、Large (1-1024)
- **线程模式**：single thread (BLAS=1)、multi thread (BLAS 随机 2-50，加权分布)

### 手动配置模式

```bash
# 指定 worker 线程数和 BLAS 线程数（跳过三十阶段测试）
./out/fuzz_test --thread 10 --blas-threads 4 --iteration 100
```

### 查看帮助信息

```bash
./out/fuzz_test -h
```

## 运行参数说明

| 参数 | 说明 |
|------|------|
| `--thread N` | 指定 worker 线程数（手动模式，跳过三十阶段测试） |
| `--blas-threads N` | 指定每个 GEMM 调用的 BLAS 线程数（需与 --thread 配合使用） |
| `--iteration N` | 指定每种精度的总迭代次数（默认 100） |
| `-h, --help` | 显示帮助信息 |

## 输出示例

### 三十阶段自动测试输出

```
======================================================================
  UniGEMM Fuzz Test - Thirty-Stage Auto Configuration
----------------------------------------------------------------------
  Workers=8 | Iterations/precision=100 | Total=500
  SGEMM:  Small=40 Medium=40 Large=20
  SHGEMM: Small=40 Medium=40 Large=20
  SBGEMM: Small=40 Medium=40 Large=20
  HGEMM:  Small=40 Medium=40 Large=20
  BGEMM:  Small=40 Medium=40 Large=20
======================================================================

┌─ Stage 1/30 Small SGEMM single thread
  ├─ Workers: 8
  ├─ Dim range: 1-128
  ├─ Threads/worker: 1 (total: 8)
  └─ Iterations: 20
  └─ Completed in 104 ms

┌─ Stage 2/30 Small SGEMM multi thread
  ├─ Workers: 8
  ├─ Dim range: 1-128
  ├─ Threads/worker: random 2-50 (weighted)
  └─ Iterations: 20
  └─ Completed in 101 ms

...

┌─ Stage 30/30 Large BGEMM multi thread
  ├─ Workers: 8
  ├─ Dim range: 1-1024
  ├─ Threads/worker: random 2-50 (weighted)
  └─ Iterations: 10
  └─ Completed in 520 ms


======================================================================
  ✓ All Success! ==================================================
----------------------------------------------------------------------
  Total Tests:            500  |  Time:          5665 ms
======================================================================
```

**阶段结构**：
- **SGEMM (1-6)**：FP32 精度，Small/Medium/Large 各配 single/multi 模式
- **SHGEMM (7-12)**：FP16 A/B + FP32 alpha/beta/C，Small/Medium/Large 各配 single/multi 模式
- **SBGEMM (13-18)**：BF16 A/B + FP32 alpha/beta/C，Small/Medium/Large 各配 single/multi 模式
- **HGEMM (19-24)**：全 FP16 (alpha/beta/A/B/C)，Small/Medium/Large 各配 single/multi 模式
- **BGEMM (25-30)**：全 BF16 (alpha/beta/A/B/C)，Small/Medium/Large 各配 single/multi 模式

### 失败输出示例

```
======================================================================
  Failure Details (first 20)
----------------------------------------------------------------------
┌─ Test Failure [Stage 2/30 SGEMM Small multi thread]
│  Parameters:
│    R | transA=N, transB=N | dims=[31,5,44]
│    α=-2, β=2 | lda=44, ldb=5, ldc=5 | threads=6
│  Mismatches (20 shown):
│    [0,0] impl=20.244116, ref=-20.243959, rel_err=1.9999923
│    [0,2] impl=17.892267, ref=-14.780161, rel_err=1.8260642
│    ...
└─────────────────────────────────────────────────────────
┌─ Test Failure [Stage 3/30 SGEMM Medium single thread]
│  Parameters:
│    C | transA=N, transB=T | dims=[64,54,12]
│    α=1, β=0 | lda=64, ldb=12, ldc=64 | threads=1
│  Mismatches (20 shown):
│    [0,0] impl=123.456, ref=123.455, rel_err=8.1e-06
│    ...
└─────────────────────────────────────────────────────────
----------------------------------------------------------------------

  Stage Failure Summary:
    Stage  2/30  SGEMM   Small        multi  thread:     12 failures
    Stage  3/30  SGEMM   Medium       single thread:      5 failures
    Stage 15/30  SBGEMM  Medium       multi  thread:      3 failures

======================================================================
  Final Results
----------------------------------------------------------------------
  Total Tests:           500  |  Passed:          480  |  Failed:           20
  Error Rate:       4.0000%  |  Time:          5665 ms
======================================================================
```

## 项目结构

```
fuzz_test/
├── CMakeLists.txt               # CMake 构建配置
├── build.sh                     # 便捷构建脚本
├── fuzz_test_main.cpp           # 主程序入口（三十阶段编排）
├── fuzz_test_worker.cpp         # 线程工作函数
├── fuzz_test_globals.cpp        # 全局变量（原子计数器、失败日志）
├── fuzz_test_buffer.h           # 内存缓冲区分配（HBM/64字节对齐）
├── fuzz_test_random.h           # 随机参数生成
├── fuzz_test_compare.h          # 矩阵结果比对
├── fuzz_test_report.h           # 失败信息输出
├── fuzz_test_config.h           # 配置常量（维度范围、线程配置、容差）
├── fuzz_test_failure.h          # 失败信息结构定义
├── fuzz_test_worker.h           # 线程相关声明
├── stubs/
│   ├── shgemm_stub.cpp/h        # SHGEMM 存根实现（FP16 A/B → float → cblas_sgemm）
│   ├── sbgemm_stub.cpp/h        # SBGEMM 存根实现（BF16 A/B → float → cblas_sgemm）
│   ├── hgemm_stub.cpp/h         # HGEMM 存根实现（全 FP16 → float → cblas_sgemm → FP16）
│   └── bgemm_stub.cpp/h         # BGEMM 存根实现（全 BF16 → float → cblas_sgemm → BF16）
├── README.md                    # 本文档
├── build/                       # CMake 构建目录（自动生成）
└── out/                         # 输出目录（可执行文件）
```

## 实现细节

### 1. 独立参考实现 (`openblas.c`)

为确保测试的有效性，参考实现采用了与 `unigemm.c` 完全不同的代码结构：

- **不同的索引计算方式**：使用 `get_elem`/`put_elem` 辅助函数封装索引计算
- **避免共享 bug**：不同的代码结构确保两个实现不会因为相同的逻辑错误而得出相同的错误结果

### 2. 五种精度类型

| 精度 | alpha/beta | A, B | C | 接口函数 | 容差 |
|------|-----------|------|---|----------|------|
| SGEMM | float | float | float | `cblas_sgemm` | 1e-4 |
| SHGEMM | float | float16_t | float | `cblas_shgemm` | 1e-3 |
| SBGEMM | float | bfloat16_t | float | `cblas_sbgemm` | 1e-3 |
| HGEMM | float16_t | float16_t | float16_t | `cblas_hgemm` | 5e-3 |
| BGEMM | bfloat16_t | bfloat16_t | bfloat16_t | `cblas_bgemm` | 5e-2 |

**数据生成策略**：所有精度类型使用 `InitMatrix` 生成 [0,1] 的 float 数据，然后转换为目标精度（FP16/BF16）给 impl 使用，同时扩展回 float 给 ref 使用。确保 impl 和 ref 操作完全相同的精度值，差异只来自 GEMM 实现本身和半精度转换的精度损失。

**精度转换**：
- FP16 ↔ float：使用位操作（`memcpy` + shift 16 bits）实现，兼容 Mac 平台
- BF16 ↔ float：使用位操作（`memcpy` + shift 16 bits）实现，Mac 上 `bfloat16_t = uint16_t` 需要特殊处理

**数据流对比**：

**SGEMM**（纯 FP32）：
```
InitMatrix → float A, B, C
impl: cblas_sgemm(alpha, A, B, beta, C_impl)
ref:  cblas_sgemm_ref(alpha, A, B, beta, C_ref)
compare(C_impl, C_ref)
```

**SHGEMM**（FP16 A/B, FP32 alpha/beta/C）：
```
InitMatrix → float16_t A, B
InitMatrix → float C, copy to C_ref
FP16 → float for ref: A_ref, B_ref
impl: cblas_shgemm(alpha, A_half, B_half, beta, C_impl)
ref:  cblas_sgemm_ref(alpha, A_ref, B_ref, beta, C_ref)
compare(C_impl, C_ref)
```

**HGEMM**（全 FP16）：
```
InitMatrix → float16_t A, B, C_half
FP16 → float for ref: A_ref, B_ref
InitMatrix → float C, copy to C_ref
impl: cblas_hgemm(alpha_half, A_half, B_half, beta_half, C_half)
ref:  cblas_sgemm_ref(alpha, A_ref, B_ref, beta, C_ref)
FP16 → float for compare: C_impl = float(C_half)
compare(C_impl, C_ref)
```

**BGEMM**（全 BF16）：
```
InitMatrix → float A_temp, B_temp, C_temp
float → BF16 for impl: A_bf16, B_bf16, C_bf16 (bit manipulation)
BF16 → float for ref: A_ref, B_ref
C_temp → C_ref
impl: cblas_bgemm(alpha_bf16, A_bf16, B_bf16, beta_bf16, C_bf16)
ref:  cblas_sgemm_ref(alpha, A_ref, B_ref, beta, C_ref)
BF16 → float for compare: C_impl = float(C_bf16)
compare(C_impl, C_ref)
```

### 3. 参数随机化策略

| 参数 | 随机化策略 |
|------|------------|
| `order` | RowMajor / ColMajor 各 50% |
| `transA/transB` | NoTrans / Trans 各 50% |
| `m, n, k` | 根据阶段固定范围：Small (1-128) / Medium (1-512) / Large (1-1024) |
| `alpha, beta` | 70% 特殊值 {0,±1,±2,±0.5,±0.25}，30% 随机值 [-10,10] |
| `lda/ldb/ldc` | 最小值（上限 clamp 到 MAX_LD） |
| `BLAS threads` | single 模式固定 1；multi 模式加权随机 (2-50，线程数越大概率越低) |

**维度分布概率**（在 `fuzz_test_config.h` 中配置）：
- Small (1-128)：40%
- Medium (1-512)：40%
- Large (1-1024)：20%

### 4. 失败报告

每个失败记录包含：
- **Stage 信息**：Stage 编号、精度类型、维度范围、线程模式
- **参数详情**：布局、转置模式、维度、标量系数、leading dimensions
- **不匹配记录**：最多 20 个位置的详细误差信息

最终报告包含 **Stage Failure Summary**，按 stage 汇总失败数量。

### 5. HBM 内存分配与对齐支持

矩阵缓冲区支持两种分配模式，通过编译宏 `-DUSE_HBM` 切换：

| 模式 | 分配函数 | 对齐 | NUMA |
|------|----------|------|------|
| 默认 | `posix_memalign` | 64 字节 | 否 |
| HBM | `AllocateMemory<T>` | 64 字节 | 是 |

### 6. 多线程测试模型

测试采用**两层嵌套并行**架构：
- **外层**：`std::thread` worker 线程并发执行测试迭代
- **内层**：OpenMP 线程在单个 GEMM 操作内部并行

**线程隔离**：每个线程使用独立的 RNG 种子、独立的缓冲区（A、B、C_impl、C_ref、A_half、B_half、C_half、A_bf16、B_bf16、C_bf16），无共享可变状态。

## 配置选项

### 维度范围与概率 (`fuzz_test_config.h`)

```cpp
/* 维度范围定义 */
constexpr int DIM_RANGE_SMALL = 128;      // 小维度范围
constexpr int DIM_RANGE_MEDIUM = 512;     // 中维度范围
constexpr int DIM_RANGE_LARGE = 1024;     // 大维度范围

/* 维度分布概率 */
constexpr int DIM_PROB_SMALL = 40;        // Small 范围概率 (%)
constexpr int DIM_PROB_MEDIUM = 40;       // Medium 范围概率 (%)
constexpr int DIM_PROB_LARGE = 20;        // Large 范围概率 (%)
```

### 线程配置 (`fuzz_test_config.h`)

```cpp
#ifdef USE_HBM
#define MAX_WORKERS 32    // HBM 模式默认 32 worker
#else
#define MAX_WORKERS 8     // 非 HBM 模式默认 8 worker
#endif

#define MAX_BLAS_THREADS 50  // 多线程阶段最大 BLAS 线程数
#define MAX_LD 1024           // Leading dimension 上限
```

**运行时覆盖**：通过 `UNIGEMM_MAX_WORKERS` 环境变量覆盖 worker 数量。

### 容差配置 (`fuzz_test_config.h`)

```cpp
constexpr float SGEMM_TOLERANCE  = 1e-4f;  // SGEMM 相对误差容差
constexpr float SHGEMM_TOLERANCE = 1e-3f;  // SHGEMM 相对误差容差
constexpr float SBGEMM_TOLERANCE = 1e-3f;  // SBGEMM 相对误差容差
constexpr float HGEMM_TOLERANCE  = 5e-3f;  // HGEMM 相对误差容差（FP16 C round-trip）
constexpr float BGEMM_TOLERANCE  = 5e-2f;  // BGEMM 相对误差容差（BF16 C round-trip）
```

**容差说明**：
- SGEMM: 纯 FP32，最小容差
- SHGEMM/SBGEMM: FP32 alpha/beta/C，容差适中
- HGEMM: 全 FP16，C 矩阵经过 FP16↔float 双重转换，容差增加
- BGEMM: 全 BF16，C 矩阵经过 BF16↔float 双重转换，精度损失最大，容差最高

### OpenMP 环境变量配置

Stage 2（多线程阶段）会创建大量 OpenMP 线程（随机 2-50）。如果遇到资源限制：

```bash
export OMP_THREAD_LIMIT=256    # 限制同时执行的最大线程数
export OMP_STACKSIZE=2M         # 减少每线程栈大小
export OMP_WAIT_POLICY=passive  # 使用被动等待策略
export KMP_BLOCKTIME=50ms       # 减少线程保持活跃时间
```

## 注意事项

1. **线程控制**：
   - `MAX_WORKERS` 控制外层 `std::thread` 数量
   - 可通过 `UNIGEMM_MAX_WORKERS` 环境变量运行时覆盖
   - 如果触发 OMP Error #34，通过 `OMP_THREAD_LIMIT` 限制 OpenMP 资源

2. **测试覆盖**：
   - 三十阶段覆盖 5 种精度 × 3 种维度 × 2 种线程模式
   - 可通过修改 `fuzz_test_config.h` 中的常量调整

3. **内存使用**：
   - 每个线程约 4MB（4 个 float 缓冲区 × 1MB）+ 半精度缓冲区
   - 缓冲区对齐到 64 字节边界

4. **HBM 模式**：使用 `-DUSE_HBM=ON` 编译，需要链接提供 `AllocateMemory`/`FreeMemory` 的库

5. **平台兼容性**：
   - Mac 平台 `bfloat16_t = uint16_t`，必须使用位操作进行 float↔BF16 转换
   - HGEMM 在 Mac 上表现正常（`float16_t = _Float16`）
   - BGEMM 在 Mac 上通过位操作正确处理，但精度特性与服务器可能不同

## 构建与运行

```bash
# MacBook 开发模式
cd fuzz_test && bash build.sh --run

# 手动配置测试
./out/fuzz_test --thread 4 --iteration 500

# HBM 模式（ARM64 服务器）
mkdir build && cd build
cmake -DUSE_HBM=ON .. && cmake --build .
./out/fuzz_test --iteration 1000
```
