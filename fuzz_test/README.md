# UniGEMM 模糊测试工具

## 项目概述

这是一个多线程随机测试工具，用于全面测试 `cblas_sgemm`/`cblas_shgemm`/`cblas_sbgemm` 函数实现的正确性。通过独立编写的参考实现进行正确性比对，支持三种精度类型：FP32 (SGEMM)、FP16 (SHGEMM) 和 BF16 (SBGEMM)。

## 运行方式

### 十八阶段自动测试（默认）

```bash
# 使用默认参数运行
./out/fuzz_test
# 将自动检测 CPU 核心数，并运行十八个测试阶段

# 自定义总迭代次数（每个精度运行指定次数）
./out/fuzz_test --iteration 100
```

**十八阶段测试模式**：
- 3 种精度 × 3 种维度范围 × 2 种 BLAS 线程模式 = 18 个阶段
- **精度**：SGEMM (FP32)、SHGEMM (FP16)、SBGEMM (BF16)
- **维度**：Small (1-128)、Medium (1-512)、Large (1-1024)
- **线程模式**：single thread (BLAS=1)、multi thread (BLAS 随机 2-50，加权分布)

### 手动配置模式

```bash
# 指定 worker 线程数和 BLAS 线程数（跳过十八阶段测试）
./out/fuzz_test --thread 10 --blas-threads 4 --iteration 100
```

### 查看帮助信息

```bash
./out/fuzz_test -h
```

## 运行参数说明

| 参数 | 说明 |
|------|------|
| `--thread N` | 指定 worker 线程数（手动模式，跳过十八阶段测试） |
| `--blas-threads N` | 指定每个 GEMM 调用的 BLAS 线程数（需与 --thread 配合使用） |
| `--iteration N` | 指定每种精度的总迭代次数（默认 100） |
| `-h, --help` | 显示帮助信息 |

## 输出示例

### 十八阶段自动测试输出

```
======================================================================
  UniGEMM Fuzz Test - Eighteen-Stage Auto Configuration
----------------------------------------------------------------------
  Workers=8 | Iterations/precision=100 | Total=300
  SGEMM:  Small=40 Medium=40 Large=20
  SHGEMM: Small=40 Medium=40 Large=20
  SBGEMM: Small=40 Medium=40 Large=20
======================================================================

┌─ Stage 1/18 Small SGEMM single thread
  ├─ Workers: 8
  ├─ Dim range: 1-128
  ├─ Threads/worker: 1 (total: 8)
  └─ Iterations: 20
  └─ Completed in 104 ms

┌─ Stage 2/18 Small SGEMM multi thread
  ├─ Workers: 8
  ├─ Dim range: 1-128
  ├─ Threads/worker: random 2-50 (weighted)
  └─ Iterations: 20
  └─ Completed in 101 ms

...

┌─ Stage 18/18 Large SBGEMM multi thread
  ├─ Workers: 8
  ├─ Dim range: 1-1024
  ├─ Threads/worker: random 2-50 (weighted)
  └─ Iterations: 10
  └─ Completed in 520 ms


======================================================================
  ✓ All Success! ==================================================
----------------------------------------------------------------------
  Total Tests:            300  |  Time:          5665 ms
======================================================================
```

**阶段结构**：
- **SGEMM (1-6)**：FP32 精度，Small/Medium/Large 各配 single/multi 模式
- **SHGEMM (7-12)**：FP16 精度，Small/Medium/Large 各配 single/multi 模式
- **SBGEMM (13-18)**：BF16 精度，Small/Medium/Large 各配 single/multi 模式

### 失败输出示例

```
======================================================================
  Failure Details (first 20)
----------------------------------------------------------------------
┌─ Test Failure [Stage 2/18 SGEMM Small multi thread]
│  Parameters:
│    R | transA=N, transB=N | dims=[31,5,44]
│    α=-2, β=2 | lda=44, ldb=5, ldc=5 | threads=6
│  Mismatches (20 shown):
│    [0,0] impl=20.244116, ref=-20.243959, rel_err=1.9999923
│    [0,2] impl=17.892267, ref=-14.780161, rel_err=1.8260642
│    ...
└─────────────────────────────────────────────────────────
┌─ Test Failure [Stage 3/18 SGEMM Medium single thread]
│  Parameters:
│    C | transA=N, transB=T | dims=[64,54,12]
│    α=1, β=0 | lda=64, ldb=12, ldc=64 | threads=1
│  Mismatches (20 shown):
│    [0,0] impl=123.456, ref=123.455, rel_err=8.1e-06
│    ...
└─────────────────────────────────────────────────────────
----------------------------------------------------------------------

  Stage Failure Summary:
    Stage  2/18  SGEMM   Small        multi  thread:     12 failures
    Stage  3/18  SGEMM   Medium       single thread:      5 failures
    Stage 15/18  SBGEMM  Medium       multi  thread:      3 failures

======================================================================
  Final Results
----------------------------------------------------------------------
  Total Tests:           300  |  Passed:          280  |  Failed:           20
  Error Rate:       6.6667%  |  Time:          5665 ms
======================================================================
```

## 项目结构

```
fuzz_test/
├── CMakeLists.txt               # CMake 构建配置
├── build.sh                     # 便捷构建脚本
├── fuzz_test_main.cpp           # 主程序入口（十八阶段编排）
├── fuzz_test_worker.cpp         # 线程工作函数
├── fuzz_test_globals.cpp        # 全局变量（原子计数器、失败日志）
├── fuzz_test_buffer.h           # 内存缓冲区分配（HBM/64字节对齐）
├── fuzz_test_random.h           # 随机参数生成
├── fuzz_test_compare.h          # 矩阵结果比对
├── fuzz_test_report.h           # 失败信息输出
├── fuzz_test_config.h           # 配置常量（维度范围、线程配置）
├── fuzz_test_failure.h          # 失败信息结构定义
├── fuzz_test_worker.h           # 线程相关声明
├── stubs/
│   ├── shgemm_stub.cpp/h        # SHGEMM 存根实现（FP16 转换）
│   └── sbgemm_stub.cpp/h        # SBGEMM 存根实现（BF16 转换）
├── README.md                    # 本文档
├── build/                       # CMake 构建目录（自动生成）
└── out/                         # 输出目录（可执行文件）
```

## 实现细节

### 1. 独立参考实现 (`openblas.c`)

为确保测试的有效性，参考实现采用了与 `unigemm.c` 完全不同的代码结构：

- **不同的索引计算方式**：使用 `get_elem`/`put_elem` 辅助函数封装索引计算
- **避免共享 bug**：不同的代码结构确保两个实现不会因为相同的逻辑错误而得出相同的错误结果

### 2. 三种精度类型

| 精度 | 数据类型 | 接口函数 | 容差 |
|------|----------|----------|------|
| SGEMM | `float` (FP32) | `cblas_sgemm` | 1e-2 |
| SHGEMM | `float16_t` (FP16) | `cblas_shgemm` | 1e-1 |
| SBGEMM | `bfloat16_t` (BF16) | `cblas_sbgemm` | 1e-1 |

**数据生成策略**：SHGEMM/SBGEMM 直接以原始精度生成数据（随机 uint16 位模式），然后扩展为 float 给 reference 使用，确保 impl 和 ref 操作完全相同的精度值，差异只来自 GEMM 实现本身。

**SHGEMM/SBGEMM 路径**：
1. 生成 FP16/BF16 原始数据（随机 uint16 位模式）
2. 将 FP16/BF16 扩展为 float（给 ref 和 impl 共用）
3. impl 走 stub（FP16/BF16 → float → cblas_sgemm）
4. ref 直接用扩展后的 float 调用 reference

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

**线程隔离**：每个线程使用独立的 RNG 种子、独立的缓冲区（A、B、C_impl、C_ref），无共享可变状态。

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
#define SGEMM_TOLERANCE  0.01f  // SGEMM 相对误差容差
#define SHGEMM_TOLERANCE 0.1f   // SHGEMM 相对误差容差
#define SBGEMM_TOLERANCE 0.1f   // SBGEMM 相对误差容差
```

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
   - 十八阶段覆盖 3 种精度 × 3 种维度 × 2 种线程模式
   - 可通过修改 `fuzz_test_config.h` 中的常量调整

3. **内存使用**：
   - 每个线程约 4MB（4 个缓冲区 × 1MB）
   - 缓冲区对齐到 64 字节边界

4. **HBM 模式**：使用 `-DUSE_HBM=ON` 编译，需要链接提供 `AllocateMemory`/`FreeMemory` 的库

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
