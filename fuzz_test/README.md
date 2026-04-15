# SGEMM 模糊测试工具

## 项目概述

这是一个多线程随机测试工具，用于全面测试 `cblas_sgemm` 函数实现的正确性。由于系统中没有参考 BLAS 库，本项目通过独立编写的参考实现进行正确性比对。

## 运行方式

### 两阶段自动测试（默认）

```bash
# 使用默认参数运行
fuzz_test
# 将自动检测 CPU 核心数，并运行两个测试阶段

# 自定义总迭代次数（50% 单线程 + 50% 多线程）
fuzz_test --iteration 5000
```

**两阶段测试模式**：
- **Stage 1**：50% 迭代次数，单线程（BLAS threads = 1）
- **Stage 2**：50% 迭代次数，多线程（BLAS threads 随机 2-50，加权分布）

### 手动配置模式

```bash
# 指定 worker 线程数和 BLAS 线程数
fuzz_test --thread 10 --blas-threads 4 --iteration 100

# 仅指定迭代次数（使用默认两阶段配置）
fuzz_test --iteration 100
```

### 查看帮助信息

```bash
fuzz_test -h
```

## 运行参数说明

| 参数 | 说明 |
|------|------|
| `--thread N` | 指定 worker 线程数（手动模式，跳过两阶段测试） |
| `--blas-threads N` | 指定每个 GEMM 调用的 BLAS 线程数（需与 --thread 配合使用） |
| `--iteration N` | 指定总迭代次数（默认 100） |
| `-h, --help` | 显示帮助信息 |

## 输出示例

### 两阶段自动测试输出

```
======================================================================
  UniGEMM Fuzz Test - Two-Stage Auto Configuration
----------------------------------------------------------------------
  Configuration: Workers=8 | Total Iterations=100
======================================================================

┌─ Stage 1/2 (BLAS threads=1, Single-threaded)
  ├─ Workers: 8
  ├─ BLAS threads/worker: 1 (total: 8 threads)
  └─ Iterations: 50
  └─ Completed in 413 ms

┌─ Stage 2/2 (BLAS threads=random 2-50, Multi-threaded)
  ├─ Workers: 8
  ├─ BLAS threads/worker: random 2-50 (weighted)
  └─ Iterations: 50
  └─ Completed in 311 ms

======================================================================
  ✓ All Success! ==================================================
----------------------------------------------------------------------
  Total Tests:            100  |  Time:           724 ms
======================================================================
```

**阶段说明**：
- **Stage 1**：单线程测试，验证基础正确性
- **Stage 2**：多线程测试，BLAS 线程数随机（2-50），使用加权分布（线程数越大概率越低）

### 手动配置模式输出

```
======================================================================
  UniGEMM Fuzz Test - Manual Configuration
======================================================================
  ├─ Workers: 10
  ├─ BLAS threads/worker: 4 (total: 40 threads)
  └─ Iterations: 100

======================================================================
  ✓ All Success! ==================================================
----------------------------------------------------------------------
  Total Tests:            100  |  Time:           542 ms
======================================================================
```

### 失败输出示例

```
========================================
Final Results:
  Total:   100
  Passed:  98
  Failed:  2
  Error rate: 2%
========================================

First 2 failures:
  Failure:
    order=RowMajor transA=Trans transB=NoTrans
    m=8 n=16 k=4 alpha=1.00000 beta=0.500000
    lda=8 ldb=5 ldc=17
    First mismatch at [3,7]: impl=142.35678 ref=142.35982 rel_error=2.13e-05
  Failure:
    order=ColMajor transA=NoTrans transB=ConjTrans
    m=32 n=0 k=15 alpha=0.000000 beta=1.00000
    lda=33 ldb=16 ldc=1
    First mismatch at [0,0]: impl=-5.4321000 ref=-5.4320984 rel_error=3.67e-07
```

## 项目结构

```
fuzz_test/
├── CMakeLists.txt           # CMake 构建配置
├── build.sh                 # 便捷构建脚本
├── test_buggy.sh            # 错误检测测试脚本
├── fuzz_test_main.cpp       # 主程序入口
├── fuzz_test_worker.cpp     # 线程工作函数
├── fuzz_test_globals.cpp    # 全局变量（原子计数器、失败日志）
├── fuzz_test_buffer.cpp     # 内存缓冲区分配（支持 HBM/64字节对齐）
├── fuzz_test_random.cpp     # 随机参数生成
├── fuzz_test_compare.cpp    # 矩阵结果比对
├── fuzz_test_report.cpp     # 失败信息输出
├── fuzz_test_config.h       # 配置常量（维度范围、线程配置）
├── fuzz_test_failure.h      # 失败信息结构定义
├── fuzz_test_worker.h       # 线程相关声明
├── README.md                # 本文档
├── build/                   # CMake 构建目录（自动生成）
└── out/                     # 输出目录（可执行文件）
```

## 实现细节

### 1. 独立参考实现 (openblas.c)

为确保测试的有效性，参考实现采用了与 `unigemm.c` 完全不同的代码结构：

- **不同的索引计算方式**：使用 `get_elem`/`put_elem` 辅助函数封装索引计算，而非内联布尔分支
- **分离的有效维度计算**：通过 `effective_rows`/`effective_cols` 函数处理转置后的矩阵维度
- **避免共享 bug**：不同的代码结构确保两个实现不会因为相同的逻辑错误而得出相同的错误结果

### 2. 参数随机化策略

测试程序通过加权随机方式覆盖各种参数组合：

| 参数 | 随机化策略 |
|------|------------|
| `order` | RowMajor / ColMajor 各 50% 概率 |
| `transA/transB` | 2 个枚举值 (NoTrans/Trans，实数 SGEMM 中 ConjTrans=Trans) |
| `m, n, k` | 40% 概率 0-128（小维度），40% 概率 0-512（中维度），20% 概率 0-1024（大维度） |
| `alpha, beta` | 70% 概率选择特殊值 {0,±1,±2,±0.5,±0.25}，30% 概率选择 [-10,10] 随机浮点数 |
| `lda/ldb/ldc` | 最小值 + 0-7 的随机 padding |
| `BLAS threads` | Stage 1 固定为 1；Stage 2 使用加权随机（2-50，线程数越大概率越低） |

**多线程阶段加权分布**：
- 线程数 2：权重最高（概率最大）
- 线程数 25：中等权重
- 线程数 50：权重最低（概率最小）

### 3. HBM 内存分配与对齐支持

矩阵缓冲区支持两种分配模式，通过编译宏 `-DUSE_HBM` 切换：

#### 普通模式（默认）
- 使用 `posix_memalign` 分配内存
- **64 字节对齐**，满足 SIMD 指令要求
- 适用于 x86/ARM 平台

#### HBM 模式 (`-DUSE_HBM`)
- 使用 `HBMAlloc(size, cpuid)` 在 HBM 上分配内存
- 根据当前 CPU ID 选择 NUMA 节点，优化内存访问延迟
- 使用 `HBMFree()` 释放内存

### 4. 多线程测试模型

测试采用**两层嵌套并行**架构：

- **外层**：`std::thread` worker 线程并发执行测试迭代
- **内层**：OpenMP 线程在单个 GEMM 操作内部并行

**两阶段自动测试**（默认模式）：
- 自动检测 CPU 核心数（默认 8 个 worker，可通过 `UNIGEMM_MAX_WORKERS` 覆盖）
- **Stage 1**：单线程测试（BLAS threads = 1）
- **Stage 2**：多线程测试（BLAS threads 随机 2-50，加权分布）
- 每个阶段运行总迭代次数的 50%

**手动配置模式**：
- 使用 `--thread` 和 `--blas-threads` 手动指定配置
- 适用于特定测试场景或性能调优

**其他特性**：
- 线程隔离：每个线程使用独立的 RNG 种子，无共享可变状态
- 内存管理：每个线程独立分配 4 个缓冲区（A、B、C_impl、C_ref）
- 统计计数：使用原子计数器统计 total/passed/failed
- 失败日志：使用互斥锁保护的失败日志（最多记录 20 个失败的详细信息）
- 进度显示：实时显示 S/M/L（小/中/大矩阵）测试进度

### 5. 正确性比对

- **相对误差容差**：`|a-b| < 1e-3 * max(|a|,|b|,1.0)`
- **失败报告**：打印完整参数配置和首个不匹配位置的详细信息
- **输出格式**：最终汇总统计 + 前 20 个失败的详细诊断

## 配置选项

### 维度范围配置

配置位于 `fuzz_test_config.h`：

```cpp
/* 维度范围定义 */
constexpr int DIM_RANGE_SMALL = 128;      // 小维度范围
constexpr int DIM_RANGE_MEDIUM = 512;     // 中维度范围
constexpr int DIM_RANGE_LARGE = 1024;     // 大维度范围

/* 维度分布概率 (总和建议为 100) */
constexpr int DIM_PROB_SMALL = 40;        // 0-128 范围概率 (%)
constexpr int DIM_PROB_MEDIUM = 40;       // 0-512 范围概率 (%)
constexpr int DIM_PROB_LARGE = 20;        // 0-1024 范围概率 (%)
```

### 线程配置

配置位于 `fuzz_test_config.h`：

```cpp
/* 线程配置 */
#ifdef USE_HBM
#define MAX_WORKERS 100  // HBM 模式默认 100 worker
#else
#define MAX_WORKERS 8    // 非 HBM 模式默认 8 worker
#endif

/* BLAS 线程数范围配置 (用于多线程阶段) */
#define MAX_BLAS_THREADS 50  // 多线程阶段最大 BLAS 线程数
```

**配置说明**：
- `MAX_WORKERS`：控制外层 `std::thread` 的数量
  - 不设置则运行时自动检测 CPU 核数
  - **HBM 模式**：默认 100 worker
  - 可通过 `UNIGEMM_MAX_WORKERS` 环境变量运行时覆盖

**两阶段测试模式**（默认）：
- **Stage 1**：单线程（BLAS threads = 1）
- **Stage 2**：多线程（BLAS threads 随机 2-`MAX_BLAS_THREADS`，加权分布）
  - 线程数越大，随机到的概率越低（线性衰减）

修改这些常量后重新编译即可生效。

### OpenMP 环境变量配置

Stage 2（多线程阶段）会创建大量 OpenMP 线程（随机 2-50）。如果遇到 OMP Error #34（资源不可用），可以通过环境变量限制 OpenMP 资源：

```bash
# 限制 OpenMP 同时执行的最大线程数
export OMP_THREAD_LIMIT=256

# 减少每线程栈大小（默认 4M）
export OMP_STACKSIZE=2M

# 使用被动等待策略（减少资源占用）
export OMP_WAIT_POLICY=passive

# 减少线程保持活跃的时间
export KMP_BLOCKTIME=50ms
```

完整运行示例：
```bash
export OMP_THREAD_LIMIT=256 OMP_STACKSIZE=2M
./out/fuzz_test --iteration 5000
```

**关键环境变量**：

| 环境变量 | 作用 | 推荐值 |
|----------|------|--------|
| `OMP_THREAD_LIMIT` | 同时执行的最大线程数 | 128-512 |
| `OMP_STACKSIZE` | 每线程栈大小 | 2M-4M |
| `OMP_WAIT_POLICY` | 等待策略（active/passive） | passive |
| `KMP_BLOCKTIME` | 线程保持活跃时间 | 50ms |

### 运行时覆盖配置

可以通过 `UNIGEMM_MAX_WORKERS` 环境变量在运行时覆盖 worker 数量：

```bash
# 临时限制到 80 worker（无需重新编译）
UNIGEMM_MAX_WORKERS=80 ./out/fuzz_test --iteration 500

# 300 核系统想跑满，确保不设置此变量即可
```

优先级：`UNIGEMM_MAX_WORKERS` > `MAX_WORKERS` 编译配置 > CPU 核数自动检测

## 注意事项

1. **线程控制**：
   - `MAX_WORKERS` 控制外层 std::thread 数量
   - 默认运行时自动检测 CPU 核数，可通过 `UNIGEMM_MAX_WORKERS` 覆盖
   - Stage 1（单线程）：只创建 std::thread workers，不创建 OpenMP 线程
   - Stage 2（多线程）：每个测试随机创建 2-50 个 OpenMP 线程
   - 如果 Stage 2 触发 OMP Error #34，可通过 `OMP_THREAD_LIMIT` 限制 OpenMP 资源

2. **测试覆盖**：当前测试维度配置（在 `fuzz_test_config.h` 中可调）：
   - 40% 概率：0-128（小维度）
   - 40% 概率：0-512（中维度）
   - 20% 概率：0-1024（大维度）
   - 可通过修改 `DIM_RANGE_*` 和 `DIM_PROB_*` 常量调整

3. **运行时间**：
   - 默认配置（100 次迭代）通常在 1 秒内完成
   - 总迭代次数平均分配给两个阶段
   - 增加迭代次数会线性增加运行时间

4. **内存使用**：
   - 每个线程约 4MB（4 个缓冲区 × 1MB）
   - 多阶段测试会为每个阶段独立分配和释放内存
   - 缓冲区对齐到 64 字节边界

5. **优化一致性**：编译使用 `-O2` 优化，与生产构建保持一致，有助于发现优化相关的 bug

6. **HBM 模式**：使用 `-DUSE_HBM` 编译时，需要链接提供 `HBMAlloc`/`HBMFree` 函数的库

## 扩展方向

如需扩展测试功能，可以考虑：

- 添加对双精度 (`cblas_dgemm`) 的支持
- 支持更大的矩阵维度
- 添加对特殊边界条件的测试（如 NaN、Infinity）
- 支持随机种子固定以实现可复现的测试
