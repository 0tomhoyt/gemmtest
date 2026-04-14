# SGEMM 模糊测试工具

## 项目概述

这是一个多线程随机测试工具，用于全面测试 `cblas_sgemm` 函数实现的正确性。由于系统中没有参考 BLAS 库，本项目通过独立编写的参考实现进行正确性比对。

## 如何使用

### 方式 1: 使用便捷构建脚本（推荐）

```bash
cd fuzz_test

# 仅构建（普通模式）
./build.sh

# 清理构建并运行
./build.sh --clean --run

# 传递参数给测试程序
./build.sh --run --thread 4 --iteration 100

# Debug 模式构建
./build.sh --debug

# 查看帮助
./build.sh --help
```

### 方式 2: 使用 CMake

```bash
cd fuzz_test

# 普通模式配置和构建
mkdir build && cd build
cmake ..
cmake --build .

# HBM 模式编译（需要 HBM 环境）
mkdir build && cd build
cmake -DUSE_HBM=ON ..
cmake --build .

# 运行
./out/fuzz_test
```

### 运行参数

```bash
# 使用默认参数运行（4 线程，总 100 次迭代）
./out/fuzz_test

# 自定义线程数
./out/fuzz_test --thread 8

# 自定义总迭代次数
./out/fuzz_test --iteration 5000

# 同时指定线程数和迭代次数
./out/fuzz_test --thread 8 --iteration 20000

# 查看帮助信息
./out/fuzz_test -h
```

### 测试错误检测功能

验证 fuzz_test 能否正确检测到实现中的错误：

```bash
cd fuzz_test

# 使用故意有错误的实现进行测试
./test_buggy.sh

# 或手动执行
mkdir build && cd build
cmake -DUSE_BUGGY_IMPL=ON ..
cmake --build .
cd ..
./out/fuzz_test --thread 2 --iteration 10
# 预期输出应该包含 Failed: > 0
```

### 清理

```bash
rm -rf build out
```

## 输出示例

### 成功输出

```
Starting fuzz test:
  Threads: 4
  Total iterations: 100
  Iterations per thread: 25

========================================
Results:
  Total:   100
  Passed:  100
  Failed:  0
  Error rate: 0%
========================================
```

### 失败输出（当检测到错误时）

```
========================================
Results:
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
├── fuzz_test_config.h       # 配置常量（维度范围、对齐要求）
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
| `transA/transB` | 4 个枚举值均匀随机分布 |
| `m, n, k` | 10% 概率 0-64（特殊维度），40% 概率 0-128，50% 概率 0-512 |
| `alpha, beta` | 70% 概率选择特殊值 {0,±1,±2,±0.5,±0.25}，30% 概率选择 [-10,10] 随机浮点数 |
| `lda/ldb/ldc` | 最小值 + 0-7 的随机 padding |

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

- **命令行参数**：可配置线程数 (`--thread`) 和总迭代次数 (`--iteration`)
- **线程隔离**：每个线程使用独立的 RNG 种子，无共享可变状态
- **内存管理**：每个线程独立分配 4 个缓冲区（A、B、C_impl、C_ref）
- **统计计数**：使用原子计数器统计 total/passed/failed
- **失败日志**：使用互斥锁保护的失败日志（最多记录 20 个失败的详细信息）

### 5. 正确性比对

- **相对误差容差**：`|a-b| < 1e-3 * max(|a|,|b|,1.0)`
- **失败报告**：打印完整参数配置和首个不匹配位置的详细信息
- **输出格式**：最终汇总统计 + 前 20 个失败的详细诊断

## 注意事项

1. **测试覆盖**：当前测试维度配置（在 [fuzz_test_config.h](fuzz_test_config.h) 中可调）：
   - 10% 概率：0-64（特殊边界维度）
   - 40% 概率：0-128
   - 50% 概率：0-512
   - 可通过修改 `DIM_RANGE_*` 和 `DIM_PROB_*` 常量调整

2. **运行时间**：默认配置（40000 次迭代）通常在几十秒内完成。增加迭代次数会线性增加运行时间

3. **内存使用**：
   - 每个线程约 4MB（4 个缓冲区 × 1MB）
   - 4 个线程总计约 16MB
   - 缓冲区对齐到 64 字节边界

4. **优化一致性**：编译使用 `-O2` 优化，与生产构建保持一致，有助于发现优化相关的 bug

5. **HBM 模式**：使用 `-DUSE_HBM` 编译时，需要链接提供 `HBMAlloc`/`HBMFree` 函数的库

## 配置维度范围

维度测试范围配置位于 [fuzz_test_config.h](fuzz_test_config.h)：

```cpp
/* 维度范围定义 */
constexpr int DIM_RANGE_SMALL = 64;      // 小维度范围
constexpr int DIM_RANGE_MEDIUM = 128;    // 中维度范围
constexpr int DIM_RANGE_LARGE = 512;     // 大维度范围

/* 维度分布概率 (总和建议为 100) */
constexpr int DIM_PROB_SMALL = 10;       // 0-64 范围概率 (%)
constexpr int DIM_PROB_MEDIUM = 40;      // 0-512 范围概率 (%)
constexpr int DIM_PROB_LARGE = 50;       // 0-1024 范围概率 (%)

/* 缓冲区对齐要求 (字节) */
constexpr int BUFFER_ALIGNMENT = 64;
```

修改这些常量后重新编译即可生效。

## 扩展

如需扩展测试功能，可以考虑：

- 添加对双精度 (`cblas_dgemm`) 的支持
- 支持更大的矩阵维度
- 添加对特殊边界条件的测试（如 NaN、Infinity）
- 支持随机种子固定以实现可复现的测试
