# GEMM 测试框架

## 项目概述

这是一个为**两种不同环境**设计的 SGEMM（单精度通用矩阵乘法）测试框架：

| 环境 | 用途 | 内存分配 | 平台 |
|------|------|----------|------|
| **MacBook** | 开发调试、快速验证 | `posix_memalign` | macOS (x86/ARM64) |
| **ARM64 服务器** | 生产环境、HBM 测试 | `AllocateMemory<T>` (HBM) | Linux + NUMA |

同一套代码通过编译宏 `-DUSE_HBM` 在两种环境间切换，无需维护两套代码库。

## 项目结构

```
unigemm_test/
├── include/
│   ├── unigemm_920f.h       # 待测试的 SGEMM 实现
│   ├── test_util.h          # ARM64 服务器 HBM 分配工具
│   └── gemm_benchmark.h     # BLAS 数据类型定义
├── fuzz_test/
│   ├── fuzz_test_*.cpp/h    # 模糊测试核心代码（双环境兼容）
│   ├── CMakeLists.txt       # CMake 构建配置
│   ├── build.sh             # 便捷构建脚本
│   └── README.md            # 测试工具详细文档
├── openblas.c               # 参考实现（C）
├── openblas.h               # 参考实现头文件
└── README.md                # 本文档
```

## 双环境设计

### MacBook 环境（默认）

用于日常开发和快速功能验证：

```bash
cd fuzz_test
./build.sh --run
```

**特点：**
- 使用 `posix_memalign` 分配 64 字节对齐内存
- 跨平台兼容（macOS/Linux，x86/ARM64）
- 快速编译迭代
- 适合开发调试和功能验证

### ARM64 服务器环境（HBM）

用于生产级测试和性能验证：

```bash
cd fuzz_test/build
cmake -DUSE_HBM=ON ..
cmake --build .
./out/fuzz_test --thread 4 --iteration 100
```

**特点：**
- 使用 HBM（高带宽内存）进行矩阵计算
- NUMA 感知内存分配，优化访问延迟
- 与生产环境一致
- 适合压力测试和性能调优

### 内存分配对比

| 特性 | MacBook (默认) | ARM64 服务器 (USE_HBM) |
|------|----------------|------------------------|
| 分配函数 | `posix_memalign` | `AllocateMemory<T>(..., true)` |
| 释放函数 | `std::free` | `FreeMemory<T>(..., true)` |
| 对齐保证 | 64 字节 | 64 字节 |
| NUMA 绑定 | 否 | 是（自动选择当前 CPU 节点） |
| 依赖 | 标准库 | `libnuma` + HBM 驱动 |

## 核心组件

### 1. SGEMM 实现

#### 待测试实现 (`include/unigemm_920f.h`)
- 目标实现，需要验证正确性
- 头文件 + 实现合并方式，便于集成
- 支持完整 BLAS API：布局、转置、标量系数

#### 参考实现 (`openblas.c`)
- 完全独立编写的参考实现
- 不同代码结构确保不会共享 bug
- 使用 `get_elem`/`put_elem` 辅助函数

### 2. 模糊测试工具

`fuzz_test/` 提供完整测试框架：

**功能：**
- 多线程并行测试（可配置）
- 随机参数生成（维度、布局、转置、系数）
- 自动正确性验证
- 双环境内存分配（编译时切换）

**快速开始：**

```bash
# MacBook
cd fuzz_test && ./build.sh --run

# ARM64 服务器
cd fuzz_test && mkdir build && cd build
cmake -DUSE_HBM=ON .. && cmake --build .
./out/fuzz_test --thread 4 --iteration 100
```

详细文档：[fuzz_test/README.md](fuzz_test/README.md)

## 代码实现

### 环境切换机制

```cpp
// fuzz_test_buffer.cpp
#ifdef USE_HBM
    #include "test_util.h"  // 仅 HBM 模式需要
#endif

bool ThreadBuffers::allocate(BLASINT max_dim, BLASINT max_ld) {
    max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

#ifdef USE_HBM
    // ARM64 服务器：HBM 分配
    a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
    b_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
    // ...
#else
    // MacBook：标准 posix 分配
    size_t byte_size = max_size * sizeof(float);
    posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, byte_size);
    // ...
#endif
}
```

### 关键设计点

1. **条件编译**：`#ifdef USE_HBM` 切换分配路径
2. **统一接口**：两种模式返回相同类型（`float*`），上层代码无感知
3. **头文件隔离**：`test_util.h` 仅在 HBM 模式下包含
4. **对齐保证**：两种模式都保证 64 字节对齐

## 技术特点

### 参数随机化

加权随机覆盖边界情况：

- **布局**：RowMajor / ColMajor（各 50%）
- **转置**：4 个枚举值均匀分布
- **维度**：40% 小维度（0-128）+ 40% 中维度（0-512）+ 20% 大维度（0-1024）
- **标量系数**：70% 特殊值 + 30% 随机值
- **LDA**：最小值 + 随机 padding
- **BLAS 线程**：Stage 1 固定为 1；Stage 2 加权随机（2-50，线程数越大概率越低）

### 正确性验证

- **容差**：`|a-b| < 1e-3 * max(|a|,|b|,1.0)`
- **失败报告**：完整参数 + 不匹配位置
- **统计汇总**：总数 / 通过 / 失败 / 错误率

### 性能优化

- **线程隔离**：独立缓冲区，无共享状态
- **内存对齐**：64 字节，满足 SIMD 要求
- **编译优化**：`-O2` 级别

## 构建要求

### MacBook（默认模式）
- C++17 编译器（Clang/GCC）
- CMake 3.10+
- OpenMP 支持

### ARM64 服务器（HBM 模式）
- GCC/Clang for Linux
- CMake 3.10+
- OpenMP 支持
- `libnuma` 库
- NUMA 感知的 HBM 驱动

## 典型工作流

### MacBook 开发流程

```bash
# 1. 快速验证
cd fuzz_test && ./build.sh --run

# 2. 调试模式
./build.sh --debug --run --thread 1 --iteration 100

# 3. 错误检测测试
./test_buggy.sh
```

### ARM64 服务器测试流程

```bash
# 1. HBM 模式构建
cd fuzz_test && mkdir build && cd build
cmake -DUSE_HBM=ON .. && cmake --build .

# 2. 两阶段自动测试（推荐）
./out/fuzz_test --iteration 50000
# Stage 1: 50% 迭代，单线程（BLAS=1）
# Stage 2: 50% 迭代，多线程（BLAS 随机 2-50，加权分布）

# 3. 手动配置特定测试
./out/fuzz_test --thread 16 --blas-threads 4 --iteration 50000

# 4. 验证通过后部署到生产环境
```

## 扩展方向

- [ ] 双精度测试（`cblas_dgemm`）
- [ ] 更大矩阵维度支持
- [ ] 特殊浮点值测试（NaN、Infinity）
- [ ] 性能基准测试集成
- [ ] CI/CD 自动化（双环境）

## 许可证

本项目用于内部测试和验证。
