# GEMM 测试框架

## 项目概述

这是一个为**两种不同环境**设计的 GEMM（通用矩阵乘法）数值正确性测试框架：

| 环境 | 用途 | 内存分配 | 平台 |
|------|------|----------|------|
| **MacBook** | 开发调试、快速验证 | `posix_memalign` | macOS (x86/ARM64) |
| **ARM64 服务器** | 生产环境、HBM 测试 | `AllocateMemory<T>` (HBM) | Linux + NUMA |

同一套代码通过编译宏 `-DUSE_HBM` 在两种环境间切换，无需维护两套代码库。

**支持的精度类型**：
- **SGEMM**：FP32 单精度
- **SHGEMM**：FP16 半精度
- **SBGEMM**：BF16 bfloat16

## 项目结构

```
unigemm_test/
├── include/
│   ├── unigemm_920f.h       # BLAS 类型定义 + cblas_*gemm 声明
│   ├── gemm_benchmark.h     # Benchmark 配置/结果结构
│   ├── test_util.h          # ARM64 服务器 HBM 分配工具
│   └── ref_test_util.h      # 简单矩阵初始化工具
├── fuzz_test/
│   ├── fuzz_test_*.cpp/h    # 模糊测试核心代码（双环境兼容）
│   ├── stubs/               # SHGEMM/SBGEMM 存根实现
│   ├── CMakeLists.txt       # CMake 构建配置
│   ├── build.sh             # 便捷构建脚本
│   └── README.md            # 测试工具详细文档
├── openblas.c               # SGEMM 参考实现（C）
├── openblas.h               # 参考实现头文件
├── unigemm.c                # 待测试的 SGEMM 实现
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

### ARM64 服务器环境（HBM）

用于生产级测试和性能验证：

```bash
cd fuzz_test && mkdir build && cd build
cmake -DUSE_HBM=ON .. && cmake --build .
./out/fuzz_test --thread 4 --iteration 100
```

**特点：**
- 使用 HBM（高带宽内存）进行矩阵计算
- NUMA 感知内存分配，优化访问延迟
- 与生产环境一致

### 内存分配对比

| 特性 | MacBook (默认) | ARM64 服务器 (USE_HBM) |
|------|----------------|------------------------|
| 分配函数 | `posix_memalign` | `AllocateMemory<T>(..., true)` |
| 释放函数 | `std::free` | `FreeMemory<T>(..., true)` |
| 对齐保证 | 64 字节 | 64 字节 |
| NUMA 绑定 | 否 | 是（自动选择当前 CPU 节点） |
| 依赖 | 标准库 | `libnuma` + HBM 驱动 |

## 核心组件

### 1. GEMM 实现

#### 待测试实现 (`unigemm.c`)
- OpenMP 多线程 SGEMM 实现
- 支持完整 BLAS API：布局、转置、标量系数
- 同时支持 RowMajor 和 ColMajor

#### 参考实现 (`openblas.c`)
- 完全独立编写的参考实现
- 不同代码结构确保不会共享 bug
- 使用 `get_elem`/`put_elem` 辅助函数

#### 存根实现
- `shgemm_stub.cpp`：FP16 → FP32 转换 + SGEMM reference
- `sbgemm_stub.cpp`：BF16 → FP32 转换 + SGEMM reference

**数据生成策略**：SHGEMM/SBGEMM 直接以原始精度生成数据（随机 uint16 位模式），然后扩展为 float 给 reference 使用。impl 和 ref 操作完全相同的精度值，差异只来自 GEMM 实现本身。

### 2. 模糊测试工具

`fuzz_test/` 提供完整的十八阶段测试框架：

**十八阶段结构**：
- 3 种精度（SGEMM/SHGEMM/SBGEMM）
- 3 种维度范围（Small 1-128 / Medium 1-512 / Large 1-1024）
- 2 种 BLAS 线程模式（single / multi）

**快速开始**：

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
// fuzz_test_buffer.h
#ifdef USE_HBM
    #include "test_util.h"  // 仅 HBM 模式需要
#endif

bool ThreadBuffers::allocate(BLASINT max_dim, BLASINT max_ld) {
    max_size = static_cast<size_t>(max_ld) * static_cast<size_t>(max_dim);

#ifdef USE_HBM
    // ARM64 服务器：HBM 分配
    a_buf = AllocateMemory<float>(max_size, BUFFER_ALIGNMENT, true);
#else
    // MacBook：标准 posix 分配
    size_t byte_size = max_size * sizeof(float);
    posix_memalign(reinterpret_cast<void**>(&a_buf), BUFFER_ALIGNMENT, byte_size);
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
- **转置**：NoTrans / Trans（各 50%）
- **维度**：根据阶段固定范围（Small/Medium/Large）
- **标量系数**：70% 特殊值 + 30% 随机值
- **LDA**：最小值（上限 clamp 到 MAX_LD）
- **BLAS 线程**：single 模式固定 1；multi 模式加权随机（2-50）

### 正确性验证

| 精度 | 容差 |
|------|------|
| SGEMM | 1e-2 |
| SHGEMM | 1e-1 |
| SBGEMM | 1e-1 |

**失败报告**：
- 每个失败显示 Stage 信息（编号、精度、维度、线程模式）
- 记录完整参数和最多 20 个不匹配位置
- 最终报告包含按 Stage 汇总的失败统计

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

# 3. 使用 buggy 实现测试错误检测
cd build && cmake -DUSE_BUGGY_IMPL=ON .. && make
./out/fuzz_test --iteration 100
```

### ARM64 服务器测试流程

```bash
# 1. HBM 模式构建
cd fuzz_test && mkdir build && cd build
cmake -DUSE_HBM=ON .. && cmake --build .

# 2. 十八阶段自动测试
./out/fuzz_test --iteration 5000

# 3. 手动配置特定测试
./out/fuzz_test --thread 16 --iteration 50000

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
