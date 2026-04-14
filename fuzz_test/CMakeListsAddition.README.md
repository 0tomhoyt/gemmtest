# CMakeListsAddition.txt 使用说明

## 简介
`CMakeListsAddition.txt` 提供独立的外部静态库链接配置，用于测试预编译的 BLAS 实现。

## 快速开始

### 在另一台环境的目录结构
```
some_project/
├── fuzz_test/
│   ├── CMakeLists.txt           # 原始配置（不修改）
│   ├── CMakeListsAddition.txt    # 外部库配置
│   ├── build.sh
│   └── ...
├── libopenblas_ref.a             # 参考实现库
└── unigemm_920f_static.a        # 被测实现库
```

### 方法 1：在 CMakeLists.txt 末尾添加（推荐）

在 `fuzz_test/CMakeLists.txt` 的**最后**添加：

```cmake
# 支持外部静态库
option(USE_EXTERNAL_LIBS "Link external static libraries" OFF)

if(USE_EXTERNAL_LIBS)
    include(CMakeListsAddition.txt)
endif()
```

然后在 CMakeLists.txt 中找到 `target_link_libraries` 部分，添加：

```cmake
# 链接 pthread 库
target_link_libraries(fuzz_test PRIVATE pthread)

# 添加外部库链接
if(USE_EXTERNAL_LIBS)
    target_link_libraries(fuzz_test PRIVATE "${OPENBLAS_REF_PATH}" "${UNIGEMM_LIB_PATH}")
endif()
```

### 构建

```bash
cd fuzz_test/build

# 使用源文件编译（默认）
cmake -DUSE_EXTERNAL_LIBS=OFF ..
make

# 使用外部静态库
cmake -DUSE_EXTERNAL_LIBS=ON ..
make

# 自定义库名称
cmake -DUSE_EXTERNAL_LIBS=ON -DUNIGEMM_LIB_NAM=custom_name ..

# 运行测试
./out/fuzz_test -t 4 -n 1000
```

## 方法 2：命令行传递配置

```bash
# 临时使用，不修改 CMakeLists.txt
cmake -DUSE_EXTERNAL_LIBS=ON -DUNIGEMM_LIB_NAM=mylib ..
```

## 环境变量

| 变量 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `UNIGEMM_LIB_NAM` | 库名称前缀 | `unigemm_920f` | `mylib` |
| `UNIGEMM_DIR` | 库文件所在目录 | `..` | `/opt/libs` |

## 外部库要求

### 必需符号（使用 `extern "C"` 导出）

**参考库** `libopenblas_ref.a`：
```c
void cblas_sgemm_ref(...);
void BlasSetNumThreadsLocal(int);
```

**被测库** `${UNIGEMM_LIB_NAM}_static.a`：
```c
void cblas_sgemm(...);
void BlasSetNumThreadsLocal(int);
```

### 库文件搜索路径

按优先级搜索：
1. `../libopenblas_ref.a`
2. `./libopenblas_ref.a`
3. `${UNIGEMM_DIR}/libopenblas_ref.a`
4. `${UNIGEMM_DIR}/openblas_ref.a`

对于被测库，搜索 `${UNIGEMM_LIB_NAME}.a` 和 `lib${UNIGEMM_LIB_NAME}.a`

## 常见问题

### Q: CMake 提示找不到库文件
**A**: 使用 `UNIGEMM_DIR` 指定路径：
```bash
cmake -DUSE_EXTERNAL_LIBS=ON -DUNIGEMM_DIR=/path/to/libs ..
```

### Q: 链接时符号未定义
**A**: 检查库是否正确导出符号：
```bash
nm libopenblas_ref.a | grep cblas_sgemm_ref
nm unigemm_920f_static.a | grep cblas_sgemm
```

### Q: 如何测试不同的库版本？
**A**: 替换库文件后重新构建：
```bash
# 复制新版本
cp /path/to/new_version.a ./unigemm_920f_static.a

# 重新构建
cd build
cmake -DUSE_EXTERNAL_LIBS=ON ..
make
./out/fuzz_test
```

## 完整示例

```bash
# 1. 准备库文件
cd /path/to/project
ls libopenblas_ref.a unigemm_920f_static.a

# 2. 修改 CMakeLists.txt（一次性）
cd fuzz_test
# 在文件末尾添加 include(CMakeListsAddition.txt) 和相关链接代码

# 3. 构建
mkdir build && cd build
cmake -DUSE_EXTERNAL_LIBS=ON ..
make

# 4. 运行测试
./out/fuzz_test -t 4 -n 100
```
