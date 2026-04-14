cmake_minimum_required(VERSION 3.10)
project(fuzz_test CXX C)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# 头文件搜索路径
include_directories(${CMAKE_SOURCE_DIR}/.. ${CMAKE_SOURCE_DIR}/../include)

# 选项：使用 buggy 实现（用于测试 fuzz_test 的错误检测能力）
option(USE_BUGGY_IMPL "Use buggy implementation to test error detection" OFF)

# C++ 源文件
set(CPP_SOURCES
    fuzz_test_main.cpp
    fuzz_test_worker.cpp
    fuzz_test_globals.cpp
)

# C 源文件（BLAS 实现）
set(C_SOURCES
    ../openblas.c
)

# 根据选项选择实现
if(USE_BUGGY_IMPL)
    message(STATUS "Using BUGGY implementation for testing")
    list(APPEND C_SOURCES ../unigemm_buggy.c)
else()
    message(STATUS "Using correct implementation")
    list(APPEND C_SOURCES ../unigemm.c)
endif()

# 合并所有源文件
set(SOURCES ${CPP_SOURCES} ${C_SOURCES})

# 创建可执行文件
add_executable(fuzz_test ${SOURCES})

# 为 C 源文件设置 C 编译器和标准
set_source_files_properties(${C_SOURCES} PROPERTIES LANGUAGE C)
target_compile_features(fuzz_test PRIVATE c_std_99)

# 链接 pthread 库
target_link_libraries(fuzz_test PRIVATE pthread)

# 查找并链接 OpenMP
# 对于 macOS with Homebrew libomp
if(APPLE)
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.a)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_omp_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_CXX_FOUND TRUE)
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(fuzz_test PRIVATE ${OpenMP_omp_LIBRARY})
    # 只为 C 源文件添加 OpenMP 编译选项
    set_source_files_properties(${C_SOURCES} PROPERTIES
        COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    )
    message(STATUS "OpenMP support found")
endif()

# 安装规则（可选）
install(TARGETS fuzz_test DESTINATION bin)
