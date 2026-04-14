#!/bin/bash
# Fuzz test build script using CMake

set -e

# 查找 cmake 命令
if command -v cmake &> /dev/null; then
    CMAKE_CMD="cmake"
elif [ -f "/opt/homebrew/bin/cmake" ]; then
    CMAKE_CMD="/opt/homebrew/bin/cmake"
elif [ -f "/usr/local/bin/cmake" ]; then
    CMAKE_CMD="/usr/local/bin/cmake"
else
    echo "Error: cmake not found!"
    echo "Please install cmake or add it to your PATH."
    echo ""
    echo "Installation options:"
    echo "  - macOS: brew install cmake"
    echo "  - Linux: sudo apt install cmake  (Ubuntu/Debian)"
    echo "           sudo yum install cmake  (CentOS/RHEL)"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Fuzz Test Build Script ==="

# 解析命令行参数
BUILD_TYPE="Release"
CLEAN=0
RUN=0
ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=1
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --run)
            RUN=1
            shift
            ;;
        --thread|--iteration)
            if [ -n "$2" ] && [[ ! "$2" =~ ^- ]]; then
                ARGS="$ARGS $1 $2"
                shift 2
            else
                echo "Error: $1 requires an argument"
                exit 1
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean         Clean build directory before building"
            echo "  --debug         Build with debug symbols"
            echo "  --run           Run the test after building"
            echo "  --thread N      Number of threads (passed to fuzz test)"
            echo "  --iteration N   Total iterations (passed to fuzz test)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Build only"
            echo "  $0 --run                  # Build and run with defaults"
            echo "  $0 --clean --run          # Clean, build and run"
            echo "  $0 --run --thread 4 --iteration 100  # Custom args"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# 清理
if [ $CLEAN -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf build out
fi

# 创建 build 目录
mkdir -p build
cd build

# 配置 CMake
echo "Configuring CMake ($BUILD_TYPE)..."
$CMAKE_CMD -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

# 构建
echo "Building..."
$CMAKE_CMD --build . --config $BUILD_TYPE

cd ..

# 运行测试
if [ $RUN -eq 1 ]; then
    echo ""
    echo "Running fuzz test..."
    echo ""
    ./out/fuzz_test $ARGS
fi

echo ""
echo "Build complete! Executable: ./out/fuzz_test"
