#!/bin/bash
set -e

if command -v cmake &> /dev/null; then
    CMAKE_CMD="cmake"
elif [ -f "/opt/homebrew/bin/cmake" ]; then
    CMAKE_CMD="/opt/homebrew/bin/cmake"
elif [ -f "/usr/local/bin/cmake" ]; then
    CMAKE_CMD="/usr/local/bin/cmake"
else
    echo "Error: cmake not found!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== System Test Build Script ==="

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
        --csv|--random)
            ARGS="$ARGS $1"
            shift
            ;;
        --precision|--iteration|--seed)
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
            echo "Build Options:"
            echo "  --clean           Clean build directory before building"
            echo "  --debug           Build with debug symbols"
            echo "  --run             Run the test after building"
            echo ""
            echo "Test Modes (default: full = csv + random):"
            echo "  --csv             Run embedded CSV test cases only"
            echo "  --random          Run random 30-stage test only"
            echo ""
            echo "Test Options:"
            echo "  --precision NAME  Single precision: sgemm, shgemm, sbgemm, hgemm, bgemm"
            echo "  --iteration N     Iterations per precision (default: 100)"
            echo "  --seed N          Seed for CSV matrix generation (default: 42)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ $CLEAN -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf build out
fi

mkdir -p build
cd build

echo "Configuring CMake ($BUILD_TYPE)..."
$CMAKE_CMD -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

echo "Building..."
$CMAKE_CMD --build . --config $BUILD_TYPE

cd ..

if [ $RUN -eq 1 ]; then
    echo ""
    echo "Running system test..."
    echo ""
    ./out/st $ARGS
fi

echo ""
echo "Build complete! Executable: ./out/st"
