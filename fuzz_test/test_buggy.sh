#!/bin/bash
# Test script to verify fuzz_test can detect errors in buggy implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Testing Fuzz Test Error Detection"
echo "=========================================="
echo ""
echo "This will compile and test an intentionally buggy SGEMM implementation."
echo "If fuzz_test works correctly, it should report failures."
echo ""

# Clean build
echo "1. Cleaning build directory..."
rm -rf build out

# Create build directory and configure with buggy implementation
echo ""
echo "2. Configuring CMake with buggy implementation..."
mkdir -p build
cd build
/opt/homebrew/bin/cmake -DUSE_BUGGY_IMPL=ON ..
cd ..

# Build
echo ""
echo "3. Building..."
/opt/homebrew/bin/cmake --build build --config Release

echo ""
echo "4. Running fuzz_test with buggy implementation..."
echo "   Expected: Should detect errors and report failures"
echo ""

# Run test with moderate iterations to catch bugs quickly
./out/fuzz_test --thread 4 --iteration 100

echo ""
echo "=========================================="
echo "If you see failures above, fuzz_test is working correctly!"
echo "=========================================="
