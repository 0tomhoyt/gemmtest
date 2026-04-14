---
name: validate
description: Validate and test the UniGEMM project (MacBook environment)
scope: [/Users/tomhoyt/Desktop/firm_project/unigemm_test]
tools: Bash
---

# UniGEMM Project Validation (MacBook)

This skill runs validation tests for the UniGEMM SGEMM testing framework on the current MacBook environment.

## Validation Commands

### Quick Validation

Fast validation for development:

```bash
cd fuzz_test
./build.sh --clean --run --thread 4 --iteration 5000
```

### Full Validation

Comprehensive testing with more iterations:

```bash
cd fuzz_test
./build.sh --clean --run --thread 8 --iteration 50000
```

### Debug Validation

Debug mode with fewer iterations:

```bash
cd fuzz_test
./build.sh --clean --debug --run --thread 1 --iteration 100
```

### Error Detection Validation

Verify that the test framework can detect bugs:

```bash
cd fuzz_test
./test_buggy.sh
```

Expected output: Should report failures (this confirms error detection works).

## Validation Checklist

When validating, check for:

1. **Build Success**: No compilation errors or warnings
2. **Test Completion**: All iterations complete without crashes
3. **Error Rate**: Should be 0% for correct implementation
4. **Memory Alignment**: Verify 64-byte alignment assertions pass
5. **Multi-threading**: Confirm thread-safe operation (default: 4 threads)

## Expected Output

### Success Output

```
Starting fuzz test:
  Threads: 4
  Total iterations: 5000
  Iterations per thread: 1250

========================================
Results:
  Total:   5000
  Passed:  5000
  Failed:  0
  Error rate: 0%
========================================
```

### Failure Output (Indicates Implementation Bug)

```
Results:
  Total:   20000
  Passed:  19998
  Failed:  2
  Error rate: 0.01%
```

## Notes

- Thread count (`--thread`) should not exceed CPU core count
- Iteration count (`--iteration`) is total across all threads
- Use `--debug` build option when investigating failures
- HBM mode (`-DUSE_HBM`) is for ARM64 server only, do not run on MacBook
