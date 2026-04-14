---
name: validate
description: Validate and test the UniGEMM project (MacBook environment)
scope: [/Users/tomhoyt/Desktop/firm_project/unigemm_test]
---

# UniGEMM Project Validation (MacBook)

This skill runs validation tests for the UniGEMM SGEMM testing framework on the current MacBook environment.

## Validation Commands

### Quick Validation

Fast validation for development:

```bash
cd fuzz_test
./build.sh --clean --run -t 4 -n 5000
```

### Full Validation

Comprehensive testing with more iterations:

```bash
cd fuzz_test
./build.sh --clean --run -t 8 -n 50000
```

### Debug Validation

Debug mode with fewer iterations:

```bash
cd fuzz_test
./build.sh --clean --debug --run -t 1 -n 100
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
  Iterations per thread: 5000
  Total iterations: 20000

========================================
Results:
  Total:   20000
  Passed:  20000
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

- Thread count (`-t`) should not exceed CPU core count
- Iteration count (`-n`) scales linearly with test time
- Use `--debug` build option when investigating failures
- HBM mode (`-DUSE_HBM`) is for ARM64 server only, do not run on MacBook
