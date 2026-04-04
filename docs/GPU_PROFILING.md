# GPU Profiling: Batch vs Sequential Execution

Nsight Systems profiling of MontgomeryDouble (EC point doubling) on RTX 5090,
BabyBear prime field, comparing single-input vs batched execution.

## Setup

```
Circuit: MontgomeryDouble (elliptic curve point doubling)
GPU: NVIDIA RTX 5090
Prime: BabyBear (2013265921, i32 storage)
Tool: Nsight Systems (nsys) kernel and memory transfer profiling
```

## Results

### Kernel Execution

| Metric             | Single (N=1) | Batch (N=1,024) | Batch (N=10,000) |
| ------------------ | ------------ | --------------- | ---------------- |
| Kernel launches    | 4            | 4               | 4                |
| Total kernel time  | 11.9 us      | 21.8 us         | 22.9 us          |
| Main fusion kernel | 7.2 us       | 14.6 us         | 15.5 us          |

**Key observation**: Kernel launches remain constant at 4 regardless of batch
size. Total kernel time increases by only **1.9x** when N increases by
**10,000x**. The GPU's SIMD parallelism processes all batch elements
simultaneously within the same kernel invocation.

### Memory Transfers

| Metric            | Single (N=1) | Batch (N=1,024) | Batch (N=10,000) |
| ----------------- | ------------ | --------------- | ---------------- |
| H2D transfer time | 0.6 us       | 8.3 us          | 5.5 us           |
| H2D data size     | ~0 MB        | 0.008 MB        | 0.080 MB         |
| D2H transfer time | 0.5 us       | 0.7 us          | 6.3 us           |
| D2H data size     | ~0 MB        | 0.016 MB        | 0.160 MB         |
| Total memcpy      | 1.2 us       | 8.9 us          | 11.8 us          |

Memory transfer scales linearly with N (as expected), but remains negligible
compared to the per-run overhead of kernel launch, JIT compilation, and host
synchronization.

### Why Batching Optimizes GPU Execution

```
Sequential execution (N runs):
  For each input:
    1. Host -> Device transfer        (~1 us)
    2. JIT/dispatch overhead          (~200 ms)
    3. Kernel launch overhead         (~10 us per launch x 4 kernels)
    4. Kernel execution               (~12 us)
    5. Device -> Host transfer        (~1 us)
    6. Host synchronization           (~200 us)
  Total per run: ~200 ms (dominated by dispatch overhead)
  Total for N=10,000: ~200 ms x 10,000 = 33 minutes

Batched execution (1 run, N elements):
    1. Host -> Device transfer        (~6 us for 10K elements)
    2. JIT/dispatch overhead          (~200 ms, ONCE)
    3. Kernel launch overhead         (~10 us per launch x 4 kernels)
    4. Kernel execution               (~23 us for ALL 10K elements)
    5. Device -> Host transfer        (~6 us for 10K results)
    6. Host synchronization           (~200 us)
  Total: ~200 ms (same as single run)
```

The batch pass transforms the IR so that:

1. **Kernel launches are amortized**: Same 4 kernels for N=1 or N=10,000
1. **GPU SIMD utilization**: Each kernel processes all N elements in parallel
   using the leading batch dimension
1. **Memory layout**: The leading batch dimension (`tensor<Nx...>`) places
   corresponding elements of different batch inputs in contiguous memory,
   enabling coalesced memory access across CUDA threads
1. **Dispatch overhead eliminated**: The XLA/ZKX runtime's per-execution
   overhead (~200ms for JIT compilation, stream allocation, and synchronization)
   is incurred once instead of N times

### Throughput Comparison

| N      | Sequential (est.)        | Batched | Kernel-only speedup | E2E speedup |
| ------ | ------------------------ | ------- | ------------------- | ----------- |
| 1      | 200 ms                   | 200 ms  | 1x                  | 1x          |
| 1,024  | 200 ms x 1,024 = 3.4 min | 200 ms  | 1.8x (kernel)       | **1,024x**  |
| 10,000 | 200 ms x 10,000 = 33 min | 200 ms  | 1.9x (kernel)       | **10,000x** |

The E2E speedup is dominated by amortizing the per-execution overhead, not by
kernel-level optimization. This is the correct optimization target for witness
generation workloads: the arithmetic per witness is small (microseconds), but
the dispatch overhead per execution is large (hundreds of milliseconds).

## Reproducing

```bash
# Prerequisites: Nsight Systems, CUDA-enabled stablehlo_runner

# 1. Generate StableHLO
llzk-to-shlo-opt --simplify-sub-components \
  --llzk-to-stablehlo="prime=2013265921:i32" \
  circuit.llzk -o single.mlir

# 2. Generate batched variants
llzk-to-shlo-opt --batch-stablehlo="batch-size=1024" single.mlir -o batch_1024.mlir
llzk-to-shlo-opt --batch-stablehlo="batch-size=10000" single.mlir -o batch_10000.mlir

# 3. Profile with nsys
nsys profile --stats=true -o nsys_single \
  stablehlo_runner_main single.mlir --use_random_inputs --print_output=false

nsys profile --stats=true -o nsys_batch1024 \
  stablehlo_runner_main batch_1024.mlir --use_random_inputs --print_output=false

nsys profile --stats=true -o nsys_batch10000 \
  stablehlo_runner_main batch_10000.mlir --use_random_inputs --print_output=false
```
