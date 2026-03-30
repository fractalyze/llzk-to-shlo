# LLZK-to-StableHLO 변환 현황

## 수치

- **변환 성공**: 47/47 (BabyBear 100%, BN254 100%)
- **GPU E2E**: 46/47 (ZKX runner), 47/47 (open-zkx + FusionWrapper)
- **LIT 테스트**: 12/12 통과
- **GPU semantic**: 8/8 통과

## Milestone 2 Deliverables

### 1. LLZK → StableHLO lowering passes ✅

47/47 circom 회로 변환 (BabyBear + BN254). SimplifySubComponents + LlzkToStablehlo
full pipeline.

### 2. StableHLO-based witness generation in ZKX pipeline ✅

E2E pipeline: `circom → LLZK → StableHLO → GPU witness generation`

- ZKX runner: 46/47 E2E
- open-zkx runner: 47/47 E2E (FusionWrapper + negate/power/convert 수정)

### 3. GPU execution optimization ✅

**Auto-vectorization pass** (`vectorizeIndependentWhileLoops`):
iteration-independent while loops를 감지하여 element-wise vectorized ops로 변환.

패턴: `while(i < N) { out[i] = f(in[i]); i++ }` → `out = f(in)`

**벤치마크** (BabyBear, RTX 5090):

| N      | GPU sequential | GPU vectorized | Speedup |
| ------ | -------------- | -------------- | ------- |
| 1,024  | 997μs          | 48μs           | 21x     |
| 16,384 | 14.9ms         | 55μs           | 271x    |

**GPU vs CPU** (BabyBear, N independent multiplications):

| N         | GPU (vectorized) | CPU (C++ -O2) | GPU/CPU  |
| --------- | ---------------- | ------------- | -------- |
| 262,144   | 106μs            | 159μs         | 1.5x GPU |
| 1,048,576 | 171μs            | 648μs         | 3.8x GPU |

## 아키텍처

```
SimplifySubComponents (LLZK IR → LLZK IR, pod dispatch 제거)
    ↓ runPipeline으로 내부 실행
LlzkToStablehlo (LLZK IR → StableHLO)
    ├── promoteArraysToWhileCarry
    ├── convertWhileBodyArgsToSSA
    ├── convertWritemToSSA
    ├── applyPartialConversion
    └── Post-passes:
        ├── convertScfWhileToStablehloWhile
        ├── scf.if → stablehlo.select
        ├── reconnect func.call to pod.read @comp
        ├── LLZK cleanup (residual pod/array → zero tensor)
        ├── arith.ori/andi → stablehlo.or/and
        ├── dead code cleanup
        └── vectorizeIndependentWhileLoops
```

## 주요 코드 파일

| 파일                        | 역할                                                      |
| --------------------------- | --------------------------------------------------------- |
| `SimplifySubComponents.cpp` | pod dispatch 제거 (flattenPod, eliminatePod, extractCall) |
| `LlzkToStablehlo.cpp`       | Main conversion + post-passes + vectorization             |
| `TypeConversion.cpp/.h`     | 타입 변환 + getPodInitializedRecords                      |
| `ArrayPatterns.cpp`         | array.read/write/extract/insert → stablehlo ops           |
| `FeltPatterns.cpp`          | felt.add/mul/shr/bit_and/pow → stablehlo ops              |
| `FunctionPatterns.cpp`      | function.call → func.call                                 |

## 알려진 제약사항

1. `getPodInitializedRecords` 문자열 파싱 workaround (fractalyze/llzk-to-shlo#2)
1. `hasArrayOfPods` guard (nested eliminatePodDispatch regression)
1. constrain function body clearing crash (sub-component call chain)
1. 2D 배열 circuit (e.g. `signal chain[N][K]`) 변환 미지원 — element당 compute-heavy
   circuit (MiMC batch 등)의 GPU E2E에 필요
1. Vectorization은 단일 element-wise op 패턴만 지원 — func.call chain, 다중 output array
   패턴은 미구현

## 관련 PR / Issue

- fractalyze/stablehlo#40 — i128/i256 integer type widths
- fractalyze/open-zkx#1 — GPU fusion for mixed babybear/s32 ops
- fractalyze/llzk-to-shlo#2 — getPodInitializedRecords API
- fractalyze/prime-ir#285 — InverseOp zero convention
