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

3단계 자동 vectorization pass:

- **Phase 1**: 1D independent while → element-wise ops
- **Phase 1.5**: 2D carry while → column write/read vectorization
- **Phase 2**: nested while inner loop → 1D carry vectorization

* **pub-only output trimming**: non-pub struct 멤버를 output에서 제거

**벤치마크** (BabyBear, RTX 5090, circom → GPU E2E):

| N       | K (muls/elem) | GPU (auto-vectorized) | CPU (C++ -O2) | Speedup      |
| ------- | ------------- | --------------------- | ------------- | ------------ |
| 65,536  | 32            | 439μs                 | 1,315μs       | **3.0x GPU** |
| 262,144 | 32            | 4.1ms                 | 5.0ms         | **1.2x GPU** |

**Sequential → Vectorized 개선** (같은 GPU에서):

| N      | K   | Sequential | Vectorized | Improvement |
| ------ | --- | ---------- | ---------- | ----------- |
| 65,536 | 32  | 5.2s       | 439μs      | **11,800x** |

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
        ├── vectorizeIndependentWhileLoops (Phase 1/1.5/2)
        └── pub-only output trimming
```

## 알려진 제약사항

1. `getPodInitializedRecords` 문자열 파싱 workaround (fractalyze/llzk-to-shlo#2)
1. `hasArrayOfPods` guard (nested eliminatePodDispatch regression)
1. constrain function body clearing crash (sub-component call chain)
1. 단일 outer loop circom 스타일 (`for i: chain[i][0]=...; for j: chain[i][j]=...`)은
   vectorize 가능. 합쳐진 스타일 (`for i: { chain[i][0]=...; for j: chain[i][j]=... }`)은
   미지원
1. dynamic_slice 인덱스가 loop counter 대신 constant 0을 사용하는 버그 (기존)

## 관련 PR / Issue

- fractalyze/stablehlo#40 — i128/i256 integer type widths
- fractalyze/open-zkx#1 — GPU fusion for mixed babybear/s32 ops
- fractalyze/llzk-to-shlo#2 — getPodInitializedRecords API
- fractalyze/prime-ir#285 — InverseOp zero convention
