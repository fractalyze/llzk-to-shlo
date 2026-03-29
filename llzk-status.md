# LLZK-to-StableHLO 변환 현황

## 수치

- **변환 성공**: 47/47 (100%)
- **LIT 테스트**: 12/12 통과
- **GPU semantic**: 8/8 통과

## 테스트 방법

```bash
# 전체 회로 변환 테스트 (47개)
OPT=$(readlink -f bazel-bin/tools/llzk-to-shlo-opt)
for f in /tmp/llzk_e2e/*_llzk/*.llzk /tmp/llzk_e2e/*/*_llzk/*.llzk; do
  timeout 60 "$OPT" --llzk-to-stablehlo="prime=2013265921:i32" "$f" > /dev/null 2>&1
done

# LIT + GPU semantic
bazel test //tests/...

# ASan 빌드 (메모리 버그 탐지)
bazel build //tools:llzk-to-shlo-opt -c dbg --copt=-fsanitize=address --linkopt=-fsanitize=address
```

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
        ├── arith.ori/andi → stablehlo.or/and
        └── dead code cleanup
```

## 주요 코드 파일

| 파일                        | 역할                                                           |
| --------------------------- | -------------------------------------------------------------- |
| `SimplifySubComponents.cpp` | pod dispatch 패턴 제거 (flattenPod, eliminatePod, extractCall) |
| `LlzkToStablehlo.cpp`       | Main conversion pass + post-passes                             |
| `TypeConversion.cpp/.h`     | 타입 변환 + getPodInitializedRecords                           |
| `ArrayPatterns.cpp`         | array.read/write/extract/insert → stablehlo ops                |
| `FunctionPatterns.cpp`      | function.call → func.call                                      |

## 테스트 LLZK IR 파일 위치

```
/tmp/llzk_e2e/multiplexer_llzk/multiplexer_test_llzk/multiplexer_test.llzk
/tmp/llzk_e2e/pedersen_llzk/pedersen_test_llzk/pedersen_test.llzk
```

이 파일들은 circom 컴파일러로 생성된 pre-compiled LLZK IR. circom 컴파일러가 현재 설치되어 있지 않으므로 재생성
불가. `/tmp`에 있으므로 reboot 시 소실.

## 알려진 제약사항

1. `getPodInitializedRecords`가 attribute를 문자열로 출력 후 파싱하는 workaround 사용 — LLZK의
   pod.new properties 접근 API 부재 때문
1. `hasArrayOfPods` guard가 아직 존재 (nested block에서 eliminatePodDispatch 실행 시 일부 회로
   regression)
1. constrain function body clearing 시도 시 multimimc7/mimcsponge_wrap crash
   (sub-component call chain 문제)

## 해결된 문제 이력

### multiplexer_test, pedersen_test domination 에러 (f441f10)

**원인 3가지:**

1. `findCapturedArrays`가 inner while block args를 outer while carry로 잘못 승격 →
   `isProperAncestor` 체크로 nested region 값 스킵
1. `flattenPodArrayWhileCarry`에서 pod array result의 post-while `array.read` users
   미처리 → nondet으로 대체
1. `ArrayWritePattern`/`ArrayInsertPattern`이 void (0-result) ops에서
   `replaceOpWithNewOp` assertion 실패 → `replaceWithDUS` 헬퍼로 void/SSA 분기 처리 +
   `convertWhileBodyArgsToSSA`로 extract→write→insert 체인 SSA 변환

### ASan "Loading dialect in multi-threaded context" 에러

`func::FuncDialect`를 `dependentDialects`에 추가하여 해결 (`LlzkToStablehlo.td`).
