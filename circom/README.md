# `circom/` — circom interop tree

First-class C++ utilities for consuming circom artifacts from llzk-to-shlo.
Currently:

- [`wtns/`](wtns/) — parser for `.wtns` (witness binary, iden3 binfileutils v2)

This tree backs the M3 measurement harness's correctness gate: the GPU witness
output must equal `circom -c`'s native witness, byte-for-byte (see
[`CLAUDE.md`](../CLAUDE.md) "Load-Bearing Invariants" — circom is the source of
truth). A future `bench/m3:witness_compare` (PR-C) will read the .wtns produced
by `bench/m3/run_baseline.sh` and compare against m3_runner's GPU output at a
sampled witness index.

## Why a self-contained parser

We considered porting
[`fractalyze/rabbitsnark`](https://github.com/fractalyze/rabbitsnark)'s
`circom/wtns` reader, but it depends on `zkx::base::ReadOnlyBuffer`,
`zkx::math::BigInt`, and `zkx::math::bn254::Fr` — all from the proprietary
`fractalyze/zkx` repo, which is **not** part of our `@open_zkx` (the public
subset we pin). Pulling those in would either require adding the private repo as
a Bazel dep (not accessible from public CI) or vendoring multiple files of
buffer + endian + Serde infrastructure.

The .wtns v2 format is small and stable (iden3 binfileutils), so a ~150-LOC
self-contained parser is cheaper to maintain than tracking upstream's transitive
zkx dependency chain.

Public API:

```cpp
#include "circom/wtns/wtns.h"
absl::StatusOr<llzk_to_shlo::circom::WitnessFile> wtns =
    llzk_to_shlo::circom::ParseWtns("path/to/file.wtns");
// wtns->field_size_bytes, wtns->num_witnesses, wtns->modulus, wtns->data
// wtns->Witness(i) returns absl::Span<const uint8_t> at the i-th witness.
```

## Test fixture provenance

`wtns/multiplier_3.wtns` is the `.wtns` emitted by `circom -c` on a 3-input
multiplier circuit with input `{"in":["3","4","5"]}`. It was copied from
[`fractalyze/rabbitsnark`](https://github.com/fractalyze/rabbitsnark/blob/597c5760/circom/wtns/multiplier_3.wtns)
@ commit `597c5760` (2025-12-10) — 268 bytes, opaque test data, used to ground
the parser against real circom output. Field is bn254 Fr (32 bytes per witness);
the six witnesses are `[1, 60, 3, 4, 5, 12]`.

## When to extend

This tree should grow as our circom interop needs grow — for example, a
`circom/r1cs/` reader if we ever need to ingest constraint systems directly. The
directory level (top-level, not under `third_party/`) signals these are
first-class utilities owned by this repo, not external dependencies.
