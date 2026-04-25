# M3 Feasibility Report — llzk-to-shlo

ETH grant Milestone 3 deliverable. Submitted **2026-05-09**.

> **Status**: Skeleton (v0). Track A2 placeholder tables; Track A1 fills the
> measurement results during Phase 1–2. Final polish in Phase 3.

______________________________________________________________________

## 1. Executive Summary

`llzk-to-shlo` lowers ZK circuit IR (LLZK) to StableHLO so that witness
generation can ride on the same ML compiler infrastructure that already
optimizes for batch and GPU. This report measures the end-to-end pipeline
(Circom → LLZK → StableHLO → batched GPU execution via open-zkx
`stablehlo_runner`) on production-grade circuits and reports where the
GPU-batched pipeline wins, where it ties, and where the bottleneck still sits.

**Important scope limit (stated up-front)**: of 123 entry points in the public
[`circom-benchmarks`](https://github.com/project-llzk/circom-benchmarks) set,
**77 fail at the upstream Circom→LLZK frontend** with a single mixed-type
subcomponent panic (`circom-llzk/llzk_backend/src/template_ext.rs:243`, "Support
mixed type subcomponent instantiations not yet implemented"); 1 more
(PointCompress, 21K-line ed25519) hits a `SimplifySubComponents` timeout in our
own pipeline. The two flagship anchors most reviewers expect (full SHA-256, full
Keccak-256) are in the upstream-failing set. The Day-1 SHA-256 wrapper
experiment (`Sha256(64)` over circomlib) confirmed the panic is reproducible
end-to-end; anchor B fell back to
`iden3-core/src/utils_verifyCredentialSubject.circom` (Polygon ID production
primitive). See **§7 Limitations** for the full breakdown.

**Headline findings** *(placeholders — Track A1 fills these from measurements)*:

- Anchor A (`aes_256_encrypt`): GPU vs circom-native at N=65,536 — \*\[GPU
  win/loss
  - factor\]\*. Saturation knee at N ≈ *[N]*.
- Anchor B (`iden3_verify_credential_subject`): GPU vs circom-native at N=65,536
  — *[GPU win/loss + factor]*. Saturation knee at N ≈ *[N]*.
- Per-stage: kernel time dominates at large N; compile + JIT amortize across the
  batch (*[concrete numbers]*). D2H is *[bound]* per *[size]*.

**Bottom line** *(placeholder)*: GPU-batched StableHLO witness generation is
viable for *[batch-heavy use cases — list]* and not yet competitive for
*[latency-bound use cases — list]*; the dominant E2E coverage gap today is the
upstream Circom frontend, not our lowering.

______________________________________________________________________

## 2. Pipeline Architecture

The pipeline has four stages; each is documented in detail in
[`docs/E2E_LOWERING_GUIDE.md`](E2E_LOWERING_GUIDE.md).

```
Circom (.circom)
   │  circom --llzk concrete [--llzk_plaintext]
   ▼
LLZK IR (.llzk)
   │  llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo
   ▼
StableHLO IR (.mlir)
   │  llzk-to-shlo-opt --batch-stablehlo="batch-size=N"
   ▼
Batched StableHLO IR (.mlir)
   │  open-zkx stablehlo_runner (GPU)
   ▼
N witnesses from one kernel launch
```

Two passes carry the bulk of the work:

- **`SimplifySubComponents`** removes Circom's pod-dispatch state machine (each
  subcomponent input becomes ~40 LLZK lines of input-counter + pending-record +
  delayed `function.call`); flattens to direct `function.call`.
- **`LlzkToStablehlo`** is the heavy pass: pre-passes (input-pod elimination,
  while-carry promotion, SSA-ification of array writes), main partial conversion
  (LLZK ops → StableHLO), post-passes (`scf.while` → `stablehlo.while`, `scf.if`
  → `stablehlo.select`, residual cleanup, DCE, while-loop vectorization).

Three vectorization phases run after conversion:

1. **Phase 1** — 1-D independent `while` → element-wise tensor ops.
1. **Phase 1.5** — 2-D carry `while` → column write/read vectorization.
1. **Phase 2** — nested-`while` inner loop → 1-D carry vectorization.

`BatchStablehlo` adds a leading `N` dimension so a single kernel launch produces
`N` witnesses; per-op rules are in
[`docs/BATCH_STABLEHLO.md`](BATCH_STABLEHLO.md).

______________________________________________________________________

## 3. Method

### Backends compared

| Label        | Backend                                 | What it runs                                                                                                                                           |
| ------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gpu_zkx`    | open-zkx `stablehlo_runner` on GPU      | Batched StableHLO IR; one kernel launch per batch.                                                                                                     |
| `cpu_circom` | circom-native C++ witness (`circom -c`) | Per-witness, sequential; the source of truth (per CLAUDE.md "Load-Bearing Invariants" — `batch[i] == single[i]` against this is the correctness gate). |

`snarkjs`, `rapidsnark`, and WASM backends are **out of scope** (M3_PLAN §1 Q3):
they add setup cost without changing the GPU-batch story.

### Batch sizes

`N ∈ {1, 64, 4 096, 65 536, 262 144}`.

The geometric grid covers single-witness latency (`N=1`), small-batch
interactive (`N=64`), large-batch throughput (`N=4 096`–`262 144`).

### Per-stage timer

The harness times four stages independently:

| Stage     | What it covers                                                |
| --------- | ------------------------------------------------------------- |
| `compile` | StableHLO → device executable (open-zkx PJRT compile).        |
| `jit`     | Device-side autotune / first-launch warmup.                   |
| `kernel`  | Pure on-device execution time (excluding D2H).                |
| `d2h`     | Device-to-host transfer of the batched witness output.        |
| `total`   | Wall-clock = compile + jit + kernel + d2h + harness overhead. |

Harness overhead (= `total − Σ stages`) is reported as the bias term per M3_PLAN
§5 (instrumentation-bias risk).

### CSV schema (shared with Track A1 measurement harness)

```
circuit, backend, N, stage, time_ms, throughput_wits_per_sec
```

Output path: `bench/m3/results/<circuit>_<backend>.csv`.

### Hardware

- **GPU**: RTX 5090 (Blackwell, 32 GB).
- **CPU baseline host**: *[fill from build host: model, cores, RAM]*.
- **ZKX commit**: *[pin]*. **stablehlo commit**: *[pin]*. **circom-llzk
  commit**: *[pin]*.

### Repeat strategy

Each `(circuit, backend, N)` cell is run **3×, median reported**. Mitigates RTX
5090 ZKX JIT autotune non-determinism (M3_PLAN §5 Risk row 3; same phenomenon
documented on the `riscv-witness` side). If an autotune-disable flag exists in
the current ZKX pin, it is enabled and noted.

### Correctness gate

For every `(circuit, N)`, the harness asserts `batch[i] == single[i]` against
the circom-native witness for at least one sampled `i ∈ [0, N)`. Per
[`CLAUDE.md`](../CLAUDE.md) "Load-Bearing Invariants", circom is the source of
truth; a divergence is a real correctness bug, not a performance result.

______________________________________________________________________

## 4. Results — Placeholder Tables

> **Track A1 fills these tables Day 3+ from `bench/m3/results/*.csv`.**

### 4.1 Per-circuit throughput vs N

Throughput in **witnesses/second** (median of 3 runs).

| Circuit                           | Backend      | N=1 | N=64 | N=4 096 | N=65 536 | N=262 144 |
| --------------------------------- | ------------ | --- | ---- | ------- | -------- | --------- |
| `aes_256_encrypt`                 | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `aes_256_encrypt`                 | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `iden3_verify_credential_subject` | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `iden3_verify_credential_subject` | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_chi`                      | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_chi`                      | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_iota3`                    | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_iota3`                    | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_iota10`                   | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_iota10`                   | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_round0`                   | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_round0`                   | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_round20`                  | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_round20`                  | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_pad`                      | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_pad`                      | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_rhopi`                    | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_rhopi`                    | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_squeeze`                  | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_squeeze`                  | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_theta`                    | `gpu_zkx`    | TBD | TBD  | TBD     | TBD      | TBD       |
| `keccak_theta`                    | `cpu_circom` | TBD | TBD  | TBD     | TBD      | TBD       |

*[Line plot — throughput vs N per circuit, log-log axes — placeholder.]*

### 4.2 Per-stage breakdown at N = 65 536

Stage time in **ms** (median of 3 runs); GPU-side stages only.

| Circuit                           | compile | jit | kernel | d2h | total | harness overhead |
| --------------------------------- | ------- | --- | ------ | --- | ----- | ---------------- |
| `aes_256_encrypt`                 | TBD     | TBD | TBD    | TBD | TBD   | TBD              |
| `iden3_verify_credential_subject` | TBD     | TBD | TBD    | TBD | TBD   | TBD              |
| `keccak_chi`                      | TBD     | TBD | TBD    | TBD | TBD   | TBD              |
| *(other Tier-2 circuits)*         | TBD     | TBD | TBD    | TBD | TBD   | TBD              |

*[Stacked bar chart per circuit at N=65,536 — placeholder.]*

### 4.3 Saturation point per circuit

The **saturation N** is the smallest N at which
`throughput(N) ≥ 0.9 × throughput(2N)` (the bottleneck has flattened).

| Circuit                           | Saturation N | Bottleneck above saturation              |
| --------------------------------- | ------------ | ---------------------------------------- |
| `aes_256_encrypt`                 | TBD          | TBD (kernel / D2H / host-side stitching) |
| `iden3_verify_credential_subject` | TBD          | TBD                                      |
| *(all Tier-2 circuits)*           | TBD          | TBD                                      |

### 4.4 Correctness gate

For every cell in §4.1, `batch[i] == single[i]` against circom-native.

| Circuit           | All N pass       | First-divergence (if any) |
| ----------------- | ---------------- | ------------------------- |
| `aes_256_encrypt` | TBD (☐ all pass) | —                         |
| *(all circuits)*  | TBD              | —                         |

A divergence is escalated per M3_PLAN §5 Risk row 7 — halt Phase 1, treat as a
correctness bug.

______________________________________________________________________

## 5. Three-Axis Narrative

The performance story is best understood along three independent axes; the GPU
win is the **product** of all three, not any one in isolation.

### Axis A — Intra-circuit step parallelism via ZKX fusion

A single witness already exposes step-level parallelism *inside* the circuit's
StableHLO graph: independent `while` loops are vectorized to element-wise tensor
ops (Phase 1), 2-D carry loops to column ops (Phase 1.5), nested-while inner
loops to 1-D carries (Phase 2). ZKX's StableHLO fusion pass then collapses
element-wise chains into single kernels.

**M2 evidence** (cite [`docs/GPU_PROFILING.md`](GPU_PROFILING.md)):
MontgomeryDouble holds at **4 kernel launches** as N grows from 1 to 10K — the
launch count is a function of the *circuit's* fused-op DAG, not of the batch
size. This is what makes the M3 batch story possible: per-batch launch count is
constant, so the kernel time grows linearly with N while launch overhead does
not.

### Axis B — Cross-circuit batch parallelism via `BatchStablehlo`

`BatchStablehlo` adds a leading `N` dimension to every tensor. Each of the
constant ~`K` kernel launches per circuit (Axis A) now operates on
shape-`[N, …]` tensors instead of `[…]`. With `N` large and `K` small, total
work is `K × N`-element kernels — the GPU's preferred regime.

**M2 evidence** (cite [`docs/BATCH_STABLEHLO.md`](BATCH_STABLEHLO.md)): BabyBear
`Multiplier2` microbenchmark — `3.0× GPU vs CPU @ N=65 536`, `K=32 muls/elem`.
M3 §4.1 will show how this generalizes to production circuits.

### Axis C — ML compiler infrastructure reuse (the structural argument)

Once a circuit is in StableHLO, every optimization the StableHLO ecosystem ships
— fusion, layout assignment, autotune, GPU codegen, async D2H, multi-stream
scheduling — applies for free. Hand-rolled ZK provers re-derive each of these
per backend; this pipeline inherits them.

The structural consequence: the M3 numbers in §4 should be read as a *lower
bound* on what the pipeline can deliver. Future StableHLO ecosystem improvements
compound onto every measurement here without code changes on our side. (This is
the "reuse ML compiler infra" core principle from [`CLAUDE.md`](../CLAUDE.md).)

______________________________________________________________________

## 6. Use-case Matrix

| Use case                                    | Typical batch size | GPU-batch wins? | Why                                                                                     |
| ------------------------------------------- | ------------------ | --------------- | --------------------------------------------------------------------------------------- |
| Rollup transaction prover (server-side)     | 10³ – 10⁵ tx/proof | **Yes**         | Throughput-bound, batch-natural. Axes A+B+C all in play.                                |
| Anonymous airdrop (claim window)            | 10⁴ – 10⁶ users    | **Yes**         | Very batch-heavy; per-claim latency unimportant within the window.                      |
| Privacy pool (anonymity-set proofs)         | 10³ – 10⁵          | **Yes**         | Same shape as airdrop: large set, batch generation.                                     |
| Recursive aggregation (folding leaves)      | 10² – 10⁴ leaves   | **Yes**         | Each level batches; depth is small.                                                     |
| Single-tx wallet sign / interactive proof   | 1                  | **No**          | Latency-bound; compile + JIT do not amortize over a single proof. CPU C++ witness wins. |
| Rate-limited per-user proof (e.g. one vote) | 1–10               | **Likely no**   | Same as above; batch too small to amortize StableHLO compile cost.                      |

§4 measurements distinguish the saturation N where the GPU win begins per
circuit; this matrix is the application-side mapping.

______________________________________________________________________

## 7. Limitations

> **Stated before recommendations** (M3_PLAN §7 convention) — EF reviewers
> calibrate trust on a writer's willingness to surface failure modes early.

### 7.1 Coverage gap is upstream, not in our lowering

123 entry points in `circom-benchmarks` (commit `6897550c`):

| Stage                     | Pass | Fail   | Notes                                                                                                                                                                                           |
| ------------------------- | ---- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Circom → LLZK (concrete)  | 46   | **77** | All 77 hit a single panic at `circom-llzk/llzk_backend/src/template_ext.rs:243` ("Support mixed type subcomponent instantiations not yet implemented"). Bug is in the upstream Circom frontend. |
| LLZK → StableHLO          | 45   | 1      | PointCompress (21K-line ed25519) — `SimplifySubComponents` timeout in *our* pipeline; tracked separately, not M3 scope.                                                                         |
| StableHLO → Batched (N=4) | 45   | 0      | All 45 LLZK-passing circuits batch cleanly.                                                                                                                                                     |

**End-to-end rate: 45/123 = 36.6 %.** Of circuits that successfully produce LLZK
IR, **45/46 = 97.8 %** complete the full pipeline. The "E2E rate looks bad"
framing is misleading; the substrate failure rate is upstream.

Flagship circuits absent because of this single upstream bug:

- Full SHA-256 (`Sha256(N)` from circomlib).
- Full Keccak-256 (`keccak_256_256_test`, `keccak_full`, etc.).
- MACI batch state-tree update.
- Tornado Cash–style mixers (`Webb-tools/*`, semaphore).
- iden3 stateTransition / EdDSA / SMT proof family.
- Hydra commitments.

### 7.2 SHA-256 wrapper experiment (Day 1)

Per M3_PLAN §3 Phase 1 Day 1, we wrote a minimal `Sha256(64)` wrapper over
circomlib and ran `circom --llzk concrete`. **Result**: panic at
`template_ext.rs:243` after expanding 100 template instances, identical to the
bug surfaced by the failing 77. Anchor B fell back to
`iden3-core/src/utils_verifyCredentialSubject.circom` (Polygon ID
production-deployed primitive; passes today).

We do **not** fix the upstream Circom frontend in M3 (M3_PLAN §6); the bug is
documented here and will surface upstream as a separate issue.

### 7.3 PointCompress — `SimplifySubComponents` timeout

The 21K-line ed25519 LLZK IR exhausts the fixed-point loop budget in
`SimplifySubComponents`. Tracked as a separate engineering item; out of M3 scope
(M3_PLAN §2 Out-of-scope). Diagnostic: pod-dispatch nesting depth appears
unbounded in this circuit.

### 7.4 RTX 5090 ZKX JIT autotune non-determinism

Documented on the `riscv-witness` side (`zkx::base::Uniform` + autotune seed
sourcing). Mitigation: 3× re-run per cell, median reported. If a seeded autotune
flag exists in the current ZKX pin, it is enabled. Run-to-run variance is
reported per cell in §4 appendices.

### 7.5 Per-stage timer instrumentation bias

Wall-clock `total` and `Σ(compile, jit, kernel, d2h)` differ by **harness
overhead**. The gap is reported per cell in §4.2; do not interpret stage sums as
the only valid timing.

### 7.6 Backend baseline scope

Only circom-native C++ witness is used as the CPU baseline (M3_PLAN §1 Q3).
`snarkjs` / `rapidsnark` / WASM are not measured; they would not change the
GPU-batch conclusion but would broaden the comparison.

______________________________________________________________________

## 8. Recommendations / Future Work

1. **Upstream Circom frontend fix** (highest leverage). A single
   `template_ext.rs` panic gates 77 of the 78 failing circuits — fixing
   mixed-type subcomponent instantiations would lift the E2E rate from 36.6 %
   toward ≈ 99 %. Out of M3 scope, but the dominant high-impact next step.
1. **PointCompress `SimplifySubComponents` budget**. Likely a separate
   engineering item: bound nested pod-dispatch peel depth or migrate the pass to
   a worklist algorithm rather than a fixed-point loop.
1. **Persistent compile cache**. Compile + JIT are amortized per `N` only
   *within a session*. Caching the compiled executable across sessions shifts
   the saturation knee of every circuit to lower N.
1. *\[Track A1 may add concrete bottleneck-driven recommendations from the §4.2
   stage table — populated Phase 3 Day 11.\]*
1. **Frontend-agnostic regression suite**. The pipeline contract is at the LLZK
   layer (per [`CLAUDE.md`](../CLAUDE.md) "frontend-agnostic target"); a
   non-Circom LLZK producer would let us decouple from the upstream frontend bug
   and broaden coverage independently.

______________________________________________________________________

## 9. Appendices

### 9.1 Raw CSVs

`bench/m3/results/<circuit>_<backend>.csv` — schema in §3. Committed in this
repo at the M3 final-submission tag.

### 9.2 Nsight artifact (Phase 3)

Per M3_PLAN §3 Phase 3 Day 10–11: Nsight Systems profile of the 1–2 most
interesting circuits (largest GPU win or sharpest saturation knee). Recipe in
[`docs/GPU_PROFILING.md`](GPU_PROFILING.md). Artifacts attached to the M3
submission.

### 9.3 Reproduction recipe

```bash
# Build all passing circuits (45 today)
bazel build //examples/...

# Run the M3 measurement harness (Track A1)
bazel run //bench/m3:run -- \
  --circuits=aes_256_encrypt,iden3_verify_credential_subject,keccak_chi,keccak_iota3,keccak_iota10,keccak_round0,keccak_round20,keccak_pad,keccak_rhopi,keccak_squeeze,keccak_theta \
  --N=1,64,4096,65536,262144 \
  --backends=gpu_zkx,cpu_circom \
  --repeats=3 \
  --output_dir="$PWD/bench/m3/results"
```

### 9.4 Circuit list

Anchors A and B + Tier-2 keccak rounds — 11 circuits in §4 tables. Tier-3
stretch (1–2 iden3 utilities, 1–2 MACI utilities) only if Phase 2 finishes ahead
per M3_PLAN §2.

### 9.5 SHA-256 wrapper experiment (reproduction)

```bash
mkdir -p /tmp/sha256_test
cat > /tmp/main.circom <<'EOF'
pragma circom 2.0.0;
include "circomlib/circuits/sha256/sha256.circom";
template Sha256Hash(N_BITS) {
    signal input in[N_BITS]; signal output out[256];
    component sha = Sha256(N_BITS);
    for (var i = 0; i < N_BITS; i++) { sha.in[i] <== in[i]; }
    for (var i = 0; i < 256; i++) { out[i] <== sha.out[i]; }
}
component main = Sha256Hash(64);
EOF
circom /tmp/main.circom --llzk concrete --llzk_plaintext \
  -o /tmp/sha256_test/ \
  -l <path-to-circom-benchmarks>/libs
# Expected: panic at circom-llzk/llzk_backend/src/template_ext.rs:243
# "Support mixed type subcomponent instantiations not yet implemented"
```

### 9.6 References

- [`E2E_LOWERING_GUIDE.md`](E2E_LOWERING_GUIDE.md) — pipeline architecture.
- [`BATCH_STABLEHLO.md`](BATCH_STABLEHLO.md) — vectorization phases.
- [`CIRCUIT_COVERAGE.md`](CIRCUIT_COVERAGE.md) — full 123-circuit coverage
  table.
- [`GPU_PROFILING.md`](GPU_PROFILING.md) — Nsight Systems methodology.
- [`CLAUDE.md`](../CLAUDE.md) — design philosophy, load-bearing invariants.
