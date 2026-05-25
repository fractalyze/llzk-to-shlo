# Design philosophy

`llzk-to-shlo` lowers LLZK circuit IR into StableHLO so that ZK witness
generation can ride the same ML-compiler infrastructure that the broader
hardware-accelerator ecosystem already optimizes. The end target is batched GPU
witness generation through [open-zkx](https://github.com/fractalyze/open-zkx)'s
`stablehlo_runner`, which collapses N independent proof inputs into a single
kernel launch instead of one per proof. Every design decision in the lowering
pipeline is shaped by four principles.

## Reuse ML compiler infrastructure

StableHLO is a production IR with GPU backends, fusion, loop vectorization, and
a broad ecosystem of tooling. Expressing ZK field arithmetic and array
operations as StableHLO ops — `felt.add` → `stablehlo.add`, structs → flat
tensors with offset-based slices — means the GPU backend, the batching pass, and
downstream optimizers all work without ZK-specific modifications. The
alternative — a bespoke ZK runtime — would require re-implementing every one of
those layers. The cost of the approach is that ZK primitives must be expressed
within StableHLO's type system and SSA constraints, which drives several of the
non-obvious lowering choices described in
[`../passes/llzk-to-stablehlo/README.md`](../passes/llzk-to-stablehlo/README.md).

## Batch first

Witness generation is embarrassingly parallel across proofs: each proof input is
independent, the arithmetic per witness is small (microseconds of GPU time), and
the dominant cost is per-execution dispatch overhead — JIT compilation, stream
allocation, and host synchronization together dominate the actual arithmetic
regardless of batch size. Amortizing that overhead across N inputs brings total
runtime to approximately one single-input execution regardless of N. To realize
this, every lowering must survive a leading batch dimension being prepended by
`BatchStablehlo`; the op-by-op bridging rules live in
[`../passes/batch-stablehlo.md`](../passes/batch-stablehlo.md).

The important corollary: the correct optimization target for witness generation
is eliminating the per-execution overhead (kernel-launch amortization,
dispatch-once), not microoptimizing per-kernel arithmetic. A change that halves
kernel arithmetic time but doubles dispatch invocations is a regression.

## GPU correctness is the gate

`@constrain` is erased during lowering — it exists only for the ZK verifier, not
for witness computation. GPU code therefore runs with no internal correctness
alarm: a miscompile produces the wrong witness silently. The only catch is
differential testing against circom's native C++ witness generator: for a given
input, `batch[i] == single[i]` against the circom `.wtns` reference means the
GPU witnesses are correct; any divergence is a miscompile, not a test framework
issue. LIT and unit tests prove IR shape; they cannot substitute for this gate.
The full correctness-gate hierarchy — why sister-circuit families are not valid
differential references, why `--O0` is required, how gate sentinels are enrolled
— is in [`../contracts/correctness-gate.md`](../contracts/correctness-gate.md).

## Frontend-agnostic target

LLZK is the stable contract between this project and its frontends. Circom is
the only production frontend today, but leaking Circom-specific assumptions into
`LlzkToStablehlo` patterns would break any future frontend that emits LLZK
differently. Concretely: pod dispatch elimination (`SimplifySubComponents`)
exists because Circom emits a specific `!pod.type` state machine for component
calls — this is a Circom artifact that must be eliminated in a dedicated
pre-pass so that `LlzkToStablehlo` sees only clean LLZK. Patterns in
`LlzkToStablehlo` that are Circom-shaped rather than LLZK-shaped are bugs.
