# Why circuits don't lower

Lowering failures do not distribute uniformly across circuits. They cluster into
a small number of structural families, each caused by a distinct gap in the
pipeline. Knowing the family tells you where to look for a fix.

## Circom → LLZK: extended-envelope timeouts

The dominant failure class at the Circom → LLZK stage is not a parse error or an
IR malformation — it is a wall-clock timeout. The Circom `llzk_backend/` crate
runs at 100% CPU on a single thread and never produces LLZK output within the
sampling envelope (60 s fast / 600 s extended), emitting no diagnostic and no
panic.

The affected circuits share a structural signature: Poseidon hash-based Merkle
trees and state-tree operations (maci batchUpdateStateTree, iden3 auth/proof,
Webb batchMerkleTreeUpdate, hydra commitment, zk-SQL, keccak full-permutation).
These templates generate large numbers of deeply-nested subcomponent instances
at Circom compile time, and the LLZK backend's code path for this shape appears
to have a pathological recursion or combinatorial expansion. The prior fast-fail
at parse (a mixed-subcomponent panic) was masking how expensive the backend
becomes once all circuits enter post-parse code paths.

Some circuits in this family recover when the envelope is extended far enough
(the PointCompress class recovered well past the fast-fail envelope but within
the extended limit); others do not recover even at the extended limit
(batchMerkleTreeUpdate at the larger sizes, gcm_siv, hydra). The classification
between "slow but terminates" and "pathologically non-terminating" is not yet
established for the majority of circuits not directly sampled at the extended
envelope.

This failure class lives entirely upstream of `llzk-to-shlo` — the fix belongs
in the Circom `llzk_backend/` crate.

## LLZK → StableHLO: unhandled shapes

Every circuit that produces LLZK IR currently lowers completely. When a new
circuit fails at this stage, the failure almost always falls into one of two
families.

**`SimplifySubComponents` normalization gap.** A `pod.new` or pod-carrying
`scf.while` shape that no existing SSC phase handles reaches the type conversion
pass, which cannot legalize `!pod.type`. The symptom is a legalization failure
with a `!pod.type` operand surviving into `LlzkToStablehlo`. Each new variant
requires either a new pre-pass phase or an extension of an existing phase (e.g.,
`unpackPodWhileCarry` or `flattenPodArrayWhileCarry`). The fixed-point driver
means a new phase integrates cleanly if it can commit to the standard
precondition: the block is pod-free except for the specific pattern the phase
targets.

**Partial-conversion gap.** A region or op shape has no registered pattern in
`LlzkToStablehlo`. The symptom is `applyPartialConversion` reporting an op as
illegal with no matching pattern. Each unhandled shape typically requires one
targeted pattern; the existing patterns for arithmetic (`FeltPatterns.cpp`),
struct access (`StructPatterns.cpp`), and array ops (`ArrayPatterns.cpp`) are
good templates. The trap to avoid: adding a pattern without verifying it in the
full pipeline — LIT tests prove IR shape but only the GPU correctness gate
catches silent miscompiles.

## PointCompress: conversion timeout

PointCompress (ed25519 point compression, ~21 K lines of LLZK IR) exposed a
distinct performance failure: the `applyPartialConversion` walk over the op
graph becomes quadratic for circuits where `materializeStructOfPodsCompField`
emits dynamic `<N x !felt>` carriers with `dynamic_slice`/`dynamic_update_slice`
reads and writes even when the carrier index K is a compile-time constant. In
the PointCompress case the materializer generated a large fraction of all tensor
ops as `tensor<1x!pf>` intermediates with dynamic indexing that the runtime
could not JIT in bounded time.

The structural fix — replacing the carrier with direct writer-to-reader SSA
binding when K is statically known, falling back to the current carrier for
runtime-K cascades — would eliminate this class of conversion timeout. Until
then, circuits with the Poseidon-style per-round template specialization pattern
(many distinct small function instances each carrying large inline constant
tables) hit the same JIT-stall at the GPU execution stage rather than at
lowering.
