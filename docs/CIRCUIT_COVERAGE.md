# Circuit Coverage Report

Full pipeline analysis of all 123 entry points in
[circom-benchmarks](https://github.com/project-llzk/circom-benchmarks) (commit
6897550c).

## Summary

| Stage                     | Pass | Fail | Rate  |
| ------------------------- | ---- | ---- | ----- |
| Circom -> LLZK (concrete) | 47   | 76   | 38.2% |
| LLZK -> StableHLO         | 47   | 0    | 100%  |
| StableHLO -> Batch (N=4)  | 47   | 0    | 100%  |

**End-to-end (Circom -> Batch): 47/123 (38.2%)**

The 76 Circom -> LLZK failures are gated by a two-layer upstream blocker stack
in the [llzk-circom](https://github.com/project-llzk/circom) frontend — not by
llzk-to-shlo. See § Failure Analysis below for details (Circom -> LLZK count
re-validated 2026-04-28; LLZK -> StableHLO count re-validated 2026-05-22 after
`pointcompress` was unblocked by the SSC Phase 4 `hasStructWritemInBody` guard
and the constant `arith.cmpi` / `scf.if` fold in the `scf.if → stablehlo.select`
post-pass — full pipeline now lands every LLZK-producing circuit). Of circuits
that successfully produce LLZK IR, **47/47 (100%)** complete the full pipeline.

The 4 additional `webb_batch_merkle_{4,8,16,32}` sibling-size BUILD targets in
`examples/BUILD.bazel` are derived sizes that live outside the 123 canonical
entry points; they now lower and batch alongside `_64` (same fix unblocks all 5
sizes) and are no longer tagged `manual`.

**M3 correctness gate**: 43 of the 46 end-to-end-passing circuits are wired into
`//bench/m3:m3_correctness_gate_test` and byte-equal `gpu_zkx` output against
the circom-native `.wtns` reference at N=1 on every PR (9 keccak step chips + 10
iden3 utility templates + 5 maci utilities + 6 EC primitives (MontgomeryDouble,
MontgomeryAdd, Edwards2Montgomery, Montgomery2Edwards, Window4, WindowMulFix) +
Num2Bits + Num2BitsCheck + LessThanBounded + 4 arithmetic/logic chips
(fulladder, onlycarry, BinSum, Decoder) + 1 bit-manipulation chip
(BitElementMulAny) + 3 AES variants gated via output-only prefix-size mode in
`PREFIX_SIZES`, plus aes_mul (GF(2⁸) finite-field multiplier with full-witness
byte-equality) and EmulatedAesencSubstituteBytes (AES S-box LUT)). See
[`M3_REPORT.md` §4.4](M3_REPORT.md) for the per-circuit gate matrix and
CLAUDE.md → "M3 correctness gate convention" for the sentinel format. The 3
end-to-end-passing chips that are intentionally held out from the 46
(`SignedFpCarryModP`, `FpMultiply`, `PointCompress`) — see "M3 gate deferred —
SignedFpCarryModP-family" below for the SFP pair, and "M3 gate not yet wired —
PointCompress" below for the ed25519 one. The 4 `webb_poseidon_vanchor_*` chips
are tracked separately under "M3 gate deferred — webb_poseidon_vanchor\_\*"
below — they are not counted in either the 43 gated or the 3 held-out-from-46
totals here.

### Building Individual Circuits

All 123 circuits have Bazel targets in `//examples`. Each target generates both
LLZK IR (`_llzk` suffix) and StableHLO output.

```bash
# Build all passing circuits
bazel build //examples/...

# Build a specific circuit
bazel build //examples:montgomerydouble          # StableHLO output
bazel build //examples:montgomerydouble_llzk     # LLZK IR only

# Inspect outputs
cat bazel-bin/examples/montgomerydouble.stablehlo.mlir
cat bazel-bin/examples/montgomerydouble_llzk.llzk

# Reproduce a failing circuit (tagged manual)
bazel build //examples:zksql_delete              # fails at circom stage
```

______________________________________________________________________

## Failure Analysis

### Circom -> LLZK Failures (76 circuits)

74 of the 76 circuits are gated upstream by a *two-layer* blocker stack in the
circom LLZK backend; the remaining 2 (`maci/batchUpdateStateTree_32_batch256`,
`maci/quadVoteTally_32_batch256`) time out before reaching either layer. The
2026-04-28 snapshot of this section listed `Webb-tools/batchMerkleTreeUpdate_64`
as a third timeout — re-validated 2026-05-21 it produces LLZK IR successfully
and crashes inside `SimplifySubComponents` instead, so it is moved to the LLZK
-> StableHLO failure bucket below (alongside its 4 already-failing sibling sizes
`_{4,8,16,32}`).

**Two-layer upstream blocker stack** (empirically re-validated 2026-04-28
against circom built from `project-llzk/circom` `dev/handle_concrete_mixed`
`9b084a6d`): the originally-reported `template_ext.rs:243` mixed-type
subcomponent panic is resolved by upstream PR #376 and `9b084a6d`, but a deeper
`Conflicting types to read array` error class on parameterized component arrays
(`inner[i] = Inner(i)` patterns) still gates the same anchor set. See
[project-llzk/circom#386](https://github.com/project-llzk/circom/issues/386) for
the minimal repro and per-anchor fact table. Both layers live in the upstream
Circom frontend, not in llzk-to-shlo.

The first-layer panic, when reproduced against an older circom build:

```
thread 'main' panicked at llzk_backend/src/template_ext.rs:243:25:
not yet implemented: Support mixed type subcomponent instantiations
```

**To reproduce** (example: zk-SQL/delete):

```bash
# Clone circom-benchmarks
git clone https://github.com/project-llzk/circom-benchmarks.git
cd circom-benchmarks && git checkout 6897550c

# Run circom with --llzk concrete
circom applications/zk-SQL/src/delete.circom \
  --llzk concrete -o /tmp/out/ \
  -l applications/zk-SQL/src \
  -l libs \
  -l libs/circomlib/circuits
# Day-1 (pre-2026-04-28) expected output:
#   panic at llzk_backend/src/template_ext.rs:243
#   "not yet implemented: Support mixed type subcomponent instantiations"
# Post-2026-04-28 (circom built from dev/handle_concrete_mixed 9b084a6d):
#   first-layer panic resolved; the same circuit now fails with
#   "Failed to generate LLZK IR: Conflicting types to read array"
# See https://github.com/project-llzk/circom/issues/386 for the second-layer
# blocker.
```

**Affected circuit families**:

| Family           | Count | Description                                     |
| ---------------- | ----- | ----------------------------------------------- |
| maci             | 26    | Voting/state tree (Poseidon hash, Merkle trees) |
| iden3-core       | 16    | Identity (EdDSA signatures, SMT proofs)         |
| Webb-tools       | 12    | Anchor protocol (Poseidon, Merkle updates)      |
| keccak256-circom | 5     | Keccak hash (multi-round permutation)           |
| aes-circom       | 3     | AES-GCM (GF multiplication, polyval)            |
| hydra            | 3     | Commitment verification (Poseidon, Merkle)      |
| zk-SQL           | 4     | SQL operations (Poseidon, comparison trees)     |
| semaphore        | 1     | Anonymous signaling (Poseidon, Merkle)          |
| Other            | 7     | batchUpdateStateTree timeout variants, etc.     |

**Common pattern**: parameterized component arrays (`inner[i] = Inner(i)`
patterns) — child templates instantiated inside an indexed loop with a parameter
that varies per index. The first-layer fix lets these expand; the second-layer
`Conflicting types to read array` check then rejects the expanded form. See
[project-llzk/circom#386](https://github.com/project-llzk/circom/issues/386) for
the canonical minimal repro.

### LLZK -> StableHLO Failures

No outstanding failures. Every circuit that produces LLZK IR completes the full
Circom -> LLZK -> StableHLO -> Batch pipeline.

`PointCompress` (21K lines of LLZK IR for ed25519 point compression) was the
last entry in this bucket. Two compounding bugs blocked it:

1. **SSC Phase 4 (`eraseDeadPodAndCountOps`) erased `scf.if` whose body's only
   side effect is a felt-typed `struct.writem`.** The canonical shape is
   `@LessThanBounded_5::@compute`:
   `%y = scf.if %cond -> (!felt.type) { struct.writem %self[@out] = %v1; scf.yield %v1 } else { struct.writem %self[@out] = %v2; scf.yield %v2 }`
   with `%y` unused. Phase 4's `isOpAndNestedResultsExternallyUnused` only
   inspects nested SSA results, so the side-effecting writem looked like dead
   bookkeeping. With every writem on `@out` gone, `collectWritemTargets`
   returned `{}` and `registerStructFieldOffsets` allocated no slot for the
   field — the downstream `struct.readm %_[@out]` in `@ModSubThree_6::@compute`
   then failed to legalize with `member offset not found for: out`. Fixed by
   adding `hasStructWritemInBody` as a Phase 4 guard sibling to the existing
   `hasNonPodArrayWriteInBody`.
1. **`arith.cmpi <pred>, %c1, %c2 : index` feeding `scf.if -> (index)` reached
   the `scf.if -> stablehlo.select` post-pass with `tensor<index>` operands** —
   a type `stablehlo.select` does not accept. PointCompress's `long_div_*` emits
   this for `min(100, 200)` trip-count guards. Fixed by folding constant-operand
   `arith.cmpi` and constant-condition `scf.if` ahead of the post-pass; both are
   textbook canonical rewrites.

The `webb_batch_merkle_{4,8,16,32,64}` family previously sat in this bucket with
a `materializePodArrayCompField` use-after-free at SSC. Root cause was the
inner-arUser handler erasing a sibling @F'-readm that was itself a later entry
in the outer `for (Operation *rm : readms)` loop — under deeply nested recursive
members the inner walk picks up readms across nesting levels, and the inner
level's @F'-readm can be a downstream user of the outer level's extracted slice.
Fixed by deferring all erases until after the for-rm loop completes (the
deeply-nested rm survives its turn as a no-op once its uses have been replaced
with `llzk.nondet`). Webb also needed `felt.bit_or` / `felt.bit_xor` lowering
patterns to reach StableHLO; both added alongside the SSC fix.

______________________________________________________________________

## M3 gate deferred — webb_poseidon_vanchor\_\*

`webb_poseidon_vanchor_{2_2, 16_2, 16_8, 2_8}` all four lower end-to-end (Circom
→ LLZK → StableHLO → Batch ✓ at PR #105) but are **held out of
`m3_correctness_gate_test`** — the GPU runner cannot JIT the resulting IR in
bounded time. Two layered causes, neither in llzk-to-shlo proper:

1. **Upstream circom emits per-round template specialization.** Poseidon's
   `for (var i = 0; i < nRoundsP; i++) ark[i] = Ark(t, C, t*i);` pattern
   produces 65–68 distinct `Ark_K_compute` functions per Poseidon instance
   (compile-time `var` loop unrolled, each `Ark_K` carries its own inline
   128-element bn254 round-constant table). For webb_2_2 that is **261 Ark
   functions × 711 lines × ~264 unique constants ≈ 105K `stablehlo.constant`
   ops**. **TODO**(project-llzk/circom or project-llzk/llzk-lib): parameterize
   Ark template by K + lift the round-constant table to module scope.

1. **Local: `materializeStructOfPodsCompField` static-K carrier.** The
   materializer emits `<N x !felt>` carriers with dynamic_slice /
   dynamic_update_slice reads/writes even though every (class, K) pair has a
   `arith.constant`-known K. In webb_2_2 this produces 124,705
   `tensor<1x!pf_bn254_sf>` ops (95% of all tensor ops; cf. `keccak_round0`
   2,095 = 33%). **TODO**(llzk-to-shlo, `SimplifySubComponents.cpp:2779`):
   replace the carrier with direct writer-to-reader SSA binding when K is
   statically known; falls back to the current carrier for runtime-K cascades.
   Independent of (1).

The 4 fixtures (`bench/m3/inputs/webb_poseidon_vanchor_*.{json,json.gate,wtns}`)
exist locally as untracked files and stay un-committed until the baseline is
green, per CLAUDE.md's "don't ship a gate sentinel before its baseline is
currently green" rule.

______________________________________________________________________

## M3 gate deferred — SignedFpCarryModP-family

`SignedFpCarryModP/src/main.circom` and `FpMultiply/src/main.circom` both
end-to-end Circom → LLZK → StableHLO → Batch but are **held out of
`m3_correctness_gate_test`** for the same JIT-stall reason as the webb chips
above — the GPU runner pegs a single CPU thread at 100 % with no progress output
and no `ptxas` child, the webb-stall fingerprint at smaller scale. Empirically
reproduced 2026-05-15 during Batch B enrollment: `signed_fp_carry_mod_p` JIT was
killed at ~12 min wall, RSS stable at 280 MB, GPU memory pinned but no util.
`FpMultiply` embeds a `SignedFpCarryModP(55, 7, …)` instance plus
`BigMultShortLong` + `PrimeReduce`, so the same path is hit at strictly larger
cost.

The lowered `@main`'s structural footprint matches the pattern:
SignedFpCarryModP returns `tensor<506x!pf_bn254_sf>` from 7 inputs (599 wires,
1,836-line `@main`); FpMultiply returns `tensor<7x!pf_bn254_sf>` from 14 inputs
but transitively embeds the same SignedFpCarryModP body (1,836 wires, larger
`@main`). Same **TODO**(llzk-to-shlo, `SimplifySubComponents.cpp:2779`) as the
webb chips — carrier reduction unlocks both families.

No fixtures are committed; the gate stays at 43/46 until the JIT path is
unblocked. Re-open `signed_fp_carry_mod_p` first (smaller of the two), then
fpmultiply once carrier reduction lands.

______________________________________________________________________

## Passing Circuits (47)

All 47 circuits pass the complete pipeline: Circom -> LLZK -> StableHLO ->
Batch(N=4).

### By Category

| Category          | Count | Circuits                                                                                                                                                                                                                      |
| ----------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arithmetic/Logic  | 5     | BinSum, Decoder, fulladder, onlycarry, FpMultiply                                                                                                                                                                             |
| Bit manipulation  | 3     | Num2Bits, Num2BitsCheck, BitElementMulAny                                                                                                                                                                                     |
| Comparison        | 2     | LessThanBounded, SignedFpCarryModP                                                                                                                                                                                            |
| Elliptic curve    | 5     | Edwards2Montgomery, Montgomery2Edwards, MontgomeryAdd, MontgomeryDouble, Window4                                                                                                                                              |
| EC fixed-base mul | 1     | WindowMulFix                                                                                                                                                                                                                  |
| AES encryption    | 4     | aes_256_ctr_test, aes_256_encrypt_test, aes_256_key_expansion_test, mul_test                                                                                                                                                  |
| Keccak primitives | 9     | chi_test, iota3_test, iota10_test, keccakfRound0_test, keccakfRound20_test, pad_test, rhopi_test, squeeze_test, theta_test                                                                                                    |
| Iden3 utilities   | 10    | inTest, queryTest, utils_GetValueByIndex, utils_getClaimExpiration, utils_getClaimSubjectOtherIden, utils_getSubjectLocation, utils_isExpirable, utils_isUpdatable, utils_verifyCredentialSubject, utils_verifyExpirationTime |
| MACI utilities    | 5     | calculateTotal_test, decrypt_test, quinGeneratePathIndices_test, quinSelector_test, splicer_test                                                                                                                              |
| Hash (AES subset) | 1     | EmulatedAesencSubstituteBytes                                                                                                                                                                                                 |
| Webb batch merkle | 1     | batchMerkleTreeUpdate_64 (the 4 derived sizes `_{4,8,16,32}` also pass — outside the 123 canonical set)                                                                                                                       |

### Full List

| #   | Circuit                                              | LLZK | StableHLO | Batch |
| --- | ---------------------------------------------------- | ---- | --------- | ----- |
| 1   | BinSum/src/main.circom                               | PASS | PASS      | PASS  |
| 2   | BitElementMulAny/src/main.circom                     | PASS | PASS      | PASS  |
| 3   | Decoder/src/main.circom                              | PASS | PASS      | PASS  |
| 4   | Edwards2Montgomery/src/main.circom                   | PASS | PASS      | PASS  |
| 5   | EmulatedAesencSubstituteBytes/src/main.circom        | PASS | PASS      | PASS  |
| 6   | FpMultiply/src/main.circom                           | PASS | PASS      | PASS  |
| 7   | LessThanBounded/src/main.circom                      | PASS | PASS      | PASS  |
| 8   | Montgomery2Edwards/src/main.circom                   | PASS | PASS      | PASS  |
| 9   | MontgomeryAdd/src/main.circom                        | PASS | PASS      | PASS  |
| 10  | MontgomeryDouble/src/main.circom                     | PASS | PASS      | PASS  |
| 11  | Num2Bits/src/main.circom                             | PASS | PASS      | PASS  |
| 12  | Num2BitsCheck/src/main.circom                        | PASS | PASS      | PASS  |
| 13  | SignedFpCarryModP/src/main.circom                    | PASS | PASS      | PASS  |
| 14  | Window4/src/main.circom                              | PASS | PASS      | PASS  |
| 15  | WindowMulFix/src/main.circom                         | PASS | PASS      | PASS  |
| 16  | aes-circom/src/aes_256_ctr_test.circom               | PASS | PASS      | PASS  |
| 17  | aes-circom/src/aes_256_encrypt_test.circom           | PASS | PASS      | PASS  |
| 18  | aes-circom/src/aes_256_key_expansion_test.circom     | PASS | PASS      | PASS  |
| 19  | aes-circom/src/mul_test.circom                       | PASS | PASS      | PASS  |
| 20  | fulladder/src/main.circom                            | PASS | PASS      | PASS  |
| 21  | iden3-core/src/inTest.circom                         | PASS | PASS      | PASS  |
| 22  | iden3-core/src/queryTest.circom                      | PASS | PASS      | PASS  |
| 23  | iden3-core/src/utils_GetValueByIndex.circom          | PASS | PASS      | PASS  |
| 24  | iden3-core/src/utils_getClaimExpiration.circom       | PASS | PASS      | PASS  |
| 25  | iden3-core/src/utils_getClaimSubjectOtherIden.circom | PASS | PASS      | PASS  |
| 26  | iden3-core/src/utils_getSubjectLocation.circom       | PASS | PASS      | PASS  |
| 27  | iden3-core/src/utils_isExpirable.circom              | PASS | PASS      | PASS  |
| 28  | iden3-core/src/utils_isUpdatable.circom              | PASS | PASS      | PASS  |
| 29  | iden3-core/src/utils_verifyCredentialSubject.circom  | PASS | PASS      | PASS  |
| 30  | iden3-core/src/utils_verifyExpirationTime.circom     | PASS | PASS      | PASS  |
| 31  | keccak256-circom/src/chi_test.circom                 | PASS | PASS      | PASS  |
| 32  | keccak256-circom/src/iota10_test.circom              | PASS | PASS      | PASS  |
| 33  | keccak256-circom/src/iota3_test.circom               | PASS | PASS      | PASS  |
| 34  | keccak256-circom/src/keccakfRound0_test.circom       | PASS | PASS      | PASS  |
| 35  | keccak256-circom/src/keccakfRound20_test.circom      | PASS | PASS      | PASS  |
| 36  | keccak256-circom/src/pad_test.circom                 | PASS | PASS      | PASS  |
| 37  | keccak256-circom/src/rhopi_test.circom               | PASS | PASS      | PASS  |
| 38  | keccak256-circom/src/squeeze_test.circom             | PASS | PASS      | PASS  |
| 39  | keccak256-circom/src/theta_test.circom               | PASS | PASS      | PASS  |
| 40  | maci/src/calculateTotal_test.circom                  | PASS | PASS      | PASS  |
| 41  | maci/src/decrypt_test.circom                         | PASS | PASS      | PASS  |
| 42  | maci/src/quinGeneratePathIndices_test.circom         | PASS | PASS      | PASS  |
| 43  | maci/src/quinSelector_test.circom                    | PASS | PASS      | PASS  |
| 44  | maci/src/splicer_test.circom                         | PASS | PASS      | PASS  |
| 45  | onlycarry/src/main.circom                            | PASS | PASS      | PASS  |
| 46  | Webb-tools/test/batchMerkleTreeUpdate_64.circom      | PASS | PASS      | PASS  |
| 47  | PointCompress/src/main.circom                        | PASS | PASS      | PASS  |

______________________________________________________________________

## Failing Circuits (72)

### LLZK Frontend Failures (72)

69 of these 72 circuits are gated upstream by the two-layer blocker stack — see
§ Failure Analysis; the remaining 3 time out before reaching either layer. The
Error column points back to that prose block.

| #   | Circuit                                                     | Stage  | Error                             |
| --- | ----------------------------------------------------------- | ------ | --------------------------------- |
| 1   | Webb-tools/test/poseidon_vanchor_2_2.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 2   | Webb-tools/test/poseidon_vanchor_2_8.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 3   | Webb-tools/test/poseidon_vanchor_16_2.circom                | CIRCOM | upstream — see § Failure Analysis |
| 4   | Webb-tools/test/poseidon_vanchor_16_8.circom                | CIRCOM | upstream — see § Failure Analysis |
| 5   | Webb-tools/test/vanchor_forest_2_2.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 6   | Webb-tools/test/vanchor_forest_2_8.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 7   | Webb-tools/test/vanchor_forest_16_2.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 8   | Webb-tools/test/vanchor_forest_16_8.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 9   | aes-circom/src/gcm_siv_dec_2_keys_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 10  | aes-circom/src/gcm_siv_enc_2_keys_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 11  | aes-circom/src/gfmul_int_test.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 12  | aes-circom/src/polyval_test.circom                          | CIRCOM | upstream — see § Failure Analysis |
| 13  | hydra/src/hydra-s1.circom                                   | CIRCOM | upstream — see § Failure Analysis |
| 14  | hydra/src/verify-hydra-commitment.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 15  | hydra/src/verify-merkle-path.circom                         | CIRCOM | upstream — see § Failure Analysis |
| 16  | iden3-core/src/auth.circom                                  | CIRCOM | upstream — see § Failure Analysis |
| 17  | iden3-core/src/authTest.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 18  | iden3-core/src/authWithRelayTest.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 19  | iden3-core/src/credentialAtomicQueryMTP.circom              | CIRCOM | upstream — see § Failure Analysis |
| 20  | iden3-core/src/credentialAtomicQueryMTPTest.circom          | CIRCOM | upstream — see § Failure Analysis |
| 21  | iden3-core/src/credentialAtomicQueryMTPWithRelay.circom     | CIRCOM | upstream — see § Failure Analysis |
| 22  | iden3-core/src/credentialAtomicQueryMTPWithRelayTest.circom | CIRCOM | upstream — see § Failure Analysis |
| 23  | iden3-core/src/credentialAtomicQuerySig.circom              | CIRCOM | upstream — see § Failure Analysis |
| 24  | iden3-core/src/credentialAtomicQuerySigTest.circom          | CIRCOM | upstream — see § Failure Analysis |
| 25  | iden3-core/src/idOwnershipBySignatureTest.circom            | CIRCOM | upstream — see § Failure Analysis |
| 26  | iden3-core/src/idOwnershipBySignatureWithRelayTest.circom   | CIRCOM | upstream — see § Failure Analysis |
| 27  | iden3-core/src/poseidon.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 28  | iden3-core/src/poseidon14.circom                            | CIRCOM | upstream — see § Failure Analysis |
| 29  | iden3-core/src/poseidon16.circom                            | CIRCOM | upstream — see § Failure Analysis |
| 30  | iden3-core/src/stateTransition.circom                       | CIRCOM | upstream — see § Failure Analysis |
| 31  | iden3-core/src/stateTransitionTest.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 32  | iden3-core/src/utils_checkIdenStateMatchesRoots.circom      | CIRCOM | upstream — see § Failure Analysis |
| 33  | iden3-core/src/utils_verifyClaimSignature.circom            | CIRCOM | upstream — see § Failure Analysis |
| 34  | keccak256-circom/src/absorb_test.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 35  | keccak256-circom/src/final_test.circom                      | CIRCOM | upstream — see § Failure Analysis |
| 36  | keccak256-circom/src/keccak_256_256_test.circom             | CIRCOM | upstream — see § Failure Analysis |
| 37  | keccak256-circom/src/keccak_32_256_test.circom              | CIRCOM | upstream — see § Failure Analysis |
| 38  | keccak256-circom/src/keccakf_test.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 39  | maci/src/batchUpdateStateTree_32.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 40  | maci/src/batchUpdateStateTree_32_batch16.circom             | CIRCOM | upstream — see § Failure Analysis |
| 41  | maci/src/batchUpdateStateTree_32_batch256.circom            | CIRCOM | timeout (30s)                     |
| 42  | maci/src/batchUpdateStateTree_large.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 43  | maci/src/batchUpdateStateTree_medium.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 44  | maci/src/batchUpdateStateTree_small.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 45  | maci/src/batchUpdateStateTree_test.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 46  | maci/src/ecdh_test.circom                                   | CIRCOM | upstream — see § Failure Analysis |
| 47  | maci/src/hasher11_test.circom                               | CIRCOM | upstream — see § Failure Analysis |
| 48  | maci/src/hasher5_test.circom                                | CIRCOM | upstream — see § Failure Analysis |
| 49  | maci/src/hashleftright_test.circom                          | CIRCOM | upstream — see § Failure Analysis |
| 50  | maci/src/merkleTreeCheckRoot_test.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 51  | maci/src/merkleTreeInclusionProof_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 52  | maci/src/merkleTreeLeafExists_test.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 53  | maci/src/performChecksBeforeUpdate_test.circom              | CIRCOM | upstream — see § Failure Analysis |
| 54  | maci/src/publicKey_test.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 55  | maci/src/quadVoteTally_32.circom                            | CIRCOM | upstream — see § Failure Analysis |
| 56  | maci/src/quadVoteTally_32_batch16.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 57  | maci/src/quadVoteTally_32_batch256.circom                   | CIRCOM | timeout (30s)                     |
| 58  | maci/src/quadVoteTally_large.circom                         | CIRCOM | upstream — see § Failure Analysis |
| 59  | maci/src/quadVoteTally_medium.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 60  | maci/src/quadVoteTally_small.circom                         | CIRCOM | upstream — see § Failure Analysis |
| 61  | maci/src/quadVoteTally_test.circom                          | CIRCOM | upstream — see § Failure Analysis |
| 62  | maci/src/quinTreeCheckRoot_test.circom                      | CIRCOM | upstream — see § Failure Analysis |
| 63  | maci/src/quinTreeInclusionProof_test.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 64  | maci/src/quinTreeLeafExists_test.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 65  | maci/src/resultCommitmentVerifier_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 66  | maci/src/updateStateTree_test.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 67  | maci/src/verifySignature_test.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 68  | semaphore/src/semaphore.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 69  | zk-SQL/src/delete.circom                                    | CIRCOM | upstream — see § Failure Analysis |
| 70  | zk-SQL/src/insert.circom                                    | CIRCOM | upstream — see § Failure Analysis |
| 71  | zk-SQL/src/select.circom                                    | CIRCOM | upstream — see § Failure Analysis |
| 72  | zk-SQL/src/update.circom                                    | CIRCOM | upstream — see § Failure Analysis |
