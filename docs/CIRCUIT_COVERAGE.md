# Circuit Coverage Report

Full pipeline analysis of all 123 entry points in
[circom-benchmarks](https://github.com/project-llzk/circom-benchmarks) (commit
6897550c).

## Summary

| Stage                     | Pass | Fail | Rate  |
| ------------------------- | ---- | ---- | ----- |
| Circom -> LLZK (concrete) | 50   | 73   | 40.7% |
| LLZK -> StableHLO         | 50   | 0    | 100%  |
| StableHLO -> Batch (N=4)  | 50   | 0    | 100%  |

**End-to-end (Circom -> Batch): 50/123 (40.7%)** at the extended-envelope (bazel
CI) measurement, +3 vs the 47/123 prior baseline. Composition: +4 new passes
(`aes-circom/gfmul_int_test`, `aes-circom/polyval_test`, `maci/ecdh_test`,
`maci/publicKey_test`) from upstream parse fixes; −1
`Webb-tools/batchMerkleTreeUpdate_64` regression at the build-server host
(verified TIMEOUT_600s; gpu-server timing not yet sampled).

Re-measured 2026-05-23 against `project-llzk/circom@llzk` HEAD (commit
`fc46d662`, post-PR #432) — built locally via `cargo build --release`.
Runner-side `$HOME/circom` is built from the same llzk branch via the
project-llzk Nix flake; commits may diverge by a day or two until the next
runner refresh. Both pipeline-aware variants (`circom --llzk concrete` and
`circom --llzk concrete --llzk_plaintext --stabilize`) produce identical
pass/fail outcomes per chip; this report's count is the intersection. Of
circuits that successfully produce LLZK IR, **50/50 (100%)** complete the full
pipeline.

**Methodology — two envelopes**: the 123 chips were first measured at 60s/chip
on the build-server (fast envelope). 47 passed within 60s, 76 exceeded it. The
76 outliers were sampled at 600s/chip:

- 1 chip recovered to PASS (`PointCompress` at ~140s — added to passing)
- 2 chips recovered to PASS (`keccak256-circom/keccakfRound0_test` at ~305s,
  `keccakfRound20_test` at ~84s — already in passing list, just slower under new
  circom)
- 8 chips confirmed TIMEOUT_600s on build-server
  (`Webb-tools/batchMerkleTreeUpdate_{16,32,64}`,
  `aes-circom/gcm_siv_{dec,enc}_2_keys_test`, all 3 `hydra/*`)
- 65 chips not retested — bazel CI envelope status uncertain

The headline 50/123 conservatively assumes the 65 untested chips remain failing.
Their true bazel CI envelope status may be higher; the doc will update as
samples are added.

**Upstream blockers cleared, compile-time stretched**: the two-layer parse-time
blocker stack that previously gated this section was the `template_ext.rs:243`
mixed-subcomponent panic together with the `Conflicting types to read array`
error class on parameterized component arrays. Both layers were resolved
upstream by
[project-llzk/circom#381](https://github.com/project-llzk/circom/pull/381) and
[#398](https://github.com/project-llzk/circom/pull/398) (closing
[issue #386](https://github.com/project-llzk/circom/issues/386)). At this
commit, **zero chips fail at parse**. Every previously-failing chip now enters
the post-parse circom backend code paths. The new circom compile profile is also
visibly slower across the board — even prior-passing chips compile in 5×-50× the
wall time they did under the predecessor binary; the headline number above
counts only chips that complete within the extended (~600s) envelope used by
samples in this report. See § Failure Analysis for the composition diff and §
Compile-time stretch for the slowdown profile.

The 4 additional `webb_batch_merkle_{4,8,16,32}` sibling-size BUILD targets in
`examples/BUILD.bazel` are derived sizes parameterized off `_64`'s source. With
the new circom, `_64` itself exceeds 600s on the build-server (verified) and the
`_{16, 32}` siblings exhibit the same deep-compile pattern (also verified at
600s). The `_{4, 8}` siblings are TIMEOUT_60s but were not directly retested at
600s; bazel-CI envelope status is uncertain.

> **M3 enrollment note (2026-05-23 circom bump):** The 43/46 totals and the
> per-family counts in the paragraph below describe the m3 wiring prior to the
> bump. With the new circom: keccakfRound0_test takes ~305s and
> keccakfRound20_test ~84s to compile (still pass — slower but within bazel CI
> envelope). `Webb-tools/batchMerkleTreeUpdate_64` exceeds the 600s envelope on
> the build-server (see § Failure Analysis) — m3 enrollment status depends on
> the gpu-server's compile time and may need a follow-up edit to
> bench/m3/BUILD.bazel. The 4 newly-Circom→LLZK passing chips (`gfmul_int_test`,
> `polyval_test`, `ecdh_test`, `publicKey_test`) are not yet enrolled in m3 (no
> committed fixtures).

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

### Circom -> LLZK Failures (73 circuits)

All 73 failing chips share a common failure mode: circom's `llzk_backend/` Rust
crate emits no diagnostic and takes longer than the report's sampling envelope
(60s/chip fast or 600s/chip extended) to produce LLZK IR. Zero chips fail at
parse, zero chips error with "Conflicting types to read array", zero chips
panic. This is the post-upstream-fix state — see § Compile-time stretch for the
per-sample timing and § Upstream resolution log below for the parse-fixes
history.

**Composition diff vs the prior measurement** (2026-04-28 baseline, 47/76 under
the predecessor circom at `dev/handle_concrete_mixed 9b084a6d`):

| Direction | Count | Chips                                                                                                                                   |
| --------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Net gain  | +3    | 50 pass vs 47                                                                                                                           |
| Unblocked | +4    | `aes-circom/gfmul_int_test`, `aes-circom/polyval_test`, `maci/ecdh_test`, `maci/publicKey_test` — were `Conflicting types`, now compile |
| Regressed | −1    | `Webb-tools/batchMerkleTreeUpdate_64` — was pass, now TIMEOUT_600s on the build-server (gpu-server timing not yet sampled)              |

The 4 unblocked chips confirm that upstream
[#381](https://github.com/project-llzk/circom/pull/381) together with
[#398](https://github.com/project-llzk/circom/pull/398) cleared the prior
blockers structurally. The 1 regression is wall-clock-only: same parse layer,
but the LLZK-backend code path Webb_64 reaches is heavier on the new compiler
than on the predecessor.

### Compile-time stretch (sample-verified)

The 73 remaining failing chips share a common signature: circom emits no
diagnostic, no panic, no error; the binary runs at ~100% CPU on a single thread
and exceeds whichever wall-clock envelope was tested. Sample verification on
representative chips:

| Chip                                                            | 60s outcome | Extended-envelope outcome | Notes                                        |
| --------------------------------------------------------------- | ----------- | ------------------------- | -------------------------------------------- |
| `PointCompress`                                                 | TIMEOUT_60s | PASS (~140s)              | Moved back to passing                        |
| `keccak256-circom/keccakfRound0_test`                           | TIMEOUT_60s | PASS (~305s)              | Already in passing — slower under new circom |
| `keccak256-circom/keccakfRound20_test`                          | TIMEOUT_60s | PASS (~84s)               | Already in passing — slower under new circom |
| `Webb-tools/batchMerkleTreeUpdate_16`                           | TIMEOUT_60s | TIMEOUT_600s              | Verified on build-server                     |
| `Webb-tools/batchMerkleTreeUpdate_32`                           | TIMEOUT_60s | TIMEOUT_600s              | Verified on build-server                     |
| `Webb-tools/batchMerkleTreeUpdate_64`                           | TIMEOUT_60s | TIMEOUT_600s              | Regressed from prior pass on build-server    |
| `aes-circom/gcm_siv_dec_2_keys_test`                            | TIMEOUT_60s | TIMEOUT_600s              | Verified on build-server                     |
| `aes-circom/gcm_siv_enc_2_keys_test`                            | TIMEOUT_60s | TIMEOUT_600s              | Verified on build-server                     |
| `hydra/{hydra-s1, verify-hydra-commitment, verify-merkle-path}` | TIMEOUT_60s | TIMEOUT_600s              | Verified on build-server (all 3)             |
| Remaining 65 chips                                              | TIMEOUT_60s | not retested              | Bazel CI envelope status uncertain           |

Some chips clearly recover under a longer envelope (`PointCompress`,
`keccakfRound0_test`, `keccakfRound20_test`); some clearly don't
(`batchMerkleTreeUpdate_{16,32,64}`, both `gcm_siv_*` chips, all 3 hydra chips).
The 65 chips not directly retested are the doc's main remaining uncertainty —
their true bazel-CI-envelope status will require either sampling at higher
timeout on the build-server or running the actual `bazel test //examples/...`
against the new circom on a gpu-server.

**Affected circuit families** (all 73 in the extended-envelope failure bucket):

| Family           | Count | Description                                                          |
| ---------------- | ----- | -------------------------------------------------------------------- |
| maci             | 27    | Voting/state tree (Poseidon hash, Merkle trees); includes 2 batch256 |
|                  |       | variants that were `timeout (30s)` in the prior report               |
| iden3-core       | 18    | Identity (EdDSA signatures, SMT proofs, Poseidon)                    |
| Webb-tools       | 13    | batchMerkleTreeUpdate ×5, poseidon_vanchor ×4, vanchor_forest ×4     |
| keccak256-circom | 5     | absorb, final, keccakf, keccak\_{256_256,32_256}                     |
| zk-SQL           | 4     | SQL operations (Poseidon, comparison trees)                          |
| hydra            | 3     | Commitment verification (Poseidon, Merkle)                           |
| aes-circom       | 2     | AES-GCM SIV (`gcm_siv_dec_2_keys_test`, `gcm_siv_enc_2_keys_test`)   |
| semaphore        | 1     | Anonymous signaling (Poseidon, Merkle)                               |

Net: **the blocker has moved from parse-time errors into a compile-time
stretch**. Every chip now begins lowering through `llzk_backend/` after upstream
PR #381 + #398 removed the mixed-subcomponent panic and the
`Conflicting types to read array` reject. The new bottleneck is wall-clock-only
— no error message, no panic, just sustained 100% CPU. Most likely a
pathological recursion or O(N!) expansion path that the prior fast-fail at parse
had been masking.

**To reproduce** (example: `Webb-tools/batchMerkleTreeUpdate_16`):

```bash
git clone -b llzk https://github.com/project-llzk/circom.git
cd circom && git checkout fc46d662  # post-PR #432
# Build per third_party/circom/workspace.bzl
export CIRCOM_PATH="$PWD/target/release/circom"

git clone https://github.com/project-llzk/circom-benchmarks.git
cd circom-benchmarks && git checkout 6897550c

time timeout 600 circom \
  applications/Webb-tools/test/batchMerkleTreeUpdate_16.circom \
  --llzk concrete -o /tmp/out/ \
  -l applications/Webb-tools/test \
  -l libs -l libs/circomlib/circuits
# Expected: exit code 124 after exactly 10:00.00 wall.
# stderr is empty; circom has produced no LLZK output.
```

### Upstream resolution log

The two-layer parse-time blocker stack that gated this section between
2026-04-28 and 2026-05-04 is now closed at the parse level. Recorded here for
reference; the issues no longer affect this measurement.

1. **`template_ext.rs:243` panic** — "not yet implemented: Support mixed type
   subcomponent instantiations". Resolved by
   [project-llzk/circom#376](https://github.com/project-llzk/circom/pull/376)
   and landed in build `9b084a6d`.
1. **`Conflicting types to read array`** on parameterized component arrays
   (`inner[i] = Inner(i)` patterns). Tracked at
   [project-llzk/circom#386](https://github.com/project-llzk/circom/issues/386).
   Resolved by [PR #398](https://github.com/project-llzk/circom/pull/398) merged
   2026-05-01; issue closed 2026-05-04.
1. **`poly.template` wrapping migration** —
   [PR #378](https://github.com/project-llzk/circom/pull/378) and
   [#381](https://github.com/project-llzk/circom/pull/381) — locally
   accommodated in `SimplifySubComponents`'s `flattenSingleEntityWrapperModules`
   pass.

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

## Passing Circuits (50)

All 50 circuits pass the complete pipeline: Circom -> LLZK -> StableHLO ->
Batch(N=4).

### By Category

| Category          | Count | Circuits                                                                                                                                                                                                                      |
| ----------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arithmetic/Logic  | 5     | BinSum, Decoder, fulladder, onlycarry, FpMultiply                                                                                                                                                                             |
| Bit manipulation  | 3     | Num2Bits, Num2BitsCheck, BitElementMulAny                                                                                                                                                                                     |
| Comparison        | 2     | LessThanBounded, SignedFpCarryModP                                                                                                                                                                                            |
| Elliptic curve    | 5     | Edwards2Montgomery, Montgomery2Edwards, MontgomeryAdd, MontgomeryDouble, Window4                                                                                                                                              |
| EC fixed-base mul | 1     | WindowMulFix                                                                                                                                                                                                                  |
| EC point compress | 1     | PointCompress                                                                                                                                                                                                                 |
| AES encryption    | 6     | aes_256_ctr_test, aes_256_encrypt_test, aes_256_key_expansion_test, mul_test, gfmul_int_test, polyval_test                                                                                                                    |
| Keccak primitives | 9     | chi_test, iota3_test, iota10_test, keccakfRound0_test, keccakfRound20_test, pad_test, rhopi_test, squeeze_test, theta_test                                                                                                    |
| Iden3 utilities   | 10    | inTest, queryTest, utils_GetValueByIndex, utils_getClaimExpiration, utils_getClaimSubjectOtherIden, utils_getSubjectLocation, utils_isExpirable, utils_isUpdatable, utils_verifyCredentialSubject, utils_verifyExpirationTime |
| MACI utilities    | 7     | calculateTotal_test, decrypt_test, ecdh_test, publicKey_test, quinGeneratePathIndices_test, quinSelector_test, splicer_test                                                                                                   |
| Hash (AES subset) | 1     | EmulatedAesencSubstituteBytes                                                                                                                                                                                                 |

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
| 16  | PointCompress/src/main.circom                        | PASS | PASS      | PASS  |
| 17  | aes-circom/src/aes_256_ctr_test.circom               | PASS | PASS      | PASS  |
| 18  | aes-circom/src/aes_256_encrypt_test.circom           | PASS | PASS      | PASS  |
| 19  | aes-circom/src/aes_256_key_expansion_test.circom     | PASS | PASS      | PASS  |
| 20  | aes-circom/src/gfmul_int_test.circom                 | PASS | PASS      | PASS  |
| 21  | aes-circom/src/mul_test.circom                       | PASS | PASS      | PASS  |
| 22  | aes-circom/src/polyval_test.circom                   | PASS | PASS      | PASS  |
| 23  | fulladder/src/main.circom                            | PASS | PASS      | PASS  |
| 24  | iden3-core/src/inTest.circom                         | PASS | PASS      | PASS  |
| 25  | iden3-core/src/queryTest.circom                      | PASS | PASS      | PASS  |
| 26  | iden3-core/src/utils_GetValueByIndex.circom          | PASS | PASS      | PASS  |
| 27  | iden3-core/src/utils_getClaimExpiration.circom       | PASS | PASS      | PASS  |
| 28  | iden3-core/src/utils_getClaimSubjectOtherIden.circom | PASS | PASS      | PASS  |
| 29  | iden3-core/src/utils_getSubjectLocation.circom       | PASS | PASS      | PASS  |
| 30  | iden3-core/src/utils_isExpirable.circom              | PASS | PASS      | PASS  |
| 31  | iden3-core/src/utils_isUpdatable.circom              | PASS | PASS      | PASS  |
| 32  | iden3-core/src/utils_verifyCredentialSubject.circom  | PASS | PASS      | PASS  |
| 33  | iden3-core/src/utils_verifyExpirationTime.circom     | PASS | PASS      | PASS  |
| 34  | keccak256-circom/src/chi_test.circom                 | PASS | PASS      | PASS  |
| 35  | keccak256-circom/src/iota10_test.circom              | PASS | PASS      | PASS  |
| 36  | keccak256-circom/src/iota3_test.circom               | PASS | PASS      | PASS  |
| 37  | keccak256-circom/src/keccakfRound0_test.circom       | PASS | PASS      | PASS  |
| 38  | keccak256-circom/src/keccakfRound20_test.circom      | PASS | PASS      | PASS  |
| 39  | keccak256-circom/src/pad_test.circom                 | PASS | PASS      | PASS  |
| 40  | keccak256-circom/src/rhopi_test.circom               | PASS | PASS      | PASS  |
| 41  | keccak256-circom/src/squeeze_test.circom             | PASS | PASS      | PASS  |
| 42  | keccak256-circom/src/theta_test.circom               | PASS | PASS      | PASS  |
| 43  | maci/src/calculateTotal_test.circom                  | PASS | PASS      | PASS  |
| 44  | maci/src/decrypt_test.circom                         | PASS | PASS      | PASS  |
| 45  | maci/src/ecdh_test.circom                            | PASS | PASS      | PASS  |
| 46  | maci/src/publicKey_test.circom                       | PASS | PASS      | PASS  |
| 47  | maci/src/quinGeneratePathIndices_test.circom         | PASS | PASS      | PASS  |
| 48  | maci/src/quinSelector_test.circom                    | PASS | PASS      | PASS  |
| 49  | maci/src/splicer_test.circom                         | PASS | PASS      | PASS  |
| 50  | onlycarry/src/main.circom                            | PASS | PASS      | PASS  |

______________________________________________________________________

## Failing Circuits (73)

### Circom -> LLZK extended-envelope failures (73)

The Outcome column tags each chip:

- `verified-600s` — directly measured at 600s timeout in this session and
  exceeded the budget.
- `TIMEOUT_60s, retest pending` — only measured at the 60s fast envelope; the
  bazel-CI envelope may still admit it as passing (cf. `keccakfRound0_test` and
  `keccakfRound20_test` which pass at ~84–305s and have been moved to § Passing
  Circuits).
- For the 2 maci batch256 chips, an earlier `timeout (30s)` flag from the prior
  report is preserved alongside.

| #   | Circuit                                                     | Stage  | Outcome                                                |
| --- | ----------------------------------------------------------- | ------ | ------------------------------------------------------ |
| 1   | Webb-tools/test/batchMerkleTreeUpdate_4.circom              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 2   | Webb-tools/test/batchMerkleTreeUpdate_8.circom              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 3   | Webb-tools/test/batchMerkleTreeUpdate_16.circom             | CIRCOM | TIMEOUT_600s verified-600s                             |
| 4   | Webb-tools/test/batchMerkleTreeUpdate_32.circom             | CIRCOM | TIMEOUT_600s verified-600s                             |
| 5   | Webb-tools/test/batchMerkleTreeUpdate_64.circom             | CIRCOM | TIMEOUT_600s verified-600s (regressed from prior pass) |
| 6   | Webb-tools/test/poseidon_vanchor_2_2.circom                 | CIRCOM | TIMEOUT_60s, retest pending                            |
| 7   | Webb-tools/test/poseidon_vanchor_2_8.circom                 | CIRCOM | TIMEOUT_60s, retest pending                            |
| 8   | Webb-tools/test/poseidon_vanchor_16_2.circom                | CIRCOM | TIMEOUT_60s, retest pending                            |
| 9   | Webb-tools/test/poseidon_vanchor_16_8.circom                | CIRCOM | TIMEOUT_60s, retest pending                            |
| 10  | Webb-tools/test/vanchor_forest_2_2.circom                   | CIRCOM | TIMEOUT_60s, retest pending                            |
| 11  | Webb-tools/test/vanchor_forest_2_8.circom                   | CIRCOM | TIMEOUT_60s, retest pending                            |
| 12  | Webb-tools/test/vanchor_forest_16_2.circom                  | CIRCOM | TIMEOUT_60s, retest pending                            |
| 13  | Webb-tools/test/vanchor_forest_16_8.circom                  | CIRCOM | TIMEOUT_60s, retest pending                            |
| 14  | aes-circom/src/gcm_siv_dec_2_keys_test.circom               | CIRCOM | TIMEOUT_600s verified-600s                             |
| 15  | aes-circom/src/gcm_siv_enc_2_keys_test.circom               | CIRCOM | TIMEOUT_600s verified-600s                             |
| 16  | hydra/src/hydra-s1.circom                                   | CIRCOM | TIMEOUT_600s verified-600s                             |
| 17  | hydra/src/verify-hydra-commitment.circom                    | CIRCOM | TIMEOUT_600s verified-600s                             |
| 18  | hydra/src/verify-merkle-path.circom                         | CIRCOM | TIMEOUT_600s verified-600s                             |
| 19  | iden3-core/src/auth.circom                                  | CIRCOM | TIMEOUT_60s, retest pending                            |
| 20  | iden3-core/src/authTest.circom                              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 21  | iden3-core/src/authWithRelayTest.circom                     | CIRCOM | TIMEOUT_60s, retest pending                            |
| 22  | iden3-core/src/credentialAtomicQueryMTP.circom              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 23  | iden3-core/src/credentialAtomicQueryMTPTest.circom          | CIRCOM | TIMEOUT_60s, retest pending                            |
| 24  | iden3-core/src/credentialAtomicQueryMTPWithRelay.circom     | CIRCOM | TIMEOUT_60s, retest pending                            |
| 25  | iden3-core/src/credentialAtomicQueryMTPWithRelayTest.circom | CIRCOM | TIMEOUT_60s, retest pending                            |
| 26  | iden3-core/src/credentialAtomicQuerySig.circom              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 27  | iden3-core/src/credentialAtomicQuerySigTest.circom          | CIRCOM | TIMEOUT_60s, retest pending                            |
| 28  | iden3-core/src/idOwnershipBySignatureTest.circom            | CIRCOM | TIMEOUT_60s, retest pending                            |
| 29  | iden3-core/src/idOwnershipBySignatureWithRelayTest.circom   | CIRCOM | TIMEOUT_60s, retest pending                            |
| 30  | iden3-core/src/poseidon.circom                              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 31  | iden3-core/src/poseidon14.circom                            | CIRCOM | TIMEOUT_60s, retest pending                            |
| 32  | iden3-core/src/poseidon16.circom                            | CIRCOM | TIMEOUT_60s, retest pending                            |
| 33  | iden3-core/src/stateTransition.circom                       | CIRCOM | TIMEOUT_60s, retest pending                            |
| 34  | iden3-core/src/stateTransitionTest.circom                   | CIRCOM | TIMEOUT_60s, retest pending                            |
| 35  | iden3-core/src/utils_checkIdenStateMatchesRoots.circom      | CIRCOM | TIMEOUT_60s, retest pending                            |
| 36  | iden3-core/src/utils_verifyClaimSignature.circom            | CIRCOM | TIMEOUT_60s, retest pending                            |
| 37  | keccak256-circom/src/absorb_test.circom                     | CIRCOM | TIMEOUT_60s, retest pending                            |
| 38  | keccak256-circom/src/final_test.circom                      | CIRCOM | TIMEOUT_60s, retest pending                            |
| 39  | keccak256-circom/src/keccak_256_256_test.circom             | CIRCOM | TIMEOUT_60s, retest pending                            |
| 40  | keccak256-circom/src/keccak_32_256_test.circom              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 41  | keccak256-circom/src/keccakf_test.circom                    | CIRCOM | TIMEOUT_60s, retest pending                            |
| 42  | maci/src/batchUpdateStateTree_32.circom                     | CIRCOM | TIMEOUT_60s, retest pending                            |
| 43  | maci/src/batchUpdateStateTree_32_batch16.circom             | CIRCOM | TIMEOUT_60s, retest pending                            |
| 44  | maci/src/batchUpdateStateTree_32_batch256.circom            | CIRCOM | TIMEOUT_60s, retest pending (was 30s in prior report)  |
| 45  | maci/src/batchUpdateStateTree_large.circom                  | CIRCOM | TIMEOUT_60s, retest pending                            |
| 46  | maci/src/batchUpdateStateTree_medium.circom                 | CIRCOM | TIMEOUT_60s, retest pending                            |
| 47  | maci/src/batchUpdateStateTree_small.circom                  | CIRCOM | TIMEOUT_60s, retest pending                            |
| 48  | maci/src/batchUpdateStateTree_test.circom                   | CIRCOM | TIMEOUT_60s, retest pending                            |
| 49  | maci/src/hasher11_test.circom                               | CIRCOM | TIMEOUT_60s, retest pending                            |
| 50  | maci/src/hasher5_test.circom                                | CIRCOM | TIMEOUT_60s, retest pending                            |
| 51  | maci/src/hashleftright_test.circom                          | CIRCOM | TIMEOUT_60s, retest pending                            |
| 52  | maci/src/merkleTreeCheckRoot_test.circom                    | CIRCOM | TIMEOUT_60s, retest pending                            |
| 53  | maci/src/merkleTreeInclusionProof_test.circom               | CIRCOM | TIMEOUT_60s, retest pending                            |
| 54  | maci/src/merkleTreeLeafExists_test.circom                   | CIRCOM | TIMEOUT_60s, retest pending                            |
| 55  | maci/src/performChecksBeforeUpdate_test.circom              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 56  | maci/src/quadVoteTally_32.circom                            | CIRCOM | TIMEOUT_60s, retest pending                            |
| 57  | maci/src/quadVoteTally_32_batch16.circom                    | CIRCOM | TIMEOUT_60s, retest pending                            |
| 58  | maci/src/quadVoteTally_32_batch256.circom                   | CIRCOM | TIMEOUT_60s, retest pending (was 30s in prior report)  |
| 59  | maci/src/quadVoteTally_large.circom                         | CIRCOM | TIMEOUT_60s, retest pending                            |
| 60  | maci/src/quadVoteTally_medium.circom                        | CIRCOM | TIMEOUT_60s, retest pending                            |
| 61  | maci/src/quadVoteTally_small.circom                         | CIRCOM | TIMEOUT_60s, retest pending                            |
| 62  | maci/src/quadVoteTally_test.circom                          | CIRCOM | TIMEOUT_60s, retest pending                            |
| 63  | maci/src/quinTreeCheckRoot_test.circom                      | CIRCOM | TIMEOUT_60s, retest pending                            |
| 64  | maci/src/quinTreeInclusionProof_test.circom                 | CIRCOM | TIMEOUT_60s, retest pending                            |
| 65  | maci/src/quinTreeLeafExists_test.circom                     | CIRCOM | TIMEOUT_60s, retest pending                            |
| 66  | maci/src/resultCommitmentVerifier_test.circom               | CIRCOM | TIMEOUT_60s, retest pending                            |
| 67  | maci/src/updateStateTree_test.circom                        | CIRCOM | TIMEOUT_60s, retest pending                            |
| 68  | maci/src/verifySignature_test.circom                        | CIRCOM | TIMEOUT_60s, retest pending                            |
| 69  | semaphore/src/semaphore.circom                              | CIRCOM | TIMEOUT_60s, retest pending                            |
| 70  | zk-SQL/src/delete.circom                                    | CIRCOM | TIMEOUT_60s, retest pending                            |
| 71  | zk-SQL/src/insert.circom                                    | CIRCOM | TIMEOUT_60s, retest pending                            |
| 72  | zk-SQL/src/select.circom                                    | CIRCOM | TIMEOUT_60s, retest pending                            |
| 73  | zk-SQL/src/update.circom                                    | CIRCOM | TIMEOUT_60s, retest pending                            |
