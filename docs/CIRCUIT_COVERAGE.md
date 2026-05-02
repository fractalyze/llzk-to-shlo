# Circuit Coverage Report

Full pipeline analysis of all 123 entry points in
[circom-benchmarks](https://github.com/project-llzk/circom-benchmarks) (commit
6897550c).

## Summary

| Stage                     | Pass | Fail | Rate  |
| ------------------------- | ---- | ---- | ----- |
| Circom -> LLZK (concrete) | 46   | 77   | 37.4% |
| LLZK -> StableHLO         | 45   | 1    | 97.8% |
| StableHLO -> Batch (N=4)  | 45   | 0    | 100%  |

**End-to-end (Circom -> Batch): 45/123 (36.6%)**

The 77 Circom -> LLZK failures are gated by a two-layer upstream blocker stack
in the [llzk-circom](https://github.com/project-llzk/circom) frontend — not by
llzk-to-shlo. See § Failure Analysis below for details (re-validated
2026-04-28). Of circuits that successfully produce LLZK IR, **45/46 (97.8%)**
complete the full pipeline.

**M3 correctness gate**: 19 of the 45 end-to-end-passing circuits are wired into
`//bench/m3:m3_correctness_gate_test` and byte-equal `gpu_zkx` output against
the circom-native `.wtns` reference at N=1 on every PR (9 keccak step chips + 9
iden3 utility templates + MontgomeryDouble; AES family held out pending an
in-flight lowering fix). See [`M3_REPORT.md` §4.4](M3_REPORT.md) for the
per-circuit gate matrix and CLAUDE.md → "M3 correctness gate convention" for the
sentinel format.

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

### Circom -> LLZK Failures (77 circuits)

74 of the 77 circuits are gated upstream by a *two-layer* blocker stack in the
circom LLZK backend; the remaining 3 (`Webb-tools/batchMerkleTreeUpdate_64`,
`maci/batchUpdateStateTree_32_batch256`, `maci/quadVoteTally_32_batch256`) time
out before reaching either layer.

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

### LLZK -> StableHLO Failure (1 circuit)

| Circuit       | Error              | Details                                                                                                                                                                 |
| ------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PointCompress | Conversion timeout | 21K lines of LLZK IR (ed25519 point compression). The `SimplifySubComponents` pass does not terminate within reasonable time due to deeply nested sub-component chains. |

______________________________________________________________________

## Passing Circuits (45)

All 45 circuits pass the complete pipeline: Circom -> LLZK -> StableHLO ->
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

______________________________________________________________________

## Failing Circuits (78)

### LLZK Frontend Failures (77)

74 of these 77 circuits are gated upstream by the two-layer blocker stack — see
§ Failure Analysis; the remaining 3 time out before reaching either layer. The
Error column points back to that prose block.

| #   | Circuit                                                     | Stage  | Error                             |
| --- | ----------------------------------------------------------- | ------ | --------------------------------- |
| 1   | Webb-tools/test/batchMerkleTreeUpdate_4.circom              | CIRCOM | upstream — see § Failure Analysis |
| 2   | Webb-tools/test/batchMerkleTreeUpdate_8.circom              | CIRCOM | upstream — see § Failure Analysis |
| 3   | Webb-tools/test/batchMerkleTreeUpdate_16.circom             | CIRCOM | upstream — see § Failure Analysis |
| 4   | Webb-tools/test/batchMerkleTreeUpdate_32.circom             | CIRCOM | upstream — see § Failure Analysis |
| 5   | Webb-tools/test/batchMerkleTreeUpdate_64.circom             | CIRCOM | timeout (30s)                     |
| 6   | Webb-tools/test/poseidon_vanchor_2_2.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 7   | Webb-tools/test/poseidon_vanchor_2_8.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 8   | Webb-tools/test/poseidon_vanchor_16_2.circom                | CIRCOM | upstream — see § Failure Analysis |
| 9   | Webb-tools/test/poseidon_vanchor_16_8.circom                | CIRCOM | upstream — see § Failure Analysis |
| 10  | Webb-tools/test/vanchor_forest_2_2.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 11  | Webb-tools/test/vanchor_forest_2_8.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 12  | Webb-tools/test/vanchor_forest_16_2.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 13  | Webb-tools/test/vanchor_forest_16_8.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 14  | aes-circom/src/gcm_siv_dec_2_keys_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 15  | aes-circom/src/gcm_siv_enc_2_keys_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 16  | aes-circom/src/gfmul_int_test.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 17  | aes-circom/src/polyval_test.circom                          | CIRCOM | upstream — see § Failure Analysis |
| 18  | hydra/src/hydra-s1.circom                                   | CIRCOM | upstream — see § Failure Analysis |
| 19  | hydra/src/verify-hydra-commitment.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 20  | hydra/src/verify-merkle-path.circom                         | CIRCOM | upstream — see § Failure Analysis |
| 21  | iden3-core/src/auth.circom                                  | CIRCOM | upstream — see § Failure Analysis |
| 22  | iden3-core/src/authTest.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 23  | iden3-core/src/authWithRelayTest.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 24  | iden3-core/src/credentialAtomicQueryMTP.circom              | CIRCOM | upstream — see § Failure Analysis |
| 25  | iden3-core/src/credentialAtomicQueryMTPTest.circom          | CIRCOM | upstream — see § Failure Analysis |
| 26  | iden3-core/src/credentialAtomicQueryMTPWithRelay.circom     | CIRCOM | upstream — see § Failure Analysis |
| 27  | iden3-core/src/credentialAtomicQueryMTPWithRelayTest.circom | CIRCOM | upstream — see § Failure Analysis |
| 28  | iden3-core/src/credentialAtomicQuerySig.circom              | CIRCOM | upstream — see § Failure Analysis |
| 29  | iden3-core/src/credentialAtomicQuerySigTest.circom          | CIRCOM | upstream — see § Failure Analysis |
| 30  | iden3-core/src/idOwnershipBySignatureTest.circom            | CIRCOM | upstream — see § Failure Analysis |
| 31  | iden3-core/src/idOwnershipBySignatureWithRelayTest.circom   | CIRCOM | upstream — see § Failure Analysis |
| 32  | iden3-core/src/poseidon.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 33  | iden3-core/src/poseidon14.circom                            | CIRCOM | upstream — see § Failure Analysis |
| 34  | iden3-core/src/poseidon16.circom                            | CIRCOM | upstream — see § Failure Analysis |
| 35  | iden3-core/src/stateTransition.circom                       | CIRCOM | upstream — see § Failure Analysis |
| 36  | iden3-core/src/stateTransitionTest.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 37  | iden3-core/src/utils_checkIdenStateMatchesRoots.circom      | CIRCOM | upstream — see § Failure Analysis |
| 38  | iden3-core/src/utils_verifyClaimSignature.circom            | CIRCOM | upstream — see § Failure Analysis |
| 39  | keccak256-circom/src/absorb_test.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 40  | keccak256-circom/src/final_test.circom                      | CIRCOM | upstream — see § Failure Analysis |
| 41  | keccak256-circom/src/keccak_256_256_test.circom             | CIRCOM | upstream — see § Failure Analysis |
| 42  | keccak256-circom/src/keccak_32_256_test.circom              | CIRCOM | upstream — see § Failure Analysis |
| 43  | keccak256-circom/src/keccakf_test.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 44  | maci/src/batchUpdateStateTree_32.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 45  | maci/src/batchUpdateStateTree_32_batch16.circom             | CIRCOM | upstream — see § Failure Analysis |
| 46  | maci/src/batchUpdateStateTree_32_batch256.circom            | CIRCOM | timeout (30s)                     |
| 47  | maci/src/batchUpdateStateTree_large.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 48  | maci/src/batchUpdateStateTree_medium.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 49  | maci/src/batchUpdateStateTree_small.circom                  | CIRCOM | upstream — see § Failure Analysis |
| 50  | maci/src/batchUpdateStateTree_test.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 51  | maci/src/ecdh_test.circom                                   | CIRCOM | upstream — see § Failure Analysis |
| 52  | maci/src/hasher11_test.circom                               | CIRCOM | upstream — see § Failure Analysis |
| 53  | maci/src/hasher5_test.circom                                | CIRCOM | upstream — see § Failure Analysis |
| 54  | maci/src/hashleftright_test.circom                          | CIRCOM | upstream — see § Failure Analysis |
| 55  | maci/src/merkleTreeCheckRoot_test.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 56  | maci/src/merkleTreeInclusionProof_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 57  | maci/src/merkleTreeLeafExists_test.circom                   | CIRCOM | upstream — see § Failure Analysis |
| 58  | maci/src/performChecksBeforeUpdate_test.circom              | CIRCOM | upstream — see § Failure Analysis |
| 59  | maci/src/publicKey_test.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 60  | maci/src/quadVoteTally_32.circom                            | CIRCOM | upstream — see § Failure Analysis |
| 61  | maci/src/quadVoteTally_32_batch16.circom                    | CIRCOM | upstream — see § Failure Analysis |
| 62  | maci/src/quadVoteTally_32_batch256.circom                   | CIRCOM | timeout (30s)                     |
| 63  | maci/src/quadVoteTally_large.circom                         | CIRCOM | upstream — see § Failure Analysis |
| 64  | maci/src/quadVoteTally_medium.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 65  | maci/src/quadVoteTally_small.circom                         | CIRCOM | upstream — see § Failure Analysis |
| 66  | maci/src/quadVoteTally_test.circom                          | CIRCOM | upstream — see § Failure Analysis |
| 67  | maci/src/quinTreeCheckRoot_test.circom                      | CIRCOM | upstream — see § Failure Analysis |
| 68  | maci/src/quinTreeInclusionProof_test.circom                 | CIRCOM | upstream — see § Failure Analysis |
| 69  | maci/src/quinTreeLeafExists_test.circom                     | CIRCOM | upstream — see § Failure Analysis |
| 70  | maci/src/resultCommitmentVerifier_test.circom               | CIRCOM | upstream — see § Failure Analysis |
| 71  | maci/src/updateStateTree_test.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 72  | maci/src/verifySignature_test.circom                        | CIRCOM | upstream — see § Failure Analysis |
| 73  | semaphore/src/semaphore.circom                              | CIRCOM | upstream — see § Failure Analysis |
| 74  | zk-SQL/src/delete.circom                                    | CIRCOM | upstream — see § Failure Analysis |
| 75  | zk-SQL/src/insert.circom                                    | CIRCOM | upstream — see § Failure Analysis |
| 76  | zk-SQL/src/select.circom                                    | CIRCOM | upstream — see § Failure Analysis |
| 77  | zk-SQL/src/update.circom                                    | CIRCOM | upstream — see § Failure Analysis |

### StableHLO Conversion Failure (1)

| #   | Circuit                       | Stage | Error                                         |
| --- | ----------------------------- | ----- | --------------------------------------------- |
| 78  | PointCompress/src/main.circom | SHLO  | Conversion timeout (21K-line ed25519 LLZK IR) |
