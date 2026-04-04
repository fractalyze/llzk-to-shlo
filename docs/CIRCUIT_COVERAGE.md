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

The 77 Circom -> LLZK failures are all caused by a single bug in the
[llzk-circom](https://github.com/project-llzk/circom) frontend — not by
llzk-to-shlo. Of circuits that successfully produce LLZK IR, **45/46 (97.8%)**
complete the full pipeline.

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

All 77 failures hit the same panic in the circom LLZK backend:

```
thread 'main' panicked at llzk_backend/src/template_ext.rs:219:25:
not yet implemented: Support mixed type subcomponent instantiations
```

**Root cause**: The circom LLZK backend (`--llzk concrete`) does not support
circuits that use **mixed-type sub-components** — where a parent template
instantiates a child template with different type parameters than its own (e.g.,
`Poseidon(4)` and `Poseidon(6)` in the same circuit). This is a frontend
limitation, not an llzk-to-shlo limitation.

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
# Expected: panicked at 'not yet implemented: Support mixed type subcomponent instantiations'
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

**Common pattern**: These circuits all use Poseidon hash, EdDSA, or Merkle tree
sub-components that are instantiated with varying parameter sizes. The concrete
LLZK mode requires all template parameters to be resolved at compile time, and
the backend panics when encountering certain mixed-parameter instantiation
patterns.

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
| Keccak primitives | 8     | chi_test, iota3_test, iota10_test, keccakfRound0_test, keccakfRound20_test, pad_test, rhopi_test, squeeze_test, theta_test                                                                                                    |
| Iden3 utilities   | 8     | inTest, queryTest, utils_GetValueByIndex, utils_getClaimExpiration, utils_getClaimSubjectOtherIden, utils_getSubjectLocation, utils_isExpirable, utils_isUpdatable, utils_verifyCredentialSubject, utils_verifyExpirationTime |
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

All fail with `template_ext.rs:219` panic in circom LLZK backend.

| #   | Circuit                                                     | Stage  | Error                     |
| --- | ----------------------------------------------------------- | ------ | ------------------------- |
| 1   | Webb-tools/test/batchMerkleTreeUpdate_4.circom              | CIRCOM | template_ext.rs:219 panic |
| 2   | Webb-tools/test/batchMerkleTreeUpdate_8.circom              | CIRCOM | template_ext.rs:219 panic |
| 3   | Webb-tools/test/batchMerkleTreeUpdate_16.circom             | CIRCOM | template_ext.rs:219 panic |
| 4   | Webb-tools/test/batchMerkleTreeUpdate_32.circom             | CIRCOM | template_ext.rs:219 panic |
| 5   | Webb-tools/test/batchMerkleTreeUpdate_64.circom             | CIRCOM | timeout (30s)             |
| 6   | Webb-tools/test/poseidon_vanchor_2_2.circom                 | CIRCOM | template_ext.rs:219 panic |
| 7   | Webb-tools/test/poseidon_vanchor_2_8.circom                 | CIRCOM | template_ext.rs:219 panic |
| 8   | Webb-tools/test/poseidon_vanchor_16_2.circom                | CIRCOM | template_ext.rs:219 panic |
| 9   | Webb-tools/test/poseidon_vanchor_16_8.circom                | CIRCOM | template_ext.rs:219 panic |
| 10  | Webb-tools/test/vanchor_forest_2_2.circom                   | CIRCOM | template_ext.rs:219 panic |
| 11  | Webb-tools/test/vanchor_forest_2_8.circom                   | CIRCOM | template_ext.rs:219 panic |
| 12  | Webb-tools/test/vanchor_forest_16_2.circom                  | CIRCOM | template_ext.rs:219 panic |
| 13  | Webb-tools/test/vanchor_forest_16_8.circom                  | CIRCOM | template_ext.rs:219 panic |
| 14  | aes-circom/src/gcm_siv_dec_2_keys_test.circom               | CIRCOM | template_ext.rs:219 panic |
| 15  | aes-circom/src/gcm_siv_enc_2_keys_test.circom               | CIRCOM | template_ext.rs:219 panic |
| 16  | aes-circom/src/gfmul_int_test.circom                        | CIRCOM | template_ext.rs:219 panic |
| 17  | aes-circom/src/polyval_test.circom                          | CIRCOM | template_ext.rs:219 panic |
| 18  | hydra/src/hydra-s1.circom                                   | CIRCOM | template_ext.rs:219 panic |
| 19  | hydra/src/verify-hydra-commitment.circom                    | CIRCOM | template_ext.rs:219 panic |
| 20  | hydra/src/verify-merkle-path.circom                         | CIRCOM | template_ext.rs:219 panic |
| 21  | iden3-core/src/auth.circom                                  | CIRCOM | template_ext.rs:219 panic |
| 22  | iden3-core/src/authTest.circom                              | CIRCOM | template_ext.rs:219 panic |
| 23  | iden3-core/src/authWithRelayTest.circom                     | CIRCOM | template_ext.rs:219 panic |
| 24  | iden3-core/src/credentialAtomicQueryMTP.circom              | CIRCOM | template_ext.rs:219 panic |
| 25  | iden3-core/src/credentialAtomicQueryMTPTest.circom          | CIRCOM | template_ext.rs:219 panic |
| 26  | iden3-core/src/credentialAtomicQueryMTPWithRelay.circom     | CIRCOM | template_ext.rs:219 panic |
| 27  | iden3-core/src/credentialAtomicQueryMTPWithRelayTest.circom | CIRCOM | template_ext.rs:219 panic |
| 28  | iden3-core/src/credentialAtomicQuerySig.circom              | CIRCOM | template_ext.rs:219 panic |
| 29  | iden3-core/src/credentialAtomicQuerySigTest.circom          | CIRCOM | template_ext.rs:219 panic |
| 30  | iden3-core/src/idOwnershipBySignatureTest.circom            | CIRCOM | template_ext.rs:219 panic |
| 31  | iden3-core/src/idOwnershipBySignatureWithRelayTest.circom   | CIRCOM | template_ext.rs:219 panic |
| 32  | iden3-core/src/poseidon.circom                              | CIRCOM | template_ext.rs:219 panic |
| 33  | iden3-core/src/poseidon14.circom                            | CIRCOM | template_ext.rs:219 panic |
| 34  | iden3-core/src/poseidon16.circom                            | CIRCOM | template_ext.rs:219 panic |
| 35  | iden3-core/src/stateTransition.circom                       | CIRCOM | template_ext.rs:219 panic |
| 36  | iden3-core/src/stateTransitionTest.circom                   | CIRCOM | template_ext.rs:219 panic |
| 37  | iden3-core/src/utils_checkIdenStateMatchesRoots.circom      | CIRCOM | template_ext.rs:219 panic |
| 38  | iden3-core/src/utils_verifyClaimSignature.circom            | CIRCOM | template_ext.rs:219 panic |
| 39  | keccak256-circom/src/absorb_test.circom                     | CIRCOM | template_ext.rs:219 panic |
| 40  | keccak256-circom/src/final_test.circom                      | CIRCOM | template_ext.rs:219 panic |
| 41  | keccak256-circom/src/keccak_256_256_test.circom             | CIRCOM | template_ext.rs:219 panic |
| 42  | keccak256-circom/src/keccak_32_256_test.circom              | CIRCOM | template_ext.rs:219 panic |
| 43  | keccak256-circom/src/keccakf_test.circom                    | CIRCOM | template_ext.rs:219 panic |
| 44  | maci/src/batchUpdateStateTree_32.circom                     | CIRCOM | template_ext.rs:219 panic |
| 45  | maci/src/batchUpdateStateTree_32_batch16.circom             | CIRCOM | template_ext.rs:219 panic |
| 46  | maci/src/batchUpdateStateTree_32_batch256.circom            | CIRCOM | timeout (30s)             |
| 47  | maci/src/batchUpdateStateTree_large.circom                  | CIRCOM | template_ext.rs:219 panic |
| 48  | maci/src/batchUpdateStateTree_medium.circom                 | CIRCOM | template_ext.rs:219 panic |
| 49  | maci/src/batchUpdateStateTree_small.circom                  | CIRCOM | template_ext.rs:219 panic |
| 50  | maci/src/batchUpdateStateTree_test.circom                   | CIRCOM | template_ext.rs:219 panic |
| 51  | maci/src/ecdh_test.circom                                   | CIRCOM | template_ext.rs:219 panic |
| 52  | maci/src/hasher11_test.circom                               | CIRCOM | template_ext.rs:219 panic |
| 53  | maci/src/hasher5_test.circom                                | CIRCOM | template_ext.rs:219 panic |
| 54  | maci/src/hashleftright_test.circom                          | CIRCOM | template_ext.rs:219 panic |
| 55  | maci/src/merkleTreeCheckRoot_test.circom                    | CIRCOM | template_ext.rs:219 panic |
| 56  | maci/src/merkleTreeInclusionProof_test.circom               | CIRCOM | template_ext.rs:219 panic |
| 57  | maci/src/merkleTreeLeafExists_test.circom                   | CIRCOM | template_ext.rs:219 panic |
| 58  | maci/src/performChecksBeforeUpdate_test.circom              | CIRCOM | template_ext.rs:219 panic |
| 59  | maci/src/publicKey_test.circom                              | CIRCOM | template_ext.rs:219 panic |
| 60  | maci/src/quadVoteTally_32.circom                            | CIRCOM | template_ext.rs:219 panic |
| 61  | maci/src/quadVoteTally_32_batch16.circom                    | CIRCOM | template_ext.rs:219 panic |
| 62  | maci/src/quadVoteTally_32_batch256.circom                   | CIRCOM | timeout (30s)             |
| 63  | maci/src/quadVoteTally_large.circom                         | CIRCOM | template_ext.rs:219 panic |
| 64  | maci/src/quadVoteTally_medium.circom                        | CIRCOM | template_ext.rs:219 panic |
| 65  | maci/src/quadVoteTally_small.circom                         | CIRCOM | template_ext.rs:219 panic |
| 66  | maci/src/quadVoteTally_test.circom                          | CIRCOM | template_ext.rs:219 panic |
| 67  | maci/src/quinTreeCheckRoot_test.circom                      | CIRCOM | template_ext.rs:219 panic |
| 68  | maci/src/quinTreeInclusionProof_test.circom                 | CIRCOM | template_ext.rs:219 panic |
| 69  | maci/src/quinTreeLeafExists_test.circom                     | CIRCOM | template_ext.rs:219 panic |
| 70  | maci/src/resultCommitmentVerifier_test.circom               | CIRCOM | template_ext.rs:219 panic |
| 71  | maci/src/updateStateTree_test.circom                        | CIRCOM | template_ext.rs:219 panic |
| 72  | maci/src/verifySignature_test.circom                        | CIRCOM | template_ext.rs:219 panic |
| 73  | semaphore/src/semaphore.circom                              | CIRCOM | template_ext.rs:219 panic |
| 74  | zk-SQL/src/delete.circom                                    | CIRCOM | template_ext.rs:219 panic |
| 75  | zk-SQL/src/insert.circom                                    | CIRCOM | template_ext.rs:219 panic |
| 76  | zk-SQL/src/select.circom                                    | CIRCOM | template_ext.rs:219 panic |
| 77  | zk-SQL/src/update.circom                                    | CIRCOM | template_ext.rs:219 panic |

### StableHLO Conversion Failure (1)

| #   | Circuit                       | Stage | Error                                         |
| --- | ----------------------------- | ----- | --------------------------------------------- |
| 78  | PointCompress/src/main.circom | SHLO  | Conversion timeout (21K-line ed25519 LLZK IR) |
