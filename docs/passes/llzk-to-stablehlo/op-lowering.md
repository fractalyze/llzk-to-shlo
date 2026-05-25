# Type and op lowering

All ZK data becomes tensors with a `!field.pf<prime>` element type. The type
converter (`TypeConversion.cpp`) and op patterns (`FeltPatterns.cpp`,
`StructPatterns.cpp`, `ArrayPatterns.cpp`, `RemovalPatterns.cpp`) make that
mapping concrete.

## Type conversion

| LLZK Type               | StableHLO Type    | Rule                           |
| ----------------------- | ----------------- | ------------------------------ |
| `!felt.type`            | `tensor<!pf>`     | 0-d scalar tensor              |
| `!array.type<N x felt>` | `tensor<N x !pf>` | shape preserved                |
| `!struct.type<@Name>`   | `tensor<K x !pf>` | flatten — K = total felt count |
| `!pod.type<...>`        | (removed)         | eliminated by SSC pre-pass     |
| `i1`                    | `tensor<i1>`      | while carry only               |

**Struct flattening.** Each felt member occupies 1 slot; each `array<N>` member
occupies N slots; nested structs are recursively flattened. Order matches the
`registerStructFieldOffsets` pre-pass output, which drives all
`struct.readm`/`struct.writem` offset calculations. The recursion in
`getMemberFlatSize` is load-bearing: for `<N x !struct<Inner>>` it returns
`N × recursive_flat(Inner)`, where only writem-targeted, non-pod members
contribute to the footprint.

**i1 carry.** `llzk.nondet : i1` in a `scf.while` carry is a pre-pass concern: a
dedicated step replaces these with `arith.constant false` before the type
converter sees them, so `tensor<i1>` is the only i1 shape the conversion
patterns handle.

**Pod removal.** Pod types have no StableHLO equivalent. They are guaranteed
gone by `SimplifySubComponents` before `LlzkToStablehlo` starts. Any surviving
pod type at conversion time is a bug in SSC, not a conversion-time decision.

## Operation patterns

### Arithmetic (1:1)

| LLZK           | StableHLO                                     |
| -------------- | --------------------------------------------- |
| `felt.add`     | `stablehlo.add`                               |
| `felt.sub`     | `stablehlo.subtract`                          |
| `felt.mul`     | `stablehlo.multiply`                          |
| `felt.div`     | `stablehlo.divide`                            |
| `felt.neg`     | `stablehlo.negate`                            |
| `felt.inv %a`  | `stablehlo.divide(1, %a)`                     |
| `felt.const N` | `stablehlo.constant dense<N>`                 |
| `felt.pow`     | `stablehlo.power` (exponent converted to i32) |

Note that `felt.div` is field division (modular inverse), while `felt.uintdiv`
(below) is integer division. Both map to `stablehlo.divide` but via different
type levels — field division stays in `!pf`; integer division goes through the
3-step convert-operate-convert pattern.

### Bitwise and integer division (3-step)

`!field.pf` carries no bitwise semantics. To perform a bitwise op, the pass
converts to integer, operates, then converts back:

```
%a_i32 = stablehlo.convert %a : tensor<!pf> -> tensor<i32>
%r_i32 = stablehlo.and %a_i32, %b_i32
%r     = stablehlo.convert %r_i32 : tensor<i32> -> tensor<!pf>
```

The same 3-step pattern applies to: `felt.shr` → `shift_right_logical`,
`felt.shl` → `shift_left`, `felt.bit_or` → `or`, `felt.bit_xor` → `xor`,
`felt.umod` → `remainder`, `felt.uintdiv` → `divide`. All go through `i32`
because `!field.pf`'s defined range is `[0, p)` with `p < 2^32` for BabyBear and
`p < 2^254` for BN254 — the conversion to `i32` is only correct for BabyBear.
For BN254 circuits the only bitwise ops that appear in practice are in
sub-circuits whose inputs are provably small.

### Struct operations (offset-based slicing)

`struct.new` becomes a zero tensor (`stablehlo.constant dense<0>`).
`struct.readm @F` becomes a `stablehlo.slice` at the precomputed offset plus a
`stablehlo.reshape` to scalar. `struct.writem @F = %v` becomes
`stablehlo.dynamic_update_slice` — an SSA chain where each write produces a new
tensor and each subsequent write uses the previous result.

The SSA chain is what makes struct mutation expressible in StableHLO's purely
functional IR. LLZK's `struct.writem` is void (in-place mutation); the pre-pass
`convertWritemToSSA` rewrites the sequence into a chain before the patterns
fire.

### Array operations

`array.read %arr[%i]` becomes `stablehlo.dynamic_slice` (index converted from
`!pf` to `i32`) plus `stablehlo.reshape` to scalar. `array.write` becomes
`stablehlo.dynamic_update_slice`, also in SSA chain form established by the
pre-pass.

### Boolean and comparison

| LLZK                       | StableHLO                      |
| -------------------------- | ------------------------------ |
| `bool.cmp lt(%a, %b)`      | `stablehlo.compare LT`         |
| `bool.and`                 | `stablehlo.and`                |
| `bool.or`                  | `stablehlo.or`                 |
| `bool.not %a : i1`         | `arith.xori %a, true` (scalar) |
| `bool.not %a : tensor<i1>` | `stablehlo.not %a` (tensor)    |

### Control flow

`scf.if` → `stablehlo.select` (post-pass, not a 1:1 pattern — see
`while-loop-transformation.md`). `function.call @A::@compute` →
`func.call @A_compute` (name flattened by the `convertAllFunctions` pre-pass).
`cast.toindex` → `stablehlo.convert` (field → i32) with a constant-preservation
invariant described below.

### Removed operations

| LLZK op                       | Handling                      | Why                                     |
| ----------------------------- | ----------------------------- | --------------------------------------- |
| `struct.def`, `struct.member` | Deleted                       | Only needed for offset calculation      |
| `function.def @constrain`     | Deleted                       | Witness gen does not verify constraints |
| `constrain.eq`                | Deleted                       | Verification is the prover's job        |
| `bool.assert`                 | Deleted                       | No runtime alarm in GPU code            |
| `felt.nondet`, `llzk.nondet`  | `stablehlo.constant dense<0>` | Zero-initialized (see invariant below)  |

`@constrain` erasure is the sharpest correctness boundary in the pipeline: the
GPU computes witnesses with no internal alarm. A miscompile in a lowering
pattern produces wrong witnesses silently. The m3 byte-equality gate against
circom's C++ output is the only catch. See
[`../../contracts/correctness-gate.md`](../../contracts/correctness-gate.md).

### `cast.toindex` constant preservation

`cast.toindex` converts a felt constant to an array index. The pattern extracts
the actual integer value from the defining `felt.const` op instead of emitting a
fallback of 0. If the fallback fires, every array write driven by a constant
index silently writes to position 0 instead of the intended slot — a silent
miscompile producing no IR-level error.

The APInt extraction requires `APInt::zextOrTrunc(storageWidth)`, not
`getSExtValue()`. `getSExtValue()` is UB for `getBitWidth() > 64` (truncates the
low 64 bits), which is triggered by the `1 << 252` LessThan offset constant on
BN254. See
[`../../development/apint-arithmetic.md`](../../development/apint-arithmetic.md).

## Lowering equivalences

**`array.new` and `llzk.nondet` both lower to `dense<0>`.** `ArrayNewPattern` in
`ArrayPatterns.cpp` and `LlzkNonDetPattern` in `RemovalPatterns.cpp` both call
`tc.createConstantAttr(tensorType, 0, rewriter)`. This is safe because LLZK's
semantics permit either as an uninitialized placeholder — the circuit is
required to write every output position before using it. Switching one for the
other in `SimplifySubComponents` is a no-op at lowering time.

Before chasing an "orphan nondet vs array.new" refactor, verify the lowered
StableHLO actually differs. If the bytes are identical, the bug is downstream of
the lowering boundary — likely in witness-output assembly or in the
vectorization passes.

**Void mutation ops are SSA-ified in two places with separate filters.**
`processBlockForArrayMutations` handles mutations inside
`scf.while`/`scf.if`/`scf.execute_region` bodies; `convertWritemToSSA` handles
function-body-scope mutations. A divergence between the two allowlists is the
canonical "void mutation silently dropped" hazard: the lowering pattern creates
a fresh `dynamic_update_slice` for void inserts, but if upstream SSA-ification
didn't fire, downstream consumers still reference the unmodified array and the
new DUS gets DCE'd. When adding a new void mutation op, update both passes.
