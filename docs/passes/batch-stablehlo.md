# BatchStablehlo

`BatchStablehlo` adds a leading batch dimension N to every tensor in the module,
transforming single-witness StableHLO into a form that processes N witnesses in
one GPU kernel launch. The pass takes one required parameter: `batch-size=N`.

## How it works

### Leading-dimension insertion

Every tensor type gains a prepended batch dimension:

| Before            | After (N)           |
| ----------------- | ------------------- |
| `tensor<!pf>`     | `tensor<Nx!pf>`     |
| `tensor<Mx!pf>`   | `tensor<NxMx!pf>`   |
| `tensor<MxKx!pf>` | `tensor<NxMxKx!pf>` |

### Implementation structure

The pass operates in two passes per function:

1. Update the function signature and block argument types.
1. **Pass 1**: Process non-constant ops — element-wise, reshape, slice, while
   bodies, calls, compare, select.
1. **Pass 2**: Broadcast constants — discriminating data constants (which must
   be broadcast to match the new batch dimension) from index constants (which
   must remain scalar).

The two-pass constant handling is the non-obvious bit: constants used as
`dynamic_slice` or `dynamic_update_slice` start indices must not be broadcast.
`isSliceIndexUse()` checks whether a constant flows into a slice index operand
and keeps those scalar; all other constants get `broadcast_in_dim`.

## Op-by-op rules

### Element-wise ops

Only the result type changes. StableHLO's broadcasting semantics propagate the
batch dimension through the operation automatically. Supported: `add`,
`subtract`, `multiply`, `divide`, `negate`, `power`, `abs`, `convert`, `rem`,
`and`, `or`, `xor`, `shift_left`, `shift_right_logical`, `not`.

### Constants → `broadcast_in_dim`

Data constants are kept at their original type and a `broadcast_in_dim` wraps
them to add the batch dimension. Index constants (identified by
`isSliceIndexUse()`) are not broadcast; they remain scalar because
`dynamic_slice`/`dynamic_update_slice` require scalar start indices.

### `dynamic_slice` / `dynamic_update_slice`

**Constant indices (common case).** Prepend a batch dim index of 0 to the
start-index list and expand `sizes` to `[N, original_sizes...]`. This handles
struct member reads and array element reads with statically-known offsets.

**Data-dependent indices (e.g., AES lookup tables).** When any index operand has
been batched (rank increased), `dynamic_slice` cannot accept per-batch indices.
The pass rewrites to a one-hot gather:

```
iota = [0, 1, ..., M-1]           // table indices
mask = compare(iota, idx, EQ)     // NxM boolean mask
selected = table * mask           // zero out non-selected elements
result = reduce_sum(dim=M_axis)   // collapse to N values
```

Detection: `hasAnyBatchedIndex()` checks whether any index operand has a batched
type after the signature update.

**Multiple data-dependent indices (e.g., AES 2D lookup).** When both row and
column indices are per-batch-element, the masks are ANDed before the reduce:

```
mask_row = compare(iota_row, row_idx, EQ)   // tensor<Nx4>
mask_col = compare(iota_col, col_idx, EQ)   // tensor<Nx256>
combined = and(mask_row, mask_col)           // tensor<Nx4x256>
result = reduce_sum(dim=1, dim=2)            // tensor<N>
```

### One-hot write (scatter)

`dynamic_update_slice` with data-dependent indices uses the same one-hot pattern
in reverse: `select(mask, broadcast(update_value), original_tensor)` writes the
update at the mask-true positions and keeps the original elsewhere.

### `stablehlo.while` (fixed trip count)

The batch dimension is added to all carry types. The condition predicate is
extracted from batch element `[0]` and reused for the whole batch — valid
because Circom while loops have compile-time trip counts and all batch elements
execute the same number of iterations. See the batching invariants below.

### `func.call`, compare, select

- `func.call`: the callee is also batched by the same pass, so only the result
  type changes.
- `compare`: result becomes `tensor<Nxi1>`.
- `select`: predicate and result both gain the batch dimension.

### LLZK residual ops

If pod dispatch was not fully eliminated before `BatchStablehlo` runs, residual
LLZK dialect ops (`pod.new`, `array.new`, etc.) may remain in the input. These
are dead code and are skipped by the batch pass. Their presence does not affect
correctness but indicates an incomplete SSC run.

## Batching invariants

**`BatchStablehlo` must see every signal — public outputs and private
internals.** The leading N dimension aliases the full per-witness state across N
proofs. Pruning to only public signals silently misaligns the batch axis;
downstream witness assembly reads from wrong offsets.

**Trip counts must be compile-time-constant.** The pass extracts the loop
predicate from element `[0]` and reuses it for all N batch elements. A frontend
emitting batch-divergent trip counts (where different inputs iterate different
numbers of times) would corrupt every batch element beyond the shortest run.
Circom's while loops always have statically-fixed trip counts; this invariant is
a property of the frontend, not something the pass verifies.

**Immutable Circom signals are what make vectorization sound.** The three
while-loop vectorization passes in `LlzkToStablehlo` — 1D element-wise, 2D carry
write, and nested-while inner — turn iterative loops into vector ops because
`out[i]` is written exactly once and never reassigned. A frontend that emitted
batch-mutable accumulators would break the vectorization pattern shape-check
even before batching; circom's signal semantics prevent this at the source
level.
