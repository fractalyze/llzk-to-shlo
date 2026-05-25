# APInt arithmetic traps

## `APInt::getSExtValue()` on a felt constant is a silent miscompile

**What goes wrong.** `getSExtValue()` is UB when `getBitWidth() > 64` — it
returns the low 64 bits silently. On bn128 circuits, `1 << 252` (the LessThan
comparator offset) truncates to `0`. The truncated value falls through the
comparator's sign-bit check and the witness output for that LessThan slot is
wrong.

Diagnostic:
`grep "value = dense<" <chip>.stablehlo.mlir | grep -v "dense<[0-9]>"` — bn128
circuits with comparators must emit at least one
`dense<7237005577332262213973186563042994240829374041602535252466099000494570602496>`
(= 2^252) per `LessThan`. Zero hits means a `getSExtValue()` call truncated it
upstream.

**Rule.** Use `APInt::zextOrTrunc(storageWidth)` — zero-extension is correct
because field elements are unsigned, ranged `[0, p)` with `p < 2^254`. Sister
site to audit: `convertToIndexTensor` in `TypeConversion.cpp`.

## `APInt::operator==` / `!=` asserts on bit-width mismatch

**What goes wrong.** `felt.const`'s `FeltConstAttr` uses `APIntParameter`'s
minimum-bits-needed sizing (e.g. `felt.const 1` produces a 4-bit APInt);
`arith.constant : index` is always 64-bit. Comparing them directly asserts in
dbg and is UB in opt: the slow-case `EqualSlowCase` reads past one side's word
boundary, often producing a corrupt pointer that segfaults much later at an
unrelated op.

The original bucket-2 S1-cluster `Type::getContext` UAF inside
`EmptyTemplateRemoval` was a downstream symptom of exactly this — the corrupt
pointer surfaced many ops after the bad `APInt` comparison.

ASAN-instrumented dbg trips at `APInt::EqualSlowCase` with mismatched
`getBitWidth()`. Opt builds bury the symptom in a delayed segfault.

**Rule.** Normalize at construction when collecting APInts from heterogeneous
sources. Apply `zextOrTrunc(kCommonWidth)` at the push_back sites (typically 64
for index-typed values). Reference site:
`SimplifySubComponents.cpp::outerIndexConstValues`.
