# MLIR C++ API gotchas

## `arith.cmpi` / `arith.cmpf` predicates live in op `properties`, not the attribute dict

**What goes wrong.** `def->getAttrOfType<IntegerAttr>("predicate")` returns null
on post-Properties-migration MLIR ops. A pass that reads the predicate via
`getAttr("predicate")` silently treats every `cmpi`/`cmpf` as "predicate
unknown" and falls through to a default branch — no error, wrong behavior. The
predicate is stored in MLIR's per-op `properties` storage and is invisible to
the discardable-attribute dict API.

This cost a full debugging session in `materializeStructOfPodsCompField`'s
`deriveReaderK`, which walks `arith.cmpi eq, %expr, %cK` predicates to
disambiguate cascade arms. Reading via `getAttr` silently always missed, and
every reader fell through to the `std::nullopt` branch.

**Rule.** Use the typed accessor:

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
if (auto cmp = dyn_cast<arith::CmpIOp>(def))
  if (cmp.getPredicate() == arith::CmpIPredicate::eq) { ... }
```

The print-form fallback (`Operation::print` into a string + `s.find(...)`) works
but is O(op_size) per check — measured at **>60× slowdown** (13 s to 14+ min) on
`iden3_query_sig`'s SSC pass alone. For any modern-MLIR op declaring inherent
attrs via TableGen `Properties`, prefer the generated typed accessor over
`getAttr(name)`.

## `gentbl_cc_library` outputs need a wide MLIR-header set

**What goes wrong.** Generated `<Dialect>Ops.{h,cpp}.inc` and
`<Dialect>Attrs.{h,cpp}.inc` transitively reference `DialectBytecodeWriter`
(`mlir/Bytecode/BytecodeOpInterface.h`), `getProperties()`
(`mlir/IR/OpDefinition.h` + `mlir/IR/Builders.h`), `OpAsmPrinter`
(`mlir/IR/OpImplementation.h`), `Diagnostics.h`, `SideEffectInterfaces.h` (when
ops use `Pure` etc.), and `DialectImplementation.h`. The manual `.cpp`/`.h` code
rarely uses these symbols directly, so cpplint flags them as "unused" — but
removing them breaks compilation of the generated `.inc` body.

**Rule.** Mirror the include set in `llzk_to_shlo/Dialect/WLA/WLA.{h,cpp}` when
adding a new dialect. Suppress cpplint with
`// NOLINT(build/include_what_you_use)` where needed. Do not trim includes from
a `gentbl_cc_library` consumer based on cpplint alone.

## Consolidate `gentbl_cc_library` rules into one per-dialect TableGen file

**What goes wrong.** Splitting one dialect's `.td` file across four separate
`gentbl_cc_library` rules re-invokes `mlir-tblgen` 4× per clean build for no
benefit. Each per-rule output must then declare the same `td_includes`
everywhere, creating four places where a missing include silently produces
incomplete generated code.

**Rule.** Mirror `llzk_to_shlo/Dialect/WLA/BUILD.bazel`'s `wla_inc_gen` rule,
which produces 8 outputs (dialect/enum/attr/op decls + defs) from a single
`mlir-tblgen` invocation. Do not run `mlir-tblgen` outside a
`gentbl_cc_library`.
