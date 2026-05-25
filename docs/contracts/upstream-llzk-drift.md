# LLZK as a moving contract

LLZK is versioned upstream at
[project-llzk/llzk-lib](https://github.com/project-llzk/llzk-lib) (pinned in
`third_party/llzk/workspace.bzl`) and changes break us. Each upstream bump tends
to introduce 2–4 new pod/scf shapes the pattern set hasn't seen, plus occasional
dialect-text syntax changes that hand-written test fixtures must absorb. The
broader catalogue of silent-miscompile traps — walker traps, SSC driver-ordering
traps, pod-dispatch miscompile signals — lives in
[design/failure-modes.md](../design/failure-modes.md). This document covers the
subset specific to upstream contract drift: how to match LLZK ops safely, how to
audit for existing string-match regressions, and the record of drift cases that
have already been resolved.

## Match ops by type, never by mnemonic string

Match LLZK and MLIR ops using the typed C++ API (`isa<>`, `dyn_cast<>`,
`llvm::TypeSwitch`). Never match by
`op->getName().getStringRef() == "mnemonic"`. String-matching a renamed or
mis-spelled mnemonic compiles without error and silently never matches — the
pattern becomes a no-op and the pass proceeds as if the op didn't exist, which
is a silent miscompile, not a build failure.

Specific namespaces and mnemonics that have caused confusion:

1. **The struct dialect's C++ namespace is `llzk::component`, not `struct`.**
   `struct` is the text-format dialect prefix (e.g. `struct.writem`), but the
   C++ type lives in `llzk::component::MemberWriteOp` (not
   `struct::MemberWriteOp`). Using the wrong namespace prefix compiles silently
   and produces a type that is never equal to anything in the IR.

1. **Mnemonic strings and C++ class names do not correspond one-to-one.**
   Notable pairs:

   - `struct.writem` → `MemberWriteOp`
   - `struct.member` → `MemberDefOp`
   - `struct.readm` → `MemberReadOp` A pass that checks
     `getStringRef() == "struct.member"` to detect member definitions will also
     silently miss the op if LLZK renames the mnemonic in a future upstream
     bump.

1. **`llzk::function::CallOp` vs `func::CallOp` are distinct types.** LLZK uses
   its own `function.call` op (`llzk::function::CallOp`); upstream MLIR's
   `func.call` is `func::CallOp`. Code that matches `func::CallOp` will miss
   LLZK function calls, and vice versa. Both can appear in the same module
   during partial conversion; distinguish them by type, not by string prefix.

**What to leave as strings:** dialect-namespace checks
(`getDialectNamespace() == "pod"`) and record/field name strings (e.g.
`sym.getName() == "@out"`) are legitimate string comparisons — the former
dispatches on dialect identity, which is a stable string; the latter matches
user-visible symbol names, which change only when the circuit changes.

## Auditing for string-mnemonic matches

To find existing string-mnemonic comparisons in the codebase, use:

```bash
grep -nE '(==|!=) *"[a-z_]+\.[a-z_0-9]+"' llzk_to_shlo/ tests/
```

This catches the most common form: `getStringRef() == "struct.writem"` or
`opName != "array.write"`. It does NOT catch the split-variable form:

```cpp
StringRef n = op->getName().getStringRef();
// ... later ...
if (n == "struct.writem") { ... }
```

For split-variable forms, search separately for `getStringRef()` call sites and
trace the stored `StringRef` to its comparison sites. The regex above is the
faster first-pass audit; the split-variable form is rare and typically
introduced when the mnemonic is compared in multiple places.

## Known drift cases

**circom PR #378 same-named template wrap.** Post-#378 circom emits every
`function.def` / `struct.def` inside a same-named `poly.template @X` to track
polymorphic typing. `EmptyTemplateRemoval` rewrites that to
`builtin.module @X { function.def @X }`. The inner symbol then shadows the
wrapping module's symbol in the parent's `SymbolTable`, and the next pass that
walks it trips with `redefinition of symbol named '<X>'` during
`applyPartialConversion`. The fix — `flattenSingleEntityWrapperModules` in
`SimplifySubComponents.cpp` — hoists the same-named single child to module
level, erases the empty wrapper, and uses `AttrTypeReplacer` to rewrite
`@X::@X[::@method]` → `@X[::@method]` including refs nested in types. A plain
attribute walk misses type-embedded refs.

**`<[]>` vs no-params on `!struct.type`.** Template removal rewrites `<[]>` to
no-params on ops in `OpClassesWithStructTypes` but leaves uncovered ops alone.
Mixing the two forms across a use-def edge produces an unresolved
`builtin.unrealized_conversion_cast` that `applyPartialConversion` won't
legalize. The fix is to strip `<[]>` on uncovered ops (any op not in
`OpClassesWithStructTypes`) but not on covered ones — template removal handles
those itself.

**`llzk.nondet : index`.** The conversion target legalizes `llzk.nondet` for
`felt`/`array`/`struct` only. Residual `pod.read` ops with `index` result type
(dispatch-pod `@count` countdown after Phase 5) survive as
`llzk.nondet : index`, which `applyPartialConversion` fails to legalize. The fix
is to substitute `arith.constant 0 : index` — the surrounding `cmpi`/`scf.if`
scaffold is structurally dead once the call is hoisted, so `0` keeps `cmpi`
false and DCE collapses the dead branch.

**Test fixtures are consumer-owned IR.** Hand-written `.mlir` test files
(everything under `tests/`) are parsed directly by `llzk-to-shlo-opt`. The
upstream IR migrator does not touch parser input — it rewrites in-tree LLZK IR
via the dialect's loader/printer, not arbitrary `.mlir` fixtures. When bumping
LLZK upstream, hand-migrate every consumer fixture in the same change. Skipping
this produces a green build whose fixtures parse on the old dialect text shape
but break for anyone trying to add a new test against the updated dialect.

**BUILD glue must move with upstream.** New `.td` files (new dialects /
interfaces) added upstream do not appear in `third_party/llzk/llzk.BUILD`
automatically. The Bazel build silently skips inc-gen rules for missing `.td`
paths. When bumping, diff `include/llzk/Dialect/**/*.td` against
`gentbl_cc_library` targets in `third_party/llzk/llzk.BUILD` and add inc-gen
rules for every missing TableGen file.

**`createEmptyTemplateRemoval` and `applyFullConversion` scope.** Its conversion
target only handles ops in `OpClassesWithStructTypes`
(`struct`/`array`/`function`/`global`/`constrain`/`polymorphic`). Anything else
(`pod.*`, `llzk.nondet` results with struct types, synthesized ops) must be gone
or already in stripped form before this pass runs — `applyFullConversion`
rejects unhandled ops with a `failed to legalize` error. Ordering: clean
residual pod traffic first, pre-strip `<[]>` only on ops outside that set, then
run template removal.

## Upstream resolution log

The following upstream issues have been resolved or locally accommodated.
Recorded here so the resolution is findable without grepping git history.

1. **`template_ext.rs:243` panic** — "not yet implemented: Support mixed type
   subcomponent instantiations". Resolved upstream by
   [project-llzk/circom#376](https://github.com/project-llzk/circom/pull/376).
   Previously caused every chip with mixed-type subcomponent instantiations to
   fail at parse with no diagnostic.

1. **`Conflicting types to read array`** on parameterized component arrays
   (`inner[i] = Inner(i)` patterns). Tracked at
   [project-llzk/circom#386](https://github.com/project-llzk/circom/issues/386).
   Resolved by
   [project-llzk/circom#398](https://github.com/project-llzk/circom/pull/398).
   Previously caused a class of chips (including `aes-circom/gfmul_int_test`,
   `maci/ecdh_test`) to fail at the circom frontend before producing any LLZK
   IR.

1. **`poly.template` wrapping migration** — introduced by
   [project-llzk/circom#378](https://github.com/project-llzk/circom/pull/378)
   and [#381](https://github.com/project-llzk/circom/pull/381). Locally
   accommodated in `SimplifySubComponents`'s `flattenSingleEntityWrapperModules`
   pass (see "Known drift cases" above). Chips built against post-#378 circom
   require the flatten pass; pre-#378 artifacts escape it.
