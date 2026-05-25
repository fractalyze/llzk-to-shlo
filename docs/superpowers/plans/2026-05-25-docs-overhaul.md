# Docs Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the eth-milestone-framed docs of `llzk-to-shlo` with a
coherent project-docs system organized around non-derivable knowledge (why /
passes / contracts / development), fronted by a narrative spine.

**Architecture:** Additive first — build the new `docs/` tree (`design/`,
`passes/`, `contracts/`, `development/`) and the spine `docs/README.md` by
relocating and de-framing durable content from existing docs. Then rewrite
`README.md` and `CLAUDE.md` to point in without duplicating prose. Then delete
the consumed source docs and the milestone/status docs. Delete the
`docs/superpowers/` workflow tree last (it holds this plan and the spec).

**Tech Stack:** Markdown only. Verification via `git grep`, a relative-link
checker, and `mdformat`. No source-code changes.

**Editorial filter (apply to every file written below):** every paragraph
carries *why / invariant / trap / contract* — never restate what the source code
already says (exception: procedural how-to not in source). No milestone/status
framing ("deliverable", "✅", ETA, velocity, point-in-time counts). `m3` survives
only as the real harness name (`bench/m3/`, `m3_runner`), never as "Milestone
3". One canonical home per topic + cross-links, no duplicated prose. Build-tool
run commands with `$PWD`-prefixed inputs. English throughout.

**Reference:** the design spec is at
`docs/superpowers/specs/2026-05-25-docs-overhaul-design.md` (§ references below
point into it).

______________________________________________________________________

### Task 0: Create the working branch

**Files:** none (git only).

- [ ] **Step 1: Branch off main**

Run:

```bash
cd /home/ryan/Workspace/llzk-to-shlo
git switch -c docs/overhaul
```

Expected: `Switched to a new branch 'docs/overhaul'`

- [ ] **Step 2: Confirm clean tree**

Run: `git status --short` Expected: only the untracked `docs/superpowers/`
spec+plan files, nothing else staged.

______________________________________________________________________

### Task 1: `design/` — the why

**Files:**

- Create: `docs/design/philosophy.md`

- Create: `docs/design/pipeline-shape.md`

- Create: `docs/design/failure-modes.md`

- [ ] **Step 1: Write `docs/design/philosophy.md`**

Source: `CLAUDE.md` "Design Philosophy" (the 4 core principles) +
`docs/GPU_PROFILING.md` "Why Batching Optimizes GPU Execution".

Sections:

- `# Design philosophy` — one-paragraph framing: lower LLZK → StableHLO to reuse
  ML-compiler infra; the end target is batched GPU witness generation via
  open-zkx `stablehlo_runner`.

- `## Reuse ML compiler infrastructure` — why express ZK primitives as StableHLO
  ops rather than a bespoke runtime.

- `## Batch first` — witness gen is embarrassingly parallel; every lowering must
  survive a leading batch dim added by `BatchStablehlo`. Fold in the durable
  rationale from GPU_PROFILING "Why Batching Optimizes GPU Execution"
  (kernel-launch overhead amortized to a single launch; batched time ≈ constant
  in N). **Drop the nsys numbers and the repro section.**

- `## GPU correctness is the gate` — `@constrain` is erased at lowering, so GPU
  code computes witnesses with no internal alarm; circom's native witness is the
  only catch. (Cross-link `../contracts/correctness-gate.md` for the full
  hierarchy.)

- `## Frontend-agnostic target` — LLZK is the stable contract; Circom is one
  producer; don't leak Circom assumptions into `LlzkToStablehlo`.

- [ ] **Step 2: Write `docs/design/pipeline-shape.md`**

Source: `CLAUDE.md` "Pipeline Overview" + "Why the Pipeline Looks This Way" +
the *why* portions of `docs/E2E_LOWERING_GUIDE.md`.

Sections:

- `# Pipeline shape` — the ASCII pipeline diagram (Circom → LLZK → StableHLO →
  Batched → GPU), one line per stage with the exact tool/flag.

- `## The two core passes` — one paragraph each on `SimplifySubComponents` and
  `LlzkToStablehlo` (role only; mechanics live in `../passes/`).

- `## Why the splits exist` — the four "merging would break correctness" splits,
  verbatim-in-spirit from CLAUDE: (1) pod-dispatch elimination is mandatory not
  cleanup; (2) SSC runs to a fixed point because nesting is arbitrary (+ the
  inner `while(phase(block))` rationale); (3) while-loop transform is four
  phases because LLZK is mutable and StableHLO is SSA; (4) post-passes exist
  because `applyPartialConversion` is 1:1 and can't restructure regions.
  Cross-link each to its `../passes/` mechanics doc.

- [ ] **Step 3: Write `docs/design/failure-modes.md`**

Source: `docs/CIRCUIT_COVERAGE.md` "Failure Analysis" + "Compile-time stretch".
**Extract the taxonomy only — no per-circuit matrix, no counts.**

Sections:

- `# Why circuits don't lower` — framing: failures cluster into families, not
  one-offs.

- `## Circom → LLZK: extended-envelope timeouts` — families that exceed circom's
  compile-time envelope (the ed25519 / large-circuit pattern; ~10-min timeout →
  no LLZK output). Describe the *category*, not the dated list.

- `## LLZK → StableHLO: unhandled shapes` — the two recurring shapes: `pod.new`
  forms SSC doesn't yet normalize, and partial-conversion gaps where no pattern
  covers a region/op shape. Each new family typically needs one dedicated SSC
  pre-pass or one targeted pattern.

- `## PointCompress: conversion timeout` — the one quadratic-walk perf case.

- [ ] **Step 4: Verify**

Run:

```bash
ls docs/design/{philosophy,pipeline-shape,failure-modes}.md
git grep -nEi 'milestone|deliverable|✅|\bETA\b|speedup|[0-9],?[0-9]{3}x' docs/design/ || echo "CLEAN: no milestone/status framing"
```

Expected: all three files listed; grep prints `CLEAN`.

- [ ] **Step 5: Commit**

```bash
git add docs/design/
git commit -m "docs(llzk-to-shlo): add design/ bucket (philosophy, pipeline-shape, failure-modes)"
```

______________________________________________________________________

### Task 2: `passes/` — what each pass does

**Files:**

- Create: `docs/passes/simplify-sub-components.md`

- Create: `docs/passes/llzk-to-stablehlo/README.md`

- Create: `docs/passes/llzk-to-stablehlo/op-lowering.md`

- Create: `docs/passes/llzk-to-stablehlo/while-loop-transformation.md`

- Create: `docs/passes/batch-stablehlo.md`

- [ ] **Step 1: Write `docs/passes/simplify-sub-components.md`**

Source: `E2E_LOWERING_GUIDE.md` "Pod Dispatch Elimination" (SSC 6-phase,
fixed-point, code organization) + `CLAUDE.md` "SSC runs to a fixed point"
rationale + `LOWERING_PITFALLS.md` "SimplifySubComponents driver-ordering traps"
and "Pod-dispatch silent miscompile signals".

Sections:

- `# SimplifySubComponents` — role: flatten pod dispatch into direct
  `function.call` so the type-aware conversion can match.

- `## The 6 phases` — name each phase and what it rewrites (concise; this
  mirrors code so keep it short and lead with *why each is needed*, not a
  line-by-line restatement).

- `## Code organization` — the `PodDispatchPhases` / `StructOfPodsConversion` /
  `PodArrayWhileCarry` / `PodArrayMaterialize` / `PodModuleCleanup` split vs the
  driver.

- `## Driver-ordering traps` — the ordering invariants (Phase 1/4/5 hoister &
  erasure ordering; `processNested` recursion guards; why a same-iteration
  consumer needs an inner `while(phase(block))` loop; outer fixed point is too
  coarse with N candidates).

- `## Silent-miscompile signals` — surviving pod-typed iter-args, sub-call count
  diffs, multi-carry stale operands, writerless dispatch pods, and the
  diagnostic each gives.

- [ ] **Step 2: Write `docs/passes/llzk-to-stablehlo/README.md`**

Source: `E2E_LOWERING_GUIDE.md` "Pipeline Overview" + `README.md` "Internal
Pipeline (LlzkToStablehlo)" listing + `CLAUDE.md` LlzkToStablehlo paragraph.

Sections:

- `# LlzkToStablehlo` — role: the heavy partial-conversion pass; pre-passes →
  main conversion → post-passes.

- `## Phase order and why it's forced` — reframe the 16-step internal pipeline
  from a *list* into *why each group exists and why the order can't change*
  (pre-passes prepare SSA-able shapes; `applyPartialConversion` is 1:1;
  post-passes restructure regions partial conversion can't). Link
  `op-lowering.md` and `while-loop-transformation.md` for mechanics.

- [ ] **Step 3: Write `docs/passes/llzk-to-stablehlo/op-lowering.md`**

Source: `E2E_LOWERING_GUIDE.md` "Type Conversion" + "Operation Patterns" +
`docs/load-bearing-invariants.md` "Lowering equivalences worth knowing".

Sections:

- `# Type and op lowering` — trimmed, why-first.

- `## Type conversion` — felt→scalar tensor, array→tensor (shape preserved),
  struct→flattened tensor (offset-based), pod→removed, i1→tensor<i1>. Keep the
  summary table; cut prose that restates the obvious.

- `## Operation patterns` — group: arithmetic 1:1; bitwise 3-step; div/rem
  3-step; struct offset-based slicing; array ops; bool/compare; control flow;
  removed ops; the `cast.toindex` constant-preservation gotcha. Lead each with
  the non-obvious bit (e.g. *why* bitwise needs 3 steps).

- `## Lowering equivalences` — `array.new` and `llzk.nondet` both lower to
  `dense<0>` (and why that's safe — an invariant of this pass).

- [ ] **Step 4: Write
  `docs/passes/llzk-to-stablehlo/while-loop-transformation.md`**

Source: `E2E_LOWERING_GUIDE.md` "While Loop Transformation" + "Post-passes" +
`CLAUDE.md` "while-loop transformation is four phases" + `LOWERING_PITFALLS.md`
"Walker traps".

Sections:

- `# While-loop transformation` — why four phases (LLZK mutable, StableHLO SSA;
  loop bodies mutate outer arrays).

- `## Step 1: promote captured arrays to while carry`

- `## Step 2: mutable → SSA conversion` — the shared
  `processBlockForArrayMutations` walker reused with different `latest`
  trackers; **it MUST gate mutating ops on the target being currently tracked,
  or untracked writes get rewritten orphan and silently DCE'd** (load-bearing).

- `## Step 3: scf.while → stablehlo.while` (post-pass).

- `## Post-passes: scf.if → stablehlo.select; vectorization` — the three
  vectorization phases (1D independent, 2D carry, nested-while inner).

- `## Walker traps` — `scf.while`/`scf.if`/`scf.execute_region` array-carry,
  passthrough detection, body-arg SSA-ification foot-guns.

- [ ] **Step 5: Write `docs/passes/batch-stablehlo.md`**

Source: `docs/BATCH_STABLEHLO.md` "How It Works" + "Op-by-Op Transformation
Rules" + `load-bearing-invariants.md` "Batching invariants". **Drop "Performance
Benchmarks".**

Sections:

- `# BatchStablehlo` — role: add a leading batch dim N to all tensors for
  parallel GPU witness gen.

- `## How it works` — leading-dimension insertion; implementation structure.

- `## Op-by-op rules` — element-wise; constants → `broadcast_in_dim`;
  `dynamic_slice`/`dynamic_update_slice`; one-hot write (scatter);
  `stablehlo.while` (fixed trip count); `func.call`/compare/select; LLZK
  residual ops.

- `## Batching invariants` — `BatchStablehlo` must see every signal; trip counts
  must be compile-time-constant; immutable circom signals are what make
  vectorization sound.

- [ ] **Step 6: Verify**

Run:

```bash
ls docs/passes/simplify-sub-components.md docs/passes/batch-stablehlo.md \
   docs/passes/llzk-to-stablehlo/{README,op-lowering,while-loop-transformation}.md
git grep -nEi 'milestone|deliverable|✅|\bETA\b|speedup|benchmark' docs/passes/ || echo "CLEAN"
```

Expected: all five files listed; grep prints `CLEAN`.

- [ ] **Step 7: Commit**

```bash
git add docs/passes/
git commit -m "docs(llzk-to-shlo): add passes/ bucket (SSC, LlzkToStablehlo, BatchStablehlo)"
```

______________________________________________________________________

### Task 3: `contracts/` — what must hold

**Files:**

- Create: `docs/contracts/correctness-gate.md`

- Create: `docs/contracts/witness-layout-anchor.md`

- Create: `docs/contracts/upstream-llzk-drift.md`

- [ ] **Step 1: Write `docs/contracts/correctness-gate.md`**

Source: `load-bearing-invariants.md` "Correctness gate hierarchy" + `CLAUDE.md`
"GPU correctness is the gate" + the two every-keystroke rules.

Sections:

- `# The correctness gate` — circom's native C++ witness is the only catch for a
  miscompile; `@constrain` is erased at lowering.

- `## The gate hierarchy` — sister-circuit families are NOT differential
  references; `<==` vs `<--` is a security boundary; GPU `batch[i] == single[i]`
  and byte-equal vs circom `.wtns` are the real signals.

- `## Two rules at every keystroke` — (1) verify against the lowered StableHLO,
  not the simplified LLZK (LLZK-level fixes are often absorbed downstream); (2)
  structural IR cleanup ≠ runtime fix — always re-run the gate after a
  structural change.

- [ ] **Step 2: Write `docs/contracts/witness-layout-anchor.md`**

Source: `docs/WITNESS_LAYOUT_ANCHOR.md` + `load-bearing-invariants.md` "Witness
layout & verifier asymmetries". **Strip the "Scope of this PR" framing.**

Sections:

- `# Witness-layout-anchor (wla)` — why it exists (the layout contract between
  lowering and the verifier).

- `## Op shape` — the `wla` op and `#wla.signal` fields.

- `## Cross-entry invariants` — what must hold across entry points;
  `getMemberFlatSize` recursion; `MemberReadOp` pub-from-outside rule;
  `--witness-layout-anchor` / `--verify-witness-layout` ordering.

- `## Background` — keep the durable background; remove PR-scoped language.

- [ ] **Step 3: Write `docs/contracts/upstream-llzk-drift.md`**

Source: `CLAUDE.md` "LLZK as a Moving Contract" + `LOWERING_PITFALLS.md`
"Upstream-LLZK contract drift" + the overhaul-memo seed content (spec §6) +
`CIRCUIT_COVERAGE.md` "Upstream resolution log".

Sections:

- `# LLZK as a moving contract` — LLZK is versioned upstream and changes break
  us; each bump tends to introduce 2–4 new pod/scf shapes.

- `## Match ops by type, never by mnemonic string` — match LLZK/MLIR ops by
  typed `isa<>`/`dyn_cast<>`, NEVER by
  `op->getName().getStringRef() == "mnemonic"` (string-matching a renamed/typo'd
  mnemonic compiles and silently never matches → silent miscompile). Gotchas:
  struct dialect C++ namespace is `llzk::component`; mnemonic ≠ class
  (`struct.writem`→`MemberWriteOp`, `struct.member`→`MemberDefOp`); LLZK
  `function.call`→`llzk::function::CallOp` vs upstream
  `func.call`→`func::CallOp`. LEAVE as strings: dialect-namespace checks
  (`getDialectNamespace()=="pod"`) and record/field NAME strings.

- `## Auditing for string-mnemonic matches` — use
  `grep -nE '(==|!=) *"[a-z_]+\.[a-z_0-9]+"'` — a plain `getStringRef() ==`
  search MISSES the split-variable form
  (`StringRef n = op->getName().getStringRef(); … n == "x"`).

- `## Known drift cases` — circom PR #378 same-named template wrap; `<[]>` vs
  no-params; `llzk.nondet : index`; test-fixture and BUILD-glue migration rules.

- `## Upstream resolution log` — the durable record of which upstream issues
  were resolved/worked around (from CIRCUIT_COVERAGE).

- [ ] **Step 4: Verify**

Run:

```bash
ls docs/contracts/{correctness-gate,witness-layout-anchor,upstream-llzk-drift}.md
git grep -nEi 'scope of this pr|milestone|deliverable|✅' docs/contracts/ || echo "CLEAN"
git grep -nE "getStringRef\(\) ==" docs/contracts/upstream-llzk-drift.md && echo "OK: audit recipe present"
```

Expected: three files listed; first grep prints `CLEAN`; second confirms the
audit recipe is documented.

- [ ] **Step 5: Commit**

```bash
git add docs/contracts/
git commit -m "docs(llzk-to-shlo): add contracts/ bucket (correctness-gate, wla, upstream-drift)"
```

______________________________________________________________________

### Task 4: `development/` — how to work in the repo

**Files:**

- Create: `docs/development/conventions.md`

- Create: `docs/development/correctness-gate-harness.md`

- Create: `docs/development/correctness-gate-fixture.md`

- Create: `docs/development/ci-and-build.md`

- Create: `docs/development/docs-style.md`

- Create: `docs/development/mlir-api-gotchas.md`

- Create: `docs/development/apint-arithmetic.md`

- Create: `docs/development/diagnostics.md`

- [ ] **Step 1: Write `docs/development/conventions.md`**

Source: `docs/conventions.md` (the index). Retarget links to the sibling files
in this folder (`correctness-gate-fixture.md`, `correctness-gate-harness.md`,
`docs-style.md`). Keep the "every entry is load-bearing" framing.

- [ ] **Step 2: Write `docs/development/correctness-gate-harness.md`**

Source: `docs/conventions/m3-correctness-gate.md`. Keep all `m3` / `bench/m3` /
`m3_runner` references (real code names). Strip any milestone framing in prose.
Sections preserved: `.json.gate` format, `.wtns` wire-layout caveat,
`circom --O0` internal-wire sentinels, duplicate-index/padding/N>1 restrictions,
enrollment workflow, circom-binary-swap cache caveat, `--stabilize`
load-bearing, `zkx_*` (not `xla_*`) flag prefix. Add the m3-gate invocation
`bazel test --config=cuda_clang //bench/m3:m3_correctness_gate_test` (note:
there is no `--config=gpu`).

- [ ] **Step 3: Write `docs/development/correctness-gate-fixture.md`**

Source: `docs/conventions/m3-fixture.md`. Sections preserved: multi-dim signals
flatten row-major; positional JSON param mapping; the lowered `@main` arg order
is NOT circom source order under `public [...]`; fixtures store one witness and
`m3_runner` auto-tiles; don't refactor toward sharing `stablehlo_runner_main.cc`
helpers.

- [ ] **Step 4: Write `docs/development/ci-and-build.md`**

Source: `docs/CI_SETUP_GUIDE.md` + `README.md` build prerequisites + the
`-c opt` testing note.

Sections:

- `# CI and build` — prerequisites (LLVM/MLIR 20.x, Bazel 7, `.bazelrc.user`
  toolchain paths).

- `## circom from the nix flake` — circom is built via the Nix flake (its ELF
  interpreter is a nix-store glibc; MLIR 20 dev libs come from the flake); the
  nix dev shell is required or the example circuits fail to build.

- `## Runner setup` — install Nix, build+install circom, verify environment,
  what's excluded from CI.

- `## Build mode and asserts` — the repo default is `-c opt` (NDEBUG), so
  `assert`s are compiled out; debug-build invariants need `-c dbg`.

- [ ] **Step 5: Write `docs/development/docs-style.md`**

Source: `docs/conventions/docs-style.md`. Keep as-is: `mdformat-footnote` is NOT
installed → use inline numeric markers.

- [ ] **Step 6: Write the three dev-gotcha files**

Each from its `LOWERING_PITFALLS.md` section, dev-time framing:

- `docs/development/mlir-api-gotchas.md` ← "MLIR C++ API gotchas":
  `arith.cmpi`/`cmpf` predicates live in `properties` not the discardable attr
  dict; `gentbl_cc_library` include-set hygiene; consolidate-the-`tblgen`-rule
  advice.

- `docs/development/apint-arithmetic.md` ← "APInt arithmetic traps":
  `getSExtValue` UB on `getBitWidth() > 64`; `operator==` bit-width-mismatch
  assertion.

- `docs/development/diagnostics.md` ← "Diagnostic foot-guns": `awk`-slicing a
  `stablehlo.while` body; m3-gate repro without `BUILD.bazel` edits.

- [ ] **Step 7: Verify**

Run:

```bash
ls docs/development/{conventions,correctness-gate-harness,correctness-gate-fixture,ci-and-build,docs-style,mlir-api-gotchas,apint-arithmetic,diagnostics}.md
git grep -nEi 'milestone|deliverable|✅' docs/development/ || echo "CLEAN"
git grep -nE "bench/m3|m3_runner" docs/development/correctness-gate-harness.md >/dev/null && echo "OK: m3 code names preserved"
```

Expected: all eight files listed; grep prints `CLEAN`; m3 code names preserved.

- [ ] **Step 8: Commit**

```bash
git add docs/development/
git commit -m "docs(llzk-to-shlo): add development/ bucket (conventions, gate, ci, gotchas)"
```

______________________________________________________________________

### Task 5: `docs/README.md` — the narrative spine

**Files:**

- Create: `docs/README.md`

- [ ] **Step 1: Write the spine** (spec §3 "narrative spine")

Sections, each ending in a "dig deeper" link:

- `# llzk-to-shlo` + one-paragraph **What this is** (LLZK IR → StableHLO so ZK
  witness generation rides ML-compiler infra and runs batched in one GPU
  launch).

- `## The bet` — reuse ML infra · batch-first · GPU-correctness-is-the-gate ·
  frontend-agnostic. → `design/philosophy.md`

- `## The shape` — the pipeline diagram + why the splits exist. →
  `design/pipeline-shape.md`

- `## The contracts you must not break` — LLZK is a moving target; circom is the
  only catch; witness layout. → `contracts/`

- `## The traps` — silent miscompiles; what to check before touching a pass. →
  the relevant `passes/*` + `contracts/`

- `## Where to dig` — one-line map of `passes/` (per-pass mechanics) and
  `development/` (conventions, gate, build, gotchas).

- [ ] **Step 2: Verify all spine links resolve**

Run:

```bash
cd /home/ryan/Workspace/llzk-to-shlo
while read -r link; do
  t="docs/$(echo "$link" | sed -E 's/#.*//')"
  [ -e "$t" ] || [ -d "$t" ] || echo "DEAD: $link"
done < <(grep -oE '\]\(([^)]+)\)' docs/README.md | sed -E 's/^\]\(//; s/\)$//' | grep -vE '^https?://')
echo "link check done"
```

Expected: `link check done` with no `DEAD:` lines.

- [ ] **Step 3: Commit**

```bash
git add docs/README.md
git commit -m "docs(llzk-to-shlo): add docs/README.md narrative spine"
```

______________________________________________________________________

### Task 6: Rewrite the front-door `README.md`

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md`** (spec §4 "Rewritten top-level")

**Keep:** the one-line overview, the pipeline ASCII diagram, the full Quick
Start (Prerequisites, `.bazelrc.user`, build, the 4-step
circom→LLZK→StableHLO→batch→GPU run commands with `$PWD`/`pwd` prefixes, the
prime-field flag options), the Examples (Bazel targets) section, Dependencies,
License.

**Delete:** the "Key Results" block (95,000x etc.), the "Architecture → Internal
Pipeline (LlzkToStablehlo)" 16-step listing (now
`docs/passes/llzk-to-stablehlo/README.md`), the "Coverage" counts block, the
"GPU Correctness" counts, and the entire "Performance" tables (status).

**Replace** the old "Documentation" table with a single pointer: "To understand
the project, start at [`docs/README.md`](docs/README.md)." plus a 4-line bucket
map (design / passes / contracts / development).

For the Testing section: keep *how to run* the suites (`bazel test //...`, the
manual E2E command) but drop the per-suite point-in-time counts ("11 batch + 5
lowering", "46 circom-benchmarks", "25-chip").

- [ ] **Step 2: Verify**

Run:

```bash
git grep -nEi 'milestone|deliverable|✅|95,?[0-9]{3}x|Key Results|ETA\b' README.md || echo "CLEAN"
grep -nE 'docs/README.md' README.md && echo "OK: points into docs"
grep -nE 'bazel run' README.md >/dev/null && echo "OK: build-tool run commands present"
```

Expected: first grep prints `CLEAN`; README points into `docs/README.md`;
build-tool run commands present.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(llzk-to-shlo): rewrite README as front door + quickstart, drop status framing"
```

______________________________________________________________________

### Task 7: Slim `CLAUDE.md` to an agent index

**Files:**

- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite `CLAUDE.md`** (spec §4 "Rewritten top-level")

New shape:

- `# llzk-to-shlo` + one sentence + "Orientation: read
  [`docs/README.md`](docs/README.md) first."
- `## Rules at every keystroke` — the three high-frequency agent rules, each one
  tight paragraph with a link to its full home:
  1. **Match ops by type, never mnemonic string.** Use `isa<>`/`dyn_cast<>`;
     string-matching a renamed mnemonic silently never matches → silent
     miscompile. →
     [`docs/contracts/upstream-llzk-drift.md`](docs/contracts/upstream-llzk-drift.md)
  1. **Verify against the lowered StableHLO, not the simplified LLZK.**
     LLZK-level fixes are often absorbed downstream. →
     [`docs/contracts/correctness-gate.md`](docs/contracts/correctness-gate.md)
  1. **Structural cleanup ≠ runtime fix.** Re-run the correctness gate after any
     structural change. →
     [`docs/contracts/correctness-gate.md`](docs/contracts/correctness-gate.md)
- `## Map` — one-line pointer to each `docs/` bucket (design / passes /
  contracts / development).

Remove all the prose now living in `docs/` (Design Philosophy, Pipeline
Overview, Why the Pipeline Looks This Way, Load-Bearing Invariants, Conventions,
Top-level docs, External sources) — replace with links. **No duplicated prose.**

- [ ] **Step 2: Verify**

Run:

```bash
wc -l CLAUDE.md   # expect well under the old 216 lines
while read -r link; do
  [ -e "$link" ] || echo "DEAD: $link"
done < <(grep -oE '\]\(([^)]+)\)' CLAUDE.md | sed -E 's/^\]\(//; s/\)$//' | grep -vE '^https?://')
echo "link check done"
```

Expected: small line count; `link check done` with no `DEAD:` lines.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(llzk-to-shlo): slim CLAUDE.md to agent index + keystroke rules"
```

______________________________________________________________________

### Task 8: Delete consumed sources and milestone/status docs

**Files (delete):** `llzk-status.md`, `STATUS_FOR_LLZK_MAINTAINER.md`,
`request.md`, `docs/M3_REPORT.md`, `docs/charts/`, `docs/GPU_PROFILING.md`,
`docs/CIRCUIT_COVERAGE.md`, `docs/E2E_LOWERING_GUIDE.md`,
`docs/LOWERING_PITFALLS.md`, `docs/BATCH_STABLEHLO.md`,
`docs/WITNESS_LAYOUT_ANCHOR.md`, `docs/load-bearing-invariants.md`,
`docs/CI_SETUP_GUIDE.md`, `docs/conventions.md`, `docs/conventions/` (folder).

- [ ] **Step 1: Confirm nothing in the repo still references these paths**

Run:

```bash
cd /home/ryan/Workspace/llzk-to-shlo
git grep -nE 'docs/(M3_REPORT|GPU_PROFILING|CIRCUIT_COVERAGE|E2E_LOWERING_GUIDE|LOWERING_PITFALLS|BATCH_STABLEHLO|WITNESS_LAYOUT_ANCHOR|load-bearing-invariants|CI_SETUP_GUIDE|conventions)\b|llzk-status\.md|STATUS_FOR_LLZK_MAINTAINER\.md' \
  -- ':!docs/superpowers/*' || echo "NO LIVE REFERENCES"
```

Expected: `NO LIVE REFERENCES` (any hit must be fixed in the offending file
before deleting — re-point it to the new `docs/` home).

- [ ] **Step 2: Delete**

```bash
git rm -r llzk-status.md STATUS_FOR_LLZK_MAINTAINER.md request.md \
  docs/M3_REPORT.md docs/charts docs/GPU_PROFILING.md docs/CIRCUIT_COVERAGE.md \
  docs/E2E_LOWERING_GUIDE.md docs/LOWERING_PITFALLS.md docs/BATCH_STABLEHLO.md \
  docs/WITNESS_LAYOUT_ANCHOR.md docs/load-bearing-invariants.md docs/CI_SETUP_GUIDE.md \
  docs/conventions.md docs/conventions
```

Expected: each path reported as `rm`.

- [ ] **Step 3: Commit**

```bash
git commit -m "docs(llzk-to-shlo): remove milestone/status docs and consumed sources"
```

______________________________________________________________________

### Task 9: Whole-tree verification

**Files:** none (verification only).

- [ ] **Step 1: No dead relative links anywhere in docs + top-level**

Run:

```bash
cd /home/ryan/Workspace/llzk-to-shlo
fail=0
for f in README.md CLAUDE.md $(find docs -name '*.md' -not -path 'docs/superpowers/*'); do
  dir=$(dirname "$f")
  while read -r link; do
    rel=$(echo "$link" | sed -E 's/#.*//'); [ -z "$rel" ] && continue
    case "$rel" in http*) continue;; esac
    target=$(realpath -m "$dir/$rel")
    [ -e "$target" ] || { echo "DEAD in $f -> $rel"; fail=1; }
  done < <(grep -oE '\]\(([^)]+)\)' "$f" | sed -E 's/^\]\(//; s/\)$//')
done
[ $fail -eq 0 ] && echo "ALL LINKS RESOLVE"
```

Expected: `ALL LINKS RESOLVE`.

- [ ] **Step 2: No milestone/status framing survives**

Run:

```bash
git grep -nEi 'milestone [0-9]|deliverable|✅|\bETA\b|velocity|95,?[0-9]{3}x|Key Results' \
  -- 'docs/**/*.md' README.md CLAUDE.md ':!docs/superpowers/*' || echo "CLEAN"
```

Expected: `CLEAN`. (Plain "m3"/"bench/m3" as a harness name is allowed and will
not match.)

- [ ] **Step 3: Non-derivable filter spot-check (manual)**

Open `docs/passes/llzk-to-stablehlo/op-lowering.md` and
`docs/development/diagnostics.md`. Confirm no section is a pure restatement of
code — each carries a *why / invariant / trap*. If a section just narrates what
the code does, trim or cut it.

- [ ] **Step 4: Markdown style conforms**

Run:
`mdformat --check docs README.md CLAUDE.md 2>/dev/null || echo "review mdformat output"`
Expected: clean, or fix per `docs/development/docs-style.md` (inline numeric
markers; no footnote syntax).

- [ ] **Step 5: Quickstart sanity (commands, not execution)**

Confirm the README Quick Start uses `bazel run …` / `circom …` with `$PWD` or
`` `pwd` `` prefixes on file inputs, and the prime-field flags match
`--llzk-to-stablehlo="prime=…"`. No commit needed if Tasks 6 left it correct;
otherwise fix and amend.

- [ ] **Step 6: Commit any fixes**

```bash
git add -A && git commit -m "docs(llzk-to-shlo): fix dead links and style nits from verification" || echo "nothing to fix"
```

______________________________________________________________________

### Task 10: Remove the workflow tree (LAST)

**Files (delete):** `docs/superpowers/` (holds this plan and the spec — must be
deleted only after Task 9 passes).

- [ ] **Step 1: Delete**

```bash
cd /home/ryan/Workspace/llzk-to-shlo
git rm -r docs/superpowers
```

Expected: spec + plan reported as `rm`.

- [ ] **Step 2: Final link re-check (superpowers gone)**

Run the Task 9 Step 1 link checker again. Expected: `ALL LINKS RESOLVE` (nothing
in the shipped docs should have linked into `docs/superpowers/`).

- [ ] **Step 3: Commit**

```bash
git commit -m "docs(llzk-to-shlo): remove brainstorming/plan workflow artifacts"
```

- [ ] **Step 4: Hand back to the user**

Do not push or open a PR — the user handles pushes/PRs for this repo. Report the
branch (`docs/overhaul`) and the commit list.

______________________________________________________________________

## Self-review against the spec

- **§2 north star / editorial filter** → enforced per-file in Tasks 1–7 and
  spot-checked in Task 9 Step 3. ✓
- **§3 IA (README / CLAUDE / docs spine + 4 buckets)** → Tasks 1–7 create
  exactly the §3 tree; subfolder for `llzk-to-stablehlo`, single-file SSC. ✓
- **§4 sourcing map** → every new file's sources named in its write step. ✓
- **§5 deletions** → Task 8 (sources + status) and Task 10 (superpowers, last).
  ✓
- **§6 seed content** (isa\<> rule, audit grep, m3 invocation,
  `-c opt`/`-c dbg`) → folded in Tasks 3 and 4. ✓
- **§7 acceptance criteria** → Task 9 covers links (4), milestone-grep (1),
  filter spot-check (6), mdformat (7), quickstart (5); criterion (8)
  superpowers-removed → Task 10; criteria (2)(3) covered by per-task verifies. ✓
- **§8 sequencing** (superpowers deleted last) → Task 10 is terminal. ✓
- **§9 out of scope** (no `bench/m3` rename, no `-c dbg` lit fixes, no code
  changes) → no task touches source code. ✓
- **No-placeholder scan** → each write step has concrete sections + sources;
  verification steps have exact commands and expected output. ✓
