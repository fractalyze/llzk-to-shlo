# Docs Overhaul — Design Spec

_2026-05-25_

> This file is a brainstorming/workflow artifact, **not** shipped project
> documentation. The `docs/superpowers/` tree is deleted as the final step of
> executing this overhaul (see §8). It exists only to drive the implementation
> plan.

## 1. Problem

The current docs were written "with the eth milestone in mind" — milestone and
deliverable framing (`Milestone 2 Deliverables ✅`, grant reports, dated status
snapshots, velocity/ETA tables, marketing-style "95,000x" results). That framing
ages badly and gets in the way of the docs' real job.

We want project docs that:

1. **Read like a project**, not a status report — a human-readable account of
   what `llzk-to-shlo` is, why it exists, and how it is shaped.
1. **Front-load what an agent (or a new engineer) cannot derive from the
   source** — the *why*, the design philosophy, the non-derivable invariants,
   the traps, the contracts. Humans will learn this codebase through an agent
   anyway, so there is one reader, not two.

## 2. North star (the editorial filter)

Every paragraph must carry **why / invariant / trap / contract** — something you
cannot recover by reading the code. If a sentence merely restates what the
source already says, cut it.

The one exception is **genuinely procedural how-to that is not in the source**:
build/run commands, the nix-flake requirement, prime-field flags, gate
enrollment steps. That is legitimate non-derivable knowledge.

Corollaries:

- **No milestone/status framing.** No "deliverable", no "✅", no ETA/velocity, no
  point-in-time pass/fail counts. Live status belongs in git history and CI, not
  in docs.
- **`m3` stays only as a code name.** The correctness-gate harness is
  `bench/m3/` / `m3_runner` / `//bench/m3:m3_correctness_gate_test` in the
  source. Docs must use that name to stay accurate, but must describe it as "the
  correctness gate," never as "Milestone 3." Renaming the harness is a separate
  code refactor, out of scope here (§9).
- **One home + cross-links.** A topic that spans buckets (pod-dispatch
  elimination: *why* in `design/`, *mechanics* in `passes/`) gets one canonical
  home per facet and is linked, never duplicated as prose.
- **Build-tool run commands**, `$PWD`-prefixed file inputs, per the repo
  playbook (`bazel run …`, not `bazel-bin/…`).
- **English throughout.**

## 3. Information architecture

Three layers with distinct jobs and **no duplicated prose**:

- **`README.md`** — repo front door. One-paragraph what+why, the quickstart
  (build/run, prime flags, circom-branch caveat, example targets), and a single
  pointer into `docs/`.
- **`CLAUDE.md`** — agent auto-context. Points at `docs/README.md` as the entry,
  plus the few highest-frequency keystroke rules inline. Everything else is a
  link, so the auto-loaded context stays small and each fact has one home.
- **`docs/`** — the knowledge, in four intuitive buckets: **why** it is the way
  it is (`design/`), **what each pass does** (`passes/`), **what must hold**
  (`contracts/`), and **how to work in the repo** (`development/`). The
  narrative spine `docs/README.md` ties them together.

### Final tree

```
README.md                          front door + quickstart
CLAUDE.md                          agent index + keystroke rules → docs/README.md
docs/
  README.md                        ★ narrative spine
  design/
    philosophy.md                  the 4 bets (+ why batching wins on GPU)
    pipeline-shape.md              the data flow & why the splits exist
    failure-modes.md               why whole circuit families don't lower
  passes/
    simplify-sub-components.md      role + pod-dispatch mechanics + SSC ordering invariants & traps
    llzk-to-stablehlo/
      README.md                    role + pre/main/post phase overview & why
      op-lowering.md               type conversion + op patterns (lean, why-first)
      while-loop-transformation.md array→carry→SSA→stablehlo.while + vectorization + walker traps
    batch-stablehlo.md             role + op-by-op batch rules + batching invariants
  contracts/                       pipeline-wide things that must hold
    correctness-gate.md            circom-only catch · <== vs <-- · @constrain erased
    witness-layout-anchor.md       the wla dialect+pass contract
    upstream-llzk-drift.md         LLZK as a moving contract + resolution log
  development/                     how to work in this repo
    conventions.md                 index
    correctness-gate-harness.md    the m3 gate convention (references bench/m3)
    correctness-gate-fixture.md    the m3 fixture convention
    ci-and-build.md                nix flake · MLIR 20 · runner setup
    docs-style.md
    mlir-api-gotchas.md            dev-time MLIR C++ traps
    apint-arithmetic.md            dev-time APInt traps
    diagnostics.md                 diagnostic foot-guns
```

`simplify-sub-components.md` stays a single file for now; promote it to a
subfolder only if it outgrows focus. `llzk-to-stablehlo/` is a subfolder because
the pass is heavy enough that one file would fight the focused-files goal.

### The narrative spine (`docs/README.md`)

Read top-to-bottom in ~1.5 screens; each section ends in a "dig deeper" link.

1. **What this is** — one paragraph: LLZK circuit IR → StableHLO so ZK witness
   generation rides ML-compiler infrastructure and runs batched in one GPU
   launch.
1. **The bet** — reuse ML infra · batch-first · GPU-correctness-is-the-gate ·
   frontend-agnostic. → `design/philosophy.md`
1. **The shape** — the pipeline diagram + why the splits exist. →
   `design/pipeline-shape.md`
1. **The contracts you must not break** — LLZK is a moving target; circom is the
   *only* catch for a miscompile; witness layout. → `contracts/`
1. **The traps** — silent miscompiles; what to check before touching a pass. →
   the relevant `passes/*` + `contracts/`
1. **Where to dig** — one-line map of `passes/` and `development/`.

## 4. Content-sourcing map

Each new file is assembled from existing material, then run through the §2
filter. Nothing is invented; durable content is relocated and de-framed.

### `design/`

- **`philosophy.md`** ← `CLAUDE.md` "Design Philosophy" (the 4 core principles)
  - `docs/GPU_PROFILING.md` "Why Batching Optimizes GPU Execution" (the durable
    rationale for the batch-first bet; drop the nsys numbers).
- **`pipeline-shape.md`** ← `CLAUDE.md` "Pipeline Overview" + "Why the Pipeline
  Looks This Way" (the four splits) + the *why* portions of
  `docs/E2E_LOWERING_GUIDE.md`.
- **`failure-modes.md`** ← `docs/CIRCUIT_COVERAGE.md` "Failure Analysis"
  taxonomy (why whole families don't lower: circom→LLZK extended-envelope
  timeouts; LLZK→StableHLO shape gaps) + "Compile-time stretch". Drop the
  per-circuit pass/fail matrix and all counts.

### `passes/`

- **`simplify-sub-components.md`** ← `E2E_LOWERING_GUIDE.md` "Pod Dispatch
  Elimination" (SSC 6-phase, fixed-point, code organization) + `CLAUDE.md` "SSC
  runs to a fixed point" rationale + `LOWERING_PITFALLS.md`
  "SimplifySubComponents driver-ordering traps" and "Pod-dispatch silent
  miscompile signals".
- **`llzk-to-stablehlo/README.md`** ← `E2E_LOWERING_GUIDE.md` "Pipeline
  Overview" + `README.md` "Internal Pipeline (LlzkToStablehlo)" listing,
  reframed from a list into *why each phase exists and why the order is forced*
  - `CLAUDE.md` LlzkToStablehlo paragraph.
- **`llzk-to-stablehlo/op-lowering.md`** ← `E2E_LOWERING_GUIDE.md` "Type
  Conversion" + "Operation Patterns" (trimmed to why-first, no exhaustive
  restatement) + `docs/load-bearing-invariants.md` "Lowering equivalences"
  (`array.new` and `llzk.nondet` both → `dense<0>`).
- **`llzk-to-stablehlo/while-loop-transformation.md`** ← `E2E_LOWERING_GUIDE.md`
  "While Loop Transformation" (3 steps) + "Post-passes" (scf.if→select,
  vectorization) + `CLAUDE.md` "while-loop transformation is four phases" why +
  `LOWERING_PITFALLS.md` "Walker traps".
- **`batch-stablehlo.md`** ← `docs/BATCH_STABLEHLO.md` "How It Works" +
  "Op-by-Op Transformation Rules" + `load-bearing-invariants.md` "Batching
  invariants". Drop the "Performance Benchmarks" tables.

### `contracts/`

- **`correctness-gate.md`** ← `load-bearing-invariants.md` "Correctness gate
  hierarchy" + `CLAUDE.md` "GPU correctness is the gate" + the two
  every-keystroke rules ("verify against the lowered StableHLO, not the
  simplified LLZK"; "structural IR cleanup ≠ runtime fix").
- **`witness-layout-anchor.md`** ← `docs/WITNESS_LAYOUT_ANCHOR.md` (strip the
  "Scope of this PR" framing) + `load-bearing-invariants.md` "Witness layout &
  verifier asymmetries".
- **`upstream-llzk-drift.md`** ← `CLAUDE.md` "LLZK as a Moving Contract" +
  `LOWERING_PITFALLS.md` "Upstream-LLZK contract drift" + the
  isa\<>-not-string-match rule and the audit-grep recipe (from the overhaul
  memo, see §6) + `CIRCUIT_COVERAGE.md` "Upstream resolution log".

### `development/`

- **`conventions.md`** ← `docs/conventions.md` (index; retarget link paths).
- **`correctness-gate-harness.md`** ← `docs/conventions/m3-correctness-gate.md`
  (strip milestone framing; keep `m3`/`bench/m3` references; keep the m3-gate
  invocation `bazel test --config=cuda_clang …`).
- **`correctness-gate-fixture.md`** ← `docs/conventions/m3-fixture.md`.
- **`ci-and-build.md`** ← `docs/CI_SETUP_GUIDE.md` + `README.md` build
  prerequisites + the `-c opt` (NDEBUG) testing note: asserts are compiled out
  under the default build, so debug-build invariants need `-c dbg`.
- **`docs-style.md`** ← `docs/conventions/docs-style.md`.
- **`mlir-api-gotchas.md`** ← `LOWERING_PITFALLS.md` "MLIR C++ API gotchas".
- **`apint-arithmetic.md`** ← `LOWERING_PITFALLS.md` "APInt arithmetic traps".
- **`diagnostics.md`** ← `LOWERING_PITFALLS.md` "Diagnostic foot-guns".

### Rewritten top-level

- **`README.md`** — keep: the one-line overview, the pipeline diagram, Quick
  Start (prereqs, `.bazelrc.user`, build, the circom→LLZK→StableHLO→batch→GPU
  run commands, prime-field flags), the Examples (bazel targets) section,
  Dependencies, License. **Drop**: "Key Results", the "Internal Pipeline"
  listing (→ `passes/`), the "Coverage" counts, "GPU Correctness" counts, and
  the "Performance" tables (status). End with a single pointer to
  `docs/README.md`.
- **`CLAUDE.md`** — shrink to: a pointer to `docs/README.md` as the orientation
  entry, plus the keystroke rules an agent hits constantly — (1) match LLZK/MLIR
  ops by typed `isa<>`/`dyn_cast<>`, never by mnemonic string compare; (2)
  verify against the lowered StableHLO, not the simplified LLZK; (3) structural
  cleanup ≠ runtime fix, re-run the gate. Everything else becomes a link into
  `docs/`.

## 5. Deletions (final)

After their durable content is rescued per §4:

- `llzk-status.md` (Milestone-2 status, Korean)
- `STATUS_FOR_LLZK_MAINTAINER.md` (stale dated maintainer snapshot)
- `docs/M3_REPORT.md` (submitted grant deliverable)
- `docs/charts/` (M3 chart assets)
- `docs/GPU_PROFILING.md` (after extracting "why batching wins")
- `docs/CIRCUIT_COVERAGE.md` (after extracting the failure-mode taxonomy)
- `request.md` (stale, unrelated task-pointer leftover)
- `docs/superpowers/{specs,plans}` (workflow artifacts — deleted last, §8)

Files consumed and replaced (their content moves into the new tree, originals
removed): `docs/E2E_LOWERING_GUIDE.md`, `docs/LOWERING_PITFALLS.md`,
`docs/BATCH_STABLEHLO.md`, `docs/WITNESS_LAYOUT_ANCHOR.md`,
`docs/load-bearing-invariants.md`, `docs/CI_SETUP_GUIDE.md`,
`docs/conventions.md`, `docs/conventions/*`.

## 6. Overhaul-memo seed content to fold in

From the project memory (lessons deliberately kept out of `CLAUDE.md` to avoid
churn), fold into the destinations noted:

- **isa\<>-not-string-match** rule, with the gotchas (struct dialect C++
  namespace is `llzk::component`; mnemonic ≠ class; `function.call` →
  `llzk::function::CallOp` vs upstream `func::CallOp`; leave dialect-namespace
  and record/field NAME checks as strings) → `contracts/upstream-llzk-drift.md`,
  summarized as keystroke rule (1) in `CLAUDE.md`.
- **Audit grep** `grep -nE '(==|!=) *"[a-z_]+\.[a-z_0-9]+"'` (catches the
  split-variable form a `getStringRef() ==` search misses) →
  `contracts/upstream-llzk-drift.md`.
- **m3 gate invocation** (`bazel test --config=cuda_clang …`; there is no
  `--config=gpu`; circom built from the nix flake needs the nix dev shell) →
  `development/correctness-gate-harness.md` + `development/ci-and-build.md`.
- **`-c opt` NDEBUG** → asserts compiled out; debug-build invariants need
  `-c dbg` → `development/ci-and-build.md`.

## 7. Acceptance criteria

1. All §5 milestone/status docs are deleted; no "deliverable", "✅", ETA,
   velocity, or point-in-time count survives in any doc (`git grep` clean for
   milestone framing).
1. The §3 tree exists, every file populated from its §4 sources.
1. No prose is duplicated across `README.md`, `CLAUDE.md`, and `docs/` — shared
   topics are single-homed and cross-linked.
1. Every internal link resolves (no dead relative links).
1. `README.md` quickstart commands are accurate and use build-tool run form with
   `$PWD`-prefixed inputs.
1. Spot-check: each new doc passes the §2 non-derivable filter (no section is
   pure code restatement).
1. Markdown conforms to `development/docs-style.md` (inline numeric markers, no
   `mdformat-footnote`).
1. `docs/superpowers/` is removed.

## 8. Sequencing note

`docs/superpowers/{specs,plans}` contains this spec and the implementation plan.
Its deletion (criterion §7.8) must be the **final** step of execution, after the
plan has been fully carried out — otherwise the plan deletes itself mid-run.

## 9. Out of scope

- Renaming the `bench/m3/` harness (code refactor; would ripple through
  BUILD/targets/scripts). Noted as an optional future follow-up.
- Fixing the ~25 pre-existing `-c dbg` lit failures.
- Any source-code change. This is a docs-only overhaul.
