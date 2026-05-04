/* Copyright 2026 The llzk-to-shlo Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef LLZK_TO_SHLO_DIALECT_WLA_WLA_TD_
#define LLZK_TO_SHLO_DIALECT_WLA_WLA_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def WLA_Dialect : Dialect {
  let name = "wla";
  let cppNamespace = "::mlir::llzk_to_shlo::wla";
  let summary =
    "Witness-Layout-Anchor: spec carrier for the canonical circom witness "
    "layout, consumed by the verify-witness-layout pass at pipeline end.";
  let description = [{
    The `wla` dialect carries a *spec* of where each circom signal lands in
    the flat witness tensor. It does not lower to anything and has no
    runtime semantics; it exists to make the witness layout
    machine-checkable. See `docs/WITNESS_LAYOUT_ANCHOR.md` for the full
    design and the contract the emit/verify passes share.
  }];
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

class WLA_Op<string mnemonic, list<Trait> traits = []>
    : Op<WLA_Dialect, mnemonic, traits>;

class WLA_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WLA_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def WLA_KindOutput   : I32EnumAttrCase<"Output",   0, "output">;
def WLA_KindInput    : I32EnumAttrCase<"Input",    1, "input">;
def WLA_KindInternal : I32EnumAttrCase<"Internal", 2, "internal">;

def WLA_SignalKind : I32EnumAttr<
    "SignalKind",
    "kind of circom signal a wla.layout entry refers to",
    [WLA_KindOutput, WLA_KindInput, WLA_KindInternal]> {
  let cppNamespace = "::mlir::llzk_to_shlo::wla";
  let genSpecializedAttr = 0;
}

def WLA_SignalKindAttr : EnumAttr<WLA_Dialect, WLA_SignalKind, "kind">;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def WLA_SignalAttr : WLA_Attr<"Signal", "signal"> {
  let summary = "One entry of a witness-layout schedule.";
  let description = [{
    Describes the position of a single circom signal in the flat witness
    tensor. The four fields name *what* the signal is and *where* it lives:

    - `name`: a textual reference to the signal in the LLZK input. By
      convention, struct-member signals use a leading `@` (e.g. `"@out"`,
      `"@xor_1"`); function-parameter signals use a leading `%` (e.g.
      `"%arg0"`). The exact spelling is established by the TRACK 3 anchor
      pass and is not enforced by this attribute's verifier.
    - `kind`: which class of circom signal this is — `output`, `input`, or
      `internal`. Used by the TRACK 4 verify pass to apply class-specific
      checks (e.g. inputs must be sourced from a function parameter slice;
      outputs from a `struct.member` write chain).
    - `offset`: zero-based start position in the flat witness tensor.
    - `length`: number of `pf` (prime field) elements the signal occupies.

    No invariants between attribute instances are enforced here; the
    op-level verifier on `wla.layout` handles single-entry sanity (length
    > 0, offset >= 0). Cross-entry invariants (no overlap, sorted by
    offset, full coverage) are documented in
    `docs/WITNESS_LAYOUT_ANCHOR.md` as the contract T3 emits and T4 may
    additionally check.
  }];
  let parameters = (ins
    StringRefParameter<"signal name in LLZK input">:$name,
    EnumParameter<WLA_SignalKind>:$kind,
    "int64_t":$offset,
    "int64_t":$length
  );
  let assemblyFormat = "`<` $name `,` $kind `,` `offset` `=` $offset `,`"
                       "    `length` `=` $length `>`";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

def WLA_LayoutOp : WLA_Op<"layout", [HasParent<"::mlir::ModuleOp">, Pure]> {
  let summary = "Module-scope spec of the canonical circom witness layout.";
  let description = [{
    The op has no operands, no results, and no runtime semantics. Its
    `signals` attribute is the spec: one `wla.signal` entry per circom
    signal, naming the offset and length each signal occupies in the flat
    witness tensor that `@main` returns at the end of the pipeline.

    The op verifier checks per-entry sanity only (`length > 0`,
    `offset >= 0`). Cross-entry invariants (single instance per module,
    sorted by offset, no overlap, canonical output/input/internal block
    order) are the emit pass's contract and the verify pass's
    enforcement target — see `docs/WITNESS_LAYOUT_ANCHOR.md`.
  }];

  let arguments = (ins
    TypedArrayAttrBase<WLA_SignalAttr, "array of wla.signal entries">:$signals
  );
  let results = (outs);

  let assemblyFormat = "`signals` `=` $signals attr-dict";
  let hasVerifier = 1;
}

#endif  // LLZK_TO_SHLO_DIALECT_WLA_WLA_TD_
