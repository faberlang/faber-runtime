# View / Alias Autograd Policy

**Document type:** Product law
**Authority:** G-A-01 Option A (AIR product authority)
**Scope:** Runtime tape (`faber-runtime` reverse-mode autograd scaffold)
**Status:** Active — documents the existing runtime boundary as written product
law for the deprecated tape path.
**Source evidence:** `faber-runtime/src/autograd.rs`, `faber-runtime/src/tensor.rs`,
`radix/crates/radix/src/semantic/passes/air_purity.rs`,
`radix/crates/radix/src/air/mir_backward_result_contract.rs`

---

## 1. Authority Ruling (G-A-01 Option A)

AIR is the product differentiation authority. The runtime reverse-mode tape
(`faber-runtime/src/autograd.rs`) is **deprecated** and survives only as an
oracle reference and debug path. All new view-related differentiation work
belongs in AIR-generated backward functions, not new tape-owned view operations.

**Constraint on this document:** This policy formalizes the tape's existing
fail-closed behavior. It does not authorize new tape-owned view operations,
new tape view kernels, or tape-level device gradient support. The existing
eight tape boundary tests are preserved as regression coverage for the
deprecated path but are not a foundation for future product AD.

---

## 2. View-Producing vs. Materializing Operations

### View-producing (single operation)

| Operation | Behavior | Code evidence |
| --- | --- | --- |
| `Tensor::sectio(axis=0, start, end)` | Creates a view with shared storage, shape, strides, offset, and `view: true` marker. Parent and slice mutations alias. | `tensor.rs:267-284` — returns `Self { data: Arc::clone(&self.data), shape, strides, offset, view: true }` |

### Materializing (always produce independent storage)

| Operation | Behavior | Code evidence |
| --- | --- | --- |
| `Tensor::forma(shape)` | Always materializes via `from_contiguous`. Never creates a view. | `tensor.rs:203-208` — calls `Self::from_contiguous(self.planata(), dims)`; no view marker set. |
| `Tensor::transpose_rank2()` | Materializes into a new contiguous buffer. | `tensor.rs:291-308` — allocates `Vec<f32>`, copies logical values row by row, returns `from_contiguous`. |
| `Tensor::permute(axes)` | Materializes into a new contiguous buffer. | `tensor.rs:316-330` — allocates `Vec<f32>`, copies via `value_at_logical`, returns `from_contiguous`. |
| `Tensor::materialize()` | Copies logical data into a new contiguous tensor, breaking storage aliasing. | `tensor.rs:285-287` — calls `Self::from_contiguous(self.planata(), self.shape.clone())`. |

### Fact-error correction (P1)

The P1 goal document for G-A-05 incorrectly claimed `Tensor::forma` "creates a
view when possible." This is false. `forma` always calls `from_contiguous`,
which allocates fresh storage and never sets `view: true`. The operation is
always materializing regardless of input layout or contiguity.

---

## 3. Autograd Leaf Boundary Rule

Raw views are rejected at `AutogradTape::leaf()`. The check is:

```rust
// autograd.rs:135-138
pub(crate) fn leaf(&mut self, tensor: Tensor<f32>) -> Result<AutogradValue, AutogradError> {
    if tensor.is_view() {
        return Err(AutogradError::Unsupported(UnsupportedAutogradOp::View));
    }
    Ok(self.record(AutogradOp::Leaf, Vec::new(), tensor))
}
```

**Error path:** `AutogradError::Unsupported(UnsupportedAutogradOp::View)`

**User guidance:** To use a view-derived tensor as an autograd leaf:
1. Call `.materialize()` first — creates an independent copy that breaks
   storage aliasing and is accepted as a normal leaf.
2. Or use `AutogradTape::sectio(parent, start, end)` to record the slice
   provenance through the tape (see Section 4).

The `is_view()` method (`tensor.rs:333-335`) returns the `view` field of the
`Tensor` struct. This field is set to `true` only by `Tensor::sectio` and is
never set by `from_contiguous`, `forma`, `transpose_rank2`, or `permute`.

---

## 4. Tape-Owned View Rule

`AutogradTape::sectio(parent, start, end)` is the only supported tape-owned view
operation.

### Forward semantics

```rust
// autograd.rs:247-254
pub(crate) fn sectio(
    &mut self,
    value: &AutogradValue,
    start: i64,
    end: i64,
) -> Result<AutogradValue, AutogradError> {
    self.ensure_member(value)?;
    let tensor = self
        .value(value.id)?
        .sectio(start, end)
        .map_err(AutogradError::Tensor)?
        .materialize();
    Ok(self.record(AutogradOp::Sectio { start }, vec![value.id], tensor))
}
```

The operation:
1. Validates the value belongs to this tape (`ensure_member`).
2. Forward-operates `Tensor::sectio(axis=0, start, end)` to create a view.
3. Immediately materializes the view via `.materialize()` — the forward value
   stored in the tape node is a contiguous copy, not a view.
4. Records the node with `AutogradOp::Sectio { start }` and a parent edge to
   the input `AutogradValue` identified by its id. The end offset is implicit
   from the slice length (end − start).

### Backward semantics

Backward scatter-adds the slice gradient into the parent gradient at the
recorded `start..start+len` offsets along axis 0. For a gradient tensor with
shape matching the slice, the scatter-add writes:

```
parent_grad[start..start+len, ...] += slice_grad
```

### Axis restriction

Only axis 0 slicing is supported. `Tensor::sectio` operates exclusively on
axis 0 (`tensor.rs:267-284`). There is no general axis slicing in the tape.

### Validation

The same validation as `Tensor::sectio`:
- Negative bounds → `ERR_NEGATIVE_SLICE`
- `end > axis-0 size` → `ERR_INDEX_OUT_OF_BOUNDS`
- `end < start` → `ERR_NEGATIVE_SLICE`
- Cross-tape parent → `AutogradError::ForeignTapeValue`

**On validation failure:** no tape node is recorded. The tape graph is not
polluted with invalid operations.

---

## 5. Consistency Statement (AIR)

AIR-generated backward functions have no view concept. AIR is pure-functional
with let-bindings and no mutation. The AIR purity pass
(`radix/crates/radix/src/semantic/passes/air_purity.rs`) rejects mutation
and guarantees that AIR programs are side-effect-free expression graphs.

The backward result contract
(`radix/crates/radix/src/air/mir_backward_result_contract.rs`) produces
`ordered-gradient-tuple-v0` where each field is a fresh gradient tensor
with independent storage — never a view into another tensor's storage.

**Implication:** Under G-A-01 Option A, the runtime tape's view policy is
legacy. All future gradient computation uses AIR-generated backward, which
operates on pure values with no view aliasing. The tape rule survives only
as an oracle for testing the compiler-generated backward against a known
reference implementation.

---

## 6. Non-Goals

The following are explicitly out of scope for this policy document and for the
runtime tape view boundary generally:

- General axis slicing (axis 0 only)
- Indexing views (single-element or fancy indexing)
- Transpose views (transpose always materializes)
- Broadcast views
- Device-side view gradient kernels (G-A-06 — deferred)
- Sparse or packed tensor view policy
- PyTorch view compatibility
- New tape-owned view operations beyond the existing `sectio`

---

## 7. Test Coverage

Eight tests at `faber-runtime/src/autograd.rs:1895-2015` prove the policy
boundary:

| Test name | Boundary proved |
| --- | --- |
| `autograd_rejects_raw_sectio_view_leaf` | Raw views rejected at leaf boundary. Constructs a `Tensor::sectio` view, passes it to `tape.leaf()`, asserts `UnsupportedAutogradOp::View` error and empty node list. |
| `autograd_accepts_materialized_copy_of_sectio_view` | Materialized views accepted as detached leaves. Creates a view, calls `.materialize()`, records as leaf. Asserts node count = 1 and correct shape. |
| `autograd_materialized_sectio_snapshot_ignores_parent_alias_mutation` | Materialized copy is a snapshot. Registers a materialized slice as leaf, builds a graph, mutates the original storage via `ponde`, runs backward, asserts gradient matches pre-mutation values. |
| `autograd_tape_owned_sectio_records_parent_edge_and_forward_value` | Tape-owned sectio records parent identity. Asserts node operation is `Sectio { start: 1 }`, parents contain base node id, and forward value has correct shape/data. |
| `backward_scatter_adds_tape_owned_sectio_gradient_into_parent` | Backward scatter-adds into parent at correct offsets. Builds a graph `sectio -> mul -> summa -> backward`, asserts base gradient has gradient values in the slice region and zeros elsewhere. |
| `backward_accumulates_overlapping_tape_owned_sectio_gradients` | Overlapping sectio gradients accumulate correctly. Creates two overlapping slices (rows 0..1 and 0..2), sums their losses, asserts base gradient has accumulated values per-row. |
| `autograd_tape_owned_sectio_rejects_invalid_bounds_without_recording_node` | Invalid bounds rejected without polluting the tape. Asserts `ERR_NEGATIVE_SLICE` for negative start, `ERR_INDEX_OUT_OF_BOUNDS` for end > axis-0, and no node count change. |
| `autograd_tape_owned_sectio_rejects_cross_tape_parent_without_recording_node` | Cross-tape parent rejected. Creates a foreign tape with a leaf, attempts `local_tape.sectio(&foreign, ...)`, asserts `ForeignTapeValue` error and no node count change. |

---

## 8. Error Message Prose

The view rejection error path is `AutogradError::Unsupported(UnsupportedAutogradOp::View)`.

The `UnsupportedAutogradOp` enum (`autograd.rs:49-53`):

```rust
pub(crate) enum UnsupportedAutogradOp {
    Mutation,
    View,
    HostAbi,
    Session,
}
```

When a raw view tensor (produced by `Tensor::sectio`) is passed to
`AutogradTape::leaf()`, the tape returns:

```
Err(AutogradError::Unsupported(UnsupportedAutogradOp::View))
```

**Meaning for the caller:** The tensor carries `view: true` with shared storage
aliasing another tensor. The tape cannot create a leaf with sound gradient
semantics because the view has no parent identity or slice provenance recorded.
Backward would have no way to scatter the leaf gradient into the parent's
storage.

**Recommended actions:**
1. **Materialize the view:** Call `.materialize()` on the view tensor to create
   an independent contiguous copy. The materialized tensor is accepted as a
   normal autograd leaf.
2. **Use tape-owned sectio:** If the view is a `Tensor::sectio(axis=0)` slice
   and you have access to the parent `AutogradValue`, call
   `AutogradTape::sectio(parent, start, end)` instead of `leaf()`. The tape
   records the slice provenance and scatter-adds gradients back to the parent.
3. **Build a different graph:** Restructure the computation to avoid views
   entering the autograd graph, using materialized tensors throughout.

The `AutogradError` enum (`autograd.rs:57-64`):

```rust
pub(crate) enum AutogradError {
    Tensor(&'static str),
    ShapeMismatch,
    MissingNode,
    ForeignTapeValue,
    BackwardRequiresScalar,
    Unsupported(UnsupportedAutogradOp),
}
```

This document is the canonical reference for the view rejection diagnostic.
All callers should cite this document when interpreting
`UnsupportedAutogradOp::View` errors.
