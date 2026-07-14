# Tensor Runtime Substrate Inventory For Minimal Autograd Proof

This is an evidence note for the current internal `faber-runtime`
reverse-mode-autodiff-style proof. It does not claim PyTorch-equivalent
behavior, session integration, optimizer support, or host ABI gradient handles.

## Existing Substrate

| Area | Current support | Evidence |
| --- | --- | --- |
| Dense carrier | `Tensor<T>` stores homogeneous numeric data behind runtime shape metadata, row-major strides, and an offset. It is `Clone`, `Send`, and `Sync` when the element type is. | `src/tensor.rs`; `src/tensor_test.rs::tensor_is_send_sync_when_elements_are` |
| Shape and indexing | Rank, shape, element count, construction from flat data, reshape, flat offset calculation, get, set, and fill are implemented with explicit negative-dimension, negative-index, out-of-bounds, mismatch, and overflow checks. | `src/tensor.rs`; `src/tensor_test.rs`; `hosts/llvm/src/tensor.rs` |
| Elementwise arithmetic | Dense tensors support broadcast-compatible add, subtract, and multiply. Broadcast mismatches fail closed with `ERR_BROADCAST_SHAPE`. | `Tensor::addita`, `Tensor::subtrahe`, `Tensor::multiplica`; `src/tensor_test.rs::addita_broadcasts_size_one_dimension` |
| Matmul | Dense tensors support rank-2 matrix multiply with receiver rank, argument rank, and inner-dimension errors. `Tensor::transpose_rank2` is now the bounded materializing transpose primitive used by the Rust autograd scaffold's rank-2 matmul VJP. This is not a general axis-permutation primitive and is not exposed through the host ABI. | `Tensor::matmul`; `Tensor::transpose_rank2`; `src/tensor_test.rs::matmul_rectangular`; `src/tensor_test.rs::transpose_rank2_materializes_rows_as_columns`; `src/autograd.rs::tests::backward_matches_rank2_matmul_sum_vjp` |
| Reductions | The Rust carrier exposes `summa` as an element-type sum and `Tensor<f32>::media` as a non-empty f32 mean. The LLVM host ABI also exposes `__faber_rt_v1_tensor_sum` and float-only `__faber_rt_v1_tensor_mean`; integer mean is rejected until conversion support is honest for that path. | `src/tensor.rs`; `src/tensor_test.rs::media_averages_f32_elements_and_rejects_empty_tensor`; `hosts/llvm/src/tensor.rs`; `hosts/llvm/src/lib_test.rs::tensor_arithmetic_family_adds_matmuls_and_reduces` |
| Views and materialization | Rust `sectio` returns an axis-0 view sharing the same `Arc<Mutex<Vec<T>>>`; parent and slice mutations alias. `materialize` copies logical data and breaks that alias. The LLVM host ABI materializes slices rather than exposing Rust view layout. | `src/tensor_test.rs::sectio_returns_axis_zero_view`; `src/tensor_test.rs::materialize_breaks_sectio_alias`; `hosts/llvm/src/tensor.rs` |
| Sparse bridge | `Sparsa<T>` stores non-default entries, reads absent entries as default, removes entries on default writes, and densifies to `Tensor<T>`. It has no sparse arithmetic kernels. | `src/sparsa.rs`; `src/sparsa_test.rs` |
| Packed numeric bridge | `PackedU4Block` records toy U4 layout facts, validates metadata, dequantizes to `Vec<f32>`, and materializes as rank-1 `Tensor<f32>`. The only tensor integration row is reference materialization into elementwise add. | `src/packed_numeric.rs`; `src/packed_numeric_test.rs::packed_u4_materialized_tensor_feeds_elementwise_add` |
| Autograd scaffold | The Rust runtime has an internal dense `Tensor<f32>` tape with node ids, parent edges, saved forward tensors, gradient accumulation, duplicate-parent accumulation, scalar-loss backward, broadcast reductions for add/sub/mul, rank-2 matmul VJP, tape-owned `forma` reshape gradients, tape-owned axis-0 `sectio` with parent-gradient scatter-add, tape-owned `media` mean backward, and fail-closed leaf rejection for raw aliased `sectio` views. Materialized `sectio` copies are accepted and snapshotted like other leaves. | `src/autograd.rs`; `src/autograd.rs::tests::backward_matches_rank2_matmul_sum_vjp`; `src/autograd.rs::tests::backward_reshapes_tape_owned_forma_gradient_into_parent_shape`; `src/autograd.rs::tests::backward_scatter_adds_tape_owned_sectio_gradient_into_parent`; `src/autograd.rs::tests::backward_distributes_tape_owned_media_gradient_over_parent`; `src/autograd.rs::tests::autograd_rejects_raw_sectio_view_leaf`; `src/autograd.rs::tests::autograd_materialized_sectio_snapshot_ignores_parent_alias_mutation` |
| Finite-difference oracle | Test-only central-difference checks cover rank-0 scalar `x * x + x`, the exact rung-3 scalar target `loss(x, weight, target) = (x * weight - target)^2` with `x=2.0`, `weight=3.0`, `target=4.0`, `loss=4.0`, and `d_weight ~= 8.0`, same-shape vector, broadcast-bias, broadcast-scale, and mean-square losses, plus a dense linear training-step mean loss `media((XW + b - target)^2)` that compares autograd gradients for input, weight, and bias against CPU finite differences. A test-only `TestOnlySgdSession` oracle owns a flat parameter set with frozen input slots and trainable weight/bias slots, computes fresh per-step gradients, zeros frozen input gradients before update, applies manual `param -= learning_rate * grad` updates, compares updated parameters against the finite-difference session step, and checks a two-step loss trace matches the finite-difference trace while strictly decreasing. | `src/autograd_reference_test.rs` |
| ABI symbols | The host ABI names tensor creation, shape, get/set, fill, flatten, materialize, slice, add/sub/mul, matmul, sum, mean, conversion, and sparse new/get/set/nonzero/rank/densify/from-tensor symbols. | `src/host_abi.rs`; `hosts/llvm/src/lib.rs` |

## Autograd-Relevant Remaining Blockers

The current scaffold is suitable for local dense proof cases, but it is missing
the runtime machinery that would make gradients first-class across the wider
runtime:

- No backward kernels for mutation, sparse, packed, sessions, optimizers, or
  host ABI gradient handles.
- No public optimizer or session API. The only training/session boundary is the
  test-only `TestOnlySgdSession` oracle in `src/autograd_reference_test.rs`.
- No general permutation primitive and no host ABI transpose/permutation symbol.
  Rank-2 matmul gradients are covered only inside the Rust autograd scaffold
  with `Tensor::transpose_rank2` for `dA = dY * B^T` and `dB = A^T * dY`.
- No generic elementwise division API in the public `Tensor<T>` carrier.
  `Tensor<f32>::scala` exists for scalar scaling, and `Tensor<f32>::media`
  covers non-empty f32 mean.
- No dedicated elementwise unary numeric primitives such as neg/exp/log/sin in
  `Tensor<T>` yet. The current unary autograd proof uses `forma` because the
  underlying tensor reshape operation already exists and has a local VJP.
- No general raw-view autograd policy for Rust `sectio` views. The current proof
  policy supports tape-owned axis-0 `sectio` from an existing `AutogradValue`
  and scatter-adds that gradient into the parent. Raw aliased views are still
  rejected at the autograd leaf boundary, while `sectio(...).materialize()` is
  allowed because it breaks storage aliasing and is snapshotted when recorded.
- No AIR/compiler-owned gradient-check harness yet. The repo has a
  runtime-local finite-difference oracle for dense `Tensor<f32>` seed subsets
  only; it is not generated-gradient behavior.
- Sparse and packed numeric carriers are bridge materialization surfaces only
  for this purpose; they do not yet provide sparse or quantized gradient rules.

## Current Proof Boundary

The honest proof boundary stays inside dense `Tensor<f32>` and avoids the LLVM
host ABI until the Rust-level invariant is broader:

1. Use only contiguous, materialized tensors created with `Tensor::structa` or
   by explicitly calling `materialize()` on a `sectio` view.
2. Restrict the proof graph to materialized dense elementwise add/sub/mul,
   broadcast add/sub/mul, rank-2 matmul, tape-owned `forma`, tape-owned axis-0
   `sectio`, `summa`, and non-empty f32 `media`, with no raw aliased `sectio`
   leaves, no mutation after graph capture, no sparse tensors, and no packed
   tensors.
3. Prove scalar-loss cases with local unit tests and finite-difference oracles
   before broadening generated-gradient claims. The current next-rung evidence
   is a dense mean-squared linear training step plus a test-only SGD session
   boundary with frozen inputs, fresh per-step gradients, manual weight/bias
   updates, and a two-step loss trace, not a production session, optimizer,
   training loop, or `torch.nn` parity claim.
4. Reuse local finite-difference tests by copying `planata()` values,
   perturbing one input element at a time, rebuilding tensors with `structa`,
   and comparing proof gradients against the oracle.

General axis permutation remains gated before broadening matmul beyond the
internal rank-2 dense scaffold. Numeric unary primitives such as neg/exp/log/sin
can follow once their Tensor-level operations exist.

## Transpose And Permutation Policy

The current decision is to expose only a bounded, materializing
`Tensor::transpose_rank2` primitive. The matmul VJP already required a rank-2
transpose, and the implementation is fully shaped by existing `Tensor` layout
helpers: logical view-aware reads, row-major materialization, checked result
allocation, and rank metadata. This removes the private autograd-only transpose
loop without introducing a second semantics for the same operation.

General permutation remains intentionally absent. A real `permute` primitive
needs axis validation, duplicate-axis diagnostics, shape/stride policy, view
versus materialized semantics, host ABI naming, and backward scatter behavior
before it can support broader generated-gradient claims. The new primitive
therefore proves only rank-2 transpose materialization for dense tensors and
keeps AutogradTape, optimizer/session APIs, sparse/packed tensors, device
execution, and PyTorch equivalence out of scope.

## General Axis Permutation Design Gate

The current decision is no broad `Tensor::permute` implementation in this
packet. A general axis permutation is small in loop mechanics but not small in
runtime policy: it decides whether permuted tensors are aliases or copies,
which errors become stable public surface, whether the LLVM host ABI needs a
symbol, and how autograd inverts the permutation during backward. Until those
contracts are admitted together, the only permutation-like public Tensor
operation is the materializing rank-2 `transpose_rank2`.

Admission criteria for a future `Tensor::permute(&[i64])`:

| Policy row | Required decision before implementation |
| --- | --- |
| Axis validation | Axis list length must equal tensor rank; every axis must be non-negative, in range, and appear exactly once. Rank-0 requires an empty axis list. Duplicate and missing axes should have dedicated diagnostics rather than collapsing into a generic shape mismatch. |
| Materialization versus view | Start with materialization unless a full non-contiguous view contract is designed. Materialization matches `forma`, `transpose_rank2`, and host slice behavior, avoids exposing arbitrary stride aliasing, and makes post-permute parent mutation unable to change the permuted value. |
| Shape and stride semantics | The result shape is `input.shape[axes[i]]` in the requested order. If materialized, result strides are normal row-major strides for that result shape; source strides are read-only implementation detail. |
| ABI boundary | Do not add `__faber_rt_v1_tensor_permute` until the Rust primitive has tests for validation, zero-sized dimensions, rank-0, views, and autograd inversion. The current host ABI non-claim remains: no transpose/permutation symbols. |
| Backward policy | Autograd support should be tape-owned only. The backward rule scatters or materializes the upstream gradient through the inverse permutation into the parent shape. Raw permuted views must remain rejected as leaves unless they are materialized and snapshotted. |
| Generated-gradient claim | No generated-gradient or broader matmul claim can depend on arbitrary permutation until the Tensor primitive, AutogradTape op, and finite-difference/reference cases are all present. |

Failure rows that should become tests with the future primitive:

| Input condition | Expected boundary |
| --- | --- |
| Axis list rank mismatch | Reject without allocating or recording an autograd node. |
| Negative axis | Reject with a specific negative-axis diagnostic. |
| Axis greater than or equal to rank | Reject with an out-of-range-axis diagnostic. |
| Duplicate axis | Reject with a duplicate-axis diagnostic. |
| Missing axis | Reject through the duplicate/mismatch validation rather than silently dropping a dimension. |
| Rank-0 tensor with non-empty axes | Reject; rank-0 accepts only `[]`. |
| Permuting a raw aliased view | If `Tensor::permute` materializes, the result must not alias the source; if a future view mode is added, raw view leaves remain fail-closed in autograd. |
| Tape-owned permute across tapes | Reject cross-tape operands without recording a node, matching existing tape identity policy. |
| Unsupported host ABI call | No symbol exists yet; host callers must not claim permutation support. |

## Raw And Tape-Owned `sectio` View Gradient Policy

Raw `Tensor::sectio` views remain fail-closed at `AutogradTape::leaf`. The
current `Tensor` view carries shared storage, shape, strides, offset, and a
`view` marker, but it does not carry autograd parent identity or the slice
operation that produced it. If `leaf` accepted such a raw view, the tape could
only create a detached leaf for the slice-shaped tensor; backward would have no
sound way to scatter the slice gradient into the parent tensor's gradient slot.

The supported proof path is `AutogradTape::sectio(parent, start, end)`, which
takes an existing `AutogradValue` parent, validates the same axis-0 bounds as
`Tensor::sectio`, records parent identity plus the start bound, returns a
slice-shaped `AutogradValue`, and scatter-adds upstream values into the parent
gradient at the recorded offsets. Current tests cover parent-gradient
scatter-add, overlapping view accumulation, invalid bounds without recording,
cross-tape rejection, raw-view leaf rejection, saved forward snapshots, and
continued acceptance of `sectio(...).materialize()` as a detached materialized
leaf. This remains an internal dense proof path, not a host ABI or generated
gradient claim.
