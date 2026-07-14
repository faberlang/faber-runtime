# Tensor Runtime Substrate Inventory For Minimal Autograd Proof

This is an evidence note for the current internal `faber-runtime`
reverse-mode-autodiff-style proof. It does not claim PyTorch-equivalent
behavior, session integration, optimizer support, or host ABI gradient handles.

## Existing Substrate

| Area | Current support | Evidence |
| --- | --- | --- |
| Dense carrier | `Tensor<T>` stores homogeneous numeric data behind runtime shape metadata, row-major strides, and an offset. It is `Clone`, `Send`, and `Sync` when the element type is. | `src/tensor.rs`; `src/tensor_test.rs::tensor_is_send_sync_when_elements_are` |
| Shape and indexing | Rank, shape, element count, construction from flat data, reshape, flat offset calculation, get, set, and fill are implemented with explicit negative-dimension, negative-index, out-of-bounds, mismatch, and overflow checks. | `src/tensor.rs`; `src/tensor_test.rs`; `hosts/llvm/src/tensor.rs` |
| Elementwise arithmetic | Dense tensors support broadcast-compatible add, subtract, multiply, checked finite f32 division, and f32 scalar scaling. Broadcast mismatches fail closed with `ERR_BROADCAST_SHAPE`. Division rejects non-finite inputs, exact zero denominators, and non-finite results before any autograd VJP claim. | `Tensor::addita`, `Tensor::subtrahe`, `Tensor::multiplica`, `Tensor<f32>::divide`, `Tensor<f32>::reciproca`, `Tensor<f32>::scala`; `src/tensor_test.rs::addita_broadcasts_size_one_dimension`; `src/tensor_test.rs::divide_broadcasts_finite_f32_tensors`; `src/tensor_test.rs::reciproca_preserves_shape_and_checks_denominators`; `src/tensor_test.rs::scala_scales_f32_elements_and_preserves_shape` |
| Matmul and permutation | Dense tensors support rank-2 matrix multiply with receiver rank, argument rank, and inner-dimension errors. `Tensor::transpose_rank2` is the bounded materializing transpose primitive used by the Rust autograd scaffold's rank-2 matmul VJP. `Tensor::permute` materializes arbitrary axis order with dedicated rank, negative-axis, out-of-range-axis, and duplicate-axis diagnostics. Neither transpose nor permutation is exposed through the host ABI. | `Tensor::matmul`; `Tensor::transpose_rank2`; `Tensor::permute`; `src/tensor_test.rs::matmul_rectangular`; `src/tensor_test.rs::transpose_rank2_materializes_rows_as_columns`; `src/tensor_test.rs::permute_materializes_general_axis_order`; `src/host_abi_test.rs::host_abi_v1_does_not_expose_tensor_permute_or_transpose_symbols`; `src/autograd.rs::tests::backward_matches_rank2_matmul_sum_vjp` |
| Reductions | The Rust carrier exposes `summa` as an element-type sum and `Tensor<f32>::media` as a non-empty f32 mean. The LLVM host ABI also exposes `__faber_rt_v1_tensor_sum` and float-only `__faber_rt_v1_tensor_mean`; integer mean is rejected until conversion support is honest for that path. | `src/tensor.rs`; `src/tensor_test.rs::media_averages_f32_elements_and_rejects_empty_tensor`; `hosts/llvm/src/tensor.rs`; `hosts/llvm/src/lib_test.rs::tensor_arithmetic_family_adds_matmuls_and_reduces` |
| Views and materialization | Rust `sectio` returns an axis-0 view sharing the same `Arc<Mutex<Vec<T>>>`; parent and slice mutations alias. `materialize` copies logical data and breaks that alias. The LLVM host ABI materializes slices rather than exposing Rust view layout. | `src/tensor_test.rs::sectio_returns_axis_zero_view`; `src/tensor_test.rs::materialize_breaks_sectio_alias`; `hosts/llvm/src/tensor.rs` |
| Sparse bridge | `Sparsa<T>` stores non-default entries, reads absent entries as default, removes entries on default writes, and densifies to `Tensor<T>`. It has no sparse arithmetic kernels. | `src/sparsa.rs`; `src/sparsa_test.rs` |
| Packed numeric bridge | `PackedU4Block` records toy U4 layout facts, validates metadata, dequantizes to `Vec<f32>`, and materializes as rank-1 `Tensor<f32>`. The only tensor integration row is reference materialization into elementwise add. | `src/packed_numeric.rs`; `src/packed_numeric_test.rs::packed_u4_materialized_tensor_feeds_elementwise_add` |
| Autograd scaffold | The Rust runtime has an internal dense `Tensor<f32>` tape with node ids, parent edges, saved forward tensors, gradient accumulation, duplicate-parent accumulation, scalar-loss backward, broadcast reductions for add/sub/mul, rank-2 matmul VJP, tape-owned scalar `scala`, tape-owned `forma` reshape gradients, tape-owned materialized `permute` with inverse-permutation backward, broadcast-aware backward after `permute` for scalar/vector/size-one tensor operands, tape-owned axis-0 `sectio` with parent-gradient scatter-add, tape-owned `media` mean backward, and fail-closed leaf rejection for raw aliased `sectio` views. Materialized `sectio` copies are accepted and snapshotted like other leaves. | `src/autograd.rs`; `src/autograd.rs::tests::backward_matches_rank2_matmul_sum_vjp`; `src/autograd.rs::tests::backward_scales_tape_owned_scala_gradient_by_factor`; `src/autograd.rs::tests::backward_zero_scala_branch_does_not_mask_repeated_parent_gradient`; `src/autograd.rs::tests::backward_reshapes_tape_owned_forma_gradient_into_parent_shape`; `src/autograd.rs::tests::backward_inverts_tape_owned_permute_gradient_into_parent_shape`; `src/autograd.rs::tests::backward_reduces_scalar_broadcast_after_permute`; `src/autograd.rs::tests::backward_reduces_vector_broadcast_mul_after_permute`; `src/autograd.rs::tests::backward_reduces_tensor_broadcast_sub_after_permute`; `src/autograd.rs::tests::backward_scatter_adds_tape_owned_sectio_gradient_into_parent`; `src/autograd.rs::tests::backward_distributes_tape_owned_media_gradient_over_parent`; `src/autograd.rs::tests::autograd_rejects_raw_sectio_view_leaf`; `src/autograd.rs::tests::autograd_materialized_sectio_snapshot_ignores_parent_alias_mutation` |
| Finite-difference oracle | Test-only central-difference checks cover rank-0 scalar `x * x + x`, the exact rung-3 scalar target `loss(x, weight, target) = (x * weight - target)^2` with `x=2.0`, `weight=3.0`, `target=4.0`, `loss=4.0`, and `d_weight ~= 8.0`, same-shape vector, broadcast-bias, broadcast-scale, mean-square, and scalar-scaled mean-square losses, plus a dense linear training-step mean loss `media((XW + b - target)^2)` that compares autograd gradients for input, weight, and bias against CPU finite differences. A test-only `TestOnlySgdSession` oracle owns a flat parameter set with frozen input slots and trainable weight/bias slots, computes fresh per-step gradients, zeros frozen input gradients before update, applies manual `param -= learning_rate * grad` updates, compares updated parameters against the finite-difference session step, and checks a two-step loss trace matches the finite-difference trace while strictly decreasing. | `src/autograd_reference_test.rs` |
| ABI symbols | The host ABI names tensor creation, shape, get/set, fill, flatten, materialize, slice, add/sub/mul, matmul, sum, mean, conversion, and sparse new/get/set/nonzero/rank/densify/from-tensor symbols. | `src/host_abi.rs`; `hosts/llvm/src/lib.rs` |

## Autograd-Relevant Remaining Blockers

The current scaffold is suitable for local dense proof cases, but it is missing
the runtime machinery that would make gradients first-class across the wider
runtime:

- No backward kernels for mutation, sparse, packed, sessions, optimizers, or
  host ABI gradient handles.
- No public optimizer or session API. The only training/session boundary is the
  test-only `TestOnlySgdSession` oracle in `src/autograd_reference_test.rs`.
- No host ABI transpose/permutation symbol. Rank-2 matmul gradients are covered
  only inside the Rust autograd scaffold with `Tensor::transpose_rank2` for
  `dA = dY * B^T` and `dB = A^T * dY`; tape-owned `permute` is still
  runtime-local and not generated-gradient behavior.
- Checked finite division exists only for dense `Tensor<f32>` forward values.
  There is still no tape-owned division VJP. `Tensor<f32>::scala` is covered by
  the internal tape-owned scalar-scale VJP, and `Tensor<f32>::media` covers
  non-empty f32 mean.
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

## Dense Primitive Gap Matrix

This matrix ranks the next small internal dense proof units after the
test-only training/session oracle. It is a planning gate for `Tensor<f32>` and
the private autograd tape only; it does not create public optimizer/session,
host ABI gradient, generated-gradient, sparse/packed, or PyTorch-equivalence
claims.

| Priority | Gap | Current substrate | Smallest acceptance gate | Explicit non-claim |
| --- | --- | --- | --- | --- |
| 1 | Elementwise division / reciprocal scaling | `Tensor<f32>::divide` and `Tensor<f32>::reciproca` enforce checked finite forward division. `Tensor<f32>::scala` covers scalar multiplication, but there is no tape-owned division VJP. | Add tape-owned division/reciprocal backward tests against finite differences for mean-square or linear residual normalization, preserving zero/non-finite fail-closed behavior. | No broad arithmetic parity and no generated-gradient division rule until compiler/AIR owns an oracle. |
| 2 | Numeric unary primitives | `Tensor::forma` is the current unary proof because reshape already exists; no `neg`, `exp`, `log`, `sin`, or similar elementwise numeric Tensor operations exist. | Pick one primitive with a local derivative and a clear domain policy, starting with `neg` if the goal is shape-preserving linearity or `exp` only after non-finite behavior is specified; prove tensor forward plus tape VJP against finite differences. | No math-library surface, activation library, or PyTorch unary parity. |
| 3 | Training-loss reductions beyond scalar `summa` / non-empty f32 `media` | `summa` and `media` are covered for scalar-loss backward; current session oracle uses mean-squared loss. | Add one named reduction only if the Tensor forward API exists and its scalar seed rule is local; otherwise keep using `media((prediction-target)^2)` as the reference loss. | No optimizer/session API and no reduction host ABI gradient handle. |
| 4 | Higher-rank matmul / batched linear algebra | Rank-2 `Tensor::matmul` plus `transpose_rank2` support the current dense linear oracle. `Tensor::permute` is materialized and tape-owned only. | Do not widen until shape policy, broadcast semantics, and AIR/generated-gradient ownership are named; first gate should be a design/test packet, not code. | No broader matmul, device execution, or generated-gradient claim. |

## Division And Reciprocal Forward Gate

Division is implemented only as an internal dense `Tensor<f32>` forward
primitive. The selected policy is checked finite division, not IEEE
pass-through of `NaN`, `inf`, or `-inf` values.

`Tensor<f32>::divide` uses the same broadcast unification as add/sub/mul and
keeps `ERR_BROADCAST_SHAPE` for incompatible operand shapes.
`Tensor<f32>::reciproca` computes elementwise `1.0 / x` while preserving the
receiver shape. Inputs must be finite before division. An exact zero
denominator is an error, and the implementation must not materialize positive
or negative infinity for that case. A non-finite result, including overflow to
infinity or any `NaN`, is also an error.

Forward tests enforce finite result, zero denominator rejection, non-finite
input rejection, non-finite result rejection, shape preservation, and broadcast
mismatch behavior. Autograd support remains blocked. The next acceptance gate
is:

1. Add a tape-owned operation and finite-difference proof for the same
   primitive.
2. Scalar division by a finite non-zero constant should scale upstream
   gradients by the reciprocal constant.
3. Reciprocal-style `c / x` must prove the local derivative `-c / (x * x)`
   before being admitted.

This gate does not add a public optimizer/session API, host ABI gradient
handle, generated-gradient rule, sparse or packed division rule, or
PyTorch-equivalence claim.

## Current Proof Boundary

The honest proof boundary stays inside dense `Tensor<f32>` and avoids the LLVM
host ABI until the Rust-level invariant is broader:

1. Use only contiguous, materialized tensors created with `Tensor::structa` or
   by explicitly calling `materialize()` on a `sectio` view.
2. Restrict the proof graph to materialized dense elementwise add/sub/mul,
   broadcast add/sub/mul, rank-2 matmul, tape-owned scalar `scala`, tape-owned `forma`, materialized
   tape-owned `permute`, tape-owned axis-0 `sectio`, `summa`, and non-empty f32
   `media`, with no raw aliased `sectio` leaves, no mutation after graph
   capture, no sparse tensors, and no packed tensors.
   Broadcast-aware backward after `permute` is proven only for local dense
   scalar, vector, and size-one tensor broadcast operands.
3. Prove scalar-loss cases with local unit tests and finite-difference oracles
   before broadening generated-gradient claims. The current next-rung evidence
   is a dense mean-squared linear training step plus a test-only SGD session
   boundary with frozen inputs, fresh per-step gradients, manual weight/bias
   updates, and a two-step loss trace, not a production session, optimizer,
   training loop, or `torch.nn` parity claim.
4. Reuse local finite-difference tests by copying `planata()` values,
   perturbing one input element at a time, rebuilding tensors with `structa`,
   and comparing proof gradients against the oracle.

General axis permutation is now admitted only as materialized dense
`Tensor::permute` plus tape-owned inverse-permutation backward. Broader matmul
or generated-gradient claims remain gated on compiler/AIR integration and host
ABI design. Numeric unary primitives such as neg/exp/log/sin can follow once
their Tensor-level operations exist.

## Transpose And Permutation Policy

The current decision is to expose materializing dense Tensor primitives only:
`Tensor::transpose_rank2` for the rank-2 matmul VJP and `Tensor::permute` for
arbitrary axis order. Both operations use existing `Tensor` layout helpers:
logical view-aware reads, row-major materialization, checked result allocation,
and rank metadata. `permute` starts with materialization rather than a
non-contiguous view contract, so permuting a raw `sectio` view produces a copy
whose values do not alias later parent mutation.

`Tensor::permute` admits only a complete axis list: axis count must equal rank,
each axis must be non-negative and in range, and each axis may appear only once.
Missing axes are rejected by rank mismatch or duplicate-axis validation rather
than being silently dropped. Rank-0 tensors accept only `[]`.

The host ABI remains intentionally unchanged: no
`__faber_rt_v1_tensor_permute` or transpose symbol exists. Autograd support is
tape-owned only; `AutogradTape::permute` records the axis list and backward
applies the inverse permutation to the upstream gradient. This still keeps
optimizer/session APIs, sparse/packed tensors, device execution, generated
gradients, and PyTorch equivalence out of scope.

## General Axis Permutation Design Gate

The current packet admits materialized `Tensor::permute`; the remaining gate is
promotion beyond the Rust runtime-local proof. A general axis permutation is
small in loop mechanics but still broad in integration policy: it decides when
LLVM host ABI callers get a symbol, how generated code names the operation, and
how compiler-owned gradient checks prove the inverse-permutation rule.

Current `Tensor::permute(&[i64])` policy:

| Policy row | Current decision |
| --- | --- |
| Axis validation | Implemented with dedicated diagnostics for rank mismatch, negative axis, out-of-range axis, and duplicate axis. Rank-0 accepts only `[]`. |
| Materialization versus view | Implemented as materialization. Result tensors have fresh storage and row-major strides. |
| Shape and stride semantics | The result shape is `input.shape[axes[i]]` in the requested order. Source strides are read-only implementation detail used only while copying logical values. |
| ABI boundary | No `__faber_rt_v1_tensor_permute` or transpose symbol. `src/host_abi_test.rs::host_abi_v1_does_not_expose_tensor_permute_or_transpose_symbols` guards that non-exposure. |
| Backward policy | Implemented only for tape-owned `AutogradTape::permute`; backward materializes the upstream gradient through the inverse axis list into the parent shape. |
| Generated-gradient claim | Still absent. The primitive and tape proof do not imply AIR/compiler-owned gradient behavior or broader matmul claims. |

Failure rows covered by the current proof:

| Input condition | Current boundary |
| --- | --- |
| Axis list rank mismatch | Rejects with `ERR_PERMUTE_RANK` and AutogradTape records no node. |
| Negative axis | Rejects with `ERR_PERMUTE_NEGATIVE_AXIS`. |
| Axis greater than or equal to rank | Rejects with `ERR_PERMUTE_AXIS_OUT_OF_RANGE`. |
| Duplicate axis | Rejects with `ERR_PERMUTE_DUPLICATE_AXIS`; the missing-axis case cannot silently pass because a complete unique axis list is required. |
| Rank-0 tensor with non-empty axes | Rejects with `ERR_PERMUTE_RANK`; rank-0 accepts only `[]`. |
| Permuting a raw aliased view | Materializes fresh storage, so later parent mutation does not change the permuted result. |
| Tape-owned permute across tapes | Rejects cross-tape operands without recording a node, matching existing tape identity policy. |
| Unsupported host ABI call | No symbol exists; host callers must not claim permutation support. |

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
