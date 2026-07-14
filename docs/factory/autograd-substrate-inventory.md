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
| Matmul | Dense tensors support rank-2 matrix multiply with receiver rank, argument rank, and inner-dimension errors. The Rust autograd scaffold records rank-2 matmul and computes its dense VJP with a private transpose helper; this is not a public transpose/permutation primitive. | `Tensor::matmul`; `src/tensor_test.rs::matmul_rectangular`; `src/autograd.rs::tests::backward_matches_rank2_matmul_sum_vjp` |
| Reductions | The Rust carrier exposes `summa` as an element-type sum. The LLVM host ABI also exposes `__faber_rt_v1_tensor_sum` and float-only `__faber_rt_v1_tensor_mean`; integer mean is rejected until conversion support is honest for that path. | `src/tensor.rs`; `hosts/llvm/src/tensor.rs`; `hosts/llvm/src/lib_test.rs::tensor_arithmetic_family_adds_matmuls_and_reduces` |
| Views and materialization | Rust `sectio` returns an axis-0 view sharing the same `Arc<Mutex<Vec<T>>>`; parent and slice mutations alias. `materialize` copies logical data and breaks that alias. The LLVM host ABI materializes slices rather than exposing Rust view layout. | `src/tensor_test.rs::sectio_returns_axis_zero_view`; `src/tensor_test.rs::materialize_breaks_sectio_alias`; `hosts/llvm/src/tensor.rs` |
| Sparse bridge | `Sparsa<T>` stores non-default entries, reads absent entries as default, removes entries on default writes, and densifies to `Tensor<T>`. It has no sparse arithmetic kernels. | `src/sparsa.rs`; `src/sparsa_test.rs` |
| Packed numeric bridge | `PackedU4Block` records toy U4 layout facts, validates metadata, dequantizes to `Vec<f32>`, and materializes as rank-1 `Tensor<f32>`. The only tensor integration row is reference materialization into elementwise add. | `src/packed_numeric.rs`; `src/packed_numeric_test.rs::packed_u4_materialized_tensor_feeds_elementwise_add` |
| Autograd scaffold | The Rust runtime has an internal dense `Tensor<f32>` tape with node ids, parent edges, saved forward tensors, gradient accumulation, duplicate-parent accumulation, scalar-loss backward, broadcast reductions for add/sub/mul, rank-2 matmul VJP, tape-owned axis-0 `sectio` with parent-gradient scatter-add, and fail-closed leaf rejection for raw aliased `sectio` views. Materialized `sectio` copies are accepted and snapshotted like other leaves. | `src/autograd.rs`; `src/autograd.rs::tests::backward_matches_rank2_matmul_sum_vjp`; `src/autograd.rs::tests::backward_scatter_adds_tape_owned_sectio_gradient_into_parent`; `src/autograd.rs::tests::autograd_rejects_raw_sectio_view_leaf`; `src/autograd.rs::tests::autograd_materialized_sectio_snapshot_ignores_parent_alias_mutation` |
| Finite-difference oracle | Test-only central-difference checks cover rank-0 scalar `x * x + x`, the exact rung-3 scalar target `loss(x, weight, target) = (x * weight - target)^2` with `x=2.0`, `weight=3.0`, `target=4.0`, `loss=4.0`, and `d_weight ~= 8.0`, same-shape vector and broadcast-bias losses, plus a dense linear training-step loss `summa((XW + b - target)^2)` that compares autograd gradients for input, weight, and bias against CPU finite differences. A test-only `LinearTrainingSession` oracle applies manual `param -= learning_rate * grad` updates to weight and bias only, keeps input frozen, compares updated parameters against the finite-difference session step, and checks a two-step loss trace matches the finite-difference trace while strictly decreasing. | `src/autograd_reference_test.rs` |
| ABI symbols | The host ABI names tensor creation, shape, get/set, fill, flatten, materialize, slice, add/sub/mul, matmul, sum, mean, conversion, and sparse new/get/set/nonzero/rank/densify/from-tensor symbols. | `src/host_abi.rs`; `hosts/llvm/src/lib.rs` |

## Autograd-Relevant Remaining Blockers

The current scaffold is suitable for local dense proof cases, but it is missing
the runtime machinery that would make gradients first-class across the wider
runtime:

- No backward kernels for mean, reshape, mutation, sparse, packed, sessions,
  optimizers, or host ABI gradient handles.
- No public transpose or permutation primitive. Rank-2 matmul gradients are
  covered only inside the Rust autograd scaffold with a private dense transpose
  helper for `dA = dY * B^T` and `dB = A^T * dY`.
- No elementwise division or scalar scaling API in the public `Tensor<T>`
  carrier; LLVM float mean handles division internally, but this is not a
  reusable tensor primitive.
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
   broadcast add/sub/mul, rank-2 matmul, tape-owned axis-0 `sectio`, and
   `summa`, with no raw aliased `sectio` leaves, no mutation after graph
   capture, no sparse tensors, and no packed tensors.
3. Prove scalar-loss cases with local unit tests and finite-difference oracles
   before broadening generated-gradient claims. The current next-rung evidence
   is a dense linear training step plus a test-only session/update boundary
   with a two-step manual weight/bias loss trace, not a production session,
   optimizer, training loop, or `torch.nn` parity claim.
4. Reuse local finite-difference tests by copying `planata()` values,
   perturbing one input element at a time, rebuilding tensors with `structa`,
   and comparing proof gradients against the oracle.

After that passes, the next promotion should decide whether to expose a public
transpose primitive before broadening matmul beyond the internal rank-2 dense
scaffold. Mean can follow once scalar scaling/division is available as a
reusable tensor operation rather than only inside the LLVM host ABI mean helper.

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
