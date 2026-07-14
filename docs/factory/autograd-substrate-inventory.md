# Tensor Runtime Substrate Inventory For Minimal Autograd Proof

This is an evidence note for how close `faber-runtime` is to a small
reverse-mode-autodiff-style proof. It does not claim that the runtime has
autograd, gradient tracking, or PyTorch-equivalent behavior today.

## Existing Substrate

| Area | Current support | Evidence |
| --- | --- | --- |
| Dense carrier | `Tensor<T>` stores homogeneous numeric data behind runtime shape metadata, row-major strides, and an offset. It is `Clone`, `Send`, and `Sync` when the element type is. | `src/tensor.rs`; `src/tensor_test.rs::tensor_is_send_sync_when_elements_are` |
| Shape and indexing | Rank, shape, element count, construction from flat data, reshape, flat offset calculation, get, set, and fill are implemented with explicit negative-dimension, negative-index, out-of-bounds, mismatch, and overflow checks. | `src/tensor.rs`; `src/tensor_test.rs`; `hosts/llvm/src/tensor.rs` |
| Elementwise arithmetic | Dense tensors support broadcast-compatible add, subtract, and multiply. Broadcast mismatches fail closed with `ERR_BROADCAST_SHAPE`. | `Tensor::addita`, `Tensor::subtrahe`, `Tensor::multiplica`; `src/tensor_test.rs::addita_broadcasts_size_one_dimension` |
| Matmul | Dense tensors support rank-2 matrix multiply with receiver rank, argument rank, and inner-dimension errors. | `Tensor::matmul`; `src/tensor_test.rs::matmul_rectangular` |
| Reductions | The Rust carrier exposes `summa` as an element-type sum. The LLVM host ABI also exposes `__faber_rt_v1_tensor_sum` and float-only `__faber_rt_v1_tensor_mean`; integer mean is rejected until conversion support is honest for that path. | `src/tensor.rs`; `hosts/llvm/src/tensor.rs`; `hosts/llvm/src/lib_test.rs::tensor_arithmetic_family_adds_matmuls_and_reduces` |
| Views and materialization | Rust `sectio` returns an axis-0 view sharing the same `Arc<Mutex<Vec<T>>>`; parent and slice mutations alias. `materialize` copies logical data and breaks that alias. The LLVM host ABI materializes slices rather than exposing Rust view layout. | `src/tensor_test.rs::sectio_returns_axis_zero_view`; `src/tensor_test.rs::materialize_breaks_sectio_alias`; `hosts/llvm/src/tensor.rs` |
| Sparse bridge | `Sparsa<T>` stores non-default entries, reads absent entries as default, removes entries on default writes, and densifies to `Tensor<T>`. It has no sparse arithmetic kernels. | `src/sparsa.rs`; `src/sparsa_test.rs` |
| Packed numeric bridge | `PackedU4Block` records toy U4 layout facts, validates metadata, dequantizes to `Vec<f32>`, and materializes as rank-1 `Tensor<f32>`. The only tensor integration row is reference materialization into elementwise add. | `src/packed_numeric.rs`; `src/packed_numeric_test.rs::packed_u4_materialized_tensor_feeds_elementwise_add` |
| Finite-difference oracle | Test-only central-difference checks cover rank-0 scalar `x * x + x` and same-shape vector `summa((x * w - target) * (x * w - target))` losses using only materialized dense `Tensor<f32>` operations. | `src/autograd_reference_test.rs` |
| ABI symbols | The host ABI names tensor creation, shape, get/set, fill, flatten, materialize, slice, add/sub/mul, matmul, sum, mean, conversion, and sparse new/get/set/nonzero/rank/densify/from-tensor symbols. | `src/host_abi.rs`; `hosts/llvm/src/lib.rs` |

## Autograd-Relevant Blockers

The current substrate is suitable for forward numeric experiments, but it is
missing the runtime machinery that would make gradients first-class:

- No gradient tape, node id, operation graph, parent edge list, or saved
  forward metadata.
- No gradient tensor allocation/accumulation API, zeroing convention, or
  duplicate-parent accumulation rule.
- No backward kernels for add, sub, mul, matmul, sum, mean, reshape, slice, or
  broadcast unification.
- No broadcast-gradient reduction helper that reduces an upstream gradient back
  to an operand's original shape.
- No transpose or permutation primitive, which blocks the standard rank-2
  matmul gradients `dA = dY * B^T` and `dB = A^T * dY`.
- No elementwise division or scalar scaling API in the public `Tensor<T>`
  carrier; LLVM float mean handles division internally, but this is not a
  reusable tensor primitive.
- No alias policy for gradients through Rust `sectio` views. The existing view
  aliasing is intentional for mutation, but a gradient proof must decide whether
  views scatter-add into the parent gradient or whether the first proof forbids
  aliased inputs.
- No AIR/compiler-owned gradient-check harness yet. The repo now has a
  runtime-local finite-difference oracle for the first dense `Tensor<f32>`
  seed subset only.
- Sparse and packed numeric carriers are bridge materialization surfaces only
  for this purpose; they do not yet provide sparse or quantized gradient rules.

## Candidate First Proof

The smallest honest proof should stay inside dense `Tensor<f32>` and avoid the
LLVM host ABI until the Rust-level invariant is proven:

1. Use only contiguous, materialized tensors created with `Tensor::structa`.
2. Restrict the first graph to same-shape elementwise add/sub/mul plus
   `summa`, with no broadcasting, no `sectio`, no mutation after graph capture,
   no sparse tensors, and no packed tensors.
3. Prove one scalar-loss case such as
   `loss = summa((x * w + b) * (x * w + b))`.
4. Reuse the local finite-difference tests by copying `planata()` values,
   perturbing one input element at a time, rebuilding tensors with `structa`,
   and comparing future proof gradients against the oracle.

After that passes, the next promotion should add broadcast-gradient reduction
for add/mul and a transpose primitive before claiming matmul gradient coverage.
Mean can follow once scalar scaling/division is available as a reusable tensor
operation rather than only inside the LLVM host ABI mean helper.
