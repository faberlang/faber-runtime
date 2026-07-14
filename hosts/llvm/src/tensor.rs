//! Arena-owned dense tensor carrier for the LLVM host ABI (Stages 4V–4W).
//!
//! Tensors store a typed flat element buffer plus an explicit shape. Views from
//! `sectio` materialize so the link surface stays honest without exposing Rust
//! layout. Element-width conversion and sparse remain residual families.

use super::array::{find_array, read_value, store_array, write_value, RuntimeArray, RuntimeValue};
use super::option::store_option;
use super::RuntimeContext;
use faber::host_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtStatusV1, FaberRtValueKindV1,
    STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC, VALUE_KIND_I64,
};
use faber::tensor::{
    tensor_flat_offset, tensor_shape_element_count, tensor_shape_has_element_count,
    ERR_INDEX_OUT_OF_BOUNDS, ERR_INVALID_SLICE_RANGE, ERR_NEGATIVE_DIM, ERR_NEGATIVE_INDEX,
    ERR_NEGATIVE_SLICE,
};
use faber::Tensor;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) struct RuntimeTensor {
    pub(super) kind: FaberRtValueKindV1,
    pub(super) shape: Vec<i64>,
    pub(super) data: Vec<RuntimeValue>,
}

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

/// Element kinds admitted by the LLVM host tensor ABI.
///
/// Keep this set aligned with `apply_binary` and `tensor_sum_value`: callers
/// should not be able to construct a tensor kind that fails only at the first
/// arithmetic or reduction boundary.
fn tensor_kind(kind: FaberRtValueKindV1) -> bool {
    matches!(
        kind,
        faber::host_abi::VALUE_KIND_F32
            | faber::host_abi::VALUE_KIND_F64
            | faber::host_abi::VALUE_KIND_I32
            | faber::host_abi::VALUE_KIND_I64
    )
}

fn default_value(kind: FaberRtValueKindV1) -> Option<RuntimeValue> {
    Some(match kind {
        faber::host_abi::VALUE_KIND_I32 => RuntimeValue::I32(0),
        faber::host_abi::VALUE_KIND_I64 => RuntimeValue::I64(0),
        faber::host_abi::VALUE_KIND_F32 => RuntimeValue::F32(0.0),
        faber::host_abi::VALUE_KIND_F64 => RuntimeValue::F64(0.0),
        _ => return None,
    })
}

pub(super) fn store_tensor(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    shape: Vec<i64>,
    data: Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let tensor = super::StableBox::new(RuntimeTensor { kind, shape, data });
    let handle = tensor.handle();
    runtime.tensors.push(tensor);
    FaberRtPtrResultV1::success(handle)
}

pub(super) fn find_tensor(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeTensor> {
    runtime
        .tensors
        .iter()
        .find(|tensor| std::ptr::eq(tensor.as_ref(), handle.cast()))
        .map(super::StableBox::as_ref)
}

fn find_tensor_mut(
    runtime: &mut RuntimeContext,
    handle: *mut c_void,
) -> Option<&mut RuntimeTensor> {
    runtime
        .tensors
        .iter_mut()
        .find(|tensor| std::ptr::eq(tensor.as_ref(), handle.cast()))
        .map(super::StableBox::as_mut)
}

fn shape_from_array(array: &RuntimeArray) -> Option<Vec<i64>> {
    if array.kind != VALUE_KIND_I64 {
        return None;
    }
    array
        .values
        .iter()
        .map(|value| match value {
            RuntimeValue::I64(dim) => Some(*dim),
            _ => None,
        })
        .collect()
}

fn validate_shape(shape: &[i64]) -> Result<usize, &'static str> {
    for dim in shape {
        if *dim < 0 {
            return Err(ERR_NEGATIVE_DIM);
        }
    }
    tensor_shape_element_count(shape).ok_or("tensor element count overflow")
}

fn indices_from_array(array: &RuntimeArray) -> Option<Vec<i64>> {
    shape_from_array(array).and_then(|indices| {
        if indices.iter().any(|index| *index < 0) {
            None
        } else {
            Some(indices)
        }
    })
}

fn flat_offset(shape: &[i64], indices: &[i64]) -> Result<usize, &'static str> {
    for index in indices {
        if *index < 0 {
            return Err(ERR_NEGATIVE_INDEX);
        }
    }
    tensor_flat_offset(shape, indices).ok_or(ERR_INDEX_OUT_OF_BOUNDS)
}

/// Rank-0 empty tensor (`vacua`) of the requested element kind.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_new(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(fill)) = (runtime(context), default_value(kind)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !tensor_kind(kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_tensor(runtime, kind, Vec::new(), vec![fill])
    })
}

/// Create a dense tensor filled with one scalar value.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_create(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
    fill: *const c_void,
    shape: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !tensor_kind(kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(fill) = (unsafe { read_value(kind, fill) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(shape) = find_array(runtime, shape).and_then(shape_from_array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Ok(count) = validate_shape(&shape) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_tensor(runtime, kind, shape, vec![fill; count])
    })
}

/// Build a tensor from a flat element lista and an i64 shape lista.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_from_flat(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
    data: *mut c_void,
    shape: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !tensor_kind(kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(data_array) = find_array(runtime, data) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if data_array.kind != kind {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(shape) = find_array(runtime, shape).and_then(shape_from_array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !tensor_shape_has_element_count(&shape, data_array.values.len()) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let data = data_array.values.clone();
        store_tensor(runtime, kind, shape, data)
    })
}

/// Tensor rank (`longitudo`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_rank(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(output) = (unsafe { output.as_mut() }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *output = tensor.shape.len() as i64;
        STATUS_OK
    })
}

/// Materialize shape as `lista<numerus>` (i64).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_shape(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let values = tensor
            .shape
            .iter()
            .copied()
            .map(RuntimeValue::I64)
            .collect();
        store_array(runtime, VALUE_KIND_I64, values)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_reshape(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    shape: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(shape) = find_array(runtime, shape).and_then(shape_from_array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !tensor_shape_has_element_count(&shape, tensor.data.len()) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_tensor(runtime, tensor.kind, shape, tensor.data.clone())
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_get(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    indices: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(indices) = find_array(runtime, indices).and_then(indices_from_array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let value = match flat_offset(&tensor.shape, &indices) {
            Ok(offset) => tensor.data.get(offset).copied(),
            Err(_) => None,
        };
        store_option(runtime, tensor.kind, value)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_set(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    indices: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(indices) = find_array(runtime, indices).and_then(indices_from_array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(tensor) = find_tensor_mut(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if tensor.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(offset) = flat_offset(&tensor.shape, &indices) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(slot) = tensor.data.get_mut(offset) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *slot = value;
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_fill(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(tensor) = find_tensor_mut(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if tensor.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        for slot in &mut tensor.data {
            *slot = value;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_flatten(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_array(runtime, tensor.kind, tensor.data.clone())
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_materialize(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_tensor(
            runtime,
            tensor.kind,
            tensor.shape.clone(),
            tensor.data.clone(),
        )
    })
}

/// Contiguous axis-0 slice `[start, end)`, materialized for link honesty.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_slice(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    start: i64,
    end: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if start < 0 || end < 0 {
            let _ = ERR_NEGATIVE_SLICE;
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        if end < start {
            let _ = ERR_INVALID_SLICE_RANGE;
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        if tensor.shape.is_empty() || end as usize > tensor.shape[0] as usize {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let mut shape = tensor.shape.clone();
        shape[0] = end - start;
        let row = tensor_shape_element_count(&tensor.shape[1..]).unwrap_or(1);
        let start_off = (start as usize).saturating_mul(row);
        let end_off = (end as usize).saturating_mul(row);
        if end_off > tensor.data.len() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let data = tensor.data[start_off..end_off].to_vec();
        store_tensor(runtime, tensor.kind, shape, data)
    })
}

fn binary_tensor_op(
    context: *mut FaberRtContextV1,
    lhs: *mut c_void,
    rhs: *mut c_void,
    op: BinaryOp,
) -> FaberRtPtrResultV1 {
    let Some(runtime) = runtime(context) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(left) = find_tensor(runtime, lhs) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(right) = find_tensor(runtime, rhs) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    if left.kind != right.kind {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    let kind = left.kind;
    let Some((shape, data)) = apply_binary(left, right, op) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    store_tensor(runtime, kind, shape, data)
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    MatMul,
}

fn apply_binary(
    left: &RuntimeTensor,
    right: &RuntimeTensor,
    op: BinaryOp,
) -> Option<(Vec<i64>, Vec<RuntimeValue>)> {
    match left.kind {
        faber::host_abi::VALUE_KIND_F32 => {
            let lhs = to_tensor_f32(left)?;
            let rhs = to_tensor_f32(right)?;
            let result = match op {
                BinaryOp::Add => lhs.addita(&rhs).ok()?,
                BinaryOp::Sub => lhs.subtrahe(&rhs).ok()?,
                BinaryOp::Mul => lhs.multiplica(&rhs).ok()?,
                BinaryOp::MatMul => lhs.matmul(&rhs).ok()?,
            };
            Some(from_tensor_f32(&result))
        }
        faber::host_abi::VALUE_KIND_F64 => {
            let lhs = to_tensor_f64(left)?;
            let rhs = to_tensor_f64(right)?;
            let result = match op {
                BinaryOp::Add => lhs.addita(&rhs).ok()?,
                BinaryOp::Sub => lhs.subtrahe(&rhs).ok()?,
                BinaryOp::Mul => lhs.multiplica(&rhs).ok()?,
                BinaryOp::MatMul => lhs.matmul(&rhs).ok()?,
            };
            Some(from_tensor_f64(&result))
        }
        faber::host_abi::VALUE_KIND_I64 => {
            let lhs = to_tensor_i64(left)?;
            let rhs = to_tensor_i64(right)?;
            let result = match op {
                BinaryOp::Add => lhs.addita(&rhs).ok()?,
                BinaryOp::Sub => lhs.subtrahe(&rhs).ok()?,
                BinaryOp::Mul => lhs.multiplica(&rhs).ok()?,
                BinaryOp::MatMul => lhs.matmul(&rhs).ok()?,
            };
            Some(from_tensor_i64(&result))
        }
        faber::host_abi::VALUE_KIND_I32 => {
            let lhs = to_tensor_i32(left)?;
            let rhs = to_tensor_i32(right)?;
            let result = match op {
                BinaryOp::Add => lhs.addita(&rhs).ok()?,
                BinaryOp::Sub => lhs.subtrahe(&rhs).ok()?,
                BinaryOp::Mul => lhs.multiplica(&rhs).ok()?,
                BinaryOp::MatMul => lhs.matmul(&rhs).ok()?,
            };
            Some(from_tensor_i32(&result))
        }
        _ => None,
    }
}

fn to_tensor_f32(tensor: &RuntimeTensor) -> Option<Tensor<f32>> {
    let data = tensor
        .data
        .iter()
        .map(|value| match value {
            RuntimeValue::F32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Tensor::structa(data, &tensor.shape).ok()
}

fn from_tensor_f32(tensor: &Tensor<f32>) -> (Vec<i64>, Vec<RuntimeValue>) {
    (
        tensor.magnitudines(),
        tensor
            .planata()
            .into_iter()
            .map(RuntimeValue::F32)
            .collect(),
    )
}

fn to_tensor_f64(tensor: &RuntimeTensor) -> Option<Tensor<f64>> {
    let data = tensor
        .data
        .iter()
        .map(|value| match value {
            RuntimeValue::F64(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Tensor::structa(data, &tensor.shape).ok()
}

fn from_tensor_f64(tensor: &Tensor<f64>) -> (Vec<i64>, Vec<RuntimeValue>) {
    (
        tensor.magnitudines(),
        tensor
            .planata()
            .into_iter()
            .map(RuntimeValue::F64)
            .collect(),
    )
}

fn to_tensor_i64(tensor: &RuntimeTensor) -> Option<Tensor<i64>> {
    let data = tensor
        .data
        .iter()
        .map(|value| match value {
            RuntimeValue::I64(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Tensor::structa(data, &tensor.shape).ok()
}

fn from_tensor_i64(tensor: &Tensor<i64>) -> (Vec<i64>, Vec<RuntimeValue>) {
    (
        tensor.magnitudines(),
        tensor
            .planata()
            .into_iter()
            .map(RuntimeValue::I64)
            .collect(),
    )
}

fn to_tensor_i32(tensor: &RuntimeTensor) -> Option<Tensor<i32>> {
    let data = tensor
        .data
        .iter()
        .map(|value| match value {
            RuntimeValue::I32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Tensor::structa(data, &tensor.shape).ok()
}

fn from_tensor_i32(tensor: &Tensor<i32>) -> (Vec<i64>, Vec<RuntimeValue>) {
    (
        tensor.magnitudines(),
        tensor
            .planata()
            .into_iter()
            .map(RuntimeValue::I32)
            .collect(),
    )
}

/// Elementwise add with broadcast.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_add(
    context: *mut FaberRtContextV1,
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| binary_tensor_op(context, lhs, rhs, BinaryOp::Add))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_sub(
    context: *mut FaberRtContextV1,
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| binary_tensor_op(context, lhs, rhs, BinaryOp::Sub))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_mul(
    context: *mut FaberRtContextV1,
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| binary_tensor_op(context, lhs, rhs, BinaryOp::Mul))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_matmul(
    context: *mut FaberRtContextV1,
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| binary_tensor_op(context, lhs, rhs, BinaryOp::MatMul))
}

/// Element-type fold (`summa`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_sum(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if tensor.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(value) = tensor_sum_value(tensor) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

/// Mean (`media`) in the element kind for float carriers.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_mean(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(tensor) = find_tensor(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if tensor.kind != kind || tensor.data.is_empty() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(value) = tensor_mean_value(tensor) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

fn tensor_sum_value(tensor: &RuntimeTensor) -> Option<RuntimeValue> {
    match tensor.kind {
        faber::host_abi::VALUE_KIND_F32 => Some(RuntimeValue::F32(to_tensor_f32(tensor)?.summa())),
        faber::host_abi::VALUE_KIND_F64 => Some(RuntimeValue::F64(to_tensor_f64(tensor)?.summa())),
        faber::host_abi::VALUE_KIND_I64 => Some(RuntimeValue::I64(to_tensor_i64(tensor)?.summa())),
        faber::host_abi::VALUE_KIND_I32 => Some(RuntimeValue::I32(to_tensor_i32(tensor)?.summa())),
        _ => None,
    }
}

fn tensor_mean_value(tensor: &RuntimeTensor) -> Option<RuntimeValue> {
    let n = tensor.data.len() as f64;
    if n == 0.0 {
        return None;
    }
    match tensor.kind {
        faber::host_abi::VALUE_KIND_F32 => {
            let sum = to_tensor_f32(tensor)?.summa();
            Some(RuntimeValue::F32((sum as f64 / n) as f32))
        }
        faber::host_abi::VALUE_KIND_F64 => {
            let sum = to_tensor_f64(tensor)?.summa();
            Some(RuntimeValue::F64(sum / n))
        }
        // Integer mean promotes to f64 carrier storage as f64 RuntimeValue is
        // wrong for i64 kind — reject integer mean until conversion family lands.
        _ => None,
    }
}

/// Element-width tensor conversion (`tensor ↦ tensor`) preserving shape.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tensor_convert(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    from_kind: FaberRtValueKindV1,
    to_kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !tensor_kind(from_kind) || !tensor_kind(to_kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(tensor) = find_tensor(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if tensor.kind != from_kind {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        if from_kind == to_kind {
            return store_tensor(runtime, to_kind, tensor.shape.clone(), tensor.data.clone());
        }
        let mut data = Vec::with_capacity(tensor.data.len());
        for value in &tensor.data {
            let Some(converted) = cast_runtime_value(*value, from_kind, to_kind) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            data.push(converted);
        }
        store_tensor(runtime, to_kind, tensor.shape.clone(), data)
    })
}

fn cast_runtime_value(
    value: RuntimeValue,
    from_kind: FaberRtValueKindV1,
    to_kind: FaberRtValueKindV1,
) -> Option<RuntimeValue> {
    // Mirror Rust `as` for numeric lattice cells used by tensor conversio.
    if matches!(
        to_kind,
        faber::host_abi::VALUE_KIND_F32
            | faber::host_abi::VALUE_KIND_F64
            | faber::host_abi::VALUE_KIND_F16
    ) {
        let float = value_as_f64(value, from_kind)?;
        return match to_kind {
            faber::host_abi::VALUE_KIND_F32 => Some(RuntimeValue::F32(float as f32)),
            faber::host_abi::VALUE_KIND_F64 => Some(RuntimeValue::F64(float)),
            faber::host_abi::VALUE_KIND_F16 => Some(RuntimeValue::F16(float as u16)),
            _ => None,
        };
    }
    let integer = value_as_i128(value, from_kind)?;
    match to_kind {
        faber::host_abi::VALUE_KIND_I1 => Some(RuntimeValue::I1(u8::from(integer != 0))),
        faber::host_abi::VALUE_KIND_I8 => Some(RuntimeValue::I8(integer as i8)),
        faber::host_abi::VALUE_KIND_I16 => Some(RuntimeValue::I16(integer as i16)),
        faber::host_abi::VALUE_KIND_I32 => Some(RuntimeValue::I32(integer as i32)),
        faber::host_abi::VALUE_KIND_I64 => Some(RuntimeValue::I64(integer as i64)),
        faber::host_abi::VALUE_KIND_U8 => Some(RuntimeValue::U8(integer as u8)),
        faber::host_abi::VALUE_KIND_U16 => Some(RuntimeValue::U16(integer as u16)),
        faber::host_abi::VALUE_KIND_U32 => Some(RuntimeValue::U32(integer as u32)),
        faber::host_abi::VALUE_KIND_U64 => Some(RuntimeValue::U64(integer as u64)),
        _ => None,
    }
}

fn value_as_f64(value: RuntimeValue, kind: FaberRtValueKindV1) -> Option<f64> {
    Some(match (kind, value) {
        (faber::host_abi::VALUE_KIND_I1, RuntimeValue::I1(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_I8, RuntimeValue::I8(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_I16, RuntimeValue::I16(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_I32, RuntimeValue::I32(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_I64, RuntimeValue::I64(v)) => v as f64,
        (faber::host_abi::VALUE_KIND_U8, RuntimeValue::U8(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_U16, RuntimeValue::U16(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_U32, RuntimeValue::U32(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_U64, RuntimeValue::U64(v)) => v as f64,
        (faber::host_abi::VALUE_KIND_F16, RuntimeValue::F16(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_F32, RuntimeValue::F32(v)) => f64::from(v),
        (faber::host_abi::VALUE_KIND_F64, RuntimeValue::F64(v)) => v,
        _ => return None,
    })
}

fn value_as_i128(value: RuntimeValue, kind: FaberRtValueKindV1) -> Option<i128> {
    Some(match (kind, value) {
        (faber::host_abi::VALUE_KIND_I1, RuntimeValue::I1(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_I8, RuntimeValue::I8(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_I16, RuntimeValue::I16(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_I32, RuntimeValue::I32(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_I64, RuntimeValue::I64(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_U8, RuntimeValue::U8(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_U16, RuntimeValue::U16(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_U32, RuntimeValue::U32(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_U64, RuntimeValue::U64(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_F16, RuntimeValue::F16(v)) => i128::from(v),
        (faber::host_abi::VALUE_KIND_F32, RuntimeValue::F32(v)) => v as i128,
        (faber::host_abi::VALUE_KIND_F64, RuntimeValue::F64(v)) => v as i128,
        _ => return None,
    })
}

pub(super) fn store_tensor_from_parts(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    shape: Vec<i64>,
    data: Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    store_tensor(runtime, kind, shape, data)
}

pub(super) fn tensor_to_runtime_values(
    tensor: &RuntimeTensor,
) -> Option<(Vec<i64>, Vec<RuntimeValue>)> {
    Some((tensor.shape.clone(), tensor.data.clone()))
}
