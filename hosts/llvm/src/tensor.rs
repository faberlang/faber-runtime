//! Arena-owned dense tensor carrier for the LLVM host ABI (Stage 4V core).
//!
//! Tensors store a typed flat element buffer plus an explicit shape. Views from
//! `sectio` materialize so the link surface stays honest without exposing Rust
//! layout. Arithmetic and element-width conversion remain residual families.

use super::array::{
    find_array, read_value, store_array, valid_kind, RuntimeArray, RuntimeValue,
};
use super::option::store_option;
use super::RuntimeContext;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtStatusV1, FaberRtValueKindV1, STATUS_INVALID_ARGUMENT,
    STATUS_OK, STATUS_PANIC, VALUE_KIND_I64,
};
use faber::tensor::{
    tensor_flat_offset, tensor_shape_element_count, tensor_shape_has_element_count, ERR_INDEX_OUT_OF_BOUNDS,
    ERR_INVALID_SLICE_RANGE, ERR_NEGATIVE_DIM, ERR_NEGATIVE_INDEX, ERR_NEGATIVE_SLICE,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) struct RuntimeTensor {
    kind: FaberRtValueKindV1,
    shape: Vec<i64>,
    data: Vec<RuntimeValue>,
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

fn tensor_kind(kind: FaberRtValueKindV1) -> bool {
    valid_kind(kind)
        && !matches!(
            kind,
            faber::llvm_abi::VALUE_KIND_TEXT
                | faber::llvm_abi::VALUE_KIND_VALOR
                | faber::llvm_abi::VALUE_KIND_OPTION_I64
                | faber::llvm_abi::VALUE_KIND_INSTANS
                | faber::llvm_abi::VALUE_KIND_ASCII
                | faber::llvm_abi::VALUE_KIND_PTR
        )
}

fn default_value(kind: FaberRtValueKindV1) -> Option<RuntimeValue> {
    Some(match kind {
        faber::llvm_abi::VALUE_KIND_I1 => RuntimeValue::I1(0),
        faber::llvm_abi::VALUE_KIND_I8 => RuntimeValue::I8(0),
        faber::llvm_abi::VALUE_KIND_I16 => RuntimeValue::I16(0),
        faber::llvm_abi::VALUE_KIND_I32 => RuntimeValue::I32(0),
        faber::llvm_abi::VALUE_KIND_I64 => RuntimeValue::I64(0),
        faber::llvm_abi::VALUE_KIND_U8 => RuntimeValue::U8(0),
        faber::llvm_abi::VALUE_KIND_U16 => RuntimeValue::U16(0),
        faber::llvm_abi::VALUE_KIND_U32 => RuntimeValue::U32(0),
        faber::llvm_abi::VALUE_KIND_U64 => RuntimeValue::U64(0),
        faber::llvm_abi::VALUE_KIND_F16 => RuntimeValue::F16(0),
        faber::llvm_abi::VALUE_KIND_F32 => RuntimeValue::F32(0.0),
        faber::llvm_abi::VALUE_KIND_F64 => RuntimeValue::F64(0.0),
        _ => return None,
    })
}

fn store_tensor(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    shape: Vec<i64>,
    data: Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let mut tensor = Box::new(RuntimeTensor { kind, shape, data });
    let handle = std::ptr::from_mut(tensor.as_mut()).cast::<c_void>();
    runtime.tensors.push(tensor);
    FaberRtPtrResultV1::success(handle)
}

fn find_tensor(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeTensor> {
    runtime
        .tensors
        .iter()
        .find(|tensor| std::ptr::eq(tensor.as_ref(), handle.cast()))
        .map(Box::as_ref)
}

fn find_tensor_mut(runtime: &mut RuntimeContext, handle: *mut c_void) -> Option<&mut RuntimeTensor> {
    runtime
        .tensors
        .iter_mut()
        .find(|tensor| std::ptr::eq(tensor.as_ref(), handle.cast()))
        .map(Box::as_mut)
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
        store_tensor(runtime, tensor.kind, tensor.shape.clone(), tensor.data.clone())
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
