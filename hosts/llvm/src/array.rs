//! Arena-owned typed arrays for the LLVM host ABI.

use super::RuntimeContext;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtStatusV1, FaberRtValueKindV1,
    STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC, VALUE_KIND_F32, VALUE_KIND_F64,
    VALUE_KIND_I1, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_PTR,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

#[derive(Clone, Copy)]
enum RuntimeValue {
    I1(u8),
    I8(i8),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ptr(*mut c_void),
}

pub(super) struct RuntimeArray {
    kind: FaberRtValueKindV1,
    values: Vec<RuntimeValue>,
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_new(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !valid_kind(kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let mut array = Box::new(RuntimeArray {
            kind,
            values: Vec::new(),
        });
        let handle = std::ptr::from_mut(array.as_mut()).cast::<c_void>();
        runtime.arrays.push(array);
        FaberRtPtrResultV1::success(handle)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_push(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        array.values.push(value);
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_extend(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    source: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(source_index) = find_array_index(runtime, source) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let source_kind = runtime.arrays[source_index].kind;
        let source_values = runtime.arrays[source_index].values.clone();
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != source_kind {
            return STATUS_INVALID_ARGUMENT;
        }
        array.values.extend(source_values);
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_length(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(length) = i64::try_from(array.values.len()) else {
            return STATUS_INVALID_ARGUMENT;
        };
        output.write(length);
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_get(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    index: i64,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(index) = usize::try_from(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = array.values.get(index).copied() else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind || !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_set(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    index: i64,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(index) = usize::try_from(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(slot) = array.values.get_mut(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        *slot = value;
        STATUS_OK
    })
}

fn valid_kind(kind: FaberRtValueKindV1) -> bool {
    matches!(
        kind,
        VALUE_KIND_I1
            | VALUE_KIND_I8
            | VALUE_KIND_I32
            | VALUE_KIND_I64
            | VALUE_KIND_F32
            | VALUE_KIND_F64
            | VALUE_KIND_PTR
    )
}

unsafe fn runtime_mut<'a>(context: *mut FaberRtContextV1) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn find_array(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeArray> {
    runtime
        .arrays
        .iter()
        .find(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
        .map(Box::as_ref)
}

fn find_array_mut(runtime: &mut RuntimeContext, handle: *mut c_void) -> Option<&mut RuntimeArray> {
    runtime
        .arrays
        .iter_mut()
        .find(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
        .map(Box::as_mut)
}

fn find_array_index(runtime: &RuntimeContext, handle: *mut c_void) -> Option<usize> {
    runtime
        .arrays
        .iter()
        .position(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
}

unsafe fn read_value(kind: FaberRtValueKindV1, value: *const c_void) -> Option<RuntimeValue> {
    Some(match kind {
        VALUE_KIND_I1 => RuntimeValue::I1(unsafe { read_typed(value) }?),
        VALUE_KIND_I8 => RuntimeValue::I8(unsafe { read_typed(value) }?),
        VALUE_KIND_I32 => RuntimeValue::I32(unsafe { read_typed(value) }?),
        VALUE_KIND_I64 => RuntimeValue::I64(unsafe { read_typed(value) }?),
        VALUE_KIND_F32 => RuntimeValue::F32(unsafe { read_typed(value) }?),
        VALUE_KIND_F64 => RuntimeValue::F64(unsafe { read_typed(value) }?),
        VALUE_KIND_PTR => RuntimeValue::Ptr(unsafe { read_typed(value) }?),
        _ => return None,
    })
}

unsafe fn write_value(value: RuntimeValue, output: *mut c_void) -> bool {
    match value {
        RuntimeValue::I1(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I8(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::Ptr(value) => unsafe { write_typed(output, value) },
    }
}

unsafe fn read_typed<T: Copy>(value: *const c_void) -> Option<T> {
    let value = value.cast::<T>();
    (!value.is_null() && value.is_aligned()).then(|| unsafe { value.read() })
}

unsafe fn write_typed<T>(output: *mut c_void, value: T) -> bool {
    let output = output.cast::<T>();
    if output.is_null() || !output.is_aligned() {
        return false;
    }
    unsafe { output.write(value) };
    true
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
