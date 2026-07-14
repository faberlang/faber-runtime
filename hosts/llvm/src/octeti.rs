//! Mutable opaque octeti operations and text conversion.

use super::array::RuntimeValue;
use super::format::{store_text, text_value};
use super::option::store_option;
use super::valor_aggregate::{find_octeti, store_octeti};
use super::RuntimeContext;
use faber::host_abi::*;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{self, AssertUnwindSafe};

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

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_append(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    value: u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(bytes) = runtime
            .octeti
            .iter_mut()
            .find(|bytes| std::ptr::eq(bytes.as_ref(), handle.cast()))
        else {
            return STATUS_INVALID_ARGUMENT;
        };
        bytes.push(value);
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_get(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    index: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let value = usize::try_from(index)
            .ok()
            .and_then(|index| find_octeti(runtime, handle)?.get(index).copied());
        store_option(runtime, VALUE_KIND_U8, value.map(RuntimeValue::U8))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_length(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(length) =
            find_octeti(runtime, handle).and_then(|bytes| i64::try_from(bytes.len()).ok())
        else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(output) = (unsafe { output.as_mut() }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *output = length;
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_from_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(value)) = (runtime(context), text_value(value)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_octeti(runtime, value.into_bytes())
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_from_ascii(
    context: *mut FaberRtContextV1,
    value: *const c_char,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if value.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let bytes = unsafe { CStr::from_ptr(value) }.to_bytes();
        if !bytes.is_ascii() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_octeti(runtime, bytes.to_vec())
    })
}

fn decode(context: *mut FaberRtContextV1, handle: *mut c_void, ascii: bool) -> FaberRtPtrResultV1 {
    let Some(runtime) = runtime(context) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(bytes) = find_octeti(runtime, handle) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    if (ascii && !bytes.is_ascii()) || std::str::from_utf8(bytes).is_err() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    if ascii {
        let mut owned = bytes.clone();
        owned.push(0);
        let owned = owned.into_boxed_slice();
        let pointer = owned.as_ptr().cast_mut().cast();
        runtime.ascii.push(super::StableBox::from_box(owned));
        FaberRtPtrResultV1::success(pointer)
    } else {
        store_text(context, String::from_utf8_lossy(bytes).into_owned())
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_get_text(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| decode(context, handle, false))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_get_ascii(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| decode(context, handle, true))
}
