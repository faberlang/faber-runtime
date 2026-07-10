//! First-order Unicode text queries and transformations for LLVM-host handles.

use super::array::{RuntimeArray, RuntimeValue};
use super::format::{store_text, text_value};
use super::RuntimeContext;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT,
    STATUS_OK, STATUS_PANIC, VALUE_KIND_PTR,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn query(
    context: *mut FaberRtContextV1,
    out: *mut u8,
    operation: impl FnOnce() -> Option<bool>,
) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if context.is_null() || out.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(value) = operation() else {
            return STATUS_INVALID_ARGUMENT;
        };
        unsafe { *out = u8::from(value) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_is_empty(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || Some(text_value(text)?.is_empty()))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_contains(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    needle: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        Some(text_value(text)?.contains(&text_value(needle)?))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_starts_with(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    prefix: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        Some(text_value(text)?.starts_with(&text_value(prefix)?))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_ends_with(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    suffix: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        Some(text_value(text)?.ends_with(&text_value(suffix)?))
    })
}

fn transform(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    operation: impl FnOnce(String) -> String,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(text) = text_value(text) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, operation(text))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_uppercase(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| text.to_uppercase())
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_lowercase(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| text.to_lowercase())
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_trim(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| text.trim().to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_slice(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    start: i64,
    end: i64,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| {
        let start = usize::try_from(start.max(0)).unwrap_or(usize::MAX);
        let end = usize::try_from(end.max(0)).unwrap_or(usize::MAX);
        text.chars()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_replace(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    old: *const FaberRtSliceV1,
    new: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(old) = text_value(old) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(new) = text_value(new) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    transform(context, text, |text| text.replace(&old, &new))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_split(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    separator: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(text) = text_value(text) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(separator) = text_value(separator) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let mut values = Vec::new();
        for part in text.split(&separator) {
            let result = store_text(context, part.to_owned());
            if result.status != STATUS_OK {
                return result;
            }
            values.push(RuntimeValue::Ptr(result.value));
        }
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        let mut array = Box::new(RuntimeArray {
            kind: VALUE_KIND_PTR,
            values,
        });
        let handle = std::ptr::from_mut(array.as_mut()).cast::<c_void>();
        runtime.arrays.push(array);
        FaberRtPtrResultV1::success(handle)
    })
}
