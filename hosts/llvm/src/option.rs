//! Arena-owned scalar and opaque-handle options for the LLVM host ABI.

use super::array::{read_value, valid_kind, write_value, RuntimeValue};
use super::RuntimeContext;
use faber::host_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtStatusV1, FaberRtValueKindV1,
    STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) struct RuntimeOption {
    pub(super) kind: FaberRtValueKindV1,
    pub(super) value: Option<RuntimeValue>,
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_none(
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
        store_option(runtime, kind, None)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_some(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_option(runtime, kind, Some(value))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_is_present(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(option) = find_option(runtime, option) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if option.kind != kind || !(unsafe { write_u8(output, u8::from(option.value.is_some())) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_get(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(option) = find_option(runtime, option) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = option.value else {
            return STATUS_INVALID_ARGUMENT;
        };
        if option.kind != kind || !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_get_or(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    fallback: *const c_void,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(option) = find_option(runtime, option) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if option.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        let value = match option.value {
            Some(value) => value,
            None => {
                let Some(value) = (unsafe { read_value(kind, fallback) }) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                value
            }
        };
        if !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

pub(super) fn store_option(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    value: Option<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let mut option = Box::new(RuntimeOption { kind, value });
    let handle = std::ptr::from_mut(option.as_mut()).cast::<c_void>();
    runtime.options.push(option);
    FaberRtPtrResultV1::success(handle)
}

fn find_option(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeOption> {
    runtime
        .options
        .iter()
        .find(|option| std::ptr::eq(option.as_ref(), handle.cast_const().cast::<RuntimeOption>()))
        .map(Box::as_ref)
}

unsafe fn runtime_mut<'a>(context: *mut FaberRtContextV1) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

unsafe fn write_u8(output: *mut c_void, value: u8) -> bool {
    let output = output.cast::<u8>();
    if output.is_null() {
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
