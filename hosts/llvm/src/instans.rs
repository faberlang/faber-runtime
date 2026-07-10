//! Precision-aware instans conversion through arena-owned handles.

use super::convert::with_valor;
use super::format::{store_text, text_value};
use super::RuntimeContext;
use faber::llvm_abi::*;
use faber::{Instans, InstansPraecisio, Valor};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

fn ffi(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn precision(value: FaberRtInstansPrecisionV1) -> Option<InstansPraecisio> {
    match value {
        INSTANS_PRECISION_SECONDS => Some(InstansPraecisio::Secunda),
        INSTANS_PRECISION_MILLIS => Some(InstansPraecisio::Millisecunda),
        INSTANS_PRECISION_MICROS => Some(InstansPraecisio::Microsecunda),
        INSTANS_PRECISION_NANOS => Some(InstansPraecisio::Nanosecunda),
        _ => None,
    }
}

fn store(runtime: &mut RuntimeContext, value: Instans) -> FaberRtPtrResultV1 {
    let mut value = Box::new(value);
    let handle = std::ptr::from_mut(value.as_mut()).cast::<c_void>();
    runtime.instants.push(value);
    FaberRtPtrResultV1::success(handle)
}

fn find(runtime: &RuntimeContext, handle: *mut c_void) -> Option<Instans> {
    runtime
        .instants
        .iter()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(|value| **value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_from_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
    requested: FaberRtInstansPrecisionV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let (Some(runtime), Some(text), Some(requested)) =
            (runtime(context), text_value(value), precision(requested))
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = Instans::try_from_valor(&Valor::Textus(text), requested) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store(runtime, value)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_from_valor(
    context: *mut FaberRtContextV1,
    valor: *const Valor,
    requested: FaberRtInstansPrecisionV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(requested) = precision(requested) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = with_valor(context, valor, |valor| {
            Instans::try_from_valor(valor, requested)
        }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store(runtime, value)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_retag(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    requested: FaberRtInstansPrecisionV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let (Some(runtime), Some(requested)) = (runtime(context), precision(requested)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = find(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store(runtime, value.ad_praecisionem(requested))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_get_text(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = find(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, value.to_rfc3339())
    })
}
