//! Scalar conversion into runtime-owned opaque `valor` handles.

use super::RuntimeContext;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC,
};
use faber::Valor;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Numerus(value))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_f64(
    context: *mut FaberRtContextV1,
    value: f64,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Fractus(value))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_i1(
    context: *mut FaberRtContextV1,
    value: u8,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Bivalens(value != 0))
}

fn store_valor(context: *mut FaberRtContextV1, value: Valor) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        let mut value = Box::new(value);
        let handle = std::ptr::from_mut(value.as_mut()).cast::<c_void>();
        runtime.valors.push(value);
        FaberRtPtrResultV1::success(handle)
    }))
    .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
