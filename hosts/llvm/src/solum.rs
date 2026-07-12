//! Filesystem boundaries used by LLVM-host `norma:solum` providers.

use super::format::{store_text, text_value};
use faber::host_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT,
    STATUS_IO_ERROR, STATUS_OK,
};

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_solum_read_text(
    context: *mut FaberRtContextV1,
    path: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(path) = text_value(path) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    match std::fs::read_to_string(path) {
        Ok(text) => store_text(context, text),
        Err(_) => FaberRtPtrResultV1::failure(STATUS_IO_ERROR),
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_solum_write_text(
    context: *mut FaberRtContextV1,
    path: *const FaberRtSliceV1,
    text: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    if context.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let (Some(path), Some(text)) = (text_value(path), text_value(text)) else {
        return STATUS_INVALID_ARGUMENT;
    };
    match std::fs::write(path, text) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_IO_ERROR,
    }
}
