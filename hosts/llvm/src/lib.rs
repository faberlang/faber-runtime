#[cfg(not(test))]
use faber::llvm_abi::FaberRtExitV1;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_IO_ERROR,
    STATUS_OK, STATUS_PANIC,
};
use std::ffi::{c_char, c_int};
use std::io::{self, Write};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;

struct RuntimeContext {
    _arguments: Vec<Vec<u8>>,
}

/// Initialize one process-lifetime LLVM host context.
///
/// # Safety
///
/// `out_context` must be writable. When `argc` is positive, `argv` must point
/// to `argc` valid C strings. A successful context must be shut down exactly
/// once with [`__faber_rt_v1_shutdown`].
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_init(
    argc: c_int,
    argv: *const *const c_char,
    out_context: *mut *mut FaberRtContextV1,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if out_context.is_null() || argc < 0 || (argc > 0 && argv.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }
        let mut arguments = Vec::with_capacity(argc as usize);
        for index in 0..argc as usize {
            let value = *argv.add(index);
            if value.is_null() {
                return STATUS_INVALID_ARGUMENT;
            }
            arguments.push(std::ffi::CStr::from_ptr(value).to_bytes().to_vec());
        }
        let context = Box::new(RuntimeContext {
            _arguments: arguments,
        });
        *out_context = Box::into_raw(context).cast();
        STATUS_OK
    })
}

/// Release a context returned by [`__faber_rt_v1_init`].
///
/// # Safety
///
/// `context` must be null or a live context returned by this runtime and not
/// previously shut down.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_shutdown(context: *mut FaberRtContextV1) {
    if context.is_null() {
        return;
    }
    drop(panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(context.cast::<RuntimeContext>()));
        drop(io::stdout().flush());
        drop(io::stderr().flush());
    })));
}

/// Write one `nota` text payload followed by its canonical newline.
///
/// # Safety
///
/// `context` must be live. `text.data` must be readable for `text.len` bytes,
/// except that a null pointer is allowed when the length is zero.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_write_nota_text(
    context: *mut FaberRtContextV1,
    text: FaberRtSliceV1,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() || (text.len > 0 && text.data.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(len) = usize::try_from(text.len) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let bytes = if len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(text.data, len)
        };
        let mut stdout = io::stdout().lock();
        match stdout
            .write_all(bytes)
            .and_then(|()| stdout.write_all(b"\n"))
            .and_then(|()| stdout.flush())
        {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_IO_ERROR,
        }
    })
}

/// Emit a fatal diagnostic and abort without unwinding across the C boundary.
///
/// # Safety
///
/// The context and message slice follow the same validity requirements as
/// [`__faber_rt_v1_write_nota_text`]. This function never returns.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_fatal(
    context: *mut FaberRtContextV1,
    message: FaberRtSliceV1,
) -> ! {
    if !context.is_null() && (message.len == 0 || !message.data.is_null()) {
        if let Ok(len) = usize::try_from(message.len) {
            let bytes = if len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(message.data, len)
            };
            drop(io::stderr().write_all(bytes));
            drop(io::stderr().write_all(b"\n"));
            drop(io::stderr().flush());
        }
    }
    std::process::abort()
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

#[cfg(not(test))]
extern "C" {
    fn __faber_program_entry_v1(context: *mut FaberRtContextV1) -> FaberRtExitV1;
}

#[cfg(not(test))]
#[no_mangle]
/// C process entry owned by the LLVM host runtime.
///
/// # Safety
///
/// The platform launcher must provide the normal C `argc`/`argv` contract.
pub unsafe extern "C" fn main(argc: c_int, argv: *const *const c_char) -> c_int {
    let mut context = ptr::null_mut();
    let init = __faber_rt_v1_init(argc, argv, &mut context);
    if !init.is_ok() {
        return init.code;
    }
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| __faber_program_entry_v1(context)))
        .unwrap_or(FaberRtExitV1 {
            process_code: STATUS_PANIC.code,
            status: STATUS_PANIC,
        });
    __faber_rt_v1_shutdown(context);
    if outcome.status.is_ok() {
        outcome.process_code
    } else {
        outcome.status.code
    }
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
