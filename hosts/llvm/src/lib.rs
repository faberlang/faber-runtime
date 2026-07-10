mod array;
mod array_numeric;
mod collection_map;
mod convert;
mod format;
mod option;
mod text;
mod valor_aggregate;

use array::RuntimeArray;
#[cfg(test)]
use array::{
    __faber_rt_v1_array_clone, __faber_rt_v1_array_contains, __faber_rt_v1_array_extend,
    __faber_rt_v1_array_get, __faber_rt_v1_array_is_empty, __faber_rt_v1_array_length,
    __faber_rt_v1_array_new, __faber_rt_v1_array_option, __faber_rt_v1_array_push,
    __faber_rt_v1_array_range, __faber_rt_v1_array_reverse, __faber_rt_v1_array_set,
};
#[cfg(test)]
use array_numeric::{__faber_rt_v1_array_sort, __faber_rt_v1_array_sum};
#[cfg(test)]
use collection_map::{
    __faber_rt_v1_map_contains, __faber_rt_v1_map_delete, __faber_rt_v1_map_is_empty,
    __faber_rt_v1_map_keys, __faber_rt_v1_map_length, __faber_rt_v1_map_new,
    __faber_rt_v1_map_option, __faber_rt_v1_map_put, __faber_rt_v1_map_values,
    __faber_rt_v1_set_add, __faber_rt_v1_set_contains, __faber_rt_v1_set_delete,
    __faber_rt_v1_set_difference, __faber_rt_v1_set_intersection, __faber_rt_v1_set_is_empty,
    __faber_rt_v1_set_is_subset, __faber_rt_v1_set_is_superset, __faber_rt_v1_set_length,
    __faber_rt_v1_set_new, __faber_rt_v1_set_symmetric_difference, __faber_rt_v1_set_union,
};
use collection_map::{RuntimeMap, RuntimeSet};
#[cfg(test)]
use convert::{
    __faber_rt_v1_valor_ascii, __faber_rt_v1_valor_f64, __faber_rt_v1_valor_get_ascii,
    __faber_rt_v1_valor_get_f64, __faber_rt_v1_valor_get_i1, __faber_rt_v1_valor_get_i64,
    __faber_rt_v1_valor_get_nihil, __faber_rt_v1_valor_get_text, __faber_rt_v1_valor_i1,
    __faber_rt_v1_valor_i64, __faber_rt_v1_valor_nihil, __faber_rt_v1_valor_text,
};
#[cfg(not(test))]
use faber::llvm_abi::FaberRtExitV1;
#[cfg(test)]
use faber::llvm_abi::FaberRtPtrResultV1;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_IO_ERROR,
    STATUS_OK, STATUS_PANIC, STATUS_UNSUPPORTED,
};
#[cfg(test)]
use faber::llvm_abi::{
    ARRAY_OPTION_FIRST, ARRAY_OPTION_INDEX, ARRAY_OPTION_LAST, ARRAY_OPTION_REMOVE_FIRST,
    ARRAY_OPTION_REMOVE_LAST, ARRAY_RANGE_DROP_FIRST, ARRAY_RANGE_SLICE, ARRAY_RANGE_TAKE,
    ARRAY_RANGE_TAKE_LAST, VALUE_KIND_F16, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1,
    VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_PTR, VALUE_KIND_TEXT,
    VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_U8,
};
use faber::Valor;
use format::RuntimeText;
#[cfg(test)]
use format::{
    __faber_rt_v1_format_f64, __faber_rt_v1_format_i64, __faber_rt_v1_format_i64_i64,
    __faber_rt_v1_format_i64_i64_i64, __faber_rt_v1_format_text, __faber_rt_v1_format_text_i64,
    __faber_rt_v1_format_text_text, __faber_rt_v1_text_f64, __faber_rt_v1_text_i1,
    __faber_rt_v1_text_i64, __faber_rt_v1_text_length,
};
use option::RuntimeOption;
#[cfg(test)]
use option::{
    __faber_rt_v1_option_get, __faber_rt_v1_option_get_or, __faber_rt_v1_option_is_present,
    __faber_rt_v1_option_none, __faber_rt_v1_option_some,
};
use std::ffi::{c_char, c_int};
use std::fmt::Display;
use std::io::{self, Write};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
#[cfg(test)]
use text::{
    __faber_rt_v1_text_contains, __faber_rt_v1_text_ends_with, __faber_rt_v1_text_is_empty,
    __faber_rt_v1_text_lowercase, __faber_rt_v1_text_parse_float, __faber_rt_v1_text_parse_integer,
    __faber_rt_v1_text_replace, __faber_rt_v1_text_slice, __faber_rt_v1_text_split,
    __faber_rt_v1_text_starts_with, __faber_rt_v1_text_trim, __faber_rt_v1_text_truthy,
    __faber_rt_v1_text_uppercase,
};
#[cfg(test)]
use valor_aggregate::{
    __faber_rt_v1_octeti_new, __faber_rt_v1_valor_array, __faber_rt_v1_valor_get_array,
    __faber_rt_v1_valor_get_map, __faber_rt_v1_valor_get_octeti, __faber_rt_v1_valor_map,
    __faber_rt_v1_valor_octeti,
};

struct RuntimeContext {
    _arguments: Vec<Vec<u8>>,
    texts: Vec<Box<RuntimeText>>,
    valors: Vec<Box<Valor>>,
    ascii: Vec<Box<[u8]>>,
    octeti: Vec<Box<[u8]>>,
    arrays: Vec<Box<RuntimeArray>>,
    options: Vec<Box<RuntimeOption>>,
    maps: Vec<Box<RuntimeMap>>,
    sets: Vec<Box<RuntimeSet>>,
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
            texts: Vec::new(),
            valors: Vec::new(),
            ascii: Vec::new(),
            octeti: Vec::new(),
            arrays: Vec::new(),
            options: Vec::new(),
            maps: Vec::new(),
            sets: Vec::new(),
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

/// Evaluate one assertion without allowing a panic to cross the C ABI.
///
/// # Safety
///
/// `context` must be live.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_assert(
    context: *mut FaberRtContextV1,
    condition: u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() {
            STATUS_INVALID_ARGUMENT
        } else if condition == 0 {
            STATUS_PANIC
        } else {
            STATUS_OK
        }
    })
}

/// Evaluate one assertion and report its literal message on failure.
///
/// # Safety
///
/// `context` must be live. `message` follows the slice validity contract of
/// [`__faber_rt_v1_write_nota_text`].
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_assert_message(
    context: *mut FaberRtContextV1,
    condition: u8,
    message: FaberRtSliceV1,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() || (message.len > 0 && message.data.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }
        if condition != 0 {
            return STATUS_OK;
        }
        let Ok(len) = usize::try_from(message.len) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let bytes = if len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(message.data, len)
        };
        let mut stderr = io::stderr().lock();
        match stderr
            .write_all(bytes)
            .and_then(|()| stderr.write_all(b"\n"))
            .and_then(|()| stderr.flush())
        {
            Ok(()) => STATUS_PANIC,
            Err(_) => STATUS_IO_ERROR,
        }
    })
}

fn write_diagnostic(
    context: *mut FaberRtContextV1,
    stderr: bool,
    value: impl Display,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let result = if stderr {
            let mut output = io::stderr().lock();
            writeln!(output, "{value}").and_then(|()| output.flush())
        } else {
            let mut output = io::stdout().lock();
            writeln!(output, "{value}").and_then(|()| output.flush())
        };
        match result {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_IO_ERROR,
        }
    })
}

fn unsupported_opaque_diagnostic(context: *mut FaberRtContextV1) -> FaberRtStatusV1 {
    if context.is_null() {
        STATUS_INVALID_ARGUMENT
    } else {
        STATUS_UNSUPPORTED
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_ptr(
    context: *mut FaberRtContextV1,
    _value: *const u8,
) -> FaberRtStatusV1 {
    unsupported_opaque_diagnostic(context)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i1(
    context: *mut FaberRtContextV1,
    value: u8,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value != 0)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_f32(
    context: *mut FaberRtContextV1,
    value: f32,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_f64(
    context: *mut FaberRtContextV1,
    value: f64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i8(
    context: *mut FaberRtContextV1,
    value: i8,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i32(
    context: *mut FaberRtContextV1,
    value: i32,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_ptr(
    context: *mut FaberRtContextV1,
    _value: *const u8,
) -> FaberRtStatusV1 {
    unsupported_opaque_diagnostic(context)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, true, value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_ptr(
    context: *mut FaberRtContextV1,
    _value: *const u8,
) -> FaberRtStatusV1 {
    unsupported_opaque_diagnostic(context)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
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

/// Abort for a message whose opaque runtime representation has no byte-length
/// contract at this ABI boundary.
///
/// # Safety
///
/// `context` must be live. `message` is intentionally never dereferenced.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_fatal_opaque(
    context: *mut FaberRtContextV1,
    _message: *const u8,
) -> ! {
    if !context.is_null() {
        drop(io::stderr().write_all(b"fatal error\n"));
        drop(io::stderr().flush());
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
