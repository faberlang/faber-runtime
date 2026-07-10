//! Scalar template formatting and runtime-owned LLVM text handles.

use super::RuntimeContext;
use faber::llvm_abi::{
    FaberRtContextV1, FaberRtPtrResultV1, FaberRtSliceV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: i64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(context, template, &[value.to_string()])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    first: i64,
    second: i64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(context, template, &[first.to_string(), second.to_string()])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64_i64_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    first: i64,
    second: i64,
    third: i64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(
        context,
        template,
        &[first.to_string(), second.to_string(), third.to_string()],
    )
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_f64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: f64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(context, template, &[value.to_string()])
}

fn format_scalar_values(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    args: &[String],
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        if context.is_null() || (template.len > 0 && template.data.is_null()) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Ok(len) = usize::try_from(template.len) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let bytes = if len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(template.data, len) }
        };
        let Ok(template) = std::str::from_utf8(bytes) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let rendered = render_template(template, args);
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        let mut text = Box::new(RuntimeText { _value: rendered });
        let handle = std::ptr::from_mut(text.as_mut()).cast::<c_void>();
        runtime.texts.push(text);
        FaberRtPtrResultV1::success(handle)
    })
}

fn render_template(template: &str, args: &[String]) -> String {
    let mut output = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    let mut next_arg = 0usize;
    while let Some(ch) = chars.next() {
        if ch != '§' {
            output.push(ch);
            continue;
        }
        let mut index = String::new();
        while chars.peek().is_some_and(char::is_ascii_digit) {
            if let Some(digit) = chars.next() {
                index.push(digit);
            }
        }
        let arg_index = if index.is_empty() {
            let current = next_arg;
            next_arg += 1;
            current
        } else {
            match index.parse::<usize>() {
                Ok(index) => index,
                Err(_) => usize::MAX,
            }
        };
        if let Some(value) = args.get(arg_index) {
            output.push_str(value);
        } else {
            output.push('§');
            output.push_str(&index);
        }
    }
    output
}
pub(super) struct RuntimeText {
    pub(super) _value: String,
}
