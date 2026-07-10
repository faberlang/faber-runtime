//! Stable C ABI shared by LLVM-emitted programs and the LLVM host runtime.

use core::ffi::c_void;

pub const ABI_VERSION: u32 = 1;

pub const SYMBOL_INIT: &str = "__faber_rt_v1_init";
pub const SYMBOL_SHUTDOWN: &str = "__faber_rt_v1_shutdown";
pub const SYMBOL_WRITE_NOTA_TEXT: &str = "__faber_rt_v1_write_nota_text";
pub const SYMBOL_ASSERT: &str = "__faber_rt_v1_assert";
pub const SYMBOL_ASSERT_MESSAGE: &str = "__faber_rt_v1_assert_message";
pub const SYMBOL_FATAL: &str = "__faber_rt_v1_fatal";
pub const SYMBOL_FATAL_OPAQUE: &str = "__faber_rt_v1_fatal_opaque";
pub const SYMBOL_FORMAT_I64: &str = "__faber_rt_v1_format_i64";
pub const SYMBOL_FORMAT_I64_I64: &str = "__faber_rt_v1_format_i64_i64";
pub const SYMBOL_FORMAT_I64_I64_I64: &str = "__faber_rt_v1_format_i64_i64_i64";
pub const SYMBOL_FORMAT_F64: &str = "__faber_rt_v1_format_f64";
pub const SYMBOL_TEXT_I64: &str = "__faber_rt_v1_text_i64";
pub const SYMBOL_TEXT_F64: &str = "__faber_rt_v1_text_f64";
pub const SYMBOL_TEXT_I1: &str = "__faber_rt_v1_text_i1";
pub const SYMBOL_VALOR_I64: &str = "__faber_rt_v1_valor_i64";
pub const SYMBOL_VALOR_F64: &str = "__faber_rt_v1_valor_f64";
pub const SYMBOL_VALOR_I1: &str = "__faber_rt_v1_valor_i1";
pub const SYMBOL_ARRAY_NEW: &str = "__faber_rt_v1_array_new";
pub const SYMBOL_ARRAY_PUSH: &str = "__faber_rt_v1_array_push";
pub const SYMBOL_ARRAY_EXTEND: &str = "__faber_rt_v1_array_extend";
pub const SYMBOL_ARRAY_LENGTH: &str = "__faber_rt_v1_array_length";
pub const SYMBOL_ARRAY_GET: &str = "__faber_rt_v1_array_get";
pub const SYMBOL_ARRAY_SET: &str = "__faber_rt_v1_array_set";
pub const SYMBOL_ARRAY_CLONE: &str = "__faber_rt_v1_array_clone";
pub const SYMBOL_ARRAY_CONTAINS: &str = "__faber_rt_v1_array_contains";
pub const SYMBOL_ARRAY_IS_EMPTY: &str = "__faber_rt_v1_array_is_empty";
pub const SYMBOL_ARRAY_REVERSE: &str = "__faber_rt_v1_array_reverse";
pub const SYMBOL_ARRAY_RANGE: &str = "__faber_rt_v1_array_range";
pub const SYMBOL_PROGRAM_ENTRY: &str = "__faber_program_entry_v1";

pub type FaberRtValueKindV1 = u32;
pub const VALUE_KIND_I1: FaberRtValueKindV1 = 1;
pub const VALUE_KIND_I8: FaberRtValueKindV1 = 2;
pub const VALUE_KIND_I32: FaberRtValueKindV1 = 3;
pub const VALUE_KIND_I64: FaberRtValueKindV1 = 4;
pub const VALUE_KIND_F32: FaberRtValueKindV1 = 5;
pub const VALUE_KIND_F64: FaberRtValueKindV1 = 6;
pub const VALUE_KIND_PTR: FaberRtValueKindV1 = 7;

pub type FaberRtArrayRangeModeV1 = u32;
pub const ARRAY_RANGE_SLICE: FaberRtArrayRangeModeV1 = 1;
pub const ARRAY_RANGE_TAKE: FaberRtArrayRangeModeV1 = 2;
pub const ARRAY_RANGE_TAKE_LAST: FaberRtArrayRangeModeV1 = 3;
pub const ARRAY_RANGE_DROP_FIRST: FaberRtArrayRangeModeV1 = 4;

pub const LLVM_SLICE_TYPE: &str = "%FaberRtSliceV1";
pub const LLVM_SLICE_TYPE_DEFINITION: &str = "%FaberRtSliceV1 = type { ptr, i64 }";
pub const LLVM_EXIT_TYPE: &str = "%FaberRtExitV1";
pub const LLVM_EXIT_TYPE_DEFINITION: &str = "%FaberRtExitV1 = type { i32, i32 }";
pub const LLVM_PTR_RESULT_TYPE: &str = "%FaberRtPtrResultV1";
pub const LLVM_PTR_RESULT_TYPE_DEFINITION: &str = "%FaberRtPtrResultV1 = type { i32, ptr }";

pub const STATUS_OK: FaberRtStatusV1 = FaberRtStatusV1 { code: 0 };
pub const STATUS_INVALID_ARGUMENT: FaberRtStatusV1 = FaberRtStatusV1 { code: 1 };
pub const STATUS_IO_ERROR: FaberRtStatusV1 = FaberRtStatusV1 { code: 2 };
pub const STATUS_PANIC: FaberRtStatusV1 = FaberRtStatusV1 { code: 3 };
pub const STATUS_UNSUPPORTED: FaberRtStatusV1 = FaberRtStatusV1 { code: 4 };

pub const DIAGNOSTIC_SYMBOLS_V1: &[(&str, &str, &str)] = &[
    ("nota", "ptr", "__faber_rt_v1_diagnostic_nota_ptr"),
    ("nota", "i64", "__faber_rt_v1_diagnostic_nota_i64"),
    ("nota", "i1", "__faber_rt_v1_diagnostic_nota_i1"),
    ("nota", "float", "__faber_rt_v1_diagnostic_nota_f32"),
    ("nota", "double", "__faber_rt_v1_diagnostic_nota_f64"),
    ("nota", "i8", "__faber_rt_v1_diagnostic_nota_i8"),
    ("nota", "i32", "__faber_rt_v1_diagnostic_nota_i32"),
    ("mone", "ptr", "__faber_rt_v1_diagnostic_mone_ptr"),
    ("mone", "i64", "__faber_rt_v1_diagnostic_mone_i64"),
    ("vide", "ptr", "__faber_rt_v1_diagnostic_vide_ptr"),
    ("vide", "i64", "__faber_rt_v1_diagnostic_vide_i64"),
];

pub fn diagnostic_symbol_v1(kind: &str, carrier: &str) -> Option<&'static str> {
    DIAGNOSTIC_SYMBOLS_V1
        .iter()
        .find(|(candidate_kind, candidate_carrier, _)| {
            *candidate_kind == kind && *candidate_carrier == carrier
        })
        .map(|(_, _, symbol)| *symbol)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtSliceV1 {
    pub data: *const u8,
    pub len: u64,
}

impl FaberRtSliceV1 {
    pub const fn from_static(bytes: &'static [u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len() as u64,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtStatusV1 {
    pub code: i32,
}

impl FaberRtStatusV1 {
    pub const fn is_ok(self) -> bool {
        self.code == STATUS_OK.code
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtPtrResultV1 {
    pub status: FaberRtStatusV1,
    pub value: *mut c_void,
}

impl FaberRtPtrResultV1 {
    pub const fn failure(status: FaberRtStatusV1) -> Self {
        Self {
            status,
            value: core::ptr::null_mut(),
        }
    }

    pub const fn success(value: *mut c_void) -> Self {
        Self {
            status: STATUS_OK,
            value,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtExitV1 {
    pub process_code: i32,
    pub status: FaberRtStatusV1,
}

impl FaberRtExitV1 {
    pub const SUCCESS: Self = Self {
        process_code: 0,
        status: STATUS_OK,
    };
}

/// Opaque process-lifetime runtime context. Only pointers cross the ABI.
#[repr(C)]
pub struct FaberRtContextV1 {
    _private: [u8; 0],
    _alignment: [*mut c_void; 0],
}

#[cfg(test)]
#[path = "llvm_abi_test.rs"]
mod tests;
