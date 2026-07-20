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
pub const SYMBOL_FORMAT_I1: &str = "__faber_rt_v1_format_i1";
pub const SYMBOL_FORMAT_I64_I64: &str = "__faber_rt_v1_format_i64_i64";
pub const SYMBOL_FORMAT_I64_I64_I64: &str = "__faber_rt_v1_format_i64_i64_i64";
pub const SYMBOL_FORMAT_F64: &str = "__faber_rt_v1_format_f64";
pub const SYMBOL_FORMAT_TEXT: &str = "__faber_rt_v1_format_text";
pub const SYMBOL_FORMAT_TEXT_TEXT: &str = "__faber_rt_v1_format_text_text";
pub const SYMBOL_FORMAT_TEXT_I64: &str = "__faber_rt_v1_format_text_i64";
pub const SYMBOL_FORMAT_I64_TEXT: &str = "__faber_rt_v1_format_i64_text";
pub const SYMBOL_FORMAT_TEXT_TEXT_TEXT: &str = "__faber_rt_v1_format_text_text_text";
pub const SYMBOL_FORMAT_TEXT_I64_I1: &str = "__faber_rt_v1_format_text_i64_i1";
pub const SYMBOL_TEXT_LENGTH: &str = "__faber_rt_v1_text_length";
pub const SYMBOL_TEXT_CONCAT: &str = "__faber_rt_v1_text_concat";
pub const SYMBOL_TEXT_IS_EMPTY: &str = "__faber_rt_v1_text_is_empty";
pub const SYMBOL_TEXT_CONTAINS: &str = "__faber_rt_v1_text_contains";
pub const SYMBOL_TEXT_STARTS_WITH: &str = "__faber_rt_v1_text_starts_with";
pub const SYMBOL_TEXT_ENDS_WITH: &str = "__faber_rt_v1_text_ends_with";
pub const SYMBOL_TEXT_UPPERCASE: &str = "__faber_rt_v1_text_uppercase";
pub const SYMBOL_TEXT_LOWERCASE: &str = "__faber_rt_v1_text_lowercase";
pub const SYMBOL_TEXT_TRIM: &str = "__faber_rt_v1_text_trim";
pub const SYMBOL_TEXT_SLICE: &str = "__faber_rt_v1_text_slice";
pub const SYMBOL_TEXT_SPLIT: &str = "__faber_rt_v1_text_split";
pub const SYMBOL_TEXT_REPLACE: &str = "__faber_rt_v1_text_replace";
pub const SYMBOL_TEXT_PARSE_INTEGER: &str = "__faber_rt_v1_text_parse_integer";
pub const SYMBOL_TEXT_PARSE_FLOAT: &str = "__faber_rt_v1_text_parse_float";
pub const SYMBOL_TEXT_TRUTHY: &str = "__faber_rt_v1_text_truthy";
pub const SYMBOL_TEXT_I64: &str = "__faber_rt_v1_text_i64";
pub const SYMBOL_TEXT_F64: &str = "__faber_rt_v1_text_f64";
pub const SYMBOL_TEXT_I1: &str = "__faber_rt_v1_text_i1";
pub const SYMBOL_ASCII_TRUTHY: &str = "__faber_rt_v1_ascii_truthy";
pub const SYMBOL_SOLUM_READ_TEXT: &str = "__faber_rt_v1_solum_read_text";
pub const SYMBOL_SOLUM_WRITE_TEXT: &str = "__faber_rt_v1_solum_write_text";
pub const SYMBOL_VALOR_I64: &str = "__faber_rt_v1_valor_i64";
pub const SYMBOL_VALOR_F64: &str = "__faber_rt_v1_valor_f64";
pub const SYMBOL_VALOR_I1: &str = "__faber_rt_v1_valor_i1";
pub const SYMBOL_VALOR_TEXT: &str = "__faber_rt_v1_valor_text";
pub const SYMBOL_VALOR_ASCII: &str = "__faber_rt_v1_valor_ascii";
pub const SYMBOL_VALOR_NIHIL: &str = "__faber_rt_v1_valor_nihil";
pub const SYMBOL_VALOR_GET_I64: &str = "__faber_rt_v1_valor_get_i64";
pub const SYMBOL_VALOR_GET_F64: &str = "__faber_rt_v1_valor_get_f64";
pub const SYMBOL_VALOR_GET_I1: &str = "__faber_rt_v1_valor_get_i1";
pub const SYMBOL_VALOR_GET_TEXT: &str = "__faber_rt_v1_valor_get_text";
pub const SYMBOL_VALOR_GET_ASCII: &str = "__faber_rt_v1_valor_get_ascii";
pub const SYMBOL_VALOR_GET_NIHIL: &str = "__faber_rt_v1_valor_get_nihil";
pub const SYMBOL_OCTETI_NEW: &str = "__faber_rt_v1_octeti_new";
pub const SYMBOL_OCTETI_APPEND: &str = "__faber_rt_v1_octeti_append";
pub const SYMBOL_OCTETI_GET: &str = "__faber_rt_v1_octeti_get";
pub const SYMBOL_OCTETI_LENGTH: &str = "__faber_rt_v1_octeti_length";
pub const SYMBOL_OCTETI_FROM_TEXT: &str = "__faber_rt_v1_octeti_from_text";
pub const SYMBOL_OCTETI_FROM_ASCII: &str = "__faber_rt_v1_octeti_from_ascii";
pub const SYMBOL_OCTETI_GET_TEXT: &str = "__faber_rt_v1_octeti_get_text";
pub const SYMBOL_OCTETI_GET_ASCII: &str = "__faber_rt_v1_octeti_get_ascii";
pub const SYMBOL_INSTANS_FROM_TEXT: &str = "__faber_rt_v1_instans_from_text";
pub const SYMBOL_INSTANS_FROM_VALOR: &str = "__faber_rt_v1_instans_from_valor";
pub const SYMBOL_INSTANS_RETAG: &str = "__faber_rt_v1_instans_retag";
pub const SYMBOL_INSTANS_GET_TEXT: &str = "__faber_rt_v1_instans_get_text";
pub const SYMBOL_VALOR_OCTETI: &str = "__faber_rt_v1_valor_octeti";
pub const SYMBOL_VALOR_ARRAY: &str = "__faber_rt_v1_valor_array";
pub const SYMBOL_VALOR_MAP: &str = "__faber_rt_v1_valor_map";
pub const SYMBOL_VALOR_GET_OCTETI: &str = "__faber_rt_v1_valor_get_octeti";
pub const SYMBOL_VALOR_GET_ARRAY: &str = "__faber_rt_v1_valor_get_array";
pub const SYMBOL_VALOR_GET_MAP: &str = "__faber_rt_v1_valor_get_map";
pub const SYMBOL_VALOR_GENUS: &str = "__faber_rt_v1_valor_genus";
pub const SYMBOL_VALOR_GET_GENUS: &str = "__faber_rt_v1_valor_get_genus";
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
pub const SYMBOL_ARRAY_OPTION: &str = "__faber_rt_v1_array_option";
pub const SYMBOL_ARRAY_SORT: &str = "__faber_rt_v1_array_sort";
pub const SYMBOL_ARRAY_SUM: &str = "__faber_rt_v1_array_sum";
pub const SYMBOL_OPTION_NONE: &str = "__faber_rt_v1_option_none";
pub const SYMBOL_OPTION_SOME: &str = "__faber_rt_v1_option_some";
pub const SYMBOL_OPTION_IS_PRESENT: &str = "__faber_rt_v1_option_is_present";
pub const SYMBOL_OPTION_GET: &str = "__faber_rt_v1_option_get";
pub const SYMBOL_OPTION_GET_OR: &str = "__faber_rt_v1_option_get_or";
pub const SYMBOL_MAP_NEW: &str = "__faber_rt_v1_map_new";
pub const SYMBOL_MAP_PUT: &str = "__faber_rt_v1_map_put";
pub const SYMBOL_MAP_OPTION: &str = "__faber_rt_v1_map_option";
pub const SYMBOL_MAP_GET: &str = "__faber_rt_v1_map_get";
pub const SYMBOL_MAP_CONTAINS: &str = "__faber_rt_v1_map_contains";
pub const SYMBOL_MAP_DELETE: &str = "__faber_rt_v1_map_delete";
pub const SYMBOL_MAP_LENGTH: &str = "__faber_rt_v1_map_length";
pub const SYMBOL_MAP_IS_EMPTY: &str = "__faber_rt_v1_map_is_empty";
pub const SYMBOL_MAP_KEYS: &str = "__faber_rt_v1_map_keys";
pub const SYMBOL_MAP_VALUES: &str = "__faber_rt_v1_map_values";
pub const SYMBOL_VALOR_TENSOR: &str = "__faber_rt_v1_valor_tensor";
pub const SYMBOL_SET_NEW: &str = "__faber_rt_v1_set_new";
pub const SYMBOL_SET_ADD: &str = "__faber_rt_v1_set_add";
pub const SYMBOL_SET_CONTAINS: &str = "__faber_rt_v1_set_contains";
pub const SYMBOL_SET_DELETE: &str = "__faber_rt_v1_set_delete";
pub const SYMBOL_SET_LENGTH: &str = "__faber_rt_v1_set_length";
pub const SYMBOL_SET_IS_EMPTY: &str = "__faber_rt_v1_set_is_empty";
pub const SYMBOL_SET_FROM_ARRAY: &str = "__faber_rt_v1_set_from_array";
pub const SYMBOL_ARRAY_FROM_SET: &str = "__faber_rt_v1_array_from_set";
pub const SYMBOL_SET_UNION: &str = "__faber_rt_v1_set_union";
pub const SYMBOL_SET_INTERSECTION: &str = "__faber_rt_v1_set_intersection";
pub const SYMBOL_SET_DIFFERENCE: &str = "__faber_rt_v1_set_difference";
pub const SYMBOL_SET_SYMMETRIC_DIFFERENCE: &str = "__faber_rt_v1_set_symmetric_difference";
pub const SYMBOL_SET_IS_SUBSET: &str = "__faber_rt_v1_set_is_subset";
pub const SYMBOL_SET_IS_SUPERSET: &str = "__faber_rt_v1_set_is_superset";
pub const SYMBOL_TENSOR_NEW: &str = "__faber_rt_v1_tensor_new";
pub const SYMBOL_TENSOR_CREATE: &str = "__faber_rt_v1_tensor_create";
pub const SYMBOL_TENSOR_FROM_FLAT: &str = "__faber_rt_v1_tensor_from_flat";
pub const SYMBOL_TENSOR_RANK: &str = "__faber_rt_v1_tensor_rank";
pub const SYMBOL_TENSOR_SHAPE: &str = "__faber_rt_v1_tensor_shape";
pub const SYMBOL_TENSOR_RESHAPE: &str = "__faber_rt_v1_tensor_reshape";
pub const SYMBOL_TENSOR_GET: &str = "__faber_rt_v1_tensor_get";
pub const SYMBOL_TENSOR_SET: &str = "__faber_rt_v1_tensor_set";
pub const SYMBOL_TENSOR_FILL: &str = "__faber_rt_v1_tensor_fill";
pub const SYMBOL_TENSOR_FLATTEN: &str = "__faber_rt_v1_tensor_flatten";
pub const SYMBOL_TENSOR_MATERIALIZE: &str = "__faber_rt_v1_tensor_materialize";
pub const SYMBOL_TENSOR_SLICE: &str = "__faber_rt_v1_tensor_slice";
pub const SYMBOL_TENSOR_ADD: &str = "__faber_rt_v1_tensor_add";
pub const SYMBOL_TENSOR_SUB: &str = "__faber_rt_v1_tensor_sub";
pub const SYMBOL_TENSOR_MUL: &str = "__faber_rt_v1_tensor_mul";
pub const SYMBOL_TENSOR_MATMUL: &str = "__faber_rt_v1_tensor_matmul";
pub const SYMBOL_TENSOR_SUM: &str = "__faber_rt_v1_tensor_sum";
pub const SYMBOL_TENSOR_MEAN: &str = "__faber_rt_v1_tensor_mean";
pub const SYMBOL_TENSOR_CONVERT: &str = "__faber_rt_v1_tensor_convert";
pub const SYMBOL_SPARSE_NEW: &str = "__faber_rt_v1_sparse_new";
pub const SYMBOL_SPARSE_GET: &str = "__faber_rt_v1_sparse_get";
pub const SYMBOL_SPARSE_SET: &str = "__faber_rt_v1_sparse_set";
pub const SYMBOL_SPARSE_NONZERO: &str = "__faber_rt_v1_sparse_nonzero";
pub const SYMBOL_SPARSE_RANK: &str = "__faber_rt_v1_sparse_rank";
pub const SYMBOL_SPARSE_DENSIFY: &str = "__faber_rt_v1_sparse_densify";
pub const SYMBOL_SPARSE_FROM_TENSOR: &str = "__faber_rt_v1_sparse_from_tensor";
pub const SYMBOL_REGEX_FROM_TEXT: &str = "__faber_rt_v1_regex_from_text";
pub const SYMBOL_REGEX_FROM_ASCII: &str = "__faber_rt_v1_regex_from_ascii";
pub const SYMBOL_REGEX_GET_TEXT: &str = "__faber_rt_v1_regex_get_text";
pub const SYMBOL_INTERVAL_NEW: &str = "__faber_rt_v1_interval_new";
pub const SYMBOL_INTERVAL_INTERSECT: &str = "__faber_rt_v1_interval_intersect";
pub const SYMBOL_INTERVAL_UNION: &str = "__faber_rt_v1_interval_union";
pub const SYMBOL_INTERVAL_LENGTH: &str = "__faber_rt_v1_interval_length";
pub const SYMBOL_INTERVAL_CONTAINS: &str = "__faber_rt_v1_interval_contains";
pub const SYMBOL_INTERVAL_CLAMP_I64: &str = "__faber_rt_v1_interval_clamp_i64";
pub const SYMBOL_INTERVAL_CLAMP: &str = "__faber_rt_v1_interval_clamp";
pub const SYMBOL_INTERVAL_MATERIALIZE_ARRAY: &str = "__faber_rt_v1_interval_materialize_array";
pub const SYMBOL_INTERVAL_MATERIALIZE_TENSOR: &str = "__faber_rt_v1_interval_materialize_tensor";
pub const SYMBOL_PROGRAM_ENTRY: &str = "__faber_program_entry_v1";

pub type FaberRtValueKindV1 = u32;
pub const VALUE_KIND_I1: FaberRtValueKindV1 = 1;
pub const VALUE_KIND_I8: FaberRtValueKindV1 = 2;
pub const VALUE_KIND_I32: FaberRtValueKindV1 = 3;
pub const VALUE_KIND_I64: FaberRtValueKindV1 = 4;
pub const VALUE_KIND_F32: FaberRtValueKindV1 = 5;
pub const VALUE_KIND_F64: FaberRtValueKindV1 = 6;
pub const VALUE_KIND_PTR: FaberRtValueKindV1 = 7;
pub const VALUE_KIND_I16: FaberRtValueKindV1 = 8;
pub const VALUE_KIND_U8: FaberRtValueKindV1 = 9;
pub const VALUE_KIND_U16: FaberRtValueKindV1 = 10;
pub const VALUE_KIND_U32: FaberRtValueKindV1 = 11;
pub const VALUE_KIND_U64: FaberRtValueKindV1 = 12;
pub const VALUE_KIND_F16: FaberRtValueKindV1 = 13;
pub const VALUE_KIND_TEXT: FaberRtValueKindV1 = 14;
pub const VALUE_KIND_VALOR: FaberRtValueKindV1 = 15;
pub const VALUE_KIND_OPTION_I64: FaberRtValueKindV1 = 16;
pub const VALUE_KIND_INSTANS: FaberRtValueKindV1 = 17;
pub const VALUE_KIND_ASCII: FaberRtValueKindV1 = 18;

pub type FaberRtInstansPrecisionV1 = u32;
pub const INSTANS_PRECISION_SECONDS: FaberRtInstansPrecisionV1 = 0;
pub const INSTANS_PRECISION_MILLIS: FaberRtInstansPrecisionV1 = 1;
pub const INSTANS_PRECISION_MICROS: FaberRtInstansPrecisionV1 = 2;
pub const INSTANS_PRECISION_NANOS: FaberRtInstansPrecisionV1 = 3;

pub type FaberRtArrayRangeModeV1 = u32;
pub const ARRAY_RANGE_SLICE: FaberRtArrayRangeModeV1 = 1;
pub const ARRAY_RANGE_TAKE: FaberRtArrayRangeModeV1 = 2;
pub const ARRAY_RANGE_TAKE_LAST: FaberRtArrayRangeModeV1 = 3;
pub const ARRAY_RANGE_DROP_FIRST: FaberRtArrayRangeModeV1 = 4;

pub type FaberRtArrayOptionModeV1 = u32;
pub const ARRAY_OPTION_INDEX: FaberRtArrayOptionModeV1 = 1;
pub const ARRAY_OPTION_FIRST: FaberRtArrayOptionModeV1 = 2;
pub const ARRAY_OPTION_LAST: FaberRtArrayOptionModeV1 = 3;
pub const ARRAY_OPTION_REMOVE_FIRST: FaberRtArrayOptionModeV1 = 4;
pub const ARRAY_OPTION_REMOVE_LAST: FaberRtArrayOptionModeV1 = 5;

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
    ("nota", "text", "__faber_rt_v1_diagnostic_nota_text"),
    ("nota", "ascii", "__faber_rt_v1_diagnostic_nota_ascii"),
    ("nota", "i64", "__faber_rt_v1_diagnostic_nota_i64"),
    ("nota", "i1", "__faber_rt_v1_diagnostic_nota_i1"),
    ("nota", "float", "__faber_rt_v1_diagnostic_nota_f32"),
    ("nota", "double", "__faber_rt_v1_diagnostic_nota_f64"),
    ("nota", "i8", "__faber_rt_v1_diagnostic_nota_i8"),
    ("nota", "i32", "__faber_rt_v1_diagnostic_nota_i32"),
    ("mone", "ptr", "__faber_rt_v1_diagnostic_mone_ptr"),
    ("mone", "text", "__faber_rt_v1_diagnostic_mone_text"),
    ("mone", "ascii", "__faber_rt_v1_diagnostic_mone_ascii"),
    ("mone", "i64", "__faber_rt_v1_diagnostic_mone_i64"),
    ("vide", "ptr", "__faber_rt_v1_diagnostic_vide_ptr"),
    ("vide", "text", "__faber_rt_v1_diagnostic_vide_text"),
    ("vide", "ascii", "__faber_rt_v1_diagnostic_vide_ascii"),
    ("vide", "i64", "__faber_rt_v1_diagnostic_vide_i64"),
];

#[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
#[path = "host_abi_test.rs"]
mod tests;
