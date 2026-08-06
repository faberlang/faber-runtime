use super::*;
use core::mem::{align_of, size_of};
use std::collections::BTreeSet;

/// Coherence: every `radix-host-abi` gradient symbol string matches the
/// corresponding `faber::host_abi` gradient symbol string.
///
/// The runtime (`faber-runtime`) mirrors the ABI contract table owned by
/// `radix-host-abi`. This test catches drift between the two crates at
/// test time in a dev build.
#[test]
fn gradient_symbols_cohere_with_radix_host_abi() {
    assert_eq!(
        radix_host_abi::SYMBOL_GRADIENT_CREATE,
        SYMBOL_GRADIENT_CREATE,
    );
    assert_eq!(
        radix_host_abi::SYMBOL_GRADIENT_ACCUMULATE,
        SYMBOL_GRADIENT_ACCUMULATE,
    );
    assert_eq!(radix_host_abi::SYMBOL_GRADIENT_READ, SYMBOL_GRADIENT_READ,);
    assert_eq!(radix_host_abi::SYMBOL_GRADIENT_ZERO, SYMBOL_GRADIENT_ZERO,);
}

#[test]
fn cursor_stream_symbol_coheres_with_radix_host_abi() {
    // P5: the shared cursor-stream row mirrors the radix-host-abi table.
    assert_eq!(radix_host_abi::SYMBOL_CURSOR_STREAM, SYMBOL_CURSOR_STREAM,);
    assert_eq!(SYMBOL_CURSOR_STREAM, "__faber_rt_v1_cursor_stream");
}

#[test]
fn or_recovery_symbols_cohere_with_radix_host_abi() {
    // P6: the `_or` recovery family mirrors the radix-host-abi table.
    for (runtime, shared) in [
        (
            SYMBOL_VALOR_GET_I64_OR,
            radix_host_abi::SYMBOL_VALOR_GET_I64_OR,
        ),
        (
            SYMBOL_VALOR_GET_F64_OR,
            radix_host_abi::SYMBOL_VALOR_GET_F64_OR,
        ),
        (
            SYMBOL_VALOR_GET_I1_OR,
            radix_host_abi::SYMBOL_VALOR_GET_I1_OR,
        ),
        (
            SYMBOL_VALOR_GET_TEXT_OR,
            radix_host_abi::SYMBOL_VALOR_GET_TEXT_OR,
        ),
        (
            SYMBOL_VALOR_GET_ASCII_OR,
            radix_host_abi::SYMBOL_VALOR_GET_ASCII_OR,
        ),
        (
            SYMBOL_VALOR_GET_OCTETI_OR,
            radix_host_abi::SYMBOL_VALOR_GET_OCTETI_OR,
        ),
        (
            SYMBOL_VALOR_GET_ARRAY_OR,
            radix_host_abi::SYMBOL_VALOR_GET_ARRAY_OR,
        ),
        (
            SYMBOL_VALOR_GET_MAP_OR,
            radix_host_abi::SYMBOL_VALOR_GET_MAP_OR,
        ),
        (
            SYMBOL_VALOR_GET_GENUS_OR,
            radix_host_abi::SYMBOL_VALOR_GET_GENUS_OR,
        ),
        (
            SYMBOL_OCTETI_GET_TEXT_OR,
            radix_host_abi::SYMBOL_OCTETI_GET_TEXT_OR,
        ),
        (
            SYMBOL_OCTETI_GET_ASCII_OR,
            radix_host_abi::SYMBOL_OCTETI_GET_ASCII_OR,
        ),
        (
            SYMBOL_INSTANS_FROM_TEXT_OR,
            radix_host_abi::SYMBOL_INSTANS_FROM_TEXT_OR,
        ),
        (
            SYMBOL_INSTANS_FROM_VALOR_OR,
            radix_host_abi::SYMBOL_INSTANS_FROM_VALOR_OR,
        ),
    ] {
        assert_eq!(runtime, shared, "runtime/radix-host-abi `_or` row drift");
        assert!(runtime.starts_with("__faber_rt_v1_"), "{runtime}");
    }
}

#[test]
fn failable_rows_cohere_with_radix_host_abi() {
    // P10: the failable status/payload rows mirror the radix-host-abi table.
    assert_eq!(radix_host_abi::SYMBOL_FALLIBLE_ERROR, SYMBOL_FALLIBLE_ERROR,);
    assert_eq!(SYMBOL_FALLIBLE_ERROR, "__faber_rt_v1_fallible_error");
    let fallible_code = radix_host_abi::STATUS_CODES
        .iter()
        .find(|(name, _)| *name == "STATUS_FALLIBLE")
        .map(|(_, code)| *code)
        .expect("radix-host-abi STATUS_FALLIBLE row");
    assert_eq!(fallible_code, STATUS_FALLIBLE.code);
    assert_eq!(STATUS_FALLIBLE.code, 5);
    assert_eq!(STATUS_OK.code, 0, "happy-path discriminator stays 0");
}

#[test]
fn solum_symbols_cohere_with_radix_host_abi() {
    assert_eq!(
        radix_host_abi::SYMBOL_SOLUM_READ_TEXT,
        SYMBOL_SOLUM_READ_TEXT,
    );
    assert_eq!(
        radix_host_abi::SYMBOL_SOLUM_READ_LINES,
        SYMBOL_SOLUM_READ_LINES,
    );
    assert_eq!(
        radix_host_abi::SYMBOL_SOLUM_READ_BYTES,
        SYMBOL_SOLUM_READ_BYTES,
    );
    assert_eq!(
        radix_host_abi::SYMBOL_SOLUM_WRITE_TEXT,
        SYMBOL_SOLUM_WRITE_TEXT,
    );
}

#[test]
fn host_abi_v1_slice_carrier_layout() {
    assert_eq!(size_of::<FaberRtSliceV1>(), 16);
    assert_eq!(align_of::<FaberRtSliceV1>(), 8);
}

#[test]
fn host_abi_v1_status_carrier_layout() {
    assert_eq!(size_of::<FaberRtStatusV1>(), 4);
    assert_eq!(align_of::<FaberRtStatusV1>(), 4);
}

#[test]
fn host_abi_v1_ptr_result_carrier_layout() {
    assert_eq!(size_of::<FaberRtPtrResultV1>(), size_of::<usize>() * 2);
    assert_eq!(align_of::<FaberRtPtrResultV1>(), align_of::<*mut c_void>());
}

#[test]
fn host_abi_v1_exit_carrier_layout() {
    assert_eq!(size_of::<FaberRtExitV1>(), 8);
    assert_eq!(align_of::<FaberRtExitV1>(), 4);
}

#[test]
fn host_abi_v1_context_carrier_layout() {
    assert_eq!(size_of::<FaberRtContextV1>(), 0);
    assert_eq!(align_of::<FaberRtContextV1>(), align_of::<*mut c_void>());
}

#[test]
fn host_abi_v1_diagnostic_symbol_count_is_23() {
    assert_eq!(DIAGNOSTIC_SYMBOLS_V1.len(), 23);
}

#[test]
fn host_abi_v1_diagnostic_symbols_are_unique() {
    let symbols = DIAGNOSTIC_SYMBOLS_V1
        .iter()
        .map(|(_, _, symbol)| *symbol)
        .collect::<BTreeSet<_>>();
    assert_eq!(symbols.len(), DIAGNOSTIC_SYMBOLS_V1.len());
}

#[test]
fn host_abi_v1_diagnostic_symbols_have_v1_prefix() {
    for (_, _, symbol) in DIAGNOSTIC_SYMBOLS_V1 {
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
}

#[test]
fn host_abi_v1_diagnostic_symbol_is_recoverable_from_kind_and_carrier() {
    for (kind, carrier, symbol) in DIAGNOSTIC_SYMBOLS_V1 {
        assert_eq!(diagnostic_symbol_v1(kind, carrier), Some(*symbol));
    }
}

#[test]
fn host_abi_v1_diagnostic_symbol_nota_float() {
    assert_eq!(
        diagnostic_symbol_v1("nota", "float"),
        Some("__faber_rt_v1_diagnostic_nota_f32")
    );
}

#[test]
fn host_abi_v1_diagnostic_symbol_nota_double() {
    assert_eq!(
        diagnostic_symbol_v1("nota", "double"),
        Some("__faber_rt_v1_diagnostic_nota_f64")
    );
}

#[test]
fn host_abi_v1_diagnostic_unsupported_differs_from_ok() {
    assert_ne!(STATUS_UNSUPPORTED, STATUS_OK);
}

#[test]
fn host_abi_v1_core_symbols_have_v1_prefix() {
    for symbol in [
        SYMBOL_INIT,
        SYMBOL_SHUTDOWN,
        SYMBOL_WRITE_NOTA_TEXT,
        SYMBOL_ASSERT,
        SYMBOL_ASSERT_MESSAGE,
        SYMBOL_FATAL,
        SYMBOL_FATAL_OPAQUE,
        SYMBOL_FORMAT_I1,
        SYMBOL_FORMAT_I64,
        SYMBOL_FORMAT_I64_I64,
        SYMBOL_FORMAT_I64_I64_I64,
        SYMBOL_FORMAT_F64,
        SYMBOL_FORMAT_F32,
        SYMBOL_TEXT_I64,
        SYMBOL_TEXT_F64,
        SYMBOL_TEXT_I1,
        SYMBOL_ASCII_TRUTHY,
        SYMBOL_TEXT_CONCAT,
        SYMBOL_VALOR_I64,
        SYMBOL_VALOR_F64,
        SYMBOL_VALOR_I1,
    ] {
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
}

#[test]
fn host_abi_v1_array_symbols_have_v1_prefix() {
    for symbol in [
        SYMBOL_ARRAY_NEW,
        SYMBOL_ARRAY_PUSH,
        SYMBOL_ARRAY_EXTEND,
        SYMBOL_ARRAY_LENGTH,
        SYMBOL_ARRAY_GET,
        SYMBOL_ARRAY_SET,
        SYMBOL_ARRAY_CLONE,
        SYMBOL_ARRAY_CONTAINS,
        SYMBOL_ARRAY_IS_EMPTY,
        SYMBOL_ARRAY_REVERSE,
        SYMBOL_ARRAY_RANGE,
        SYMBOL_ARRAY_OPTION,
        SYMBOL_ARRAY_SORT,
        SYMBOL_ARRAY_SUM,
    ] {
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
}

#[test]
fn host_abi_v1_option_symbols_have_v1_prefix() {
    for symbol in [
        SYMBOL_OPTION_NONE,
        SYMBOL_OPTION_SOME,
        SYMBOL_OPTION_IS_PRESENT,
        SYMBOL_OPTION_GET,
        SYMBOL_OPTION_GET_OR,
    ] {
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
}

#[test]
fn host_abi_v1_program_entry_exact_value() {
    assert_eq!(SYMBOL_PROGRAM_ENTRY, "__faber_program_entry_v1");
}

#[test]
fn host_abi_v1_array_option_symbols_are_distinct() {
    assert_eq!(
        [
            ARRAY_OPTION_INDEX,
            ARRAY_OPTION_FIRST,
            ARRAY_OPTION_LAST,
            ARRAY_OPTION_REMOVE_FIRST,
            ARRAY_OPTION_REMOVE_LAST,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len(),
        5
    );
}

#[test]
fn host_abi_v1_value_kind_symbols_are_distinct() {
    assert_eq!(
        [
            VALUE_KIND_I1,
            VALUE_KIND_I8,
            VALUE_KIND_I32,
            VALUE_KIND_I64,
            VALUE_KIND_F32,
            VALUE_KIND_F64,
            VALUE_KIND_PTR,
            VALUE_KIND_I16,
            VALUE_KIND_U8,
            VALUE_KIND_U16,
            VALUE_KIND_U32,
            VALUE_KIND_U64,
            VALUE_KIND_F16,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len(),
        13
    );
}

#[test]
fn host_abi_v1_array_range_symbols_are_distinct() {
    assert_eq!(
        [
            ARRAY_RANGE_SLICE,
            ARRAY_RANGE_TAKE,
            ARRAY_RANGE_TAKE_LAST,
            ARRAY_RANGE_DROP_FIRST,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len(),
        4
    );
}

#[test]
fn host_abi_v1_llvm_slice_type_definition() {
    assert_eq!(
        LLVM_SLICE_TYPE_DEFINITION,
        "%FaberRtSliceV1 = type { ptr, i64 }"
    );
}

#[test]
fn host_abi_v1_llvm_exit_type_definition() {
    assert_eq!(LLVM_EXIT_TYPE_DEFINITION, "%FaberRtExitV1 = type i64");
}

#[test]
fn host_abi_v1_llvm_ptr_result_type_definition() {
    assert_eq!(
        LLVM_PTR_RESULT_TYPE_DEFINITION,
        "%FaberRtPtrResultV1 = type { i32, ptr }"
    );
}

#[test]
fn host_abi_v1_gradient_symbols_have_v1_gradient_prefix() {
    for symbol in [
        SYMBOL_GRADIENT_CREATE,
        SYMBOL_GRADIENT_ACCUMULATE,
        SYMBOL_GRADIENT_READ,
        SYMBOL_GRADIENT_ZERO,
    ] {
        assert!(symbol.starts_with("__faber_rt_v1_gradient_"), "{symbol}");
    }
}

#[test]
fn host_abi_v1_gradient_symbols_are_distinct() {
    let expected = [
        "SYMBOL_GRADIENT_CREATE",
        "SYMBOL_GRADIENT_ACCUMULATE",
        "SYMBOL_GRADIENT_READ",
        "SYMBOL_GRADIENT_ZERO",
    ];
    let mut seen: Vec<&str> = expected
        .iter()
        .map(|name| match *name {
            "SYMBOL_GRADIENT_CREATE" => SYMBOL_GRADIENT_CREATE,
            "SYMBOL_GRADIENT_ACCUMULATE" => SYMBOL_GRADIENT_ACCUMULATE,
            "SYMBOL_GRADIENT_READ" => SYMBOL_GRADIENT_READ,
            "SYMBOL_GRADIENT_ZERO" => SYMBOL_GRADIENT_ZERO,
            _ => unreachable!(),
        })
        .collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 4, "gradient symbols must all be distinct");
}

#[test]
fn host_abi_v1_does_not_expose_tensor_permute_or_transpose_symbols() {
    for symbol in [
        SYMBOL_TENSOR_NEW,
        SYMBOL_TENSOR_CREATE,
        SYMBOL_TENSOR_FROM_FLAT,
        SYMBOL_TENSOR_RANK,
        SYMBOL_TENSOR_SHAPE,
        SYMBOL_TENSOR_RESHAPE,
        SYMBOL_TENSOR_GET,
        SYMBOL_TENSOR_SET,
        SYMBOL_TENSOR_FILL,
        SYMBOL_TENSOR_FLATTEN,
        SYMBOL_TENSOR_MATERIALIZE,
        SYMBOL_TENSOR_SLICE,
        SYMBOL_TENSOR_ADD,
        SYMBOL_TENSOR_SUB,
        SYMBOL_TENSOR_MUL,
        SYMBOL_TENSOR_MATMUL,
        SYMBOL_TENSOR_SUM,
        SYMBOL_TENSOR_MEAN,
        SYMBOL_TENSOR_CONVERT,
    ] {
        assert!(!symbol.contains("permute"), "{symbol}");
        assert!(!symbol.contains("transpose"), "{symbol}");
    }
}
