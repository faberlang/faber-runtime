use super::*;
use core::mem::{align_of, size_of};
use std::collections::BTreeSet;

#[test]
fn llvm_abi_v1_carriers_have_stable_host_layout() {
    assert_eq!(size_of::<FaberRtSliceV1>(), 16);
    assert_eq!(align_of::<FaberRtSliceV1>(), 8);
    assert_eq!(size_of::<FaberRtStatusV1>(), 4);
    assert_eq!(align_of::<FaberRtStatusV1>(), 4);
    assert_eq!(size_of::<FaberRtPtrResultV1>(), size_of::<usize>() * 2);
    assert_eq!(align_of::<FaberRtPtrResultV1>(), align_of::<*mut c_void>());
    assert_eq!(size_of::<FaberRtExitV1>(), 8);
    assert_eq!(align_of::<FaberRtExitV1>(), 4);
    assert_eq!(size_of::<FaberRtContextV1>(), 0);
    assert_eq!(align_of::<FaberRtContextV1>(), align_of::<*mut c_void>());
}

#[test]
fn llvm_abi_v1_diagnostic_family_is_complete_and_unique() {
    assert_eq!(DIAGNOSTIC_SYMBOLS_V1.len(), 11);
    let symbols = DIAGNOSTIC_SYMBOLS_V1
        .iter()
        .map(|(_, _, symbol)| *symbol)
        .collect::<BTreeSet<_>>();
    assert_eq!(symbols.len(), DIAGNOSTIC_SYMBOLS_V1.len());
    for (kind, carrier, symbol) in DIAGNOSTIC_SYMBOLS_V1 {
        assert_eq!(diagnostic_symbol_v1(kind, carrier), Some(*symbol));
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
    assert_eq!(
        diagnostic_symbol_v1("nota", "float"),
        Some("__faber_rt_v1_diagnostic_nota_f32")
    );
    assert_eq!(
        diagnostic_symbol_v1("nota", "double"),
        Some("__faber_rt_v1_diagnostic_nota_f64")
    );
    assert_ne!(STATUS_UNSUPPORTED, STATUS_OK);
}

#[test]
fn llvm_abi_v1_symbol_namespace_is_versioned() {
    for symbol in [
        SYMBOL_INIT,
        SYMBOL_SHUTDOWN,
        SYMBOL_WRITE_NOTA_TEXT,
        SYMBOL_ASSERT,
        SYMBOL_ASSERT_MESSAGE,
        SYMBOL_FATAL,
        SYMBOL_FATAL_OPAQUE,
        SYMBOL_FORMAT_I64,
        SYMBOL_FORMAT_I64_I64,
        SYMBOL_FORMAT_I64_I64_I64,
        SYMBOL_FORMAT_F64,
        SYMBOL_TEXT_I64,
        SYMBOL_TEXT_F64,
        SYMBOL_TEXT_I1,
        SYMBOL_VALOR_I64,
        SYMBOL_VALOR_F64,
        SYMBOL_VALOR_I1,
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
    ] {
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
    assert_eq!(SYMBOL_PROGRAM_ENTRY, "__faber_program_entry_v1");
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
    assert_eq!(
        [
            VALUE_KIND_I1,
            VALUE_KIND_I8,
            VALUE_KIND_I32,
            VALUE_KIND_I64,
            VALUE_KIND_F32,
            VALUE_KIND_F64,
            VALUE_KIND_PTR,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len(),
        7
    );
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
    assert_eq!(
        LLVM_SLICE_TYPE_DEFINITION,
        "%FaberRtSliceV1 = type { ptr, i64 }"
    );
    assert_eq!(
        LLVM_EXIT_TYPE_DEFINITION,
        "%FaberRtExitV1 = type { i32, i32 }"
    );
    assert_eq!(
        LLVM_PTR_RESULT_TYPE_DEFINITION,
        "%FaberRtPtrResultV1 = type { i32, ptr }"
    );
}
