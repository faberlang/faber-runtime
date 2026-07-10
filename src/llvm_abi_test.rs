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
    }
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
    ] {
        assert!(symbol.starts_with("__faber_rt_v1_"), "{symbol}");
    }
    assert_eq!(SYMBOL_PROGRAM_ENTRY, "__faber_program_entry_v1");
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
