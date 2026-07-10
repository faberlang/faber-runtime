use super::*;
use core::mem::{align_of, size_of};

#[test]
fn llvm_abi_v1_carriers_have_stable_host_layout() {
    assert_eq!(size_of::<FaberRtSliceV1>(), 16);
    assert_eq!(align_of::<FaberRtSliceV1>(), 8);
    assert_eq!(size_of::<FaberRtStatusV1>(), 4);
    assert_eq!(align_of::<FaberRtStatusV1>(), 4);
    assert_eq!(size_of::<FaberRtExitV1>(), 8);
    assert_eq!(align_of::<FaberRtExitV1>(), 4);
    assert_eq!(size_of::<FaberRtContextV1>(), 0);
    assert_eq!(align_of::<FaberRtContextV1>(), align_of::<*mut c_void>());
}

#[test]
fn llvm_abi_v1_symbol_namespace_is_versioned() {
    for symbol in [
        SYMBOL_INIT,
        SYMBOL_SHUTDOWN,
        SYMBOL_WRITE_NOTA_TEXT,
        SYMBOL_FATAL,
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
}
