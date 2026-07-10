//! Stable C ABI shared by LLVM-emitted programs and the LLVM host runtime.

use core::ffi::c_void;

pub const ABI_VERSION: u32 = 1;

pub const SYMBOL_INIT: &str = "__faber_rt_v1_init";
pub const SYMBOL_SHUTDOWN: &str = "__faber_rt_v1_shutdown";
pub const SYMBOL_WRITE_NOTA_TEXT: &str = "__faber_rt_v1_write_nota_text";
pub const SYMBOL_FATAL: &str = "__faber_rt_v1_fatal";
pub const SYMBOL_PROGRAM_ENTRY: &str = "__faber_program_entry_v1";

pub const LLVM_SLICE_TYPE: &str = "%FaberRtSliceV1";
pub const LLVM_SLICE_TYPE_DEFINITION: &str = "%FaberRtSliceV1 = type { ptr, i64 }";
pub const LLVM_EXIT_TYPE: &str = "%FaberRtExitV1";
pub const LLVM_EXIT_TYPE_DEFINITION: &str = "%FaberRtExitV1 = type { i32, i32 }";

pub const STATUS_OK: FaberRtStatusV1 = FaberRtStatusV1 { code: 0 };
pub const STATUS_INVALID_ARGUMENT: FaberRtStatusV1 = FaberRtStatusV1 { code: 1 };
pub const STATUS_IO_ERROR: FaberRtStatusV1 = FaberRtStatusV1 { code: 2 };
pub const STATUS_PANIC: FaberRtStatusV1 = FaberRtStatusV1 { code: 3 };

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
