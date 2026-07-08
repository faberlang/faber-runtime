//! Faber language runtime types for generated Rust code.
//!
//! WHY: language-owned carriers (`Valor`, `Ascii`, `Regex`, frame types) live here
//! so codegen emits `faber::*` instead of inlined `Faber*` prelude types.

pub mod ascii;
pub mod display;
pub mod frame;
pub mod instans;
pub mod intervallum;
pub mod packed_numeric;
pub mod regex;
pub mod sparsa;
pub mod tensor;
pub mod textus;
pub mod valor;

pub use ascii::Ascii;
pub use display::{
    display_bivalens, display_fractus, display_option, display_option_bivalens,
    display_option_fractus, display_option_vacuum, display_text_payload, display_valor,
    FractusDisplay,
};
pub use frame::{FrameStatus, IntoFrameStatus, IntoScrinium, Meus, Scrinium, Sermo, Tuus};
pub use instans::{Instans, InstansPraecisio};
pub use intervallum::{Intervallum, IntervallumKind};
pub use packed_numeric::{
    packed_u4_tensor_integration_rows, PackedBitOrder, PackedTensorIntegrationOperation,
    PackedTensorIntegrationRow, PackedTensorIntegrationStatus, PackedU4Block, PackedU4Layout,
    PackedWidenedType,
};
pub use regex::Regex;
pub use sparsa::Sparsa;
pub use tensor::Tensor;
pub use textus::unicode_scalar_value;
pub use valor::{FromValor, Valor};

#[cfg(test)]
#[path = "display_test.rs"]
mod display_test;

#[cfg(test)]
#[path = "textus_test.rs"]
mod textus_test;

#[cfg(test)]
#[path = "instans_test.rs"]
mod instans_test;

#[cfg(test)]
#[path = "intervallum_test.rs"]
mod intervallum_test;

#[cfg(test)]
#[path = "packed_numeric_test.rs"]
mod packed_numeric_test;

#[cfg(test)]
#[path = "valor_from_valor_test.rs"]
mod valor_from_valor_test;

#[cfg(test)]
#[path = "valor_aggregate_test.rs"]
mod valor_aggregate_test;

#[cfg(test)]
#[path = "frame_test.rs"]
mod frame_test;

#[cfg(test)]
#[path = "frame_live_test.rs"]
mod frame_live_test;
