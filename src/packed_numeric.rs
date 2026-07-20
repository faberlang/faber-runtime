//! Packed numeric block runtime carriers.
//!
//! TARGET: reference runtime support for quant-shaped systems values. Packed
//! blocks keep byte storage behind explicit layout facts and materialize only
//! through named widen/dequant operations.

use crate::tensor::Tensor;

pub const ERR_PACKED_U4_BYTE_COUNT: &str = "packed u4 block byte count mismatch";
pub const ERR_PACKED_U4_INDEX: &str = "packed u4 block index out of bounds";
pub const ERR_PACKED_U4_SCALE: &str = "packed u4 block scale must be finite";
pub const ERR_PACKED_U4_ZERO_POINT: &str = "packed u4 block zero point out of range";
pub const ERR_PACKED_U4_TENSOR_SHAPE: &str = "packed u4 tensor materialization shape mismatch";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedBitOrder {
    LowNibbleFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedWidenedType {
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedU4Layout {
    pub element_width_bits: u8,
    pub values_per_byte: u8,
    pub block_values: usize,
    pub packed_bytes: usize,
    pub bit_order: PackedBitOrder,
    pub widened_type: PackedWidenedType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedTensorIntegrationOperation {
    Rank1F32ElementwiseAdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedTensorIntegrationStatus {
    ReferenceMaterialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedTensorIntegrationRow {
    pub layout: PackedU4Layout,
    pub operation: PackedTensorIntegrationOperation,
    pub status: PackedTensorIntegrationStatus,
    pub evidence: &'static str,
}

pub const PACKED_U4_TENSOR_INTEGRATION_ROWS: &[PackedTensorIntegrationRow] = &[PackedTensorIntegrationRow {
    layout: PackedU4Layout::toy_u4(),
    operation: PackedTensorIntegrationOperation::Rank1F32ElementwiseAdd,
    status: PackedTensorIntegrationStatus::ReferenceMaterialization,
    evidence: "crates/faber/src/packed_numeric_test.rs::packed_u4_materialized_tensor_feeds_elementwise_add",
}];

impl PackedU4Layout {
    pub const ELEMENT_WIDTH_BITS: u8 = 4;
    pub const VALUES_PER_BYTE: u8 = 2;
    pub const BLOCK_VALUES: usize = 8;
    pub const PACKED_BYTES: usize = Self::BLOCK_VALUES / Self::VALUES_PER_BYTE as usize;
    pub const BIT_ORDER: PackedBitOrder = PackedBitOrder::LowNibbleFirst;
    pub const WIDENED_TYPE: PackedWidenedType = PackedWidenedType::F32;

    #[must_use]
    pub const fn toy_u4() -> Self {
        Self {
            element_width_bits: Self::ELEMENT_WIDTH_BITS,
            values_per_byte: Self::VALUES_PER_BYTE,
            block_values: Self::BLOCK_VALUES,
            packed_bytes: Self::PACKED_BYTES,
            bit_order: Self::BIT_ORDER,
            widened_type: Self::WIDENED_TYPE,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedU4Block {
    bytes: [u8; PackedU4Layout::PACKED_BYTES],
    scale: f32,
    zero_point: u8,
}

impl PackedU4Block {
    pub fn new(bytes: &[u8], scale: f32, zero_point: u8) -> Result<Self, &'static str> {
        if bytes.len() != PackedU4Layout::PACKED_BYTES {
            return Err(ERR_PACKED_U4_BYTE_COUNT);
        }
        if !scale.is_finite() {
            return Err(ERR_PACKED_U4_SCALE);
        }
        if zero_point > 0x0f {
            return Err(ERR_PACKED_U4_ZERO_POINT);
        }

        let mut stored = [0_u8; PackedU4Layout::PACKED_BYTES];
        stored.copy_from_slice(bytes);
        Ok(Self {
            bytes: stored,
            scale,
            zero_point,
        })
    }

    #[must_use]
    pub const fn layout(&self) -> PackedU4Layout {
        PackedU4Layout::toy_u4()
    }

    #[must_use]
    pub fn packed_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    #[must_use]
    pub fn zero_point(&self) -> u8 {
        self.zero_point
    }

    pub fn extract(&self, index: usize) -> Result<u8, &'static str> {
        if index >= PackedU4Layout::BLOCK_VALUES {
            return Err(ERR_PACKED_U4_INDEX);
        }
        Ok(self.extract_in_bounds(index))
    }

    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        (0..PackedU4Layout::BLOCK_VALUES)
            .map(|index| {
                let raw = self.extract_in_bounds(index);
                (f32::from(raw) - f32::from(self.zero_point)) * self.scale
            })
            .collect()
    }

    pub fn dequantize_tensor(&self) -> Result<Tensor<f32>, &'static str> {
        Tensor::structa(
            self.dequantize(),
            &[i64::try_from(PackedU4Layout::BLOCK_VALUES)
                .map_err(|_| ERR_PACKED_U4_TENSOR_SHAPE)?],
        )
    }

    fn extract_in_bounds(&self, index: usize) -> u8 {
        let byte = self.bytes[index / PackedU4Layout::VALUES_PER_BYTE as usize];
        if index.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        }
    }
}

#[must_use]
pub fn packed_u4_tensor_integration_rows() -> &'static [PackedTensorIntegrationRow] {
    PACKED_U4_TENSOR_INTEGRATION_ROWS
}
