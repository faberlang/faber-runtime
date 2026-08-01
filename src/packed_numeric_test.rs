use crate::packed_numeric::{
    packed_u4_tensor_integration_rows, PackedBitOrder, PackedTensorIntegrationOperation,
    PackedTensorIntegrationStatus, PackedU4Block, PackedU4Layout, PackedWidenedType,
    ERR_PACKED_U4_BYTE_COUNT, ERR_PACKED_U4_INDEX, ERR_PACKED_U4_SCALE, ERR_PACKED_U4_ZERO_POINT,
};
use crate::Tensor;

#[test]
fn packed_u4_layout_records_element_width_bits() {
    assert_eq!(PackedU4Layout::toy_u4().element_width_bits, 4);
}

#[test]
fn packed_u4_layout_records_values_per_byte() {
    assert_eq!(PackedU4Layout::toy_u4().values_per_byte, 2);
}

#[test]
fn packed_u4_layout_records_block_values() {
    assert_eq!(PackedU4Layout::toy_u4().block_values, 8);
}

#[test]
fn packed_u4_layout_records_packed_bytes() {
    assert_eq!(PackedU4Layout::toy_u4().packed_bytes, 4);
}

#[test]
fn packed_u4_layout_records_bit_order() {
    assert_eq!(
        PackedU4Layout::toy_u4().bit_order,
        PackedBitOrder::LowNibbleFirst
    );
}

#[test]
fn packed_u4_layout_records_widened_type() {
    assert_eq!(
        PackedU4Layout::toy_u4().widened_type,
        PackedWidenedType::F32
    );
}

#[test]
fn packed_u4_extracts_low_nibble_first() {
    let block = PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], 1.0, 0).expect("valid toy block");

    let extracted: Vec<u8> = (0..8)
        .map(|index| block.extract(index).expect("in bounds"))
        .collect();

    assert_eq!(extracted, vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn packed_u4_dequantizes_with_scale_and_zero_point() {
    let block = PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], 0.5, 4).expect("valid toy block");

    assert_eq!(
        block.dequantize(),
        vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    );
}

#[test]
fn packed_u4_rejects_short_byte_slice() {
    assert_eq!(
        PackedU4Block::new(&[0x21, 0x43, 0x65], 1.0, 0).unwrap_err(),
        ERR_PACKED_U4_BYTE_COUNT
    );
}

#[test]
fn packed_u4_rejects_long_byte_slice() {
    assert_eq!(
        PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87, 0x99], 1.0, 0).unwrap_err(),
        ERR_PACKED_U4_BYTE_COUNT
    );
}

#[test]
fn packed_u4_rejects_empty_byte_slice() {
    assert_eq!(
        PackedU4Block::new(&[], 1.0, 0).unwrap_err(),
        ERR_PACKED_U4_BYTE_COUNT
    );
}

#[test]
fn packed_u4_rejects_nan_scale() {
    assert_eq!(
        PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], f32::NAN, 0).unwrap_err(),
        ERR_PACKED_U4_SCALE
    );
}

#[test]
fn packed_u4_rejects_overflow_zero_point() {
    assert_eq!(
        PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], 1.0, 16).unwrap_err(),
        ERR_PACKED_U4_ZERO_POINT
    );
}

#[test]
fn packed_u4_rejects_out_of_range_extract_index() {
    let block = PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], 1.0, 0).expect("valid toy block");
    assert_eq!(block.extract(8).unwrap_err(), ERR_PACKED_U4_INDEX);
}

#[test]
fn packed_u4_tensor_integration_records_reference_materialization_path() {
    let row = packed_u4_tensor_integration_rows()
        .iter()
        .find(|row| row.operation == PackedTensorIntegrationOperation::Rank1F32ElementwiseAdd)
        .expect("packed u4 tensor integration row");

    assert_eq!(row.layout, PackedU4Layout::toy_u4());
    assert_eq!(
        row.status,
        PackedTensorIntegrationStatus::ReferenceMaterialization
    );
    assert!(
        row.evidence
            .contains("packed_u4_materialized_tensor_feeds_elementwise_add"),
        "integration row should cite the focused tensor operation proof: {row:?}"
    );
}

#[test]
fn packed_u4_dequantizes_to_rank1_f32_tensor() {
    let block = PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], 0.5, 4).expect("valid toy block");
    let tensor = block
        .dequantize_tensor()
        .expect("toy packed block materializes to rank-1 tensor");

    assert_eq!(tensor.magnitudines(), vec![8]);
    assert_eq!(
        tensor.planata(),
        vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    );
}

#[test]
fn packed_u4_materialized_tensor_feeds_elementwise_add() {
    let block = PackedU4Block::new(&[0x21, 0x43, 0x65, 0x87], 0.5, 4).expect("valid toy block");
    let packed_tensor = block
        .dequantize_tensor()
        .expect("toy packed block materializes to rank-1 tensor");
    let bias = Tensor::structa(vec![1.0_f32; PackedU4Layout::BLOCK_VALUES], &[8])
        .expect("rank-1 bias tensor");

    let result = packed_tensor
        .addita(&bias)
        .expect("packed tensor and bias shapes match");

    assert_eq!(
        result.planata(),
        vec![-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    );
}
