use super::{
    tensor_flat_offset, tensor_shape_element_count, tensor_shape_has_element_count, Tensor,
    ERR_BROADCAST_SHAPE, ERR_DIVIDE_NON_FINITE_INPUT, ERR_DIVIDE_NON_FINITE_RESULT,
    ERR_DIVIDE_ZERO_DENOMINATOR, ERR_ELEMENT_COUNT_OVERFLOW, ERR_MATMUL_ARGUMENT_RANK,
    ERR_MATMUL_INNER_DIMENSION, ERR_MATMUL_RECEIVER_RANK, ERR_MEDIA_EMPTY,
    ERR_PERMUTE_AXIS_OUT_OF_RANGE, ERR_PERMUTE_DUPLICATE_AXIS, ERR_PERMUTE_NEGATIVE_AXIS,
    ERR_PERMUTE_RANK, ERR_TRANSPOSE_RANK,
};

#[test]
fn vacua_has_rank_zero() {
    let tensor: Tensor<f32> = Tensor::vacua();
    assert_eq!(tensor.longitudo(), 0);
    assert_eq!(tensor.element_count(), 1);
}

#[test]
fn crea_rejects_negative_shape_dimension() {
    let err = Tensor::<f32>::crea(&[-1, 0], 0.0).unwrap_err();
    assert_eq!(err, "tensor shape dimension must be non-negative");
}

#[test]
fn crea_rejects_overflowing_shape_product() {
    let err = Tensor::<f32>::crea(&[i64::MAX, 2], 0.0).unwrap_err();
    assert_eq!(err, ERR_ELEMENT_COUNT_OVERFLOW);
}

#[test]
fn tensor_shape_element_count_rejects_negative_and_overflow() {
    assert_eq!(tensor_shape_element_count(&[2, 3, 4]), Some(24));
    assert_eq!(tensor_shape_element_count(&[-1, 4]), None);
    assert_eq!(tensor_shape_element_count(&[i64::MAX, i64::MAX]), None);
    assert!(tensor_shape_has_element_count(&[2, 3], 6));
    assert!(!tensor_shape_has_element_count(&[2, 3], 5));
    assert_eq!(ERR_ELEMENT_COUNT_OVERFLOW, "tensor element count overflow");
}

#[test]
fn tensor_flat_offset_checks_rank_bounds_and_overflow() {
    assert_eq!(tensor_flat_offset(&[2, 3], &[1, 2]), Some(5));
    assert_eq!(tensor_flat_offset(&[2, 3], &[2, 0]), None);
    assert_eq!(tensor_flat_offset(&[2, 3], &[0]), None);
    assert_eq!(tensor_flat_offset(&[2, 3], &[-1, 0]), None);
}

#[test]
fn ponde_reports_out_of_bounds_and_negative_index() {
    let mut tensor = Tensor::crea(&[2, 2], 0.0f32).expect("valid shape");
    assert!(tensor.ponde(&[0, 0], 1.0).is_ok());
    assert_eq!(
        tensor.ponde(&[9, 9], 9.0),
        Err("tensor index out of bounds")
    );
    assert_eq!(
        tensor.ponde(&[-1, 0], 9.0),
        Err("tensor index must be non-negative")
    );
    assert_eq!(tensor.accipe(&[0, 0]).expect("valid index"), Some(1.0));
}

#[test]
fn accipe_rejects_negative_index() {
    let tensor = Tensor::crea(&[2, 2], 0.0f32).expect("valid shape");
    assert_eq!(
        tensor.accipe(&[-1, 0]),
        Err("tensor index must be non-negative")
    );
    assert_eq!(tensor.accipe(&[9, 9]).expect("in-range type"), None);
}

#[test]
fn structa_and_planata_round_trip() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("shape matches data");
    assert_eq!(tensor.magnitudines(), vec![2, 2]);
    assert_eq!(tensor.planata(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn structa_rejects_negative_shape_dimension() {
    let err = Tensor::structa(vec![1.0f32], &[-1]).unwrap_err();
    assert_eq!(err, "tensor shape dimension must be non-negative");
}

#[test]
fn convert_elements_preserves_shape_and_maps_values() {
    let tensor = Tensor::structa(vec![1i64, 2, 3, 4], &[2, 2]).expect("shape matches data");
    let converted = tensor.convert_elements(|value| value as f64);
    assert_eq!(converted.magnitudines(), vec![2, 2]);
    assert_eq!(converted.planata(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn sectio_slices_axis_zero() {
    let tensor = Tensor::crea(&[3, 2], 1.0f32).expect("valid shape");
    let slice = tensor.sectio(1, 3).expect("valid slice");
    assert_eq!(slice.longitudo(), 2);
    assert_eq!(slice.magnitudines(), vec![2, 2]);
}

#[test]
fn sectio_returns_axis_zero_view() {
    let mut tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .expect("shape matches data");
    let mut slice = tensor.sectio(1, 3).expect("valid slice");

    tensor.ponde(&[1, 0], 30.0).expect("parent write succeeds");
    assert_eq!(
        slice.accipe(&[0, 0]).expect("slice read succeeds"),
        Some(30.0)
    );

    slice.ponde(&[1, 1], 60.0).expect("slice write succeeds");
    assert_eq!(
        tensor.accipe(&[2, 1]).expect("parent read succeeds"),
        Some(60.0)
    );
}

#[test]
fn materialize_breaks_sectio_alias() {
    let mut tensor =
        Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("shape matches data");
    let mut materialized = tensor.sectio(0, 1).expect("valid slice").materialize();

    tensor.ponde(&[0, 0], 10.0).expect("parent write succeeds");
    assert_eq!(
        materialized
            .accipe(&[0, 0])
            .expect("materialized read succeeds"),
        Some(1.0)
    );

    materialized
        .ponde(&[0, 1], 20.0)
        .expect("materialized write succeeds");
    assert_eq!(
        tensor.accipe(&[0, 1]).expect("parent read succeeds"),
        Some(2.0)
    );
}

#[test]
fn tensor_is_send_sync_when_elements_are() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tensor<f32>>();
}

#[test]
fn sectio_rejects_negative_bounds() {
    let tensor = Tensor::crea(&[3, 2], 1.0f32).expect("valid shape");
    assert_eq!(
        tensor.sectio(-1, 2).unwrap_err(),
        "tensor slice bounds must be non-negative"
    );
    assert_eq!(
        tensor.sectio(2, 1).unwrap_err(),
        "tensor slice end must be at least start"
    );
}

#[test]
fn addita_sums_elementwise() {
    let a = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::structa(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();
    let c = a.addita(&b).expect("broadcast-compatible shape");
    assert_eq!(c.magnitudines(), vec![3]);
    assert_eq!(c.planata(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn addita_broadcasts_size_one_dimension() {
    let a = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::structa(vec![10.0f32, 20.0], &[2, 1]).unwrap();
    let c = a.addita(&b).expect("broadcast-compatible shape");
    assert_eq!(c.magnitudines(), vec![2, 2]);
    // a = [[1,2],[3,4]]; b = [[10],[20]] broadcasts to [[10,10],[20,20]].
    assert_eq!(c.planata(), vec![11.0, 12.0, 23.0, 24.0]);
}

#[test]
fn addita_rejects_broadcast_shape_mismatch() {
    let a = Tensor::structa(vec![1.0f32, 2.0], &[2]).unwrap();
    let b = Tensor::structa(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();
    assert_eq!(a.addita(&b).unwrap_err(), ERR_BROADCAST_SHAPE);
}

#[test]
fn addita_broadcasts_zero_extent_with_size_one_axis_to_empty_result() {
    let empty = Tensor::<f32>::structa(Vec::new(), &[0, 3]).unwrap();
    let row = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[1, 3]).unwrap();

    let result = empty
        .addita(&row)
        .expect("zero/one broadcast is compatible");

    assert_eq!(result.magnitudines(), vec![0, 3]);
    assert_eq!(result.planata(), Vec::<f32>::new());
}

#[test]
fn subtrahe_broadcasts_zero_extent_with_size_one_axis_to_empty_result() {
    let row = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
    let empty = Tensor::<f32>::structa(Vec::new(), &[0, 3]).unwrap();

    let result = row
        .subtrahe(&empty)
        .expect("one/zero broadcast is compatible");

    assert_eq!(result.magnitudines(), vec![0, 3]);
    assert_eq!(result.planata(), Vec::<f32>::new());
}

#[test]
fn multiplica_broadcasts_zero_extent_with_size_one_axis_to_empty_result() {
    let empty = Tensor::<f32>::structa(Vec::new(), &[2, 0, 3]).unwrap();
    let lane = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[1, 1, 3]).unwrap();

    let result = empty
        .multiplica(&lane)
        .expect("zero/one broadcast is compatible");

    assert_eq!(result.magnitudines(), vec![2, 0, 3]);
    assert_eq!(result.planata(), Vec::<f32>::new());
}

#[test]
fn zero_extent_broadcast_rejects_non_one_mismatch_for_each_arithmetic_op() {
    let empty = Tensor::<f32>::structa(Vec::new(), &[0, 3]).unwrap();
    let rows = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    assert_eq!(empty.addita(&rows).unwrap_err(), ERR_BROADCAST_SHAPE);
    assert_eq!(empty.subtrahe(&rows).unwrap_err(), ERR_BROADCAST_SHAPE);
    assert_eq!(empty.multiplica(&rows).unwrap_err(), ERR_BROADCAST_SHAPE);
}

#[test]
fn subtrahe_and_multiplica_are_elementwise() {
    let a = Tensor::structa(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();
    let b = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert_eq!(
        a.subtrahe(&b)
            .expect("broadcast-compatible shape")
            .planata(),
        vec![9.0, 18.0, 27.0]
    );
    assert_eq!(
        a.multiplica(&b)
            .expect("broadcast-compatible shape")
            .planata(),
        vec![10.0, 40.0, 90.0]
    );
}

#[test]
fn addita_integer_tensors_sum_without_widening() {
    let a = Tensor::structa(vec![1i64, 2, 3], &[3]).unwrap();
    let b = Tensor::structa(vec![4i64, 5, 6], &[3]).unwrap();
    assert_eq!(
        a.addita(&b).expect("broadcast-compatible shape").planata(),
        vec![5, 7, 9]
    );
}

#[test]
fn summa_folds_all_elements_to_element_type() {
    let grid = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(grid.summa(), 10.0);
    let ints = Tensor::structa(vec![1i64, 2, 3, 4], &[4]).unwrap();
    assert_eq!(ints.summa(), 10);
}

#[test]
fn scala_scales_f32_elements_and_preserves_shape() {
    let tensor = Tensor::structa(vec![1.0f32, -2.0, 3.5, 4.0], &[2, 2]).unwrap();

    let scaled = tensor.scala(0.5);

    assert_eq!(scaled.magnitudines(), vec![2, 2]);
    assert_eq!(scaled.planata(), vec![0.5, -1.0, 1.75, 2.0]);
}

#[test]
fn divide_broadcasts_finite_f32_tensors() {
    let lhs = Tensor::structa(vec![8.0f32, 18.0, -24.0, 40.0], &[2, 2]).unwrap();
    let rhs = Tensor::structa(vec![2.0f32, -4.0], &[2, 1]).unwrap();

    let divided = lhs.divide(&rhs).expect("finite broadcast division");

    assert_eq!(divided.magnitudines(), vec![2, 2]);
    assert_eq!(divided.planata(), vec![4.0, 9.0, 6.0, -10.0]);
}

#[test]
fn reciproca_preserves_shape_and_checks_denominators() {
    let tensor = Tensor::structa(vec![2.0f32, -4.0, 0.25, 8.0], &[2, 2]).unwrap();

    let reciprocal = tensor.reciproca().expect("finite reciprocal");

    assert_eq!(reciprocal.magnitudines(), vec![2, 2]);
    assert_eq!(reciprocal.planata(), vec![0.5, -0.25, 4.0, 0.125]);

    let zero = Tensor::structa(vec![1.0f32, 0.0], &[2]).unwrap();
    assert_eq!(zero.reciproca().unwrap_err(), ERR_DIVIDE_ZERO_DENOMINATOR);
}

#[test]
fn divide_rejects_zero_denominator_without_materializing_infinity() {
    let lhs = Tensor::structa(vec![1.0f32, -2.0], &[2]).unwrap();
    let rhs = Tensor::structa(vec![1.0f32, -0.0], &[2]).unwrap();

    assert_eq!(lhs.divide(&rhs).unwrap_err(), ERR_DIVIDE_ZERO_DENOMINATOR);
}

#[test]
fn divide_rejects_non_finite_inputs_before_dividing() {
    let lhs = Tensor::structa(vec![1.0f32, f32::INFINITY], &[2]).unwrap();
    let rhs = Tensor::structa(vec![1.0f32, 2.0], &[2]).unwrap();
    assert_eq!(lhs.divide(&rhs).unwrap_err(), ERR_DIVIDE_NON_FINITE_INPUT);

    let lhs = Tensor::structa(vec![1.0f32], &[]).unwrap();
    let rhs = Tensor::structa(vec![f32::NAN], &[]).unwrap();
    assert_eq!(lhs.divide(&rhs).unwrap_err(), ERR_DIVIDE_NON_FINITE_INPUT);
}

#[test]
fn divide_rejects_non_finite_results() {
    let lhs = Tensor::structa(vec![f32::MAX], &[]).unwrap();
    let rhs = Tensor::structa(vec![f32::MIN_POSITIVE], &[]).unwrap();

    assert_eq!(lhs.divide(&rhs).unwrap_err(), ERR_DIVIDE_NON_FINITE_RESULT);
}

#[test]
fn divide_rejects_broadcast_shape_mismatch() {
    let lhs = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let rhs = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();

    assert_eq!(lhs.divide(&rhs).unwrap_err(), ERR_BROADCAST_SHAPE);
}

#[test]
fn media_averages_f32_elements_and_rejects_empty_tensor() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let empty = Tensor::<f32>::structa(Vec::new(), &[0]).unwrap();

    assert_eq!(tensor.media().unwrap(), 2.5);
    assert_eq!(empty.media().unwrap_err(), ERR_MEDIA_EMPTY);
}

#[test]
fn transpose_rank2_materializes_rows_as_columns() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let transposed = tensor.transpose_rank2().expect("rank-2 transpose");

    assert_eq!(transposed.magnitudines(), vec![3, 2]);
    assert_eq!(transposed.planata(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_rank2_materializes_views_without_aliasing() {
    let mut tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let view = tensor.sectio(1, 3).expect("axis-0 view");
    let transposed = view.transpose_rank2().expect("rank-2 view transpose");

    tensor.ponde(&[1, 0], 99.0).unwrap();

    assert_eq!(transposed.magnitudines(), vec![2, 2]);
    assert_eq!(transposed.planata(), vec![3.0, 5.0, 4.0, 6.0]);
}

#[test]
fn transpose_rank2_rejects_non_rank2_tensor() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();

    assert_eq!(tensor.transpose_rank2().unwrap_err(), ERR_TRANSPOSE_RANK);
}

#[test]
fn permute_materializes_general_axis_order() {
    let tensor = Tensor::structa((0..24).collect::<Vec<i32>>(), &[2, 3, 4]).unwrap();

    let permuted = tensor.permute(&[2, 0, 1]).expect("valid axis order");

    assert_eq!(permuted.magnitudines(), vec![4, 2, 3]);
    assert_eq!(
        permuted.planata(),
        vec![0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23]
    );
}

#[test]
fn permute_materializes_views_without_aliasing() {
    let mut tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let view = tensor.sectio(1, 3).expect("axis-0 view");
    let permuted = view.permute(&[1, 0]).expect("rank-2 view permute");

    tensor.ponde(&[1, 0], 99.0).unwrap();

    assert_eq!(permuted.magnitudines(), vec![2, 2]);
    assert_eq!(permuted.planata(), vec![3.0, 5.0, 4.0, 6.0]);
}

#[test]
fn permute_accepts_rank_zero_empty_axis_list() {
    let tensor = Tensor::structa(vec![42_i32], &[]).unwrap();

    let permuted = tensor.permute(&[]).expect("rank-0 identity permute");

    assert_eq!(permuted.magnitudines(), Vec::<i64>::new());
    assert_eq!(permuted.planata(), vec![42]);
}

#[test]
fn permute_rejects_rank_mismatch_and_missing_axis() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(tensor.permute(&[0]).unwrap_err(), ERR_PERMUTE_RANK);
}

#[test]
fn permute_rejects_negative_axis() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(
        tensor.permute(&[0, -1]).unwrap_err(),
        ERR_PERMUTE_NEGATIVE_AXIS
    );
}

#[test]
fn permute_rejects_axis_out_of_range() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(
        tensor.permute(&[0, 2]).unwrap_err(),
        ERR_PERMUTE_AXIS_OUT_OF_RANGE
    );
}

#[test]
fn permute_rejects_duplicate_axis() {
    let tensor = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(
        tensor.permute(&[0, 0]).unwrap_err(),
        ERR_PERMUTE_DUPLICATE_AXIS
    );
}

#[test]
fn permute_rejects_rank_zero_non_empty_axis_list() {
    let tensor = Tensor::structa(vec![42_i32], &[]).unwrap();

    assert_eq!(tensor.permute(&[0]).unwrap_err(), ERR_PERMUTE_RANK);
}

#[test]
fn matmul_square_identity() {
    // I₃ × A = A
    let eye = Tensor::structa(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]).unwrap();
    let a = Tensor::structa(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();
    let result = eye.matmul(&a).expect("valid matmul");
    assert_eq!(result.magnitudines(), vec![3, 3]);
    assert_eq!(
        result.planata(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn matmul_rectangular() {
    // [2,3] × [3,4] → [2,4]
    let a = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::structa(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    )
    .unwrap();
    let result = a.matmul(&b).expect("valid matmul");
    assert_eq!(result.magnitudines(), vec![2, 4]);
    // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
    //       = [38, 44, 50, 56]
    // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
    //       = [83, 98, 113, 128]
    assert_eq!(
        result.planata(),
        vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
    );
}

#[test]
fn matmul_receiver_rank_rejects_with_error() {
    let a = Tensor::structa(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::structa(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(a.matmul(&b).unwrap_err(), ERR_MATMUL_RECEIVER_RANK);
}

#[test]
fn matmul_argument_rank_rejects_with_error() {
    let a = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::structa(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert_eq!(a.matmul(&b).unwrap_err(), ERR_MATMUL_ARGUMENT_RANK);
}

#[test]
fn matmul_inner_mismatch_rejects_with_error() {
    let a = Tensor::structa(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::structa(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
    assert_eq!(a.matmul(&b).unwrap_err(), ERR_MATMUL_INNER_DIMENSION);
}

#[test]
fn matmul_rejects_overflowing_result_shape_before_allocation() {
    let a = Tensor::<f32>::crea(&[i64::MAX, 0], 0.0).expect("zero-element huge receiver");
    let b = Tensor::<f32>::crea(&[0, 2], 0.0).expect("zero-element argument");

    assert_eq!(a.matmul(&b).unwrap_err(), ERR_ELEMENT_COUNT_OVERFLOW);
}
