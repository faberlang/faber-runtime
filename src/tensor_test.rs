use super::{
    tensor_flat_offset, tensor_shape_element_count, tensor_shape_has_element_count, Tensor,
    ERR_BROADCAST_SHAPE, ERR_ELEMENT_COUNT_OVERFLOW, ERR_MATMUL_ARGUMENT_RANK,
    ERR_MATMUL_INNER_DIMENSION, ERR_MATMUL_RECEIVER_RANK,
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
