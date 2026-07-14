use crate::Tensor;

const FINITE_DIFFERENCE_EPSILON: f32 = 1.0e-3;
const FINITE_DIFFERENCE_TOLERANCE: f32 = 2.0e-3;

fn finite_difference_gradient<F>(params: &[f32], loss: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    (0..params.len())
        .map(|index| {
            let mut plus = params.to_vec();
            plus[index] += FINITE_DIFFERENCE_EPSILON;
            let mut minus = params.to_vec();
            minus[index] -= FINITE_DIFFERENCE_EPSILON;
            (loss(&plus) - loss(&minus)) / (2.0 * FINITE_DIFFERENCE_EPSILON)
        })
        .collect()
}

fn assert_gradient_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= FINITE_DIFFERENCE_TOLERANCE,
            "gradient[{index}] expected {expected}, got {actual}, delta {delta}"
        );
    }
}

fn rank_zero_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(vec![params[0]], &[]).expect("rank-zero scalar tensor");
    let x_squared = x.multiplica(&x).expect("same-shape scalar multiply");
    x_squared.addita(&x).expect("same-shape scalar add").summa()
}

fn same_shape_vector_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(params[0..3].to_vec(), &[3]).expect("x tensor");
    let w = Tensor::structa(params[3..6].to_vec(), &[3]).expect("w tensor");
    let target = Tensor::structa(vec![1.0, -2.0, 0.5], &[3]).expect("target tensor");

    let prediction = x.multiplica(&w).expect("same-shape elementwise multiply");
    let residual = prediction
        .subtrahe(&target)
        .expect("same-shape elementwise subtract");
    residual
        .multiplica(&residual)
        .expect("same-shape square")
        .summa()
}

#[test]
fn finite_difference_reference_checks_rank_zero_scalar_loss() {
    let params = vec![1.75_f32];
    let gradient = finite_difference_gradient(&params, rank_zero_loss);
    let expected = vec![2.0 * params[0] + 1.0];

    assert_gradient_close(&gradient, &expected);
}

#[test]
fn finite_difference_reference_checks_same_shape_vector_loss() {
    let params = vec![0.5_f32, -1.0, 2.0, 3.0, -0.25, 0.75];
    let gradient = finite_difference_gradient(&params, same_shape_vector_loss);

    let x = &params[0..3];
    let w = &params[3..6];
    let target = [1.0_f32, -2.0, 0.5];
    let residuals: Vec<f32> = x
        .iter()
        .zip(w.iter())
        .zip(target.iter())
        .map(|((x, w), target)| x * w - target)
        .collect();
    let mut expected = Vec::with_capacity(params.len());
    expected.extend(
        residuals
            .iter()
            .zip(w.iter())
            .map(|(residual, w)| 2.0 * residual * w),
    );
    expected.extend(
        residuals
            .iter()
            .zip(x.iter())
            .map(|(residual, x)| 2.0 * residual * x),
    );

    assert_gradient_close(&gradient, &expected);
}
