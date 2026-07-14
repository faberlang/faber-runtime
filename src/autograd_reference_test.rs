use crate::autograd::{AutogradTape, AutogradValue};
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

fn broadcast_bias_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(params[0..4].to_vec(), &[2, 2]).expect("x tensor");
    let bias = Tensor::structa(params[4..6].to_vec(), &[2, 1]).expect("bias tensor");
    let target = Tensor::structa(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]).expect("target tensor");

    let prediction = x.addita(&bias).expect("row bias broadcasts across columns");
    let residual = prediction
        .subtrahe(&target)
        .expect("same-shape elementwise subtract");
    residual
        .multiplica(&residual)
        .expect("same-shape square")
        .summa()
}

fn linear_training_step_loss(params: &[f32]) -> f32 {
    let input = Tensor::structa(params[0..4].to_vec(), &[2, 2]).expect("input tensor");
    let weight = Tensor::structa(params[4..8].to_vec(), &[2, 2]).expect("weight tensor");
    let bias = Tensor::structa(params[8..10].to_vec(), &[1, 2]).expect("bias tensor");
    let target = Tensor::structa(vec![0.25, -1.0, 1.5, 0.75], &[2, 2]).expect("target tensor");

    let prediction = input.matmul(&weight).expect("rank-2 linear matmul");
    let shifted = prediction.addita(&bias).expect("batch bias broadcasts");
    let residual = shifted
        .subtrahe(&target)
        .expect("same-shape elementwise subtract");
    residual
        .multiplica(&residual)
        .expect("same-shape square")
        .summa()
}

fn linear_training_step_autograd_gradient(params: &[f32]) -> Vec<f32> {
    let mut tape = AutogradTape::new();
    let input = leaf(&mut tape, tensor(&params[0..4], &[2, 2]));
    let weight = leaf(&mut tape, tensor(&params[4..8], &[2, 2]));
    let bias = leaf(&mut tape, tensor(&params[8..10], &[1, 2]));
    let target = leaf(&mut tape, tensor(&[0.25, -1.0, 1.5, 0.75], &[2, 2]));

    let prediction = tape.matmul(&input, &weight).expect("linear matmul");
    let shifted = tape.add(&prediction, &bias).expect("batch bias broadcasts");
    let residual = tape.sub(&shifted, &target).expect("prediction - target");
    let squared = tape.mul(&residual, &residual).expect("residual squared");
    let loss = tape.summa(&squared).expect("scalar loss");
    let gradients = tape.backward(&loss).expect("backward succeeds");

    let mut actual = Vec::with_capacity(params.len());
    actual.extend(
        gradients
            .gradient(input.id())
            .expect("input gradient")
            .planata(),
    );
    actual.extend(
        gradients
            .gradient(weight.id())
            .expect("weight gradient")
            .planata(),
    );
    actual.extend(
        gradients
            .gradient(bias.id())
            .expect("bias gradient")
            .planata(),
    );
    actual
}

fn apply_linear_parameter_update(params: &[f32], gradient: &[f32], learning_rate: f32) -> Vec<f32> {
    let mut updated = params.to_vec();
    for index in 4..10 {
        updated[index] -= learning_rate * gradient[index];
    }
    updated
}

#[derive(Clone, Debug, PartialEq)]
struct LinearTrainingSession {
    params: Vec<f32>,
    learning_rate: f32,
}

impl LinearTrainingSession {
    fn new(params: Vec<f32>, learning_rate: f32) -> Self {
        Self {
            params,
            learning_rate,
        }
    }

    fn loss(&self) -> f32 {
        linear_training_step_loss(&self.params)
    }

    fn autograd_step(&mut self) {
        let gradient = linear_training_step_autograd_gradient(&self.params);
        self.params = apply_linear_parameter_update(&self.params, &gradient, self.learning_rate);
    }

    fn finite_difference_step(&mut self) {
        let gradient = finite_difference_gradient(&self.params, linear_training_step_loss);
        self.params = apply_linear_parameter_update(&self.params, &gradient, self.learning_rate);
    }
}

fn rung3_scalar_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(vec![params[0]], &[]).expect("rank-zero x tensor");
    let weight = Tensor::structa(vec![params[1]], &[]).expect("rank-zero weight tensor");
    let target = Tensor::structa(vec![params[2]], &[]).expect("rank-zero target tensor");

    let prediction = x.multiplica(&weight).expect("same-shape scalar multiply");
    let residual = prediction
        .subtrahe(&target)
        .expect("same-shape scalar subtract");
    residual
        .multiplica(&residual)
        .expect("same-shape scalar square")
        .summa()
}

fn tensor(values: &[f32], shape: &[i64]) -> Tensor<f32> {
    Tensor::structa(values.to_vec(), shape).expect("test tensor shape matches")
}

fn leaf(tape: &mut AutogradTape, tensor: Tensor<f32>) -> AutogradValue {
    tape.leaf(tensor)
        .expect("materialized tensor is differentiable")
}

#[test]
fn finite_difference_reference_checks_rank_zero_scalar_loss() {
    let params = vec![1.75_f32];
    let gradient = finite_difference_gradient(&params, rank_zero_loss);
    let expected = vec![2.0 * params[0] + 1.0];

    assert_gradient_close(&gradient, &expected);
}

#[test]
fn finite_difference_reference_checks_broadcast_bias_gradient_reduction() {
    let params = vec![0.5_f32, -1.0, 2.0, 4.0, 0.25, -0.75];
    let gradient = finite_difference_gradient(&params, broadcast_bias_loss);

    let x = &params[0..4];
    let bias = &params[4..6];
    let target = [1.0_f32, -2.0, 0.5, 3.0];
    let residuals: Vec<f32> = x
        .chunks_exact(2)
        .zip(bias.iter())
        .flat_map(|(row, bias)| row.iter().map(move |x| x + bias))
        .zip(target.iter())
        .map(|(prediction, target)| prediction - target)
        .collect();

    let mut expected = Vec::with_capacity(params.len());
    expected.extend(residuals.iter().map(|residual| 2.0 * residual));
    expected.extend(
        residuals
            .chunks_exact(2)
            .map(|row| row.iter().map(|residual| 2.0 * residual).sum::<f32>()),
    );

    assert_gradient_close(&gradient, &expected);
}

#[test]
fn autograd_matches_finite_difference_linear_training_step_gradients() {
    let params = vec![0.5_f32, -1.0, 2.0, 0.75, 1.25, -0.5, 0.8, 1.1, 0.2, -0.3];
    let reference = finite_difference_gradient(&params, linear_training_step_loss);
    let actual = linear_training_step_autograd_gradient(&params);

    assert_gradient_close(&actual, &reference);
}

#[test]
fn autograd_parameter_update_matches_finite_difference_linear_oracle() {
    let params = vec![0.5_f32, -1.0, 2.0, 0.75, 1.25, -0.5, 0.8, 1.1, 0.2, -0.3];
    let learning_rate = 0.01;
    let mut reference = LinearTrainingSession::new(params.clone(), learning_rate);
    let mut autograd = LinearTrainingSession::new(params.clone(), learning_rate);
    let initial_loss = autograd.loss();

    reference.finite_difference_step();
    autograd.autograd_step();

    assert_gradient_close(&autograd.params, &reference.params);
    assert_eq!(&autograd.params[0..4], &params[0..4]);
    assert!(
        autograd.loss() < initial_loss,
        "manual weight/bias update should lower the local training loss"
    );
}

#[test]
fn finite_difference_reference_checks_exec_approved_rung3_weight_gradient() {
    let params = vec![2.0_f32, 3.0, 4.0];
    let loss = rung3_scalar_loss(&params);
    let gradient = finite_difference_gradient(&params, rung3_scalar_loss);

    assert_eq!(loss, 4.0);
    assert_gradient_close(&[gradient[1]], &[8.0]);
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
