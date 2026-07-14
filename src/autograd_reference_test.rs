use crate::autograd::{AutogradTape, AutogradValue};
use crate::Tensor;

const FINITE_DIFFERENCE_EPSILON: f32 = 1.0e-3;
const FINITE_DIFFERENCE_TOLERANCE: f32 = 2.0e-3;
const LINEAR_INPUT_RANGE: std::ops::Range<usize> = 0..4;
const LINEAR_WEIGHT_RANGE: std::ops::Range<usize> = 4..8;
const LINEAR_BIAS_RANGE: std::ops::Range<usize> = 8..10;

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

fn assert_strictly_decreasing(trace: &[f32]) {
    assert!(trace.len() >= 2);
    for window in trace.windows(2) {
        assert!(
            window[1] < window[0],
            "loss trace should decrease, got {} then {}",
            window[0],
            window[1]
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

fn mean_square_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(params.to_vec(), &[2, 2]).expect("x tensor");
    let squared = x.multiplica(&x).expect("same-shape square");
    squared.media().expect("non-empty mean")
}

fn scaled_mean_square_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(params.to_vec(), &[2, 2]).expect("x tensor");
    let scaled = x.scala(0.25);
    let squared = scaled.multiplica(&scaled).expect("same-shape square");
    squared.media().expect("non-empty mean")
}

fn mean_square_autograd_gradient(params: &[f32]) -> Vec<f32> {
    let mut tape = AutogradTape::new();
    let x = leaf(&mut tape, tensor(params, &[2, 2]));

    let squared = tape.mul(&x, &x).expect("same-shape square");
    let loss = tape.media(&squared).expect("mean loss");
    let gradients = tape.backward(&loss).expect("backward succeeds");

    gradients.gradient(x.id()).expect("x gradient").planata()
}

fn scaled_mean_square_autograd_gradient(params: &[f32]) -> Vec<f32> {
    let mut tape = AutogradTape::new();
    let x = leaf(&mut tape, tensor(params, &[2, 2]));

    let scaled = tape.scala(&x, 0.25).expect("scala records");
    let squared = tape.mul(&scaled, &scaled).expect("same-shape square");
    let loss = tape.media(&squared).expect("mean loss");
    let gradients = tape.backward(&loss).expect("backward succeeds");

    gradients.gradient(x.id()).expect("x gradient").planata()
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

fn broadcast_bias_autograd_gradient(params: &[f32]) -> Vec<f32> {
    let mut tape = AutogradTape::new();
    let x = leaf(&mut tape, tensor(&params[0..4], &[2, 2]));
    let bias = leaf(&mut tape, tensor(&params[4..6], &[2, 1]));
    let target = leaf(&mut tape, tensor(&[1.0, -2.0, 0.5, 3.0], &[2, 2]));

    let prediction = tape.add(&x, &bias).expect("row bias broadcasts");
    let residual = tape.sub(&prediction, &target).expect("prediction - target");
    let squared = tape.mul(&residual, &residual).expect("residual squared");
    let loss = tape.summa(&squared).expect("scalar loss");
    let gradients = tape.backward(&loss).expect("backward succeeds");

    let mut actual = Vec::with_capacity(params.len());
    actual.extend(gradients.gradient(x.id()).expect("x gradient").planata());
    actual.extend(
        gradients
            .gradient(bias.id())
            .expect("bias gradient")
            .planata(),
    );
    actual
}

fn broadcast_scale_loss(params: &[f32]) -> f32 {
    let x = Tensor::structa(params[0..4].to_vec(), &[2, 2]).expect("x tensor");
    let scale = Tensor::structa(params[4..6].to_vec(), &[2, 1]).expect("scale tensor");
    let target = Tensor::structa(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]).expect("target tensor");

    let prediction = x
        .multiplica(&scale)
        .expect("row scale broadcasts across columns");
    let residual = prediction
        .subtrahe(&target)
        .expect("same-shape elementwise subtract");
    residual
        .multiplica(&residual)
        .expect("same-shape square")
        .summa()
}

fn broadcast_scale_autograd_gradient(params: &[f32]) -> Vec<f32> {
    let mut tape = AutogradTape::new();
    let x = leaf(&mut tape, tensor(&params[0..4], &[2, 2]));
    let scale = leaf(&mut tape, tensor(&params[4..6], &[2, 1]));
    let target = leaf(&mut tape, tensor(&[1.0, -2.0, 0.5, 3.0], &[2, 2]));

    let prediction = tape.mul(&x, &scale).expect("row scale broadcasts");
    let residual = tape.sub(&prediction, &target).expect("prediction - target");
    let squared = tape.mul(&residual, &residual).expect("residual squared");
    let loss = tape.summa(&squared).expect("scalar loss");
    let gradients = tape.backward(&loss).expect("backward succeeds");

    let mut actual = Vec::with_capacity(params.len());
    actual.extend(gradients.gradient(x.id()).expect("x gradient").planata());
    actual.extend(
        gradients
            .gradient(scale.id())
            .expect("scale gradient")
            .planata(),
    );
    actual
}

fn linear_training_step_loss(params: &[f32]) -> f32 {
    let input =
        Tensor::structa(params[LINEAR_INPUT_RANGE].to_vec(), &[2, 2]).expect("input tensor");
    let weight =
        Tensor::structa(params[LINEAR_WEIGHT_RANGE].to_vec(), &[2, 2]).expect("weight tensor");
    let bias = Tensor::structa(params[LINEAR_BIAS_RANGE].to_vec(), &[1, 2]).expect("bias tensor");
    let target = Tensor::structa(vec![0.25, -1.0, 1.5, 0.75], &[2, 2]).expect("target tensor");

    let prediction = input.matmul(&weight).expect("rank-2 linear matmul");
    let shifted = prediction.addita(&bias).expect("batch bias broadcasts");
    let residual = shifted
        .subtrahe(&target)
        .expect("same-shape elementwise subtract");
    residual
        .multiplica(&residual)
        .expect("same-shape square")
        .media()
        .expect("non-empty mean")
}

fn linear_training_step_autograd_gradient(params: &[f32]) -> Vec<f32> {
    let mut tape = AutogradTape::new();
    let input = leaf(&mut tape, tensor(&params[LINEAR_INPUT_RANGE], &[2, 2]));
    let weight = leaf(&mut tape, tensor(&params[LINEAR_WEIGHT_RANGE], &[2, 2]));
    let bias = leaf(&mut tape, tensor(&params[LINEAR_BIAS_RANGE], &[1, 2]));
    let target = leaf(&mut tape, tensor(&[0.25, -1.0, 1.5, 0.75], &[2, 2]));

    let prediction = tape.matmul(&input, &weight).expect("linear matmul");
    let shifted = tape.add(&prediction, &bias).expect("batch bias broadcasts");
    let residual = tape.sub(&shifted, &target).expect("prediction - target");
    let squared = tape.mul(&residual, &residual).expect("residual squared");
    let loss = tape.media(&squared).expect("mean loss");
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

fn zero_frozen_linear_gradient(gradient: &mut [f32]) {
    for index in LINEAR_INPUT_RANGE {
        gradient[index] = 0.0;
    }
}

fn apply_test_only_sgd_update(params: &[f32], gradient: &[f32], learning_rate: f32) -> Vec<f32> {
    let mut updated = params.to_vec();
    for (value, gradient) in updated.iter_mut().zip(gradient.iter()) {
        *value -= learning_rate * gradient;
    }
    updated
}

#[derive(Clone, Debug, PartialEq)]
struct TestOnlySgdSession {
    params: Vec<f32>,
    learning_rate: f32,
}

impl TestOnlySgdSession {
    fn new(params: Vec<f32>, learning_rate: f32) -> Self {
        Self {
            params,
            learning_rate,
        }
    }

    fn loss(&self) -> f32 {
        linear_training_step_loss(&self.params)
    }

    fn autograd_trainable_gradient(&self) -> Vec<f32> {
        let mut gradient = linear_training_step_autograd_gradient(&self.params);
        zero_frozen_linear_gradient(&mut gradient);
        gradient
    }

    fn finite_difference_trainable_gradient(&self) -> Vec<f32> {
        let mut gradient = finite_difference_gradient(&self.params, linear_training_step_loss);
        zero_frozen_linear_gradient(&mut gradient);
        gradient
    }

    fn autograd_step(&mut self) -> Vec<f32> {
        let gradient = self.autograd_trainable_gradient();
        self.params = apply_test_only_sgd_update(&self.params, &gradient, self.learning_rate);
        gradient
    }

    fn finite_difference_step(&mut self) -> Vec<f32> {
        let gradient = self.finite_difference_trainable_gradient();
        self.params = apply_test_only_sgd_update(&self.params, &gradient, self.learning_rate);
        gradient
    }

    fn autograd_loss_trace(&mut self, steps: usize) -> Vec<f32> {
        let mut trace = Vec::with_capacity(steps + 1);
        trace.push(self.loss());
        for _ in 0..steps {
            let _gradient = self.autograd_step();
            trace.push(self.loss());
        }
        trace
    }

    fn finite_difference_loss_trace(&mut self, steps: usize) -> Vec<f32> {
        let mut trace = Vec::with_capacity(steps + 1);
        trace.push(self.loss());
        for _ in 0..steps {
            let _gradient = self.finite_difference_step();
            trace.push(self.loss());
        }
        trace
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
fn autograd_matches_finite_difference_broadcast_add_gradient_reduction() {
    let params = vec![0.5_f32, -1.0, 2.0, 4.0, 0.25, -0.75];
    let reference = finite_difference_gradient(&params, broadcast_bias_loss);
    let actual = broadcast_bias_autograd_gradient(&params);

    assert_gradient_close(&actual, &reference);
}

#[test]
fn autograd_matches_finite_difference_broadcast_mul_gradient_reduction() {
    let params = vec![0.5_f32, -1.0, 1.2, -0.7, 0.6, -0.4];
    let reference = finite_difference_gradient(&params, broadcast_scale_loss);
    let actual = broadcast_scale_autograd_gradient(&params);

    assert_gradient_close(&actual, &reference);
}

#[test]
fn autograd_matches_finite_difference_mean_square_gradient() {
    let params = vec![0.5_f32, -1.0, 2.0, -0.75];
    let reference = finite_difference_gradient(&params, mean_square_loss);
    let actual = mean_square_autograd_gradient(&params);

    assert_gradient_close(&actual, &reference);
}

#[test]
fn autograd_matches_finite_difference_scaled_mean_square_gradient() {
    let params = vec![0.5_f32, -1.0, 2.0, -0.75];
    let reference = finite_difference_gradient(&params, scaled_mean_square_loss);
    let actual = scaled_mean_square_autograd_gradient(&params);

    assert_gradient_close(&actual, &reference);
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
    let mut reference = TestOnlySgdSession::new(params.clone(), learning_rate);
    let mut autograd = TestOnlySgdSession::new(params.clone(), learning_rate);
    let initial_loss = autograd.loss();

    let reference_gradient = reference.finite_difference_step();
    let autograd_gradient = autograd.autograd_step();

    assert_gradient_close(&autograd_gradient, &reference_gradient);
    assert_gradient_close(&autograd.params, &reference.params);
    assert_eq!(
        &autograd.params[LINEAR_INPUT_RANGE],
        &params[LINEAR_INPUT_RANGE]
    );
    assert!(
        autograd.loss() < initial_loss,
        "manual weight/bias update should lower the local training loss"
    );
}

#[test]
fn test_only_sgd_session_zeroes_frozen_input_gradient_before_update() {
    let params = vec![0.5_f32, -1.0, 2.0, 0.75, 1.25, -0.5, 0.8, 1.1, 0.2, -0.3];
    let mut session = TestOnlySgdSession::new(params.clone(), 0.01);

    let gradient = session.autograd_step();

    assert_eq!(&gradient[LINEAR_INPUT_RANGE], &[0.0, 0.0, 0.0, 0.0]);
    assert!(
        gradient[LINEAR_WEIGHT_RANGE]
            .iter()
            .chain(gradient[LINEAR_BIAS_RANGE].iter())
            .any(|value| value.abs() > 1.0e-6),
        "trainable weight/bias gradient should be nonzero"
    );
    assert_eq!(
        &session.params[LINEAR_INPUT_RANGE],
        &params[LINEAR_INPUT_RANGE]
    );
}

#[test]
fn autograd_two_step_loss_trace_matches_finite_difference_session_oracle() {
    let params = vec![0.5_f32, -1.0, 2.0, 0.75, 1.25, -0.5, 0.8, 1.1, 0.2, -0.3];
    let learning_rate = 0.01;
    let mut reference = TestOnlySgdSession::new(params.clone(), learning_rate);
    let mut autograd = TestOnlySgdSession::new(params.clone(), learning_rate);

    let reference_trace = reference.finite_difference_loss_trace(2);
    let autograd_trace = autograd.autograd_loss_trace(2);

    assert_gradient_close(&autograd_trace, &reference_trace);
    assert_strictly_decreasing(&autograd_trace);
    assert_gradient_close(&autograd.params, &reference.params);
    assert_eq!(
        &autograd.params[LINEAR_INPUT_RANGE],
        &params[LINEAR_INPUT_RANGE]
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
