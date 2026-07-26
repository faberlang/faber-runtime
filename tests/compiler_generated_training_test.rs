//! Integration tests: compiler-generated backward companion + training loop
//! match the tape oracle end-to-end.
//!
//! Stage E2 — single-step linear+MSE training step (regression gate).
//! Stage E3 — multi-step loss trace (≥8 SGD steps, strictly decreasing).
//!
//! Compiles `examples/training/linear-regression/` through the MIR pipeline
//! and compares output against oracles computed with the public `faber::Tensor` API.
//!
//! Run: cargo test -p faber-runtime compiler_generated_training

use faber::Tensor;
use std::process::Command;

const FINITE_DIFFERENCE_TOLERANCE: f32 = 2.0e-3;

// Fixed initial parameters — same-shape bias (2×2) to avoid broadcast
// gradient reduction in the backward companion (Stage C gap).
const INPUT: &[f32] = &[0.5, -1.0, 2.0, 0.75];
const WEIGHT: &[f32] = &[1.25, -0.5, 0.8, 1.1];
const BIAS: &[f32] = &[0.2, -0.3, 0.2, -0.3];
const TARGET: &[f32] = &[0.25, -1.0, 1.5, 0.75];

/// Compute the oracle loss using the same formula as the exemplum's forward:
///   loss = mean((input·weight + bias − target)²)
fn oracle_loss() -> f32 {
    let input = Tensor::structa(INPUT.to_vec(), &[2, 2]).expect("input tensor");
    let weight = Tensor::structa(WEIGHT.to_vec(), &[2, 2]).expect("weight tensor");
    let bias = Tensor::structa(BIAS.to_vec(), &[2, 2]).expect("bias tensor");
    let target = Tensor::structa(TARGET.to_vec(), &[2, 2]).expect("target tensor");

    let prediction = input.matmul(&weight).expect("matmul");
    let shifted = prediction.addita(&bias).expect("add bias");
    let residual = shifted.subtrahe(&target).expect("subtract target");
    let squared = residual.multiplica(&residual).expect("square");
    squared.media().expect("mean loss")
}

/// Compute oracle gradients for weight and bias using finite differences.
fn oracle_weight_gradient() -> Vec<f32> {
    let eps = 1.0e-3;
    let mut params: Vec<f32> = INPUT.to_vec();
    params.extend_from_slice(WEIGHT);
    params.extend_from_slice(BIAS);

    let loss = |p: &[f32]| -> f32 {
        let input = Tensor::structa(p[0..4].to_vec(), &[2, 2]).expect("input");
        let weight = Tensor::structa(p[4..8].to_vec(), &[2, 2]).expect("weight");
        let bias = Tensor::structa(p[8..12].to_vec(), &[2, 2]).expect("bias");
        let target = Tensor::structa(TARGET.to_vec(), &[2, 2]).expect("target");
        let prediction = input.matmul(&weight).expect("matmul");
        let shifted = prediction.addita(&bias).expect("add bias");
        let residual = shifted.subtrahe(&target).expect("subtract target");
        let squared = residual.multiplica(&residual).expect("square");
        squared.media().expect("mean loss")
    };

    // Gradients for weight (indices 4..8) + bias (indices 8..12).
    let mut gradients = Vec::with_capacity(8);
    for i in 4..params.len() {
        let orig = params[i];
        params[i] = orig + eps;
        let loss_plus = loss(&params);
        params[i] = orig - eps;
        let loss_minus = loss(&params);
        params[i] = orig;
        gradients.push((loss_plus - loss_minus) / (2.0 * eps));
    }
    gradients
}

/// Compute oracle params after one SGD step (weight + bias updated).
fn oracle_sgd_step() -> (Vec<f32>, Vec<f32>) {
    let grads = oracle_weight_gradient();
    let lr = 0.01;
    let weight_grad: Vec<f32> = grads[0..4].to_vec();
    let bias_grad: Vec<f32> = grads[4..8].to_vec();

    let new_weight: Vec<f32> = WEIGHT
        .iter()
        .zip(weight_grad.iter())
        .map(|(w, g)| w - lr * g)
        .collect();
    let new_bias: Vec<f32> = BIAS
        .iter()
        .zip(bias_grad.iter())
        .map(|(b, g)| b - lr * g)
        .collect();

    (new_weight, new_bias)
}

/// Path to the linear-regression exemplum package relative to
/// `faber-runtime/` (the `CARGO_MANIFEST_DIR` for this test crate).
fn exemplum_path() -> String {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    format!("{}/../examples/training/linear-regression/", manifest_dir)
}

/// Path to the `faber` CLI `Cargo.toml` relative to `faber-runtime/`.
fn faber_manifest_path() -> String {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    format!("{}/../faber/Cargo.toml", manifest_dir)
}

/// Run `faber run <exemplum_path>` and return stdout + stderr.
fn run_exemplum() -> std::io::Result<std::process::Output> {
    let faber_toml = faber_manifest_path();
    let exemplum = exemplum_path();

    Command::new("cargo")
        .args([
            "run",
            "--manifest-path",
            &faber_toml,
            "--",
            "run",
            "-t",
            "fmir",
            &exemplum,
        ])
        .output()
}

/// Parse the loss value from `nota` output.
///
/// The exemplum's `incipit` calls `nota loss` after the forward pass.
/// The output is expected to contain a fractus value on its own line.
/// This parser extracts the first parseable f32 from stdout.
fn parse_loss(stdout: &str) -> Option<f32> {
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Ok(value) = trimmed.parse::<f32>() {
            return Some(value);
        }
    }
    None
}

#[test]
fn compiler_generated_training_step_matches_tape_oracle() {
    let output = run_exemplum().expect("faber run should succeed — companion must be present after f4c5313bf");

    assert!(
        output.status.success(),
        "faber run exited with code {} — companion or forward pass failure.\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Forward pass loss should be present and correct.
    let exemplum_loss = parse_loss(&stdout).unwrap_or_else(|| {
        panic!(
            "exemplum stdout did not contain a parseable loss value.\n\
             stdout:\n{stdout}\n\
             (Companion is present: check for runtime errors in stderr.)"
        );
    });

    let oracle_loss = oracle_loss();
    let delta = (exemplum_loss - oracle_loss).abs();
    assert!(
        delta <= FINITE_DIFFERENCE_TOLERANCE,
        "exemplum loss {exemplum_loss} differs from oracle loss {oracle_loss} \
         by {delta} (tolerance {FINITE_DIFFERENCE_TOLERANCE})"
    );

    // Parse weight and bias after SGD update from nota output.
    // The exemplum outputs: loss, then weight values (on multiple lines
    // if the tensor is multiline), then bias values.
    let all_values: Vec<f32> = stdout
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect();

    // We need at least 1 (loss) + 4 (weight) + 4 (bias) = 9 values.
    assert!(
        all_values.len() >= 9,
        "expected at least 9 numeric values (loss + 4 weight + 4 bias), got {}.\nstdout:\n{stdout}",
        all_values.len(),
    );

    // Loss is all_values[0], weight is all_values[1..5], bias is all_values[5..9].
    let exemplum_weight = &all_values[1..5];
    let exemplum_bias = &all_values[5..9];

    let (oracle_weight, oracle_bias) = oracle_sgd_step();

    for (i, (&actual, &expected)) in exemplum_weight.iter().zip(oracle_weight.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= FINITE_DIFFERENCE_TOLERANCE,
            "weight[{i}]: exemplum {actual} vs oracle {expected} (delta {delta})"
        );
    }

    for (i, (&actual, &expected)) in exemplum_bias.iter().zip(oracle_bias.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= FINITE_DIFFERENCE_TOLERANCE,
            "bias[{i}]: exemplum {actual} vs oracle {expected} (delta {delta})"
        );
    }
}

// ---------------------------------------------------------------------------
// Stage E3 — multi-step loss trace
// ---------------------------------------------------------------------------

/// Same-shape (2×2) bias forward loss matching the exemplum's `linear_loss`.
/// Uses 12-element flat params: [4 input, 4 weight, 4 bias].
fn linear_loss_2x2_bias(params: &[f32]) -> f32 {
    let input = Tensor::structa(params[0..4].to_vec(), &[2, 2]).expect("input");
    let weight = Tensor::structa(params[4..8].to_vec(), &[2, 2]).expect("weight");
    let bias = Tensor::structa(params[8..12].to_vec(), &[2, 2]).expect("bias");
    let target = Tensor::structa(vec![0.25, -1.0, 1.5, 0.75], &[2, 2]).expect("target");

    let prediction = input.matmul(&weight).expect("matmul");
    let shifted = prediction.addita(&bias).expect("add bias");
    let residual = shifted.subtrahe(&target).expect("subtract target");
    let squared = residual.multiplica(&residual).expect("square");
    squared.media().expect("mean loss")
}

/// Compute full multi-step loss trace using finite-difference gradients for
/// trainable params (weight indices 4..8, bias indices 8..12).
///
/// Matches the exemplum's [2,2] same-shape bias exactly.  The
/// `TestOnlySgdSession` oracle in the autograd module uses [1,2] bias, so we
/// compute our own oracle here rather than reusing it.
fn oracle_multi_step_loss_trace(steps: usize) -> Vec<f32> {
    // Flat 12-param layout: [4 input, 4 weight, 4 bias] with [2,2] bias.
    let mut params: Vec<f32> = vec![0.5, -1.0, 2.0, 0.75, 1.25, -0.5, 0.8, 1.1, 0.2, -0.3, 0.2, -0.3];
    let lr = 0.01;
    let eps = 1.0e-3;
    let mut trace = Vec::with_capacity(steps + 1);

    for s in 0..=steps {
        trace.push(linear_loss_2x2_bias(&params));
        if s < steps {
            // Finite-difference gradient for trainable params only.
            let mut gradient = [0.0_f32; 12];
            for i in 4..params.len() {
                let original = params[i];
                params[i] = original + eps;
                let loss_plus = linear_loss_2x2_bias(&params);
                params[i] = original - eps;
                let loss_minus = linear_loss_2x2_bias(&params);
                params[i] = original;
                gradient[i] = (loss_plus - loss_minus) / (2.0 * eps);
            }
            // SGD update for trainable params.
            for i in 4..params.len() {
                params[i] -= lr * gradient[i];
            }
        }
    }
    trace
}

#[test]
fn compiler_generated_loss_trace_matches_tape_oracle() {
    let output = run_exemplum().expect("faber run should succeed for multi-step trace");

    assert!(
        output.status.success(),
        "faber run exited with code {} — companion or forward pass failure.\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse loss trace from nota output — collect all parseable f32 values.
    let exemplum_trace: Vec<f32> = stdout
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect();

    const STEPS: usize = 8;
    assert!(
        exemplum_trace.len() >= STEPS + 1,
        "expected ≥ {} loss values, got {}.\nstdout:\n{stdout}",
        STEPS + 1,
        exemplum_trace.len()
    );
    let exemplum_trace = &exemplum_trace[..STEPS + 1];

    // Oracle trace from finite-difference over same-shape [2,2] bias.
    let oracle_trace = oracle_multi_step_loss_trace(STEPS);

    for (i, (actual, expected)) in exemplum_trace.iter().zip(oracle_trace.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= FINITE_DIFFERENCE_TOLERANCE,
            "step {i}: exemplum loss {actual} vs oracle {expected} (delta {delta})"
        );
    }

    // Strictly decreasing.
    for i in 1..exemplum_trace.len() {
        assert!(
            exemplum_trace[i] < exemplum_trace[i - 1],
            "step {i}: loss {} is not less than previous loss {}",
            exemplum_trace[i],
            exemplum_trace[i - 1]
        );
    }
}
