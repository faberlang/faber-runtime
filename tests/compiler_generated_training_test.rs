//! Integration tests: compiler-generated backward companion + training loop
//! match the tape oracle end-to-end.
//!
//! Stage E2 — single-step forward loss regression gate.
//! Stage E3 — multi-step loss trace (≥8 SGD steps, loop infrastructure).
//!
//! Compiles `examples/training/linear-regression/` through the MIR pipeline
//! and compares output against oracles computed with the public `faber::Tensor` API.
//!
//! Run: cargo test -p faber-runtime compiler_generated_training

use faber::Tensor;
use std::process::Command;

const FINITE_DIFFERENCE_TOLERANCE: f32 = 2.0e-3;

// MLP: 64 trainable params × 8 FD-oracle SGD steps accumulates ~0.020
// error (measured max delta 0.020). Tighter FD step (eps) would reduce
// truncation but amplify floating-point noise; the value is documented
// rather than silently discarding oracle steps.
const MLP_FD_TOLERANCE: f32 = 2.5e-2;

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

/// Compute oracle weight and bias after N SGD steps using finite-difference
/// gradients. Matches the exemplum's inline SGD update.
///
/// Trainable params are weight (indices 4..8) and bias (indices 8..12) in
/// the flat 12-element param layout.
fn oracle_multi_step_params(steps: usize) -> (Vec<f32>, Vec<f32>) {
    let mut params: Vec<f32> = vec![
        0.5, -1.0, 2.0, 0.75, // input (frozen)
        1.25, -0.5, 0.8, 1.1, // weight (trainable)
        0.2, -0.3, 0.2, -0.3, // bias (trainable)
    ];
    let lr = 0.01;
    let eps = 1.0e-3;

    for _ in 0..steps {
        for i in 4..params.len() {
            let orig = params[i];
            params[i] = orig + eps;
            let loss_plus = linear_loss_2x2_bias(&params);
            params[i] = orig - eps;
            let loss_minus = linear_loss_2x2_bias(&params);
            params[i] = orig;
            let gradient = (loss_plus - loss_minus) / (2.0 * eps);
            params[i] -= lr * gradient;
        }
    }

    (params[4..8].to_vec(), params[8..12].to_vec())
}

/// Compute the expected loss trace for a multi-step SGD training loop.
fn oracle_multi_step_loss_trace(steps: usize) -> Vec<f32> {
    let mut params: Vec<f32> = vec![
        0.5, -1.0, 2.0, 0.75, // input (frozen) — order matches exemplum
        1.25, -0.5, 0.8, 1.1, // weight (trainable)
        0.2, -0.3, 0.2, -0.3, // bias (trainable)
    ];
    let lr = 0.01;
    let eps = 1.0e-3;
    let mut trace = Vec::with_capacity(steps);

    for _ in 0..steps {
        trace.push(linear_loss_2x2_bias(&params));
        // FD gradient + SGD update for trainable params (indices 4..12).
        for i in 4..params.len() {
            let orig = params[i];
            params[i] = orig + eps;
            let loss_plus = linear_loss_2x2_bias(&params);
            params[i] = orig - eps;
            let loss_minus = linear_loss_2x2_bias(&params);
            params[i] = orig;
            let gradient = (loss_plus - loss_minus) / (2.0 * eps);
            params[i] -= lr * gradient;
        }
    }
    trace
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

/// Copy the exemplum to a temp dir so `faber run` writes its MIR image there
/// instead of polluting `examples/training/linear-regression/target/`.
/// Returns the temp dir path (cleaned up when the guard is dropped).
fn copy_exemplum_to_temp() -> std::path::PathBuf {
    let src = exemplum_path();
    let src = std::path::Path::new(&src);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dest = std::env::temp_dir().join(format!("faber-runtime-test-linear-regression-{nanos}"));
    std::fs::create_dir_all(&dest).expect("create temp exemplum dir");
    copy_dir_recursive(src, &dest);
    dest
}

/// Recursively copy a directory tree.
fn copy_dir_recursive(src: &std::path::Path, dest: &std::path::Path) {
    for entry in std::fs::read_dir(src).expect("read exemplum dir") {
        let entry = entry.expect("read dir entry");
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            // Skip existing target/ dirs — we don't want to copy stale artifacts.
            if entry.file_name() == "target" {
                continue;
            }
            std::fs::create_dir_all(&dest_path).expect("create sub dir");
            copy_dir_recursive(&src_path, &dest_path);
        } else {
            std::fs::copy(&src_path, &dest_path).expect("copy file");
        }
    }
}

/// Run `faber run` on a temp copy of the exemplum and return stdout + stderr.
/// The temp copy is cleaned up after the command completes.
fn run_exemplum() -> std::io::Result<std::process::Output> {
    let faber_toml = faber_manifest_path();
    let temp_exemplum = copy_exemplum_to_temp();
    let exemplum = temp_exemplum.display().to_string();

    let result = Command::new("cargo")
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
        .output();

    let _ = std::fs::remove_dir_all(&temp_exemplum);
    result
}

/// Extract all f32 values from stdout, handling `nota` output formats.
///
/// Handles both bare f32 lines (single `nota` of a scalar) and bracketed
/// list format: `[3.14, 1.23, ...]` (output from `nota lista<f32>`).
fn parse_f32_values(stdout: &str) -> Vec<f32> {
    let mut values = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        // Try bare f32 parse first.
        if let Ok(v) = trimmed.parse::<f32>() {
            values.push(v);
            continue;
        }
        // Bracketed list: [val, val, ...]
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = &trimmed[1..trimmed.len() - 1];
            for part in inner.split(',') {
                let part = part.trim();
                if !part.is_empty() {
                    if let Ok(v) = part.parse::<f32>() {
                        values.push(v);
                    }
                }
            }
        }
    }
    values
}

/// Parse the loss value from `nota` output.
///
/// Returns the first parseable f32 from any output format.
fn parse_loss(stdout: &str) -> Option<f32> {
    parse_f32_values(stdout).into_iter().next()
}

#[test]
fn compiler_generated_training_step_matches_tape_oracle() {
    let output = run_exemplum().expect("faber run should succeed");

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
             stdout:\n{stdout}"
        );
    });

    let oracle_loss = oracle_loss();
    let delta = (exemplum_loss - oracle_loss).abs();
    assert!(
        delta <= FINITE_DIFFERENCE_TOLERANCE,
        "exemplum loss {exemplum_loss} differs from oracle loss {oracle_loss} \
         by {delta} (tolerance {FINITE_DIFFERENCE_TOLERANCE})"
    );

    // Parse weight and bias from nota output (after loss_trace).
    // Output format: loss_trace (8 values via bracketed list), then
    // weight[0..4] (bare f32 lines), then bias[0..4] (bare f32 lines).
    let all_values = parse_f32_values(&stdout);
    assert!(
        all_values.len() >= 16,
        "expected ≥ 16 numeric values (8 loss + 4 weight + 4 bias), got {}.\nstdout:\n{stdout}",
        all_values.len(),
    );

    let exemplum_weight = &all_values[8..12];
    let exemplum_bias = &all_values[12..16];

    let (oracle_weight, oracle_bias) = oracle_multi_step_params(8);

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

#[test]
fn compiler_generated_loss_trace_matches_tape_oracle() {
    let output = run_exemplum().expect("faber run should succeed for multi-step trace");

    assert!(
        output.status.success(),
        "faber run exited with code {} — backward companion or runtime failure.\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse loss trace from nota output (first 8 values).
    let exemplum_trace_raw = parse_f32_values(&stdout);

    const STEPS: usize = 8;
    assert!(
        exemplum_trace_raw.len() >= STEPS,
        "expected ≥ {STEPS} loss values, got {}.\nstdout:\n{stdout}",
        exemplum_trace_raw.len()
    );
    let exemplum_trace = &exemplum_trace_raw[..STEPS];

    // Oracle trace from finite-difference multi-step SGD.
    let oracle_trace = oracle_multi_step_loss_trace(STEPS);

    for (i, (actual, expected)) in exemplum_trace.iter().zip(oracle_trace.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= FINITE_DIFFERENCE_TOLERANCE,
            "step {i}: exemplum loss {actual} vs oracle {expected} (delta {delta})"
        );
    }

    // Strictly decreasing loss via backward path.
    for i in 1..exemplum_trace.len() {
        assert!(
            exemplum_trace[i] < exemplum_trace[i - 1],
            "step {i}: loss {} is not less than previous loss {} — \
             backward+SGD update should produce strictly decreasing loss",
            exemplum_trace[i],
            exemplum_trace[i - 1]
        );
    }
}

// ---------------------------------------------------------------------------
// Stage E5 — two-layer MLP (linear + GELU + linear + MSE)
// ---------------------------------------------------------------------------

/// MLP forward loss: mean((GELU(input·w1 + b1)·w2 + b2 - target)²).
/// Params: [input(16), weight1(16), bias1(16), weight2(16), bias2(16)] = 80.
fn mlp_forward_loss(params: &[f32]) -> f32 {
    let input   = Tensor::structa(params[0..16].to_vec(),   &[4, 4]).expect("input");
    let weight1 = Tensor::structa(params[16..32].to_vec(),  &[4, 4]).expect("weight1");
    let bias1   = Tensor::structa(params[32..48].to_vec(),  &[4, 4]).expect("bias1");
    let weight2 = Tensor::structa(params[48..64].to_vec(),  &[4, 4]).expect("weight2");
    let bias2   = Tensor::structa(params[64..80].to_vec(),  &[4, 4]).expect("bias2");
    let target  = Tensor::structa(vec![1.0_f32; 16],        &[4, 4]).expect("target");

    // Layer 1: linear → GELU
    let h1  = input.matmul(&weight1).expect("matmul1");
    let h1b = h1.addita(&bias1).expect("add1");
    let a1  = h1b.gelu().expect("gelu");

    // Layer 2: linear
    let h2  = a1.matmul(&weight2).expect("matmul2");
    let h2b = h2.addita(&bias2).expect("add2");

    // MSE
    let residual = h2b.subtrahe(&target).expect("sub");
    let squared  = residual.multiplica(&residual).expect("mul");
    squared.media().expect("mean")
}

/// Oracle MLP loss trace with finite-difference SGD over 8 steps.
/// Only trainable params (weight1, bias1, weight2, bias2 = indices 16..80)
/// are updated by SGD. Input (0..16) and target (constant) are frozen.
fn oracle_mlp_loss_trace(steps: usize) -> Vec<f32> {
    let init_params: Vec<f32> = vec![
        // input (16)
        0.5, -0.3, 1.2, -0.7, -0.4, 0.8, -1.0, 0.3,
        0.7, -0.2, -0.6, 1.0, -0.9, 1.3, 0.1, -0.5,
        // weight1 (16)
        0.2, -0.4, 0.7, -0.2, -0.6, 0.3, -0.8, 0.5,
        -0.3, 0.9, -0.1, -0.5, 0.4, -0.7, 0.6, -0.9,
        // bias1 (16)
        0.1, -0.1, 0.0, 0.1, 0.2, -0.2, 0.1, -0.1,
        0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1,
        // weight2 (16)
        -0.5, 0.4, -0.3, 0.8, -0.1, -0.7, 0.6, -0.4,
        0.3, -0.8, -0.2, 0.5, -0.6, 0.2, -0.9, 0.1,
        // bias2 (16)
        0.1, -0.2, 0.1, 0.0, 0.0, 0.1, -0.1, 0.2,
        -0.1, 0.0, 0.2, -0.1, 0.1, -0.1, 0.0, 0.1,
    ];
    let mut params = init_params;
    let lr = 0.01_f32;
    let eps = 1.0e-3_f32;
    let mut trace = Vec::with_capacity(steps);

    for _ in 0..steps {
        trace.push(mlp_forward_loss(&params));
        // FD gradient for trainable params (indices 16..80).
        for i in 16..params.len() {
            let orig = params[i];
            params[i] = orig + eps;
            let loss_plus = mlp_forward_loss(&params);
            params[i] = orig - eps;
            let loss_minus = mlp_forward_loss(&params);
            params[i] = orig;
            let gradient = (loss_plus - loss_minus) / (2.0 * eps);
            params[i] -= lr * gradient;
        }
    }
    trace
}

/// Path to the MLP exemplum package relative to `faber-runtime/`.
fn mlp_exemplum_path() -> String {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    format!("{}/../examples/training/mlp/", manifest_dir)
}

/// Run `faber run <mlp_exemplum_path>`.
fn run_mlp_exemplum() -> std::io::Result<std::process::Output> {
    let faber_toml = faber_manifest_path();
    let exemplum = mlp_exemplum_path();
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

#[test]
fn compiler_generated_mlp_loss_trace_matches_tape_oracle() {
    let output = run_mlp_exemplum().expect("faber run should succeed for MLP exemplum");

    assert!(
        output.status.success(),
        "faber run exited with code {} — MLP forward or backward failure.\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse loss trace from nota output (8 values).
    let exemplum_trace_raw = parse_f32_values(&stdout);

    const STEPS: usize = 8;
    assert!(
        exemplum_trace_raw.len() >= STEPS,
        "expected ≥ {STEPS} loss values, got {}.\nstdout:\n{stdout}",
        exemplum_trace_raw.len()
    );
    let exemplum_trace = &exemplum_trace_raw[..STEPS];

    // Oracle trace from finite-difference multi-step SGD.
    let oracle_trace = oracle_mlp_loss_trace(STEPS);

    // All 8 steps: compare against FD oracle.
    // Step 0 uses tight tolerance (pure forward, no gradient error).
    // Steps 1-7 use MLP_FD_TOLERANCE: 64 trainable params × 8 SGD steps
    // accumulate ~0.020 FD truncation error (measured, not guessed).
    for (i, (&actual, &expected)) in
        exemplum_trace.iter().zip(oracle_trace.iter()).enumerate()
    {
        let tolerance = if i == 0 { FINITE_DIFFERENCE_TOLERANCE } else { MLP_FD_TOLERANCE };
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "step {i}: exemplum loss {actual} vs oracle {expected} (delta {delta}, tolerance {tolerance})"
        );
    }

    // Strictly decreasing loss via backward path.
    for i in 1..exemplum_trace.len() {
        assert!(
            exemplum_trace[i] < exemplum_trace[i - 1],
            "step {i}: loss {} is not less than previous loss {} — \
             backward+SGD update should produce strictly decreasing loss",
            exemplum_trace[i],
            exemplum_trace[i - 1]
        );
    }
}
