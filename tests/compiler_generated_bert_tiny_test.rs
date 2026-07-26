//! Integration test: BERT-tiny training fragment loss trace matches
//! a finite-difference oracle.
//!
//! Compiles `examples/training/bert-tiny-fragment/` through the MIR pipeline
//! and compares the 8-step loss trace against an oracle computed with the
//! public `faber::Tensor` API.
//!
//! Run: cargo test -p faber-runtime compiler_generated_bert_tiny

use faber::Tensor;
use std::process::Command;

// BERT-tiny FD oracle tolerance — wider than MLP (2.5e-2) due to
// 528 trainable params × 8 SGD steps accumulating FD truncation error
// through 3×LayerNorm + Softmax + 7×MatMul gradient chains.
// Measured after implementation; adjust as needed.
const BERT_TINY_FD_TOLERANCE: f32 = 5.0e-2;

// ---------------------------------------------------------------------------
// Paths and runners
// ---------------------------------------------------------------------------

/// Path to the BERT-tiny exemplum package relative to `faber-runtime/`.
fn bert_tiny_exemplum_path() -> String {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    format!("{}/../examples/training/bert-tiny-fragment/", manifest_dir)
}

/// Path to the `faber` CLI `Cargo.toml` relative to `faber-runtime/`.
fn faber_manifest_path() -> String {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    format!("{}/../faber/Cargo.toml", manifest_dir)
}

/// Run `faber run <bert_tiny_exemplum_path>`.
fn run_bert_tiny_exemplum() -> std::io::Result<std::process::Output> {
    let faber_toml = faber_manifest_path();
    let exemplum = bert_tiny_exemplum_path();
    Command::new("cargo")
        .args(["run", "--manifest-path", &faber_toml, "--", "run", "-t", "fmir", &exemplum])
        .output()
}

/// Extract all f32 values from stdout, handling `nota` output formats.
fn parse_f32_values(stdout: &str) -> Vec<f32> {
    let mut values = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Ok(v) = trimmed.parse::<f32>() {
            values.push(v);
            continue;
        }
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

// ---------------------------------------------------------------------------
// BERT-tiny forward loss oracle
// ---------------------------------------------------------------------------
// Param layout (flat Vec<f32>, 564 floats total):
//   [  0.. 16] input         (2×8, frozen)
//   [ 16.. 80] wq            (8×8, trainable)
//   [ 80.. 96] bq            (2×8, trainable)
//   [ 96..160] wk            (8×8, trainable)
//   [160..176] bk            (2×8, trainable)
//   [176..240] wv            (8×8, trainable)
//   [240..256] bv            (2×8, trainable)
//   [256..320] wo            (8×8, trainable)
//   [320..336] bo            (2×8, trainable)
//   [336..400] wf1           (8×8, trainable)
//   [400..416] bf1           (2×8, trainable)
//   [416..480] wf2           (8×8, trainable)
//   [480..496] bf2           (2×8, trainable)
//   [496..504] ln1_s         (8,   trainable)
//   [504..512] ln1_o         (8,   trainable)
//   [512..520] ln2_s         (8,   trainable)
//   [520..528] ln2_o         (8,   trainable)
//   [528..536] ln3_s         (8,   trainable)
//   [536..544] ln3_o         (8,   trainable)
//   [544..548] dk_scale      (2×2, frozen)
//   [548..564] target        (2×8, frozen)
// Trainable: indices 16..544 (528 floats).
//
// NOTE: Biases are [2, 8] (not [8]) to avoid rank-extension broadcast
// unsupported by reverse AD. This matches the exemplum's shape.

/// BERT-tiny forward loss: full single-layer transformer with MSE.
fn bert_tiny_forward_loss(params: &[f32]) -> f32 {
    let input = Tensor::structa(params[0..16].to_vec(), &[2, 8]).expect("input");
    let wq = Tensor::structa(params[16..80].to_vec(), &[8, 8]).expect("wq");
    let bq = Tensor::structa(params[80..96].to_vec(), &[2, 8]).expect("bq");
    let wk = Tensor::structa(params[96..160].to_vec(), &[8, 8]).expect("wk");
    let bk = Tensor::structa(params[160..176].to_vec(), &[2, 8]).expect("bk");
    let wv = Tensor::structa(params[176..240].to_vec(), &[8, 8]).expect("wv");
    let bv = Tensor::structa(params[240..256].to_vec(), &[2, 8]).expect("bv");
    let wo = Tensor::structa(params[256..320].to_vec(), &[8, 8]).expect("wo");
    let bo = Tensor::structa(params[320..336].to_vec(), &[2, 8]).expect("bo");
    let wf1 = Tensor::structa(params[336..400].to_vec(), &[8, 8]).expect("wf1");
    let bf1 = Tensor::structa(params[400..416].to_vec(), &[2, 8]).expect("bf1");
    let wf2 = Tensor::structa(params[416..480].to_vec(), &[8, 8]).expect("wf2");
    let bf2 = Tensor::structa(params[480..496].to_vec(), &[2, 8]).expect("bf2");
    let ln1_s = Tensor::structa(params[496..504].to_vec(), &[8]).expect("ln1_s");
    let ln1_o = Tensor::structa(params[504..512].to_vec(), &[8]).expect("ln1_o");
    let ln2_s = Tensor::structa(params[512..520].to_vec(), &[8]).expect("ln2_s");
    let ln2_o = Tensor::structa(params[520..528].to_vec(), &[8]).expect("ln2_o");
    let ln3_s = Tensor::structa(params[528..536].to_vec(), &[8]).expect("ln3_s");
    let ln3_o = Tensor::structa(params[536..544].to_vec(), &[8]).expect("ln3_o");
    let target = Tensor::structa(params[548..564].to_vec(), &[2, 8]).expect("target");

    // Pre-LN 1
    let ln1 = input.layernorm(1, 1e-5, Some(&ln1_s), Some(&ln1_o)).expect("ln1");

    // QKV projections
    let q = ln1.matmul(&wq).expect("q");
    let qb = q.addita(&bq).expect("qb");
    let k = ln1.matmul(&wk).expect("k");
    let kb = k.addita(&bk).expect("kb");
    let v = ln1.matmul(&wv).expect("v");
    let vb = v.addita(&bv).expect("vb");

    // Attention scores
    let kt = kb.transpose_rank2().expect("kt");
    let scores = qb.matmul(&kt).expect("scores");

    // Scale: scores * (1/sqrt(8))
    let dk_scale_tensor =
        Tensor::structa(vec![0.35355339_f32; 4], &[2, 2]).expect("scale");
    let scaled = scores.multiplica(&dk_scale_tensor).expect("scaled");

    // Softmax
    let attn_weights = scaled.softmax().expect("softmax");

    // Context
    let context = attn_weights.matmul(&vb).expect("context");

    // Output + residual
    let ao = context.matmul(&wo).expect("ao");
    let aob = ao.addita(&bo).expect("aob");
    let r1 = input.addita(&aob).expect("r1");

    // Post-attention LN
    let ln2 = r1.layernorm(1, 1e-5, Some(&ln2_s), Some(&ln2_o)).expect("ln2");

    // FFN
    let h1 = ln2.matmul(&wf1).expect("h1");
    let h1b = h1.addita(&bf1).expect("h1b");
    let a1 = h1b.gelu().expect("a1");
    let h2 = a1.matmul(&wf2).expect("h2");
    let h2b = h2.addita(&bf2).expect("h2b");
    let r2 = ln2.addita(&h2b).expect("r2");

    // Pre-loss LN
    let ln3 = r2.layernorm(1, 1e-5, Some(&ln3_s), Some(&ln3_o)).expect("ln3");

    // MSE
    let residual = ln3.subtrahe(&target).expect("sub");
    let squared = residual.multiplica(&residual).expect("mul");
    squared.media().expect("mean")
}

// ---------------------------------------------------------------------------
// FD oracle: 8-step SGD loss trace
// ---------------------------------------------------------------------------

/// Oracle BERT-tiny loss trace with finite-difference SGD.
///
/// Only trainable params (indices 16..544) are updated. Input (0..16),
/// dk_scale (544..548), and target (548..564) are frozen.
fn oracle_bert_tiny_loss_trace(steps: usize) -> Vec<f32> {
    let init_params: Vec<f32> = vec![
        // input (16)
        0.5, -0.3, 1.2, -0.7, -0.4, 0.8, -1.0, 0.3,
        0.7, -0.2, -0.6, 1.0, -0.9, 1.3, 0.1, -0.5,
        // wq (64)
        0.2, -0.4, 0.7, -0.2, -0.6, 0.3, -0.8, 0.5,
        -0.3, 0.9, -0.1, -0.5, 0.4, -0.7, 0.6, -0.9,
        0.1, -0.2, -0.8, 0.6, -0.4, 0.5, -0.3, 0.7,
        0.0, -0.6, 0.3, -0.9, 0.8, -0.1, -0.7, 0.2,
        0.5, -0.5, 0.0, 0.4, -0.4, -0.2, 0.9, -0.8,
        0.2, 0.1, -0.7, -0.3, 0.6, -0.5, 0.3, -0.1,
        -0.9, 0.4, 0.5, -0.6, 0.1, 0.8, -0.4, -0.3,
        -0.2, 0.7, -0.8, 0.3, -0.9, -0.5, 0.0, 0.6,
        // bq (16) — duplicated [8]→[2,8]
        0.1, -0.1, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0,
        0.1, -0.1, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0,
        // wk (64)
        -0.5, 0.4, -0.3, 0.8, -0.1, -0.7, 0.6, -0.4,
        0.3, -0.8, -0.2, 0.5, -0.6, 0.2, -0.9, 0.1,
        -0.3, 0.6, -0.7, 0.0, 0.4, -0.5, 0.8, -0.2,
        0.1, -0.4, 0.9, -0.6, -0.3, 0.7, -0.8, 0.5,
        0.2, -0.9, 0.1, -0.5, 0.6, -0.2, 0.0, -0.7,
        -0.4, 0.8, -0.3, 0.4, -0.1, 0.5, -0.6, -0.0,
        0.7, -0.2, 0.3, -0.8, 0.9, -0.4, -0.5, 0.1,
        -0.6, 0.0, -0.7, 0.2, 0.5, -0.3, 0.8, -0.9,
        // bk (16)
        0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.1, -0.1,
        0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.1, -0.1,
        // wv (64)
        0.4, -0.6, 0.2, -0.8, 0.5, -0.3, 0.9, -0.1,
        -0.7, 0.1, -0.4, 0.6, -0.9, 0.0, -0.5, 0.3,
        0.8, -0.2, -0.6, 0.1, 0.4, -0.7, 0.5, -0.3,
        -0.1, 0.9, -0.5, 0.2, -0.4, 0.8, -0.6, 0.0,
        0.3, -0.8, 0.7, -0.1, -0.5, 0.6, -0.2, 0.4,
        -0.9, 0.5, -0.3, 0.1, 0.0, -0.6, 0.8, -0.7,
        0.2, -0.4, 0.9, -0.8, -0.3, 0.7, -0.1, 0.6,
        -0.5, -0.6, 0.4, -0.2, 0.3, -0.9, 0.1, -0.8,
        // bv (16)
        -0.1, 0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.1,
        -0.1, 0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.1,
        // wo (64)
        -0.3, 0.5, -0.7, 0.1, -0.9, 0.4, -0.2, 0.6,
        0.0, -0.4, 0.8, -0.6, 0.2, -0.5, 0.9, -0.1,
        0.7, -0.3, -0.1, 0.5, -0.8, 0.6, -0.4, 0.2,
        -0.9, 0.3, -0.6, 0.4, -0.7, 0.1, 0.8, -0.5,
        0.2, 0.6, -0.2, -0.8, 0.4, -0.3, -0.5, 0.7,
        -0.1, -0.9, 0.5, -0.4, 0.6, 0.0, -0.7, 0.3,
        0.8, -0.2, 0.1, -0.5, -0.6, 0.9, -0.3, -0.4,
        -0.8, 0.4, -0.9, 0.7, -0.1, 0.2, 0.5, -0.6,
        // bo (16)
        0.1, 0.0, -0.1, 0.1, 0.0, -0.1, 0.2, -0.2,
        0.1, 0.0, -0.1, 0.1, 0.0, -0.1, 0.2, -0.2,
        // wf1 (64)
        0.6, -0.3, 0.9, -0.5, 0.1, -0.8, 0.4, -0.2,
        -0.7, 0.2, -0.4, 0.8, -0.6, 0.0, -0.9, 0.5,
        0.3, -0.1, 0.5, -0.7, 0.2, -0.5, 0.8, -0.4,
        -0.6, 0.9, -0.2, 0.0, -0.3, 0.7, -0.8, 0.1,
        0.4, -0.6, 0.3, -0.9, 0.5, -0.1, -0.4, 0.7,
        -0.2, 0.8, -0.5, 0.6, -0.7, -0.3, 0.9, -0.1,
        0.0, -0.4, 0.7, -0.2, -0.8, 0.6, -0.5, 0.3,
        0.1, -0.9, -0.6, 0.4, -0.2, 0.8, -0.3, 0.5,
        // bf1 (16)
        0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1,
        0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1,
        // wf2 (64)
        -0.4, 0.7, -0.2, 0.5, -0.9, 0.3, -0.6, 0.1,
        0.8, -0.5, 0.0, -0.3, 0.6, -0.1, 0.7, -0.4,
        -0.2, 0.9, -0.8, 0.4, -0.5, 0.1, -0.7, 0.6,
        0.3, -0.6, 0.2, -0.1, 0.8, -0.4, 0.5, -0.9,
        0.1, -0.3, 0.6, -0.7, -0.2, 0.9, -0.5, 0.4,
        -0.8, 0.5, -0.1, 0.3, -0.6, 0.0, -0.4, 0.7,
        0.2, -0.9, 0.8, -0.5, 0.4, -0.7, -0.3, 0.6,
        -0.1, 0.4, -0.6, 0.9, -0.3, 0.2, -0.8, 0.5,
        // bf2 (16)
        -0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1, 0.1,
        -0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1, 0.1,
        // ln1_s (8)
        1.0, 0.9, 1.1, 0.8, 1.0, 1.2, 0.9, 1.1,
        // ln1_o (8)
        0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1,
        // ln2_s (8)
        0.9, 1.0, 1.1, 0.8, 1.2, 0.9, 1.0, 1.1,
        // ln2_o (8)
        -0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0,
        // ln3_s (8)
        1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.1, 0.9,
        // ln3_o (8)
        0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1,
        // dk_scale (4, frozen)
        0.35355339, 0.35355339, 0.35355339, 0.35355339,
        // target (16, frozen)
        1.0, 0.5, 0.0, 0.8, 1.0, 0.2, 0.7, 0.5,
        1.0, 0.8, 0.3, 0.6, 0.9, 0.2, 0.5, 0.7,
    ];
    let mut params = init_params;
    let lr = 0.01_f32;
    let eps = 1.0e-3_f32;
    let mut trace = Vec::with_capacity(steps);

    for _ in 0..steps {
        trace.push(bert_tiny_forward_loss(&params));
        // FD gradient for trainable params (indices 16..544).
        for i in 16..544 {
            let orig = params[i];
            params[i] = orig + eps;
            let loss_plus = bert_tiny_forward_loss(&params);
            params[i] = orig - eps;
            let loss_minus = bert_tiny_forward_loss(&params);
            params[i] = orig;
            let gradient = (loss_plus - loss_minus) / (2.0 * eps);
            params[i] -= lr * gradient;
        }
    }
    trace
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn compiler_generated_bert_tiny_loss_trace_matches_tape_oracle() {
    // Compute the FD oracle trace.
    const STEPS: usize = 8;
    let oracle_trace = oracle_bert_tiny_loss_trace(STEPS);

    // Run the exemplum through fmir.
    let output = run_bert_tiny_exemplum().expect("faber run should succeed for BERT-tiny exemplum");

    assert!(
        output.status.success(),
        "faber run exited with code {} — BERT-tiny forward or backward failure.\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse loss trace from nota output (8 values).
    let exemplum_trace_raw = parse_f32_values(&stdout);
    assert!(
        exemplum_trace_raw.len() >= STEPS,
        "expected ≥ {STEPS} loss values, got {}.\nstdout:\n{stdout}",
        exemplum_trace_raw.len()
    );
    let exemplum_trace = &exemplum_trace_raw[..STEPS];

    // Compare each step against FD oracle.
    for (i, (&actual, &expected)) in
        exemplum_trace.iter().zip(oracle_trace.iter()).enumerate()
    {
        let delta = (actual - expected).abs();
        assert!(
            delta <= BERT_TINY_FD_TOLERANCE,
            "step {i}: exemplum {actual} vs oracle {expected} \
             (delta {delta}, tolerance {BERT_TINY_FD_TOLERANCE})"
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
