//! GI2-2 — CPU decoder op surface tests.
//!
//! Families:
//! 1. **Op semantics** (always run, no model needed) — hand-computed
//!    vectors prove each op against its documented formula:
//!    - RMSNorm of a hand-computed vector (incl. no-mean-subtraction and the
//!      weight scale);
//!    - RoPE rotation of a pinned `(pos, k)` matches the documented angle
//!      `theta[k] = pos·freq_base^(−2k/dim)` (llama-arch NORM consecutive
//!      pairs; NOT the NEOX half-split);
//!    - GQA causal attention: scale, row softmax, context, the causal mask,
//!      the decode case (n_q < n_kv), and the llama.cpp head-grouping
//!      (`h % n_kv_heads`);
//!    - SwiGLU `silu(gate) ⊙ up`; dense matmul (`Wᵀ x`, GGUF K-major);
//!      residual add; SiLU; row softmax incl. masked `-inf` rows.
//! 2. **Fail-closed shape guards** — every op rejects a length mismatch with
//!    a typed [`OpError`].
//! 3. **Per-operation golden fixtures for GI3** (exit gate bullet 5) —
//!    self-contained fixtures (op id, pinned input f32 bytes from the
//!    admitted row at the pinned position, expected f32 output, provenance),
//!    hash-accounted (SHA-256 recorded), byte-stable: two independent
//!    generations are byte-identical and match the committed fixture files.
//!    Emission is env-gated (`FABER_GI22_EMIT_GOLDENS=1`); the committed
//!    fixtures + manifest live under `testdata/gi2-2-op-goldens/` (Q3
//!    default: faber-runtime crate-local testdata).
//!
//! Model-dependent tests follow the `tensor_view_test` / `dequant_test`
//! convention: skipped (with a loud note) when the pinned row is absent.

use crate::decoder_ops::*;
use crate::dequant::{dequant_block, dequant_tensor};
use crate::gguf::{admit_gguf, hex, sha256, PINNED_SHA256_HEX};
use crate::json::Json;
use crate::tensor_view::TensorView;
use crate::valor::Valor;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Fixture + pinned-row constants
// ---------------------------------------------------------------------------

/// Machine-local pinned row (skipped when absent — same convention as
/// `tensor_view_test` / `dequant_test`).
const PINNED_MODEL_PATH: &str = "/Users/ianzepp/ai/models/SmolLM2-360M-Instruct-Q4_K_M.gguf";

/// Committed op-golden fixtures (Q3 default: faber-runtime crate-local
/// testdata, `include_bytes!`-able).
const GOLDENS_REL_DIR: &str = "testdata/gi2-2-op-goldens";

/// Generator + fixture schema version (byte-stability contract).
const GENERATOR_VERSION: &str = "gi2-2-op-goldens-v1";
const FIXTURE_SCHEMA: &str = "gi2-2-op-goldens-v1";
const MANIFEST_SCHEMA: &str = "gi2-2-op-goldens-manifest-v1";

/// The pinned correctness prompt (gi0-workloads §3.1): 9 BOS-free tokens.
const PROMPT_TOKENS: [i64; 9] = [504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767];
/// The pinned token whose embedding is the decoder input at the pinned
/// position (the last prompt token — window position 0 = prompt position 8).
const PINNED_TOKEN: i64 = 2767;
/// Generation-window position the fixtures pin (0 = prompt end).
const PINNED_WINDOW_POS: u64 = 0;
/// Prompt position of [`PINNED_TOKEN`] (9 prompt tokens, 0-based).
const PINNED_PROMPT_POS: u64 = 8;

fn pinned_model_bytes() -> Option<Vec<u8>> {
    let path = std::path::Path::new(PINNED_MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: pinned model not present at {PINNED_MODEL_PATH}");
        return None;
    }
    std::fs::read(path).ok()
}

fn goldens_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDENS_REL_DIR)
}

/// Build the pinned tensor view, gated on exact coverage (GI1-4 residual:
/// dequant consumers refuse an un-covered view).
fn build_pinned_view(bytes: &[u8]) -> TensorView<'_> {
    let admission = admit_gguf(bytes).expect("pinned row must admit");
    let view = TensorView::build(&admission, bytes).expect("view must build");
    assert!(view.coverage_ok(), "the pinned view must tile exactly");
    assert_eq!(view.sha256_hex(), PINNED_SHA256_HEX, "pinned row identity");
    view
}

fn f32_le_hex(values: &[f32]) -> String {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    hex(&out)
}

fn f32_sha256_hex(values: &[f32]) -> String {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    hex(&sha256(&out))
}

/// Dequantize a bounded element range `[start, start+count)` of a tensor
/// entry through the view's raw block accessors (the range may be
/// block-misaligned; only the requested elements are kept). Used to carve
/// pinned row/column slices of the large admitted tensors (e.g. one
/// `token_embd.weight` row) without materializing the whole tensor.
fn dequant_element_range(
    view: &TensorView<'_>,
    entry: &crate::tensor_view::TensorViewEntry,
    start: u64,
    count: u64,
) -> Vec<f32> {
    let blk_elems = entry.layout.block_elements();
    assert!(
        start + count <= entry.element_count,
        "range beyond tensor elements"
    );
    let first_blk = start / blk_elems;
    let last_blk = (start + count - 1) / blk_elems;
    let mut out = Vec::with_capacity(count as usize);
    for b in first_blk..=last_blk {
        let block = view
            .raw_block(entry, b)
            .unwrap_or_else(|e| panic!("{} block {b}: {e}", entry.name));
        let vals = dequant_block(&entry.layout, block)
            .unwrap_or_else(|e| panic!("{} block {b}: {e}", entry.name));
        let blk_start = b * blk_elems;
        for (i, v) in vals.into_iter().enumerate() {
            let abs = blk_start + i as u64;
            if (start..start + count).contains(&abs) {
                out.push(v);
            }
        }
    }
    out
}

/// Apply [`rope_norm`] to each of `n_heads` contiguous `dim`-wide heads.
fn rope_all_heads(x: &[f32], n_heads: usize, pos: u32, freq_base: f32, dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), n_heads * dim);
    let mut out = Vec::with_capacity(x.len());
    for h in 0..n_heads {
        out.extend(rope_norm(&x[h * dim..(h + 1) * dim], pos, freq_base, dim).expect("rope_norm"));
    }
    out
}

// ---------------------------------------------------------------------------
// Valor walking helpers (same pattern as dequant_test)
// ---------------------------------------------------------------------------

fn text(v: &Valor) -> &str {
    let Valor::Textus(s) = v else {
        panic!("expected JSON string")
    };
    s
}

fn int(v: &Valor) -> i64 {
    let Valor::Numerus(n) = v else {
        panic!("expected JSON integer")
    };
    *n
}

fn flag(v: &Valor) -> bool {
    let Valor::Bivalens(b) = v else {
        panic!("expected JSON boolean")
    };
    *b
}

fn list<'a>(v: &'a Valor) -> &'a [Valor] {
    let Valor::Lista(items) = v else {
        panic!("expected JSON array")
    };
    items
}

fn field<'a>(v: &'a Valor, key: &str) -> &'a Valor {
    let Valor::Tabula(fields) = v else {
        panic!("expected JSON object")
    };
    fields
        .get(key)
        .unwrap_or_else(|| panic!("missing JSON field {key:?}"))
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        let bound = tol * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= bound,
            "index {i}: got {g}, expected {e} (|Δ| = {})",
            (g - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Op semantics — hand-computed vectors
// ---------------------------------------------------------------------------

#[test]
fn rms_norm_matches_hand_computed_vector() {
    // x = [1, 2, 3, 4], w = [2, 1, 0.5, 1], eps = 1e-5.
    // mean(x²) = (1+4+9+16)/4 = 7.5; scale = 1/sqrt(7.5 + 1e-5) ≈ 0.3651477.
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let w = [2.0f32, 1.0, 0.5, 1.0];
    let y = rms_norm(&x, &w, 1e-5).expect("rms_norm");
    let expected = [
        0.73029536, // 1·scale·2
        0.73029536, // 2·scale·1
        0.54772152, // 3·scale·0.5
        1.46059072, // 4·scale·1
    ];
    assert_close(&y, &expected, 1e-4);

    // Unit-weight RMSNorm of the same vector (pure 1/sqrt(mean(x²)+eps)).
    let ones = [1.0f32; 4];
    let y = rms_norm(&x, &ones, 1e-5).expect("rms_norm");
    let expected = [
        0.36514768, // 1·scale
        0.73029536, // 2·scale
        1.09544304, // 3·scale
        1.46059072, // 4·scale
    ];
    assert_close(&y, &expected, 1e-4);

    // No mean subtraction: y must equal x·scale·w — NOT (x − mean(x))·… .
    // A mean-subtracting norm would produce x' = [−1.5, −0.5, 0.5, 1.5]
    // scaled by sqrt(mean(x'²)) = sqrt(1.25); the literal above pins the
    // RMS form.
    assert!(
        (y[0] + y[1] + y[2] + y[3]) > 3.0,
        "signs must follow x (no mean shift)"
    );
}

#[test]
fn rms_norm_is_dimension_agnostic_and_deterministic() {
    let x: Vec<f32> = (0..960).map(|i| (i as f32 - 480.0) / 971.0).collect();
    let w: Vec<f32> = (0..960).map(|i| 0.5 + 0.001 * i as f32 % 1.0).collect();
    let a = rms_norm(&x, &w, RMS_EPS).expect("rms_norm");
    let b = rms_norm(&x, &w, RMS_EPS).expect("rms_norm");
    assert_eq!(a.len(), 960);
    // Byte-determinism: identical inputs → identical f32 outputs.
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.to_bits(), y.to_bits(), "deterministic f32 bits");
    }
}

#[test]
fn silu_matches_hand_computed_values() {
    assert_eq!(silu(0.0), 0.0);
    // silu(1) = 1·sigmoid(1) = 1/(1+e^-1) ≈ 0.73105858
    assert!((silu(1.0) - 0.7310586).abs() < 1e-6);
    // silu(-1) = -1·sigmoid(-1) ≈ -0.26894142
    assert!((silu(-1.0) - (-0.2689414)).abs() < 1e-6);
    // asymptotics: silu(x) → x for large x, → 0 for large −x
    assert!((silu(20.0) - 20.0).abs() < 1e-3);
    assert!(silu(-20.0).abs() < 1e-6);
}

#[test]
fn rope_norm_rotation_matches_documented_angle() {
    // Pinned (pos = 8, k = 0): theta_0 = 8·100000^(−0/64) = 8 rad.
    // Rotating the unit vector e0 = [1, 0] gives [cos(8), sin(8)].
    let mut x = vec![0.0f32; 64];
    x[0] = 1.0;
    let y = rope_norm(&x, 8, ROPE_FREQ_BASE, ROPE_DIM).expect("rope_norm");
    let cos8 = -0.14550003380861354f32;
    let sin8 = 0.9893582466233818f32;
    assert!(
        (y[0] - cos8).abs() < 1e-5,
        "pair 0 cos: got {}, want {}",
        y[0],
        cos8
    );
    assert!(
        (y[1] - sin8).abs() < 1e-5,
        "pair 0 sin: got {}, want {}",
        y[1],
        sin8
    );
    // All other pairs are untouched for the k=0 unit vector.
    for i in 2..64 {
        assert_eq!(y[i], 0.0, "pair 0 rotation must not touch element {i}");
    }

    // Pinned (pos = 8, k = 1): theta_1 = 8·100000^(−2/64). Rotating e1 =
    // [0, 0, 1, 0] gives [cos(theta_1), sin(theta_1)] at pair 1.
    let mut x = vec![0.0f32; 64];
    x[2] = 1.0;
    let y = rope_norm(&x, 8, ROPE_FREQ_BASE, ROPE_DIM).expect("rope_norm");
    let theta1 = 8.0f32 * ROPE_FREQ_BASE.powf(-2.0 / ROPE_DIM as f32);
    assert!(
        (y[2] - theta1.cos()).abs() < 1e-5,
        "pair 1 cos matches the documented angle"
    );
    assert!(
        (y[3] - theta1.sin()).abs() < 1e-5,
        "pair 1 sin matches the documented angle"
    );

    // pos = 0 is the identity rotation (cos 0 = 1, sin 0 = 0).
    let x: Vec<f32> = (0..64).map(|i| 0.25 * (i as f32 + 1.0)).collect();
    let y = rope_norm(&x, 0, ROPE_FREQ_BASE, ROPE_DIM).expect("rope_norm");
    assert_eq!(x, y, "pos 0 must be the identity rotation");

    // The rotation preserves the vector norm (it is a per-pair Givens
    // rotation).
    let y = rope_norm(&x, 8, ROPE_FREQ_BASE, ROPE_DIM).expect("rope_norm");
    let nx: f32 = x.iter().map(|v| v * v).sum();
    let ny: f32 = y.iter().map(|v| v * v).sum();
    assert!((nx - ny).abs() < 1e-3, "rotation must preserve the norm");
}

#[test]
fn rope_norm_uses_consecutive_pairs_not_half_split() {
    // The llama-arch NORM layout rotates (x[2k], x[2k+1]) — NOT the NEOX
    // half-split (x[k], x[k+32]). With a unit vector at x[0], the NORM
    // rotation moves energy into x[1] only; a half-split rotation would
    // move it into x[32] instead.
    let mut x = vec![0.0f32; 64];
    x[0] = 1.0;
    let y = rope_norm(&x, 8, ROPE_FREQ_BASE, ROPE_DIM).expect("rope_norm");
    assert!(
        y[1].abs() > 0.9,
        "consecutive partner must receive the energy"
    );
    assert_eq!(
        y[32], 0.0,
        "half-split partner must stay untouched (NORM layout)"
    );
}

#[test]
fn softmax_row_matches_hand_computed_values() {
    // scores [1, 2, 3]: exp(1−3)=e⁻², exp(2−3)=e⁻¹, exp(3−3)=1 over sum.
    let p = softmax_row(&[1.0, 2.0, 3.0]);
    let sum = 1.0f32 + (-1.0f32).exp() + (-2.0f32).exp();
    assert_close(
        &p,
        &[(-2.0f32).exp() / sum, (-1.0f32).exp() / sum, 1.0 / sum],
        1e-6,
    );
    let total: f32 = p.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "rows sum to 1");

    // Masked (-inf) entries get probability 0.
    let p = softmax_row(&[1.0, f32::NEG_INFINITY, 2.0]);
    let sum = 1.0f32 + (-1.0f32).exp();
    assert_close(&p, &[(-1.0f32).exp() / sum, 0.0, 1.0 / sum], 1e-6);

    // Fully masked row → zeros (documented degenerate case).
    let p = softmax_row(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    assert_eq!(p, vec![0.0, 0.0]);

    // Single element → 1.0.
    assert_eq!(softmax_row(&[3.5]), vec![1.0]);
}

#[test]
fn attention_causal_matches_hand_computed_pinned_row() {
    // 2 Q heads, 1 KV head (GQA 2:1), head_dim 2, 2 tokens, scale 0.5.
    let n_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 2usize;
    let scale = 0.5f32;
    // q rows: [head0, head1] per token.
    let q = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    // k rows (single kv head): token0 [1,1], token1 [3,3].
    let k = [1.0, 1.0, 3.0, 3.0];
    // v rows: token0 [10,20], token1 [30,40].
    let v = [10.0, 20.0, 30.0, 40.0];

    let ctx = attention_causal(&q, &k, &v, 2, 2, n_heads, n_kv_heads, head_dim, scale)
        .expect("attention_causal");

    // Token 0 (global pos 0): single key → context = v[0] for both heads.
    assert_eq!(&ctx[0..4], &[10.0, 20.0, 10.0, 20.0]);

    // Token 1 (global pos 1): both heads use kv head 0.
    // h0: q=[2,0]: scores 0.5·(2·1)=1 and 0.5·(2·3)=3 → softmax [e¹,e³].
    // h1: q=[0,2]: scores 0.5·(2·1)=1 and 0.5·(2·3)=3 → same probs.
    let p0_exact = 1.0 / (1.0 + (2.0f32).exp());
    let p1_exact = (2.0f32).exp() / (1.0 + (2.0f32).exp());
    assert!(
        (p0_exact + p1_exact - 1.0).abs() < 1e-6,
        "softmax normalizes"
    );
    let c0 = p0_exact * 10.0 + p1_exact * 30.0;
    let c1 = p0_exact * 20.0 + p1_exact * 40.0;
    assert_close(&ctx[4..8], &[c0, c1, c0, c1], 1e-5);
}

#[test]
fn attention_causal_masks_future_keys() {
    // 1 head, 1 kv head, head_dim 1, 2 tokens, scale 1.
    let ctx = attention_causal(
        &[1.0, 1.0],
        &[1.0, 1000.0],
        &[7.0, 999.0],
        2,
        2,
        1,
        1,
        1,
        1.0,
    )
    .expect("attention_causal");
    // Row 0 attends only to key 0 (causal) → context exactly v[0].
    assert_eq!(ctx[0], 7.0, "first row must not see the future key");
    // Row 1 attends to both keys; the 1000·1 score dominates → ≈ v[1].
    assert!((ctx[1] - 999.0).abs() < 1e-3, "row 1 blends both keys");

    // Decode case: n_q = 1, n_kv = 2 → the single query sits at global
    // position 1 and attends to both keys.
    let ctx = attention_causal(&[1.0], &[1.0, 1000.0], &[7.0, 999.0], 1, 2, 1, 1, 1, 1.0)
        .expect("attention_causal");
    assert!(
        (ctx[0] - 999.0).abs() < 1e-3,
        "decode row attends to the full context"
    );
}

#[test]
fn attention_causal_uses_llama_cpp_head_grouping() {
    // 2 Q heads, 2 KV heads, head_dim 2, 1 query token, 2 keys, scale 1.
    // Head h uses KV head h % 2: head0 → kv0, head1 → kv1.
    let q = [1.0, 0.0, 0.0, 1.0];
    let k = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    let v = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    let ctx = attention_causal(&q, &k, &v, 1, 2, 2, 2, 2, 1.0).expect("attention_causal");
    // head0: kv0 scores [1, 2] → probs [e⁻¹, 1]/(1+e⁻¹) → ctx ≈ 1.731; head1: kv1 → same.
    let s = 1.0f32 + (-1.0f32).exp();
    let c0 = ((-1.0f32).exp() + 2.0 * 1.0) / s;
    assert_close(&ctx, &[c0, 0.0, 0.0, c0], 1e-5);
}

#[test]
fn swiglu_matches_hand_computed_values() {
    // gate [1, 2] ⊙ up [3, 4]: [silu(1)·3, silu(2)·4].
    let out = swiglu(&[1.0, 2.0], &[3.0, 4.0]).expect("swiglu");
    let silu1 = 0.73105858f32; // 1/(1+e^-1)
    let silu2 = 1.76159416f32; // 2/(1+e^-2)
    assert_close(&out, &[silu1 * 3.0, silu2 * 4.0], 1e-5);
    // Length 2560 smoke (the pinned FFN size) stays deterministic.
    let gate: Vec<f32> = (0..2560).map(|i| 0.01 * (i as f32 % 97.0)).collect();
    let up: Vec<f32> = (0..2560).map(|i| 0.02 * (i as f32 % 61.0)).collect();
    let a = swiglu(&gate, &up).expect("swiglu");
    let b = swiglu(&gate, &up).expect("swiglu");
    assert_eq!(a.len(), 2560);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.to_bits(), y.to_bits(), "deterministic f32 bits");
    }
}

#[test]
fn dense_matches_hand_computed_matmul() {
    // K-major W = [[1, 3], [2, 4]] (W[i,j] at i + 2j), x = [5, 6]:
    // y = [5·1 + 6·2, 5·3 + 6·4] = [17, 39].
    let w = [1.0, 2.0, 3.0, 4.0];
    let y = dense(&w, &[5.0, 6.0], 2, 2).expect("dense");
    assert_eq!(y, vec![17.0, 39.0]);

    // 960 → 64 slice shape (attn_q head-0 style) stays deterministic.
    let x: Vec<f32> = (0..960).map(|i| 0.01 * (i as f32 % 83.0)).collect();
    let w: Vec<f32> = (0..960 * 64).map(|i| 0.005 * (i as f32 % 211.0)).collect();
    let a = dense(&w, &x, 960, 64).expect("dense");
    let b = dense(&w, &x, 960, 64).expect("dense");
    assert_eq!(a.len(), 64);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.to_bits(), y.to_bits(), "deterministic f32 bits");
    }
}

#[test]
fn residual_add_matches_hand_computed_values() {
    let y = residual_add(&[1.0, 2.0], &[10.0, 20.0]).expect("residual_add");
    assert_eq!(y, vec![11.0, 22.0]);
}

// ---------------------------------------------------------------------------
// 2. Fail-closed shape guards
// ---------------------------------------------------------------------------

#[test]
fn ops_fail_closed_on_shape_mismatch() {
    assert!(rms_norm(&[1.0, 2.0], &[1.0], 1e-5).is_err());
    assert!(rope_norm(&[1.0, 2.0], 8, 100000.0, 4).is_err()); // x.len() != dim
    assert!(rope_norm(&[1.0, 2.0, 3.0], 8, 100000.0, 2).is_err());
    assert!(swiglu(&[1.0], &[1.0, 2.0]).is_err());
    assert!(residual_add(&[1.0], &[1.0, 2.0]).is_err());
    let err = dense(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2, 2).expect_err("short weight");
    assert_eq!(err.op, "dense");

    // attention_causal: mismatched q/k/v lengths, n_q > n_kv, non-GQA heads.
    let err = attention_causal(&[0.0; 3], &[0.0; 2], &[0.0; 2], 1, 1, 1, 1, 1, 1.0)
        .expect_err("q length");
    assert_eq!(err.op, "attention_causal");
    let err = attention_causal(&[0.0; 2], &[0.0; 2], &[0.0; 2], 2, 1, 1, 1, 1, 1.0)
        .expect_err("n_q > n_kv");
    assert!(err.detail.contains("n_q"));
    let err = attention_causal(&[0.0; 2], &[0.0; 2], &[0.0; 2], 1, 1, 2, 3, 1, 1.0)
        .expect_err("GQA ratio");
    assert!(err.detail.contains("multiple"));
}

// ---------------------------------------------------------------------------
// 3. Per-operation golden fixtures for GI3 (exit gate bullet 5)
// ---------------------------------------------------------------------------

/// One self-contained op golden: op id, pinned input vectors (name + source
/// provenance), expected output, and op-specific metadata.
struct OpFixture {
    op: &'static str,
    inputs: Vec<(&'static str, &'static str, Vec<f32>)>,
    output: (&'static str, Vec<f32>),
    extra: Vec<(&'static str, Valor)>,
}

/// Build the model-provenance + pinned-position object every fixture shares.
fn fixture_provenance(view: &TensorView<'_>) -> BTreeMap<String, Valor> {
    let mut model = BTreeMap::new();
    model.insert("sha256".to_string(), Valor::from(view.sha256_hex()));
    model.insert("bytes".to_string(), Valor::from(view.file_size() as i64));
    model.insert("path".to_string(), Valor::from(PINNED_MODEL_PATH));
    let mut provenance = BTreeMap::new();
    provenance.insert("schema".to_string(), Valor::from(FIXTURE_SCHEMA));
    provenance.insert("generator".to_string(), Valor::from(GENERATOR_VERSION));
    provenance.insert("model".to_string(), model.into());
    provenance.insert("coverage_ok".to_string(), Valor::from(view.coverage_ok()));
    let mut pinned = BTreeMap::new();
    pinned.insert("window".to_string(), Valor::from(PINNED_WINDOW_POS as i64));
    pinned.insert("prompt".to_string(), Valor::from(PINNED_PROMPT_POS as i64));
    pinned.insert("token".to_string(), Valor::from(PINNED_TOKEN));
    provenance.insert("pinned_position".to_string(), pinned.into());
    provenance
}

/// Serialize one fixture to its byte-stable JSON wire form.
fn fixture_to_json(fx: &OpFixture, view: &TensorView<'_>) -> String {
    let mut root = fixture_provenance(view);
    root.insert("op".to_string(), Valor::from(fx.op));
    for (key, value) in &fx.extra {
        root.insert((*key).to_string(), value.clone());
    }
    let inputs: Vec<Valor> = fx
        .inputs
        .iter()
        .map(|(name, source, values)| {
            let mut m = BTreeMap::new();
            m.insert("name".to_string(), Valor::from(*name));
            m.insert("source".to_string(), Valor::from(*source));
            m.insert("elements".to_string(), Valor::from(values.len() as i64));
            m.insert("f32_le_hex".to_string(), Valor::from(f32_le_hex(values)));
            m.into()
        })
        .collect();
    root.insert("inputs".to_string(), inputs.into());
    let (name, values) = &fx.output;
    let mut out = BTreeMap::new();
    out.insert("name".to_string(), Valor::from(*name));
    out.insert("elements".to_string(), Valor::from(values.len() as i64));
    out.insert("sha256".to_string(), Valor::from(f32_sha256_hex(values)));
    out.insert("f32_le_hex".to_string(), Valor::from(f32_le_hex(values)));
    root.insert("expected_output".to_string(), out.into());
    let json = Json::from_object(root).expect("fixture JSON is valid");
    format!("{}\n", json.to_wire())
}

/// Emit all six per-op goldens from the admitted row at the pinned position
/// (window 0 = prompt position 8, token 2767). Every input is pinned f32
/// bytes derived from real admitted-row tensors (documented in `source`).
fn generate_all_fixtures(view: &TensorView<'_>) -> Vec<OpFixture> {
    let embd = view.tensor("token_embd.weight").expect("token_embd.weight");
    let norm_w0 = dequant_tensor(
        view,
        view.tensor("blk.0.attn_norm.weight").expect("attn_norm"),
    )
    .expect("attn_norm dequant");
    let x0 = dequant_element_range(
        view,
        embd,
        PINNED_TOKEN as u64 * HIDDEN_SIZE as u64,
        HIDDEN_SIZE as u64,
    );
    assert_eq!(x0.len(), HIDDEN_SIZE, "embedding row must be 960 wide");
    let a0 = rms_norm(&x0, &norm_w0, RMS_EPS).expect("rms_norm");

    // --- dense golden: attn_q head-0 output slice (columns 0..64) ---------
    let attn_q_entry = view.tensor("blk.0.attn_q.weight").expect("attn_q.weight");
    let q_w_head0 = dequant_element_range(view, attn_q_entry, 0, (HEAD_DIM * HIDDEN_SIZE) as u64);
    let q_head0 = dense(&q_w_head0, &a0, HIDDEN_SIZE, HEAD_DIM).expect("dense");

    // --- rope golden: head-0 Q projection at the pinned position ----------
    let q_head0_rot =
        rope_norm(&q_head0, PINNED_PROMPT_POS as u32, ROPE_FREQ_BASE, ROPE_DIM).expect("rope_norm");

    // --- attention golden: post-rope Q/K/V over the 9 prompt positions -----
    let attn_q_full = dequant_tensor(view, attn_q_entry).expect("attn_q dequant");
    let attn_k_full = dequant_tensor(view, view.tensor("blk.0.attn_k.weight").expect("attn_k"))
        .expect("attn_k dequant");
    let attn_v_full = dequant_tensor(view, view.tensor("blk.0.attn_v.weight").expect("attn_v"))
        .expect("attn_v dequant");
    let mut k_rot = Vec::with_capacity(9 * KV_HEAD_COUNT * HEAD_DIM);
    let mut v_all = Vec::with_capacity(9 * KV_HEAD_COUNT * HEAD_DIM);
    let mut q8_rot = Vec::with_capacity(HIDDEN_SIZE);
    for (t, &tok) in PROMPT_TOKENS.iter().enumerate() {
        let emb_t = dequant_element_range(
            view,
            embd,
            tok as u64 * HIDDEN_SIZE as u64,
            HIDDEN_SIZE as u64,
        );
        let a_t = rms_norm(&emb_t, &norm_w0, RMS_EPS).expect("rms_norm");
        let q_t = dense(&attn_q_full, &a_t, HIDDEN_SIZE, HIDDEN_SIZE).expect("dense");
        let k_t = dense(&attn_k_full, &a_t, HIDDEN_SIZE, KV_HEAD_COUNT * HEAD_DIM).expect("dense");
        let v_t = dense(&attn_v_full, &a_t, HIDDEN_SIZE, KV_HEAD_COUNT * HEAD_DIM).expect("dense");
        if t == PINNED_PROMPT_POS as usize {
            q8_rot = rope_all_heads(
                &q_t,
                HEAD_COUNT,
                PINNED_PROMPT_POS as u32,
                ROPE_FREQ_BASE,
                HEAD_DIM,
            );
        }
        k_rot.extend(rope_all_heads(
            &k_t,
            KV_HEAD_COUNT,
            t as u32,
            ROPE_FREQ_BASE,
            HEAD_DIM,
        ));
        v_all.extend(v_t);
    }
    assert_eq!(q8_rot.len(), HIDDEN_SIZE);
    assert_eq!(k_rot.len(), 9 * KV_HEAD_COUNT * HEAD_DIM);
    assert_eq!(v_all.len(), 9 * KV_HEAD_COUNT * HEAD_DIM);
    let context8 = attention_causal(
        &q8_rot,
        &k_rot,
        &v_all,
        1,
        PROMPT_TOKENS.len(),
        HEAD_COUNT,
        KV_HEAD_COUNT,
        HEAD_DIM,
        ATTENTION_SCALE,
    )
    .expect("attention_causal");
    assert_eq!(context8.len(), HIDDEN_SIZE);

    // --- residual golden: layer-0 output projection + add -------------------
    let attn_out_full = dequant_tensor(
        view,
        view.tensor("blk.0.attn_output.weight")
            .expect("attn_output"),
    )
    .expect("attn_output dequant");
    let attn_out8 = dense(&attn_out_full, &context8, HIDDEN_SIZE, HIDDEN_SIZE).expect("dense");
    let ffn_inp8 = residual_add(&x0, &attn_out8).expect("residual_add");

    // --- swiglu golden: ffn gate/up projections at the pinned position ------
    let ffn_norm_w0 = dequant_tensor(
        view,
        view.tensor("blk.0.ffn_norm.weight").expect("ffn_norm"),
    )
    .expect("ffn_norm dequant");
    let f = rms_norm(&ffn_inp8, &ffn_norm_w0, RMS_EPS).expect("rms_norm");
    let ffn_gate_full = dequant_tensor(
        view,
        view.tensor("blk.0.ffn_gate.weight").expect("ffn_gate"),
    )
    .expect("ffn_gate dequant");
    let ffn_up_full = dequant_tensor(view, view.tensor("blk.0.ffn_up.weight").expect("ffn_up"))
        .expect("ffn_up dequant");
    let gate8 = dense(&ffn_gate_full, &f, HIDDEN_SIZE, FFN_SIZE).expect("dense");
    let up8 = dense(&ffn_up_full, &f, HIDDEN_SIZE, FFN_SIZE).expect("dense");
    let h8 = swiglu(&gate8, &up8).expect("swiglu");

    vec![
        OpFixture {
            op: "rms_norm",
            inputs: vec![
                (
                    "x",
                    "token_embd.weight[2767, 0..960) (Q8_0 row slice) — decoder input at the pinned position",
                    x0.clone(),
                ),
                (
                    "weight",
                    "blk.0.attn_norm.weight (F32, dequantized)",
                    norm_w0.clone(),
                ),
            ],
            output: ("y", a0.clone()),
            extra: vec![(
                "eps",
                Valor::from("1e-5 (stored F32 9.999999747378752e-06)"),
            )],
        },
        OpFixture {
            op: "dense",
            inputs: vec![
                (
                    "weight",
                    "blk.0.attn_q.weight[:, 0..64) (Q5_0, GGUF K-major, head-0 output slice)",
                    q_w_head0,
                ),
                (
                    "input",
                    "rms_norm golden output — normed activation at the pinned position",
                    a0.clone(),
                ),
            ],
            output: ("y", q_head0.clone()),
            extra: vec![
                ("in_features", Valor::from(HIDDEN_SIZE as i64)),
                ("out_features", Valor::from(HEAD_DIM as i64)),
            ],
        },
        OpFixture {
            op: "rope",
            inputs: vec![(
                "head",
                "dense golden output — head-0 Q projection at the pinned position",
                q_head0,
            )],
            output: ("rotated", q_head0_rot),
            extra: vec![
                ("pos", Valor::from(PINNED_PROMPT_POS as i64)),
                ("freq_base", Valor::from(ROPE_FREQ_BASE as f64)),
                ("dim", Valor::from(ROPE_DIM as i64)),
            ],
        },
        OpFixture {
            op: "attention",
            inputs: vec![
                (
                    "q",
                    "rope_norm(dense(blk.0.attn_q.weight, a_t), pos=8) per head — post-rope query at the pinned position",
                    q8_rot,
                ),
                (
                    "k",
                    "rope_norm(dense(blk.0.attn_k.weight, a_t), pos=t) per head for prompt positions 0..8",
                    k_rot,
                ),
                (
                    "v",
                    "dense(blk.0.attn_v.weight, a_t) for prompt positions 0..8 (never rotated)",
                    v_all,
                ),
            ],
            output: ("context", context8),
            extra: vec![
                ("n_q", Valor::from(1i64)),
                ("n_kv", Valor::from(PROMPT_TOKENS.len() as i64)),
                ("n_heads", Valor::from(HEAD_COUNT as i64)),
                ("n_kv_heads", Valor::from(KV_HEAD_COUNT as i64)),
                ("head_dim", Valor::from(HEAD_DIM as i64)),
                ("scale", Valor::from(ATTENTION_SCALE as f64)),
            ],
        },
        OpFixture {
            op: "residual",
            inputs: vec![
                (
                    "a",
                    "token_embd.weight[2767, 0..960) — the pinned-position layer input",
                    x0,
                ),
                (
                    "b",
                    "dense(blk.0.attn_output.weight, attention golden context)",
                    attn_out8,
                ),
            ],
            output: ("y", ffn_inp8),
            extra: vec![],
        },
        OpFixture {
            op: "swiglu",
            inputs: vec![
                (
                    "gate",
                    "dense(blk.0.ffn_gate.weight, rms_norm(ffn_inp8, blk.0.ffn_norm.weight))",
                    gate8,
                ),
                (
                    "up",
                    "dense(blk.0.ffn_up.weight, rms_norm(ffn_inp8, blk.0.ffn_norm.weight))",
                    up8,
                ),
            ],
            output: ("y", h8),
            extra: vec![("in_features", Valor::from(FFN_SIZE as i64))],
        },
    ]
}

/// The committed manifest: schema/generator/model provenance + per-op
/// hash-accounting (SHA-256 of each fixture file).
fn build_manifest(view: &TensorView<'_>, fixtures: &[OpFixture]) -> String {
    let mut root = fixture_provenance(view);
    root.insert("schema".to_string(), Valor::from(MANIFEST_SCHEMA));
    let mut entries: Vec<Valor> = Vec::new();
    for fx in fixtures {
        let wire = fixture_to_json(fx, view);
        let mut m = BTreeMap::new();
        m.insert("op".to_string(), Valor::from(fx.op));
        m.insert("file".to_string(), Valor::from(format!("{}.json", fx.op)));
        m.insert(
            "sha256".to_string(),
            Valor::from(hex(&sha256(wire.as_bytes()))),
        );
        m.insert(
            "output_elements".to_string(),
            Valor::from(fx.output.1.len() as i64),
        );
        entries.push(m.into());
    }
    root.insert("fixtures".to_string(), entries.into());
    let json = Json::from_object(root).expect("manifest JSON is valid");
    format!("{}\n", json.to_wire())
}

fn all_committed_wires(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let manifest = std::fs::read_to_string(dir.join("manifest.json"))
        .unwrap_or_else(|_e| panic!("committed manifest must exist ({})", dir.display()));
    let json = Json::parse(&manifest).expect("manifest must parse");
    let mut out = Vec::new();
    for item in list(field(json.as_valor(), "fixtures")) {
        let file = text(field(item, "file")).to_string();
        let bytes = std::fs::read(dir.join(&file))
            .unwrap_or_else(|e| panic!("committed fixture {file}: {e}"));
        out.push((file, bytes));
    }
    out
}

/// One-time emission of the committed fixtures (env-gated). Run with
/// `FABER_GI22_EMIT_GOLDENS=1` to (re)generate the fixture files + manifest
/// under `testdata/gi2-2-op-goldens/`. This is setup evidence — the
/// committed files are the artifacts the determinism test compares against.
#[test]
fn emit_op_goldens_when_requested() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    if std::env::var_os("FABER_GI22_EMIT_GOLDENS").is_none() {
        eprintln!("set FABER_GI22_EMIT_GOLDENS=1 to (re)emit the committed goldens");
        return;
    }
    let view = build_pinned_view(&bytes);
    let fixtures = generate_all_fixtures(&view);
    let dir = goldens_dir();
    std::fs::create_dir_all(&dir).expect("create goldens dir");
    for fx in &fixtures {
        let wire = fixture_to_json(fx, &view);
        std::fs::write(dir.join(format!("{}.json", fx.op)), wire)
            .unwrap_or_else(|e| panic!("write {}: {e}", fx.op));
    }
    std::fs::write(dir.join("manifest.json"), build_manifest(&view, &fixtures))
        .expect("write manifest");
    eprintln!("wrote GI2-2 op goldens to {}", dir.display());
}

/// The GI3 consumption contract: per-op fixtures are emitted from the
/// admitted row, **byte-identical across two independent runs**, hash-
/// accounted (manifest SHA-256), and byte-identical to the committed files.
#[test]
fn op_goldens_are_byte_stable_deterministic_and_hash_accounted() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_pinned_view(&bytes);

    // Two independent generations are byte-identical (determinism).
    let first = generate_all_fixtures(&view);
    let second = generate_all_fixtures(&view);
    let wires_a: Vec<String> = first.iter().map(|fx| fixture_to_json(fx, &view)).collect();
    let wires_b: Vec<String> = second.iter().map(|fx| fixture_to_json(fx, &view)).collect();
    assert_eq!(wires_a, wires_b, "fixture generation must be deterministic");
    assert_eq!(first.len(), 6, "exactly six per-op fixtures");

    // Committed fixtures match regeneration byte-for-byte and the manifest
    // hash-accounts them.
    let dir = goldens_dir();
    let committed = all_committed_wires(&dir);
    assert_eq!(committed.len(), 6, "manifest must list all six fixtures");
    for (fx, wire) in first.iter().zip(&wires_a) {
        let file = format!("{}.json", fx.op);
        let committed_bytes = committed
            .iter()
            .find(|(name, _)| name == &file)
            .unwrap_or_else(|| panic!("{file} must be listed in the manifest"))
            .1
            .clone();
        assert_eq!(
            committed_bytes,
            wire.as_bytes(),
            "{} fixture must be byte-identical to the committed file",
            fx.op
        );
        // Hash accounting: recorded manifest SHA-256 == actual file digest.
        let manifest =
            Json::parse(&std::fs::read_to_string(dir.join("manifest.json")).expect("manifest"))
                .expect("manifest parse");
        let recorded = list(field(manifest.as_valor(), "fixtures"))
            .iter()
            .find_map(|entry| {
                let entry_file = text(field(entry, "file"));
                (entry_file == file).then(|| text(field(entry, "sha256")))
            })
            .unwrap_or_else(|| panic!("{file} manifest entry"));
        assert_eq!(
            recorded,
            hex(&sha256(wire.as_bytes())),
            "{file} manifest hash must account the fixture bytes"
        );
    }

    // Provenance: the fixture model facts match the live pinned view.
    for fx in &first {
        let wire = fixture_to_json(fx, &view);
        let json = Json::parse(&wire).expect("fixture parses");
        let root = json.as_valor();
        let model = field(root, "model");
        assert_eq!(
            text(field(model, "sha256")),
            PINNED_SHA256_HEX,
            "{} provenance model SHA",
            fx.op
        );
        assert_eq!(
            int(field(model, "bytes")),
            view.file_size() as i64,
            "{} provenance model bytes",
            fx.op
        );
        assert_eq!(
            flag(field(root, "coverage_ok")),
            true,
            "{} provenance coverage",
            fx.op
        );
        // Expected output f32 byte count matches the op's declared width.
        let out = field(root, "expected_output");
        assert_eq!(
            int(field(out, "elements")),
            fx.output.1.len() as i64,
            "{} output elements",
            fx.op
        );
    }
}

/// The committed manifest is always present and lists exactly the six ops
/// (runs even when the pinned model is absent).
#[test]
fn committed_manifest_lists_all_six_ops() {
    let dir = goldens_dir();
    let wire = std::fs::read_to_string(dir.join("manifest.json"))
        .unwrap_or_else(|e| panic!("committed manifest must exist at {} ({e})", dir.display()));
    let json = Json::parse(&wire).expect("manifest must parse");
    let root = json.as_valor();
    assert_eq!(
        text(field(root, "schema")),
        MANIFEST_SCHEMA,
        "manifest schema"
    );
    assert_eq!(
        text(field(root, "generator")),
        GENERATOR_VERSION,
        "manifest generator"
    );
    let mut ops: Vec<&str> = list(field(root, "fixtures"))
        .iter()
        .map(|entry| text(field(entry, "op")))
        .collect();
    ops.sort_unstable();
    let expected = [
        "attention",
        "dense",
        "residual",
        "rms_norm",
        "rope",
        "swiglu",
    ];
    assert_eq!(
        ops, expected,
        "manifest must list exactly the six per-op fixtures"
    );
}
