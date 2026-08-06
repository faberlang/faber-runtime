//! GI2-2 — CPU decoder op surface for the pinned row (SmolLM2-360M-Instruct
//! Q4_K_M; `gi0-model-contract.md` v1.0.0 §2, FC2/FC7): **readable f32 CPU
//! ops** in textbook/unfused form — a correctness oracle, not a performance
//! implementation.
//!
//! Ops:
//! - [`rms_norm`] — row-wise RMSNorm (eps 1e-5, **no mean subtraction**);
//! - [`rope_norm`] — llama-arch **NORM consecutive-pair** RoPE rotation
//!   (`theta[k] = pos·freq_base^(−2k/dim)`, freq_base 100000, dim 64);
//! - [`attention_causal`] — GQA causal attention (15/5 heads × 64, scale
//!   0.125 = 1/√64, llama.cpp head-grouping, causal mask, row-wise softmax);
//! - [`swiglu`] — SiLU(gate) ⊙ up;
//! - [`dense`] — dense matmul in the GGUF/ggml K-major weight layout
//!   (`y = Wᵀ x`);
//! - [`residual_add`] — element-wise add.
//!
//! Every op consumes **dequantized f32 materializations** (GI2-1 `dequant`):
//! each such `Vec<f32>` is a CPU-oracle materialization accompanied by its
//! [`crate::dequant::OracleReceipt`]. The ops themselves are pure functions
//! over f32 slices (they do not re-derive or re-carve any model fact).
//!
//! CONTRACT — the semantics mirror the pinned comparator's arch binding
//! (local llama.cpp tree @ `a957b7747`: `src/models/llama.cpp` `build_llama`,
//! `src/llama-graph.cpp` `build_qkv` / `build_attn_mha` / `build_ffn`,
//! `ggml/src/ggml-cpu/ops.cpp` rope + rms_norm kernels):
//!
//! - **RMSNorm**: `sum = Σ x²` accumulated in f64 (ggml's `ggml_float`),
//!   `mean = sum/n` cast to f32, `scale = 1/sqrtf(mean + eps)`, then
//!   `y[i] = x[i]·scale·w[i]` (the fused rms_norm+mul order). No mean
//!   subtraction.
//! - **RoPE NORM**: each head's `dim` consecutive elements are rotated as
//!   consecutive pairs `(x[2k], x[2k+1])` with
//!   `theta[k] = pos·freq_base^(−2k/dim)` (`rotate_pairs(n_dims, n_offset=1,
//!   scale=1)` at `ggml/src/ggml-cpu/ops.cpp`; `head_dim == n_rot == 64` for
//!   the pinned row, so the whole head rotates). V is never rotated.
//! - **GQA**: Q head `h` uses KV head `h % n_kv_heads` (the ggml mul_mat
//!   broadcast replication — 15/5 → each consecutive triple of Q heads
//!   shares one KV head); `score = scale·⟨q, k⟩`; causal mask (`j ≤` the
//!   query's global position); row-wise softmax; context = Σ p·v.
//! - **SwiGLU**: `silu(gate) ⊙ up` (`ggml_swiglu_split`, LLM_FFN_SILU +
//!   LLM_FFN_PAR); `silu(x) = x/(1 + e^(−x))` (`ggml_silu`).
//! - **dense**: `y[j] = Σ_i W[i, j]·x[i]` for the GGUF K-major storage order
//!   (element `(i, j)` at `i + j·in_features`) — exactly `ggml_mul_mat(W, x)`.
//!
//! DETERMINISM — every op is plain IEEE-754 f32 (rms_norm's sum in f64) and
//! depends only on its inputs: identical inputs yield byte-identical outputs
//! on the same machine. The per-operation golden fixtures (GI3 consumption
//! contract, exit gate bullet 5) are emitted from the admitted row and depend
//! on exactly this property.
//!
//! FAIL CLOSED — shape mismatches panic with a precise message (an oracle
//! must never silently truncate or reinterpret).

use std::fmt;

// ---------------------------------------------------------------------------
// Pinned-row arch constants (gi0-model-contract.md v1.0.0 §2 — FC2)
// ---------------------------------------------------------------------------

/// Hidden size (`llama.embedding_length` = 960).
pub const HIDDEN_SIZE: usize = 960;
/// Q heads (`llama.attention.head_count` = 15).
pub const HEAD_COUNT: usize = 15;
/// KV heads (`llama.attention.head_count_kv` = 5).
pub const KV_HEAD_COUNT: usize = 5;
/// Head dim (960 / 15 = 64; `llama.rope.dimension_count`).
pub const HEAD_DIM: usize = 64;
/// FFN intermediate size (`llama.feed_forward_length` = 2560).
pub const FFN_SIZE: usize = 2560;
/// RMSNorm epsilon (`llama.attention.layer_norm_rms_epsilon` — stored F32
/// `9.999999747378752e-06`, the f32 nearest to 1e-5).
pub const RMS_EPS: f32 = 1e-5f32;
/// RoPE frequency base (`llama.rope.freq_base` = 100000.0).
pub const ROPE_FREQ_BASE: f32 = 100000.0;
/// RoPE dimension count (`llama.rope.dimension_count` = 64 == head dim).
pub const ROPE_DIM: usize = 64;
/// Attention scale = 1/√head_dim = 1/8 = 0.125 (llama.cpp derives
/// `1.0f/sqrtf(n_embd_head)`; sqrtf(64) = 8 exactly).
pub const ATTENTION_SCALE: f32 = 0.125;

/// A typed, machine-parseable op rejection (fail-closed on shape mismatch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpError {
    /// Which op rejected the call.
    pub op: &'static str,
    /// The shape constraint that was violated.
    pub detail: String,
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.op, self.detail)
    }
}

impl std::error::Error for OpError {}

/// Fail-closed shape guard shared by the ops: every op's arguments must have
/// exactly the documented lengths, otherwise a precise [`OpError`].
fn check(cond: bool, op: &'static str, detail: impl Into<String>) -> Result<(), OpError> {
    if cond {
        Ok(())
    } else {
        Err(OpError {
            op,
            detail: detail.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Row-wise RMSNorm (eps, **no mean subtraction**), matching the ggml CPU
/// kernel exactly (`ggml_compute_forward_rms_norm_f32`): the sum of squares
/// is accumulated in f64, `mean` is cast to f32, `scale = 1/sqrtf(mean+eps)`,
/// and the output is `x[i]·scale·w[i]` (the fused rms_norm+mul order — the
/// same operation order as llama.cpp `build_norm(…, LLM_NORM_RMS, …)`).
///
/// `x` and `weight` must have the same length.
///
/// # Errors
///
/// Returns a typed [`OpError`] on shape mismatch (fail-closed).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, OpError> {
    check(
        x.len() == weight.len(),
        "rms_norm",
        format!(
            "x ({}) and weight ({}) must have equal length",
            x.len(),
            weight.len()
        ),
    )?;
    let n = x.len();
    let sum: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let mean = (sum / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    Ok(x.iter()
        .zip(weight)
        .map(|(xi, wi)| xi * scale * wi)
        .collect())
}

// ---------------------------------------------------------------------------
// RoPE — llama-arch NORM consecutive-pair rotation
// ---------------------------------------------------------------------------

/// llama-arch **NORM** RoPE: rotates the `dim` consecutive elements of one
/// head as consecutive pairs `(x[2k], x[2k+1])` with
/// `theta[k] = pos·freq_base^(−2k/dim)` (`freq_base` 100000, dim 64 for the
/// pinned row — head_dim == n_rot, so the whole head rotates).
///
/// Exact kernel reference: `ggml/src/ggml-cpu/ops.cpp`
/// `ggml_rope_cache_init` + `rotate_pairs(n_dims, n_offset=1, …, scale=1)`
/// for `GGML_ROPE_TYPE_NORMAL` (`LLAMA_ROPE_TYPE_NORM`) — NOT the NEOX
/// half-split layout (model contract §2.4 correction).
///
/// `x` must have exactly `dim` elements. `pos` is the absolute token
/// position (0-based across the sequence).
///
/// # Errors
///
/// Returns a typed [`OpError`] on shape mismatch (fail-closed).
pub fn rope_norm(x: &[f32], pos: u32, freq_base: f32, dim: usize) -> Result<Vec<f32>, OpError> {
    check(
        x.len() == dim,
        "rope_norm",
        format!("x ({}) must have exactly dim ({dim}) elements", x.len()),
    )?;
    check(
        dim % 2 == 0,
        "rope_norm",
        format!("dim ({dim}) must be even"),
    )?;
    let mut out = x.to_vec();
    for k in 0..dim / 2 {
        // theta[k] = pos · freq_base^(−2k/dim) — the documented angle.
        let theta = pos as f32 * freq_base.powf(-(2.0 * k as f32) / dim as f32);
        let cos = theta.cos();
        let sin = theta.sin();
        let (x0, x1) = (out[2 * k], out[2 * k + 1]);
        out[2 * k] = x0 * cos - x1 * sin;
        out[2 * k + 1] = x0 * sin + x1 * cos;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SiLU / softmax building blocks
// ---------------------------------------------------------------------------

/// SiLU (swish): `x/(1 + e^(−x))` — `ggml_silu`'s exact formula.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

/// Row-wise softmax over `scores` (stable: subtracts the row max).
///
/// `-inf` entries are masked (probability 0) — the causal-mask convention
/// (llama.cpp `ggml_soft_max_ext` with `kq_mask` `-INFINITY` entries). A row
/// with no finite entry (fully masked) yields all zeros (documented
/// degenerate case; causal rows always have at least one unmasked key).
#[must_use]
pub fn softmax_row(scores: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for s in scores {
        if s.is_finite() && *s > max {
            max = *s;
        }
    }
    if !max.is_finite() {
        return vec![0.0; scores.len()];
    }
    let mut sum = 0.0f32;
    let mut exps = Vec::with_capacity(scores.len());
    for s in scores {
        let e = (s - max).exp();
        exps.push(e);
        sum += e;
    }
    if sum == 0.0 || !sum.is_finite() {
        return vec![0.0; scores.len()];
    }
    let inv = 1.0 / sum;
    exps.iter().map(|e| e * inv).collect()
}

// ---------------------------------------------------------------------------
// GQA causal attention
// ---------------------------------------------------------------------------

/// GQA causal attention (textbook/unfused form) for the pinned row's
/// head-grouping convention (FC7).
///
/// Layouts (row-major, contiguous — the GGUF dequant order):
/// - `q`: `n_q × (n_heads·head_dim)` — the query rows;
/// - `k`: `n_kv × (n_kv_heads·head_dim)` — the key rows (already RoPE'd by
///   the caller; V is never rotated);
/// - `v`: `n_kv × (n_kv_heads·head_dim)` — the value rows;
/// - returns `n_q × (n_heads·head_dim)` context vectors.
///
/// Causal semantics: the query at batch row `i` has global position
/// `n_kv − n_q + i` and attends to key rows `0 ..= global_position`
/// (llama.cpp's `kq_mask` convention: allowed when `j ≤ n_past + i`). With
/// `n_q == n_kv` this is the plain prompt-forward causal mask; with `n_q == 1`
/// and `n_kv > 1` it is the single-token decode case against a longer
/// context. Q head `h` uses KV head `h % n_kv_heads` (ggml's broadcast
/// replication). `score = scale·⟨q, k⟩` (scale 0.125), then row softmax over
/// the allowed keys, then `context = Σ_j p_j·v_j`.
///
/// # Errors
///
/// Returns a typed [`OpError`] on shape mismatch (fail-closed).
#[allow(clippy::too_many_arguments)]
pub fn attention_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, OpError> {
    let q_span = n_heads * head_dim;
    let kv_span = n_kv_heads * head_dim;
    check(
        n_q <= n_kv,
        "attention_causal",
        format!("n_q ({n_q}) must be ≤ n_kv ({n_kv})"),
    )?;
    check(
        n_heads % n_kv_heads == 0,
        "attention_causal",
        format!("n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"),
    )?;
    check(
        q.len() == n_q * q_span,
        "attention_causal",
        format!(
            "q ({}) must be n_q·n_heads·head_dim ({n_q}·{n_heads}·{head_dim})",
            q.len()
        ),
    )?;
    check(
        k.len() == n_kv * kv_span,
        "attention_causal",
        format!(
            "k ({}) must be n_kv·n_kv_heads·head_dim ({n_kv}·{n_kv_heads}·{head_dim})",
            k.len()
        ),
    )?;
    check(
        v.len() == n_kv * kv_span,
        "attention_causal",
        format!(
            "v ({}) must be n_kv·n_kv_heads·head_dim ({n_kv}·{n_kv_heads}·{head_dim})",
            v.len()
        ),
    )?;

    let mut ctx = vec![0.0f32; n_q * q_span];
    for i in 0..n_q {
        let global_pos = n_kv - n_q + i;
        let row_base = i * q_span;
        for h in 0..n_heads {
            let kh = h % n_kv_heads;
            let qh = &q[row_base + h * head_dim..row_base + (h + 1) * head_dim];
            // scores over the causal window 0..=global_pos
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let kj = &k[j * kv_span + kh * head_dim..j * kv_span + (kh + 1) * head_dim];
                let mut dot = 0.0f32;
                for (a, b) in qh.iter().zip(kj) {
                    dot += a * b;
                }
                scores.push(dot * scale);
            }
            let probs = softmax_row(&scores);
            let out_base = row_base + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=global_pos {
                    acc += probs[j] * v[j * kv_span + kh * head_dim + d];
                }
                ctx[out_base + d] = acc;
            }
        }
    }
    Ok(ctx)
}

// ---------------------------------------------------------------------------
// SwiGLU / FFN
// ---------------------------------------------------------------------------

/// SwiGLU gated activation: `silu(gate) ⊙ up` — `ggml_swiglu_split` for the
/// pinned row's `LLM_FFN_SILU` + `LLM_FFN_PAR` binding (FFN order: gate and
/// up projections first, then this gated multiply, then the down projection
/// via [`dense`]).
///
/// `gate` and `up` must have the same length (2560 for the pinned row).
///
/// # Errors
///
/// Returns a typed [`OpError`] on shape mismatch (fail-closed).
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, OpError> {
    check(
        gate.len() == up.len(),
        "swiglu",
        format!(
            "gate ({}) and up ({}) must have equal length",
            gate.len(),
            up.len()
        ),
    )?;
    Ok(gate.iter().zip(up).map(|(g, u)| silu(*g) * u).collect())
}

// ---------------------------------------------------------------------------
// Dense matmul
// ---------------------------------------------------------------------------

/// Dense matmul in the GGUF/ggml K-major weight layout:
/// `y[j] = Σ_i weight[i + j·in_features] · input[i]` — exactly
/// `ggml_mul_mat(W, x)` with `W` stored as `ne = (in_features, out_features)`
/// (the contiguous dim is the input dim; the dequantized GGUF tensor can be
/// sliced directly). `input` has `in_features` elements, `weight` has
/// `in_features·out_features`, the output has `out_features` elements.
///
/// # Errors
///
/// Returns a typed [`OpError`] on shape mismatch (fail-closed).
pub fn dense(
    weight: &[f32],
    input: &[f32],
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>, OpError> {
    check(
        input.len() == in_features,
        "dense",
        format!(
            "input ({}) must have in_features ({in_features}) elements",
            input.len()
        ),
    )?;
    check(
        weight.len() == in_features * out_features,
        "dense",
        format!(
            "weight ({}) must have in_features·out_features ({in_features}·{out_features}) elements",
            weight.len()
        ),
    )?;
    let mut out = vec![0.0f32; out_features];
    for j in 0..out_features {
        let col = &weight[j * in_features..(j + 1) * in_features];
        out[j] = input.iter().zip(col).map(|(a, b)| a * b).sum();
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Residual add
// ---------------------------------------------------------------------------

/// Element-wise residual add: `y[i] = a[i] + b[i]` (the llama.cpp
/// `ggml_add(cur, inpSA)` / `ggml_add(cur, ffn_inp)` residual edges).
///
/// `a` and `b` must have the same length.
///
/// # Errors
///
/// Returns a typed [`OpError`] on shape mismatch (fail-closed).
pub fn residual_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, OpError> {
    check(
        a.len() == b.len(),
        "residual_add",
        format!("a ({}) and b ({}) must have equal length", a.len(), b.len()),
    )?;
    Ok(a.iter().zip(b).map(|(x, y)| x + y).collect())
}
