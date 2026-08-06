//! GI2-3 — CPU one-position logits oracle + teacher-forced numeric-contract
//! window (pinned row: SmolLM2-360M-Instruct Q4_K_M; `gi0-model-contract.md`
//! v1.0.0 §2, FC2/FC7).
//!
//! The oracle is the smallest correct forward path from pinned token ids to
//! **full-vocab raw logits for one position** (the next-token distribution
//! after a token sequence), using readable CPU operations and the admitted
//! GGUF tensors: GI1-1 admission + GI1-2 `QuantizedTensorLayout` + GI1-4
//! `TensorView` + GI2-1 dequant + GI2-2 op surface. It is a correctness
//! oracle — **not** a performance implementation, and never a GPU/device
//! path.
//!
//! FORWARD PATH (mirrors llama.cpp `build_llama` at the pinned checkout
//! `a957b7747`, `src/models/llama.cpp`; cf. `gi2-delivery.md` §GI2-3):
//!
//! ```text
//! token_embd gather (Q8_0 [960, 49152]; TIED — the same weight is the
//!   output head; no `output.weight` exists)                          → h [n, 960]
//! ×32 layers:
//!   attn_norm (RMSNorm eps 1e-5) → QKV dense (Q [960,960] → 15×64,
//!     K/V [960,320] → 5×64) → RoPE llama-arch NORM consecutive-pair
//!     (freq_base 100000, dim 64) → causal attention (GQA 15/5, scale 0.125,
//!     head-grouping h_q → h_q / (n_head/n_head_kv)) → attn_output dense →
//!     residual → ffn_norm (RMSNorm) → SwiGLU (silu(gate)·up, 2560) →
//!     ffn_down dense → residual
//! output_norm (RMSNorm F32, eps 1e-5) → final dense via the tied
//!   token_embd.weightᵀ → full-vocab raw logits [49,152]
//! ```
//!
//! Public surface: [`CpuOracle::forward_one`] — raw logits for the position
//! after `tokens` — plus the numeric-faithful incremental runner
//! [`ForwardRun`] that produces **byte-identical** logits while caching the
//! per-layer K/V, so the teacher-forced 17-window harness runs without
//! re-computing the whole context at every position (the durable KV cache is
//! GI4's; this cache changes no numerics — proven by test).
//!
//! COMPARATOR-FIDELITY FACTS (required oracle behavior, not tuning knobs):
//!
//! 1. **BOS-free encode** (`add_bos_token=false`, model contract §5): the
//!    oracle evaluates the given token ids exactly — no implicit BOS.
//! 2. **RoPE llama-arch NORM consecutive-pair** rotation (model contract
//!    §2.4 correction — NOT the NEOX half-split).
//! 3. **KV-cache f16 rounding**: the pinned comparator runs
//!    `--cache-type-k f16 --cache-type-v f16` and its attention always
//!    consumes K/V as **f16-rounded** values — prefill casts the freshly
//!    computed K/V to f16 (`ggml_cast` in the flash path, `llama-graph.cpp`
//!    `build_attn_mha`), decode reads the f16 KV cache (`build_attn` KV path,
//!    `mctx_cur->get_k`). The oracle therefore rounds every K/V to f16
//!    (round-to-nearest-even, `ggml_fp32_to_fp16` at `ggml/src/ggml-impl.h`)
//!    **after** RoPE and before attention; Q stays f32.
//! 4. **Metal f16 weight materialization**: the pinned comparator's Metal
//!    matmul kernels dequantize quantized weights into **f16 registers**
//!    (`ggml-metal.metal` `dequantize_q5_0`/`dequantize_q8_0`/
//!    `dequantize_q4_K`/`dequantize_q6_K`, `type4x4` = `half4x4`), so every
//!    quantized weight the comparator multiplies is f16-rounded. The oracle
//!    materializes all quantized weights through f16 (RN) via
//!    [`metal_f16_round_weights`]; the F32 norms are untouched. (The
//!    comparator's **CPU** kernels instead quantize activations to Q8_0/Q8_K
//!    before the dot — a *different* arithmetic; the oracle models the Metal
//!    path, which is what produced the pinned reference.)
//! 5. **EOG-exclusion is a top-1 SURFACE rule** (numeric contract §2.1), not
//!    a logits instruction: the oracle's raw logits stay finite; the
//!    teacher-forced comparison applies the EOG-excluding argmax over
//!    {0, 2} (`<|endoftext|>`, `<|im_end|>`).
//!
//! NUMERIC CONTRACT — two versions, per the operator-approved v2.0.0
//! two-level contract (decision need `41da94f3`, closed 2026-08-06):
//!
//! - **v1.0.0** (`gi0-numeric-contract.md` §4 — the original band): window
//!   positions 0..16; top-1 exact over non-EOG {0,2} vs
//!   `correctness-top1.json`; top-k k=5 ≥4/5 on raw normalized logits;
//!   per-element band Δ=1e-5 in log-softmax space full vocab; finite gate;
//!   first-divergence rule. **Preserved as an honest failure** — the oracle
//!   does NOT meet the 1e-5 band (see BAND STATUS); the v1.0.0 comparison
//!   test keeps recording the divergence (two-version history is the
//!   credibility asset).
//! - **v2.0.0** (decision `41da94f3` + council input `4488ccab`,
//!   operator-approved): the hard gates stay — top-1 exact over non-EOG
//!   {0,2}, top-k k=5 ≥4/5, finite gate, first-divergence rule, model hash,
//!   tokenizer, RoPE, tied-head facts separately reported — and the
//!   per-element band is replaced by **`Delta_comparator_metal` = 2.5e-2**,
//!   a **pinned-row empirical compatibility envelope** over the normalized-
//!   logp surface (frozen model, comparator binary, workload, positions) —
//!   explicitly **not** an f32 precision bound and **not** generalizable
//!   (the compared surfaces are the readable Faber CPU oracle vs the pinned
//!   llama.cpp Metal comparator). The receipt records the calibration
//!   maximum + headroom; a future observation above the envelope **FAILS**
//!   and triggers diagnosis/versioning (no auto-widen). DISCLOSURE (council
//!   condition 5): 2.5e-2 is 2,500× the v1.0.0 band and exceeds the min
//!   effective top-1 margin M=9.634e-3, so the envelope cannot prove
//!   decision invariance — the product decision/ranking claim is preserved
//!   by the hard top-1/top-k gates. Training `numeric-policy.md` rows never
//!   apply (memo `fdc2a448`).
//!
//! BAND STATUS (honest, v2.0.0 closeout): the oracle meets the **v2.0.0**
//! contract at all 17 window positions — top-1 exact (including the
//! EOG-exclusion case at position 1), top-k ≥4/5, finite, and
//! `max |logp_oracle − logp_comparator|` ≤ `Delta_comparator_metal` (2.5e-2)
//! at every position, so the v2.0.0 divergence field is `none`. The v1.0.0
//! 1e-5 band remains unmet (~1.3e-2..2.2e-2 max per position — the residual
//! is the comparator's Metal-kernel arithmetic: f16 dequant structures for
//! Q6_K/Q5_0, flash-attention softmax/accumulation, matrix-core accumulation
//! order, which a readable CPU op surface does not reproduce bit-for-bit).
//! That v1.0.0 failure is preserved and recorded, never weakened or hidden.
//!
//! FAIL CLOSED — [`OracleError`] for: a gapped/forged view (GI1-4 residual,
//! dequant gates on `coverage_ok()`), an empty token sequence, a token id
//! outside `[0, vocab_size)`, or any op/layout contradiction.
//!
//! ORACLE MATERIALIZATIONS — every cached weight `Vec<f32>` is a CPU-oracle
//! materialization (GI2-1): produced via `dequant_tensor` and accompanied by
//! an [`OracleReceipt`] (exposed via [`CpuOracle::receipts`]). This
//! conversion neither changes `RepackIdentity::Native` nor authorizes
//! converted-weight GPU/headline execution.

use crate::decoder_ops::{
    dense, residual_add, rms_norm, rope_norm, swiglu, OpError, ATTENTION_SCALE, FFN_SIZE,
    HEAD_COUNT, HEAD_DIM, HIDDEN_SIZE, KV_HEAD_COUNT, RMS_EPS,
};
use crate::dequant::{dequant_tensor, half_to_f32, DequantError, OracleReceipt};
use crate::tensor_view::TensorView;
use std::fmt;

// ---------------------------------------------------------------------------
// Pinned-row oracle constants (model contract §2, FC2)
// ---------------------------------------------------------------------------

/// Layer count (`llama.block_count` = 32).
pub const LAYER_COUNT: usize = 32;
/// Vocab size (`llama.vocab_size` = 49,152).
pub const VOCAB_SIZE: usize = 49_152;
/// EOG set for the pinned row (`<|endoftext|>` 0, `<|im_end|>` 2; model
/// contract §4/§5) — the top-1 surface excludes these (numeric contract
/// §2.1).
pub const EOG_TOKENS: [i64; 2] = [0, 2];
/// Contract window: prompt end + first 16 decode positions (numeric contract
/// §3).
pub const WINDOW_POSITIONS: usize = 17;
/// v1.0.0 per-element band Δ (`gi0-numeric-contract.md` v1.0.0 §4.3) — an
/// **unmet** band, preserved as an honest failure: the v1.0.0 comparison
/// test records the divergence and never weakens it.
pub const BAND_DELTA: f32 = 1e-5;
/// v2.0.0 per-element envelope **`Delta_comparator_metal`** (decision need
/// `41da94f3`, operator-approved 2026-08-06): the pinned-row empirical
/// compatibility envelope over the normalized-logp surface (frozen model,
/// comparator binary, workload, positions) — explicitly NOT an f32 precision
/// bound and NOT generalizable. The receipt records the calibration maximum
/// + headroom; a future observation above the envelope FAILS and triggers
/// diagnosis/versioning (no auto-widen).
pub const BAND_DELTA_V2: f32 = 2.5e-2;
/// Top-k default (numeric contract §4.2).
pub const TOPK_K: usize = 5;
/// Top-k minimum overlap (≥4/5, numeric contract §4.2).
pub const TOPK_MIN_OVERLAP: usize = 4;

// ---------------------------------------------------------------------------
// Typed fail-closed diagnostics
// ---------------------------------------------------------------------------

/// A typed, machine-parseable oracle rejection.
#[derive(Debug, Clone, PartialEq)]
pub enum OracleError {
    /// The dequant layer refused a tensor (gapped view, uncovered entry,
    /// repack, byte mismatch, view access failure).
    Dequant(DequantError),
    /// An op rejected a shape contradiction.
    Op(OpError),
    /// No tokens — the oracle cannot produce a next-position distribution.
    EmptyTokens,
    /// A token id is outside the admitted vocab `[0, vocab_size)`.
    TokenOutOfRange { token: i64, vocab: usize },
    /// A logits/logp vector was empty or contained a non-finite value where
    /// the finite gate requires finite values.
    NonFinite { detail: String },
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dequant(err) => write!(f, "dequant: {err}"),
            Self::Op(err) => write!(f, "op: {err}"),
            Self::EmptyTokens => write!(f, "oracle needs at least one token"),
            Self::TokenOutOfRange { token, vocab } => {
                write!(
                    f,
                    "token {token} is outside the admitted vocab [0, {vocab})"
                )
            }
            Self::NonFinite { detail } => write!(f, "finite gate: {detail}"),
        }
    }
}

impl std::error::Error for OracleError {}

impl From<DequantError> for OracleError {
    fn from(err: DequantError) -> Self {
        Self::Dequant(err)
    }
}

impl From<OpError> for OracleError {
    fn from(err: OpError) -> Self {
        Self::Op(err)
    }
}

// ---------------------------------------------------------------------------
// f32 → f16 (round-to-nearest-even) — the KV-cache fidelity conversion
// ---------------------------------------------------------------------------

/// Convert `x` to an IEEE-754 binary16 bit pattern with **round-to-nearest-
/// even**, matching `ggml_fp32_to_fp16` (`ggml/src/ggml-impl.h`) and the
/// Metal `half` conversion — the rounding the pinned comparator applies to
/// every K/V value it stores in the f16 KV cache (`--cache-type-k f16
/// --cache-type-v f16`) before attention.
///
/// Rounding rules: exact values map exactly; half-way values round to the
/// even significand; values in `[65504, 65520)` round to 65504 or overflow
/// to ±Inf per RN (the exact tie at 65512.0 rounds up to Inf — a documented
/// corner that cannot occur in the pinned row's activation range); `|x| ≥
/// 65520` overflows to ±Inf; subnormals round via `|x|·2²⁴` with RN (the
/// f16 subnormal ulp is 2⁻²⁴).
#[must_use]
pub fn f32_to_f16_rn(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let m = bits & 0x7fff_ffff; // |x| raw bits
    if m >= 0x7f80_0000 {
        // NaN / ±Inf
        if m == 0x7f80_0000 {
            return sign | 0x7c00;
        }
        // NaN: keep the high 10 payload bits (exact for NaN values that came
        // from a half, whose payload is 13-bit-aligned); a zero payload (a
        // signaling f32 NaN with no high mantissa bits) becomes the quiet
        // canonical NaN 0x7c01 — a NaN must never collapse to Inf.
        let mant = ((m >> 13) & 0x3ff) as u16;
        if mant == 0 {
            return sign | 0x7c01;
        }
        return sign | 0x7c00 | mant;
    }
    if m >= 0x47ff_f000 {
        // |x| ≥ 65520 → ±Inf (f16 max is 65504)
        return sign | 0x7c00;
    }
    if m >= 0x3880_0000 {
        // Normal f16: 2⁻¹⁴ ≤ |x| < 65520. Rebase the exponent (f32 bias 127
        // → f16 bias 15) and round the 23-bit fraction to 10 bits with
        // round-half-to-even on the 13 discarded bits.
        let unexp = (m >> 23) & 0xff;
        let mant = m & 0x7fffff;
        let mut word = ((unexp - 112) << 10) | (mant >> 13);
        let rest = mant & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && (word & 1) != 0) {
            word += 1;
        }
        return sign | word as u16;
    }
    // Subnormal f16 (or ±0): 0 < |x| < 2⁻¹⁴ → value = m10·2⁻²⁴ with
    // m10 = round-half-to-even(|x|·2²⁴) ∈ {0..1023}. The scaling is exact
    // (≤ 24 significant bits), so the RN is exact.
    let scaled = f32::from_bits(m) * 16_777_216.0f32;
    let m10 = round_half_to_even(scaled);
    debug_assert!(m10 <= 1023);
    sign | m10 as u16
}

/// Round a non-negative f32 in `[0, 1024)` to the nearest integer,
/// ties-to-even. `v - floor(v)` is exact for the callers' inputs.
fn round_half_to_even(v: f32) -> u32 {
    let f = v.floor();
    let frac = v - f;
    if frac < 0.5 {
        f as u32
    } else if frac > 0.5 {
        (f + 1.0) as u32
    } else {
        let fi = f as u32;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// Apply the KV-cache f16 rounding to a value: `f32 → f16 (RN) → f32 (exact)`.
#[must_use]
fn kv_f16_round(x: f32) -> f32 {
    half_to_f32(f32_to_f16_rn(x))
}

// ---------------------------------------------------------------------------
// Materialized layer weights
// ---------------------------------------------------------------------------

/// One decoder layer's materialized weights (CPU-oracle f32 materializations
/// of the admitted row; all dequantized via `dequant_tensor` under the
/// coverage gate).
struct LayerWeights {
    attn_norm: Vec<f32>,
    attn_q: Vec<f32>,
    attn_k: Vec<f32>,
    attn_v: Vec<f32>,
    attn_output: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

impl LayerWeights {
    /// Dequantize the nine `blk.{il}.*` tensors of layer `il` through the
    /// view, appending one [`OracleReceipt`] per materialization.
    fn build(
        view: &TensorView<'_>,
        il: usize,
        receipts: &mut Vec<OracleReceipt>,
    ) -> Result<Self, OracleError> {
        let name = |base: &str| format!("blk.{il}.{base}");
        let mut tensor = |base: &str| -> Result<Vec<f32>, OracleError> {
            let full = name(base);
            let entry = view.tensor(&full).ok_or_else(|| {
                OracleError::Dequant(DequantError::EntryNotCovered { name: full.clone() })
            })?;
            let mut values = dequant_tensor(view, entry)?;
            metal_f16_round_weights(&mut values);
            receipts.push(OracleReceipt::for_tensor(view, entry)?);
            Ok(values)
        };
        Ok(Self {
            attn_norm: tensor("attn_norm.weight")?,
            attn_q: tensor("attn_q.weight")?,
            attn_k: tensor("attn_k.weight")?,
            attn_v: tensor("attn_v.weight")?,
            attn_output: tensor("attn_output.weight")?,
            ffn_norm: tensor("ffn_norm.weight")?,
            ffn_gate: tensor("ffn_gate.weight")?,
            ffn_up: tensor("ffn_up.weight")?,
            ffn_down: tensor("ffn_down.weight")?,
        })
    }
}

/// The pinned comparator's Metal matmul kernels store dequantized weight
/// values as **f16 registers** (`ggml-metal.metal` `dequantize_q5_0` /
/// `dequantize_q8_0` / `dequantize_q4_K` / `dequantize_q6_K`, then
/// `type4x4`/`half4x4`), so every quantized weight the comparator multiplies
/// is f16-rounded. For Q8_0 (`f16(q·d)`) and Q4_K (`f16(dl·q − ml)`) this is
/// exactly `f16(value)` of the f32 dequant; for Q5_0 (`f16(d·x0 − 16·d)`)
/// the f32 structure can differ by ≤1 f16 ulp on rare ties. The oracle
/// rounds every materialized quantized weight through f16 (RN) to reproduce
/// the comparator's arithmetic (the F32 norms are untouched).
fn metal_f16_round_weights(w: &mut [f32]) {
    for v in w.iter_mut() {
        *v = half_to_f32(f32_to_f16_rn(*v));
    }
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

/// The CPU one-position logits oracle for the pinned row.
///
/// Built once from the admitted tensor view (all 290 tensors materialized to
/// f32 under the `coverage_ok()` gate), then used for [`CpuOracle::forward_one`]
/// and the incremental [`ForwardRun`].
pub struct CpuOracle {
    /// Tied embedding / output head (`token_embd.weight`, Q8_0, [960,49152]
    /// K-major) — the only embedding tensor (no `output.weight`).
    tok_embd: Vec<f32>,
    /// The 32 decoder layers.
    layers: Vec<LayerWeights>,
    /// `output_norm.weight` (F32 [960]) — the final norm.
    output_norm: Vec<f32>,
    /// Model identity + provenance (hash-accounted).
    model_sha256_hex: String,
    file_size: u64,
    /// One [`OracleReceipt`] per materialized tensor (290 for the pinned
    /// row) — the oracle-materialization contract.
    receipts: Vec<OracleReceipt>,
}

impl fmt::Debug for CpuOracle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuOracle")
            .field("model_sha256_hex", &self.model_sha256_hex)
            .field("file_size", &self.file_size)
            .field("tok_embd_elements", &self.tok_embd.len())
            .field("layers", &self.layers.len())
            .field("receipts", &self.receipts.len())
            .finish()
    }
}

impl CpuOracle {
    /// Materialize the oracle from the admitted tensor view.
    ///
    /// GI1-4 residual folded in: construction **gates on `coverage_ok()`** —
    /// a gapped or forged view is refused before any byte is touched (every
    /// `dequant_tensor` call re-checks the same gate).
    ///
    /// # Errors
    ///
    /// Returns the first typed [`OracleError`] the view contradicts.
    pub fn build(view: &TensorView<'_>) -> Result<Self, OracleError> {
        if !view.coverage_ok() {
            return Err(DequantError::CoverageNotOk.into());
        }
        let mut receipts = Vec::with_capacity(290);

        let tok_embd_entry = view.tensor("token_embd.weight").ok_or_else(|| {
            OracleError::Dequant(DequantError::EntryNotCovered {
                name: "token_embd.weight".into(),
            })
        })?;
        let mut tok_embd = dequant_tensor(view, tok_embd_entry)?;
        metal_f16_round_weights(&mut tok_embd);
        receipts.push(OracleReceipt::for_tensor(view, tok_embd_entry)?);

        let mut layers = Vec::with_capacity(LAYER_COUNT);
        for il in 0..LAYER_COUNT {
            layers.push(LayerWeights::build(view, il, &mut receipts)?);
        }

        let out_norm_entry = view.tensor("output_norm.weight").ok_or_else(|| {
            OracleError::Dequant(DequantError::EntryNotCovered {
                name: "output_norm.weight".into(),
            })
        })?;
        let output_norm = dequant_tensor(view, out_norm_entry)?;
        receipts.push(OracleReceipt::for_tensor(view, out_norm_entry)?);

        Ok(Self {
            model_sha256_hex: view.sha256_hex(),
            file_size: view.file_size(),
            tok_embd,
            layers,
            output_norm,
            receipts,
        })
    }

    /// Model SHA-256 (hex) recorded at build.
    #[must_use]
    pub fn model_sha256_hex(&self) -> &str {
        &self.model_sha256_hex
    }

    /// Admitted file size (bytes).
    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// One [`OracleReceipt`] per materialized tensor — the CPU-oracle
    /// materialization contract (290 for the pinned row).
    #[must_use]
    pub fn receipts(&self) -> &[OracleReceipt] {
        &self.receipts
    }

    /// Degenerate test-only oracle (empty weights) for the fail-closed guard
    /// tests — never reaches a tensor (empty-token and out-of-range guards
    /// run before any byte is touched).
    #[cfg(test)]
    pub(crate) fn degenerate_for_test() -> Self {
        Self {
            tok_embd: vec![0.0; 1],
            layers: Vec::new(),
            output_norm: vec![0.0; 1],
            model_sha256_hex: String::new(),
            file_size: 0,
            receipts: Vec::new(),
        }
    }

    // -- temporary diagnostics (removed before commit) -----------------------
    #[cfg(test)]
    pub(crate) fn embed_rows_for_diag(&self, tokens: &[i64]) -> Vec<f32> {
        self.embed_rows(tokens)
    }
    #[cfg(test)]
    pub(crate) fn forward_layer_for_diag(
        &self,
        il: usize,
        h: &[f32],
    ) -> Result<Vec<f32>, OracleError> {
        self.forward_layer(&self.layers[il], h)
    }
    #[cfg(test)]
    pub(crate) fn output_norm_for_diag(&self) -> &[f32] {
        &self.output_norm
    }
    #[cfg(test)]
    pub(crate) fn tok_embd_for_diag(&self) -> &[f32] {
        &self.tok_embd
    }

    /// Full-vocab raw logits for the position **after** `tokens`.
    ///
    /// The one-position forward: embedding gather → 32 layers → output norm →
    /// tied final projection. `tokens` are evaluated exactly (BOS-free;
    /// `add_bos_token=false`).
    ///
    /// # Errors
    ///
    /// Empty `tokens`, an out-of-range token id, or an op/layout
    /// contradiction (fail closed).
    pub fn forward_one(&self, tokens: &[i64]) -> Result<Vec<f32>, OracleError> {
        if tokens.is_empty() {
            return Err(OracleError::EmptyTokens);
        }
        for &t in tokens {
            if !(0..VOCAB_SIZE as i64).contains(&t) {
                return Err(OracleError::TokenOutOfRange {
                    token: t,
                    vocab: VOCAB_SIZE,
                });
            }
        }
        let mut h = self.embed_rows(tokens);
        for layer in &self.layers {
            h = self.forward_layer(layer, &h)?;
        }
        // Output norm (F32, eps 1e-5) + tied final projection.
        let n = tokens.len();
        let last = &h[(n - 1) * HIDDEN_SIZE..n * HIDDEN_SIZE];
        let normed = rms_norm(last, &self.output_norm, RMS_EPS)?;
        dense(&self.tok_embd, &normed, HIDDEN_SIZE, VOCAB_SIZE).map_err(OracleError::from)
    }

    /// Gather the token-embedding rows (`token_embd.weight` Q8_0 rows) for
    /// `tokens` → `[n, 960]`.
    fn embed_rows(&self, tokens: &[i64]) -> Vec<f32> {
        let mut out = Vec::with_capacity(tokens.len() * HIDDEN_SIZE);
        for &t in tokens {
            let ti = t as usize;
            out.extend_from_slice(&self.tok_embd[ti * HIDDEN_SIZE..(ti + 1) * HIDDEN_SIZE]);
        }
        out
    }

    /// One decoder layer over a `[n, 960]` hidden state (batch form; every
    /// row attends causally to `0..=row`).
    fn forward_layer(&self, layer: &LayerWeights, h: &[f32]) -> Result<Vec<f32>, OracleError> {
        let n = h.len() / HIDDEN_SIZE;
        debug_assert_eq!(h.len(), n * HIDDEN_SIZE);

        // attn_norm → QKV (per-token rows: `dense` is a single-row matmul).
        let mut a = Vec::with_capacity(h.len());
        for t in 0..n {
            a.extend(rms_norm(
                &h[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE],
                &layer.attn_norm,
                RMS_EPS,
            )?);
        }
        let mut q = Vec::with_capacity(n * HIDDEN_SIZE);
        let mut k = Vec::with_capacity(n * KV_HEAD_COUNT * HEAD_DIM);
        let mut v = Vec::with_capacity(n * KV_HEAD_COUNT * HEAD_DIM);
        for t in 0..n {
            let a_t = &a[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
            q.extend(dense(&layer.attn_q, a_t, HIDDEN_SIZE, HIDDEN_SIZE)?);
            k.extend(dense(
                &layer.attn_k,
                a_t,
                HIDDEN_SIZE,
                KV_HEAD_COUNT * HEAD_DIM,
            )?);
            v.extend(dense(
                &layer.attn_v,
                a_t,
                HIDDEN_SIZE,
                KV_HEAD_COUNT * HEAD_DIM,
            )?);
        }

        // RoPE (NORM consecutive-pair) + KV-cache f16 rounding (Q stays f32).
        let mut q_rot = Vec::with_capacity(q.len());
        let mut k_rot = Vec::with_capacity(k.len());
        let mut k_h16 = Vec::with_capacity(k.len());
        let mut v_h16 = Vec::with_capacity(v.len());
        for t in 0..n {
            q_rot.extend(rope_all_heads(
                &q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE],
                HEAD_COUNT,
                t as u32,
            )?);
            k_rot.extend(rope_all_heads(
                &k[t * KV_HEAD_COUNT * HEAD_DIM..(t + 1) * KV_HEAD_COUNT * HEAD_DIM],
                KV_HEAD_COUNT,
                t as u32,
            )?);
            for j in 0..KV_HEAD_COUNT * HEAD_DIM {
                k_h16.push(kv_f16_round(k_rot[t * KV_HEAD_COUNT * HEAD_DIM + j]));
                v_h16.push(kv_f16_round(v[t * KV_HEAD_COUNT * HEAD_DIM + j]));
            }
        }

        // Causal GQA attention (verified consecutive-triples grouping),
        // attn_output (per-token rows), residual.
        let ctx = attention_gqa(
            &q_rot,
            &k_h16,
            &v_h16,
            n,
            n,
            HEAD_COUNT,
            KV_HEAD_COUNT,
            HEAD_DIM,
            ATTENTION_SCALE,
        )?;
        let mut o = Vec::with_capacity(ctx.len());
        for t in 0..n {
            o.extend(dense(
                &layer.attn_output,
                &ctx[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE],
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            )?);
        }
        let h2 = residual_add(h, &o)?;

        // ffn_norm → SwiGLU → ffn_down → residual (per-token rows).
        let mut f = Vec::with_capacity(h2.len());
        for t in 0..n {
            f.extend(rms_norm(
                &h2[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE],
                &layer.ffn_norm,
                RMS_EPS,
            )?);
        }
        let mut hh = Vec::with_capacity(n * HIDDEN_SIZE);
        for t in 0..n {
            let f_t = &f[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
            let gate = dense(&layer.ffn_gate, f_t, HIDDEN_SIZE, FFN_SIZE)?;
            let up = dense(&layer.ffn_up, f_t, HIDDEN_SIZE, FFN_SIZE)?;
            hh.extend(swiglu(&gate, &up)?);
        }
        let mut down = Vec::with_capacity(n * HIDDEN_SIZE);
        for t in 0..n {
            down.extend(dense(
                &layer.ffn_down,
                &hh[t * FFN_SIZE..(t + 1) * FFN_SIZE],
                FFN_SIZE,
                HIDDEN_SIZE,
            )?);
        }
        residual_add(&h2, &down).map_err(OracleError::from)
    }
}

/// Apply [`rope_norm`] to each of `n_heads` contiguous `HEAD_DIM`-wide heads.
fn rope_all_heads(x: &[f32], n_heads: usize, pos: u32) -> Result<Vec<f32>, OpError> {
    let mut out = Vec::with_capacity(n_heads * HEAD_DIM);
    for h in 0..n_heads {
        out.extend(rope_norm(
            &x[h * HEAD_DIM..(h + 1) * HEAD_DIM],
            pos,
            crate::decoder_ops::ROPE_FREQ_BASE,
            crate::decoder_ops::ROPE_DIM,
        )?);
    }
    Ok(out)
}

/// GQA causal attention for the pinned row's **verified** head-grouping.
///
/// Query head `h` uses KV head `h / (n_heads / n_kv_heads)` — the
/// consecutive-triples grouping (15/5 → q heads 0,1,2 share kv head 0; 3,4,5
/// share 1; …). This is the FC7 verification outcome against the pinned
/// llama.cpp tree (`a957b7747`):
/// - CPU non-flash path: `kq = ggml_mul_mat(k, q)` broadcasts the KV batch
///   dimension with `r2 = ne12 / ne02`, `i02 = i12 / r2` (`ggml-cpu.c`,
///   `mul_mat_f32` — K batch = Q batch / r2);
/// - Metal flash path (the comparator's actual path):
///   `ikv2 = iq2 / (args.ne02 / args.ne_12_2)` (`ggml-metal.metal`,
///   `kernel_flash_attn_ext_impl`).
///
/// NOTE — this differs from [`crate::decoder_ops::attention_causal`]'s
/// `h % n_kv_heads` grouping, which matches the OLD llama.cpp convention and
/// coincides only when `n_heads == n_kv_heads`. The comparator executes the
/// consecutive-triples grouping, so the oracle must too. (The GI2-2 op's
/// attention golden was originally committed under the older `h % n_kv_heads`
/// convention and is **superseded**: corrected to this grouping and re-pinned
/// in `testdata/gi2-2-op-goldens/manifest.json` per CTO 8173f0cf — GI3 must
/// consume the corrected fixture, not the stale one.)
///
/// Layouts (row-major, contiguous — the dequant order):
/// - `q`: `n_q × (n_heads·HEAD_DIM)`;
/// - `k`, `v`: `n_kv × (n_kv_heads·HEAD_DIM)` (already f16-rounded post-RoPE
///   K / f16-rounded V by the caller);
/// - returns `n_q × (n_heads·HEAD_DIM)` context rows.
///
/// Causal semantics: query row `i` sits at global position `n_kv − n_q + i`
/// and attends to key rows `0 ..= global_position`. `score = scale·⟨q,k⟩`
/// (scale 0.125), row-wise softmax, context `Σ p·v`.
///
/// # Errors
///
/// Returns a typed [`OpError`] on a shape contradiction (fail-closed; the
/// oracle's shapes are fixed by the pinned row).
///
/// `pub(crate)` so the GI2-2 op-golden generator shares this single verified
/// grouping (the fixture must be produced by the same math the oracle and the
/// comparator execute, not by a parallel copy).
pub(crate) fn attention_gqa(
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
    let check = |cond: bool, detail: String| {
        if cond {
            Ok(())
        } else {
            Err(OpError {
                op: "attention_gqa",
                detail,
            })
        }
    };
    check(n_q <= n_kv, format!("n_q ({n_q}) must be ≤ n_kv ({n_kv})"))?;
    check(
        n_heads % n_kv_heads == 0,
        format!("n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"),
    )?;
    check(
        q.len() == n_q * q_span,
        format!(
            "q ({}) must be n_q·n_heads·head_dim ({n_q}·{n_heads}·{head_dim})",
            q.len()
        ),
    )?;
    check(
        k.len() == n_kv * kv_span,
        format!(
            "k ({}) must be n_kv·n_kv_heads·head_dim ({n_kv}·{n_kv_heads}·{head_dim})",
            k.len()
        ),
    )?;
    check(
        v.len() == n_kv * kv_span,
        format!(
            "v ({}) must be n_kv·n_kv_heads·head_dim ({n_kv}·{n_kv_heads}·{head_dim})",
            v.len()
        ),
    )?;

    let mut ctx = vec![0.0f32; n_q * q_span];
    let ratio = n_heads / n_kv_heads;
    for i in 0..n_q {
        let global_pos = n_kv - n_q + i;
        let row_base = i * q_span;
        for h in 0..n_heads {
            // Verified comparator grouping: consecutive triples of Q heads
            // share one KV head.
            let kh = h / ratio;
            let qh = &q[row_base + h * head_dim..row_base + (h + 1) * head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let kj = &k[j * kv_span + kh * head_dim..j * kv_span + (kh + 1) * head_dim];
                let mut dot = 0.0f32;
                for (a, b) in qh.iter().zip(kj) {
                    dot += a * b;
                }
                scores.push(dot * scale);
            }
            let probs = crate::decoder_ops::softmax_row(&scores);
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
// Incremental forward runner (numeric-faithful position cache)
// ---------------------------------------------------------------------------

/// An incremental, **numeric-faithful** one-position forward runner.
///
/// [`ForwardRun::push_token`] appends one token and caches its per-layer
/// K/V (f16-rounded, post-RoPE — the same values the full forward computes);
/// [`ForwardRun::position_logits`] returns the raw logits at the current
/// sequence end. Because a token's residual stream and K/V depend only on the
/// tokens up to and including it, the incremental runner computes the same
/// arithmetic as [`CpuOracle::forward_one`] for the query row — **byte-
/// identical logits** (proven by test). The durable KV cache is GI4's; this
/// cache changes no numerics (GI2 non-goal).
pub struct ForwardRun<'o> {
    oracle: &'o CpuOracle,
    n: usize,
    /// The last token's hidden state after all 32 layers (`[960]`).
    h_last: Vec<f32>,
    /// Per-layer cached K (f16-rounded, post-RoPE), `[n, 320]` each.
    k_cache: Vec<Vec<f32>>,
    /// Per-layer cached V (f16-rounded), `[n, 320]` each.
    v_cache: Vec<Vec<f32>>,
}

impl<'o> ForwardRun<'o> {
    /// Start a run by prefilling `tokens` (e.g. the 9 pinned prompt tokens).
    ///
    /// # Errors
    ///
    /// Empty `tokens` or an out-of-range token id (fail closed).
    pub fn new(oracle: &'o CpuOracle, tokens: &[i64]) -> Result<Self, OracleError> {
        if tokens.is_empty() {
            return Err(OracleError::EmptyTokens);
        }
        let mut run = Self {
            oracle,
            n: 0,
            h_last: vec![0.0; HIDDEN_SIZE],
            k_cache: vec![Vec::new(); LAYER_COUNT],
            v_cache: vec![Vec::new(); LAYER_COUNT],
        };
        for &t in tokens {
            run.push_token(t)?;
        }
        Ok(run)
    }

    /// Number of tokens in the current sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the sequence is empty (never true after a successful `new`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Append one token (teacher-forced) and update the caches.
    ///
    /// # Errors
    ///
    /// An out-of-range token id or an op/layout contradiction (fail closed).
    pub fn push_token(&mut self, token: i64) -> Result<(), OracleError> {
        if !(0..VOCAB_SIZE as i64).contains(&token) {
            return Err(OracleError::TokenOutOfRange {
                token,
                vocab: VOCAB_SIZE,
            });
        }
        let pos = self.n as u32;
        let mut h = self.oracle.embed_rows(&[token]);
        for (il, layer) in self.oracle.layers.iter().enumerate() {
            // attn_norm → QKV.
            let a = rms_norm(&h, &layer.attn_norm, RMS_EPS)?;
            let q = dense(&layer.attn_q, &a, HIDDEN_SIZE, HIDDEN_SIZE)?;
            let k = dense(&layer.attn_k, &a, HIDDEN_SIZE, KV_HEAD_COUNT * HEAD_DIM)?;
            let v = dense(&layer.attn_v, &a, HIDDEN_SIZE, KV_HEAD_COUNT * HEAD_DIM)?;

            // RoPE + f16 rounding; append to the per-layer caches.
            let q_rot = rope_all_heads(&q, HEAD_COUNT, pos)?;
            let k_rot = rope_all_heads(&k, KV_HEAD_COUNT, pos)?;
            for &kv in &k_rot {
                self.k_cache[il].push(kv_f16_round(kv));
            }
            for &vv in &v {
                self.v_cache[il].push(kv_f16_round(vv));
            }

            // Causal attention for the single query at global position `pos`
            // against the full cached context, then attn_output + residual.
            let ctx = attention_gqa(
                &q_rot,
                &self.k_cache[il],
                &self.v_cache[il],
                1,
                self.n + 1,
                HEAD_COUNT,
                KV_HEAD_COUNT,
                HEAD_DIM,
                ATTENTION_SCALE,
            )?;
            let o = dense(&layer.attn_output, &ctx, HIDDEN_SIZE, HIDDEN_SIZE)?;
            let h2 = residual_add(&h, &o)?;

            // ffn_norm → SwiGLU → ffn_down → residual.
            let f = rms_norm(&h2, &layer.ffn_norm, RMS_EPS)?;
            let gate = dense(&layer.ffn_gate, &f, HIDDEN_SIZE, FFN_SIZE)?;
            let up = dense(&layer.ffn_up, &f, HIDDEN_SIZE, FFN_SIZE)?;
            let hh = swiglu(&gate, &up)?;
            let down = dense(&layer.ffn_down, &hh, FFN_SIZE, HIDDEN_SIZE)?;
            h = residual_add(&h2, &down)?;
        }
        self.n += 1;
        self.h_last = h;
        Ok(())
    }

    /// Full-vocab raw logits at the current sequence end (the next-token
    /// distribution after the sequence so far).
    ///
    /// # Errors
    ///
    /// Empty sequence or an op contradiction (fail closed).
    pub fn position_logits(&self) -> Result<Vec<f32>, OracleError> {
        if self.n == 0 {
            return Err(OracleError::EmptyTokens);
        }
        let normed = rms_norm(&self.h_last, &self.oracle.output_norm, RMS_EPS)?;
        dense(&self.oracle.tok_embd, &normed, HIDDEN_SIZE, VOCAB_SIZE).map_err(OracleError::from)
    }
}

// ---------------------------------------------------------------------------
// Numeric-contract comparison surface (gi0-numeric-contract.md v1.0.0)
// ---------------------------------------------------------------------------

/// log-softmax of raw logits over the full vocab — the normalized surface
/// the band and the top-k rule operate on (shift-invariant; the comparator's
/// `logprob` values are the same quantity).
///
/// `logp[t] = (logits[t] − max) − ln(Σ exp(logits − max))`.
///
/// # Errors
///
/// An empty vector or a non-finite raw logit fails the finite gate.
pub fn log_softmax(logits: &[f32]) -> Result<Vec<f32>, OracleError> {
    if logits.is_empty() {
        return Err(OracleError::NonFinite {
            detail: "empty logits".into(),
        });
    }
    let mut max = f32::NEG_INFINITY;
    for &x in logits {
        if !x.is_finite() {
            return Err(OracleError::NonFinite {
                detail: format!("raw logit {x} is not finite"),
            });
        }
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f32;
    for &x in logits {
        sum += (x - max).exp();
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(OracleError::NonFinite {
            detail: format!("log-softmax denominator {sum} is not a positive finite value"),
        });
    }
    let ln_sum = sum.ln();
    Ok(logits.iter().map(|&x| (x - max) - ln_sum).collect())
}

/// The greedy top-1 token id: argmax of the raw logits over **non-EOG**
/// tokens (numeric contract §4.1). EOG tokens are never the oracle's
/// prediction; an EOG token at raw argmax does not count.
///
/// All values are assumed finite (the caller runs the finite gate); an
/// all-masked logits vector yields token 0 (documented degenerate).
#[must_use]
pub fn top1_non_eog(logits: &[f32], eog: &[i64]) -> i64 {
    let mut best = -1i64;
    let mut best_val = f32::NEG_INFINITY;
    for (t, &v) in logits.iter().enumerate() {
        if eog.contains(&(t as i64)) {
            continue;
        }
        if v > best_val {
            best_val = v;
            best = t as i64;
        }
    }
    if best < 0 {
        // all values masked / non-finite — degenerate; report 0.
        return 0;
    }
    best
}

/// The top-`k` token ids of a normalized logp vector (EOG **included** —
/// numeric contract §4.2), descending by logp.
#[must_use]
pub fn topk_ids(logp: &[f32], k: usize) -> Vec<i64> {
    let mut order: Vec<usize> = (0..logp.len()).collect();
    order.sort_by(|&a, &b| {
        logp[b]
            .partial_cmp(&logp[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order.into_iter().map(|t| t as i64).collect()
}

/// Set overlap of two top-`k` id lists (numeric contract §4.2).
#[must_use]
pub fn topk_overlap(a: &[i64], b: &[i64]) -> usize {
    a.iter().filter(|t| b.contains(t)).count()
}

/// Max per-element |deviation| between two full-vocab logp vectors (numeric
/// contract §4.3). Assumes equal length (fail-closed callers check).
#[must_use]
pub fn max_band_deviation(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// A named numeric-contract threshold that can fail at a window position
/// (numeric contract §4.5: the divergence record names the failing
/// threshold(s), not just a boolean).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailingThreshold {
    /// §4.1 — exact top-1 over non-EOG {0, 2} vs the comparator trace token.
    Top1,
    /// §4.2 — top-k (k=5) overlap ≥4/5 vs the comparator top-5 set.
    TopK,
    /// §4.3 — per-element log-softmax band: v1.0.0 Δ ≤ 1e-5, v2.0.0
    /// `Delta_comparator_metal` ≤ 2.5e-2 over the full vocab.
    Band,
    /// §4.4 — finite-value gate (finite logits/logp, in-range ids).
    Finite,
}

impl fmt::Display for FailingThreshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Top1 => write!(f, "top-1"),
            Self::TopK => write!(f, "top-k"),
            Self::Band => write!(f, "band"),
            Self::Finite => write!(f, "finite"),
        }
    }
}

/// Per-position verdict of the numeric contract at one window position.
///
/// Every verdict carries the contract's first-divergence fields (numeric
/// contract §4.5): the comparator trace token, the oracle's EOG-excluded
/// top-1, and the named failing thresholds — per position, so the window is
/// replayable position by position.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionVerdict {
    /// Window position (0 = prompt end; `i` = after `i` teacher-forced trace
    /// tokens).
    pub position: u32,
    /// §4.1: exact top-1 over non-EOG {0,2} vs the pinned trace token.
    pub top1_matches: bool,
    /// §4.1 probe: the oracle's raw (EOG-included) argmax at this position —
    /// records the EOG-exclusion scenario (e.g. position 1 raw argmax = 2).
    pub raw_argmax: i64,
    /// §4.1/§4.5: the pinned comparator greedy trace token id at this
    /// position (`correctness-top1.json`).
    pub trace_token: i64,
    /// §4.1/§4.5: the oracle's **EOG-excluded** top-1 id at this position —
    /// the contract surface (not the raw argmax; EOG {0,2} never counts).
    pub oracle_top1: i64,
    /// §4.2: overlap size of the oracle vs comparator top-5 sets (≥4 = pass).
    pub topk_overlap: usize,
    /// §4.2 pass (≥4/5).
    pub topk_matches: bool,
    /// §4.3: max per-element |logp_oracle − logp_comparator| over the full
    /// vocab.
    pub max_band_deviation: f32,
    /// §4.3 pass (≤ 1e-5).
    pub band_matches: bool,
    /// §4.4: every raw logit and normalized logp finite, ids in range.
    pub all_finite: bool,
    /// §4.5: the named failing threshold(s) at this position; empty when
    /// every threshold passes (`ok`).
    pub failing_thresholds: Vec<FailingThreshold>,
    /// All of §4.1–§4.4.
    pub ok: bool,
}

/// The durable first-divergence record (numeric contract §4.5): taken at the
/// **first** diverging token position — the lowest generation position `i`
/// at which any threshold fails — with the comparator trace token id, the
/// oracle token id (the EOG-excluded top-1, §4.1), the named failing
/// threshold(s), and the max band deviation at that position. Later
/// disagreements never replace or obscure the first diverging position.
#[derive(Debug, Clone, PartialEq)]
pub struct DivergenceRecord {
    /// Window position of the first diverging token (0 = prompt end).
    pub position: u32,
    /// §4.1/§4.5: the pinned comparator greedy trace token id.
    pub comparator_trace_token: i64,
    /// §4.1/§4.5: the oracle's EOG-excluded top-1 token id.
    pub oracle_top1: i64,
    /// §4.5: the named failing threshold(s) (top-1, top-k, band, finite).
    pub failing_thresholds: Vec<FailingThreshold>,
    /// §4.3/§4.5: max per-element |logp_oracle − logp_comparator| at the
    /// first diverging position.
    pub max_band_deviation: f32,
}

impl DivergenceRecord {
    /// The first-divergence record over a full window of verdicts, or `None`
    /// when every position passes (contract §4.5: the lowest generation
    /// position `i` at which any threshold fails wins; later failures never
    /// replace it).
    #[must_use]
    pub fn first(verdicts: &[PositionVerdict]) -> Option<DivergenceRecord> {
        let v = verdicts.iter().find(|v| !v.ok)?;
        Some(DivergenceRecord {
            position: v.position,
            comparator_trace_token: v.trace_token,
            oracle_top1: v.oracle_top1,
            failing_thresholds: v.failing_thresholds.clone(),
            max_band_deviation: v.max_band_deviation,
        })
    }
}

/// Compare one window position against the pinned comparator reference under
/// the binding numeric contract.
///
/// - `oracle_logits`: the oracle's raw full-vocab logits (finite gate);
/// - `comparator_logp`: the comparator's full-vocab normalized logp (from the
///   committed 17×49152 reference fixture), indexed by token id;
/// - `trace_token`: the pinned greedy trace token at this window position
///   (`correctness-top1.json`);
/// - `band_delta`: the binding band/envelope threshold — v1.0.0 [`BAND_DELTA`]
///   (1e-5, honest failure) or v2.0.0 [`BAND_DELTA_V2`]
///   (`Delta_comparator_metal` = 2.5e-2, the closeout contract).
///
/// # Errors
///
/// Length mismatches or non-finite comparator values fail closed.
pub fn compare_position(
    position: u32,
    oracle_logits: &[f32],
    comparator_logp: &[f32],
    trace_token: i64,
    band_delta: f32,
) -> Result<PositionVerdict, OracleError> {
    if oracle_logits.len() != VOCAB_SIZE || comparator_logp.len() != VOCAB_SIZE {
        return Err(OracleError::NonFinite {
            detail: format!(
                "compare_position needs {VOCAB_SIZE}-wide vectors, got oracle {} / comparator {}",
                oracle_logits.len(),
                comparator_logp.len()
            ),
        });
    }
    let oracle_logp = log_softmax(oracle_logits)?;
    let all_finite = comparator_logp.iter().all(|v| v.is_finite())
        && oracle_logp.iter().all(|v| v.is_finite())
        && oracle_logits.iter().all(|v| v.is_finite());

    let raw_argmax = topk_ids(&oracle_logp, 1)[0];
    let oracle_top1 = top1_non_eog(oracle_logits, &EOG_TOKENS);
    let top1_matches = oracle_top1 == trace_token;

    let oracle_top5 = topk_ids(&oracle_logp, TOPK_K);
    let comparator_top5 = topk_ids(comparator_logp, TOPK_K);
    let overlap = topk_overlap(&oracle_top5, &comparator_top5);
    let topk_matches = overlap >= TOPK_MIN_OVERLAP;

    let max_dev = max_band_deviation(&oracle_logp, comparator_logp);
    let band_matches = max_dev <= band_delta;

    // §4.5: the named failing threshold(s) at this position.
    let mut failing_thresholds = Vec::new();
    if !top1_matches {
        failing_thresholds.push(FailingThreshold::Top1);
    }
    if !topk_matches {
        failing_thresholds.push(FailingThreshold::TopK);
    }
    if !band_matches {
        failing_thresholds.push(FailingThreshold::Band);
    }
    if !all_finite {
        failing_thresholds.push(FailingThreshold::Finite);
    }
    let ok = failing_thresholds.is_empty();

    Ok(PositionVerdict {
        position,
        top1_matches,
        raw_argmax,
        trace_token,
        oracle_top1,
        topk_overlap: overlap,
        topk_matches,
        max_band_deviation: max_dev,
        band_matches,
        all_finite,
        failing_thresholds,
        ok,
    })
}
