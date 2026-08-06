//! GI3-5 — prefill program builder + GPU-vs-oracle comparison harness.
//!
//! The prefill execution + correctness surface of the GI3 serial integration
//! (`gi3-delivery.md` §GI3-5). The prefill computes the **prompt-final
//! full-vocab logits** `[49152]` for the pinned 9-token correctness prompt
//! (position 0 = prompt end):
//!
//! ```text
//! embedding gather over the tied `token_embd.weight` [960, 49152] (the same
//!   weight is the output head — no `output.weight` exists)           → h [n, 960]
//! ×32 layers:
//!   attn_norm (RMSNorm, eps 1e-5) → QKV matmul (Q [960,960] → 15×64,
//!     K/V [960,320] → 5×64) → RoPE (llama-arch NORM consecutive-pair,
//!     freq_base 100000, dim 64) → causal attention (scores · 0.125 →
//!     CausalMaskedSoftmax → ·V, GQA 15/5, head grouping h/(n_head/n_kv_head))
//!     → attn_output matmul → residual → ffn_norm (RMSNorm) → SwiGLU
//!     (silu(gate)·up, 2560) → ffn_down matmul → residual
//! output_norm (RMSNorm, eps 1e-5) → final projection via the tied
//!   `token_embd.weight`ᵀ → full-vocab raw logits [49152]
//! ```
//!
//! [`PrefillProgramBuilder`] assembles this **program description** from the
//! admitted row facts — the ordered recipe sequence with the source weight
//! tensor names and shapes — mirroring the CPU oracle's forward path
//! ([`crate::cpu_oracle`]) exactly (the same math, the same order; proven by
//! the byte-identical reference-executor test). The faber Q1-default driver
//! maps the same description to the wire program; the device executes it.
//!
//! [`compare_gpu_logits`] is the **Q2 comparison harness**: GPU prompt-final
//! logits vs the committed GI2 logits golden under the Q2 thresholds
//! (`gi3-delivery.md` §Open Questions Q2 default — the training
//! `numeric-policy.md` v1.0.0 family mapping applied by analogy; the logits
//! surface is the tied-head projection, so the **matmul row** applies):
//! - **top-1 exact** over non-EOG {0, 2} at the prompt end vs the golden;
//! - **per-element numeric rows**: `|gpu − golden| ≤ atol + rtol·|golden|`
//!   with the matmul row `atol 1e-5 / rtol 1e-5` (elementwise rows
//!   `1e-6/1e-5`, reduction rows `1e-6/1e-6` are recorded for the op-level
//!   localization floor, never the gate);
//! - **finite gate**: every GPU and golden logit finite;
//! - **first-divergence rule** (`gi0-numeric-contract.md` §4.5): divergence is
//!   recorded at the first failing position (here the single prompt-end
//!   position), never hidden by text-level similarity.
//!
//! [`PrefillReceipt`] is the receipt type carrying the S6 prefill-regime
//! fields (shape class / representation / algorithm / workspace / evidence
//! recorded separately, CTO S6) + the device execution facts (transfers /
//! allocations / launches / syncs / observations) + the repack/conversion,
//! module-prep, persistent-upload, and first-invocation/capture timing
//! (CTO S11). The populated records land in the committed evidence
//! (`radix/docs/factory/gpu-inference-gguf/evidence/gi3-prefill-receipts.md`).
//!
//! Constraints carried in (delivery done-when): no per-op host recomputation
//! in the execution path; no hidden llama.cpp execution; the GI2 per-op
//! goldens (`testdata/gi2-2-op-goldens/`) stay the localization floor —
//! never the exit gate.

use crate::cpu_oracle::{top1_non_eog, FailingThreshold, EOG_TOKENS, LAYER_COUNT, VOCAB_SIZE};
use crate::decoder_ops::{
    ATTENTION_SCALE, FFN_SIZE, HEAD_COUNT, HEAD_DIM, HIDDEN_SIZE, KV_HEAD_COUNT, RMS_EPS,
};
use crate::json::Json;
use crate::valor::Valor;
use std::collections::BTreeMap;

/// The pinned 9 BOS-free prompt tokens (gi0-workloads §3.1).
pub const PROMPT_TOKENS: [i64; 9] = [504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767];

/// Q2 per-element numeric thresholds — the training `numeric-policy.md`
/// v1.0.0 family mapping applied by analogy to the GPU-vs-oracle logits
/// (gi3-delivery §Q2 default). The logits surface is produced by the
/// tied-head projection (a matmul), so the **matmul row** is the gate.
pub const Q2_ATOL: f32 = 1e-5;
/// Q2 relative tolerance (matmul row).
pub const Q2_RTOL: f32 = 1e-5;
/// Elementwise row (op-level localization floor, never the gate).
pub const Q2_ELEMENTWISE_ATOL: f32 = 1e-6;
/// Elementwise row (op-level localization floor, never the gate).
pub const Q2_ELEMENTWISE_RTOL: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Prefill program builder
// ---------------------------------------------------------------------------

/// The prefill recipe kinds (the GI3 kernel-plan family + the existing
/// elementwise/matmul recipes the prefill assembles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillRecipe {
    /// Embedding gather over the tied `token_embd.weight` rows (the GI3
    /// `Gather` recipe).
    EmbeddingGather,
    /// RMSNorm (`RmsNormalization` recipe).
    RmsNorm,
    /// Quantized matmul (the `TiledMatMul`/`MatMul` recipe family).
    Matmul,
    /// RoPE (`Rope` recipe, NORM consecutive-pair).
    Rope,
    /// Causal masked softmax over the scaled attention scores
    /// (`CausalMaskedSoftmax` recipe; scores · 0.125).
    CausalMaskedSoftmax,
    /// SwiGLU activation (elementwise `silu(gate)·up`).
    Swiglu,
    /// Residual add (elementwise `TensorAdd`).
    ResidualAdd,
}

impl PrefillRecipe {
    /// The recipe's name (the receipt / localization vocabulary).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::EmbeddingGather => "gather",
            Self::RmsNorm => "rms_norm",
            Self::Matmul => "matmul",
            Self::Rope => "rope",
            Self::CausalMaskedSoftmax => "causal_masked_softmax",
            Self::Swiglu => "swiglu",
            Self::ResidualAdd => "residual_add",
        }
    }
}

/// One kernel slot of the prefill program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillKernel {
    /// The recipe kind.
    pub recipe: PrefillRecipe,
    /// The source weight tensor this kernel consumes (the admitted row's
    /// tensor name), when the kernel carries weights.
    pub weight_tensor: Option<&'static str>,
    /// Static output shape of the kernel slot.
    pub shape: Vec<u64>,
}

/// The prefill program description for the pinned row.
///
/// The ordered kernel sequence (gather → 32 decoder layers → output norm →
/// tied-head projection) mirrors the CPU oracle's forward path exactly; the
/// faber Q1-default driver maps it to the wire program for the device.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillProgram {
    /// The pinned correctness prompt.
    pub prompt_tokens: [i64; 9],
    /// Row facts (frozen by the model contract v1.0.0).
    pub layer_count: usize,
    pub hidden_size: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub ffn_size: usize,
    pub vocab_size: usize,
    /// Attention score scale (`1/sqrt(head_dim)` = 0.125).
    pub attention_scale: f32,
    /// RMSNorm epsilon (`1e-5`).
    pub rms_eps: f32,
    /// The ordered kernel sequence.
    pub kernels: Vec<PrefillKernel>,
}

/// Assembles the prefill program description from the admitted row facts.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillProgramBuilder;

impl PrefillProgramBuilder {
    /// Build the pinned-row prefill program: embedding gather → 32 layers ×
    /// 11 kernels → output norm → tied-head projection.
    #[must_use]
    pub fn build() -> PrefillProgram {
        let mut kernels = Vec::with_capacity(1 + LAYER_COUNT * 11 + 2);
        // Embedding gather over the tied token_embd rows → h [n, 960].
        kernels.push(PrefillKernel {
            recipe: PrefillRecipe::EmbeddingGather,
            weight_tensor: Some("token_embd.weight"),
            shape: vec![9, HIDDEN_SIZE as u64],
        });
        let mut layer = |base: String, shape: Vec<u64>, recipe: PrefillRecipe| PrefillKernel {
            recipe,
            weight_tensor: Some(Box::leak(base.into_boxed_str())),
            shape,
        };
        for il in 0..LAYER_COUNT {
            // attn_norm (RMSNorm) → Q/K/V matmuls.
            kernels.push(layer(
                format!("blk.{il}.attn_norm.weight"),
                vec![9, HIDDEN_SIZE as u64],
                PrefillRecipe::RmsNorm,
            ));
            kernels.push(layer(
                format!("blk.{il}.attn_q.weight"),
                vec![9, HIDDEN_SIZE as u64],
                PrefillRecipe::Matmul,
            ));
            kernels.push(layer(
                format!("blk.{il}.attn_k.weight"),
                vec![9, (KV_HEAD_COUNT * HEAD_DIM) as u64],
                PrefillRecipe::Matmul,
            ));
            kernels.push(layer(
                format!("blk.{il}.attn_v.weight"),
                vec![9, (KV_HEAD_COUNT * HEAD_DIM) as u64],
                PrefillRecipe::Matmul,
            ));
            // RoPE (q + k) → causal attention (scores·0.125 →
            // CausalMaskedSoftmax → ·V) → attn_output matmul.
            kernels.push(PrefillKernel {
                recipe: PrefillRecipe::Rope,
                weight_tensor: None,
                shape: vec![9, HIDDEN_SIZE as u64],
            });
            kernels.push(PrefillKernel {
                recipe: PrefillRecipe::Rope,
                weight_tensor: None,
                shape: vec![9, (KV_HEAD_COUNT * HEAD_DIM) as u64],
            });
            kernels.push(PrefillKernel {
                recipe: PrefillRecipe::CausalMaskedSoftmax,
                weight_tensor: None,
                shape: vec![9, 9],
            });
            kernels.push(layer(
                format!("blk.{il}.attn_output.weight"),
                vec![9, HIDDEN_SIZE as u64],
                PrefillRecipe::Matmul,
            ));
            // Residual → ffn_norm → SwiGLU → ffn_down → residual.
            kernels.push(PrefillKernel {
                recipe: PrefillRecipe::ResidualAdd,
                weight_tensor: None,
                shape: vec![9, HIDDEN_SIZE as u64],
            });
            kernels.push(layer(
                format!("blk.{il}.ffn_norm.weight"),
                vec![9, HIDDEN_SIZE as u64],
                PrefillRecipe::RmsNorm,
            ));
            kernels.push(PrefillKernel {
                recipe: PrefillRecipe::Swiglu,
                weight_tensor: None,
                shape: vec![9, FFN_SIZE as u64],
            });
            kernels.push(layer(
                format!("blk.{il}.ffn_down.weight"),
                vec![9, HIDDEN_SIZE as u64],
                PrefillRecipe::Matmul,
            ));
            kernels.push(PrefillKernel {
                recipe: PrefillRecipe::ResidualAdd,
                weight_tensor: None,
                shape: vec![9, HIDDEN_SIZE as u64],
            });
        }
        // output_norm → tied-head projection → logits [49152].
        kernels.push(PrefillKernel {
            recipe: PrefillRecipe::RmsNorm,
            weight_tensor: Some("output_norm.weight"),
            shape: vec![HIDDEN_SIZE as u64],
        });
        kernels.push(PrefillKernel {
            recipe: PrefillRecipe::Matmul,
            weight_tensor: Some("token_embd.weight"),
            shape: vec![VOCAB_SIZE as u64],
        });

        PrefillProgram {
            prompt_tokens: PROMPT_TOKENS,
            layer_count: LAYER_COUNT,
            hidden_size: HIDDEN_SIZE,
            head_count: HEAD_COUNT,
            kv_head_count: KV_HEAD_COUNT,
            head_dim: HEAD_DIM,
            ffn_size: FFN_SIZE,
            vocab_size: VOCAB_SIZE,
            attention_scale: ATTENTION_SCALE,
            rms_eps: RMS_EPS,
            kernels,
        }
    }

    /// Build the per-layer kernel sequence for layer `il` (13 kernels per
    /// decoder layer).
    #[must_use]
    pub fn layer_kernels(il: usize) -> Vec<PrefillKernel> {
        const LAYER_KERNEL_COUNT: usize = 13;
        let program = Self::build();
        let start = 1 + il * LAYER_KERNEL_COUNT;
        program.kernels[start..start + LAYER_KERNEL_COUNT].to_vec()
    }
}

impl PrefillProgram {
    /// The kernel count (1 gather + 32×11 + output_norm + tied head = 356).
    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// The weight tensors the program consumes (the runtime weight values
    /// the Q1-default driver supplies at session creation), in deterministic
    /// first-use order — 1 (token_embd) + 32×9 (per-layer) + 1 (output_norm)
    /// = 290, the same 290 tensors the CPU oracle materializes.
    #[must_use]
    pub fn weight_tensors(&self) -> Vec<&'static str> {
        let mut tensors = vec!["token_embd.weight"];
        for il in 0..self.layer_count {
            for base in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                tensors.push(Box::leak(format!("blk.{il}.{base}").into_boxed_str()));
            }
        }
        tensors.push("output_norm.weight");
        tensors
    }
}

// ---------------------------------------------------------------------------
// Q2 GPU-vs-oracle comparison harness
// ---------------------------------------------------------------------------

/// The Q2 comparison verdict for the prompt-final logits (position 0).
///
/// Divergence field per `gi0-numeric-contract.md` §4.5: `none` when every
/// Q2 threshold passes, otherwise the first-diverging position + the named
/// failing thresholds + the max deviation. Never text-level.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillComparison {
    /// §Q2: the GPU's EOG-excluded top-1 token id.
    pub gpu_top1: i64,
    /// §Q2: the golden's EOG-excluded top-1 token id.
    pub golden_top1: i64,
    /// §Q2: top-1 exact over non-EOG {0, 2} at the prompt end.
    pub top1_matches: bool,
    /// §Q2: max per-element |gpu − golden| over the full vocab.
    pub max_delta: f32,
    /// §Q2: per-element numeric row (matmul `atol 1e-5 / rtol 1e-5`)
    /// `|gpu − golden| ≤ atol + rtol·|golden|` for every element.
    pub numeric_matches: bool,
    /// Finite gate: every GPU and golden logit finite.
    pub all_finite: bool,
    /// The named failing threshold(s) (§Q2 / §4.5); empty when every
    /// threshold passes.
    pub failing_thresholds: Vec<FailingThreshold>,
    /// Whether every Q2 threshold passes.
    pub ok: bool,
}

impl PrefillComparison {
    /// The §4.5 divergence field: `none` or the first-diverging position +
    /// details.
    #[must_use]
    pub fn divergence_field(&self) -> String {
        if self.ok {
            "none".to_owned()
        } else {
            format!(
                "position 0 (prompt end): GPU top-1 {} vs golden {}; failing [{}]; max delta {:.3e}",
                self.gpu_top1,
                self.golden_top1,
                self.failing_thresholds
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
                self.max_delta,
            )
        }
    }
}

/// Compare GPU prompt-final logits against the golden logits under the Q2
/// thresholds.
///
/// - `gpu_logits`: the device's full-vocab raw logits `[49152]`;
/// - `golden_logits`: the committed GI2 logits golden's raw logits
///   (`testdata/gi2-3-logits-golden/logits-pos0.json` `raw_logits`).
///
/// # Errors
///
/// A length mismatch or non-finite golden input fails closed.
pub fn compare_gpu_logits(
    gpu_logits: &[f32],
    golden_logits: &[f32],
) -> Result<PrefillComparison, &'static str> {
    if gpu_logits.len() != VOCAB_SIZE || golden_logits.len() != VOCAB_SIZE {
        return Err("Q2 comparison needs 49152-wide logits vectors");
    }
    let gpu_top1 = top1_non_eog(gpu_logits, &EOG_TOKENS);
    let golden_top1 = top1_non_eog(golden_logits, &EOG_TOKENS);
    let top1_matches = gpu_top1 == golden_top1;

    let mut max_delta = 0.0f32;
    let mut numeric_matches = true;
    let mut all_finite = true;
    for (gpu, golden) in gpu_logits.iter().zip(golden_logits.iter()) {
        if !gpu.is_finite() || !golden.is_finite() {
            all_finite = false;
            numeric_matches = false;
            continue;
        }
        let delta = (gpu - golden).abs();
        if delta > max_delta {
            max_delta = delta;
        }
        // The matmul row (the tied-head projection produced the logits).
        let bound = Q2_ATOL + Q2_RTOL * golden.abs();
        if delta > bound {
            numeric_matches = false;
        }
    }

    let mut failing_thresholds = Vec::new();
    if !top1_matches {
        failing_thresholds.push(FailingThreshold::Top1);
    }
    if !numeric_matches {
        failing_thresholds.push(FailingThreshold::Band);
    }
    if !all_finite {
        failing_thresholds.push(FailingThreshold::Finite);
    }
    let ok = failing_thresholds.is_empty();

    Ok(PrefillComparison {
        gpu_top1,
        golden_top1,
        top1_matches,
        max_delta,
        numeric_matches,
        all_finite,
        failing_thresholds,
        ok,
    })
}

// ---------------------------------------------------------------------------
// Receipts (S6 prefill-regime fields + execution facts + timing)
// ---------------------------------------------------------------------------

/// The executable regime of a GI3 receipt — always `prefill` (GI4 owns
/// decode; a decode number is never mislabeled here, CTO S6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutableRegime {
    /// The prefill regime (prompt-final logits for the correctness prompt).
    Prefill,
}

impl ExecutableRegime {
    /// The regime label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
        }
    }
}

/// The S6 prefill-regime fields — recorded separately (CTO S6), never
/// conflated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillRegimeFields {
    /// Shape class of the executed program (the prefill workload class).
    pub shape_class: String,
    /// Weight representation (the declared f32 conversion per the repack
    /// plan — never presented as direct GGUF quantized execution).
    pub representation: String,
    /// The algorithm family (declared f32 conversion reusing the GI2-1
    /// dequant semantics; attention GQA consecutive-triples grouping).
    pub algorithm: String,
    /// Workspace facts (buffers/scratch the prefill program needs).
    pub workspace: String,
    /// Evidence (the committed comparison record + receipts paths).
    pub evidence: String,
}

/// The prefill execution receipt: device facts + S6 regime fields + the
/// repack/conversion/upload timing (CTO S11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillReceipt {
    /// Executable regime — always `prefill`.
    pub regime: ExecutableRegime,
    /// Device execution facts (from the host execution receipt).
    pub transfers: u32,
    pub allocations: u32,
    pub launches: u32,
    pub syncs: u32,
    pub observations: u32,
    /// S6 prefill-regime fields.
    pub regime_fields: PrefillRegimeFields,
    /// Repack/conversion wall time (µs).
    pub repack_conversion_us: u64,
    /// Module preparation wall time (µs).
    pub module_prep_us: u64,
    /// Persistent weight upload wall time (µs).
    pub persistent_upload_us: u64,
    /// First invocation wall time (µs).
    pub first_invocation_us: u64,
    /// Capture wall time (µs).
    pub capture_us: u64,
}

/// Serialize a prefill receipt to the committed JSON schema
/// (`gi3-prefill-receipt-v1`).
#[must_use]
pub fn receipt_to_json(receipt: &PrefillReceipt) -> String {
    let mut root = BTreeMap::new();
    root.insert("schema".to_string(), Valor::from("gi3-prefill-receipt-v1"));
    root.insert("regime".to_string(), Valor::from(receipt.regime.label()));
    root.insert(
        "transfers".to_string(),
        Valor::from(receipt.transfers as i64),
    );
    root.insert(
        "allocations".to_string(),
        Valor::from(receipt.allocations as i64),
    );
    root.insert("launches".to_string(), Valor::from(receipt.launches as i64));
    root.insert("syncs".to_string(), Valor::from(receipt.syncs as i64));
    root.insert(
        "observations".to_string(),
        Valor::from(receipt.observations as i64),
    );
    let mut regime = BTreeMap::new();
    regime.insert(
        "shape_class".to_string(),
        Valor::from(receipt.regime_fields.shape_class.clone()),
    );
    regime.insert(
        "representation".to_string(),
        Valor::from(receipt.regime_fields.representation.clone()),
    );
    regime.insert(
        "algorithm".to_string(),
        Valor::from(receipt.regime_fields.algorithm.clone()),
    );
    regime.insert(
        "workspace".to_string(),
        Valor::from(receipt.regime_fields.workspace.clone()),
    );
    regime.insert(
        "evidence".to_string(),
        Valor::from(receipt.regime_fields.evidence.clone()),
    );
    root.insert("s6_regime".to_string(), regime.into());
    root.insert(
        "repack_conversion_us".to_string(),
        Valor::from(receipt.repack_conversion_us as i64),
    );
    root.insert(
        "module_prep_us".to_string(),
        Valor::from(receipt.module_prep_us as i64),
    );
    root.insert(
        "persistent_upload_us".to_string(),
        Valor::from(receipt.persistent_upload_us as i64),
    );
    root.insert(
        "first_invocation_us".to_string(),
        Valor::from(receipt.first_invocation_us as i64),
    );
    root.insert(
        "capture_us".to_string(),
        Valor::from(receipt.capture_us as i64),
    );
    let json = Json::from_object(root).expect("prefill receipt JSON is valid");
    format!("{}\n", json.to_wire())
}
