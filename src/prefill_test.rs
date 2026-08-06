//! GI3-5 — prefill program builder + Q2 comparison harness tests.
//!
//! Families:
//! 1. **Program builder**: the prefill program description for the pinned row
//!    — kernel count (1 gather + 32×11 + output_norm + tied head = 355), the
//!    ordered recipe sequence, the per-layer tensor names, and the 290-weight
//!    enumeration (the same 290 tensors the CPU oracle materializes).
//! 2. **Q2 comparison harness**: GPU logits vs golden — top-1 exact over
//!    non-EOG {0,2}, the per-element matmul-row thresholds
//!    (`atol 1e-5 / rtol 1e-5`), the finite gate, and the §4.5 divergence
//!    field. The identity case (GPU == golden) PASSES; a flipped top-1, an
//!    out-of-band delta, and a non-finite value each FAIL with the named
//!    threshold — never weakened.
//! 3. **The committed golden** (`testdata/gi2-3-logits-golden/`): a GPU
//!    result equal to the golden raw logits passes the full Q2 gate, and the
//!    golden's declared top-1 (30) is reproduced by the harness's EOG-
//!    excluded argmax.
//! 4. **Receipts**: the S6 prefill-regime fields + execution facts + timing
//!    serialize/round-trip under `gi3-prefill-receipt-v1`.

use crate::cpu_oracle::{FailingThreshold, EOG_TOKENS, VOCAB_SIZE};
use crate::json::Json;
use crate::prefill::*;
use crate::valor::Valor;

fn goldens_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/gi2-3-logits-golden")
}

/// Parse the golden's `f32_le_hex` byte stream into f32 values.
fn hex_f32s(hex: &str) -> Vec<f32> {
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
        .collect();
    assert_eq!(bytes.len(), VOCAB_SIZE * 4);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn field<'a>(v: &'a Valor, key: &str) -> &'a Valor {
    let Valor::Tabula(fields) = v else {
        panic!("expected JSON object")
    };
    fields
        .get(key)
        .unwrap_or_else(|| panic!("missing JSON field {key:?}"))
}

fn text(v: &Valor) -> &str {
    let Valor::Textus(s) = v else {
        panic!("expected JSON string")
    };
    s
}

/// Load the committed GI2-3 logits golden's raw logits (position 0).
fn golden_raw_logits() -> Vec<f32> {
    let wire = std::fs::read_to_string(goldens_dir().join("logits-pos0.json"))
        .expect("committed logits golden must exist");
    let root = Json::parse(&wire)
        .expect("golden parses")
        .as_valor()
        .clone();
    let raw = field(field(&root, "raw_logits"), "f32_le_hex");
    hex_f32s(text(raw))
}

// ---------------------------------------------------------------------------
// 1. Program builder
// ---------------------------------------------------------------------------

#[test]
fn program_builder_assembles_the_pinned_row_prefill() {
    let program = PrefillProgramBuilder::build();

    // Row facts (model contract v1.0.0).
    assert_eq!(program.prompt_tokens, PROMPT_TOKENS);
    assert_eq!(program.layer_count, 32);
    assert_eq!(program.hidden_size, 960);
    assert_eq!(program.head_count, 15);
    assert_eq!(program.kv_head_count, 5);
    assert_eq!(program.head_dim, 64);
    assert_eq!(program.ffn_size, 2560);
    assert_eq!(program.vocab_size, 49152);
    assert_eq!(program.attention_scale, 0.125);
    assert_eq!(program.rms_eps, 1e-5);

    // Kernel sequence: 1 gather + 32×13 + output_norm + tied head = 419.
    assert_eq!(program.kernel_count(), 419);
    assert_eq!(program.kernels[0].recipe, PrefillRecipe::EmbeddingGather);
    assert_eq!(program.kernels[0].weight_tensor, Some("token_embd.weight"));
    assert_eq!(program.kernels[0].shape, vec![9, 960]);

    // Layer 0's 13-kernel sequence in order.
    let layer0 = PrefillProgramBuilder::layer_kernels(0);
    assert_eq!(layer0.len(), 13);
    let recipes: Vec<PrefillRecipe> = layer0.iter().map(|k| k.recipe).collect();
    assert_eq!(
        recipes,
        vec![
            PrefillRecipe::RmsNorm,
            PrefillRecipe::Matmul,
            PrefillRecipe::Matmul,
            PrefillRecipe::Matmul,
            PrefillRecipe::Rope,
            PrefillRecipe::Rope,
            PrefillRecipe::CausalMaskedSoftmax,
            PrefillRecipe::Matmul,
            PrefillRecipe::ResidualAdd,
            PrefillRecipe::RmsNorm,
            PrefillRecipe::Swiglu,
            PrefillRecipe::Matmul,
            PrefillRecipe::ResidualAdd,
        ]
    );
    assert_eq!(layer0[0].weight_tensor, Some("blk.0.attn_norm.weight"));
    assert_eq!(layer0[1].weight_tensor, Some("blk.0.attn_q.weight"));
    assert_eq!(layer0[2].weight_tensor, Some("blk.0.attn_k.weight"));
    assert_eq!(layer0[3].weight_tensor, Some("blk.0.attn_v.weight"));
    assert_eq!(layer0[7].weight_tensor, Some("blk.0.attn_output.weight"));
    assert_eq!(layer0[9].weight_tensor, Some("blk.0.ffn_norm.weight"));

    // Weight-free kernels carry no weight tensor.
    assert_eq!(layer0[4].weight_tensor, None, "rope has no weight tensor");
    assert_eq!(
        layer0[6].weight_tensor, None,
        "causal softmax has no weight"
    );
    assert_eq!(layer0[8].weight_tensor, None, "residual has no weight");
    assert_eq!(
        layer0[10].weight_tensor, None,
        "swiglu has no single weight"
    );

    // Tail: output_norm + tied-head projection.
    let n = program.kernel_count();
    assert_eq!(program.kernels[n - 2].recipe, PrefillRecipe::RmsNorm);
    assert_eq!(
        program.kernels[n - 2].weight_tensor,
        Some("output_norm.weight")
    );
    assert_eq!(program.kernels[n - 1].recipe, PrefillRecipe::Matmul);
    assert_eq!(
        program.kernels[n - 1].weight_tensor,
        Some("token_embd.weight")
    );
    assert_eq!(program.kernels[n - 1].shape, vec![49152]);
}

#[test]
fn program_weight_tensors_are_the_290_admitted_tensors() {
    let program = PrefillProgramBuilder::build();
    let weights = program.weight_tensors();
    // 1 (token_embd) + 32×9 (per-layer) + 1 (output_norm) — the same 290 the
    // CPU oracle materializes.
    assert_eq!(weights.len(), 290);
    assert_eq!(weights[0], "token_embd.weight");
    assert!(weights.contains(&"blk.0.attn_norm.weight"));
    assert!(weights.contains(&"blk.0.ffn_gate.weight"));
    assert!(weights.contains(&"blk.0.ffn_up.weight"));
    assert!(weights.contains(&"blk.31.ffn_down.weight"));
    assert_eq!(weights[289], "output_norm.weight");
    // No duplicates.
    let mut sorted = weights.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 290, "the weight set must be distinct");
}

// ---------------------------------------------------------------------------
// 2. Q2 comparison harness
// ---------------------------------------------------------------------------

#[test]
fn q2_identity_logits_pass_and_divergence_field_is_none() {
    let golden = golden_raw_logits();
    let comparison = compare_gpu_logits(&golden, &golden).expect("compare");
    assert!(comparison.top1_matches, "top-1 exact for identical logits");
    assert!(
        comparison.numeric_matches,
        "numeric rows pass for identical logits"
    );
    assert!(comparison.all_finite, "finite gate");
    assert!(comparison.ok, "every Q2 threshold passes");
    assert_eq!(comparison.failing_thresholds, vec![]);
    assert_eq!(comparison.divergence_field(), "none");
    assert_eq!(comparison.max_delta, 0.0);
    assert_eq!(
        comparison.golden_top1, 30,
        "the golden's EOG-excluded top-1 is token 30"
    );
}

#[test]
fn q2_small_in_band_delta_passes_but_flipped_top1_fails() {
    let golden = golden_raw_logits();
    // In-band perturbation (below the matmul row bound) still passes.
    let small: Vec<f32> = golden.iter().map(|v| v + 1e-7).collect();
    let comparison = compare_gpu_logits(&small, &golden).expect("compare");
    assert!(comparison.ok, "an in-band perturbation passes");

    // Flip the top-1: raise a non-EOG token above the golden's argmax.
    let mut flipped = golden.clone();
    let winner = 1234;
    flipped[winner] = flipped[golden_top1_idx(&golden)] + 10.0;
    let comparison = compare_gpu_logits(&flipped, &golden).expect("compare");
    assert!(!comparison.top1_matches, "top-1 must be exact over non-EOG");
    assert_eq!(comparison.gpu_top1, winner as i64);
    assert!(
        comparison
            .failing_thresholds
            .contains(&FailingThreshold::Top1),
        "the flipped top-1 names the top-1 threshold: {:?}",
        comparison.failing_thresholds
    );
    assert!(!comparison.ok);
    assert!(
        comparison.divergence_field().starts_with("position 0"),
        "the divergence field names the position: {}",
        comparison.divergence_field()
    );
}

#[test]
fn q2_out_of_band_delta_fails_the_numeric_row() {
    let golden = golden_raw_logits();
    // Out-of-band perturbation on one element (> atol + rtol·|b|).
    let mut perturbed = golden.clone();
    let idx = 100;
    perturbed[idx] = golden[idx] + 1e-3;
    let comparison = compare_gpu_logits(&perturbed, &golden).expect("compare");
    assert!(
        !comparison.numeric_matches,
        "an out-of-band delta fails the numeric row"
    );
    assert!(
        comparison
            .failing_thresholds
            .contains(&FailingThreshold::Band),
        "the failing threshold must name the band/numeric row: {:?}",
        comparison.failing_thresholds
    );
    assert!(!comparison.ok);
}

#[test]
fn q2_non_finite_gpu_logit_fails_the_finite_gate() {
    let golden = golden_raw_logits();
    let mut nan = golden.clone();
    nan[500] = f32::NAN;
    let comparison = compare_gpu_logits(&nan, &golden).expect("compare");
    assert!(
        !comparison.all_finite,
        "a NaN GPU logit fails the finite gate"
    );
    assert!(comparison
        .failing_thresholds
        .contains(&FailingThreshold::Finite));
    assert!(!comparison.ok);
}

#[test]
fn q2_length_mismatch_fails_closed() {
    assert!(compare_gpu_logits(&[0.0; 10], &golden_raw_logits()).is_err());
    assert!(compare_gpu_logits(&golden_raw_logits(), &[0.0; 10]).is_err());
}

#[test]
fn gpu_top1_excludes_eog_tokens() {
    // The EOG-exclusion surface: a GPU result with an EOG token at the raw
    // argmax must still resolve the non-EOG top-1 (contract §2.1).
    let golden = golden_raw_logits();
    let mut eog_at_top = golden.clone();
    eog_at_top[2] = eog_at_top[golden_top1_idx(&golden)] + 10.0; // <|im_end|> wins raw
    let comparison = compare_gpu_logits(&eog_at_top, &golden).expect("compare");
    assert!(
        !EOG_TOKENS.contains(&comparison.gpu_top1),
        "the GPU top-1 must never be an EOG token (got {})",
        comparison.gpu_top1
    );
    assert_eq!(
        comparison.gpu_top1, 30,
        "the non-EOG argmax is still token 30"
    );
}

/// The index of the golden's raw argmax (the EOG-excluded top-1 = 30).
fn golden_top1_idx(logits: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if !EOG_TOKENS.contains(&(i as i64)) && v > logits[best] {
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// 3. Receipts (S6 regime fields + execution facts + timing)
// ---------------------------------------------------------------------------

#[test]
fn prefill_receipt_serializes_with_s6_regime_fields() {
    let receipt = PrefillReceipt {
        regime: ExecutableRegime::Prefill,
        transfers: 1,
        allocations: 2,
        launches: 355,
        syncs: 356,
        observations: 1,
        regime_fields: PrefillRegimeFields {
            shape_class: "prefill-9t".to_owned(),
            representation:
                "declared f32 conversion (GI2-1 dequant; never direct GGUF quantized execution)"
                    .to_owned(),
            algorithm: "GQA consecutive-triples; CausalMaskedSoftmax; declared-f32 matmul"
                .to_owned(),
            workspace: "29 program buffers + 1 output".to_owned(),
            evidence: "gi3-prefill-comparison.json + gi3-prefill-receipts.md".to_owned(),
        },
        repack_conversion_us: 1_234_567,
        module_prep_us: 2_345,
        persistent_upload_us: 345_678,
        first_invocation_us: 12_345,
        capture_us: 4_567,
    };
    let wire = receipt_to_json(&receipt);
    let root = Json::parse(&wire)
        .expect("receipt parses")
        .as_valor()
        .clone();
    assert_eq!(text(field(&root, "schema")), "gi3-prefill-receipt-v1");
    assert_eq!(text(field(&root, "regime")), "prefill");
    assert_eq!(int(field(&root, "launches")), 355);
    assert_eq!(int(field(&root, "observations")), 1);
    let regime = field(&root, "s6_regime");
    assert_eq!(text(field(regime, "shape_class")), "prefill-9t");
    assert_eq!(
        text(field(regime, "representation")),
        "declared f32 conversion (GI2-1 dequant; never direct GGUF quantized execution)"
    );
    assert_eq!(int(field(&root, "repack_conversion_us")), 1_234_567);
    assert_eq!(int(field(&root, "persistent_upload_us")), 345_678);
}

fn int(v: &Valor) -> i64 {
    let Valor::Numerus(n) = v else {
        panic!("expected JSON integer")
    };
    *n
}
