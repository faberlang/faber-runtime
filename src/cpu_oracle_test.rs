//! GI2-3 — CPU one-position logits oracle + teacher-forced numeric-contract
//! window tests.
//!
//! Families:
//! 1. **f32 → f16 (round-to-nearest-even)** — the KV-cache fidelity
//!    conversion (comparator `--cache-type-k/v f16`): known values, exact
//!    tie-to-even behavior, and the exhaustive round-trip over all 65,536
//!    half patterns (`f32_to_f16_rn(half_to_f32(h)) == h`).
//! 2. **Fail-closed oracle construction/forward**: empty tokens, out-of-range
//!    token ids, gapped/forged view (GI1-4 residual), and the materialization
//!    receipts (290).
//! 3. **Forward determinism + numeric-faithful incremental runner**: two
//!    independent `forward_one` runs are byte-identical, and
//!    `ForwardRun::position_logits` is **byte-identical** to `forward_one`
//!    for the same context (the cache changes no numerics — GI2 non-goal).
//! 4. **The teacher-forced 17-window numeric-contract comparison**
//!    (two contract versions — decision `41da94f3`): prompt end (position 0)
//!    + positions 1..16 fed the pinned `correctness-top1.json` trace
//!    teacher-forced, against the committed comparator logp reference window
//!    (17×49152, hash-pinned; captured per the GI0-4 probe protocol).
//!    **v1.0.0** (1e-5 band) is the honest-failure record — top-1/top-k/finite
//!    hold at all 17 positions, the band fails at every position, and the
//!    divergence is recorded, never weakened. **v2.0.0**
//!    (`Delta_comparator_metal` = 2.5e-2 envelope + hard gates, the
//!    operator-approved closeout contract) is MET at all 17 positions —
//!    divergence field = `none`, with the calibration maximum + headroom
//!    recorded. The EOG-exclusion scenario at fixture position 1 (raw
//!    argmax=2/EOS, trace=198) is asserted as a fidelity probe.
//! 5. **GI3 logits golden** (exit gate bullet 5): position-0 prompt-end raw
//!    logits + normalized logp, byte-stable across two independent runs,
//!    hash-accounted, byte-identical to the committed fixture under
//!    `testdata/gi2-3-logits-golden/`.
//!
//! Model-dependent tests follow the `tensor_view_test` / `dequant_test`
//! convention: skipped (with a loud note) when the pinned row is absent.

use crate::cpu_oracle::*;
use crate::dequant::half_to_f32;
use crate::gguf::{admit_gguf, hex, sha256, PINNED_SHA256_HEX};
use crate::json::Json;
use crate::tensor_view::TensorView;
use crate::valor::Valor;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Pinned-row constants
// ---------------------------------------------------------------------------

/// Machine-local pinned row (skipped when absent — same convention as
/// `tensor_view_test` / `dequant_test`).
const PINNED_MODEL_PATH: &str = "/Users/ianzepp/ai/models/SmolLM2-360M-Instruct-Q4_K_M.gguf";

/// The committed comparator logp reference window (17×49152 f32 LE,
/// position-major, vocab-id indexed) + capture receipt — under the Q1
/// default path (sibling radix docs evidence; `gi2-3-capture-receipt.md`).
/// Relative to this crate root, like the dequant goldens.
const LOGP_REFERENCE_REL_PATH: &str =
    "../radix/docs/factory/gpu-inference-gguf/evidence/gi2-3-logp-reference.bin";

/// Hash pin of the committed logp reference fixture (documented in the
/// capture receipt; a fixture change breaks this test loudly).
const LOGP_REFERENCE_SHA256_HEX: &str =
    "d105b49bde0ee5070020bb3a545578ba1195900937f44446893ad3bbaabad8ec";

/// Committed GI3 logits golden (Q3 default: faber-runtime crate-local
/// testdata, `include_bytes!`-able).
const GOLDENS_REL_DIR: &str = "testdata/gi2-3-logits-golden";
/// Golden generator + fixture schema version (byte-stability contract).
const GENERATOR_VERSION: &str = "gi2-3-logits-golden-v1";
const FIXTURE_SCHEMA: &str = "gi2-3-logits-golden-v1";
const MANIFEST_SCHEMA: &str = "gi2-3-logits-golden-manifest-v1";

/// The pinned correctness prompt (gi0-workloads §3.1): 9 BOS-free tokens.
const PROMPT_TOKENS: [i64; 9] = [504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767];

/// The pinned expected greedy trace head — `correctness-top1.json`
/// `trace_tokens[0..17]` (256-token trace; head `[30, 198, 198, 504, …]`,
/// FC5). `TRACE_TOKENS[i]` is the expected greedy token at window position
/// `i`; positions 1..16 are fed teacher-forced (`TRACE_TOKENS[i-1]`).
const TRACE_TOKENS: [i64; 17] = [
    30, 198, 198, 504, 808, 6330, 314, 253, 2232, 4814, 282, 1027, 28, 979, 260, 1796, 6330,
];

// ---------------------------------------------------------------------------
// Model loading helpers
// ---------------------------------------------------------------------------

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

/// A process-lifetime shared oracle — the dequant materialization (~7s in an
/// unoptimized build) happens once for all model-dependent tests.
fn shared_oracle() -> Option<&'static CpuOracle> {
    static ORACLE: std::sync::OnceLock<Option<CpuOracle>> = std::sync::OnceLock::new();
    ORACLE
        .get_or_init(|| {
            if !std::path::Path::new(PINNED_MODEL_PATH).exists() {
                eprintln!("SKIP: pinned model not present at {PINNED_MODEL_PATH}");
                return None;
            }
            let bytes = std::fs::read(PINNED_MODEL_PATH).expect("pinned model read");
            let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            let view = build_pinned_view(bytes);
            Some(CpuOracle::build(&view).expect("oracle build"))
        })
        .as_ref()
}

/// Load the committed comparator logp reference window (17×49152 f32 LE,
/// position-major, vocab-id indexed). Skips loudly when the sibling radix
/// layout changed (a broken path must never silently skip the reference
/// comparison).
fn load_logp_reference() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(LOGP_REFERENCE_REL_PATH);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            eprintln!(
                "SKIP: comparator logp reference not readable at {} ({err})",
                path.display()
            );
            None
        }
    }
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

// ---------------------------------------------------------------------------
// Valor walking helpers (same pattern as decoder_ops_test)
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

// ---------------------------------------------------------------------------
// 1. f32 → f16 round-to-nearest-even (KV-cache fidelity conversion)
// ---------------------------------------------------------------------------

#[test]
fn f32_to_f16_rn_matches_known_values() {
    // Exact representable values map exactly.
    assert_eq!(f32_to_f16_rn(0.0), 0x0000);
    assert_eq!(f32_to_f16_rn(-0.0), 0x8000);
    assert_eq!(f32_to_f16_rn(1.0), 0x3C00);
    assert_eq!(f32_to_f16_rn(-1.0), 0xBC00);
    assert_eq!(f32_to_f16_rn(0.5), 0x3800);
    assert_eq!(f32_to_f16_rn(2.0), 0x4000);
    assert_eq!(f32_to_f16_rn(6.103515625e-05), 0x0400); // 2⁻¹⁴
    assert_eq!(f32_to_f16_rn(65504.0), 0x7BFF); // f16 max
    assert_eq!(f32_to_f16_rn(-65504.0), 0xFBFF);
    assert_eq!(f32_to_f16_rn(65520.0), 0x7C00); // overflow → +Inf
    assert_eq!(f32_to_f16_rn(f32::INFINITY), 0x7C00);
    assert_eq!(f32_to_f16_rn(f32::NEG_INFINITY), 0xFC00);
    // π (1.1001001...b × 2¹) → 0x4248.
    assert_eq!(f32_to_f16_rn(std::f32::consts::PI), 0x4248);
    // A value with a full 11-bit fraction (rounding of the 13 discarded bits).
    assert_eq!(f32_to_f16_rn(1.0009765625), 0x3C01); // 1 + 2⁻¹⁰
}

#[test]
fn f32_to_f16_rn_is_round_to_nearest_even() {
    // Tie-to-even: 1 + 2⁻¹¹ is exactly halfway between 1.0 (0x3C00, even
    // significand) and 1 + 2⁻¹⁰ (0x3C01, odd) → rounds to the even one.
    let tie_low = 1.0f32 + 2.0f32.powi(-11);
    assert_eq!(f32_to_f16_rn(tie_low), 0x3C00);
    // Halfway between 0x3C01 (odd) and 0x3C02 (even) → even (0x3C02).
    let tie_high = 1.0f32 + 2.0f32.powi(-10) + 2.0f32.powi(-11);
    assert_eq!(f32_to_f16_rn(tie_high), 0x3C02);
    // Strictly below the low tie → down; strictly above → up.
    assert_eq!(f32_to_f16_rn(1.0f32 + 2.0f32.powi(-12)), 0x3C00);
    assert_eq!(
        f32_to_f16_rn(1.0f32 + 2.0f32.powi(-10) + 2.0f32.powi(-12)),
        0x3C01
    );
    // Subnormal tie: 2⁻²⁵ is halfway between 0 and the smallest subnormal
    // 2⁻²⁴ → rounds to 0 (even).
    assert_eq!(f32_to_f16_rn(2.0f32.powi(-25)), 0x0000);
    // Subnormal tie between mantissa 1 (odd) and 2 (even): 1.5·2⁻²⁴ → even.
    assert_eq!(f32_to_f16_rn(3.0f32 * 2.0f32.powi(-25)), 0x0002);
    // Just above 2⁻²⁵ → smallest subnormal.
    assert_eq!(f32_to_f16_rn(2.0f32.powi(-25) + 2.0f32.powi(-26)), 0x0001);
}

#[test]
fn f32_to_f16_rn_round_trips_all_half_patterns() {
    // Exhaustive: converting an f16's exact f32 value back to f16 (RN) is
    // the identity — the conversion never rounds an exactly-representable
    // value away.
    for h in 0u16..=u16::MAX {
        let x = half_to_f32(h);
        let back = f32_to_f16_rn(x);
        assert_eq!(back, h, "round-trip broken for half pattern {h:#06x} = {x}");
    }
}

// ---------------------------------------------------------------------------
// 2. Fail-closed construction + forward
// ---------------------------------------------------------------------------

#[test]
fn oracle_refuses_empty_tokens_and_out_of_range_ids() {
    // No model needed: the guards run before any byte is touched.
    let oracle = CpuOracle::degenerate_for_test();
    // Empty token sequence.
    let err = oracle.forward_one(&[]).expect_err("empty tokens");
    assert!(matches!(err, OracleError::EmptyTokens));
    // Out-of-range token id (the pinned vocab is [0, 49152)).
    let err = oracle
        .forward_one(&[0, VOCAB_SIZE as i64])
        .expect_err("out of range");
    assert!(matches!(
        err,
        OracleError::TokenOutOfRange { token: 49_152, .. }
    ));
    let err = oracle.forward_one(&[-1]).expect_err("negative id");
    assert!(matches!(err, OracleError::TokenOutOfRange { .. }));
}

#[test]
fn oracle_build_materializes_all_tensors_with_receipts() {
    let Some(oracle) = shared_oracle() else {
        return;
    };
    assert_eq!(oracle.model_sha256_hex(), PINNED_SHA256_HEX);
    // 1 (token_embd) + 32×9 (per-layer) + 1 (output_norm) = 290 receipts.
    assert_eq!(
        oracle.receipts().len(),
        290,
        "one receipt per materialized tensor"
    );
    let sources: std::collections::BTreeSet<&str> = oracle
        .receipts()
        .iter()
        .map(|r| r.source_tensor.as_str())
        .collect();
    assert_eq!(sources.len(), 290, "receipts must name distinct tensors");
    assert!(sources.contains("token_embd.weight"));
    assert!(sources.contains("blk.0.attn_norm.weight"));
    assert!(sources.contains("blk.31.ffn_down.weight"));
    assert!(sources.contains("output_norm.weight"));
}

// ---------------------------------------------------------------------------
// 3. Forward determinism + numeric-faithful incremental runner
// ---------------------------------------------------------------------------

#[test]
fn forward_one_is_deterministic_and_incremental_run_is_byte_identical() {
    let Some(oracle) = shared_oracle() else {
        return;
    };

    // Two independent full forwards (position 0, the 9 pinned prompt tokens)
    // are byte-identical.
    let a = oracle.forward_one(&PROMPT_TOKENS).expect("forward_one");
    let b = oracle.forward_one(&PROMPT_TOKENS).expect("forward_one");
    assert_eq!(a.len(), VOCAB_SIZE);
    assert_eq!(a, b, "forward_one must be byte-deterministic");

    // The incremental runner is byte-identical to the full forward for the
    // same context — the K/V cache changes no numerics. Checked at position
    // 0 (full 9-token batch, n_q = n_kv = 9) and on a small 2-token context
    // (the batch multi-row vs single-row attention share the same arithmetic
    // for the last row; every window position's row is computed by exactly
    // that single-row path — exercised across all 17 positions by the
    // comparison test).
    let run0 = ForwardRun::new(oracle, &PROMPT_TOKENS).expect("run");
    assert_eq!(run0.len(), PROMPT_TOKENS.len());
    let p0 = run0.position_logits().expect("position 0");
    assert_eq!(
        p0, a,
        "position 0: ForwardRun must equal forward_one bit-for-bit"
    );

    let small = [PROMPT_TOKENS[0], PROMPT_TOKENS[1]];
    let full_small = oracle.forward_one(&small).expect("small full forward");
    let run_small = ForwardRun::new(oracle, &small).expect("small run");
    let incr_small = run_small.position_logits().expect("small position logits");
    assert_eq!(
        incr_small, full_small,
        "small context: ForwardRun must equal forward_one bit-for-bit"
    );
    assert_eq!(incr_small.len(), VOCAB_SIZE);
}

#[test]
fn log_softmax_normalizes_and_finite_gate_rejects_bad_input() {
    // log-softmax of [1,2,3] = log(softmax): S = 1 + e⁻¹ + e⁻²; logp[i] =
    // (x[i] − 3) − ln(S).
    let lp = log_softmax(&[1.0, 2.0, 3.0]).expect("log_softmax");
    let s = 1.0f32 + (-1.0f32).exp() + (-2.0f32).exp();
    let ln_s = s.ln();
    let expected = [(-2.0f32) - ln_s, (-1.0f32) - ln_s, -ln_s];
    for (i, (g, e)) in lp.iter().zip(expected).enumerate() {
        assert!((g - e).abs() < 1e-5, "logp[{i}]: got {g}, expected {e}");
    }
    // Normalization: Σ exp(logp) == 1.
    let mut sum = 0.0f32;
    for &v in &lp {
        sum += v.exp();
    }
    assert!((sum - 1.0).abs() < 1e-6, "log-softmax must be normalized");

    // Empty / non-finite inputs fail the finite gate.
    assert!(log_softmax(&[]).is_err());
    assert!(log_softmax(&[1.0, f32::NAN]).is_err());
    assert!(log_softmax(&[f32::INFINITY, 1.0]).is_err());

    // top1_non_eog excludes {0, 2} and reproduces the pinned EOG scenario.
    let mut logits = vec![0.0f32; 16];
    logits[2] = 10.0; // EOS would win the raw argmax...
    logits[7] = 9.5;
    assert_eq!(top1_non_eog(&logits, &EOG_TOKENS), 7);
    assert_eq!(topk_ids(&logits, 5)[0], 2, "top-k keeps EOG (raw surface)");
}

// ---------------------------------------------------------------------------
// 4. Teacher-forced 17-window numeric-contract comparison
// ---------------------------------------------------------------------------

#[test]
fn teacher_forced_window_meets_numeric_contract_v1_0_0() {
    let Some(oracle) = shared_oracle() else {
        return;
    };
    let Some(fixture) = load_logp_reference() else {
        return;
    };
    // Hash pin: the committed fixture is the captured reference window.
    assert_eq!(
        hex(&sha256(&fixture)),
        LOGP_REFERENCE_SHA256_HEX,
        "logp reference fixture must be hash-pinned (capture receipt documents it)"
    );
    // 17 positions × 49152 vocab f32 LE.
    assert_eq!(
        fixture.len(),
        WINDOW_POSITIONS * VOCAB_SIZE * 4,
        "fixture must be the full 17×49152 f32 LE window"
    );

    let mut run = ForwardRun::new(oracle, &PROMPT_TOKENS).expect("teacher-forced run");
    let mut verdicts = Vec::with_capacity(WINDOW_POSITIONS);
    let mut raw_argmax_pos1 = -1i64;

    for i in 0..WINDOW_POSITIONS {
        if i > 0 {
            run.push_token(TRACE_TOKENS[i - 1])
                .expect("teacher-forced push");
        }
        let logits = run.position_logits().expect("position logits");
        let base = i * VOCAB_SIZE * 4;
        let comparator_logp: Vec<f32> = fixture[base..base + VOCAB_SIZE * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let verdict = compare_position(
            i as u32,
            &logits,
            &comparator_logp,
            TRACE_TOKENS[i],
            BAND_DELTA,
        )
        .expect("compare_position");
        if i == 1 {
            raw_argmax_pos1 = verdict.raw_argmax;
        }
        verdicts.push(verdict);
    }

    // EOG-exclusion fidelity probe: at fixture position 1 the raw (EOG-
    // included) argmax is token 2 (`<|im_end|>`/EOS) while the sampled trace
    // token is 198 — the oracle reproduces the pinned EOG scenario, so the
    // top-1 check is not vacuous.
    assert_eq!(
        raw_argmax_pos1, 2,
        "position 1 raw argmax must be the EOG token 2 (numeric contract §2.1)"
    );

    // The report prints the comparator trace token and the EOG-excluded
    // oracle top-1 (contract §4.1 surface) per position, plus the named
    // failing thresholds (contract §4.5) — not just booleans + raw argmax.
    let report = format!(
        "window positions:\n{}",
        verdicts
            .iter()
            .map(|v| {
                let failing = if v.failing_thresholds.is_empty() {
                    "none".to_string()
                } else {
                    v.failing_thresholds
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                };
                format!(
                    "  pos {:2}: top1={} trace={:5} oracle_top1={:5} raw_argmax={:5} topk={}/{} band={:.3e} finite={} failing=[{}] ok={}",
                    v.position,
                    v.top1_matches,
                    v.trace_token,
                    v.oracle_top1,
                    v.raw_argmax,
                    v.topk_overlap,
                    TOPK_K,
                    v.max_band_deviation,
                    v.all_finite,
                    failing,
                    v.ok
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Per-position contract fields (contract §4.5): every verdict records
    // the comparator trace token, carries the EOG-excluded oracle top-1
    // (never an EOG token), and names its failing thresholds (non-empty
    // exactly when the position fails).
    for (i, v) in verdicts.iter().enumerate() {
        assert_eq!(
            v.trace_token, TRACE_TOKENS[i],
            "verdict must record the comparator trace token at pos {}:\n{report}",
            v.position
        );
        assert!(
            !EOG_TOKENS.contains(&v.oracle_top1),
            "pos {}: oracle_top1 {} must never be an EOG token (contract §2.1/§4.1):\n{report}",
            v.position,
            v.oracle_top1
        );
        assert_eq!(
            v.failing_thresholds.is_empty(),
            v.ok,
            "pos {}: failing thresholds empty iff ok:\n{report}",
            v.position
        );
    }

    // (a) top-1 exact over non-EOG at every position — the EOG-excluded
    // oracle top-1 therefore equals the comparator trace token everywhere.
    assert!(
        verdicts.iter().all(|v| v.top1_matches),
        "top-1 exact over non-EOG must hold everywhere:\n{report}"
    );
    for (i, v) in verdicts.iter().enumerate() {
        assert_eq!(
            v.oracle_top1, TRACE_TOKENS[i],
            "pos {}: oracle_top1 must equal the comparator trace token:\n{report}",
            v.position
        );
    }
    // (b) top-k k=5 ≥4/5 everywhere.
    assert!(
        verdicts.iter().all(|v| v.topk_matches),
        "top-k overlap must be ≥4/5 everywhere:\n{report}"
    );
    // (c) per-element band Δ=1e-5 log-softmax — the band does NOT hold
    // (module doc BAND STATUS: ~1.3e-2..2.2e-2 residual vs 1e-5); every
    // position carries the honest named-failure record. Truth over safety:
    // the failure is recorded, never weakened or hidden.
    assert!(
        verdicts
            .iter()
            .all(|v| v.failing_thresholds == [FailingThreshold::Band]),
        "the band must fail at every position (the only failing threshold):\n{report}"
    );
    // (d) finite gate everywhere.
    assert!(
        verdicts.iter().all(|v| v.all_finite),
        "finite gate must hold everywhere:\n{report}"
    );

    // (e) first-divergence rule (contract §4.5): the durable record is taken
    // at the FIRST failing position (0 = prompt end) and carries the
    // comparator trace token, the EOG-excluded oracle top-1, the named
    // failing thresholds, and the max band deviation at that position.
    let record = DivergenceRecord::first(&verdicts).expect("band divergence must be recorded");
    assert_eq!(
        record.position, 0,
        "the first failing position is the prompt end:\n{report}"
    );
    assert_eq!(
        record.comparator_trace_token, TRACE_TOKENS[0],
        "record must carry the comparator trace token:\n{report}"
    );
    assert_eq!(
        record.oracle_top1, TRACE_TOKENS[0],
        "record must carry the EOG-excluded oracle top-1 (top-1 is exact at position 0):\n{report}"
    );
    assert_eq!(
        record.oracle_top1, verdicts[0].oracle_top1,
        "record oracle_top1 must match the per-position verdict:\n{report}"
    );
    assert_eq!(
        record.failing_thresholds,
        vec![FailingThreshold::Band],
        "record must name the failing threshold(s):\n{report}"
    );
    assert_eq!(
        record.max_band_deviation, verdicts[0].max_band_deviation,
        "record must carry the max band deviation at the first failing position:\n{report}"
    );
    assert!(
        record.max_band_deviation > BAND_DELTA,
        "the recorded band failure must exceed the 1e-5 threshold:\n{report}"
    );
}

/// The **v2.0.0** closeout (decision `41da94f3`, operator-approved): the
/// per-element band is `Delta_comparator_metal` = 2.5e-2 — a pinned-row
/// empirical compatibility envelope over the normalized-logp surface (NOT an
/// f32 precision bound, NOT generalizable) — while top-1 exact over non-EOG
/// {0,2}, top-k k=5 ≥4/5, the finite gate, and the first-divergence rule
/// stay hard. All 17 window positions must meet the envelope + hard gates;
/// the divergence field must be `none`. The receipt records the calibration
/// maximum + headroom (a future observation above the envelope FAILS — no
/// auto-widen).
#[test]
fn teacher_forced_window_meets_numeric_contract_v2_0_0() {
    let Some(oracle) = shared_oracle() else {
        return;
    };
    let Some(fixture) = load_logp_reference() else {
        return;
    };
    // Hash pin: the same committed reference window is the v2.0.0 behavior
    // reference (decision item 2 — the Metal fixture is retained).
    assert_eq!(
        hex(&sha256(&fixture)),
        LOGP_REFERENCE_SHA256_HEX,
        "logp reference fixture must be hash-pinned (capture receipt documents it)"
    );
    assert_eq!(
        fixture.len(),
        WINDOW_POSITIONS * VOCAB_SIZE * 4,
        "fixture must be the full 17×49152 f32 LE window"
    );

    let mut run = ForwardRun::new(oracle, &PROMPT_TOKENS).expect("teacher-forced run");
    let mut verdicts = Vec::with_capacity(WINDOW_POSITIONS);

    for i in 0..WINDOW_POSITIONS {
        if i > 0 {
            run.push_token(TRACE_TOKENS[i - 1])
                .expect("teacher-forced push");
        }
        let logits = run.position_logits().expect("position logits");
        let base = i * VOCAB_SIZE * 4;
        let comparator_logp: Vec<f32> = fixture[base..base + VOCAB_SIZE * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let verdict = compare_position(
            i as u32,
            &logits,
            &comparator_logp,
            TRACE_TOKENS[i],
            BAND_DELTA_V2,
        )
        .expect("compare_position");
        verdicts.push(verdict);
    }

    let report = format!(
        "v2.0.0 window positions:\n{}",
        verdicts
            .iter()
            .map(|v| {
                format!(
                    "  pos {:2}: top1={} trace={:5} oracle_top1={:5} topk={}/{} band={:.3e} finite={} ok={}",
                    v.position,
                    v.top1_matches,
                    v.trace_token,
                    v.oracle_top1,
                    v.topk_overlap,
                    TOPK_K,
                    v.max_band_deviation,
                    v.all_finite,
                    v.ok
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Calibration maximum over the full window + headroom vs the envelope
    // (CTO receipt condition: calibration maximum + headroom recorded; a
    // future observation above the envelope FAILS — no auto-widen).
    let calibration_max = verdicts
        .iter()
        .fold(0.0f32, |m, v| m.max(v.max_band_deviation));
    let headroom = (BAND_DELTA_V2 - calibration_max) / BAND_DELTA_V2;
    eprintln!(
        "v2.0.0 calibration max = {calibration_max:.3e}, envelope = {BAND_DELTA_V2:.3e}, headroom = {headroom:.3} ({:.1}%)",
        headroom * 100.0,
    );
    assert!(
        headroom >= 0.0,
        "calibration max {calibration_max:.3e} must fit the 2.5e-2 envelope:\n{report}",
    );

    // The envelope is doing real work: every position's deviation exceeds the
    // v1.0.0 1e-5 band (v1.0.0 remains an honest, recorded failure — never
    // weakened, never relabeled as a pass).
    assert!(
        verdicts.iter().all(|v| v.max_band_deviation > BAND_DELTA),
        "every position must exceed the v1.0.0 band (the 1e-5 band is an honest failure):\n{report}",
    );

    // Hard gates at every position: top-1 exact over non-EOG {0,2}, top-k
    // k=5 ≥4/5, finite gate.
    assert!(
        verdicts.iter().all(|v| v.top1_matches),
        "top-1 exact over non-EOG must hold everywhere:\n{report}",
    );
    assert!(
        verdicts.iter().all(|v| v.topk_matches),
        "top-k overlap must be ≥4/5 everywhere:\n{report}",
    );
    assert!(
        verdicts.iter().all(|v| v.all_finite),
        "finite gate must hold everywhere:\n{report}",
    );
    assert!(
        verdicts.iter().all(|v| v.ok),
        "every v2.0.0 threshold must pass at every position:\n{report}",
    );

    // The v2.0.0 divergence field is `none` — a fully passing window has no
    // first-divergence record.
    assert_eq!(
        DivergenceRecord::first(&verdicts),
        None,
        "v2.0.0 divergence field must be none (all 17 positions pass):\n{report}",
    );
}

/// The durable first-divergence record is taken at the FIRST failing position
/// and carries the contract fields — comparator trace token, EOG-excluded
/// oracle top-1, named failing thresholds, max band deviation — and a later
/// failure never replaces it (numeric contract §4.5). No model needed:
/// hand-built verdicts.
#[test]
fn divergence_record_first_uses_contract_fields() {
    let pass = PositionVerdict {
        position: 0,
        top1_matches: true,
        raw_argmax: 30,
        trace_token: 30,
        oracle_top1: 30,
        topk_overlap: 5,
        topk_matches: true,
        max_band_deviation: 0.0,
        band_matches: true,
        all_finite: true,
        failing_thresholds: vec![],
        ok: true,
    };
    let first_fail = PositionVerdict {
        position: 3,
        top1_matches: false,
        raw_argmax: 2,
        trace_token: 808,
        oracle_top1: 9,
        topk_overlap: 5,
        topk_matches: true,
        max_band_deviation: 0.0,
        band_matches: true,
        all_finite: true,
        failing_thresholds: vec![FailingThreshold::Top1],
        ok: false,
    };
    let later_fail = PositionVerdict {
        position: 7,
        top1_matches: true,
        raw_argmax: 253,
        trace_token: 253,
        oracle_top1: 253,
        topk_overlap: 3,
        topk_matches: false,
        max_band_deviation: 2.2e-2,
        band_matches: false,
        all_finite: false,
        failing_thresholds: vec![
            FailingThreshold::TopK,
            FailingThreshold::Band,
            FailingThreshold::Finite,
        ],
        ok: false,
    };

    // No failing position → no record.
    assert_eq!(
        DivergenceRecord::first(&[pass.clone(), pass.clone()]),
        None,
        "a fully-passing window has no divergence record"
    );
    // The FIRST failing position wins; the later, "worse" failure at
    // position 7 never replaces it.
    let record = DivergenceRecord::first(&[pass, first_fail.clone(), later_fail])
        .expect("first divergence must be recorded");
    assert_eq!(record.position, 3);
    assert_eq!(record.comparator_trace_token, 808, "comparator trace token");
    assert_eq!(record.oracle_top1, 9, "EOG-excluded oracle top-1");
    assert_eq!(
        record.failing_thresholds,
        vec![FailingThreshold::Top1],
        "named failing thresholds"
    );
    assert_eq!(
        record.max_band_deviation, 0.0,
        "max band deviation at pos 3"
    );
}

// ---------------------------------------------------------------------------
// 5. GI3 logits golden (exit gate bullet 5)
// ---------------------------------------------------------------------------

/// The GI3 logits golden content for window position 0 (prompt end).
struct LogitsGolden {
    raw_logits: Vec<f32>,
    logp: Vec<f32>,
    top1_non_eog_id: i64,
    raw_argmax_id: i64,
}

fn golden_provenance(view: &TensorView<'_>) -> BTreeMap<String, Valor> {
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
    pinned.insert("window".to_string(), Valor::from(0i64));
    pinned.insert("prompt_pos".to_string(), Valor::from(8i64));
    pinned.insert(
        "prompt_tokens".to_string(),
        Valor::from(PROMPT_TOKENS.to_vec()),
    );
    pinned.insert(
        "n_tokens".to_string(),
        Valor::from(PROMPT_TOKENS.len() as i64),
    );
    provenance.insert("pinned_position".to_string(), pinned.into());
    let mut contract = BTreeMap::new();
    contract.insert(
        "schema".to_string(),
        Valor::from("gi0-numeric-contract.md v1.0.0"),
    );
    contract.insert("band_delta".to_string(), Valor::from(BAND_DELTA as f64));
    provenance.insert("numeric_contract".to_string(), contract.into());
    provenance
}

fn golden_to_json(fx: &LogitsGolden, view: &TensorView<'_>) -> String {
    let mut root = golden_provenance(view);
    root.insert("op".to_string(), Valor::from("logits"));
    root.insert(
        "tied_head".to_string(),
        Valor::from("token_embd.weightᵀ (no output.weight exists)"),
    );
    root.insert("top1_non_eog".to_string(), Valor::from(fx.top1_non_eog_id));
    root.insert("raw_argmax".to_string(), Valor::from(fx.raw_argmax_id));
    let raw = BTreeMap::from([
        ("name".to_string(), Valor::from("raw_logits")),
        (
            "elements".to_string(),
            Valor::from(fx.raw_logits.len() as i64),
        ),
        (
            "sha256".to_string(),
            Valor::from(f32_sha256_hex(&fx.raw_logits)),
        ),
        (
            "f32_le_hex".to_string(),
            Valor::from(f32_le_hex(&fx.raw_logits)),
        ),
    ]);
    root.insert("raw_logits".to_string(), raw.into());
    let lp = BTreeMap::from([
        ("name".to_string(), Valor::from("logp")),
        ("elements".to_string(), Valor::from(fx.logp.len() as i64)),
        ("sha256".to_string(), Valor::from(f32_sha256_hex(&fx.logp))),
        ("f32_le_hex".to_string(), Valor::from(f32_le_hex(&fx.logp))),
    ]);
    root.insert("logp".to_string(), lp.into());
    let json = Json::from_object(root).expect("golden JSON is valid");
    format!("{}\n", json.to_wire())
}

/// Build the logits golden from the oracle at window position 0.
fn generate_logits_golden(oracle: &CpuOracle) -> LogitsGolden {
    let raw = oracle.forward_one(&PROMPT_TOKENS).expect("forward_one");
    let lp = log_softmax(&raw).expect("log_softmax");
    let top1 = top1_non_eog(&raw, &EOG_TOKENS);
    let raw_argmax = topk_ids(&lp, 1)[0];
    LogitsGolden {
        raw_logits: raw,
        logp: lp,
        top1_non_eog_id: top1,
        raw_argmax_id: raw_argmax,
    }
}

/// One-time emission of the committed logits golden (env-gated). Run with
/// `FABER_GI23_EMIT_GOLDEN=1` to (re)generate the fixture + manifest under
/// `testdata/gi2-3-logits-golden/`. Setup evidence — the committed files are
/// what the byte-stability test compares against.
#[test]
fn emit_logits_golden_when_requested() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    if std::env::var_os("FABER_GI23_EMIT_GOLDEN").is_none() {
        eprintln!("set FABER_GI23_EMIT_GOLDEN=1 to (re)emit the committed logits golden");
        return;
    }
    let view = build_pinned_view(&bytes);
    let oracle = CpuOracle::build(&view).expect("oracle build");
    let fx = generate_logits_golden(&oracle);
    let dir = goldens_dir();
    std::fs::create_dir_all(&dir).expect("create goldens dir");
    let wire = golden_to_json(&fx, &view);
    std::fs::write(dir.join("logits-pos0.json"), wire).expect("write golden");
    let manifest = build_manifest(&view, &fx);
    std::fs::write(dir.join("manifest.json"), manifest).expect("write manifest");
    eprintln!("wrote GI2-3 logits golden to {}", dir.display());
}

/// The committed logits golden manifest: schema/generator/model provenance +
/// hash accounting.
fn build_manifest(view: &TensorView<'_>, fx: &LogitsGolden) -> String {
    let mut root = golden_provenance(view);
    root.insert("schema".to_string(), Valor::from(MANIFEST_SCHEMA));
    let wire = golden_to_json(fx, view);
    let mut entry = BTreeMap::new();
    entry.insert("op".to_string(), Valor::from("logits"));
    entry.insert("file".to_string(), Valor::from("logits-pos0.json"));
    entry.insert(
        "sha256".to_string(),
        Valor::from(hex(&sha256(wire.as_bytes()))),
    );
    entry.insert(
        "raw_logits_elements".to_string(),
        Valor::from(fx.raw_logits.len() as i64),
    );
    entry.insert(
        "logp_elements".to_string(),
        Valor::from(fx.logp.len() as i64),
    );
    let entry_valor: Valor = entry.into();
    root.insert("fixtures".to_string(), Valor::from(vec![entry_valor]));
    let json = Json::from_object(root).expect("manifest JSON is valid");
    format!("{}\n", json.to_wire())
}

/// The GI3 consumption contract for the logits golden: emitted from the
/// admitted row at position 0, hash-accounted, byte-identical to the
/// committed file, and self-consistent (logp = log-softmax of the raw
/// logits; top-1 documented). Byte-determinism across runs is additionally
/// proven by the committed-vs-regenerated byte identity (the committed file
/// was generated by an earlier independent run) and by
/// `forward_one_is_deterministic_and_incremental_run_is_byte_identical`.
#[test]
fn logits_golden_is_byte_stable_deterministic_and_hash_accounted() {
    let Some(oracle) = shared_oracle() else {
        return;
    };
    let bytes = std::fs::read(PINNED_MODEL_PATH).expect("pinned model read");
    let view = build_pinned_view(&bytes);
    let first = generate_logits_golden(oracle);
    let wire_a = golden_to_json(&first, &view);

    // The committed fixture must match regeneration byte-for-byte.
    let dir = goldens_dir();
    let committed = std::fs::read_to_string(dir.join("logits-pos0.json"))
        .unwrap_or_else(|e| panic!("committed golden must exist at {} ({e})", dir.display()));
    assert_eq!(
        committed, wire_a,
        "logits golden must be byte-identical to the committed file"
    );

    // Hash accounting: the manifest records the fixture file digest.
    let manifest_wire = std::fs::read_to_string(dir.join("manifest.json"))
        .unwrap_or_else(|e| panic!("committed manifest must exist ({e})"));
    let manifest = Json::parse(&manifest_wire).expect("manifest parses");
    let entry = &list(field(manifest.as_valor(), "fixtures"))[0];
    assert_eq!(text(field(entry, "file")), "logits-pos0.json");
    assert_eq!(
        text(field(entry, "sha256")),
        hex(&sha256(wire_a.as_bytes())),
        "manifest hash must account the golden bytes"
    );

    // Provenance + self-consistency: logp is the log-softmax of raw_logits,
    // and the documented top-1/trace facts hold at position 0.
    let parsed = Json::parse(&committed).expect("golden parses");
    let root = parsed.as_valor();
    let model = field(root, "model");
    assert_eq!(text(field(model, "sha256")), PINNED_SHA256_HEX);
    assert_eq!(flag(field(root, "coverage_ok")), true);
    assert_eq!(int(field(root, "top1_non_eog")), TRACE_TOKENS[0]);
    let raw_hex = text(field(field(root, "raw_logits"), "f32_le_hex"));
    let logp_hex = text(field(field(root, "logp"), "f32_le_hex"));
    assert_eq!(raw_hex.len(), VOCAB_SIZE * 8);
    assert_eq!(logp_hex.len(), VOCAB_SIZE * 8);
    // Re-derive logp from the raw hex and compare with the golden's own.
    // The fixture stores the f32 LE byte stream as hex (byte-pairs in order).
    let hex_f32s = |s: &str| -> Vec<f32> {
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex byte"))
            .collect();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let raw: Vec<f32> = hex_f32s(raw_hex);
    let golden_lp: Vec<f32> = hex_f32s(logp_hex);
    let recomputed = log_softmax(&raw).expect("log_softmax recompute");
    assert_eq!(
        recomputed, golden_lp,
        "golden logp must be log-softmax(raw)"
    );
}

/// The committed manifest is always present and names the logits golden
/// (runs even when the pinned model is absent).
#[test]
fn committed_logits_golden_manifest_lists_the_logits_fixture() {
    let dir = goldens_dir();
    let manifest = std::fs::read_to_string(dir.join("manifest.json"))
        .unwrap_or_else(|e| panic!("committed manifest must exist at {} ({e})", dir.display()));
    let root = Json::parse(&manifest)
        .expect("manifest parses")
        .as_valor()
        .clone();
    assert_eq!(text(field(&root, "schema")), MANIFEST_SCHEMA);
    assert_eq!(text(field(&root, "generator")), GENERATOR_VERSION);
    let entries = list(field(&root, "fixtures"));
    assert_eq!(entries.len(), 1, "exactly the logits fixture");
    assert_eq!(text(field(&entries[0], "op")), "logits");
    assert_eq!(text(field(&entries[0], "file")), "logits-pos0.json");
}
