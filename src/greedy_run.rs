//! GI2-4 — free-running greedy decode + agreement/first-divergence record.
//!
//! The second GI2 acceptance surface: a **free-running** greedy decode that
//! feeds its own predictions (a broken oracle must not be masked by the
//! teacher-forced GI2-3 window). Starts from the 9 pinned BOS-free prompt
//! tokens; at each of the 256 generated positions the next token is the
//! **argmax over non-EOG {0, 2}** of the oracle's raw logits (temperature 0;
//! the EOG exclusion reproduces the pinned sampler's `ignore_eos=true` −∞
//! bias — FC1, cf. `correctness-top1.json` sampling block); the prediction
//! is fed back; the run completes 256 generated tokens.
//!
//! No durable KV cache (GI4 owns it): the run rides the numeric-faithful
//! incremental runner [`ForwardRun`] (GI2-3), whose per-position K/V cache
//! provably changes no numerics (byte-identical to `forward_one`). This
//! module adds no cache of its own.
//!
//! The result is the **agreement/first-divergence record** per the numeric
//! contract §4.5 (first-divergence rule, `gi0-numeric-contract.md`):
//! - divergence = any generated position whose greedy token disagrees with
//!   `correctness-top1.json` `trace_tokens` (the greedy surface is §4.1,
//!   top-1 exact over non-EOG {0,2});
//! - the record is taken at the **first** diverging position with the
//!   position index, the comparator trace token vs the oracle token, the
//!   named failing threshold(s), and the max band deviation at that position
//!   (computed against the captured comparator logp window when the position
//!   is inside the 17-position captured window 0..16; `None` beyond it — no
//!   comparator logp reference exists there);
//! - divergence is **never hidden by text-level similarity** — the record is
//!   token-level, per position;
//! - the finite gate (every raw logit, every normalized logp, every token id
//!   finite and in `[0, vocab_size)`) is checked at every covered position.
//!
//! The divergence field is `none` only when the oracle's 256-token greedy
//! trace agrees with the comparator trace at every position.

use crate::cpu_oracle::{
    log_softmax, max_band_deviation, top1_non_eog, CpuOracle, FailingThreshold, ForwardRun,
    OracleError, BAND_DELTA_V2, EOG_TOKENS, VOCAB_SIZE, WINDOW_POSITIONS,
};
use crate::json::Json;
use crate::valor::Valor;
use std::collections::BTreeMap;
use std::fmt;

/// Generated-token count of the correctness fixture (gi0-workloads §4:
/// 256 generated tokens).
pub const GREEDY_TOKEN_COUNT: usize = 256;

/// The 9 pinned BOS-free prompt tokens (gi0-workloads §3.1) — the greedy run
/// starts from these.
pub const PROMPT_TOKENS: [i64; 9] = [504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767];

/// The pinned 256-token greedy trace (`correctness-top1.json` `trace_tokens`,
/// schema `gi0-expected-top1-trace` 1.0.0; comparator revision 10150
/// `dee2a846b`, sampling `ignore_eos=true` / temperature 0 / seed 42).
/// Embedded so the default-lane record tests are hermetic (no trials read).
pub const TRACE_TOKENS: [i64; GREEDY_TOKEN_COUNT] = [
    30, 198, 198, 504, 808, 6330, 314, 253, 2232, 4814, 282, 1027, 28, 979, 260, 1796, 6330, 314,
    253, 540, 1784, 6330, 338, 2925, 253, 29219, 17503, 288, 1538, 3171, 1096, 30, 378, 1796, 6330,
    597, 2925, 253, 540, 5764, 284, 26048, 9238, 28, 527, 314, 5712, 327, 253, 2855, 1694, 30, 198,
    198, 788, 2656, 282, 6330, 2678, 28, 260, 808, 6330, 314, 253, 2232, 6330, 351, 253, 2244,
    4791, 17503, 28, 979, 260, 1796, 6330, 314, 253, 8568, 6330, 351, 827, 4791, 25593, 7238, 411,
    253, 25530, 14061, 365, 397, 595, 669, 2022, 260, 1796, 6330, 540, 1784, 284, 3684, 288, 1012,
    30, 198, 198, 16814, 28, 260, 808, 6330, 314, 540, 19484, 284, 1454, 28, 979, 260, 1796, 6330,
    314, 540, 13992, 284, 2229, 105, 30, 378, 1796, 6330, 2925, 540, 5764, 1789, 284, 28766, 260,
    722, 282, 260, 2229, 476, 38046, 1627, 527, 314, 253, 540, 13270, 2229, 4187, 30, 198, 198,
    20832, 28, 260, 808, 6330, 314, 540, 14857, 284, 288, 260, 1225, 28, 979, 260, 1796, 6330, 314,
    540, 1784, 284, 4798, 30, 378, 1796, 6330, 597, 2925, 540, 9733, 1789, 284, 28766, 260, 722,
    282, 260, 2229, 476, 38046, 1627, 527, 314, 253, 540, 13270, 2229, 4187, 30, 198, 198, 788,
    2656, 282, 3997, 28, 260, 808, 6330, 314, 540, 5764, 284, 6987, 28, 979, 260, 1796, 6330, 314,
    540, 16344, 284, 33820, 30, 378, 808, 6330, 314, 540, 5712, 327, 253, 5764, 3794, 355, 2524,
    28, 979, 260, 1796, 6330, 314, 540, 5712, 327, 253, 5862, 1681, 355,
];

// ---------------------------------------------------------------------------
// The first-divergence record (numeric contract §4.5)
// ---------------------------------------------------------------------------

/// The first-divergence record of a greedy run (§4.5): taken at the **first**
/// diverging generated position with the comparator trace token vs the oracle
/// token, the named failing threshold(s), and the max band deviation at that
/// position. Later disagreements never replace or obscure the first
/// diverging position, and a divergence is never excused by text-level
/// similarity — the record is token-level.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyDivergence {
    /// Generated position of the first divergence (0 = first generated
    /// token after the 9-token prompt).
    pub position: u32,
    /// §4.1/§4.5: the comparator greedy trace token id at this position.
    pub comparator_trace_token: i64,
    /// §4.1/§4.5: the oracle's EOG-excluded greedy token id.
    pub oracle_token: i64,
    /// §4.5: the named failing threshold(s) — `Top1` always at a token
    /// mismatch; `Band` when a captured comparator logp reference exists at
    /// the position and the per-element envelope is exceeded; `Finite` when
    /// any logit/logp at the position is non-finite.
    pub failing_thresholds: Vec<FailingThreshold>,
    /// §4.3/§4.5: max per-element |logp_oracle − logp_comparator| at the
    /// diverging position. `Some` only for positions inside the captured
    /// 17-position comparator window (0..16, `gi2-3-logp-reference.bin`);
    /// `None` beyond it — no comparator logp reference exists there.
    pub max_band_deviation: Option<f32>,
}

impl fmt::Display for GreedyDivergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "first divergence at generated position {}: comparator trace token {} vs oracle token {}; failing threshold(s) [{}]; max band deviation {:?}",
            self.position,
            self.comparator_trace_token,
            self.oracle_token,
            self.failing_thresholds
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            self.max_band_deviation,
        )
    }
}

/// The agreement/first-divergence record of one greedy run (§4.5): the
/// generated tokens, the finite-gate verdict, and the first-divergence record
/// (`None` = the full covered trace agrees).
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyRunRecord {
    /// The oracle-generated token ids, in generation order.
    pub generated_tokens: Vec<i64>,
    /// Finite gate: every raw logit and normalized logp finite, every token
    /// id in `[0, vocab_size)`, at every covered position.
    pub all_finite: bool,
    /// The first-divergence record (§4.5), or `None` when the covered trace
    /// agrees at every position (divergence field = `none`).
    pub first_divergence: Option<GreedyDivergence>,
}

impl GreedyRunRecord {
    /// The divergence field per §4.5: `none` when the full covered trace
    /// agrees, otherwise the first-diverging position + details.
    #[must_use]
    pub fn divergence_field(&self) -> String {
        match &self.first_divergence {
            Some(d) => format!(
                "position {} (comparator {} vs oracle {}; {})",
                d.position,
                d.comparator_trace_token,
                d.oracle_token,
                d.failing_thresholds
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            None => "none".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// The free-running greedy runner
// ---------------------------------------------------------------------------

/// Run the free-running greedy decode over the pinned prompt and compare
/// against the pinned trace.
///
/// - Starts from [`PROMPT_TOKENS`]; generates `trace_tokens.len().min(
///   [`GREEDY_TOKEN_COUNT`])` tokens (the full acceptance run passes the
///   full 256-token [`TRACE_TOKENS`]; a bounded-prefix default-lane test
///   passes a short prefix so the loop mechanics stay fast in the default
///   lane).
/// - At each position the next token is the argmax over non-EOG {0,2} of the
///   raw logits (temperature 0; `ignore_eos=true` bias, FC1) and the
///   prediction is fed back (free-running — never teacher-forced).
/// - `comparator_logp_window`, when provided, is the captured 17×49152
///   normalized-logp window (position-major, vocab-id indexed) used to
///   compute the §4.3 max band deviation at a diverging position inside the
///   captured window; `None` skips the band (recorded as `None` in the
///   divergence).
///
/// # Errors
///
/// Empty `trace_tokens`, an empty token run, or an op/layout contradiction
/// (fail closed).
pub fn run_greedy(
    oracle: &CpuOracle,
    trace_tokens: &[i64],
    comparator_logp_window: Option<&[f32]>,
) -> Result<GreedyRunRecord, OracleError> {
    if trace_tokens.is_empty() {
        return Err(OracleError::EmptyTokens);
    }
    let run_len = trace_tokens.len().min(GREEDY_TOKEN_COUNT);
    let mut run = ForwardRun::new(oracle, &PROMPT_TOKENS)?;

    let mut generated = Vec::with_capacity(run_len);
    let mut all_finite = true;
    let mut first_divergence: Option<GreedyDivergence> = None;

    for i in 0..run_len {
        let logits = run.position_logits()?;
        let logp = log_softmax(&logits)?;
        let finite = logits.iter().all(|v| v.is_finite()) && logp.iter().all(|v| v.is_finite());
        if !finite {
            all_finite = false;
        }
        let token = top1_non_eog(&logits, &EOG_TOKENS);
        generated.push(token);

        if first_divergence.is_none() && token != trace_tokens[i] {
            // §4.5: the record is taken at the FIRST diverging position.
            let mut failing = vec![FailingThreshold::Top1];
            let mut band = None;
            if let Some(window) = comparator_logp_window {
                let pos = i as usize;
                if pos < WINDOW_POSITIONS && window.len() >= (pos + 1) * VOCAB_SIZE {
                    let base = pos * VOCAB_SIZE;
                    let dev = max_band_deviation(&logp, &window[base..base + VOCAB_SIZE]);
                    band = Some(dev);
                    if dev > BAND_DELTA_V2 {
                        failing.push(FailingThreshold::Band);
                    }
                }
            }
            if !finite {
                failing.push(FailingThreshold::Finite);
            }
            first_divergence = Some(GreedyDivergence {
                position: i as u32,
                comparator_trace_token: trace_tokens[i],
                oracle_token: token,
                failing_thresholds: failing,
                max_band_deviation: band,
            });
        }

        if i + 1 < run_len {
            run.push_token(token)?;
        }
    }

    Ok(GreedyRunRecord {
        generated_tokens: generated,
        all_finite,
        first_divergence,
    })
}

// ---------------------------------------------------------------------------
// Committed run-record serialization (the GI2-4 receipt, §4.5)
// ---------------------------------------------------------------------------

/// Schema of the committed run record (byte-stability contract).
pub const RECORD_SCHEMA: &str = "gi2-4-greedy-record-v1";
/// Manifest schema of the committed run record.
pub const RECORD_MANIFEST_SCHEMA: &str = "gi2-4-greedy-record-manifest-v1";
/// Schema of the pinned comparator trace the embedded `TRACE_TOKENS` came
/// from (`correctness-top1.json`).
pub const TRACE_SCHEMA: &str = "gi0-expected-top1-trace 1.0.0";
/// The pinned comparator revision recorded by the trace.
pub const TRACE_COMPARATOR_REVISION: &str = "10150 (dee2a846b)";
/// SHA-256 of the `correctness-top1.json` trace file (trials, hash-accounted
/// in the committed record provenance).
pub const TRACE_FILE_SHA256_HEX: &str =
    "66417f431376a384df1e26faf192d4ca69aaddcc33cdb5e1b949ae9c1b6f6e1a";

/// Serialize a greedy run record to the committed-record JSON (schema
/// `gi2-4-greedy-record-v1`).
#[must_use]
pub fn record_to_json(record: &GreedyRunRecord) -> String {
    let mut root = BTreeMap::new();
    root.insert("schema".to_string(), Valor::from(RECORD_SCHEMA));
    root.insert(
        "prompt_tokens".to_string(),
        Valor::from(PROMPT_TOKENS.to_vec()),
    );
    root.insert(
        "generated_token_count".to_string(),
        Valor::from(record.generated_tokens.len() as i64),
    );
    root.insert("all_finite".to_string(), Valor::from(record.all_finite));
    root.insert(
        "all_agree".to_string(),
        Valor::from(record.first_divergence.is_none()),
    );
    let mut trace = BTreeMap::new();
    trace.insert("schema".to_string(), Valor::from(TRACE_SCHEMA));
    trace.insert(
        "comparator_revision".to_string(),
        Valor::from(TRACE_COMPARATOR_REVISION),
    );
    trace.insert(
        "file_sha256".to_string(),
        Valor::from(TRACE_FILE_SHA256_HEX),
    );
    trace.insert("sampling_ignore_eos".to_string(), Valor::from(true));
    trace.insert("temperature".to_string(), Valor::from(0.0f64));
    root.insert("trace".to_string(), trace.into());

    match &record.first_divergence {
        Some(d) => {
            let mut div = BTreeMap::new();
            div.insert("position".to_string(), Valor::from(d.position as i64));
            div.insert(
                "comparator_trace_token".to_string(),
                Valor::from(d.comparator_trace_token),
            );
            div.insert("oracle_token".to_string(), Valor::from(d.oracle_token));
            div.insert(
                "failing_thresholds".to_string(),
                Valor::from(
                    d.failing_thresholds
                        .iter()
                        .map(|t| Valor::from(t.to_string()))
                        .collect::<Vec<_>>(),
                ),
            );
            div.insert(
                "max_band_deviation".to_string(),
                d.max_band_deviation.map(|v| Valor::from(v as f64)).into(),
            );
            root.insert("first_divergence".to_string(), div.into());
        }
        None => {
            root.insert("first_divergence".to_string(), Valor::from(None::<Valor>));
        }
    }

    root.insert(
        "generated_tokens".to_string(),
        Valor::from(record.generated_tokens.clone()),
    );
    root.insert(
        "generated_tokens_sha256".to_string(),
        Valor::from(f32_sha256_hex_i64(&record.generated_tokens)),
    );
    let json = Json::from_object(root).expect("greedy record JSON is valid");
    format!("{}\n", json.to_wire())
}

/// SHA-256 hex of the i64 token-id stream encoded as little-endian u64s
/// (deterministic hash accounting for the committed record).
fn f32_sha256_hex_i64(tokens: &[i64]) -> String {
    let mut out = Vec::with_capacity(tokens.len() * 8);
    for t in tokens {
        out.extend_from_slice(&(*t as u64).to_le_bytes());
    }
    crate::gguf::hex(&crate::gguf::sha256(&out))
}
