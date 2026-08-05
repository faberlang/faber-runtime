//! GI1-4 deterministic host-readable tensor view tests.
//!
//! Families (all against the machine-local pinned row; skipped when absent):
//! 1. Exact 290-tensor enumeration in GGUF tensor-table order, with
//!    bound-checked indexing.
//! 2. Determinism: two loads of the same file yield byte-identical
//!    descriptors.
//! 3. Named lookup: every pinned base name (`PINNED_BASE_NAMES`) resolves,
//!    exact full-name lookup works, unknown names fail closed.
//! 4. Shapes/types match the model contract §2.3 per-layer pattern.
//! 5. Mixed-quant placements resolve: `ffn_down` Q4_K/Q6_K split on the 16
//!    named layers; `attn_v` Q8_0/Q5_0 split.
//! 6. Byte ranges bound-checked + hash-accounted (data region, coverage,
//!    whole-file SHA-256, raw-bytes/raw-block access).
//! 7. Out-of-range access fails closed (typed errors).
//!
//! The view is CPU-only by construction (a pure descriptor layer; no
//! device/GPU import in `tensor_view.rs`).

use crate::gguf::{admit_gguf, GgmlType, PINNED_SHA256_HEX};
use crate::quantized_tensor_layout::ByteRange;
use crate::tensor_view::*;

// ---------------------------------------------------------------------------
// Machine-local pinned row fixture (skipped when absent — see gi1-delivery §1)
// ---------------------------------------------------------------------------

const PINNED_MODEL_PATH: &str = "/Users/ianzepp/ai/models/SmolLM2-360M-Instruct-Q4_K_M.gguf";

fn pinned_model_bytes() -> Option<Vec<u8>> {
    let path = std::path::Path::new(PINNED_MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: pinned model not present at {PINNED_MODEL_PATH}");
        return None;
    }
    std::fs::read(path).ok()
}

fn build_view(bytes: &[u8]) -> TensorView<'_> {
    let admission = admit_gguf(bytes).expect("pinned row must admit");
    TensorView::build(&admission, bytes).expect("pinned row view must build")
}

/// Extract the layer number from a `"blk.N.<base>.weight"` name.
fn layer_of(name: &str) -> u64 {
    let part = name.split('.').nth(1).expect("blk.N form");
    part.parse::<u64>().expect("numeric layer")
}

// ---------------------------------------------------------------------------
// Contract §2.3 fixtures
// ---------------------------------------------------------------------------

/// `ffn_down` Q4_K layers (model contract §2.3 — the 16 named layers);
/// Q6_K on the other 16.
const FFN_DOWN_Q4_K_LAYERS: [u64; 16] =
    [3, 4, 12, 13, 15, 16, 18, 19, 20, 21, 23, 24, 26, 27, 29, 31];

/// `attn_v` Q8_0 layers (evidence `contract-gguf-metadata.txt`, verified live
/// 2026-08-05); Q5_0 on the other 16.
const ATTN_V_Q8_0_LAYERS: [u64; 16] =
    [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 14, 17, 22, 25, 28, 30];

/// Expected family size per pinned base name (§2.3: 1 + 32×9 + 1 = 290).
fn expected_family_size(base: &str) -> usize {
    match base {
        "token_embd.weight" | "output_norm.weight" => 1,
        _ => 32,
    }
}

/// Expected §2.3 dims for a pinned base name's entries.
fn expected_dims(base: &str) -> &'static [u64] {
    match base {
        "token_embd.weight" => &[960, 49_152],
        "attn_norm.weight" | "ffn_norm.weight" | "output_norm.weight" => &[960],
        "attn_q.weight" | "attn_output.weight" => &[960, 960],
        "attn_k.weight" | "attn_v.weight" => &[960, 320],
        "ffn_gate.weight" | "ffn_up.weight" => &[960, 2560],
        "ffn_down.weight" => &[2560, 960],
        _ => panic!("unexpected base name"),
    }
}

// ---------------------------------------------------------------------------
// 1. Enumeration: exactly 290, GGUF tensor-table order, bound-checked
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_view_enumerates_290_in_gguf_order() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_view(&bytes);
    let admission = admit_gguf(&bytes).expect("pinned row must admit");

    assert_eq!(view.len(), 290);
    assert!(!view.is_empty());

    // Enumeration order == GGUF tensor-table order (the admission order).
    let view_names: Vec<&str> = view.entries().iter().map(|e| e.name.as_str()).collect();
    let admission_names: Vec<&str> = admission
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(view_names, admission_names);

    // First/last entries per the evidence listing (`contract-gguf-metadata.txt`).
    assert_eq!(view_names[0], "token_embd.weight");
    assert_eq!(view_names[289], "output_norm.weight");

    // Bound-checked enumeration.
    assert_eq!(view.entry(0).map(|e| e.name.as_str()), Some("token_embd.weight"));
    assert!(view.entry(289).is_some());
    assert!(view.entry(290).is_none());
    assert!(view.entry(usize::MAX).is_none());
}

// ---------------------------------------------------------------------------
// 2. Determinism: two loads, byte-identical descriptors
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_two_loads_yield_byte_identical_descriptors() {
    let Some(bytes_a) = pinned_model_bytes() else {
        return;
    };
    // A second independent read of the same file.
    let bytes_b = std::fs::read(PINNED_MODEL_PATH).expect("second read");
    assert_eq!(bytes_a, bytes_b, "the pinned file must be stable");

    let a = build_view(&bytes_a);
    let b = build_view(&bytes_b);

    assert_eq!(a, b, "two loads must yield byte-identical descriptors");
    assert_eq!(a.entries(), b.entries());
    assert_eq!(a.sha256_hex(), b.sha256_hex());
    assert_eq!(a.data_offset(), b.data_offset());
    assert_eq!(a.data_len(), b.data_len());
    assert_eq!(a.file_size(), b.file_size());
}

// ---------------------------------------------------------------------------
// 3. Named lookup: every pinned base name resolves
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_named_lookup_resolves_all_pinned_base_names() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_view(&bytes);

    // Family sizes: 1 + 32×9 + 1 = 290 total.
    let mut total = 0;
    for base in PINNED_BASE_NAMES {
        let family = view
            .family(base)
            .unwrap_or_else(|| panic!("pinned base name {base:?} must resolve"));
        assert_eq!(
            family.len(),
            expected_family_size(base),
            "{base:?} family size"
        );
        total += family.len();
    }
    assert_eq!(total, 290);

    // Exact full-name lookup.
    assert_eq!(
        view.tensor("token_embd.weight").map(|e| e.name.as_str()),
        Some("token_embd.weight")
    );
    assert!(view.tensor("blk.0.attn_v.weight").is_some());
    assert!(view.tensor("blk.31.ffn_down.weight").is_some());

    // Unknown / non-pinned names fail closed.
    assert!(view.tensor("output.weight").is_none());
    assert!(view.tensor("bogus.weight").is_none());
    assert!(view.tensor("").is_none());
    assert!(view.family("bogus.weight").is_none());
}

// ---------------------------------------------------------------------------
// 4. Shapes/types match the model contract §2.3 per-layer pattern
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_shapes_and_types_match_contract_s23() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_view(&bytes);

    // Single tensors.
    let token_embd = view
        .tensor("token_embd.weight")
        .expect("token_embd.weight");
    assert_eq!(token_embd.ggml_type, GgmlType::Q8_0);
    assert_eq!(token_embd.dims, [960, 49_152]);
    assert_eq!(token_embd.element_count, 47_185_920);

    let output_norm = view
        .tensor("output_norm.weight")
        .expect("output_norm.weight");
    assert_eq!(output_norm.ggml_type, GgmlType::F32);
    assert_eq!(output_norm.dims, [960]);
    assert_eq!(output_norm.element_count, 960);

    // Per-layer families (type is uniform except the two mixed-quant rows,
    // which are asserted exactly in test 5).
    for base in [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
    ] {
        let family = view.family(base).expect(base);
        assert_eq!(family.len(), 32, "{base} family size");
        for entry in family {
            assert_eq!(entry.dims, expected_dims(base), "{} dims", entry.name);
            assert_eq!(
                entry.element_count,
                entry.dims.iter().product::<u64>(),
                "{} element count",
                entry.name
            );
        }
    }

    // Uniform per-layer types.
    assert!(view
        .family("attn_norm.weight")
        .unwrap()
        .iter()
        .all(|e| e.ggml_type == GgmlType::F32));
    assert!(view
        .family("attn_q.weight")
        .unwrap()
        .iter()
        .all(|e| e.ggml_type == GgmlType::Q5_0));
    assert!(view
        .family("attn_output.weight")
        .unwrap()
        .iter()
        .all(|e| e.ggml_type == GgmlType::Q5_0));
}

// ---------------------------------------------------------------------------
// 5. Mixed-quant placements resolve (contract §2.3)
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_mixed_quant_placements_resolve() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_view(&bytes);

    // ffn_down: Q4_K exactly on the 16 named layers, Q6_K on the other 16.
    let ffn_down = view.family("ffn_down.weight").expect("ffn_down.weight");
    assert_eq!(ffn_down.len(), 32);
    let mut q4_k_layers = Vec::new();
    let mut q6_k_layers = Vec::new();
    for entry in &ffn_down {
        assert_eq!(entry.dims, [2560, 960], "{} dims", entry.name);
        match entry.ggml_type {
            GgmlType::Q4_K => q4_k_layers.push(layer_of(&entry.name)),
            GgmlType::Q6_K => q6_k_layers.push(layer_of(&entry.name)),
            other => panic!("ffn_down has unexpected type {other:?}"),
        }
    }
    q4_k_layers.sort_unstable();
    q6_k_layers.sort_unstable();
    assert_eq!(q4_k_layers, FFN_DOWN_Q4_K_LAYERS, "Q4_K layers");
    assert_eq!(q4_k_layers.len(), 16);
    assert_eq!(q6_k_layers.len(), 16);
    assert!(
        q4_k_layers.iter().all(|l| !q6_k_layers.contains(l)),
        "Q4_K and Q6_K layers must be disjoint"
    );

    // attn_v: Q8_0 exactly on the 16 observed layers, Q5_0 on the other 16.
    let attn_v = view.family("attn_v.weight").expect("attn_v.weight");
    assert_eq!(attn_v.len(), 32);
    let mut q8_0_layers = Vec::new();
    let mut q5_0_layers = Vec::new();
    for entry in &attn_v {
        assert_eq!(entry.dims, [960, 320], "{} dims", entry.name);
        match entry.ggml_type {
            GgmlType::Q8_0 => q8_0_layers.push(layer_of(&entry.name)),
            GgmlType::Q5_0 => q5_0_layers.push(layer_of(&entry.name)),
            other => panic!("attn_v has unexpected type {other:?}"),
        }
    }
    q8_0_layers.sort_unstable();
    q5_0_layers.sort_unstable();
    assert_eq!(q8_0_layers, ATTN_V_Q8_0_LAYERS, "Q8_0 layers");
    assert_eq!(q8_0_layers.len(), 16);
    assert_eq!(q5_0_layers.len(), 16);

    // The two splits are distinct placement sets.
    assert_ne!(q8_0_layers, q4_k_layers, "attn_v and ffn_down splits differ");
}

// ---------------------------------------------------------------------------
// 6. Byte ranges bound-checked + hash-accounted
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_byte_ranges_bound_checked_and_hash_accounted() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_view(&bytes);

    // Data region (values verified live on 2026-08-05 — see gguf_test.rs).
    assert_eq!(view.data_offset(), 1_787_040);
    assert_eq!(view.data_len(), 268_803_840);
    assert_eq!(view.data_offset() + view.data_len(), view.file_size());
    assert_eq!(view.file_size(), 270_590_880);
    assert_eq!(
        view.data_region(),
        ByteRange::new(1_787_040, 270_590_880)
    );

    // Whole-file SHA-256 recorded at GI1-1.
    assert_eq!(view.sha256_hex(), PINNED_SHA256_HEX);
    assert!(view.sha256_matches(&bytes));
    let mut flipped = bytes.clone();
    *flipped.last_mut().unwrap() ^= 0xFF;
    assert!(!view.sha256_matches(&flipped));

    // Per-tensor coverage + exact tiling of the data region.
    for entry in view.entries() {
        assert!(view.per_tensor_covered(entry), "{} covered", entry.name);
    }
    assert!(view.coverage_ok());
    let sum_ranges: u64 = view.entries().iter().map(|e| e.byte_range.len()).sum();
    assert_eq!(sum_ranges, view.data_len());

    // token_embd: first tensor, Q8_0, 32-elem/34-byte blocks.
    let token_embd = view.tensor("token_embd.weight").expect("token_embd.weight");
    assert_eq!(token_embd.byte_range.start, view.data_offset());
    assert_eq!(token_embd.layout.block_elements(), 32);
    assert_eq!(token_embd.layout.block_bytes(), 34);
    assert_eq!(token_embd.layout.blocks(), 47_185_920 / 32);
    assert_eq!(token_embd.byte_range.len(), 1_474_560 * 34);

    // Bounded raw access.
    let raw = view.raw_bytes(token_embd).expect("token_embd raw bytes");
    assert_eq!(raw.len() as u64, token_embd.byte_range.len());
    assert_eq!(raw, &bytes[token_embd.byte_range.start as usize..token_embd.byte_range.end as usize]);
    let block0 = view.raw_block(token_embd, 0).expect("block 0");
    assert_eq!(block0.len(), 34);
    assert_eq!(block0, &raw[..34]);
    let last_block = view
        .raw_block(token_embd, token_embd.layout.blocks() - 1)
        .expect("last block");
    assert_eq!(last_block.len(), 34);
    assert_eq!(
        last_block,
        &raw[(raw.len() - 34)..]
    );

    // Out-of-range block index fails closed.
    let err = view
        .raw_block(token_embd, token_embd.layout.blocks())
        .expect_err("block index == blocks must fail closed");
    assert!(matches!(err, TensorViewError::BlockIndexOutOfBounds { .. }));
}

// ---------------------------------------------------------------------------
// 7. Out-of-range access fails closed (typed errors)
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_out_of_range_access_fails_closed() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let view = build_view(&bytes);
    let token_embd = view.tensor("token_embd.weight").expect("token_embd.weight");

    // Forged entry whose declared range lies entirely beyond the file.
    let past_file = TensorViewEntry {
        name: "forged.past_file.weight".to_string(),
        ggml_type: token_embd.ggml_type,
        dims: token_embd.dims.clone(),
        element_count: token_embd.element_count,
        byte_range: ByteRange::new(
            view.file_size() + 100,
            view.file_size() + 200,
        ),
        layout: token_embd.layout.clone(),
    };
    let err = view
        .raw_bytes(&past_file)
        .expect_err("range past the file must fail closed");
    assert!(matches!(err, TensorViewError::AccessOutOfBounds { .. }));

    // Forged entry whose declared range straddles the file end.
    let straddling = TensorViewEntry {
        name: "forged.straddling.weight".to_string(),
        ggml_type: token_embd.ggml_type,
        dims: token_embd.dims.clone(),
        element_count: token_embd.element_count,
        byte_range: ByteRange::new(
            view.file_size() - 10,
            view.file_size() + 10,
        ),
        layout: token_embd.layout.clone(),
    };
    let err = view
        .raw_bytes(&straddling)
        .expect_err("straddling range must fail closed");
    assert!(matches!(err, TensorViewError::AccessOutOfBounds { .. }));

    // Forged entry whose declared range is too small for one real block:
    // a valid block index but a range that cannot hold the block fails closed.
    let too_small = TensorViewEntry {
        name: "forged.too_small.weight".to_string(),
        ggml_type: token_embd.ggml_type,
        dims: token_embd.dims.clone(),
        element_count: token_embd.element_count,
        byte_range: ByteRange::new(
            view.data_offset(),
            view.data_offset() + 10,
        ),
        layout: token_embd.layout.clone(),
    };
    let err = view
        .raw_block(&too_small, 0)
        .expect_err("block larger than the declared range must fail closed");
    assert!(matches!(err, TensorViewError::AccessOutOfBounds { .. }));
}

// ---------------------------------------------------------------------------
// 8. Build rejects a wrong-sized buffer (fail-closed envelope)
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_build_rejects_wrong_buffer_size() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let admission = admit_gguf(&bytes).expect("pinned row must admit");
    let truncated = &bytes[..bytes.len() - 1];
    let err = TensorView::build(&admission, truncated)
        .expect_err("a truncated buffer must be rejected");
    assert!(matches!(
        err,
        TensorViewError::WrongFileSize { expected: 270_590_880, .. }
    ));
}
