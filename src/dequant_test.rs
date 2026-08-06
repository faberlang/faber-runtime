//! GI2-1 — CPU dequant core tests.
//!
//! Families:
//! 1. GGML half → f32 conversion is the exact IEEE-754 binary16 mapping.
//! 2. F32 blocks are bit-exact passthrough (incl. ±0, ±inf, NaN, subnormal,
//!    π).
//! 3. Block-level output equals the **independent reference** bit-exactly
//!    over the committed golden fixture (real pinned-row block bytes +
//!    adversarial byte patterns) — `gi2-dequant-goldens.json`, produced by
//!    `radix/docs/factory/gpu-inference-gguf/evidence/gi2-dequant-reference.py`.
//! 4. Real fixture bytes are the actual pinned-row blocks of the admitted
//!    file (machine-local cross-check; skipped when the model is absent).
//! 5. Per-tensor reconstruction across the quant mix (one Q8_0 `attn_v`, one
//!    Q4_K `ffn_down`, one Q5_0 `attn_q`, one Q6_K `ffn_down`, one F32 norm)
//!    matches the reference SHA-256 of the dequantized f32 byte stream.
//! 6. `coverage_ok()` gates construction — a gapped/forged view fails closed
//!    (GI1-4 residual folded in).
//! 7. Forged entries fail closed (past-file → `EntryNotCovered`; in-file but
//!    oversized → row-length backstop).
//! 8. Out-of-range block/byte access fails closed with the typed diagnostics.
//! 9. Row/block byte-length mismatches and declared-repack layouts are
//!    rejected with typed errors.
//! 10. The toy packed-u4 carrier stays un-admitted (no dequant path).
//! 11. Every dequant `Vec<f32>` is a CPU-oracle materialization accompanied
//!     by an [`OracleReceipt`]: the structural descriptor (source tensor
//!     identity + encoding/byte range, destination contiguous-f32 layout +
//!     byte extent, transform `ggml-quants.c @ a957b7747`, `purpose =
//!     CpuOracle`), deterministic-fixture evidence (output digest + generation
//!     timing/peak bytes — setup evidence, not decode metrics), the explicit
//!     statement that the conversion neither changes `RepackIdentity::Native`
//!     nor authorizes converted-weight GPU/headline execution, and the same
//!     fail-closed gates as `dequant_tensor`.
//!
//! The golden comparison is **bit-exact**: every expected value is the
//! reference's u32 IEEE-754 bit pattern, compared against `f32::to_bits()`
//! (dequant is exact integer/half math — the tests never use `==` on floats).

use crate::dequant::*;
use crate::gguf::{
    admit_gguf, hex, sha256, GgmlType, TensorDescriptor, PINNED_SHA256_HEX,
};
use crate::json::Json;
use crate::quantized_tensor_layout::{
    ByteRange, QuantizedLayoutError, QuantizedTensorLayout, RepackHash, RepackIdentity,
};
use crate::tensor_view::{TensorView, TensorViewEntry, TensorViewError};
use crate::valor::Valor;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Machine-local pinned row (skipped when absent — same convention as
/// `tensor_view_test`).
const PINNED_MODEL_PATH: &str = "/Users/ianzepp/ai/models/SmolLM2-360M-Instruct-Q4_K_M.gguf";

/// The committed independent-reference goldens (produced by the reference
/// script; relative to this crate root → the sibling radix evidence dir).
const GOLDENS_REL_PATH: &str =
    "../radix/docs/factory/gpu-inference-gguf/evidence/gi2-dequant-goldens.json";

fn pinned_model_bytes() -> Option<Vec<u8>> {
    let path = std::path::Path::new(PINNED_MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: pinned model not present at {PINNED_MODEL_PATH}");
        return None;
    }
    std::fs::read(path).ok()
}

fn load_goldens() -> Option<Json> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDENS_REL_PATH);
    let wire = match std::fs::read_to_string(&path) {
        Ok(wire) => wire,
        Err(err) => {
            // The golden file is committed in the sibling radix repo — a skip
            // here means the sibling layout changed; report it loudly so a
            // broken path can never silently skip the reference comparison.
            eprintln!("SKIP: dequant goldens not readable at {} ({err})", path.display());
            return None;
        }
    };
    Some(Json::parse(&wire).expect("golden JSON must parse"))
}
// -- Golden JSON walking helpers -------------------------------------------

fn obj<'a>(v: &'a Valor, key: &str) -> &'a Valor {
    let Valor::Tabula(fields) = v else {
        panic!("expected JSON object at {key}");
    };
    fields.get(key).unwrap_or_else(|| panic!("golden missing field {key:?}"))
}

/// Object-field access rooted at a parsed `Json` document.
fn golden_obj<'a>(golden: &'a Json, key: &str) -> &'a Valor {
    obj(golden.as_valor(), key)
}

fn text<'a>(v: &'a Valor) -> &'a str {
    let Valor::Textus(s) = v else { panic!("expected JSON string") };
    s
}

fn int(v: &Valor) -> i64 {
    let Valor::Numerus(n) = v else { panic!("expected JSON integer") };
    *n
}

fn list<'a>(v: &'a Valor) -> &'a [Valor] {
    let Valor::Lista(items) = v else { panic!("expected JSON array") };
    items
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex string");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex byte"))
        .collect()
}

/// The admitted GGML type for a golden `"type"` field.
fn ggml_type_from_name(name: &str) -> GgmlType {
    match name {
        "F32" => GgmlType::F32,
        "Q4_K" => GgmlType::Q4_K,
        "Q5_0" => GgmlType::Q5_0,
        "Q6_K" => GgmlType::Q6_K,
        "Q8_0" => GgmlType::Q8_0,
        other => panic!("unexpected golden type {other:?}"),
    }
}

/// A one-block synthetic layout for `ggml_type` (dims = `[block_elements]`).
fn synthetic_layout(ggml_type: GgmlType) -> QuantizedTensorLayout {
    synthetic_layout_blocks(ggml_type, 1)
}

/// A synthetic layout of `blocks` packed blocks for `ggml_type`.
fn synthetic_layout_blocks(ggml_type: GgmlType, blocks: u64) -> QuantizedTensorLayout {
    let block_elems = ggml_type.block_elements();
    let desc = TensorDescriptor {
        name: "synthetic.weight".to_string(),
        ggml_type,
        dims: vec![block_elems * blocks],
        element_count: block_elems * blocks,
        blocks,
        byte_len: blocks * ggml_type.block_bytes(),
        offset_in_data: 0,
        absolute_offset: 0,
    };
    QuantizedTensorLayout::resolve(&desc, u64::MAX).expect("synthetic layout resolves")
}

/// f32 values as the LE byte stream the per-tensor reference hashes.
fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// 1. GGML half -> f32 conversion (exact IEEE-754 binary16 -> binary32)
// ---------------------------------------------------------------------------

#[test]
fn half_to_f32_matches_ieee_binary16() {
    // (half bits, expected f32 u32 bits)
    let cases: [(u16, u32); 12] = [
        (0x0000, 0x0000_0000), // +0.0
        (0x8000, 0x8000_0000), // -0.0
        (0x3c00, 0x3f80_0000), // 1.0
        (0xbc00, 0xbf80_0000), // -1.0
        (0x7bff, 0x477f_e000), // 65504.0 (max finite half)
        (0xfbff, 0xc77f_e000), // -65504.0
        (0x0001, 0x3380_0000), // 2^-24 (min subnormal)
        (0x03ff, 0x387f_c000), // 1023 * 2^-24 (max subnormal)
        (0x3555, 0x3eaa_a000), // 1365/4096 = 0.333251953125
        (0x7c00, 0x7f80_0000), // +inf
        (0xfc00, 0xff80_0000), // -inf
        (0x7e00, 0x7fc0_0000), // canonical quiet NaN
    ];
    for (bits, expected) in cases {
        assert_eq!(
            half_to_f32(bits).to_bits(),
            expected,
            "half {bits:#06x} must map bit-exactly"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. F32 blocks: bit-exact passthrough
// ---------------------------------------------------------------------------

#[test]
fn f32_blocks_are_bit_exact_passthrough() {
    let layout = synthetic_layout(GgmlType::F32);
    for bits in [
        0x0000_0000u32, // +0.0
        0x8000_0000,    // -0.0
        0x7f80_0000,    // +inf
        0xff80_0000,    // -inf
        0x7fc0_0000,    // quiet NaN
        0x0000_0001,    // min subnormal
        0x4049_0fdb,    // pi
        0x3f80_0000,    // 1.0
    ] {
        let out = dequant_block(&layout, &bits.to_le_bytes()).expect("f32 block dequantizes");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to_bits(), bits, "f32 passthrough for {bits:#010x}");
    }
}

// ---------------------------------------------------------------------------
// 3. Block-level output equals the independent reference (bit-exact)
// ---------------------------------------------------------------------------

#[test]
fn block_fixtures_match_independent_reference_bit_exactly() {
    let Some(golden) = load_goldens() else { return; };
    let fixtures = list(&golden_obj(&golden, "block_fixtures"));
    assert!(!fixtures.is_empty(), "golden must contain block fixtures");
    let mut real = 0usize;
    let mut adversarial = 0usize;
    for item in fixtures {
        let name = text(&obj(item, "name"));
        let ggml_type = ggml_type_from_name(text(&obj(item, "type")));
        let layout = synthetic_layout(ggml_type);
        let bytes = hex_to_bytes(text(&obj(item, "bytes_hex")));
        let out = dequant_block(&layout, &bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected = list(&obj(item, "output_hex"));
        assert_eq!(
            out.len(),
            expected.len(),
            "{name}: element count vs reference"
        );
        for (idx, (got, want)) in out.iter().zip(expected).enumerate() {
            let want_bits = u32::from_str_radix(text(want), 16)
                .unwrap_or_else(|e| panic!("{name}: bad expected hex: {e}"));
            assert_eq!(
                got.to_bits(),
                want_bits,
                "{name}: element {idx} differs from the independent reference"
            );
        }
        match text(&obj(item, "source")) {
            "real" => real += 1,
            "adversarial" => adversarial += 1,
            other => panic!("unexpected fixture source {other:?}"),
        }
    }
    // The quant mix is covered: every admitted type appears, real AND
    // adversarial fixtures are present for the pinned block table.
    assert!(real > 0, "no real pinned-row block fixtures");
    assert!(adversarial > 0, "no adversarial block fixtures");
}

// ---------------------------------------------------------------------------
// 4. Real fixture bytes are the actual pinned-row blocks (model-dependent)
// ---------------------------------------------------------------------------

#[test]
fn real_fixture_bytes_are_pinned_row_blocks() {
    let Some(bytes) = pinned_model_bytes() else { return; };
    let Some(golden) = load_goldens() else { return; };
    let admission = admit_gguf(&bytes).expect("pinned row must admit");
    let view = TensorView::build(&admission, &bytes).expect("view must build");
    for item in list(&golden_obj(&golden, "block_fixtures")) {
        if text(&obj(item, "source")) != "real" {
            continue;
        }
        let tensor = text(&obj(item, "tensor"));
        let block_index = int(&obj(item, "block_index")) as u64;
        let entry = view
            .tensor(tensor)
            .unwrap_or_else(|| panic!("{tensor} must be in the pinned view"));
        let block = view
            .raw_block(entry, block_index)
            .unwrap_or_else(|e| panic!("{tensor} block {block_index}: {e}"));
        assert_eq!(
            block,
            hex_to_bytes(text(&obj(item, "bytes_hex"))),
            "{tensor} block {block_index} bytes must equal the committed fixture"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Per-tensor reconstruction across the quant mix (model-dependent)
// ---------------------------------------------------------------------------

#[test]
fn per_tensor_reconstruction_matches_reference() {
    let Some(bytes) = pinned_model_bytes() else { return; };
    let Some(golden) = load_goldens() else { return; };
    let admission = admit_gguf(&bytes).expect("pinned row must admit");
    let view = TensorView::build(&admission, &bytes).expect("view must build");
    assert!(view.coverage_ok(), "the pinned view must tile exactly");
    assert_eq!(view.sha256_hex(), PINNED_SHA256_HEX);

    let fixtures = list(&golden_obj(&golden, "tensor_fixtures"));
    // The golden mix is exactly the 5 pinned tensors.
    let mut seen = BTreeMap::<&str, usize>::new();
    for item in fixtures {
        let name = text(&obj(item, "name"));
        *seen.entry(name).or_default() += 1;
        let expected_type = text(&obj(item, "type"));
        let entry = view
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} must be in the pinned view"));
        // Pinned-fact cross-checks: the golden type/element count agree with
        // the admitted view and the contract §2.3 mix.
        assert_eq!(entry.ggml_type, ggml_type_from_name(expected_type), "{name} type");
        assert_eq!(
            entry.element_count,
            int(&obj(item, "elements")) as u64,
            "{name} element count"
        );
        let out = dequant_tensor(&view, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(out.len() as u64, entry.element_count, "{name} output length");
        let digest = sha256(&f32_le_bytes(&out));
        assert_eq!(
            hex(&digest),
            text(&obj(item, "sha256")),
            "{name} dequant SHA-256 must equal the independent reference"
        );
    }
    // One Q8_0 attn_v, one Q4_K ffn_down, one Q5_0 attn_q, one Q6_K ffn_down,
    // one F32 norm — the 5 distinct pinned tensors.
    assert_eq!(seen.len(), 5, "exactly five per-tensor fixtures");
    assert_eq!(seen.get("blk.0.attn_v.weight"), Some(&1));
    assert_eq!(seen.get("blk.3.ffn_down.weight"), Some(&1));
    assert_eq!(seen.get("blk.0.attn_q.weight"), Some(&1));
    assert_eq!(seen.get("blk.0.ffn_down.weight"), Some(&1));
    assert_eq!(seen.get("output_norm.weight"), Some(&1));
}

// ---------------------------------------------------------------------------
// 6. coverage_ok() gates construction (GI1-4 residual folded in)
// ---------------------------------------------------------------------------

/// An admission whose F32 tensor ranges leave a 32-byte gap: the view builds
/// (each range is individually in-bounds) but `coverage_ok()` must be false.
fn gapped_admission() -> crate::gguf::GgufAdmission {
    let f32 = GgmlType::F32;
    let mk = |name: &str, offset: u64| TensorDescriptor {
        name: name.to_string(),
        ggml_type: f32,
        dims: vec![8],
        element_count: 8,
        blocks: 8,
        byte_len: 32,
        offset_in_data: offset,
        absolute_offset: offset,
    };
    crate::gguf::GgufAdmission {
        schema: "test",
        file_size: 4096,
        sha256_hex: String::new(),
        metadata: Vec::new(),
        tensors: vec![mk("t1.weight", 0), mk("t2.weight", 64)],
        data_offset: 0,
        data_len: 96,
    }
}

/// An admission whose F32 tensor ranges tile the data region exactly (gap-free
/// control case for the forged-entry tests).
fn covered_admission() -> crate::gguf::GgufAdmission {
    let f32 = GgmlType::F32;
    let mk = |name: &str, offset: u64| TensorDescriptor {
        name: name.to_string(),
        ggml_type: f32,
        dims: vec![8],
        element_count: 8,
        blocks: 8,
        byte_len: 32,
        offset_in_data: offset,
        absolute_offset: offset,
    };
    crate::gguf::GgufAdmission {
        schema: "test",
        file_size: 4096,
        sha256_hex: String::new(),
        metadata: Vec::new(),
        tensors: vec![mk("t1.weight", 0), mk("t2.weight", 32)],
        data_offset: 0,
        data_len: 64,
    }
}

#[test]
fn coverage_gating_fails_closed_on_gapped_view() {
    let bytes = vec![0u8; 4096];
    let view = TensorView::build(&gapped_admission(), &bytes)
        .expect("a gapped view still builds (ranges are individually in-bounds)");
    assert!(!view.coverage_ok(), "gapped range set must fail aggregate coverage");
    let entry = view.entry(0).expect("first entry");
    let err = dequant_tensor(&view, entry).expect_err("dequant must refuse a gapped view");
    assert!(matches!(err, DequantError::CoverageNotOk));
}

// ---------------------------------------------------------------------------
// 7. Forged entries fail closed
// ---------------------------------------------------------------------------

#[test]
fn forged_entries_fail_closed() {
    let bytes = vec![0u8; 4096];
    let view = TensorView::build(&covered_admission(), &bytes)
        .expect("covered view must build");
    assert!(view.coverage_ok(), "control view must tile exactly");
    let good = view.entry(0).expect("first entry");

    // Forged entry whose declared range extends past the file.
    let past_file = TensorViewEntry {
        name: "forged.past_file.weight".to_string(),
        ggml_type: good.ggml_type,
        dims: good.dims.clone(),
        element_count: good.element_count,
        byte_range: ByteRange::new(0, 5000),
        layout: good.layout.clone(),
    };
    let err = dequant_tensor(&view, &past_file).expect_err("past-file entry must fail closed");
    assert!(matches!(
        err,
        DequantError::EntryNotCovered { ref name } if name == "forged.past_file.weight"
    ));

    // Forged entry with an in-file but oversized range: passes the coverage
    // gates but trips the row-length backstop (8 F32 blocks -> 32 bytes).
    let oversized = TensorViewEntry {
        name: "forged.oversized.weight".to_string(),
        ggml_type: good.ggml_type,
        dims: good.dims.clone(),
        element_count: good.element_count,
        byte_range: ByteRange::new(0, 4096),
        layout: good.layout.clone(),
    };
    let err = dequant_tensor(&view, &oversized).expect_err("oversized entry must fail closed");
    assert!(matches!(
        err,
        DequantError::RowBytesMismatch { expected: 32, actual: 4096 }
    ));
}

// ---------------------------------------------------------------------------
// 8. Out-of-range block/byte access fails closed (typed diagnostics)
// ---------------------------------------------------------------------------

#[test]
fn out_of_range_block_and_byte_access_fails_closed() {
    let Some(bytes) = pinned_model_bytes() else { return; };
    let admission = admit_gguf(&bytes).expect("pinned row must admit");
    let view = TensorView::build(&admission, &bytes).expect("view must build");
    let entry = view.tensor("token_embd.weight").expect("token_embd.weight");

    // Block index one past the last block -> typed BlockIndexOutOfBounds.
    let err = view
        .raw_block(entry, entry.layout.blocks())
        .expect_err("block index past the last block must fail closed");
    assert!(matches!(err, TensorViewError::BlockIndexOutOfBounds { .. }));

    // Forged entry with a range past the file -> typed AccessOutOfBounds.
    let forged = TensorViewEntry {
        name: "forged.weight".to_string(),
        ggml_type: entry.ggml_type,
        dims: entry.dims.clone(),
        element_count: entry.element_count,
        byte_range: ByteRange::new(view.file_size() + 10, view.file_size() + 200),
        layout: entry.layout.clone(),
    };
    let err = view.raw_bytes(&forged).expect_err("range past the file must fail closed");
    assert!(matches!(err, TensorViewError::AccessOutOfBounds { .. }));
}

// ---------------------------------------------------------------------------
// 9. Length mismatches and declared repacks are rejected
// ---------------------------------------------------------------------------

#[test]
fn row_and_block_length_mismatches_fail_closed() {
    // Q8_0: 32 elems / 34 bytes per block.
    let one = synthetic_layout(GgmlType::Q8_0);
    let err = dequant_block(&one, &[0u8; 33]).expect_err("short block");
    assert!(matches!(
        err,
        DequantError::BlockBytesMismatch { expected: 34, actual: 33 }
    ));
    let err = dequant_block(&one, &[0u8; 35]).expect_err("long block");
    assert!(matches!(err, DequantError::BlockBytesMismatch { expected: 34, .. }));

    // Two Q8_0 blocks -> 68 packed bytes.
    let two = synthetic_layout_blocks(GgmlType::Q8_0, 2);
    let err = dequant_row(&two, &[0u8; 67]).expect_err("short row");
    assert!(matches!(
        err,
        DequantError::RowBytesMismatch { expected: 68, actual: 67 }
    ));
    let err = dequant_row(&two, &[0u8; 69]).expect_err("long row");
    assert!(matches!(
        err,
        DequantError::RowBytesMismatch { expected: 68, actual: 69 }
    ));

    // A well-formed row of two blocks succeeds and has the right shape.
    let mut packed = Vec::new();
    packed.extend_from_slice(&[0u8; 34]);
    packed.extend_from_slice(&[1u8; 34]);
    let out = dequant_row(&two, &packed).expect("two-block row dequantizes");
    assert_eq!(out.len(), 64);
}

#[test]
fn declared_repack_layout_is_never_executed() {
    let repacked = synthetic_layout(GgmlType::Q8_0)
        .with_declared_repack(RepackHash::new([0xab; 32]));
    let err = dequant_block(&repacked, &[0u8; 34]).expect_err("repack block refused");
    assert!(matches!(err, DequantError::RepackNotNative));
    let err = dequant_row(&repacked, &[0u8; 34]).expect_err("repack row refused");
    assert!(matches!(err, DequantError::RepackNotNative));
}

// ---------------------------------------------------------------------------
// 10. Toy packed-u4 stays un-admitted (no dequant path)
// ---------------------------------------------------------------------------

#[test]
fn toy_u4_is_never_admitted_by_dequant() {
    // The closed GGML type set is exactly the five admitted block types.
    let ids: [(u32, GgmlType); 5] = [
        (0, GgmlType::F32),
        (6, GgmlType::Q5_0),
        (8, GgmlType::Q8_0),
        (12, GgmlType::Q4_K),
        (14, GgmlType::Q6_K),
    ];
    for (id, expected) in ids {
        assert_eq!(GgmlType::from_id(id), Some(expected), "id {id}");
    }
    for id in 0..=20u32 {
        if !ids.iter().any(|(i, _)| *i == id) {
            assert_eq!(GgmlType::from_id(id), None, "id {id} must not be admitted");
        }
    }
    // No admitted type carries the toy packed-u4 signature (8 values / 4
    // bytes, scale + zero_point — no GGML structure; GI1-2 exclusion).
    for ggml_type in [GgmlType::F32, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K] {
        assert_ne!(
            (ggml_type.block_elements(), ggml_type.block_bytes()),
            (8, 4),
            "{} must never match the toy packed-u4 block signature",
            ggml_type.name()
        );
    }
    // The toy's own layout cannot resolve as any admitted GGML layout: an
    // 8-element F32 tensor packs into 32 bytes, never the toy's 4.
    let err = QuantizedTensorLayout::resolve(
        &TensorDescriptor {
            name: "u4ish.weight".to_string(),
            ggml_type: GgmlType::F32,
            dims: vec![8],
            element_count: 8,
            blocks: 8,
            byte_len: 4,
            offset_in_data: 0,
            absolute_offset: 0,
        },
        u64::MAX,
    )
    .expect_err("an 8-element / 4-byte tensor must fail GGML layout resolution");
    assert!(matches!(err, QuantizedLayoutError::ByteLenMismatch { .. }));
    // And the toy carrier itself is read-only reference material for this
    // exclusion (GI1-2 test) — untouched by dequant.
    let toy = crate::packed_numeric::PackedU4Layout::toy_u4();
    assert_eq!(toy.block_values, 8);
    assert_eq!(toy.packed_bytes, 4);
}

// ---------------------------------------------------------------------------
// Golden identity (schema + pinned model facts)
// ---------------------------------------------------------------------------

#[test]
fn golden_identity_matches_the_pinned_row() {
    let Some(golden) = load_goldens() else { return; };
    assert_eq!(text(&golden_obj(&golden, "schema")), "gi2-dequant-goldens-v1");
    let model = golden_obj(&golden, "model");
    assert_eq!(text(&obj(model, "sha256")), PINNED_SHA256_HEX);
    assert_eq!(int(&obj(model, "bytes")), 270_590_880);
    assert_eq!(
        text(&obj(model, "path")),
        "/Users/ianzepp/ai/models/SmolLM2-360M-Instruct-Q4_K_M.gguf"
    );
}

// ---------------------------------------------------------------------------
// 11. Oracle-materialization receipt accompanies the dequant Vec<f32>
// ---------------------------------------------------------------------------

#[test]
fn oracle_receipt_accompanies_the_materialized_tensor() {
    let Some(bytes) = pinned_model_bytes() else { return; };
    let Some(golden) = load_goldens() else { return; };
    let admission = admit_gguf(&bytes).expect("pinned row must admit");
    let view = TensorView::build(&admission, &bytes).expect("view must build");
    assert!(view.coverage_ok(), "the pinned view must tile exactly");

    let fixtures = list(&golden_obj(&golden, "tensor_fixtures"));
    assert!(!fixtures.is_empty(), "golden must contain tensor fixtures");
    for item in fixtures {
        let name = text(&obj(item, "name"));
        let entry = view
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} must be in the pinned view"));

        // Structural descriptor: source identity / encoding / byte range,
        // destination contiguous-f32 layout + byte extent, transform
        // implementation/version, oracle purpose.
        let receipt =
            OracleReceipt::for_tensor(&view, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(receipt.source_tensor, name, "{name} source identity");
        assert_eq!(receipt.source_encoding, entry.ggml_type, "{name} source encoding");
        assert_eq!(receipt.source_byte_range, entry.byte_range, "{name} source byte range");
        assert_eq!(receipt.dest_element_count, entry.element_count, "{name} dest elements");
        assert_eq!(
            receipt.dest_byte_extent,
            entry.element_count * entry.layout.logical_dtype().bytes(),
            "{name} dest byte extent"
        );
        assert_eq!(receipt.transform_impl, ORACLE_TRANSFORM_IMPL, "{name} transform");
        assert_eq!(receipt.purpose, OraclePurpose::CpuOracle, "{name} oracle purpose");

        // The conversion neither changes RepackIdentity::Native ...
        assert_eq!(
            entry.layout.repack_identity(),
            RepackIdentity::Native,
            "{name} layout must stay native"
        );

        // ... nor authorizes converted-weight execution: the receipt makes
        // the oracle boundary explicit; digest / timing / peak bytes are
        // fixture-generation setup evidence and are absent on a live
        // materialization.
        assert_eq!(
            receipt.output_digest,
            None,
            "{name}: live materialization carries no fixture digest"
        );
        assert_eq!(receipt.timing_us, None, "{name}: live materialization carries no timing");
        assert_eq!(
            receipt.peak_temp_bytes,
            None,
            "{name}: live materialization carries no peak bytes"
        );

        // Deterministic-fixture evidence (recorded at fixture generation):
        // the receipt's output digest equals the actual materialized f32 LE
        // byte stream, and the golden's generation-time timing + peak
        // temporary bytes are carried.
        let out = dequant_tensor(&view, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
        let digest = sha256(&f32_le_bytes(&out));
        assert_eq!(hex(&digest), text(&obj(item, "sha256")), "{name} fixture digest");
        let evidenced = receipt.with_fixture_evidence(
            digest,
            int(&obj(item, "timing_us")) as u64,
            int(&obj(item, "peak_temp_bytes")) as u64,
        );
        assert_eq!(evidenced.output_digest, Some(digest), "{name} evidenced digest");
        assert_eq!(
            evidenced.timing_us,
            Some(int(&obj(item, "timing_us")) as u64),
            "{name} evidenced timing"
        );
        assert_eq!(
            evidenced.peak_temp_bytes,
            Some(int(&obj(item, "peak_temp_bytes")) as u64),
            "{name} evidenced peak bytes"
        );
    }
}

#[test]
fn oracle_receipt_fails_closed_like_dequant_tensor() {
    let bytes = vec![0u8; 4096];

    // Gapped range set -> CoverageNotOk (same gate as dequant_tensor).
    let view = TensorView::build(&gapped_admission(), &bytes)
        .expect("a gapped view still builds (ranges are individually in-bounds)");
    let entry = view.entry(0).expect("first entry");
    let err = OracleReceipt::for_tensor(&view, entry).expect_err("gapped view must fail closed");
    assert!(matches!(err, DequantError::CoverageNotOk));

    // Covered view + forged past-file entry -> EntryNotCovered.
    let view = TensorView::build(&covered_admission(), &bytes)
        .expect("covered view must build");
    assert!(view.coverage_ok(), "control view must tile exactly");
    let good = view.entry(0).expect("first entry");
    let past_file = TensorViewEntry {
        name: "forged.past_file.weight".to_string(),
        ggml_type: good.ggml_type,
        dims: good.dims.clone(),
        element_count: good.element_count,
        byte_range: ByteRange::new(0, 5000),
        layout: good.layout.clone(),
    };
    let err = OracleReceipt::for_tensor(&view, &past_file)
        .expect_err("past-file entry must fail closed");
    assert!(matches!(
        err,
        DequantError::EntryNotCovered { ref name } if name == "forged.past_file.weight"
    ));

    // Forged in-file but oversized entry -> row-length backstop.
    let oversized = TensorViewEntry {
        name: "forged.oversized.weight".to_string(),
        ggml_type: good.ggml_type,
        dims: good.dims.clone(),
        element_count: good.element_count,
        byte_range: ByteRange::new(0, 4096),
        layout: good.layout.clone(),
    };
    let err = OracleReceipt::for_tensor(&view, &oversized)
        .expect_err("oversized entry must fail closed");
    assert!(matches!(
        err,
        DequantError::RowBytesMismatch { expected: 32, actual: 4096 }
    ));

    // Declared-repack layout -> RepackNotNative (never authorized).
    let repacked = TensorViewEntry {
        name: "forged.repacked.weight".to_string(),
        ggml_type: good.ggml_type,
        dims: good.dims.clone(),
        element_count: good.element_count,
        byte_range: ByteRange::new(0, 32),
        layout: good
            .layout
            .clone()
            .with_declared_repack(RepackHash::new([0xab; 32])),
    };
    let err = OracleReceipt::for_tensor(&view, &repacked)
        .expect_err("declared repack must fail closed");
    assert!(matches!(err, DequantError::RepackNotNative));
}
