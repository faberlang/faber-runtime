//! GI1-2 `QuantizedTensorLayout` contract tests.
//!
//! Families:
//! 1. Block-size verification for every tensor type in the pinned row
//!    against the model-contract §3 table (`gi0-model-contract.md`).
//! 2. Scale/min encoding structure per §3.
//! 3. Full CTO field set present on the type (test per field).
//! 4. Round-trips: elements-per-block × blocks == tensor element count and
//!    bytes-per-block × blocks == byte-range length, per type (pinned
//!    aggregates, §2.3) and per representative pinned tensor.
//! 5. Byte range validated fail-closed (out-of-bounds, non-block-aligned,
//!    misaligned start, overflow).
//! 6. Repack identity explicit and hash-accounted.
//! 7. Toy packed-u4 provably distinct and never admitted (exclusion).
//! 8. No `U8`-as-quantization carrier in the layout surface.
//! 9. Machine-local pinned-row round-trip (skipped when the file is absent).

use crate::gguf::{
    align_up, EXPECTED_ELEMENTS_PER_TYPE, GgmlType, TensorDescriptor,
};
use crate::packed_numeric::{PackedU4Block, PackedU4Layout};
use crate::quantized_tensor_layout::*;

// ---------------------------------------------------------------------------
// Synthetic descriptor helpers (round-trip fixtures from the pinned row)
// ---------------------------------------------------------------------------

/// Build a block-consistent tensor descriptor (byte range padded to 32).
fn desc(name: &str, ggml_type: GgmlType, dims: &[u64], offset_in_data: u64) -> TensorDescriptor {
    let element_count: u64 = dims.iter().product();
    let blocks = element_count / ggml_type.block_elements();
    let byte_len = blocks * ggml_type.block_bytes();
    TensorDescriptor {
        name: name.to_string(),
        ggml_type,
        dims: dims.to_vec(),
        element_count,
        blocks,
        byte_len,
        offset_in_data,
        absolute_offset: offset_in_data,
    }
}

/// Append `t` at the next 32-aligned offset, returning its layout.
fn push_desc(
    tensors: &mut Vec<TensorDescriptor>,
    name: &str,
    ggml_type: GgmlType,
    dims: &[u64],
    next_offset: &mut u64,
) {
    let d = desc(name, ggml_type, dims, *next_offset);
    *next_offset = align_up(*next_offset + d.byte_len, LAYOUT_ALIGNMENT);
    tensors.push(d);
}

// ---------------------------------------------------------------------------
// 1. Block-size verification against the model-contract §3 table
// ---------------------------------------------------------------------------

/// §3 block table (`gi0-model-contract.md` §3): (elems/block, bytes/block).
const S3_BLOCK_TABLE: [(GgmlType, u64, u64); 5] = [
    (GgmlType::Q4_K, 256, 144),
    (GgmlType::Q5_0, 32, 22),
    (GgmlType::Q6_K, 256, 210),
    (GgmlType::Q8_0, 32, 34),
    (GgmlType::F32, 1, 4),
];

#[test]
fn block_sizes_match_contract_s3_table() {
    for (ggml_type, s3_elems, s3_bytes) in S3_BLOCK_TABLE {
        // The layout's block geometry must match §3 exactly.
        assert_eq!(
            ggml_type.block_elements(),
            s3_elems,
            "{} block element count",
            ggml_type.name()
        );
        assert_eq!(
            ggml_type.block_bytes(),
            s3_bytes,
            "{} block byte width",
            ggml_type.name()
        );
        let layout = QuantizedTensorLayout::resolve(
            &desc("t", ggml_type, &[s3_elems * 2], 0),
            u64::MAX,
        )
        .expect("admitted type resolves");
        assert_eq!(layout.block_elements(), s3_elems);
        assert_eq!(layout.block_bytes(), s3_bytes);
    }
}

// ---------------------------------------------------------------------------
// 2. Scale/min encoding per §3
// ---------------------------------------------------------------------------

#[test]
fn scale_min_encoding_matches_contract_s3() {
    // Q4_K: `d` half + `dmin` half + `scales[12]` + `qs[128]` = 144 B.
    assert_eq!(
        ScaleMinEncoding::from_ggml_type(GgmlType::Q4_K),
        ScaleMinEncoding::Q4_K {
            d_half_bytes: 2,
            dmin_half_bytes: 2,
            scales_len: 12,
            qs_len: 128,
        }
    );
    // Q5_0: `d` half + `qh[4]` + `qs[16]` = 22 B.
    assert_eq!(
        ScaleMinEncoding::from_ggml_type(GgmlType::Q5_0),
        ScaleMinEncoding::Q5_0 {
            d_half_bytes: 2,
            qh_len: 4,
            qs_len: 16,
        }
    );
    // Q8_0: `d` half + `qs[32]` = 34 B.
    assert_eq!(
        ScaleMinEncoding::from_ggml_type(GgmlType::Q8_0),
        ScaleMinEncoding::Q8_0 {
            d_half_bytes: 2,
            qs_len: 32,
        }
    );
    // Q6_K: super-block, 256 elems / 210 B.
    assert_eq!(
        ScaleMinEncoding::from_ggml_type(GgmlType::Q6_K),
        ScaleMinEncoding::Q6_K {
            super_block_elements: 256,
            super_block_bytes: 210,
        }
    );
    // F32: scalar, 4 B, no scale/min fields.
    assert_eq!(
        ScaleMinEncoding::from_ggml_type(GgmlType::F32),
        ScaleMinEncoding::F32 { scalar_bytes: 4 }
    );

    // Every encoding's structural width sums to the GGML block byte width.
    for &(ggml_type, _, s3_bytes) in &S3_BLOCK_TABLE {
        let encoding = ScaleMinEncoding::from_ggml_type(ggml_type);
        assert_eq!(encoding.block_bytes(), s3_bytes, "{}", ggml_type.name());
        assert_eq!(encoding.block_bytes(), ggml_type.block_bytes());
    }
}

// ---------------------------------------------------------------------------
// 3. Full CTO field set (test per field)
// ---------------------------------------------------------------------------

#[test]
fn full_cto_field_set_present() {
    // Q4_K ffn_down [2560, 960] (a §2.3 mixed-quant placement).
    let q4k = QuantizedTensorLayout::resolve(&desc("blk.0.ffn_down.weight", GgmlType::Q4_K, &[2560, 960], 0), u64::MAX)
        .expect("Q4_K resolves");
    assert_eq!(q4k.format_id(), GgmlType::Q4_K);
    assert_eq!(q4k.logical_dtype(), LogicalElementDtype::F32);
    assert_eq!(q4k.logical_dtype().name(), "f32");
    assert_eq!(q4k.logical_dtype().bytes(), 4);
    assert_eq!(q4k.dims(), &[2560, 960]);
    assert_eq!(q4k.element_count(), 2_457_600);
    assert_eq!(q4k.block_elements(), 256);
    assert_eq!(q4k.block_bytes(), 144);
    assert_eq!(q4k.blocks(), 9_600);
    assert_eq!(q4k.scale_min_encoding(), ScaleMinEncoding::from_ggml_type(GgmlType::Q4_K));
    assert_eq!(q4k.alignment(), LAYOUT_ALIGNMENT);
    assert_eq!(q4k.byte_range(), ByteRange { start: 0, end: 9_600 * 144 });
    assert_eq!(q4k.repack_identity(), RepackIdentity::Native);

    // Q8_0 token_embd.weight [960, 49152] (opens the pinned tensor table).
    let q8_0 = QuantizedTensorLayout::resolve(
        &desc("token_embd.weight", GgmlType::Q8_0, &[960, 49_152], 32),
        u64::MAX,
    )
    .expect("Q8_0 resolves");
    assert_eq!(q8_0.format_id(), GgmlType::Q8_0);
    assert_eq!(q8_0.dims(), &[960, 49_152]);
    assert_eq!(q8_0.element_count(), 47_185_920);
    assert_eq!(q8_0.block_elements(), 32);
    assert_eq!(q8_0.block_bytes(), 34);
    assert_eq!(q8_0.blocks(), 1_474_560);
    assert_eq!(q8_0.scale_min_encoding(), ScaleMinEncoding::from_ggml_type(GgmlType::Q8_0));
    assert_eq!(q8_0.byte_range(), ByteRange { start: 32, end: 32 + 50_135_040 });

    // F32 norm [960].
    let f32 = QuantizedTensorLayout::resolve(
        &desc("blk.0.attn_norm.weight", GgmlType::F32, &[960], 64),
        u64::MAX,
    )
    .expect("F32 resolves");
    assert_eq!(f32.format_id(), GgmlType::F32);
    assert_eq!(f32.dims(), &[960]);
    assert_eq!(f32.element_count(), 960);
    assert_eq!(f32.block_elements(), 1);
    assert_eq!(f32.block_bytes(), 4);
    assert_eq!(f32.blocks(), 960);
    assert_eq!(f32.scale_min_encoding(), ScaleMinEncoding::from_ggml_type(GgmlType::F32));
    assert_eq!(f32.byte_range(), ByteRange { start: 64, end: 64 + 3_840 });

    // Q5_0 attn_q.weight [960, 960].
    let q5_0 = QuantizedTensorLayout::resolve(
        &desc("blk.0.attn_q.weight", GgmlType::Q5_0, &[960, 960], 0),
        u64::MAX,
    )
    .expect("Q5_0 resolves");
    assert_eq!(q5_0.format_id(), GgmlType::Q5_0);
    assert_eq!(q5_0.element_count(), 921_600);
    assert_eq!(q5_0.block_elements(), 32);
    assert_eq!(q5_0.block_bytes(), 22);
    assert_eq!(q5_0.blocks(), 28_800);
    assert_eq!(q5_0.scale_min_encoding(), ScaleMinEncoding::from_ggml_type(GgmlType::Q5_0));

    // Q6_K ffn_down [2560, 960] (the other half of the §2.3 split).
    let q6_k = QuantizedTensorLayout::resolve(
        &desc("blk.1.ffn_down.weight", GgmlType::Q6_K, &[2560, 960], 0),
        u64::MAX,
    )
    .expect("Q6_K resolves");
    assert_eq!(q6_k.format_id(), GgmlType::Q6_K);
    assert_eq!(q6_k.element_count(), 2_457_600);
    assert_eq!(q6_k.block_elements(), 256);
    assert_eq!(q6_k.block_bytes(), 210);
    assert_eq!(q6_k.blocks(), 9_600);
    assert_eq!(q6_k.scale_min_encoding(), ScaleMinEncoding::from_ggml_type(GgmlType::Q6_K));

    // Every admitted format id maps to its GGUF id (the closed set).
    assert_eq!(GgmlType::from_id(GgmlType::F32.id()), Some(GgmlType::F32));
    assert_eq!(GgmlType::from_id(GgmlType::Q5_0.id()), Some(GgmlType::Q5_0));
    assert_eq!(GgmlType::from_id(GgmlType::Q8_0.id()), Some(GgmlType::Q8_0));
    assert_eq!(GgmlType::from_id(GgmlType::Q4_K.id()), Some(GgmlType::Q4_K));
    assert_eq!(GgmlType::from_id(GgmlType::Q6_K.id()), Some(GgmlType::Q6_K));
}

// ---------------------------------------------------------------------------
// 4. Round-trips against the pinned row
// ---------------------------------------------------------------------------

/// §2.3 per-type block counts (`gi0-model-contract.md` §2.3 Blocks column).
const S23_BLOCKS_PER_TYPE: [(GgmlType, u64); 5] = [
    (GgmlType::F32, 62_400),
    (GgmlType::Q4_K, 153_600),
    (GgmlType::Q5_0, 7_219_200),
    (GgmlType::Q6_K, 153_600),
    (GgmlType::Q8_0, 1_628_160),
];

#[test]
fn per_type_round_trip_against_pinned_aggregates() {
    // Every admitted type's layout round-trips against the §2.3 per-type
    // element totals: elements-per-block × blocks == tensor element count
    // and bytes-per-block × blocks == byte-range length.
    for &(ggml_type, total_elements) in &EXPECTED_ELEMENTS_PER_TYPE {
        let layout = QuantizedTensorLayout::resolve(
            &desc("aggregate", ggml_type, &[total_elements], 0),
            u64::MAX,
        )
        .unwrap_or_else(|err| panic!("{} aggregate must resolve: {err}", ggml_type.name()));

        // elements-per-block × blocks == tensor element count
        assert_eq!(
            layout.block_elements() * layout.blocks(),
            layout.element_count(),
            "{} elements round-trip",
            ggml_type.name()
        );
        assert_eq!(layout.element_count(), total_elements);

        // bytes-per-block × blocks == byte-range length
        assert_eq!(
            layout.block_bytes() * layout.blocks(),
            layout.byte_range().len(),
            "{} bytes round-trip",
            ggml_type.name()
        );

        // blocks match the §2.3 table.
        let expected_blocks = S23_BLOCKS_PER_TYPE
            .iter()
            .find(|(t, _)| *t == ggml_type)
            .map(|(_, b)| *b)
            .expect("§2.3 row");
        assert_eq!(layout.blocks(), expected_blocks, "{} blocks", ggml_type.name());
    }
}

#[test]
fn per_tensor_round_trip_pinned_pattern() {
    // Representative pinned-row tensors (§2.3 per-layer pattern + the two
    // one-offs), laid out sequentially at 32-aligned offsets.
    let mut tensors = Vec::new();
    let mut next = 0u64;
    push_desc(&mut tensors, "token_embd.weight", GgmlType::Q8_0, &[960, 49_152], &mut next);
    push_desc(&mut tensors, "blk.0.attn_norm.weight", GgmlType::F32, &[960], &mut next);
    push_desc(&mut tensors, "blk.0.attn_q.weight", GgmlType::Q5_0, &[960, 960], &mut next);
    push_desc(&mut tensors, "blk.0.attn_k.weight", GgmlType::Q5_0, &[960, 320], &mut next);
    push_desc(&mut tensors, "blk.0.attn_v.weight", GgmlType::Q8_0, &[960, 320], &mut next);
    push_desc(&mut tensors, "blk.1.attn_v.weight", GgmlType::Q5_0, &[960, 320], &mut next);
    push_desc(&mut tensors, "blk.0.attn_output.weight", GgmlType::Q5_0, &[960, 960], &mut next);
    push_desc(&mut tensors, "blk.0.ffn_norm.weight", GgmlType::F32, &[960], &mut next);
    push_desc(&mut tensors, "blk.0.ffn_gate.weight", GgmlType::Q5_0, &[960, 2560], &mut next);
    push_desc(&mut tensors, "blk.0.ffn_up.weight", GgmlType::Q5_0, &[960, 2560], &mut next);
    push_desc(&mut tensors, "blk.0.ffn_down.weight", GgmlType::Q4_K, &[2560, 960], &mut next);
    push_desc(&mut tensors, "blk.1.ffn_down.weight", GgmlType::Q6_K, &[2560, 960], &mut next);
    push_desc(&mut tensors, "output_norm.weight", GgmlType::F32, &[960], &mut next);

    let file_size = next; // data region exactly fills the file
    let mut prev_end = 0u64;
    for t in &tensors {
        let layout =
            QuantizedTensorLayout::resolve(t, file_size).unwrap_or_else(|err| {
                panic!("{} must resolve: {err}", t.name)
            });
        assert_eq!(layout.format_id(), t.ggml_type);
        assert_eq!(layout.dims(), t.dims.as_slice());
        assert_eq!(layout.element_count(), t.element_count);
        // Round-trip: elements-per-block × blocks == element count.
        assert_eq!(layout.block_elements() * layout.blocks(), t.element_count);
        // Round-trip: bytes-per-block × blocks == byte-range length.
        assert_eq!(layout.block_bytes() * layout.blocks(), layout.byte_range().len());
        assert_eq!(layout.byte_range().len(), t.byte_len);
        assert_eq!(layout.byte_range().start, t.absolute_offset);
        // Ranges are sequential, non-overlapping, and in-file.
        assert_eq!(layout.byte_range().start, prev_end);
        assert!(layout.byte_range().end <= file_size);
        assert_eq!(layout.byte_range().start % LAYOUT_ALIGNMENT, 0);
        prev_end = layout.byte_range().end;
    }
}

// ---------------------------------------------------------------------------
// 5. Byte range validated fail-closed
// ---------------------------------------------------------------------------

#[test]
fn byte_range_out_of_bounds_fails_closed() {
    let d = desc("t0", GgmlType::Q8_0, &[32], 0); // 1 block, 34 bytes
    let err = QuantizedTensorLayout::resolve(&d, 33).unwrap_err();
    assert_eq!(
        err,
        QuantizedLayoutError::ByteRangeOutOfBounds {
            start: 0,
            end: 34,
            file_size: 33,
        }
    );
}

#[test]
fn byte_range_not_block_aligned_fails_closed() {
    // Q5_0 block is 22 bytes; a 23-byte range is not block-aligned.
    let mut d = desc("t0", GgmlType::Q5_0, &[32], 0);
    d.byte_len = 23;
    let err = QuantizedTensorLayout::resolve(&d, u64::MAX).unwrap_err();
    assert_eq!(
        err,
        QuantizedLayoutError::ByteRangeNotBlockAligned {
            byte_len: 23,
            block_bytes: 22,
        }
    );
}

#[test]
fn byte_len_mismatch_fails_closed() {
    // 1 block × 22 bytes expected; a 44-byte range (block-aligned but
    // wrong for 1 block) fails closed.
    let mut d = desc("t0", GgmlType::Q5_0, &[32], 0);
    d.byte_len = 44;
    let err = QuantizedTensorLayout::resolve(&d, u64::MAX).unwrap_err();
    assert_eq!(
        err,
        QuantizedLayoutError::ByteLenMismatch {
            byte_len: 44,
            expected: 22,
        }
    );
}

#[test]
fn misaligned_range_start_fails_closed() {
    let d = desc("t0", GgmlType::F32, &[960], 16); // not 32-aligned
    let err = QuantizedTensorLayout::resolve(&d, u64::MAX).unwrap_err();
    assert_eq!(
        err,
        QuantizedLayoutError::MisalignedRangeStart {
            start: 16,
            alignment: LAYOUT_ALIGNMENT,
        }
    );
}

#[test]
fn range_arithmetic_overflow_fails_closed() {
    let mut d = desc("t0", GgmlType::Q8_0, &[32], 0); // 34 bytes
    d.absolute_offset = u64::MAX - 20; // + 34 overflows u64
    let err = QuantizedTensorLayout::resolve(&d, u64::MAX).unwrap_err();
    assert!(matches!(
        err,
        QuantizedLayoutError::ArithmeticOverflow { ref context }
            if context.contains("byte-range end")
    ));
}

#[test]
fn elements_not_block_aligned_fails_closed() {
    let d = desc("t0", GgmlType::Q4_K, &[100], 0); // not a multiple of 256
    let err = QuantizedTensorLayout::resolve(&d, u64::MAX).unwrap_err();
    assert_eq!(
        err,
        QuantizedLayoutError::ElementsNotBlockAligned {
            elements: 100,
            block_elements: 256,
        }
    );
}

// ---------------------------------------------------------------------------
// 6. Repack identity explicit and hash-accounted
// ---------------------------------------------------------------------------

#[test]
fn repack_identity_is_explicit_and_hash_accounted() {
    let base = QuantizedTensorLayout::resolve(
        &desc("blk.0.ffn_down.weight", GgmlType::Q4_K, &[2560, 960], 0),
        u64::MAX,
    )
    .expect("resolves");
    assert_eq!(base.repack_identity(), RepackIdentity::Native);

    // A declared repack carries a measurable identity (hash) and is explicit.
    let digest = [0xAB_u8; 32];
    let declared = base.with_declared_repack(RepackHash::new(digest));
    assert_eq!(declared.repack_identity(), RepackIdentity::Declared(RepackHash::new(digest)));
    assert_ne!(declared.repack_identity(), RepackIdentity::Native);
    // Direct native block execution remains the contract: repack ≠ native.
    assert_ne!(declared, base);
    match declared.repack_identity() {
        RepackIdentity::Declared(hash) => {
            assert_eq!(hash.0, digest);
            assert_eq!(hash.hex(), "abababababababababababababababababababababababababababababababab");
        }
        RepackIdentity::Native => panic!("repack declaration must be retained"),
    }
}

// ---------------------------------------------------------------------------
// 7. Toy packed-u4 exclusion (explicit)
// ---------------------------------------------------------------------------

#[test]
fn toy_packed_u4_is_distinct_and_never_admitted() {
    // The toy's own signature (FC6): 8 values / 4 bytes, scale f32 +
    // zero_point u8, no GGML structure.
    let toy = PackedU4Layout::toy_u4();
    assert_eq!(toy.block_values, 8);
    assert_eq!(toy.packed_bytes, 4);
    assert_eq!(PackedU4Layout::ELEMENT_WIDTH_BITS, 4);

    // (a) Type/value distinctness: no admitted GGML type shares the toy's
    // (block values, packed bytes) signature.
    for &(ggml_type, s3_elems, s3_bytes) in &S3_BLOCK_TABLE {
        assert!(
            (s3_elems, s3_bytes) != (8, 4),
            "{} must not equal the toy packed-u4 (8 values / 4 bytes)",
            ggml_type.name()
        );
    }

    // (b) No QuantizedTensorLayout resolves a packed-u4 block: an 8-element
    // tensor is block-incompatible with every quantized GGML type and, as
    // F32, packs into 32 bytes — never the toy's 4 bytes.
    for ggml_type in [GgmlType::Q4_K, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q6_K] {
        let err = QuantizedTensorLayout::resolve(&desc("u4ish", ggml_type, &[8], 0), u64::MAX)
            .expect_err("an 8-element tensor is never block-aligned for quantized GGML types");
        assert!(matches!(
            err,
            QuantizedLayoutError::ElementsNotBlockAligned {
                elements: 8,
                block_elements,
            } if block_elements != 8
        ));
    }
    let f32_layout = QuantizedTensorLayout::resolve(
        &desc("u4ish", GgmlType::F32, &[8], 0),
        u64::MAX,
    )
    .expect("F32 divides 8");
    assert_eq!(f32_layout.element_count(), 8);
    assert_eq!(f32_layout.byte_range().len(), 32);
    assert_ne!(f32_layout.byte_range().len(), toy.packed_bytes as u64);

    // (c) Structural distinctness: the toy carries a zero_point concept and a
    // widened F32 layout; no GGML scale/min encoding has a zero-point field
    // and every layout's scale/min encoding is exactly the §3 table shape.
    for &(ggml_type, _, _) in &S3_BLOCK_TABLE {
        let layout = QuantizedTensorLayout::resolve(
            &desc("t", ggml_type, &[ggml_type.block_elements()], 0),
            u64::MAX,
        )
        .expect("resolves");
        assert_eq!(
            layout.scale_min_encoding(),
            ScaleMinEncoding::from_ggml_type(ggml_type),
            "{} scale/min encoding is the §3 shape",
            ggml_type.name()
        );
    }
    // The toy's widened_type is F32, but the toy is a distinct carrier type
    // (its block carries scale + zero_point; a GGML block never does).
    assert_eq!(toy.widened_type, crate::packed_numeric::PackedWidenedType::F32);
}

// ---------------------------------------------------------------------------
// 8. No U8-as-quantization carrier in the layout surface
// ---------------------------------------------------------------------------

#[test]
fn no_u8_as_quantization_carrier() {
    // The layout is a pure descriptor: it holds no packed payload bytes. The
    // only byte data anywhere in its surface is the declared-repack digest.
    // Guard: the type must stay small — embedding a byte array (a
    // u8-as-quantization carrier) would blow past this bound.
    assert!(
        std::mem::size_of::<QuantizedTensorLayout>() < 200,
        "QuantizedTensorLayout must never carry packed bytes (u8-as-quantization)"
    );

    // Its byte footprint is purely derived: byte-range length ==
    // blocks × block-bytes, never stored.
    let layout = QuantizedTensorLayout::resolve(
        &desc("blk.0.ffn_down.weight", GgmlType::Q4_K, &[2560, 960], 0),
        u64::MAX,
    )
    .expect("resolves");
    assert_eq!(layout.byte_range().len(), layout.blocks() * layout.block_bytes());

    // Distinctness from the toy carrier type: QuantizedTensorLayout is a
    // different type from PackedU4Layout/PackedU4Block, so packed bytes are
    // only ever accessed through the byte range.
    let _ = std::mem::size_of::<PackedU4Layout>();
    let _ = std::mem::size_of::<PackedU4Block>();
}

// ---------------------------------------------------------------------------
// 9. Machine-local pinned-row round-trip (skipped when the file is absent)
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

#[test]
fn pinned_row_every_tensor_round_trips() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let admission = crate::gguf::admit_gguf(&bytes).expect("pinned row must admit");

    let layouts = resolve_admission(&admission).expect("all 290 tensors resolve");
    assert_eq!(layouts.len(), admission.tensors.len());
    assert_eq!(layouts.len(), 290);

    // Per-tensor round-trip and in-file bounds for every tensor of every
    // admitted type in the pinned row.
    for (t, layout) in admission.tensors.iter().zip(&layouts) {
        assert_eq!(layout.format_id(), t.ggml_type);
        assert_eq!(layout.dims(), t.dims.as_slice());
        assert_eq!(layout.element_count(), t.element_count);
        assert_eq!(layout.block_elements(), t.ggml_type.block_elements());
        assert_eq!(layout.block_bytes(), t.ggml_type.block_bytes());
        assert_eq!(layout.blocks(), t.blocks);
        // elements-per-block × blocks == tensor element count
        assert_eq!(layout.block_elements() * layout.blocks(), t.element_count);
        // bytes-per-block × blocks == byte-range length
        assert_eq!(
            layout.block_bytes() * layout.blocks(),
            layout.byte_range().len(),
            "{}",
            t.name
        );
        assert_eq!(layout.byte_range().len(), t.byte_len);
        assert_eq!(layout.byte_range().start, t.absolute_offset);
        assert!(layout.byte_range().end <= admission.file_size);
        assert_eq!(layout.alignment(), LAYOUT_ALIGNMENT);
        assert_eq!(layout.repack_identity(), RepackIdentity::Native);
        assert_eq!(layout.logical_dtype(), LogicalElementDtype::F32);
    }

    // Spot checks on the pinned row.
    let first = &layouts[0];
    assert_eq!(first.format_id(), GgmlType::Q8_0);
    assert_eq!(first.dims(), &[960, 49_152]);
    assert_eq!(first.blocks(), 1_474_560);
    assert_eq!(first.byte_range(), ByteRange { start: admission.data_offset, end: admission.data_offset + 50_135_040 });

    let last = &layouts[289];
    assert_eq!(last.format_id(), GgmlType::F32);
    assert_eq!(last.dims(), &[960]);
    assert_eq!(last.byte_range().len(), 3_840);

    // Layouts enumerate in GGUF tensor-table order (determinism).
    let second = resolve_admission(&admission).expect("second resolution");
    assert_eq!(layouts, second);
}

// ---------------------------------------------------------------------------
// Determinism of the descriptor surface (no file dependence)
// ---------------------------------------------------------------------------

#[test]
fn resolution_is_deterministic() {
    let a = QuantizedTensorLayout::resolve(
        &desc("blk.0.ffn_down.weight", GgmlType::Q4_K, &[2560, 960], 0),
        u64::MAX,
    )
    .expect("first resolve");
    let b = QuantizedTensorLayout::resolve(
        &desc("blk.0.ffn_down.weight", GgmlType::Q4_K, &[2560, 960], 0),
        u64::MAX,
    )
    .expect("second resolve");
    assert_eq!(a, b);
    assert_eq!(a.byte_range(), b.byte_range());
}
