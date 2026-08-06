//! GI3-2 — S1 per-tensor-class representation/repack plan tests.
//!
//! Families:
//! 1. The pinned-row class facts cover the closed set with the FC5 counts
//!    (Q4_K 16 / Q5_0 176 / Q6_K 16 / Q8_0 17 / F32 65) and agree with the
//!    GGML block geometry (source extents == blocks × block_bytes).
//! 2. The initial selection gives every class a selected representation —
//!    the declared f32 conversion — and no class is left undecided.
//! 3. The declared f32-conversion descriptor carries the full §7.2 field set:
//!    identity (source tensor + encoding), destination (contiguous f32, f32
//!    logical elements), algorithm family, shape/alignment/byte extent,
//!    transform implementation + version, and the fixture (digest/budget/
//!    timing) evidence.
//! 4. A converted tensor never claims direct GGUF quantized execution; the
//!    only direct-native path is explicitly pending a second representation.
//! 5. The unexercised descriptor fields (`backend`, `persistence_policy`,
//!    `executable_compatibility`) are explicitly
//!    `pending_second_representation` (council G3 trim).
//! 6. Fixture evidence is deterministic-only (unset on a live descriptor).
//! 7. Row identity binds to the pinned digest.
//! 8. `QuantizedTensorLayout` + `RepackIdentity::Native` are byte-identical
//!    after this unit (this module only consumes layout facts).

use crate::dequant::ORACLE_TRANSFORM_IMPL;
use crate::gguf::{GgmlType, TensorDescriptor};
use crate::quantized_tensor_layout::{QuantizedTensorLayout, RepackIdentity};
use crate::repack_plan::*;

// ---------------------------------------------------------------------------
// 1. Pinned-row class facts
// ---------------------------------------------------------------------------

#[test]
fn pinned_row_class_facts_cover_the_closed_set() {
    let facts = pinned_row_class_facts();
    assert_eq!(facts.len(), 5, "closed set has exactly five classes");
    let count = |class: GgmlType| {
        facts
            .iter()
            .find(|f| f.ggml_type == class)
            .unwrap_or_else(|| panic!("missing class {}", class.name()))
            .tensor_count
    };
    // FC5: quant mix Q4_K 16 / Q5_0 176 / Q6_K 16 / Q8_0 17 / F32 65 = 290.
    assert_eq!(count(GgmlType::Q4_K), 16);
    assert_eq!(count(GgmlType::Q5_0), 176);
    assert_eq!(count(GgmlType::Q6_K), 16);
    assert_eq!(count(GgmlType::Q8_0), 17);
    assert_eq!(count(GgmlType::F32), 65);
    let total: u64 = facts.iter().map(|f| f.tensor_count).sum();
    assert_eq!(total, 290, "the pinned row has 290 tensors");
}

#[test]
fn class_source_extents_agree_with_ggml_block_geometry() {
    for facts in pinned_row_class_facts() {
        let blocks = facts.total_elements / facts.ggml_type.block_elements();
        assert_eq!(
            facts.source_byte_extent,
            blocks * facts.ggml_type.block_bytes(),
            "{}: source extent must be blocks × block_bytes",
            facts.ggml_type.name()
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Initial selection
// ---------------------------------------------------------------------------

#[test]
fn initial_selection_gives_every_class_the_declared_f32_conversion() {
    let selection = RepackSelection::initial_declared_f32_conversion(RowIdentity::pinned_row());
    assert!(
        selection.every_class_selected(),
        "every class must carry a selected representation"
    );
    assert_eq!(selection.per_class.len(), 5);
    for sel in &selection.per_class {
        let SelectedRepresentation::DeclaredF32Conversion(desc) = &sel.representation else {
            panic!(
                "{}: expected the declared f32 conversion, got {:?}",
                sel.facts.ggml_type.name(),
                sel.representation
            );
        };
        assert_eq!(desc.source_encoding, sel.facts.ggml_type);
    }
}

// ---------------------------------------------------------------------------
// 3. Full repack descriptor field set
// ---------------------------------------------------------------------------

#[test]
fn declared_f32_descriptor_carries_the_full_field_set() {
    for facts in pinned_row_class_facts() {
        let desc = RepackDescriptor::declared_f32_conversion(facts);
        assert_eq!(desc.source_tensor, facts.representative_tensor);
        assert_eq!(desc.source_encoding, facts.ggml_type);
        assert_eq!(desc.destination_layout, DestinationLayout::ContiguousF32);
        assert_eq!(
            desc.element_interpretation,
            ElementInterpretation::F32LogicalElement
        );
        assert_eq!(
            desc.algorithm_family,
            AlgorithmFamily::DeclaredF32Conversion
        );
        assert_eq!(desc.shape.element_count, facts.total_elements);
        assert_eq!(desc.padding, Padding::None);
        assert_eq!(desc.alignment_bytes, F32_ELEMENT_BYTES);
        assert_eq!(desc.byte_extent, facts.f32_destination_byte_extent());
        assert_eq!(desc.transform_impl, ORACLE_TRANSFORM_IMPL);
        assert!(
            desc.is_declared_f32_conversion(),
            "{}: descriptor must identify as the declared f32 conversion",
            facts.ggml_type.name()
        );
    }
}

// ---------------------------------------------------------------------------
// 4. No direct-GGUF claim for a converted tensor
// ---------------------------------------------------------------------------

#[test]
fn converted_tensor_never_claims_direct_quantized_execution() {
    let selection = RepackSelection::initial_declared_f32_conversion(RowIdentity::pinned_row());
    for sel in &selection.per_class {
        assert!(
            !sel.representation.claims_direct_quantized_execution(),
            "{}: the declared f32 conversion must never claim direct GGUF quantized execution",
            sel.facts.ggml_type.name()
        );
    }
    // The only direct-native path is the explicitly-pending variant.
    let pending = SelectedRepresentation::DirectNative(PendingSecondRepresentation);
    assert!(pending.claims_direct_quantized_execution());
}

// ---------------------------------------------------------------------------
// 5. Unexercised fields are explicitly pending (council G3 trim)
// ---------------------------------------------------------------------------

#[test]
fn unexercised_descriptor_fields_are_explicitly_pending_second_representation() {
    for facts in pinned_row_class_facts() {
        let desc = RepackDescriptor::declared_f32_conversion(facts);
        assert_eq!(
            desc.backend,
            PendingSecondRepresentation,
            "{}: backend must be explicitly pending",
            facts.ggml_type.name()
        );
        assert_eq!(
            desc.persistence_policy,
            PendingSecondRepresentation,
            "{}: persistence_policy must be explicitly pending",
            facts.ggml_type.name()
        );
        assert_eq!(
            desc.executable_compatibility,
            PendingSecondRepresentation,
            "{}: executable_compatibility must be explicitly pending",
            facts.ggml_type.name()
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Fixture evidence is deterministic-only
// ---------------------------------------------------------------------------

#[test]
fn fixture_evidence_is_deterministic_only() {
    let desc = RepackDescriptor::declared_f32_conversion(pinned_row_class_facts()[0]);
    assert_eq!(desc.output_digest, None, "live descriptors carry no digest");
    assert_eq!(desc.setup_time_us, None);
    assert_eq!(desc.peak_temp_bytes, None);

    let digest = [0x11u8; 32];
    let evidenced = desc.with_fixture_evidence(digest, 807_317, 20_898_128);
    assert_eq!(evidenced.output_digest, Some(digest));
    assert_eq!(evidenced.setup_time_us, Some(807_317));
    assert_eq!(evidenced.peak_temp_bytes, Some(20_898_128));
    // The base descriptor is unchanged (evidence attaches on a copy).
    assert_eq!(desc.output_digest, None);
}

// ---------------------------------------------------------------------------
// 7. Row identity
// ---------------------------------------------------------------------------

#[test]
fn row_identity_binds_to_the_pinned_digest() {
    let row = RowIdentity::pinned_row();
    assert_eq!(row.model_name, "SmolLM2-360M-Instruct Q4_K_M");
    assert_eq!(
        row.sha256_hex,
        "2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2"
    );
    assert_eq!(row.file_bytes, 270_590_880);
}

// ---------------------------------------------------------------------------
// 8. Stored-layout authority invariant
// ---------------------------------------------------------------------------

#[test]
fn quantized_layout_and_native_repack_identity_are_byte_identical_after_this_unit() {
    // This unit never touches `QuantizedTensorLayout`: a resolved layout
    // still carries `RepackIdentity::Native`, and the repack plan is a
    // *separate* physical-plan surface (it consumes layout facts only).
    let desc = TensorDescriptor {
        name: "blk.0.attn_v.weight".to_string(),
        ggml_type: GgmlType::Q8_0,
        dims: vec![307_200],
        element_count: 307_200,
        blocks: 9_600,
        byte_len: 9_600 * GgmlType::Q8_0.block_bytes(),
        offset_in_data: 0,
        absolute_offset: 0,
    };
    let layout = QuantizedTensorLayout::resolve(&desc, u64::MAX).expect("layout resolves");
    assert_eq!(layout.repack_identity(), RepackIdentity::Native);
    assert_eq!(layout.element_count(), 307_200);
}
