//! Fail-closed admission tests for the GGUF admission core (GI1-1).
//!
//! Two families:
//! 1. A negative matrix of crafted mutant files — each fails closed with a
//!    typed `AdmissionError`, and no allocation proportional to an
//!    attacker-controlled count precedes validation.
//! 2. Positive admission of the pinned row (machine-local artifact at
//!    `identity.path`; skipped when the file is absent).

use crate::gguf::*;
use std::io::Write as _;

// ---------------------------------------------------------------------------
// Small GGUF writer used to craft mutants
// ---------------------------------------------------------------------------

struct W {
    buf: Vec<u8>,
}

impl W {
    fn new() -> W {
        W { buf: Vec::new() }
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn zeroes(&mut self, n: usize) {
        self.buf.extend(std::iter::repeat(0u8).take(n));
    }

    fn pad_to(&mut self, align: u64) {
        let extra = align_up(self.buf.len() as u64, align) - self.buf.len() as u64;
        self.zeroes(extra as usize);
    }
}

fn write_header(w: &mut W, kv_count: u64, tensor_count: u64) {
    w.u32(GGUF_MAGIC);
    w.u32(GGUF_VERSION);
    w.u64(tensor_count);
    w.u64(kv_count);
}

/// The full 37-KV metadata section with values matching the contract.
fn write_valid_kvs(w: &mut W) {
    w.str("general.architecture");
    w.u32(8);
    w.str("llama");
    w.str("general.type");
    w.u32(8);
    w.str("model");
    w.str("general.name");
    w.u32(8);
    w.str("Smollm2 360M 8k Lc100K Mix1 Ep2");
    w.str("general.organization");
    w.u32(8);
    w.str("Loubnabnl");
    w.str("general.finetune");
    w.u32(8);
    w.str("8k-lc100k-mix1-ep2");
    w.str("general.basename");
    w.u32(8);
    w.str("smollm2");
    w.str("general.size_label");
    w.u32(8);
    w.str("360M");
    w.str("general.license");
    w.u32(8);
    w.str("apache-2.0");
    w.str("general.languages");
    w.u32(9);
    w.u32(8);
    w.u64(1);
    w.str("en");
    w.str("llama.block_count");
    w.u32(4);
    w.u32(32);
    w.str("llama.context_length");
    w.u32(4);
    w.u32(8192);
    w.str("llama.embedding_length");
    w.u32(4);
    w.u32(960);
    w.str("llama.feed_forward_length");
    w.u32(4);
    w.u32(2560);
    w.str("llama.attention.head_count");
    w.u32(4);
    w.u32(15);
    w.str("llama.attention.head_count_kv");
    w.u32(4);
    w.u32(5);
    w.str("llama.rope.freq_base");
    w.u32(6);
    w.f32(100_000.0);
    w.str("llama.attention.layer_norm_rms_epsilon");
    w.u32(6);
    w.f32(1e-5);
    w.str("general.file_type");
    w.u32(4);
    w.u32(15);
    w.str("llama.vocab_size");
    w.u32(4);
    w.u32(49_152);
    w.str("llama.rope.dimension_count");
    w.u32(4);
    w.u32(64);
    w.str("tokenizer.ggml.add_space_prefix");
    w.u32(7);
    w.u8(0);
    w.str("tokenizer.ggml.add_bos_token");
    w.u32(7);
    w.u8(0);
    w.str("tokenizer.ggml.model");
    w.u32(8);
    w.str("gpt2");
    w.str("tokenizer.ggml.pre");
    w.u32(8);
    w.str("smollm");
    w.str("tokenizer.ggml.tokens");
    w.u32(9);
    w.u32(8);
    w.u64(49_152);
    for i in 0..49_152u64 {
        w.str(&format!("t_{i}"));
    }
    w.str("tokenizer.ggml.token_type");
    w.u32(9);
    w.u32(5);
    w.u64(49_152);
    for _ in 0..49_152 {
        w.i32(0);
    }
    w.str("tokenizer.ggml.merges");
    w.u32(9);
    w.u32(8);
    w.u64(48_900);
    for i in 0..48_900u64 {
        w.str(&format!("m_{i}"));
    }
    w.str("tokenizer.ggml.bos_token_id");
    w.u32(4);
    w.u32(1);
    w.str("tokenizer.ggml.eos_token_id");
    w.u32(4);
    w.u32(2);
    w.str("tokenizer.ggml.unknown_token_id");
    w.u32(4);
    w.u32(0);
    w.str("tokenizer.ggml.padding_token_id");
    w.u32(4);
    w.u32(2);
    w.str("tokenizer.chat_template");
    w.u32(8);
    w.str(CHAT_TEMPLATE);
    w.str("general.quantization_version");
    w.u32(4);
    w.u32(2);
    w.str("quantize.imatrix.file");
    w.u32(8);
    w.str("/models_out/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.imatrix");
    w.str("quantize.imatrix.dataset");
    w.u32(8);
    w.str("/training_dir/calibration_datav3.txt");
    w.str("quantize.imatrix.entries_count");
    w.u32(5);
    w.i32(224);
    w.str("quantize.imatrix.chunks_count");
    w.u32(5);
    w.i32(141);
}

/// Header + full valid KV section with the requested `tensor_count`.
fn valid_prelude(tensor_count: u64) -> W {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, tensor_count);
    write_valid_kvs(&mut w);
    w
}

/// One Q5_0 block (32 elements, 22 bytes) at `offset_in_data`.
fn tiny_q5_0_block(w: &mut W, name: &str, offset_in_data: u64) {
    w.str(name);
    w.u32(1);
    w.u64(32);
    w.u32(6);
    w.u64(offset_in_data);
}

// ---------------------------------------------------------------------------
// SHA-256 correctness
// ---------------------------------------------------------------------------

#[test]
fn sha256_known_vectors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // FIPS 180-4 long-vector: SHA-256 of one million `a` bytes.
    assert_eq!(
        hex(&sha256(&vec![b'a'; 1_000_000])),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

// ---------------------------------------------------------------------------
// Negative matrix — each mutant fails closed with its typed diagnostic
// ---------------------------------------------------------------------------

#[test]
fn wrong_magic_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::InvalidMagic { magic: 0xDEAD_BEEF });
}

#[test]
fn wrong_version_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.buf[4..8].copy_from_slice(&2u32.to_le_bytes());
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::UnsupportedVersion { version: 2 });
}

#[test]
fn wrong_architecture_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("general.architecture");
    w.u32(8);
    w.str("mistral");
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::ArchitectureMismatch { actual: "mistral".to_string() }
    );
}

#[test]
fn wrong_kv_count_fails_closed() {
    for kv_count in [36u64, 38] {
        let mut w = W::new();
        write_header(&mut w, kv_count, EXPECTED_TENSOR_COUNT);
        let err = admit_gguf(&w.buf).unwrap_err();
        assert_eq!(
            err,
            AdmissionError::MetadataKvCountMismatch { expected: 37, actual: kv_count }
        );
    }
}

#[test]
fn unknown_kv_key_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("general.evil");
    w.u32(8);
    w.str("x");
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::UnknownMetadataKey { key: "general.evil".to_string() });
}

#[test]
fn duplicate_metadata_key_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("general.type");
    w.u32(8);
    w.str("model");
    w.str("general.type");
    w.u32(8);
    w.str("model");
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::DuplicateMetadataKey { key: "general.type".to_string() });
}

#[test]
fn wrong_kv_value_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("llama.block_count");
    w.u32(4);
    w.u32(31);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert!(matches!(
        err,
        AdmissionError::MetadataValueMismatch { ref key, .. } if key == "llama.block_count"
    ));
}

#[test]
fn wrong_kv_value_type_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("llama.block_count");
    w.u32(10); // UINT64 instead of the contracted UINT32
    w.u64(32);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert!(matches!(
        err,
        AdmissionError::MetadataValueMismatch { ref key, .. } if key == "llama.block_count"
    ));
}

#[test]
fn wrong_languages_array_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("general.languages");
    w.u32(9);
    w.u32(8);
    w.u64(2);
    w.str("en");
    w.str("fr");
    let err = admit_gguf(&w.buf).unwrap_err();
    assert!(matches!(
        err,
        AdmissionError::TokenizerArrayCountMismatch { ref array, .. } if array == "general.languages"
    ));
}

#[test]
fn wrong_tokenizer_flags_fail_closed() {
    // add_bos_token must be false for the pinned row.
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.add_bos_token");
    w.u32(7);
    w.u8(1);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert!(matches!(
        err,
        AdmissionError::MetadataValueMismatch { ref key, .. }
            if key == "tokenizer.ggml.add_bos_token"
    ));
}

#[test]
fn malformed_bool_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.add_bos_token");
    w.u32(7);
    w.u8(2);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::MalformedBool { value: 2 });
}

#[test]
fn oversized_tensor_count_fails_closed_before_allocation() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, 1u64 << 40);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TensorCountMismatch { expected: 290, actual: 1u64 << 40 }
    );
}

#[test]
fn oversized_kv_count_fails_closed_before_allocation() {
    let mut w = W::new();
    write_header(&mut w, 1u64 << 40, EXPECTED_TENSOR_COUNT);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::MetadataKvCountMismatch { expected: 37, actual: 1u64 << 40 }
    );
}

#[test]
fn oversized_key_fails_closed_before_allocation() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.u64(u64::MAX); // key length prefix
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::KeyTooLong { ceiling: 128, actual: u64::MAX });
}

#[test]
fn oversized_string_value_fails_closed_before_allocation() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("general.name");
    w.u32(8);
    w.u64(5000); // over the 4096-byte ceiling
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::StringTooLong {
            key: "general.name".to_string(),
            ceiling: 4096,
            actual: 5000,
        }
    );
}

#[test]
fn oversized_array_element_fails_closed_before_allocation() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.merges");
    w.u32(9);
    w.u32(8);
    w.u64(48_900); // exact count passes, then the first element exceeds the ceiling
    w.u64(5000);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::StringTooLong {
            key: "tokenizer.ggml.merges".to_string(),
            ceiling: 4096,
            actual: 5000,
        }
    );
}

#[test]
fn oversized_tokenizer_arrays_fail_closed_before_allocation() {
    // tokens 49152 exactly
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.tokens");
    w.u32(9);
    w.u32(8);
    w.u64(49_153);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TokenizerArrayCountMismatch {
            array: "tokenizer.ggml.tokens".to_string(),
            expected: 49_152,
            actual: 49_153,
        }
    );

    // token_type 49152 exactly
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.token_type");
    w.u32(9);
    w.u32(5);
    w.u64(49_153);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TokenizerArrayCountMismatch {
            array: "tokenizer.ggml.token_type".to_string(),
            expected: 49_152,
            actual: 49_153,
        }
    );

    // merges 48900 exactly
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.merges");
    w.u32(9);
    w.u32(8);
    w.u64(48_899);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TokenizerArrayCountMismatch {
            array: "tokenizer.ggml.merges".to_string(),
            expected: 48_900,
            actual: 48_899,
        }
    );

    // an astronomical array count must fail before any allocation
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    w.str("tokenizer.ggml.tokens");
    w.u32(9);
    w.u32(8);
    w.u64(1u64 << 40);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TokenizerArrayCountMismatch {
            array: "tokenizer.ggml.tokens".to_string(),
            expected: 49_152,
            actual: 1u64 << 40,
        }
    );
}

#[test]
fn wrong_tensor_dtype_fails_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    w.str("blk.0.attn_norm.weight");
    w.u32(1);
    w.u64(960);
    w.u32(99); // unknown dtype id
    w.u64(0);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::UnknownDtype {
            name: "blk.0.attn_norm.weight".to_string(),
            dtype_id: 99,
        }
    );
}

#[test]
fn duplicate_tensor_name_fails_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    tiny_q5_0_block(&mut w, "a.weight", 0);
    tiny_q5_0_block(&mut w, "a.weight", 32);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::DuplicateTensorName { name: "a.weight".to_string() });
}

#[test]
fn misaligned_tensor_offset_fails_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    tiny_q5_0_block(&mut w, "t0", 0);
    tiny_q5_0_block(&mut w, "t1", 16); // not 32-aligned
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::MisalignedTensorOffset { name: "t1".to_string(), offset: 16 }
    );
}

#[test]
fn overlapping_tensor_ranges_fail_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    tiny_q5_0_block(&mut w, "t0", 0); // [0, 22)
    tiny_q5_0_block(&mut w, "t1", 0); // aligned, but starts before the previous range ends
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::OverlappingTensorRanges {
            name: "t1".to_string(),
            offset: 0,
            previous_end: 22,
        }
    );
}

#[test]
fn out_of_order_tensor_data_fails_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    tiny_q5_0_block(&mut w, "t0", 0); // [0, 22) → next expected offset 32
    tiny_q5_0_block(&mut w, "t1", 64); // gap: 64 != 32
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::NonSequentialTensorOffsets {
            name: "t1".to_string(),
            offset: 64,
            expected: 32,
        }
    );
}

#[test]
fn arithmetic_overflow_fails_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    tiny_q5_0_block(&mut w, "t0", 0);
    // Q8_0 block (34 bytes) at the largest 32-aligned offset:
    // (u64::MAX - 31) + 34 == u64::MAX + 3, which overflows u64.
    w.str("t1");
    w.u32(1);
    w.u64(32);
    w.u32(8);
    w.u64(u64::MAX - 31);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert!(matches!(err, AdmissionError::ArithmeticOverflow { .. }));
}

#[test]
fn truncated_tensor_data_fails_closed() {
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    write_valid_kvs(&mut w);
    for i in 0..290u64 {
        tiny_q5_0_block(&mut w, &format!("t{i}"), 32 * i);
    }
    let infos_end = w.buf.len();
    let data_start = align_up(infos_end as u64, GGUF_ALIGNMENT);
    w.pad_to(GGUF_ALIGNMENT);
    // 290 × 22 = 6380 bytes are needed; provide only 6000.
    w.zeroes(6000);
    let file_size = w.buf.len() as u64;
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TruncatedTensorData {
            data_end: data_start + 6380,
            file_size,
        }
    );
}

#[test]
fn per_type_element_facts_fail_closed() {
    // 290 tensors with the correct per-type DISTRIBUTION (counts pass) but
    // tiny element sizes: the per-type element totals must fail closed.
    let mut w = W::new();
    write_header(&mut w, EXPECTED_KV_COUNT, EXPECTED_TENSOR_COUNT);
    write_valid_kvs(&mut w);
    let mut offset = 0u64;
    let mut push = |w: &mut W, name: &str, dims: &[u64], dtype_id: u32| {
        w.str(name);
        w.u32(dims.len() as u32);
        for d in dims {
            w.u64(*d);
        }
        w.u32(dtype_id);
        w.u64(offset);
        let ggml = GgmlType::from_id(dtype_id).expect("admitted dtype");
        let elements: u64 = dims.iter().product();
        let byte_len = (elements / ggml.block_elements()) * ggml.block_bytes();
        offset = align_up(offset + byte_len, GGUF_ALIGNMENT);
    };
    for i in 0..65u64 {
        push(&mut w, &format!("f{i}"), &[1], 0); // F32
    }
    for i in 0..16u64 {
        push(&mut w, &format!("q4k{i}"), &[256], 12); // Q4_K
    }
    for i in 0..176u64 {
        push(&mut w, &format!("q5o{i}"), &[32], 6); // Q5_0
    }
    for i in 0..16u64 {
        push(&mut w, &format!("q6k{i}"), &[256], 14); // Q6_K
    }
    for i in 0..17u64 {
        push(&mut w, &format!("q8o{i}"), &[32], 8); // Q8_0
    }
    w.pad_to(GGUF_ALIGNMENT);
    w.zeroes(offset as usize);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::PerTypeElementCountMismatch {
            ggml_type: GgmlType::F32,
            expected: 62_400,
            actual: 65,
        }
    );
}

#[test]
fn oversized_tensor_dims_fail_closed() {
    // n_dims outside {1, 2}
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    w.str("t0");
    w.u32(3);
    w.u64(32);
    w.u64(32);
    w.u64(32);
    w.u32(6);
    w.u64(0);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::TensorDimCountMismatch { name: "t0".to_string(), n_dims: 3 });

    // dim over the 65536 ceiling
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    w.str("t0");
    w.u32(1);
    w.u64(70_000);
    w.u32(6);
    w.u64(0);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::TensorDimTooLarge { name: "t0".to_string(), dim: 70_000 });
}

#[test]
fn tensor_name_ceiling_fails_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    w.u64(u64::MAX); // name length prefix
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(err, AdmissionError::TensorNameTooLong { ceiling: 128, actual: u64::MAX });
}

#[test]
fn non_block_aligned_elements_fail_closed() {
    let mut w = valid_prelude(EXPECTED_TENSOR_COUNT);
    w.str("t0");
    w.u32(1);
    w.u64(100); // not a multiple of the Q5_0 block size 32
    w.u32(6);
    w.u64(0);
    let err = admit_gguf(&w.buf).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::TensorElementsNotBlockAligned {
            name: "t0".to_string(),
            elements: 100,
            block_elements: 32,
        }
    );
}

#[test]
fn wrong_file_size_fails_closed_before_read() {
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(&[0u8; 100]).expect("write");
    let err = admit_file(file.path()).unwrap_err();
    assert_eq!(
        err,
        AdmissionError::FileSizeMismatch { expected: PINNED_FILE_SIZE, actual: 100 }
    );
}

// ---------------------------------------------------------------------------
// Pinned-row positive admission (machine-local artifact; skipped when absent)
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
fn pinned_row_admits_with_full_descriptor_set() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let admission = admit_gguf(&bytes).expect("pinned row must admit");
    assert_eq!(admission.schema, CONTRACT_SCHEMA);
    assert_eq!(admission.file_size, PINNED_FILE_SIZE);
    assert_eq!(admission.sha256_hex, PINNED_SHA256_HEX);
    assert_eq!(admission.metadata.len(), EXPECTED_KV_COUNT as usize);
    assert_eq!(admission.tensors.len(), EXPECTED_TENSOR_COUNT as usize);
    // Verified against the live file on 2026-08-05 (GI0-3 evidence).
    assert_eq!(admission.data_offset, 1_787_040);
    assert_eq!(admission.data_len, 268_803_840);
    assert_eq!(admission.data_offset + admission.data_len, PINNED_FILE_SIZE);

    for &(ggml_type, expected) in &EXPECTED_TENSOR_COUNT_PER_TYPE {
        let actual = admission
            .tensors
            .iter()
            .filter(|t| t.ggml_type == ggml_type)
            .count();
        assert_eq!(actual as u64, expected, "{} tensor count", ggml_type.name());
    }

    // token_embd.weight opens the table.
    let first = &admission.tensors[0];
    assert_eq!(first.name, "token_embd.weight");
    assert_eq!(first.ggml_type, GgmlType::Q8_0);
    assert_eq!(first.dims, vec![960, 49_152]);
    assert_eq!(first.element_count, 47_185_920);
    assert_eq!(first.blocks, 1_474_560);
    assert_eq!(first.byte_len, 50_135_040);
    assert_eq!(first.offset_in_data, 0);
    assert_eq!(first.absolute_offset, admission.data_offset);

    // The 1-dim F32 norms follow (n_dims == 1 is part of the pinned row).
    let norm = &admission.tensors[1];
    assert_eq!(norm.name, "blk.0.attn_norm.weight");
    assert_eq!(norm.ggml_type, GgmlType::F32);
    assert_eq!(norm.dims, vec![960]);
    assert_eq!(norm.element_count, 960);
    assert_eq!(norm.byte_len, 3840);

    // output_norm.weight closes the table.
    let last = &admission.tensors[289];
    assert_eq!(last.name, "output_norm.weight");
    assert_eq!(last.ggml_type, GgmlType::F32);

    // Metadata: 37 KVs, file-order spot checks, tokenizer array sizes.
    assert_eq!(admission.metadata[0].key, "general.architecture");
    assert_eq!(
        admission.metadata[0].value,
        MetadataValue::String("llama".to_string())
    );
    assert_eq!(admission.metadata[24].key, "tokenizer.ggml.tokens");
    match &admission.metadata[24].value {
        MetadataValue::StringArray(items) => assert_eq!(items.len(), 49_152),
        other => panic!("expected StringArray, got {other:?}"),
    }
    match &admission.metadata[25].value {
        MetadataValue::Int32Array(items) => assert_eq!(items.len(), 49_152),
        other => panic!("expected Int32Array, got {other:?}"),
    }
    match &admission.metadata[26].value {
        MetadataValue::StringArray(items) => assert_eq!(items.len(), 48_900),
        other => panic!("expected StringArray, got {other:?}"),
    }

    // Absolute offsets are invariant: every tensor maps into the data region.
    for t in &admission.tensors {
        assert_eq!(t.absolute_offset, admission.data_offset + t.offset_in_data);
        assert!(t.absolute_offset >= admission.data_offset);
        assert!(
            t.absolute_offset + t.byte_len <= admission.data_offset + admission.data_len
        );
    }
}

#[test]
fn pinned_row_admit_file_wrapper() {
    if !std::path::Path::new(PINNED_MODEL_PATH).exists() {
        eprintln!("SKIP: pinned model not present at {PINNED_MODEL_PATH}");
        return;
    }
    let admission = admit_file(PINNED_MODEL_PATH).expect("admit_file must admit the pinned row");
    assert_eq!(admission.tensors.len(), EXPECTED_TENSOR_COUNT as usize);
    assert_eq!(admission.sha256_hex, PINNED_SHA256_HEX);
}

#[test]
fn pinned_row_sha_mismatch_fails_closed() {
    let Some(mut bytes) = pinned_model_bytes() else {
        return;
    };
    // Flip the last byte (inside the tensor-data region): the structure stays
    // identical, only the whole-file digest changes.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    let err = admit_gguf(&bytes).unwrap_err();
    assert!(matches!(err, AdmissionError::Sha256Mismatch { .. }));
}

#[test]
fn pinned_row_admission_is_deterministic() {
    let Some(bytes) = pinned_model_bytes() else {
        return;
    };
    let a = admit_gguf(&bytes).expect("first admit");
    let b = admit_gguf(&bytes).expect("second admit");
    assert_eq!(a.tensors, b.tensors);
    assert_eq!(a.metadata, b.metadata);
    assert_eq!(a.data_offset, b.data_offset);
    assert_eq!(a.data_len, b.data_len);
    assert_eq!(a.sha256_hex, b.sha256_hex);
}
