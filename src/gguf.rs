//! GGUF v3 container admission core for the pinned SmolLM2-360M-Instruct row.
//!
//! Fails closed: every intake fact is validated against the frozen
//! `gi0-model-contract` v1.0.0 §9 (40-field schema) + §10 intake checklist
//! (the bounded input — never re-discovered) before any allocation sized by a
//! parsed count and before any descriptor set is returned. One pinned row only
//! (`identity.path` / `identity.sha256` / `identity.bytes`, §1); no fact is
//! generalised beyond it. Identical bytes always produce identical descriptors.
//!
//! Validation order (each check gates the next, so an attacker-controlled
//! count can never drive an allocation or an iteration larger than the exact
//! contracted value):
//!
//! 1. container magic / version;
//! 2. exact `tensor_count` (== 290) and `metadata_kv_count` (== 37) ceilings;
//! 3. the 37 metadata KVs — key length ≤ 128, key in the admitted 37-key set,
//!    value type and value matching the contract (unknown keys, wrong types,
//!    and contradicting values fail closed); tokenizer arrays exactly
//!    tokens 49152 / token_type 49152 / merges 48900, every string ≤ 4096 B;
//! 4. the 290-entry tensor-info table — name length ≤ 128, unique names,
//!    `n_dims` in {1, 2}, every dim ≤ 65536, dtype in the admitted GGML type
//!    set, 32-aligned sequential non-overlapping byte ranges, in-bounds data;
//! 5. aggregate intake facts — per-type tensor counts, per-type element
//!    totals, grand total == 361,821,120;
//! 6. identity facts — exact file size == 270,590,880 and whole-file
//!    SHA-256 == `2fa3f013…bac9c2`.
//!
//! All offset/length/size arithmetic is checked (`checked_add` / `checked_mul`,
//! u64); arithmetic overflow anywhere is rejected before it can panic or wrap.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Pinned-row contract constants (gi0-model-contract v1.0.0 §1/§9/§10)
// ---------------------------------------------------------------------------

/// Schema version consumed by this admission core.
pub const CONTRACT_SCHEMA: &str = "gi0-model-contract v1.0.0";

/// `identity.bytes` — exact pinned file size.
pub const PINNED_FILE_SIZE: u64 = 270_590_880;

/// `identity.sha256` — exact pinned whole-file digest (hex).
pub const PINNED_SHA256_HEX: &str =
    "2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2";

/// `identity.sha256` — exact pinned whole-file digest (bytes).
pub const PINNED_SHA256: [u8; 32] = [
    0x2f, 0xa3, 0xf0, 0x13, 0xdc, 0xdd, 0x7b, 0x99, 0xf9, 0xb2, 0x37, 0x71, 0x7f, 0xa0, 0xb1, 0x2d,
    0x75, 0xbb, 0xb8, 0x99, 0x84, 0xcc, 0x12, 0x74, 0xbe, 0x14, 0x71, 0xa4, 0x65, 0xba, 0xc9, 0xc2,
];

/// GGUF container magic (`"GGUF"` little-endian; FC1).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF container version (FC1).
pub const GGUF_VERSION: u32 = 3;

/// GGUF default alignment (FC1; the pinned row carries no `general.alignment`).
pub const GGUF_ALIGNMENT: u64 = 32;

/// Explicit ceilings — checked before any allocation sized by a parsed count.
pub const MAX_STRING_BYTES: u64 = 4096;
pub const MAX_KEY_BYTES: u64 = 128;
pub const MAX_TENSOR_NAME_BYTES: u64 = 128;
pub const MAX_TENSOR_DIMS: u32 = 2;
pub const MAX_DIM: u64 = 65_536;

/// Exact pinned counts.
pub const EXPECTED_KV_COUNT: u64 = 37;
pub const EXPECTED_TENSOR_COUNT: u64 = 290;
pub const EXPECTED_TOTAL_ELEMENTS: u64 = 361_821_120;
pub const EXPECTED_TOKENS: u64 = 49_152;
pub const EXPECTED_TOKEN_TYPE: u64 = 49_152;
pub const EXPECTED_MERGES: u64 = 48_900;

/// The pinned 37-key metadata set (every key must appear exactly once).
pub const ADMITTED_KEYS: [&str; 37] = [
    "general.architecture",
    "general.basename",
    "general.file_type",
    "general.finetune",
    "general.languages",
    "general.license",
    "general.name",
    "general.organization",
    "general.quantization_version",
    "general.size_label",
    "general.type",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.attention.layer_norm_rms_epsilon",
    "llama.block_count",
    "llama.context_length",
    "llama.embedding_length",
    "llama.feed_forward_length",
    "llama.rope.dimension_count",
    "llama.rope.freq_base",
    "llama.vocab_size",
    "quantize.imatrix.chunks_count",
    "quantize.imatrix.dataset",
    "quantize.imatrix.entries_count",
    "quantize.imatrix.file",
    "tokenizer.chat_template",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.model",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.unknown_token_id",
];

/// Admitted GGML/GGUF tensor dtype set for the pinned row (FC3): the complete
/// closed set `{F32, Q5_0, Q8_0, Q4_K, Q6_K}`. Any other dtype id fails closed.
#[allow(non_camel_case_types)] // variants mirror the canonical GGML type names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    Q5_0,
    Q8_0,
    Q4_K,
    Q6_K,
}

impl GgmlType {
    /// Map a GGUF dtype id to an admitted type (`None` for unknown ids).
    #[must_use]
    pub fn from_id(id: u32) -> Option<GgmlType> {
        match id {
            0 => Some(GgmlType::F32),
            6 => Some(GgmlType::Q5_0),
            8 => Some(GgmlType::Q8_0),
            12 => Some(GgmlType::Q4_K),
            14 => Some(GgmlType::Q6_K),
            _ => None,
        }
    }

    /// The GGUF dtype id for this type.
    #[must_use]
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::Q5_0 => 6,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4_K => 12,
            GgmlType::Q6_K => 14,
        }
    }

    /// Canonical type name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q6_K => "Q6_K",
        }
    }

    /// Logical elements per block (§3: F32 1, Q5_0 32, Q8_0 32, Q4_K 256, Q6_K 256).
    #[must_use]
    pub fn block_elements(self) -> u64 {
        match self {
            GgmlType::F32 => 1,
            GgmlType::Q5_0 => 32,
            GgmlType::Q8_0 => 32,
            GgmlType::Q4_K => 256,
            GgmlType::Q6_K => 256,
        }
    }

    /// Packed bytes per block (§3: F32 4, Q5_0 22, Q8_0 34, Q4_K 144, Q6_K 210).
    #[must_use]
    pub fn block_bytes(self) -> u64 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::Q5_0 => 22,
            GgmlType::Q8_0 => 34,
            GgmlType::Q4_K => 144,
            GgmlType::Q6_K => 210,
        }
    }

    fn index(self) -> usize {
        match self {
            GgmlType::F32 => 0,
            GgmlType::Q5_0 => 1,
            GgmlType::Q8_0 => 2,
            GgmlType::Q4_K => 3,
            GgmlType::Q6_K => 4,
        }
    }
}

/// Per-type tensor-count expectation (§2.3 / intake checklist item 3).
pub(crate) const EXPECTED_TENSOR_COUNT_PER_TYPE: [(GgmlType, u64); 5] = [
    (GgmlType::F32, 65),
    (GgmlType::Q4_K, 16),
    (GgmlType::Q5_0, 176),
    (GgmlType::Q6_K, 16),
    (GgmlType::Q8_0, 17),
];

/// Per-type element-total expectation (§2.3).
pub(crate) const EXPECTED_ELEMENTS_PER_TYPE: [(GgmlType, u64); 5] = [
    (GgmlType::F32, 62_400),
    (GgmlType::Q4_K, 39_321_600),
    (GgmlType::Q5_0, 231_014_400),
    (GgmlType::Q6_K, 39_321_600),
    (GgmlType::Q8_0, 52_101_120),
];

// ---------------------------------------------------------------------------
// Typed fail-closed diagnostics
// ---------------------------------------------------------------------------

/// A typed, machine-parseable admission rejection.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionError {
    /// First four bytes are not `"GGUF"`.
    InvalidMagic { magic: u32 },
    /// Container version is not 3.
    UnsupportedVersion { version: u32 },
    /// `tensor_count` != 290 (checked before any tensor allocation).
    TensorCountMismatch { expected: u64, actual: u64 },
    /// `metadata_kv_count` != 37 (checked before any KV allocation).
    MetadataKvCountMismatch { expected: u64, actual: u64 },
    /// Input ended while a parsed field was still expected.
    TruncatedFile { needed: u64, available: u64 },
    /// Metadata key exceeds the 128-byte ceiling.
    KeyTooLong { ceiling: u64, actual: u64 },
    /// Metadata value string (scalar or array element) exceeds 4096 bytes.
    StringTooLong { key: String, ceiling: u64, actual: u64 },
    /// Metadata key not in the admitted 37-key set.
    UnknownMetadataKey { key: String },
    /// The same admitted key appears more than once.
    DuplicateMetadataKey { key: String },
    /// `general.architecture` != `"llama"`.
    ArchitectureMismatch { actual: String },
    /// A metadata value contradicts the contract (wrong type or wrong value).
    MetadataValueMismatch { key: String, expected: String, actual: String },
    /// A tokenizer array count is not the exact contracted value.
    TokenizerArrayCountMismatch { array: String, expected: u64, actual: u64 },
    /// A BOOL metadata value is neither 0 nor 1.
    MalformedBool { value: u8 },
    /// Tensor name exceeds the 128-byte ceiling.
    TensorNameTooLong { ceiling: u64, actual: u64 },
    /// The same tensor name appears more than once in the tensor-info table.
    DuplicateTensorName { name: String },
    /// `n_dims` is not in {1, 2} for the pinned row.
    TensorDimCountMismatch { name: String, n_dims: u32 },
    /// A dimension exceeds the 65536 ceiling.
    TensorDimTooLarge { name: String, dim: u64 },
    /// A tensor dtype id is outside the admitted GGML set.
    UnknownDtype { name: String, dtype_id: u32 },
    /// Tensor data offset is not 32-aligned (no `general.alignment` override).
    MisalignedTensorOffset { name: String, offset: u64 },
    /// Tensor element count is not a multiple of the type's block size.
    TensorElementsNotBlockAligned { name: String, elements: u64, block_elements: u64 },
    /// Checked offset/length/size arithmetic overflowed u64.
    ArithmeticOverflow { context: String },
    /// A tensor byte range starts before the previous range ends.
    OverlappingTensorRanges { name: String, offset: u64, previous_end: u64 },
    /// A tensor byte range starts after the aligned end of the previous range.
    NonSequentialTensorOffsets { name: String, offset: u64, expected: u64 },
    /// The tensor-data region extends past the end of the file.
    TruncatedTensorData { data_end: u64, file_size: u64 },
    /// Per-type tensor count contradicts §2.3.
    PerTypeTensorCountMismatch { ggml_type: GgmlType, expected: u64, actual: u64 },
    /// Per-type element total contradicts §2.3.
    PerTypeElementCountMismatch { ggml_type: GgmlType, expected: u64, actual: u64 },
    /// Grand total tensor elements != 361,821,120.
    TotalElementsMismatch { expected: u64, actual: u64 },
    /// Exact pinned file size not matched.
    FileSizeMismatch { expected: u64, actual: u64 },
    /// Whole-file SHA-256 != `2fa3f013…bac9c2`.
    Sha256Mismatch { expected_hex: String, actual_hex: String },
    /// A parsed string is not valid UTF-8.
    InvalidUtf8 { context: String },
    /// Underlying filesystem error in `admit_file`.
    Io(String),
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdmissionError::InvalidMagic { magic } => {
                write!(f, "invalid GGUF magic: expected \"GGUF\", got {magic:#010x}")
            }
            AdmissionError::UnsupportedVersion { version } => {
                write!(f, "unsupported GGUF version {version} (pinned row is version 3)")
            }
            AdmissionError::TensorCountMismatch { expected, actual } => {
                write!(f, "tensor_count {actual} != expected {expected}")
            }
            AdmissionError::MetadataKvCountMismatch { expected, actual } => {
                write!(f, "metadata_kv_count {actual} != expected {expected}")
            }
            AdmissionError::TruncatedFile { needed, available } => {
                write!(f, "truncated file: needed {needed} bytes, only {available} available")
            }
            AdmissionError::KeyTooLong { ceiling, actual } => {
                write!(f, "metadata key length {actual} exceeds ceiling {ceiling}")
            }
            AdmissionError::StringTooLong { key, ceiling, actual } => {
                write!(f, "string for {key:?} is {actual} bytes, exceeds ceiling {ceiling}")
            }
            AdmissionError::UnknownMetadataKey { key } => {
                write!(f, "unknown metadata key {key:?} (pinned 37-key set only)")
            }
            AdmissionError::DuplicateMetadataKey { key } => {
                write!(f, "duplicate metadata key {key:?}")
            }
            AdmissionError::ArchitectureMismatch { actual } => {
                write!(f, "architecture {actual:?} != pinned \"llama\"")
            }
            AdmissionError::MetadataValueMismatch {
                key,
                expected,
                actual,
            } => {
                write!(f, "metadata value for {key:?} contradicts the contract: expected {expected}, got {actual}")
            }
            AdmissionError::TokenizerArrayCountMismatch { array, expected, actual } => {
                write!(f, "tokenizer array {array:?} has {actual} elements, expected exactly {expected}")
            }
            AdmissionError::MalformedBool { value } => {
                write!(f, "malformed BOOL metadata value {value} (must be 0 or 1)")
            }
            AdmissionError::TensorNameTooLong { ceiling, actual } => {
                write!(f, "tensor name length {actual} exceeds ceiling {ceiling}")
            }
            AdmissionError::DuplicateTensorName { name } => {
                write!(f, "duplicate tensor name {name:?} in the tensor-info table")
            }
            AdmissionError::TensorDimCountMismatch { name, n_dims } => {
                write!(f, "tensor {name:?} declares n_dims {n_dims}, pinned row allows 1..=2")
            }
            AdmissionError::TensorDimTooLarge { name, dim } => {
                write!(f, "tensor {name:?} has dimension {dim}, exceeds ceiling {MAX_DIM}")
            }
            AdmissionError::UnknownDtype { name, dtype_id } => {
                write!(f, "tensor {name:?} has unknown dtype id {dtype_id}")
            }
            AdmissionError::MisalignedTensorOffset { name, offset } => {
                write!(f, "tensor {name:?} data offset {offset} is not {GGUF_ALIGNMENT}-aligned")
            }
            AdmissionError::TensorElementsNotBlockAligned { name, elements, block_elements } => {
                write!(f, "tensor {name:?} has {elements} elements, not a multiple of block size {block_elements}")
            }
            AdmissionError::ArithmeticOverflow { context } => {
                write!(f, "arithmetic overflow while computing {context}")
            }
            AdmissionError::OverlappingTensorRanges { name, offset, previous_end } => {
                write!(f, "tensor {name:?} range starts at {offset}, overlapping previous range ending at {previous_end}")
            }
            AdmissionError::NonSequentialTensorOffsets { name, offset, expected } => {
                write!(f, "tensor {name:?} data offset {offset} is out of order (expected {expected})")
            }
            AdmissionError::TruncatedTensorData { data_end, file_size } => {
                write!(f, "tensor-data region ends at {data_end}, past file size {file_size}")
            }
            AdmissionError::PerTypeTensorCountMismatch { ggml_type, expected, actual } => {
                write!(f, "{} tensor count {actual} != expected {expected}", ggml_type.name())
            }
            AdmissionError::PerTypeElementCountMismatch { ggml_type, expected, actual } => {
                write!(f, "{} element total {actual} != expected {expected}", ggml_type.name())
            }
            AdmissionError::TotalElementsMismatch { expected, actual } => {
                write!(f, "total tensor elements {actual} != expected {expected}")
            }
            AdmissionError::FileSizeMismatch { expected, actual } => {
                write!(f, "file size {actual} != pinned {expected}")
            }
            AdmissionError::Sha256Mismatch { expected_hex, actual_hex } => {
                write!(f, "SHA-256 mismatch: expected {expected_hex}, got {actual_hex}")
            }
            AdmissionError::InvalidUtf8 { context } => {
                write!(f, "invalid UTF-8 while reading {context}")
            }
            AdmissionError::Io(err) => write!(f, "io error: {err}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Admission descriptors
// ---------------------------------------------------------------------------

/// One metadata key-value pair, in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataKv {
    pub key: String,
    pub value: MetadataValue,
}

/// Typed metadata values admitted by the pinned row.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    String(String),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    StringArray(Vec<String>),
    Int32Array(Vec<i32>),
}

/// One validated tensor descriptor from the GGUF tensor-info table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub ggml_type: GgmlType,
    /// Logical dimensions in GGUF order (1 or 2 entries for the pinned row).
    pub dims: Vec<u64>,
    /// Total logical elements (`product(dims)`).
    pub element_count: u64,
    /// Number of packed blocks (`element_count / block_elements`, exact).
    pub blocks: u64,
    /// Packed byte length of this tensor (`blocks * block_bytes`).
    pub byte_len: u64,
    /// Offset of this tensor's bytes within the tensor-data region.
    pub offset_in_data: u64,
    /// Absolute file offset (`data_offset + offset_in_data`).
    pub absolute_offset: u64,
}

/// The validated admission for the pinned row.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufAdmission {
    /// Frozen intake schema identity (`gi0-model-contract v1.0.0`).
    pub schema: &'static str,
    /// Verified whole-file size (== `PINNED_FILE_SIZE`).
    pub file_size: u64,
    /// Verified whole-file SHA-256 (== `PINNED_SHA256_HEX`).
    pub sha256_hex: String,
    /// The 37 metadata KVs in file order.
    pub metadata: Vec<MetadataKv>,
    /// The 290 tensor descriptors in tensor-info-table order.
    pub tensors: Vec<TensorDescriptor>,
    /// Absolute file offset where the tensor-data region begins (32-aligned).
    pub data_offset: u64,
    /// Total tensor-data bytes (== sum of every tensor's `byte_len`).
    pub data_len: u64,
}

// ---------------------------------------------------------------------------
// Admission entry points
// ---------------------------------------------------------------------------

/// Admit a byte buffer as the pinned SmolLM2-360M-Instruct Q4_K_M row.
///
/// Every intake fact is validated fail-closed, including the whole-file
/// SHA-256. The only errors that can reach this call from `admit_file` are
/// `Io`; callers that already know the exact pinned size may call this
/// directly. Identical bytes always yield identical descriptors.
///
/// # Errors
///
/// Returns the first typed `AdmissionError` that the input contradicts.
pub fn admit_gguf(bytes: &[u8]) -> Result<GgufAdmission, AdmissionError> {
    let file_size = bytes.len() as u64;
    let mut cur = Cursor::new(bytes);

    // 1. Container identity: magic + version.
    let magic = cur.u32_le()?;
    if magic != GGUF_MAGIC {
        return Err(AdmissionError::InvalidMagic { magic });
    }
    let version = cur.u32_le()?;
    if version != GGUF_VERSION {
        return Err(AdmissionError::UnsupportedVersion { version });
    }

    // 2. Exact count ceilings — checked before any allocation sized by them.
    let tensor_count = cur.u64_le()?;
    if tensor_count != EXPECTED_TENSOR_COUNT {
        return Err(AdmissionError::TensorCountMismatch {
            expected: EXPECTED_TENSOR_COUNT,
            actual: tensor_count,
        });
    }
    let kv_count = cur.u64_le()?;
    if kv_count != EXPECTED_KV_COUNT {
        return Err(AdmissionError::MetadataKvCountMismatch {
            expected: EXPECTED_KV_COUNT,
            actual: kv_count,
        });
    }

    // 3. The 37 metadata KVs.
    let mut metadata = Vec::with_capacity(EXPECTED_KV_COUNT as usize);
    let mut seen_keys = HashSet::with_capacity(EXPECTED_KV_COUNT as usize);
    for _ in 0..kv_count {
        let key = read_key(&mut cur)?;
        if !ADMITTED_KEYS.contains(&key.as_str()) {
            return Err(AdmissionError::UnknownMetadataKey { key });
        }
        if !seen_keys.insert(key.clone()) {
            return Err(AdmissionError::DuplicateMetadataKey { key });
        }
        let expected = expected_for_key(&key);
        let value = parse_kv_value(&mut cur, &key, &expected)?;
        validate_kv_value(&key, &expected, &value)?;
        metadata.push(MetadataKv { key, value });
    }

    // 4. The 290-entry tensor-info table.
    let mut tensors = Vec::with_capacity(EXPECTED_TENSOR_COUNT as usize);
    let mut seen_tensor_names = HashSet::with_capacity(EXPECTED_TENSOR_COUNT as usize);
    let mut per_type_count = [0u64; 5];
    let mut per_type_elements = [0u64; 5];
    let mut total_data_len: u64 = 0;
    let mut prev_end: u64 = 0;

    for _ in 0..tensor_count {
        let name = read_tensor_name(&mut cur)?;
        if !seen_tensor_names.insert(name.clone()) {
            return Err(AdmissionError::DuplicateTensorName { name });
        }
        let n_dims = cur.u32_le()?;
        if n_dims == 0 || n_dims > MAX_TENSOR_DIMS {
            return Err(AdmissionError::TensorDimCountMismatch {
                name: name.clone(),
                n_dims,
            });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = cur.u64_le()?;
            if dim > MAX_DIM {
                return Err(AdmissionError::TensorDimTooLarge {
                    name: name.clone(),
                    dim,
                });
            }
            elements = elements.checked_mul(dim).ok_or_else(|| {
                AdmissionError::ArithmeticOverflow {
                    context: format!("dimensions of {name}"),
                }
            })?;
            dims.push(dim);
        }
        let dtype_id = cur.u32_le()?;
        let ggml_type = GgmlType::from_id(dtype_id).ok_or_else(|| AdmissionError::UnknownDtype {
            name: name.clone(),
            dtype_id,
        })?;
        let offset_in_data = cur.u64_le()?;

        // Byte-range validation: alignment first, then checked arithmetic,
        // then overlap/order, so a crafted offset cannot reach unchecked math.
        if offset_in_data % GGUF_ALIGNMENT != 0 {
            return Err(AdmissionError::MisalignedTensorOffset {
                name: name.clone(),
                offset: offset_in_data,
            });
        }
        let block_elems = ggml_type.block_elements();
        let block_bytes = ggml_type.block_bytes();
        if elements % block_elems != 0 {
            return Err(AdmissionError::TensorElementsNotBlockAligned {
                name: name.clone(),
                elements,
                block_elements: block_elems,
            });
        }
        let blocks = elements / block_elems;
        let byte_len = blocks.checked_mul(block_bytes).ok_or_else(|| {
            AdmissionError::ArithmeticOverflow {
                context: format!("byte length of {name}"),
            }
        })?;
        let end = offset_in_data.checked_add(byte_len).ok_or_else(|| {
            AdmissionError::ArithmeticOverflow {
                context: format!("byte-range end of {name}"),
            }
        })?;
        if offset_in_data < prev_end {
            return Err(AdmissionError::OverlappingTensorRanges {
                name: name.clone(),
                offset: offset_in_data,
                previous_end: prev_end,
            });
        }
        let expected_offset = align_up(prev_end, GGUF_ALIGNMENT);
        if offset_in_data != expected_offset {
            return Err(AdmissionError::NonSequentialTensorOffsets {
                name: name.clone(),
                offset: offset_in_data,
                expected: expected_offset,
            });
        }
        prev_end = end;
        total_data_len = total_data_len.checked_add(byte_len).ok_or_else(|| {
            AdmissionError::ArithmeticOverflow {
                context: "total tensor-data length".into(),
            }
        })?;
        per_type_count[ggml_type.index()] += 1;
        per_type_elements[ggml_type.index()] =
            per_type_elements[ggml_type.index()]
                .checked_add(elements)
                .ok_or_else(|| AdmissionError::ArithmeticOverflow {
                    context: format!("element total of {name}"),
                })?;
        tensors.push(TensorDescriptor {
            name,
            ggml_type,
            dims,
            element_count: elements,
            blocks,
            byte_len,
            offset_in_data,
            absolute_offset: 0, // filled once the data region offset is known
        });
    }

    // 5. Tensor-data region bounds.
    let data_offset = align_up(cur.pos as u64, GGUF_ALIGNMENT);
    let data_end = data_offset.checked_add(total_data_len).ok_or_else(|| {
        AdmissionError::ArithmeticOverflow {
            context: "tensor-data region end".into(),
        }
    })?;
    if data_end > file_size {
        return Err(AdmissionError::TruncatedTensorData {
            data_end,
            file_size,
        });
    }
    for t in &mut tensors {
        t.absolute_offset = data_offset + t.offset_in_data;
    }

    // 6. Aggregate intake facts (per-type counts and element totals).
    for &(ggml_type, expected) in &EXPECTED_TENSOR_COUNT_PER_TYPE {
        let actual = per_type_count[ggml_type.index()];
        if actual != expected {
            return Err(AdmissionError::PerTypeTensorCountMismatch {
                ggml_type,
                expected,
                actual,
            });
        }
    }
    for &(ggml_type, expected) in &EXPECTED_ELEMENTS_PER_TYPE {
        let actual = per_type_elements[ggml_type.index()];
        if actual != expected {
            return Err(AdmissionError::PerTypeElementCountMismatch {
                ggml_type,
                expected,
                actual,
            });
        }
    }
    let total_elements: u64 = per_type_elements.iter().sum();
    if total_elements != EXPECTED_TOTAL_ELEMENTS {
        return Err(AdmissionError::TotalElementsMismatch {
            expected: EXPECTED_TOTAL_ELEMENTS,
            actual: total_elements,
        });
    }

    // 7. Identity facts: exact file size, then whole-file SHA-256.
    if file_size != PINNED_FILE_SIZE {
        return Err(AdmissionError::FileSizeMismatch {
            expected: PINNED_FILE_SIZE,
            actual: file_size,
        });
    }
    let digest = sha256(bytes);
    if digest != PINNED_SHA256 {
        return Err(AdmissionError::Sha256Mismatch {
            expected_hex: PINNED_SHA256_HEX.to_string(),
            actual_hex: hex(&digest),
        });
    }

    Ok(GgufAdmission {
        schema: CONTRACT_SCHEMA,
        file_size,
        sha256_hex: PINNED_SHA256_HEX.to_string(),
        metadata,
        tensors,
        data_offset,
        data_len: total_data_len,
    })
}

/// Admit the pinned model file at `path`.
///
/// The exact-size ceiling is checked **before** the file is read, so no
/// allocation proportional to an on-disk size precedes validation.
///
/// # Errors
///
/// Returns `AdmissionError::Io` for filesystem failures,
/// `AdmissionError::FileSizeMismatch` for a size other than the pinned
/// 270,590,880 bytes, or the first typed structural/identity `AdmissionError`.
pub fn admit_file<P: AsRef<Path>>(path: P) -> Result<GgufAdmission, AdmissionError> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path).map_err(|err| AdmissionError::Io(err.to_string()))?;
    if metadata.len() != PINNED_FILE_SIZE {
        return Err(AdmissionError::FileSizeMismatch {
            expected: PINNED_FILE_SIZE,
            actual: metadata.len(),
        });
    }
    let bytes = std::fs::read(path).map_err(|err| AdmissionError::Io(err.to_string()))?;
    admit_gguf(&bytes)
}

// ---------------------------------------------------------------------------
// Expected metadata values (the bounded input — every contracted value)
// ---------------------------------------------------------------------------

/// The exact contracted value for one admitted metadata key.
#[derive(Debug, Clone, Copy)]
enum ExpectedValue {
    StringValue(&'static str),
    Uint32Value(u32),
    Int32Value(i32),
    Float32Value(f32),
    BoolValue(bool),
    StringArrayValue(&'static [&'static str]),
    StringArrayLen(u64),
    Int32ArrayLen(u64),
}

impl ExpectedValue {
    fn tag(&self) -> u32 {
        match self {
            ExpectedValue::StringValue(_) => 8,
            ExpectedValue::Uint32Value(_) => 4,
            ExpectedValue::Int32Value(_) => 5,
            ExpectedValue::Float32Value(_) => 6,
            ExpectedValue::BoolValue(_) => 7,
            ExpectedValue::StringArrayValue(_)
            | ExpectedValue::StringArrayLen(_)
            | ExpectedValue::Int32ArrayLen(_) => 9,
        }
    }
}

/// The chat template stored verbatim (`tokenizer.chat_template`, contract §6).
/// The `\n` escapes below are real newline bytes (0x0a) — verified against the
/// pinned file on 2026-08-05 (368 bytes, no literal backslashes). Note: the
/// contract §6 / evidence reproduction shows a trailing `%}}`; the pinned file
/// itself stores `{% endif %}` (368 bytes) — the file wins (hash-pinned).
pub(crate) const CHAT_TEMPLATE: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

#[allow(clippy::too_many_lines)]
fn expected_for_key(key: &str) -> ExpectedValue {
    match key {
        "general.architecture" => ExpectedValue::StringValue("llama"),
        "general.type" => ExpectedValue::StringValue("model"),
        "general.name" => ExpectedValue::StringValue("Smollm2 360M 8k Lc100K Mix1 Ep2"),
        "general.organization" => ExpectedValue::StringValue("Loubnabnl"),
        "general.finetune" => ExpectedValue::StringValue("8k-lc100k-mix1-ep2"),
        "general.basename" => ExpectedValue::StringValue("smollm2"),
        "general.size_label" => ExpectedValue::StringValue("360M"),
        "general.license" => ExpectedValue::StringValue("apache-2.0"),
        "general.languages" => ExpectedValue::StringArrayValue(&["en"]),
        "llama.block_count" => ExpectedValue::Uint32Value(32),
        "llama.context_length" => ExpectedValue::Uint32Value(8192),
        "llama.embedding_length" => ExpectedValue::Uint32Value(960),
        "llama.feed_forward_length" => ExpectedValue::Uint32Value(2560),
        "llama.attention.head_count" => ExpectedValue::Uint32Value(15),
        "llama.attention.head_count_kv" => ExpectedValue::Uint32Value(5),
        "llama.rope.freq_base" => ExpectedValue::Float32Value(100_000.0),
        "llama.attention.layer_norm_rms_epsilon" => ExpectedValue::Float32Value(1e-5),
        "general.file_type" => ExpectedValue::Uint32Value(15),
        "llama.vocab_size" => ExpectedValue::Uint32Value(49_152),
        "llama.rope.dimension_count" => ExpectedValue::Uint32Value(64),
        "tokenizer.ggml.add_space_prefix" => ExpectedValue::BoolValue(false),
        "tokenizer.ggml.add_bos_token" => ExpectedValue::BoolValue(false),
        "tokenizer.ggml.model" => ExpectedValue::StringValue("gpt2"),
        "tokenizer.ggml.pre" => ExpectedValue::StringValue("smollm"),
        "tokenizer.ggml.tokens" => ExpectedValue::StringArrayLen(EXPECTED_TOKENS),
        "tokenizer.ggml.token_type" => ExpectedValue::Int32ArrayLen(EXPECTED_TOKEN_TYPE),
        "tokenizer.ggml.merges" => ExpectedValue::StringArrayLen(EXPECTED_MERGES),
        "tokenizer.ggml.bos_token_id" => ExpectedValue::Uint32Value(1),
        "tokenizer.ggml.eos_token_id" => ExpectedValue::Uint32Value(2),
        "tokenizer.ggml.unknown_token_id" => ExpectedValue::Uint32Value(0),
        "tokenizer.ggml.padding_token_id" => ExpectedValue::Uint32Value(2),
        "tokenizer.chat_template" => ExpectedValue::StringValue(CHAT_TEMPLATE),
        "general.quantization_version" => ExpectedValue::Uint32Value(2),
        "quantize.imatrix.file" => ExpectedValue::StringValue(
            "/models_out/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct.imatrix",
        ),
        "quantize.imatrix.dataset" => {
            ExpectedValue::StringValue("/training_dir/calibration_datav3.txt")
        }
        "quantize.imatrix.entries_count" => ExpectedValue::Int32Value(224),
        "quantize.imatrix.chunks_count" => ExpectedValue::Int32Value(141),
        _ => unreachable!("key admission guarantees membership in the 37-key set"),
    }
}

// ---------------------------------------------------------------------------
// Cursor + primitive readers (all checked arithmetic)
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], AdmissionError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| AdmissionError::ArithmeticOverflow {
                context: "cursor advance".into(),
            })?;
        if end > self.bytes.len() {
            return Err(AdmissionError::TruncatedFile {
                needed: end as u64,
                available: self.bytes.len() as u64,
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, AdmissionError> {
        Ok(self.take(1)?[0])
    }

    fn u32_le(&mut self) -> Result<u32, AdmissionError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32_le(&mut self) -> Result<i32, AdmissionError> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64_le(&mut self) -> Result<u64, AdmissionError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn f32_le(&mut self) -> Result<f32, AdmissionError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn read_key(cur: &mut Cursor<'_>) -> Result<String, AdmissionError> {
    let len = cur.u64_le()?;
    if len > MAX_KEY_BYTES {
        return Err(AdmissionError::KeyTooLong {
            ceiling: MAX_KEY_BYTES,
            actual: len,
        });
    }
    let bytes = cur.take(len as usize)?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| AdmissionError::InvalidUtf8 { context: "metadata key".into() })
}

fn read_string(cur: &mut Cursor<'_>, key: &str) -> Result<String, AdmissionError> {
    let len = cur.u64_le()?;
    if len > MAX_STRING_BYTES {
        return Err(AdmissionError::StringTooLong {
            key: key.to_string(),
            ceiling: MAX_STRING_BYTES,
            actual: len,
        });
    }
    let bytes = cur.take(len as usize)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| AdmissionError::InvalidUtf8 {
        context: format!("value of {key}"),
    })
}

fn read_tensor_name(cur: &mut Cursor<'_>) -> Result<String, AdmissionError> {
    let len = cur.u64_le()?;
    if len > MAX_TENSOR_NAME_BYTES {
        return Err(AdmissionError::TensorNameTooLong {
            ceiling: MAX_TENSOR_NAME_BYTES,
            actual: len,
        });
    }
    let bytes = cur.take(len as usize)?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| AdmissionError::InvalidUtf8 { context: "tensor name".into() })
}

fn parse_kv_value(
    cur: &mut Cursor<'_>,
    key: &str,
    expected: &ExpectedValue,
) -> Result<MetadataValue, AdmissionError> {
    let tag = cur.u32_le()?;
    if tag != expected.tag() {
        return Err(AdmissionError::MetadataValueMismatch {
            key: key.to_string(),
            expected: format!("GGUF value tag {}", expected.tag()),
            actual: format!("GGUF value tag {tag}"),
        });
    }
    match expected {
        ExpectedValue::StringValue(_) => Ok(MetadataValue::String(read_string(cur, key)?)),
        ExpectedValue::Uint32Value(_) => Ok(MetadataValue::Uint32(cur.u32_le()?)),
        ExpectedValue::Int32Value(_) => Ok(MetadataValue::Int32(cur.i32_le()?)),
        ExpectedValue::Float32Value(_) => Ok(MetadataValue::Float32(cur.f32_le()?)),
        ExpectedValue::BoolValue(_) => match cur.u8()? {
            0 => Ok(MetadataValue::Bool(false)),
            1 => Ok(MetadataValue::Bool(true)),
            other => Err(AdmissionError::MalformedBool { value: other }),
        },
        ExpectedValue::StringArrayValue(items) => {
            let count = items.len() as u64;
            Ok(MetadataValue::StringArray(parse_string_array(cur, key, count)?))
        }
        ExpectedValue::StringArrayLen(count) => {
            Ok(MetadataValue::StringArray(parse_string_array(cur, key, *count)?))
        }
        ExpectedValue::Int32ArrayLen(count) => {
            let elem_tag = cur.u32_le()?;
            if elem_tag != 5 {
                return Err(AdmissionError::MetadataValueMismatch {
                    key: key.to_string(),
                    expected: "array<I32>".into(),
                    actual: format!("array element tag {elem_tag}"),
                });
            }
            let n = cur.u64_le()?;
            if n != *count {
                return Err(AdmissionError::TokenizerArrayCountMismatch {
                    array: key.to_string(),
                    expected: *count,
                    actual: n,
                });
            }
            let mut items = Vec::with_capacity(*count as usize);
            for _ in 0..*count {
                items.push(cur.i32_le()?);
            }
            Ok(MetadataValue::Int32Array(items))
        }
    }
}

fn parse_string_array(
    cur: &mut Cursor<'_>,
    key: &str,
    count: u64,
) -> Result<Vec<String>, AdmissionError> {
    let elem_tag = cur.u32_le()?;
    if elem_tag != 8 {
        return Err(AdmissionError::MetadataValueMismatch {
            key: key.to_string(),
            expected: "array<STRING>".into(),
            actual: format!("array element tag {elem_tag}"),
        });
    }
    let n = cur.u64_le()?;
    if n != count {
        return Err(AdmissionError::TokenizerArrayCountMismatch {
            array: key.to_string(),
            expected: count,
            actual: n,
        });
    }
    let mut items = Vec::with_capacity(count as usize);
    for _ in 0..count {
        items.push(read_string(cur, key)?);
    }
    Ok(items)
}

fn validate_kv_value(
    key: &str,
    expected: &ExpectedValue,
    value: &MetadataValue,
) -> Result<(), AdmissionError> {
    let mismatch =
        |expected: String, actual: String| AdmissionError::MetadataValueMismatch {
            key: key.to_string(),
            expected,
            actual,
        };
    match (expected, value) {
        (ExpectedValue::StringValue(exp), MetadataValue::String(act)) => {
            if *exp != act {
                if key == "general.architecture" {
                    return Err(AdmissionError::ArchitectureMismatch {
                        actual: act.clone(),
                    });
                }
                return Err(mismatch(
                    format!("string {exp:?}"),
                    format!("string {act:?}"),
                ));
            }
            Ok(())
        }
        (ExpectedValue::Uint32Value(exp), MetadataValue::Uint32(act)) => {
            if *exp != *act {
                Err(mismatch(format!("uint32 {exp}"), format!("uint32 {act}")))
            } else {
                Ok(())
            }
        }
        (ExpectedValue::Int32Value(exp), MetadataValue::Int32(act)) => {
            if *exp != *act {
                Err(mismatch(format!("int32 {exp}"), format!("int32 {act}")))
            } else {
                Ok(())
            }
        }
        (ExpectedValue::Float32Value(exp), MetadataValue::Float32(act)) => {
            if exp.to_bits() != act.to_bits() {
                Err(mismatch(
                    format!("float32 bits {:08x}", exp.to_bits()),
                    format!("float32 bits {:08x}", act.to_bits()),
                ))
            } else {
                Ok(())
            }
        }
        (ExpectedValue::BoolValue(exp), MetadataValue::Bool(act)) => {
            if *exp != *act {
                Err(mismatch(format!("bool {exp}"), format!("bool {act}")))
            } else {
                Ok(())
            }
        }
        (ExpectedValue::StringArrayValue(exp), MetadataValue::StringArray(act)) => {
            let act_refs: Vec<&str> = act.iter().map(String::as_str).collect();
            if *exp != act_refs.as_slice() {
                Err(mismatch(
                    format!("array {exp:?}"),
                    format!("array {act:?}"),
                ))
            } else {
                Ok(())
            }
        }
        // Array counts were validated exactly at parse time.
        (ExpectedValue::StringArrayLen(_), MetadataValue::StringArray(_)) => Ok(()),
        (ExpectedValue::Int32ArrayLen(_), MetadataValue::Int32Array(_)) => Ok(()),
        _ => unreachable!("value variant is enforced by the parse-time tag check"),
    }
}

pub(crate) fn align_up(v: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    v.wrapping_add(align - 1) & !(align - 1)
}
// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), pure Rust — no new dependency is permitted by the
// unit's write scope, so the whole-file digest is computed in-module.
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
    0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
    0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
    0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
    0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
    0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
    0xc671_78f2,
];

pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut w = [0u32; 64];

    let mut compress = |block: &[u8; 64]| {
        for (j, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                block[4 * j],
                block[4 * j + 1],
                block[4 * j + 2],
                block[4 * j + 3],
            ]);
        }
        for j in 16..64 {
            let s0 = w[j - 15].rotate_right(7) ^ w[j - 15].rotate_right(18) ^ (w[j - 15] >> 3);
            let s1 = w[j - 2].rotate_right(17) ^ w[j - 2].rotate_right(19) ^ (w[j - 2] >> 10);
            w[j] = w[j - 16]
                .wrapping_add(s0)
                .wrapping_add(w[j - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for (j, &k) in SHA256_K.iter().enumerate() {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(w[j]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    };

    let mut blocks = data.chunks_exact(64);
    for block in &mut blocks {
        let block: [u8; 64] = block.try_into().expect("64-byte chunk");
        compress(&block);
    }
    let rem = blocks.remainder();

    // FIPS 180-4 padding: 0x80, zeroes to 56 mod 64, then the 64-bit
    // big-endian bit length. `tail` holds at most two 64-byte blocks.
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut tail = [0u8; 128];
    let mut n = rem.len();
    tail[..n].copy_from_slice(rem);
    tail[n] = 0x80;
    n += 1;
    while n % 64 != 56 {
        tail[n] = 0;
        n += 1;
    }
    tail[n..n + 8].copy_from_slice(&bit_len.to_be_bytes());
    n += 8;
    for chunk in tail[..n].chunks_exact(64) {
        let chunk: [u8; 64] = chunk.try_into().expect("64-byte chunk");
        compress(&chunk);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
