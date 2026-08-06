//! `QuantizedTensorLayout` — the **sole admitted quantized storage contract**
//! (GI1-2; CTO report `1e7602b1`, memo `fdc2a448`, decision (b)).
//!
//! One pinned row only: SmolLM2-360M-Instruct Q4_K_M, with its closed GGML
//! dtype set `{F32, Q5_0, Q8_0, Q4_K, Q6_K}` (model contract §2.3 / §3,
//! `gi0-model-contract.md` v1.0.0). Every layout resolves from the §3 block
//! table and carries the full CTO field set:
//!
//! 1. **admitted GGML/GGUF format id** — GGML_TYPE ids F32=0, Q5_0=6,
//!    Q8_0=8, Q4_K=12, Q6_K=14 (the complete closed set; FC3);
//! 2. **logical element dtype + dimensions**;
//! 3. **block element count** (Q4_K 256, Q5_0 32, Q6_K 256, Q8_0 32, F32 1);
//! 4. **block byte width** (144 / 22 / 210 / 34 / 4);
//! 5. **scale/min encoding** — Q4_K `d`+`dmin` halves + `scales[12]` +
//!    `qs[128]`; Q5_0 `d` half + `qh[4]` + `qs[16]`; Q8_0 `d` half +
//!    `qs[32]`; Q6_K super-block (256 elems / 210 B); F32 scalar;
//! 6. **alignment** — 32 (GGUF default; the row has no `general.alignment`
//!    override);
//! 7. **byte range** — per-tensor, validated within the file and
//!    block-aligned;
//! 8. **repack identity** — a declared repack carries a measurable identity
//!    (SHA-256 hash) and is explicit; direct native block execution is the
//!    contract (decision (f)); repack is NOT the admitted layout.
//!
//! **Never `DeviceDataType::U8`-as-quantization.** This layout is a distinct,
//! purely descriptive type: it holds *no packed payload bytes at all* (only
//! the byte range and structural metadata). Packed bytes are only ever
//! accessed through the range. The toy packed-u4 layout
//! (`crate::packed_numeric::PackedU4Layout::toy_u4` — 8 values / 4 bytes,
//! scale `f32` + zero_point `u8`, no GGML structure) is **not** a GGML block
//! and is never admitted as one (the exclusion is asserted by test).
//!
//! Construction is fail-closed: `resolve` re-validates the block geometry and
//! the absolute byte range (checked arithmetic, in-file bounds, block
//! alignment, 32-aligned range start) even though the GI1-1 admission already
//! validated the same facts — a hand-built descriptor must fail the same way.

use crate::gguf::{GgmlType, GgufAdmission, TensorDescriptor};
use std::fmt;

/// GGUF default alignment (`GGUF_DEFAULT_ALIGNMENT`; FC1). The pinned row
/// carries no `general.alignment` override, so 32 is the contract alignment.
pub const LAYOUT_ALIGNMENT: u64 = 32;

/// Fixed byte width of a GGML half-precision (`ggml_half`) field.
pub const GGML_HALF_BYTES: u64 = 2;

/// Q4_K `scales[K_SCALE_SIZE]` length (`K_SCALE_SIZE` = 12, `ggml-common.h`).
pub const Q4_K_SCALES_LEN: u64 = 12;

/// Q4_K `qs[QK_K/2]` byte count (128).
pub const Q4_K_QS_BYTES: u64 = 128;

/// Q5_0 `qh[4]` byte count.
pub const Q5_0_QH_BYTES: u64 = 4;

/// Q5_0 `qs[16]` byte count.
pub const Q5_0_QS_BYTES: u64 = 16;

/// Q8_0 `qs[32]` byte count.
pub const Q8_0_QS_BYTES: u64 = 32;

/// Q6_K super-block element count (`QK_K` = 256, `ggml-common.h`).
pub const Q6_K_SUPER_BLOCK_ELEMENTS: u64 = 256;

/// Q6_K super-block byte width (210).
pub const Q6_K_SUPER_BLOCK_BYTES: u64 = 210;

/// F32 scalar byte width (4).
pub const F32_SCALAR_BYTES: u64 = 4;

// ---------------------------------------------------------------------------
// Logical element dtype
// ---------------------------------------------------------------------------

/// Logical element dtype of an admitted layout. Every GGML block type in the
/// pinned row widens each element to F32; no other logical dtype is admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalElementDtype {
    /// IEEE-754 binary32.
    F32,
}

impl LogicalElementDtype {
    /// Canonical spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "f32"
    }

    /// Widened element byte width (4).
    #[must_use]
    pub const fn bytes(self) -> u64 {
        4
    }
}

// ---------------------------------------------------------------------------
// Scale/min encoding (§3 block table)
// ---------------------------------------------------------------------------

/// Structural field widths of one admitted GGML block type's scale/min
/// encoding, mapped 1:1 from the model-contract §3 table. These are
/// structural descriptors (byte counts, array lengths) — never a
/// quantization carrier and never packed payload bytes.
#[allow(non_camel_case_types)] // variants mirror the canonical GGML type names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMinEncoding {
    /// `block_q4_K` (§3): `d` half, `dmin` half, `scales[12]`, `qs[128]`
    /// — 144 bytes total ("8 blocks of 32 elements each", 4.5 bits/weight).
    Q4_K {
        /// `d` — ggml_half, 2 bytes (block scale).
        d_half_bytes: u64,
        /// `dmin` — ggml_half, 2 bytes (block minimum).
        dmin_half_bytes: u64,
        /// `scales[12]` — `K_SCALE_SIZE` bytes.
        scales_len: u64,
        /// `qs[128]` — `QK_K/2` nibble-packed bytes.
        qs_len: u64,
    },
    /// `block_q5_0` (§3): `d` half, `qh[4]`, `qs[16]` — 22 bytes total.
    Q5_0 {
        /// `d` — ggml_half, 2 bytes.
        d_half_bytes: u64,
        /// `qh[4]` — high-nibble bitmask bytes.
        qh_len: u64,
        /// `qs[16]` — low-nibble byte values.
        qs_len: u64,
    },
    /// `block_q8_0` (§3): `d` half, `qs[32]` — 34 bytes total.
    Q8_0 {
        /// `d` — ggml_half, 2 bytes.
        d_half_bytes: u64,
        /// `qs[32]` — signed int8 weight bytes.
        qs_len: u64,
    },
    /// `block_q6_K` (§3): 256-element super-block, 210 bytes total.
    Q6_K {
        /// Super-block element count (`QK_K` = 256).
        super_block_elements: u64,
        /// Super-block byte width (210).
        super_block_bytes: u64,
    },
    /// F32 (§3): plain scalar float32 — 4 bytes, no scale/min fields.
    F32 {
        /// Scalar byte width (4).
        scalar_bytes: u64,
    },
}

impl ScaleMinEncoding {
    /// The §3 block-table encoding for an admitted GGML type.
    #[must_use]
    pub const fn from_ggml_type(ggml_type: GgmlType) -> Self {
        match ggml_type {
            GgmlType::Q4_K => Self::Q4_K {
                d_half_bytes: GGML_HALF_BYTES,
                dmin_half_bytes: GGML_HALF_BYTES,
                scales_len: Q4_K_SCALES_LEN,
                qs_len: Q4_K_QS_BYTES,
            },
            GgmlType::Q5_0 => Self::Q5_0 {
                d_half_bytes: GGML_HALF_BYTES,
                qh_len: Q5_0_QH_BYTES,
                qs_len: Q5_0_QS_BYTES,
            },
            GgmlType::Q8_0 => Self::Q8_0 {
                d_half_bytes: GGML_HALF_BYTES,
                qs_len: Q8_0_QS_BYTES,
            },
            GgmlType::Q6_K => Self::Q6_K {
                super_block_elements: Q6_K_SUPER_BLOCK_ELEMENTS,
                super_block_bytes: Q6_K_SUPER_BLOCK_BYTES,
            },
            GgmlType::F32 => Self::F32 {
                scalar_bytes: F32_SCALAR_BYTES,
            },
        }
    }

    /// Sum of this encoding's structural field widths — must equal the
    /// type's packed block byte width (`GgmlType::block_bytes()`).
    #[must_use]
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::Q4_K {
                d_half_bytes,
                dmin_half_bytes,
                scales_len,
                qs_len,
            } => d_half_bytes + dmin_half_bytes + scales_len + qs_len,
            Self::Q5_0 {
                d_half_bytes,
                qh_len,
                qs_len,
            } => d_half_bytes + qh_len + qs_len,
            Self::Q8_0 {
                d_half_bytes,
                qs_len,
            } => d_half_bytes + qs_len,
            Self::Q6_K {
                super_block_elements: _,
                super_block_bytes,
            } => super_block_bytes,
            Self::F32 { scalar_bytes } => scalar_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Byte range
// ---------------------------------------------------------------------------

/// A validated per-tensor absolute byte range in the admitted file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// First byte of the tensor's packed data (inclusive).
    pub start: u64,
    /// One past the last byte of the tensor's packed data (exclusive).
    pub end: u64,
}

impl ByteRange {
    /// Build a range (callers of the public constructor are responsible for
    /// `end >= start`; resolved ranges are validated).
    #[must_use]
    pub const fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Length in bytes (`end - start`; saturating so a malformed range never
    /// underflows).
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.end == self.start
    }
}

// ---------------------------------------------------------------------------
// Repack identity
// ---------------------------------------------------------------------------

/// SHA-256 digest identifying a declared repack's output bytes. Repacks are
/// explicit and never implicit: a declared repack is hash-accounted and
/// disclosed, and is never the admitted layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepackHash(pub [u8; 32]);

impl RepackHash {
    /// Wrap a SHA-256 digest.
    #[must_use]
    pub const fn new(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Hex spelling of the digest.
    #[must_use]
    pub fn hex(&self) -> String {
        crate::gguf::hex(&self.0)
    }
}

/// Repack identity of a layout. Direct native block execution is the contract
/// (decision (f)); a repack is NOT the admitted layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepackIdentity {
    /// No repack: tensors execute directly in native GGML block form.
    Native,
    /// An explicit declared repack of this tensor's packed bytes,
    /// hash-accounted.
    Declared(RepackHash),
}

// ---------------------------------------------------------------------------
// The layout
// ---------------------------------------------------------------------------

/// The sole admitted quantized storage contract for the pinned row.
///
/// A purely descriptive descriptor: it holds **no packed payload bytes** (no
/// `u8`-as-quantization carrier anywhere in the type). The only byte-related
/// data is the validated `byte_range`; packed bytes are accessed through that
/// range by downstream consumers (GI1-4 tensor view, GI3 decoder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedTensorLayout {
    /// Admitted GGML/GGUF format id (the closed pinned-row set).
    format_id: GgmlType,
    /// Logical element dtype (F32 for every admitted type).
    logical_dtype: LogicalElementDtype,
    /// Logical dimensions in GGUF order (1 or 2 entries for the pinned row).
    dims: Vec<u64>,
    /// Total logical elements (`product(dims)`).
    element_count: u64,
    /// Packed block element count (§3).
    block_elements: u64,
    /// Packed block byte width (§3).
    block_bytes: u64,
    /// Exact block count (`element_count / block_elements`).
    blocks: u64,
    /// Scale/min encoding of the block type (§3).
    scale_min_encoding: ScaleMinEncoding,
    /// Byte alignment of tensor-data offsets (32; GGUF default).
    alignment: u64,
    /// Validated absolute byte range of this tensor in the file.
    byte_range: ByteRange,
    /// Repack identity (native by default).
    repack_identity: RepackIdentity,
}

/// Typed, machine-parseable layout rejection.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizedLayoutError {
    /// Checked offset/size arithmetic overflowed u64.
    ArithmeticOverflow { context: String },
    /// Element count is not a multiple of the type's block size.
    ElementsNotBlockAligned { elements: u64, block_elements: u64 },
    /// The byte range length is not a multiple of the block byte width.
    ByteRangeNotBlockAligned { byte_len: u64, block_bytes: u64 },
    /// The byte range is not the exact `blocks * block_bytes` length.
    ByteLenMismatch { byte_len: u64, expected: u64 },
    /// The byte range extends past the end of the file.
    ByteRangeOutOfBounds {
        start: u64,
        end: u64,
        file_size: u64,
    },
    /// The byte range start is not 32-aligned.
    MisalignedRangeStart { start: u64, alignment: u64 },
    /// The scale/min encoding width contradicts the GGML block byte width.
    EncodingMismatch {
        format_id: GgmlType,
        encoding_bytes: u64,
        expected_bytes: u64,
    },
}

impl fmt::Display for QuantizedLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow { context } => {
                write!(f, "arithmetic overflow while computing {context}")
            }
            Self::ElementsNotBlockAligned {
                elements,
                block_elements,
            } => write!(
                f,
                "tensor has {elements} elements, not a multiple of block size {block_elements}"
            ),
            Self::ByteRangeNotBlockAligned {
                byte_len,
                block_bytes,
            } => write!(
                f,
                "byte-range length {byte_len} is not a multiple of block byte width {block_bytes}"
            ),
            Self::ByteLenMismatch { byte_len, expected } => {
                write!(f, "byte-range length {byte_len} != expected {expected}")
            }
            Self::ByteRangeOutOfBounds {
                start,
                end,
                file_size,
            } => write!(f, "byte range [{start}, {end}) extends past file size {file_size}"),
            Self::MisalignedRangeStart { start, alignment } => {
                write!(f, "byte range start {start} is not {alignment}-aligned")
            }
            Self::EncodingMismatch {
                format_id,
                encoding_bytes,
                expected_bytes,
            } => write!(
                f,
                "{} scale/min encoding sums to {encoding_bytes} bytes, expected block width {expected_bytes}",
                format_id.name()
            ),
        }
    }
}

impl QuantizedTensorLayout {
    /// Resolve the layout for one admitted tensor descriptor, re-validating
    /// the block geometry and the absolute byte range against `file_size`
    /// (checked arithmetic; in-file bounds; block alignment; 32-aligned
    /// start).
    ///
    /// # Errors
    ///
    /// Returns the first typed `QuantizedLayoutError` the descriptor
    /// contradicts.
    pub fn resolve(desc: &TensorDescriptor, file_size: u64) -> Result<Self, QuantizedLayoutError> {
        let format_id = desc.ggml_type;
        let block_elements = format_id.block_elements();
        let block_bytes = format_id.block_bytes();

        // 1. Block geometry: element count must be an exact multiple of the
        //    block element count.
        if desc.element_count % block_elements != 0 {
            return Err(QuantizedLayoutError::ElementsNotBlockAligned {
                elements: desc.element_count,
                block_elements,
            });
        }
        let blocks = desc.element_count / block_elements;

        // 2. Block alignment of the byte range, then exact length.
        if desc.byte_len % block_bytes != 0 {
            return Err(QuantizedLayoutError::ByteRangeNotBlockAligned {
                byte_len: desc.byte_len,
                block_bytes,
            });
        }
        let expected_len = blocks.checked_mul(block_bytes).ok_or_else(|| {
            QuantizedLayoutError::ArithmeticOverflow {
                context: format!("byte length of {}", desc.name),
            }
        })?;
        if desc.byte_len != expected_len {
            return Err(QuantizedLayoutError::ByteLenMismatch {
                byte_len: desc.byte_len,
                expected: expected_len,
            });
        }

        // 3. Byte range: checked arithmetic, in-file bounds, 32-aligned start.
        let start = desc.absolute_offset;
        let end = start.checked_add(desc.byte_len).ok_or_else(|| {
            QuantizedLayoutError::ArithmeticOverflow {
                context: format!("byte-range end of {}", desc.name),
            }
        })?;
        if end > file_size {
            return Err(QuantizedLayoutError::ByteRangeOutOfBounds {
                start,
                end,
                file_size,
            });
        }
        if start % LAYOUT_ALIGNMENT != 0 {
            return Err(QuantizedLayoutError::MisalignedRangeStart {
                start,
                alignment: LAYOUT_ALIGNMENT,
            });
        }

        // 4. Scale/min encoding must agree with the GGML block byte width.
        let scale_min_encoding = ScaleMinEncoding::from_ggml_type(format_id);
        let encoding_bytes = scale_min_encoding.block_bytes();
        if encoding_bytes != block_bytes {
            return Err(QuantizedLayoutError::EncodingMismatch {
                format_id,
                encoding_bytes,
                expected_bytes: block_bytes,
            });
        }

        Ok(Self {
            format_id,
            logical_dtype: LogicalElementDtype::F32,
            dims: desc.dims.clone(),
            element_count: desc.element_count,
            block_elements,
            block_bytes,
            blocks,
            scale_min_encoding,
            alignment: LAYOUT_ALIGNMENT,
            byte_range: ByteRange { start, end },
            repack_identity: RepackIdentity::Native,
        })
    }

    /// Declare an explicit, hash-accounted repack of this tensor's packed
    /// bytes. Direct native block execution remains the admitted layout
    /// (`RepackIdentity::Native`); a repack is a disclosed deviation, never
    /// the admitted layout (decision (f)).
    #[must_use]
    pub fn with_declared_repack(&self, hash: RepackHash) -> Self {
        let mut layout = self.clone();
        layout.repack_identity = RepackIdentity::Declared(hash);
        layout
    }

    /// Admitted GGML/GGUF format id.
    #[must_use]
    pub const fn format_id(&self) -> GgmlType {
        self.format_id
    }

    /// Logical element dtype.
    #[must_use]
    pub const fn logical_dtype(&self) -> LogicalElementDtype {
        self.logical_dtype
    }

    /// Logical dimensions in GGUF order.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Total logical elements (`product(dims)`).
    #[must_use]
    pub const fn element_count(&self) -> u64 {
        self.element_count
    }

    /// Packed block element count (§3).
    #[must_use]
    pub const fn block_elements(&self) -> u64 {
        self.block_elements
    }

    /// Packed block byte width (§3).
    #[must_use]
    pub const fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    /// Exact block count (`element_count / block_elements`).
    #[must_use]
    pub const fn blocks(&self) -> u64 {
        self.blocks
    }

    /// Scale/min encoding of the block type (§3).
    #[must_use]
    pub const fn scale_min_encoding(&self) -> ScaleMinEncoding {
        self.scale_min_encoding
    }

    /// Byte alignment of tensor-data offsets (32; GGUF default).
    #[must_use]
    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Validated absolute byte range of this tensor in the file.
    #[must_use]
    pub const fn byte_range(&self) -> ByteRange {
        self.byte_range
    }

    /// Repack identity (native by default; declared repacks are explicit).
    #[must_use]
    pub const fn repack_identity(&self) -> RepackIdentity {
        self.repack_identity
    }
}

/// Resolve every tensor of an admission to its layout (pinned row: 290
/// layouts in GGUF tensor-table order). Any failing tensor fails the whole
/// resolution fail-closed.
///
/// # Errors
///
/// Returns the first typed `QuantizedLayoutError` any descriptor contradicts.
pub fn resolve_admission(
    admission: &GgufAdmission,
) -> Result<Vec<QuantizedTensorLayout>, QuantizedLayoutError> {
    let mut layouts = Vec::with_capacity(admission.tensors.len());
    for desc in &admission.tensors {
        layouts.push(QuantizedTensorLayout::resolve(desc, admission.file_size)?);
    }
    Ok(layouts)
}
