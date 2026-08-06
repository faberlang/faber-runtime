//! GI1-4 — deterministic, host-readable (CPU) tensor view over the admitted
//! GGUF row: the GI2 seam (campaign §GI1 exit gate bullet 7: "GI2 receives a
//! deterministic host-readable tensor view").
//!
//! Built once from a validated [`GgufAdmission`] (GI1-1) plus the admitted
//! file bytes. For each of the 290 tensors it exposes: name, GGML type,
//! logical dimensions, the [`QuantizedTensorLayout`] (GI1-2), the validated
//! absolute byte range in the file, the element count, and bounded raw-bytes
//! accessors.
//!
//! DETERMINISTIC: entries enumerate in GGUF tensor-table order (the order
//! validated at GI1-1); two loads of the same file yield byte-identical
//! descriptors (`TensorView` and `TensorViewEntry` implement `PartialEq` over
//! the descriptor fields only).
//!
//! HASH-ACCOUNTED: the whole-file SHA-256 (verified at GI1-1 admission)
//! covers every tensor byte range. The view records the digest and every
//! range, so per-tensor coverage is checkable (`per_tensor_covered`,
//! `coverage_ok`) and re-verifiable against any byte buffer
//! (`sha256_matches`).
//!
//! FAIL CLOSED: `build` re-validates every byte range against the admitted
//! file size and the tensor-data region (a hand-rolled admission fails the
//! same way); byte access outside a tensor's declared range is rejected with
//! a typed [`TensorViewError`]; raw accessors never borrow beyond the range
//! they declare.
//!
//! CPU-ONLY: this module is a pure descriptor layer — no device/GPU code
//! path, no packed payload is copied (the file slice is borrowed), and no
//! dequantization happens here (GI2 owns dequant for its logits oracle —
//! §Open questions Q1 default; the view exposes layout facts + bounded raw
//! block bytes only).

use crate::gguf::{hex, sha256, GgmlType, GgufAdmission, PINNED_SHA256};
use crate::quantized_tensor_layout::{
    resolve_admission, ByteRange, QuantizedLayoutError, QuantizedTensorLayout,
};
use std::fmt;

// ---------------------------------------------------------------------------
// Pinned-row named lookup surface (gi0-model-contract §2.3 / task body)
// ---------------------------------------------------------------------------

/// The pinned tensor base names (one-row boundary): `token_embd.weight`,
/// per-layer `attn_norm` / `attn_q` / `attn_k` / `attn_v` / `attn_output` /
/// `ffn_norm` / `ffn_gate` / `ffn_up` / `ffn_down`, and `output_norm.weight`.
///
/// In the file each per-layer name is `blk.N.<base>`; `family` resolves both
/// spellings. Sum of family sizes: 1 + 32×9 + 1 = 290.
pub const PINNED_BASE_NAMES: [&str; 11] = [
    "token_embd.weight",
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "output_norm.weight",
];

// ---------------------------------------------------------------------------
// View entry
// ---------------------------------------------------------------------------

/// One deterministic view entry for one admitted tensor, in GGUF tensor-table
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorViewEntry {
    /// Tensor name (`"token_embd.weight"`, `"blk.3.attn_v.weight"`, …).
    pub name: String,
    /// Admitted GGML type (the closed pinned-row set).
    pub ggml_type: GgmlType,
    /// Logical dimensions in GGUF order (1 or 2 entries for the pinned row).
    pub dims: Vec<u64>,
    /// Total logical elements (`product(dims)`).
    pub element_count: u64,
    /// Validated absolute byte range of this tensor in the file (bound-
    /// checked at `TensorView::build`; identical to `layout.byte_range()`).
    pub byte_range: ByteRange,
    /// The sole admitted quantized storage contract (GI1-2) for this tensor.
    pub layout: QuantizedTensorLayout,
}

// ---------------------------------------------------------------------------
// Typed fail-closed diagnostics
// ---------------------------------------------------------------------------

/// A typed, machine-parseable view rejection.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorViewError {
    /// A tensor's layout failed to resolve (GI1-2 re-validation).
    Layout(QuantizedLayoutError),
    /// The byte buffer handed to `build` is not the admitted file size.
    WrongFileSize { expected: u64, actual: u64 },
    /// The tensor-data region extends past the admitted file size.
    DataRegionOutOfBounds {
        data_offset: u64,
        data_end: u64,
        file_size: u64,
    },
    /// A tensor's byte range extends past the end of the admitted file.
    ByteRangeOutOfBounds {
        name: String,
        start: u64,
        end: u64,
        file_size: u64,
    },
    /// A tensor's byte range lies outside the tensor-data region.
    OutsideDataRegion {
        name: String,
        start: u64,
        end: u64,
        data_offset: u64,
        data_end: u64,
    },
    /// Checked offset/length/size arithmetic overflowed u64.
    ArithmeticOverflow { context: String },
    /// A raw-bytes request is outside the entry's declared range.
    AccessOutOfBounds {
        name: String,
        start: u64,
        end: u64,
        available: u64,
    },
    /// A raw-block request is outside the entry's declared block count.
    BlockIndexOutOfBounds {
        name: String,
        blocks: u64,
        block_index: u64,
    },
}

impl fmt::Display for TensorViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(err) => write!(f, "layout resolution failed: {err}"),
            Self::WrongFileSize { expected, actual } => {
                write!(f, "byte buffer is {actual} bytes, expected the admitted {expected}")
            }
            Self::DataRegionOutOfBounds {
                data_offset,
                data_end,
                file_size,
            } => write!(
                f,
                "tensor-data region [{data_offset}, {data_end}) extends past file size {file_size}"
            ),
            Self::ByteRangeOutOfBounds {
                name,
                start,
                end,
                file_size,
            } => write!(
                f,
                "tensor {name:?} byte range [{start}, {end}) extends past file size {file_size}"
            ),
            Self::OutsideDataRegion {
                name,
                start,
                end,
                data_offset,
                data_end,
            } => write!(
                f,
                "tensor {name:?} byte range [{start}, {end}) lies outside the tensor-data region [{data_offset}, {data_end})"
            ),
            Self::ArithmeticOverflow { context } => {
                write!(f, "arithmetic overflow while computing {context}")
            }
            Self::AccessOutOfBounds {
                name,
                start,
                end,
                available,
            } => write!(
                f,
                "byte access [{start}, {end}) for {name:?} is outside the declared range (only {available} bytes available)"
            ),
            Self::BlockIndexOutOfBounds {
                name,
                blocks,
                block_index,
            } => write!(
                f,
                "block {block_index} for {name:?} is outside the declared block count {blocks}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

/// A deterministic, host-readable (CPU) tensor view over the admitted GGUF
/// row.
///
/// A pure descriptor layer: it holds **no packed payload copy** — the file
/// slice is borrowed — and exposes every tensor's name, GGML type, logical
/// dimensions, element count, validated byte range, and
/// [`QuantizedTensorLayout`], plus bounded raw-bytes accessors.
///
/// `PartialEq` compares descriptor fields only (the borrowed file buffer is
/// not compared), so `assert_eq!(a, b)` proves byte-identical descriptors.
#[derive(Debug)]
pub struct TensorView<'a> {
    file: &'a [u8],
    schema: &'static str,
    file_size: u64,
    sha256: [u8; 32],
    data_offset: u64,
    data_len: u64,
    entries: Vec<TensorViewEntry>,
}

impl PartialEq for TensorView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.schema == other.schema
            && self.file_size == other.file_size
            && self.sha256 == other.sha256
            && self.data_offset == other.data_offset
            && self.data_len == other.data_len
            && self.entries == other.entries
    }
}

impl<'a> TensorView<'a> {
    /// Build the view over the admitted file bytes.
    ///
    /// `admission` must be the validated `GgufAdmission` for the same file
    /// `bytes` holds (length is checked here; content is re-verifiable with
    /// [`Self::sha256_matches`] — the whole-file digest was already verified
    /// once at GI1-1 admission).
    ///
    /// # Errors
    ///
    /// Returns the first typed [`TensorViewError`] the input contradicts:
    /// wrong buffer length, a layout that fails GI1-2 re-validation, a byte
    /// range outside the file or the tensor-data region, or an arithmetic
    /// overflow. Identical bytes always yield an identical view.
    pub fn build(admission: &GgufAdmission, bytes: &'a [u8]) -> Result<Self, TensorViewError> {
        // 1. The buffer must be the exact admitted file (cheap wrong-buffer
        //    check; content is re-verifiable via `sha256_matches`).
        if bytes.len() as u64 != admission.file_size {
            return Err(TensorViewError::WrongFileSize {
                expected: admission.file_size,
                actual: bytes.len() as u64,
            });
        }

        // 2. Every tensor resolves to its `QuantizedTensorLayout` (GI1-2):
        //    block geometry + absolute byte range re-validated fail-closed.
        let layouts = resolve_admission(admission).map_err(TensorViewError::Layout)?;

        // 3. Deterministic entries in GGUF tensor-table order.
        let mut entries = Vec::with_capacity(admission.tensors.len());
        for (desc, layout) in admission.tensors.iter().zip(layouts) {
            let byte_range = layout.byte_range();
            entries.push(TensorViewEntry {
                name: desc.name.clone(),
                ggml_type: desc.ggml_type,
                dims: desc.dims.clone(),
                element_count: desc.element_count,
                byte_range,
                layout,
            });
        }

        // 4. Fail-closed view invariants — re-validated even though GI1-1
        //    admission already proved them, so a view built from a hand-rolled
        //    admission fails the same way.
        let data_end = admission
            .data_offset
            .checked_add(admission.data_len)
            .ok_or_else(|| TensorViewError::ArithmeticOverflow {
                context: "tensor-data region end".into(),
            })?;
        if data_end > admission.file_size {
            return Err(TensorViewError::DataRegionOutOfBounds {
                data_offset: admission.data_offset,
                data_end,
                file_size: admission.file_size,
            });
        }
        for entry in &entries {
            let r = entry.byte_range;
            if r.end > admission.file_size {
                return Err(TensorViewError::ByteRangeOutOfBounds {
                    name: entry.name.clone(),
                    start: r.start,
                    end: r.end,
                    file_size: admission.file_size,
                });
            }
            if r.start < admission.data_offset || r.end > data_end {
                return Err(TensorViewError::OutsideDataRegion {
                    name: entry.name.clone(),
                    start: r.start,
                    end: r.end,
                    data_offset: admission.data_offset,
                    data_end,
                });
            }
        }

        Ok(Self {
            file: bytes,
            schema: admission.schema,
            file_size: admission.file_size,
            // 5. The whole-file digest verified at GI1-1 (== the pinned
            //    digest by construction of the admission) covers every tensor
            //    byte range just re-validated inside the file.
            sha256: PINNED_SHA256,
            data_offset: admission.data_offset,
            data_len: admission.data_len,
            entries,
        })
    }

    // -- Identity / envelope ------------------------------------------------

    /// Frozen intake schema identity (`gi0-model-contract v1.0.0`).
    #[must_use]
    pub const fn schema(&self) -> &'static str {
        self.schema
    }

    /// Verified whole-file SHA-256 of the admitted file (hex).
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex(&self.sha256)
    }

    /// Verified whole-file SHA-256 of the admitted file (bytes).
    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// Exact admitted file size (bytes).
    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Absolute offset of the tensor-data region (32-aligned).
    #[must_use]
    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Total tensor-data bytes (== sum of every tensor's `byte_range.len()`).
    #[must_use]
    pub const fn data_len(&self) -> u64 {
        self.data_len
    }

    /// The tensor-data region as a validated byte range.
    #[must_use]
    pub fn data_region(&self) -> ByteRange {
        ByteRange::new(self.data_offset, self.data_offset + self.data_len)
    }

    // -- Enumeration (deterministic, GGUF tensor-table order) ---------------

    /// Number of view entries (290 for the pinned row).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the view has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries in GGUF tensor-table order.
    #[must_use]
    pub fn entries(&self) -> &[TensorViewEntry] {
        &self.entries
    }

    /// Bound-checked enumeration: entry at `index`, or `None` when out of
    /// bounds.
    #[must_use]
    pub fn entry(&self, index: usize) -> Option<&TensorViewEntry> {
        self.entries.get(index)
    }

    // -- Named lookup -------------------------------------------------------

    /// Exact lookup by full tensor name (`"blk.3.attn_v.weight"`).
    #[must_use]
    pub fn tensor(&self, full_name: &str) -> Option<&TensorViewEntry> {
        self.entries.iter().find(|e| e.name == full_name)
    }

    /// Pinned base-name family lookup: `"attn_v.weight"` resolves every
    /// `"blk.N.attn_v.weight"` (plus the bare name when it exists). Entries
    /// come back in GGUF tensor-table order. `None` when no entry matches.
    #[must_use]
    pub fn family(&self, base_name: &str) -> Option<Vec<&TensorViewEntry>> {
        let suffix = format!(".{base_name}");
        let hits: Vec<&TensorViewEntry> = self
            .entries
            .iter()
            .filter(|e| e.name == base_name || e.name.ends_with(&suffix))
            .collect();
        if hits.is_empty() {
            None
        } else {
            Some(hits)
        }
    }

    // -- Hash accounting / coverage -----------------------------------------

    /// Whether `file`'s whole-file SHA-256 matches the digest recorded at
    /// GI1-1 admission — the caller's bytes really are the admitted file, so
    /// the recorded hash covers every tensor byte range in the view.
    #[must_use]
    pub fn sha256_matches(&self, file: &[u8]) -> bool {
        sha256(file) == self.sha256
    }

    /// Per-tensor coverage: whether `entry`'s declared byte range lies inside
    /// the admitted file, i.e. is covered by the whole-file SHA-256 recorded
    /// at GI1-1. Rejects forged entries (ranges outside the file).
    #[must_use]
    pub fn per_tensor_covered(&self, entry: &TensorViewEntry) -> bool {
        let r = entry.byte_range;
        r.start <= r.end && r.end <= self.file_size
    }

    /// Aggregate coverage: every entry is per-tensor covered, the ranges are
    /// non-overlapping and gap-free in GGUF order, and they tile the
    /// tensor-data region exactly — every packed byte of the data region is
    /// accounted for by exactly one tensor, all of them under the one
    /// whole-file hash.
    #[must_use]
    pub fn coverage_ok(&self) -> bool {
        let Some(data_end) = self.data_offset.checked_add(self.data_len) else {
            return false;
        };
        let mut prev_end: Option<u64> = None;
        for entry in &self.entries {
            if !self.per_tensor_covered(entry) {
                return false;
            }
            let r = entry.byte_range;
            if let Some(prev) = prev_end {
                if r.start != prev {
                    return false;
                }
            }
            prev_end = Some(r.end);
        }
        match (self.entries.first(), self.entries.last()) {
            (Some(first), Some(last)) => {
                first.byte_range.start == self.data_offset && last.byte_range.end == data_end
            }
            _ => false,
        }
    }

    // -- Bounded raw-bytes access (fail closed) -----------------------------

    /// The entry's full packed bytes, bounded by its declared byte range.
    ///
    /// Fails closed with [`TensorViewError::AccessOutOfBounds`] when the
    /// declared range is outside the admitted file (e.g. a forged entry).
    ///
    /// # Errors
    ///
    /// Returns `AccessOutOfBounds` for an out-of-range declaration.
    pub fn raw_bytes(&self, entry: &TensorViewEntry) -> Result<&'a [u8], TensorViewError> {
        let r = entry.byte_range;
        if r.start > r.end || r.end > self.file.len() as u64 {
            return Err(TensorViewError::AccessOutOfBounds {
                name: entry.name.clone(),
                start: r.start,
                end: r.end,
                available: self.file.len() as u64,
            });
        }
        Ok(&self.file[r.start as usize..r.end as usize])
    }

    /// One packed GGML block of `entry` (block `block_index` in
    /// `[0, layout.blocks())`), bounded by the entry's declared range.
    ///
    /// Fails closed with [`TensorViewError::BlockIndexOutOfBounds`] or
    /// [`TensorViewError::AccessOutOfBounds`] for out-of-range requests.
    ///
    /// # Errors
    ///
    /// Returns the first typed error the request contradicts.
    pub fn raw_block(
        &self,
        entry: &TensorViewEntry,
        block_index: u64,
    ) -> Result<&'a [u8], TensorViewError> {
        let blocks = entry.layout.blocks();
        if block_index >= blocks {
            return Err(TensorViewError::BlockIndexOutOfBounds {
                name: entry.name.clone(),
                blocks,
                block_index,
            });
        }
        let block_bytes = entry.layout.block_bytes();
        let r = entry.byte_range;
        let block_offset = block_index.checked_mul(block_bytes).ok_or_else(|| {
            TensorViewError::ArithmeticOverflow {
                context: format!("{} block offset", entry.name),
            }
        })?;
        let block_start = r.start.checked_add(block_offset).ok_or_else(|| {
            TensorViewError::ArithmeticOverflow {
                context: format!("{} block start", entry.name),
            }
        })?;
        let block_end = block_start.checked_add(block_bytes).ok_or_else(|| {
            TensorViewError::ArithmeticOverflow {
                context: format!("{} block end", entry.name),
            }
        })?;
        if block_start > r.end || block_end > r.end || r.end > self.file.len() as u64 {
            return Err(TensorViewError::AccessOutOfBounds {
                name: entry.name.clone(),
                start: block_start,
                end: block_end,
                available: self.file.len() as u64,
            });
        }
        Ok(&self.file[block_start as usize..block_end as usize])
    }
}
