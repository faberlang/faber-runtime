//! GI2-1 — CPU dequant core: GGML block dequantization for the admitted row.
//!
//! Faber-owned CPU dequantization of the four admitted GGML block types of
//! the pinned SmolLM2-360M-Instruct Q4_K_M row — Q4_K 256/144, Q5_0 32/22,
//! Q6_K 256/210, Q8_0 32/34 — plus F32 1/4, to f32 logical elements
//! (model contract §3; `QuantizedTensorLayout`; GI1-2). Dequant is
//! **GI2-owned** (GI1 §Open questions Q1 default; CTO memo `fdc2a448`): the
//! tensor view exposes layout facts + bounded raw block bytes only; this
//! module widens them.
//!
//! CONTRACT — semantics match llama.cpp exactly (`ggml/src/ggml-quants.c`,
//! commit `a957b7747` at the pinned checkout): GGML half→f32 (exact IEEE-754
//! binary16→binary32, `ggml_compute_fp16_to_fp32`), integer qs, per-block
//! scale/min/dmin/kscale application (`dequantize_row_q5_0`,
//! `dequantize_row_q8_0`, `dequantize_row_q4_K`, `dequantize_row_q6_K` +
//! `get_scale_min_k4`). Dequant is exact integer/half math: all arithmetic is
//! plain IEEE-754 f32 in the same operation order as the C kernels, so
//! outputs are **bit-exact** against the independent Python reference
//! (`radix/docs/factory/gpu-inference-gguf/evidence/gi2-dequant-reference.py`
//! and its committed goldens `gi2-dequant-goldens.json`).
//!
//! FAIL CLOSED:
//! - the tensor entry point (`dequant_tensor`) **gates on `coverage_ok()` +
//!   `per_tensor_covered`** (GI1-4 residual folded in): a gapped or forged
//!   view, or an un-covered entry, is rejected with a typed
//!   [`DequantError`] before any byte is touched;
//! - byte buffers must match the layout's packed size exactly (typed
//!   `RowBytesMismatch` / `BlockBytesMismatch`);
//! - a declared repack layout is never executed (decision (f): direct native
//!   block execution is the contract; repack is NOT the admitted layout);
//! - out-of-range block/byte access is rejected by the view with its typed
//!   diagnostics, propagated as [`DequantError::View`].
//!
//! BOUNDARY — never `U8`-as-quantization: this module consumes
//! [`QuantizedTensorLayout`] + `&[u8]` and produces `Vec<f32>`. The toy
//! packed-u4 carrier (`crate::packed_numeric::PackedU4Layout::toy_u4`) is not
//! a GGML block and is not expressible here (the layout type can only resolve
//! the closed set {F32, Q5_0, Q8_0, Q4_K, Q6_K} — GI1-2 exclusion test stays
//! green).
//!
//! ORACLE RECEIPT (GI2-1 CTO S1 amendment) — every dequant `Vec<f32>` is a
//! **CPU-oracle materialization** and is accompanied by an [`OracleReceipt`]:
//! the smallest-correct descriptor carrying source tensor identity +
//! encoding/byte range, destination contiguous-f32 layout + byte extent, the
//! transformation implementation/version (`ggml-quants.c @ a957b7747`),
//! the output digest for the deterministic fixtures, `purpose = CpuOracle`,
//! and — recorded at tensor-level fixture generation only — timing + peak
//! temporary bytes (setup evidence, **not** a decode metric). This conversion
//! **neither changes [`RepackIdentity::Native`]** (direct native block
//! execution stays the contract, decision (f)) **nor authorizes
//! converted-weight GPU/headline execution**: the f32 output exists to be
//! bit-compared against the independent reference and to feed downstream CPU
//! consumers of the logits oracle.

use crate::gguf::GgmlType;
use crate::quantized_tensor_layout::{
    ByteRange, QuantizedTensorLayout, RepackIdentity,
};
use crate::tensor_view::{TensorView, TensorViewEntry};
use std::fmt;

// ---------------------------------------------------------------------------
// Typed fail-closed diagnostics
// ---------------------------------------------------------------------------

/// A typed, machine-parseable dequant rejection.
#[derive(Debug, Clone, PartialEq)]
pub enum DequantError {
    /// The tensor view's aggregate coverage is not exact (a gapped or
    /// overlapping range set, or a range set that does not tile the data
    /// region). GI1-4 residual: dequant never consumes an un-covered view.
    CoverageNotOk,
    /// The entry's declared byte range is not covered by the admitted file.
    EntryNotCovered { name: String },
    /// The layout declares a repack; only native GGML blocks are executed
    /// (decision (f) — repack is not the admitted layout).
    RepackNotNative,
    /// The packed row buffer is not exactly `blocks × block_bytes` long.
    RowBytesMismatch { expected: u64, actual: u64 },
    /// A block buffer is not exactly the layout's `block_bytes` long.
    BlockBytesMismatch { expected: u64, actual: u64 },
    /// The tensor view rejected the byte access (fail-closed diagnostics).
    View(crate::tensor_view::TensorViewError),
}

impl fmt::Display for DequantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoverageNotOk => write!(
                f,
                "tensor-view coverage is not exact (gapped/overlapping ranges); dequant refuses the view"
            ),
            Self::EntryNotCovered { name } => {
                write!(f, "tensor {name:?} byte range is not covered by the admitted file")
            }
            Self::RepackNotNative => {
                write!(f, "layout declares a repack; only native GGML blocks are executed")
            }
            Self::RowBytesMismatch { expected, actual } => write!(
                f,
                "packed row is {actual} bytes, expected exactly {expected} (blocks × block_bytes)"
            ),
            Self::BlockBytesMismatch { expected, actual } => write!(
                f,
                "block buffer is {actual} bytes, expected the layout's block width {expected}"
            ),
            Self::View(err) => write!(f, "tensor-view access failed closed: {err}"),
        }
    }
}

impl std::error::Error for DequantError {}

// ---------------------------------------------------------------------------
// GGML half -> f32 (exact IEEE-754 binary16 -> binary32)
// ---------------------------------------------------------------------------

/// Bit-exact IEEE-754 binary16 → binary32 conversion
/// (`ggml_compute_fp16_to_fp32`, `ggml-impl.h`). Every half value is exactly
/// representable in f32, so the mapping is a pure bit expansion (sign, exp,
/// mantissa); subnormals widen to their exact value (`frac × 2⁻²⁴`),
/// infinities/NaNs map with the standard bit pattern.
#[must_use]
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) & 0x1;
    let exp = u32::from(bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x3ff);
    let bits32 = if exp == 0 {
        if frac == 0 {
            sign << 31 // ±0
        } else {
            // Subnormal: value = frac × 2^-24, frac ∈ [1, 1023]. Normalize
            // the leading bit: with p = bit_length(frac), the value is
            // (frac × 2^(1-p)) × 2^(p-25), so the f32 exponent field is
            // p - 25 + 127 = p + 102 and the mantissa field is
            // (frac × 2^(1-p) - 1) × 2^23 = frac × 2^(24-p) - 2^23.
            let p = 32 - frac.leading_zeros();
            (sign << 31) | ((p + 102) << 23) | ((frac << (24 - p)) - (1 << 23))
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13) // ±inf / NaN
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13) // 127 - 15 = 112
    };
    f32::from_bits(bits32)
}

// ---------------------------------------------------------------------------
// Per-block dequant kernels (mirror llama.cpp dequantize_row_* exactly)
// ---------------------------------------------------------------------------

/// `block_q8_0`: `d` (half) + `qs[32]` int8 → `y[j] = qs[j] * d`.
fn dequant_q8_0(block: &[u8]) -> Vec<f32> {
    let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
    (0..32)
        .map(|j| f32::from(block[2 + j] as i8) * d)
        .collect()
}

/// `block_q5_0`: `d` (half) + `qh[4]` bitmask + `qs[16]` nibbles.
///
/// Output order matches llama.cpp: `y[0..16]` from the low nibbles,
/// `y[16..32]` from the high nibbles.
fn dequant_q5_0(block: &[u8]) -> Vec<f32> {
    let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    let mut y = vec![0.0f32; 32];
    for j in 0..16usize {
        let xh_0 = ((qh >> (j as u32 + 0)) << 4) & 0x10;
        let xh_1 = (qh >> (j as u32 + 12)) & 0x10;
        let x0 = i32::from((qs[j] & 0x0f) | (xh_0 as u8)) - 16;
        let x1 = i32::from((qs[j] >> 4) | (xh_1 as u8)) - 16;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
    y
}

/// `get_scale_min_k4` (llama.cpp `ggml-quants.c`): 6-bit scale + 6-bit min
/// from `scales[K_SCALE_SIZE]`.
fn get_scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

/// `block_q4_K`: `d` (half), `dmin` (half), `scales[12]`, `qs[128]`.
///
/// Per 64-element sub-block pair (`is`, `is+1`): `d1 = d·sc(is)`,
/// `m1 = dmin·m(is)`, `d2 = d·sc(is+1)`, `m2 = dmin·m(is+1)`; then
/// `y = d1·(qs&0xF) − m1` for 32 elements, `y = d2·(qs>>4) − m2` for 32.
fn dequant_q4_k(block: &[u8]) -> Vec<f32> {
    let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut y = vec![0.0f32; 256];
    let mut is = 0usize;
    let mut qoff = 0usize;
    for chunk in [0usize, 64, 128, 192] {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = min * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = min * f32::from(m);
        for l in 0..32usize {
            y[chunk + l] = d1 * f32::from(qs[qoff + l] & 0x0f) - m1;
            y[chunk + 32 + l] = d2 * f32::from(qs[qoff + l] >> 4) - m2;
        }
        qoff += 32;
        is += 2;
    }
    y
}

/// `block_q6_K`: `ql[128]` low nibbles, `qh[64]` high bits, `scales[16]`
/// int8, `d` (half) at byte offset 208.
///
/// Output order matches llama.cpp: within each 128-element group the four
/// quants are written at strides 0/32/64/96 over `l`.
fn dequant_q6_k(block: &[u8]) -> Vec<f32> {
    let d = half_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut y = vec![0.0f32; 256];
    for n in [0usize, 128] {
        let sc_base = n / 16; // the C `sc += 8` per 128-group
        for l in 0..32usize {
            let is = l / 16;
            let q1 = i32::from((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            let sc0 = f32::from(sc[sc_base + is + 0] as i8);
            let sc2 = f32::from(sc[sc_base + is + 2] as i8);
            let sc4 = f32::from(sc[sc_base + is + 4] as i8);
            let sc6 = f32::from(sc[sc_base + is + 6] as i8);
            // Same left-associative f32 multiplication order as the C kernels
            // ((d * sc) * q), so outputs are bit-exact vs the reference.
            y[n + l] = d * sc0 * q1 as f32;
            y[n + l + 32] = d * sc2 * q2 as f32;
            y[n + l + 64] = d * sc4 * q3 as f32;
            y[n + l + 96] = d * sc6 * q4 as f32;
        }
    }
    y
}

/// F32 scalar block: bit-exact passthrough of the 4 little-endian bytes.
fn dequant_f32(block: &[u8]) -> Vec<f32> {
    vec![f32::from_le_bytes([block[0], block[1], block[2], block[3]])]
}

// ---------------------------------------------------------------------------
// Public surface: block / row / tensor dequant
// ---------------------------------------------------------------------------

/// Dequantize one packed GGML block to its f32 logical elements.
///
/// `block` must be exactly `layout.block_bytes()` bytes (fail closed with
/// [`DequantError::BlockBytesMismatch`]); a declared-repack layout is
/// rejected ([`DequantError::RepackNotNative`]).
///
/// ORACLE CONTRACT — the returned `Vec<f32>` is a **CPU-oracle
/// materialization**: it must be accompanied by an [`OracleReceipt`]
/// (tensor-level: [`OracleReceipt::for_tensor`]; block/row outputs are
/// intermediate steps of that tensor-level materialization). This conversion
/// neither changes [`RepackIdentity::Native`] nor authorizes converted-weight
/// GPU/headline execution.
///
/// # Errors
///
/// Returns the first typed [`DequantError`] the input contradicts.
pub fn dequant_block(
    layout: &QuantizedTensorLayout,
    block: &[u8],
) -> Result<Vec<f32>, DequantError> {
    if layout.repack_identity() != RepackIdentity::Native {
        return Err(DequantError::RepackNotNative);
    }
    let expected = layout.block_bytes() as usize;
    if block.len() != expected {
        return Err(DequantError::BlockBytesMismatch {
            expected: expected as u64,
            actual: block.len() as u64,
        });
    }
    let out = match layout.format_id() {
        GgmlType::F32 => dequant_f32(block),
        GgmlType::Q4_K => dequant_q4_k(block),
        GgmlType::Q5_0 => dequant_q5_0(block),
        GgmlType::Q6_K => dequant_q6_k(block),
        GgmlType::Q8_0 => dequant_q8_0(block),
    };
    debug_assert_eq!(out.len() as u64, layout.block_elements());
    Ok(out)
}

/// Dequantize a packed row (all of `layout.blocks()` blocks concatenated) to
/// its f32 logical elements, in GGUF block order.
///
/// `packed` must be exactly `blocks × block_bytes` bytes (fail closed with
/// [`DequantError::RowBytesMismatch`]).
///
/// ORACLE CONTRACT — the returned `Vec<f32>` is a **CPU-oracle
/// materialization**: it must be accompanied by an [`OracleReceipt`]
/// (tensor-level: [`OracleReceipt::for_tensor`]; block/row outputs are
/// intermediate steps of that tensor-level materialization). This conversion
/// neither changes [`RepackIdentity::Native`] nor authorizes converted-weight
/// GPU/headline execution.
///
/// # Errors
///
/// Returns the first typed [`DequantError`] the input contradicts.
pub fn dequant_row(
    layout: &QuantizedTensorLayout,
    packed: &[u8],
) -> Result<Vec<f32>, DequantError> {
    let expected = layout.blocks() * layout.block_bytes();
    if packed.len() as u64 != expected {
        return Err(DequantError::RowBytesMismatch {
            expected,
            actual: packed.len() as u64,
        });
    }
    let block_bytes = layout.block_bytes() as usize;
    let mut out = Vec::with_capacity(layout.element_count() as usize);
    for chunk in packed.chunks_exact(block_bytes) {
        out.extend(dequant_block(layout, chunk)?);
    }
    Ok(out)
}

/// Dequantize a whole tensor through the bounded tensor view.
///
/// GI1-4 residual folded in: construction **gates on `coverage_ok()`** and on
/// `per_tensor_covered(entry)` — a gapped/forged view or an un-covered entry
/// fails closed with [`DequantError::CoverageNotOk`] /
/// [`DequantError::EntryNotCovered`] before any byte is touched. The entry's
/// packed bytes are read through the view's bounded accessors (their typed
/// errors propagate as [`DequantError::View`]).
///
/// ORACLE CONTRACT — the returned `Vec<f32>` is a **CPU-oracle
/// materialization** and must be accompanied by an [`OracleReceipt`]
/// ([`OracleReceipt::for_tensor`], which fails closed on the same gates).
/// This conversion neither changes [`RepackIdentity::Native`] nor authorizes
/// converted-weight GPU/headline execution.
///
/// # Errors
///
/// Returns the first typed [`DequantError`] the input contradicts.
pub fn dequant_tensor(
    view: &TensorView<'_>,
    entry: &TensorViewEntry,
) -> Result<Vec<f32>, DequantError> {
    if !view.coverage_ok() {
        return Err(DequantError::CoverageNotOk);
    }
    if !view.per_tensor_covered(entry) {
        return Err(DequantError::EntryNotCovered {
            name: entry.name.clone(),
        });
    }
    let bytes = view.raw_bytes(entry).map_err(DequantError::View)?;
    dequant_row(&entry.layout, bytes)
}

// ---------------------------------------------------------------------------
// Oracle-materialization receipt (GI2-1 CTO S1 amendment — correct before
// next phase: GI2-1 stays closed, GI3 admission depends on this)
// ---------------------------------------------------------------------------

/// Oracle purpose of a dequant materialization.
///
/// The dequant `Vec<f32>` is a **CPU-oracle** materialization: it exists to be
/// bit-compared against the independent reference (the goldens) and to feed
/// downstream CPU consumers of the logits oracle. It **never authorizes
/// converted-weight execution** — this conversion neither changes
/// [`RepackIdentity::Native`] (the layout stays native; direct native block
/// execution remains the contract, decision (f)) nor authorizes running the
/// converted f32 weights on GPU or in the headline path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OraclePurpose {
    /// CPU-only oracle materialization (bit-exact verification surface).
    CpuOracle,
}

/// The transformation implementation + pinned version the dequant output is
/// produced with: llama.cpp `ggml/src/ggml-quants.c` at the pinned checkout
/// (commit `a957b7747`) — the same reference the golden fixtures derive from.
pub const ORACLE_TRANSFORM_IMPL: &str = "ggml/src/ggml-quants.c @ a957b7747";

/// The oracle-materialization descriptor/receipt that must accompany every
/// dequant `Vec<f32>` (from [`dequant_block`], [`dequant_row`],
/// [`dequant_tensor`]).
///
/// Smallest-correct form (GI2-1 CTO S1 amendment): the receipt is a pure
/// descriptor of one CPU-oracle materialization of one admitted tensor — it
/// carries no packed bytes and no decoded values. `output_digest`,
/// `timing_us` and `peak_temp_bytes` are **deterministic-fixture setup
/// evidence**: they are recorded at tensor-level fixture generation (see
/// `gi2-dequant-reference.py` + the committed goldens) and are `None` on a
/// live materialization — they are never decode metrics.
///
/// The receipt exists to make the oracle boundary explicit: this conversion
/// **neither changes [`RepackIdentity::Native`]** nor **authorizes
/// converted-weight GPU/headline execution**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleReceipt {
    /// Source tensor identity (the GGUF tensor name in the admitted file).
    pub source_tensor: String,
    /// Source GGML encoding (the closed pinned-row set).
    pub source_encoding: GgmlType,
    /// Source packed byte range within the admitted file (absolute).
    pub source_byte_range: ByteRange,
    /// Destination layout: contiguous f32, no padding
    /// (`entry.layout.logical_dtype()`).
    pub dest_element_count: u64,
    /// Destination byte extent (`element_count × 4` for the f32 logical
    /// dtype).
    pub dest_byte_extent: u64,
    /// Transformation implementation + pinned version
    /// ([`ORACLE_TRANSFORM_IMPL`]).
    pub transform_impl: &'static str,
    /// Purpose: CPU-oracle materialization only.
    pub purpose: OraclePurpose,
    /// SHA-256 of the f32 LE byte stream of the materialized output —
    /// recorded for the deterministic fixtures (the goldens), `None` on a
    /// live materialization.
    pub output_digest: Option<[u8; 32]>,
    /// Tensor-level fixture-generation wall time (µs) — **setup evidence,
    /// not a decode metric**; `None` on a live materialization.
    pub timing_us: Option<u64>,
    /// Peak temporary bytes recorded at tensor-level fixture generation —
    /// **setup evidence, not a decode metric**; `None` on a live
    /// materialization.
    pub peak_temp_bytes: Option<u64>,
}

impl OracleReceipt {
    /// Build the receipt for a tensor-level materialization of `entry`
    /// through `view` — the descriptor that accompanies the `Vec<f32>` that
    /// [`dequant_tensor`] would produce for the same `(view, entry)`.
    ///
    /// Fails closed on the same descriptor gates as [`dequant_tensor`]
    /// without touching bytes: gapped/forged view →
    /// [`DequantError::CoverageNotOk`]; un-covered entry →
    /// [`DequantError::EntryNotCovered`]; declared repack →
    /// [`DequantError::RepackNotNative`]; row byte-length mismatch →
    /// [`DequantError::RowBytesMismatch`].
    ///
    /// # Errors
    ///
    /// Returns the first typed [`DequantError`] the input contradicts.
    pub fn for_tensor(
        view: &TensorView<'_>,
        entry: &TensorViewEntry,
    ) -> Result<Self, DequantError> {
        if !view.coverage_ok() {
            return Err(DequantError::CoverageNotOk);
        }
        if !view.per_tensor_covered(entry) {
            return Err(DequantError::EntryNotCovered {
                name: entry.name.clone(),
            });
        }
        if entry.layout.repack_identity() != RepackIdentity::Native {
            return Err(DequantError::RepackNotNative);
        }
        let expected = entry.layout.blocks() * entry.layout.block_bytes();
        if entry.byte_range.len() != expected {
            return Err(DequantError::RowBytesMismatch {
                expected,
                actual: entry.byte_range.len(),
            });
        }
        Ok(Self {
            source_tensor: entry.name.clone(),
            source_encoding: entry.ggml_type,
            source_byte_range: entry.byte_range,
            dest_element_count: entry.element_count,
            dest_byte_extent: entry.element_count * entry.layout.logical_dtype().bytes(),
            transform_impl: ORACLE_TRANSFORM_IMPL,
            purpose: OraclePurpose::CpuOracle,
            output_digest: None,
            timing_us: None,
            peak_temp_bytes: None,
        })
    }

    /// Record deterministic-fixture evidence on a copy of this receipt: the
    /// output digest (SHA-256 of the f32 LE byte stream) plus the tensor-level
    /// fixture-generation wall time (µs) and peak temporary bytes. Setup
    /// evidence only — never a decode metric.
    #[must_use]
    pub fn with_fixture_evidence(
        &self,
        output_digest: [u8; 32],
        timing_us: u64,
        peak_temp_bytes: u64,
    ) -> Self {
        Self {
            output_digest: Some(output_digest),
            timing_us: Some(timing_us),
            peak_temp_bytes: Some(peak_temp_bytes),
            ..self.clone()
        }
    }
}
