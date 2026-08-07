//! GI4-1 — typed `KvCacheLayout` tests.
//!
//! Families:
//! 1. **Layout construction**: the pinned-row layout admits; any zero
//!    dimension is rejected (`None`) — a zero-byte KV is not a layout.
//! 2. **Byte accounting (the single authority)**: `total_bytes` derives the
//!    deterministic "KV bytes per `KvCacheLayout`" figure (slots × context ×
//!    layers × kv-heads × head-dim × K+V × dtype), and the partition ledger
//!    **consumes, never re-derives** it (`partition.rs` contract).
//! 3. **Dtype + reserve policy**: dtype bytes/spelling; the reserve is a
//!    separate declared bound, never folded silently into storage.
//! 4. **Determinism**: equal layouts produce equal bytes; the layout is a
//!    `Copy`/`Eq`/`Hash` value type.

use crate::cpu_oracle::LAYER_COUNT;
use crate::decoder_ops::{HEAD_DIM, KV_HEAD_COUNT};
use crate::kv_cache::{KvCacheDtype, KvCacheLayout, KvReservePolicy};
use crate::partition::PartitionBudgetLedger;

/// The pinned row's context length (ctx 8192 — the `llama.context_length`
/// fact, `gguf.rs`).
const CONTEXT_LENGTH: u32 = 8192;

/// The pinned-row layout with the given slot count.
fn pinned_layout(slots: u32) -> KvCacheLayout {
    KvCacheLayout::new(
        slots,
        CONTEXT_LENGTH,
        LAYER_COUNT as u32,
        KV_HEAD_COUNT as u32,
        HEAD_DIM as u32,
        KvCacheDtype::F32,
        KvReservePolicy::Fixed { bytes: 0 },
    )
    .expect("pinned-row layout is valid")
}

#[test]
fn pinned_row_layout_admits() {
    let layout = pinned_layout(1);
    assert_eq!(layout.slots(), 1);
    assert_eq!(layout.context_length(), CONTEXT_LENGTH);
    assert_eq!(layout.layer_count(), LAYER_COUNT as u32);
    assert_eq!(layout.kv_head_count(), KV_HEAD_COUNT as u32);
    assert_eq!(layout.head_dim(), HEAD_DIM as u32);
    assert_eq!(layout.dtype(), KvCacheDtype::F32);
}

#[test]
fn zero_dimensions_are_rejected() {
    let ok = |slots, ctx, layers, heads, dim| {
        KvCacheLayout::new(
            slots,
            ctx,
            layers,
            heads,
            dim,
            KvCacheDtype::F32,
            KvReservePolicy::Fixed { bytes: 0 },
        )
    };
    assert!(ok(0, 8192, 32, 5, 64).is_none());
    assert!(ok(1, 0, 32, 5, 64).is_none());
    assert!(ok(1, 8192, 0, 5, 64).is_none());
    assert!(ok(1, 8192, 32, 0, 64).is_none());
    assert!(ok(1, 8192, 32, 5, 0).is_none());
}

#[test]
fn pinned_row_storage_bytes() {
    // slots 1 × ctx 8192 × layers 32 × kv-heads 5 × head-dim 64 × (K+V = 2)
    // × f32 (4 bytes) = 671_088_640 bytes.
    let layout = pinned_layout(1);
    assert_eq!(
        layout.elements_per_slot(),
        u64::from(CONTEXT_LENGTH) * 32 * 5 * 64 * 2
    );
    assert_eq!(layout.storage_bytes_per_slot(), 671_088_640 / 1);
    assert_eq!(layout.total_bytes(), Some(671_088_640));
}

#[test]
fn slot_scaling_is_linear() {
    assert_eq!(pinned_layout(2).total_bytes(), Some(671_088_640 * 2));
    assert_eq!(pinned_layout(4).total_bytes(), Some(671_088_640 * 4));
}

#[test]
fn dtype_facts() {
    assert_eq!(KvCacheDtype::F32.byte_size(), 4);
    assert_eq!(KvCacheDtype::F32.spelling(), "f32");
}

#[test]
fn reserve_policy_is_a_separate_declared_bound() {
    let layout = KvCacheLayout::new(
        1,
        CONTEXT_LENGTH,
        32,
        5,
        64,
        KvCacheDtype::F32,
        KvReservePolicy::Fixed { bytes: 4096 },
    )
    .expect("layout valid");
    assert_eq!(layout.reserve_policy().bytes(), 4096);
    // admitted = deterministic storage + declared reserve; storage itself is
    // untouched by the reserve.
    assert_eq!(layout.total_bytes(), Some(671_088_640));
    assert_eq!(layout.admitted_bytes(), Some(671_088_640 + 4096));
}

#[test]
fn ledger_consumes_the_layout_figure_never_rederives() {
    // The partition contract (`partition.rs`): KV bytes per `KvCacheLayout`
    // are consumed, not re-derived — the layout's own figure is the explicit
    // class-2 bound supplied at admission.
    let layout = pinned_layout(1);
    let kv_bytes = layout.total_bytes().expect("layout bytes finite");
    let ledger = PartitionBudgetLedger {
        weight_bytes: 0,
        kv_cache_bytes: kv_bytes,
        activation_scratch_bytes: 0,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: 0,
        concurrent_state_bytes: 0,
    };
    // The ledger's admitted total is exactly the consumed figure; it never
    // recomputes a layout from dims.
    assert_eq!(ledger.total_bytes(), Some(kv_bytes));
}

#[test]
fn deterministic_bytes() {
    let a = pinned_layout(1);
    let b = pinned_layout(1);
    assert_eq!(a, b);
    assert_eq!(a.total_bytes(), b.total_bytes());
}
