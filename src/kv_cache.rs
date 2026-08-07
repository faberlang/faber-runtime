//! GI4-1 — typed `KvCacheLayout` (CTO S5; lane queue §2).
//!
//! The typed KV layout frozen by the GI4-1 session contract
//! (`radix/docs/factory/gpu-inference-gguf/gi4-delivery.md` §GI4-1; contract
//! record `gi4-contract.md`): slots, context length, layers/heads, dtype, and
//! reserve policy. Its byte accounting is the **single authority** for "KV
//! bytes per `KvCacheLayout`" — **consumed, not re-derived**
//! ([`crate::partition`]: the partition ledger never recomputes a layout's
//! bytes; the layout's own figure is the explicit bound supplied at
//! admission). This is the typed KV identity the anonymous `PerProgram` f32
//! buffers lack (FC2/FC3).

/// Version of the typed KV layout. Bumped only by a contract revision.
pub const KV_CACHE_LAYOUT_VERSION: u32 = 1;

/// The KV cache element dtype. GI4 uses the declared-f32 representation
/// (GI3-2 repack plan; the campaign never presents converted weights as
/// direct GGUF quantized execution); the type is closed so a dtype change is
/// a contract revision, never a silent widening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheDtype {
    /// IEEE-754 binary32 (4 bytes per element) — the declared conversion
    /// representation for the GI4 row.
    F32,
}

impl KvCacheDtype {
    /// Bytes per element of this dtype.
    #[must_use]
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::F32 => 4,
        }
    }

    /// Stable diagnostic spelling (`"f32"`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }
}

/// The reserve policy of a KV layout (the fifth layout fact, lane queue §2).
///
/// The *storage* bytes are derived deterministically from the layout dims
/// ([`KvCacheLayout::total_bytes`]); the *reserve* is the policy-declared
/// extra bound (e.g. headroom declared at admission) — a separate fact, never
/// folded silently into the storage figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvReservePolicy {
    /// A fixed reserved byte count declared at admission (the partition
    /// ledger's class-2 bound when a reserve is declared).
    Fixed {
        /// The reserved bytes, on top of the deterministic storage figure.
        bytes: u64,
    },
}

impl KvReservePolicy {
    /// The declared reserve bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Fixed { bytes } => bytes,
        }
    }
}

/// The typed KV cache layout (slots, context, layers/heads, dtype, reserve).
///
/// GI4 begins with **one sequence, one slot** ([`KvCacheLayout::new`] with
/// `slots = 1`); the layout is typed so MD3I / later stages can admit more
/// slots without re-deriving a new layout vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCacheLayout {
    slots: u32,
    context_length: u32,
    layer_count: u32,
    kv_head_count: u32,
    head_dim: u32,
    dtype: KvCacheDtype,
    reserve_policy: KvReservePolicy,
}

impl KvCacheLayout {
    /// Build a typed KV layout. `None` when any dimension is zero (a zero
    /// slot/context/layer/head layout is not a layout — it would silently
    /// produce a zero-byte KV).
    #[must_use]
    pub fn new(
        slots: u32,
        context_length: u32,
        layer_count: u32,
        kv_head_count: u32,
        head_dim: u32,
        dtype: KvCacheDtype,
        reserve_policy: KvReservePolicy,
    ) -> Option<Self> {
        if slots == 0
            || context_length == 0
            || layer_count == 0
            || kv_head_count == 0
            || head_dim == 0
        {
            return None;
        }
        Some(Self {
            slots,
            context_length,
            layer_count,
            kv_head_count,
            head_dim,
            dtype,
            reserve_policy,
        })
    }

    /// The sequence-slot count (GI4: 1).
    #[must_use]
    pub const fn slots(self) -> u32 {
        self.slots
    }

    /// The maximum context length per slot (the pinned row: ctx 8192).
    #[must_use]
    pub const fn context_length(self) -> u32 {
        self.context_length
    }

    /// The transformer layer count (the pinned row: 32).
    #[must_use]
    pub const fn layer_count(self) -> u32 {
        self.layer_count
    }

    /// The KV head count per layer (the pinned row: 5, GQA 15/5).
    #[must_use]
    pub const fn kv_head_count(self) -> u32 {
        self.kv_head_count
    }

    /// The head dimension (the pinned row: 64).
    #[must_use]
    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    /// The element dtype.
    #[must_use]
    pub const fn dtype(self) -> KvCacheDtype {
        self.dtype
    }

    /// The declared reserve policy.
    #[must_use]
    pub const fn reserve_policy(self) -> KvReservePolicy {
        self.reserve_policy
    }

    /// Bytes per element (from the dtype).
    #[must_use]
    pub const fn dtype_byte_size(self) -> u32 {
        self.dtype.byte_size()
    }

    /// Storage elements per slot: `context × layers × kv_heads × head_dim`
    /// for K **plus** V (both caches are resident).
    #[must_use]
    pub const fn elements_per_slot(self) -> u64 {
        let per_layer = self.context_length as u64
            * self.kv_head_count as u64
            * self.head_dim as u64;
        // K + V: both resident caches per layer.
        per_layer * self.layer_count as u64 * 2
    }

    /// Storage bytes per slot (elements × dtype size).
    #[must_use]
    pub const fn storage_bytes_per_slot(self) -> u64 {
        self.elements_per_slot() * self.dtype_byte_size() as u64
    }

    /// The authoritative KV storage byte figure — the "KV bytes per
    /// `KvCacheLayout`" bound the partition ledger **consumes, never
    /// re-derives**. `None` on arithmetic overflow (a layout too large to
    /// account for fails closed, deterministically).
    #[must_use]
    pub fn total_bytes(self) -> Option<u64> {
        self.storage_bytes_per_slot().checked_mul(u64::from(self.slots))
    }

    /// The full admitted byte figure: storage + declared reserve. `None` on
    /// overflow.
    #[must_use]
    pub fn admitted_bytes(self) -> Option<u64> {
        self.total_bytes()?.checked_add(self.reserve_policy.bytes())
    }
}
