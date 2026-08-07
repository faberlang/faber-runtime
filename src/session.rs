//! GI4-1 — frozen inference session contract (CTO S5 surface).
//!
//! The **invocation/session contract freeze** of the GI4 serial gate
//! (`radix/docs/factory/gpu-inference-gguf/gi4-delivery.md` §GI4-1; contract
//! record `gi4-contract.md`). The GI4-1 discovery (`evidence/
//! gi4-session-discovery.md`) proved that the existing `SingleRun` session +
//! `PerProgram` resident buffers **cannot** express the required inference
//! state without semantic distortion — `SingleRun` refuses once-init params
//! (a weight upload would re-copy per execution), `RepeatingStep` once-inits
//! but is training-shaped (loss observation cadence, end-of-run readback),
//! and `PerProgram` buffers are anonymous f32 buffers with no per-invocation
//! input cadence / KV identity / sequence position / workload mode / reset.
//! The missing facts are exactly the CTO S5 surface (`6badaa01` S5): the
//! four identities/lifetimes, the typed `KvCacheLayout`, the workload modes,
//! the transactional token mutation, and the reuse/invalidation keys.
//!
//! This module freezes those facts as types: [`ModelInstance`],
//! [`ExecutionSession`], [`SequenceState`], [`Invocation`], plus
//! [`InputUpdateCadence`] and [`InvocationMode`] (the per-invocation input
//! cadence + workload-mode vocabulary the live tree lacked — FC3), the
//! transactional [`TokenCommit`] rule, and the [`ReuseKey`] reuse/invalidation
//! key. **No wire edit here** — the carriage on the wire is GI4-2's unit
//! (nothing in this module touches `radix-mir-fmir/src/schema/**`).

use crate::kv_cache::KvCacheLayout;
use std::fmt;

/// Version of this session contract freeze. Bumped only by a *contract*
/// revision; the GI4-2 wire carriage rides the accepted wire version and
/// never invents an unversioned fact.
pub const SESSION_CONTRACT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// ModelInstance — model bytes + config identity, load-once (identity 1 of 4)
// ---------------------------------------------------------------------------

/// The **`ModelInstance`** identity: model bytes + config identity, **load-once**.
///
/// A model is identified by its stable id and the SHA-256 of its model bytes
/// (for the pinned row: SmolLM2-360M-Instruct Q4_K_M, SHA-256 `2fa3f013…bac9c2`
/// — the full hash lives in the model contract, `gi0-model-contract.md`). The
/// byte length is carried so a load-once/never-re-copy decision is
/// *computable*: a per-execution re-copy of `bytes_len` is exactly the
/// `SingleRun` semantic the discovery proved infeasible (a 270 MB weight
/// upload would re-copy per execution). The model bytes themselves live in
/// the owning host session's registry; this identity never carries them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInstance {
    model_id: String,
    bytes_sha256: [u8; 32],
    bytes_len: u64,
}

impl ModelInstance {
    /// Build a model identity from its stable id, byte hash, and byte length.
    #[must_use]
    pub fn new(model_id: impl Into<String>, bytes_sha256: [u8; 32], bytes_len: u64) -> Self {
        Self {
            model_id: model_id.into(),
            bytes_sha256,
            bytes_len,
        }
    }

    /// The stable model id (e.g. the row name).
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The SHA-256 of the model bytes (the load-once identity fact).
    #[must_use]
    pub const fn bytes_sha256(&self) -> &[u8; 32] {
        &self.bytes_sha256
    }

    /// The model byte length (a per-execution re-copy would re-upload this
    /// many bytes — the discovery's infeasibility figure).
    #[must_use]
    pub const fn bytes_len(&self) -> u64 {
        self.bytes_len
    }
}

impl fmt::Display for ModelInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model:{} sha256:{:.16}…", self.model_id, self.short_sha256())
    }
}

impl ModelInstance {
    /// The first 16 hex chars of the byte hash — a stable short identity.
    #[must_use]
    pub fn short_sha256(&self) -> String {
        let mut out = String::with_capacity(32);
        for byte in &self.bytes_sha256[..8] {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ReuseKey — reuse/invalidation keys (session / sequence / epoch)
// ---------------------------------------------------------------------------

/// The **reuse/invalidation key**: `(session, sequence, epoch)`.
///
/// A resident resource (weights, KV) is reusable by a request exactly when
/// every component of its stored key matches the requested key — a new
/// session, a new sequence, or a bumped epoch invalidates it. The epoch is
/// the invalidation handle for reset/reuse decisions (GI4-3/4 flesh the
/// runtime behavior; the key rule freezes here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReuseKey {
    session_id: u64,
    sequence_id: u64,
    epoch: u64,
}

impl ReuseKey {
    /// Build a reuse key from its three identity components.
    #[must_use]
    pub const fn new(session_id: u64, sequence_id: u64, epoch: u64) -> Self {
        Self {
            session_id,
            sequence_id,
            epoch,
        }
    }

    /// The session component.
    #[must_use]
    pub const fn session_id(self) -> u64 {
        self.session_id
    }

    /// The sequence component.
    #[must_use]
    pub const fn sequence_id(self) -> u64 {
        self.sequence_id
    }

    /// The epoch component (bumping it invalidates resident resources).
    #[must_use]
    pub const fn epoch(self) -> u64 {
        self.epoch
    }

    /// A copy with a new epoch — the invalidation operation.
    #[must_use]
    pub const fn with_epoch(self, epoch: u64) -> Self {
        Self { epoch, ..self }
    }
}

/// Whether a resident resource stored under `stored` is reusable by a
/// request carrying `requested`. All three components must match.
#[must_use]
pub fn resident_reusable(stored: &ReuseKey, requested: &ReuseKey) -> bool {
    stored.session_id == requested.session_id
        && stored.sequence_id == requested.sequence_id
        && stored.epoch == requested.epoch
}

// ---------------------------------------------------------------------------
// Workload mode + per-invocation input cadence (the missing vocabulary, FC3)
// ---------------------------------------------------------------------------

/// The **workload mode** of an invocation (CTO S5; lane queue §2): one
/// prefill step or one scalar-decode step. The regime label for engine
/// receipts (prefill time vs steady-state decode time reported separately,
/// GI4-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvocationMode {
    /// A prefill step (consumes the next prompt token; writes one KV
    /// generation per layer at the current position).
    Prefill,
    /// A one-token scalar decode step (consumes the fed-back token; writes
    /// the single new score row into KV).
    ScalarDecode,
}

impl InvocationMode {
    /// Stable diagnostic spelling (`"prefill"` / `"scalar_decode"`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::ScalarDecode => "scalar_decode",
        }
    }

    /// True for a prefill step.
    #[must_use]
    pub const fn is_prefill(self) -> bool {
        matches!(self, Self::Prefill)
    }

    /// True for a scalar-decode step.
    #[must_use]
    pub const fn is_decode(self) -> bool {
        matches!(self, Self::ScalarDecode)
    }
}

impl fmt::Display for InvocationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.spelling())
    }
}

/// The **per-invocation input cadence** (lane queue §2 `InputUpdateCadence`):
/// which inputs update per token versus stay resident across the session.
///
/// Resident inputs (weights, KV K/V) are uploaded exactly once at session
/// creation and never re-copied; per-invocation inputs (decode token id, the
/// per-position RoPE cos/sin table row) update on every invocation. This is
/// the cadence fact `SingleRun`'s copy-per-execution surface and the
/// training-shaped `RepeatingStep` surface both lack (FC2/FC3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputUpdateCadence {
    /// Uploaded once at session creation; never re-copied per invocation
    /// (weights, KV K/V).
    Resident,
    /// Updated on every invocation (decode token id, per-position RoPE
    /// table — a per-invocation updated input, never a plan change).
    PerInvocation,
}

impl InputUpdateCadence {
    /// Stable diagnostic spelling (`"resident"` / `"per_invocation"`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::PerInvocation => "per_invocation",
        }
    }

    /// True for the resident (once-at-session) cadence.
    #[must_use]
    pub const fn is_resident(self) -> bool {
        matches!(self, Self::Resident)
    }

    /// True for the per-invocation cadence.
    #[must_use]
    pub const fn is_per_invocation(self) -> bool {
        matches!(self, Self::PerInvocation)
    }
}

// ---------------------------------------------------------------------------
// SequenceState + the transactional token mutation (identity 3 of 4)
// ---------------------------------------------------------------------------

/// The **`SequenceState`** identity: position, token history, KV generations.
///
/// Holds the per-sequence inference state: the next token position, the
/// token history (prompt then generated), and the committed KV generation
/// count. All of it advances **only** through a committed [`TokenCommit`]
/// (the transactional token-mutation rule) — nothing here moves on a failed
/// or uncommitted step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceState {
    position: u32,
    token_history: Vec<i64>,
    kv_generations: u64,
}

impl SequenceState {
    /// The initial sequence state for a prompt: `position` = prompt length,
    /// `kv_generations` = prompt length (prefill writes one K/V generation
    /// per prompt position per layer), `token_history` = the prompt tokens.
    #[must_use]
    pub fn new(prompt: Vec<i64>) -> Self {
        let len = prompt.len();
        Self {
            position: len as u32,
            token_history: prompt,
            kv_generations: len as u64,
        }
    }

    /// The next token position (absolute; prompt positions then generated).
    #[must_use]
    pub const fn position(&self) -> u32 {
        self.position
    }

    /// The token history: the prompt tokens followed by every committed
    /// generated token.
    #[must_use]
    pub fn token_history(&self) -> &[i64] {
        &self.token_history
    }

    /// The committed KV generation count (prompt positions + committed
    /// decode tokens).
    #[must_use]
    pub const fn kv_generations(&self) -> u64 {
        self.kv_generations
    }

    /// Commit one token transactionally (the GI4-1 transactional
    /// token-mutation rule; MD-A13 precedent).
    ///
    /// The token id, the sequence position, the KV generations, and the
    /// visible output advance **together**, and only after the token
    /// commits. A commit that is out of order (wrong position or a KV
    /// generation gap) fails **before** any field moves, leaving the last
    /// committed token/position authoritative. Retry is disabled — the only
    /// resumption point is the last committed generation, and replay from it
    /// must be proven deterministic before a retry may be attempted
    /// ([`SequenceCommitError::is_retryable`] is `false` for every variant).
    ///
    /// The visible output of a greedy decode is the committed token itself
    /// (host-side argmax over the read-back logits, GI4-3 default); because
    /// the commit is all-or-nothing, the visible output can never advance
    /// past a failed commit.
    pub fn commit(&mut self, commit: &TokenCommit) -> Result<(), SequenceCommitError> {
        if commit.position != self.position {
            return Err(SequenceCommitError::OutOfOrderPosition {
                expected: self.position,
                attempted: commit.position,
            });
        }
        // KV generations advance together with the position: the commit must
        // name the *next* generation (current + 1) — a gap means the KV
        // cache would be written out of order.
        let expected_generation = self.kv_generations + 1;
        if commit.kv_generation != expected_generation {
            return Err(SequenceCommitError::KvGenerationGap {
                expected: expected_generation,
                attempted: commit.kv_generation,
            });
        }
        self.token_history.push(commit.token);
        self.position += 1;
        self.kv_generations += 1;
        Ok(())
    }
}

/// A token that is about to commit against a [`SequenceState`].
///
/// The four mutation facts — token id, sequence position, KV generation, and
/// (for greedy decode) the visible output — are declared together so the
/// transactional rule can advance them atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCommit {
    /// The token id being committed.
    pub token: i64,
    /// The sequence position this token commits at (must equal the
    /// sequence's current position).
    pub position: u32,
    /// The KV generation this token writes (must equal the sequence's next
    /// generation — current + 1).
    pub kv_generation: u64,
}

/// A refused token commit — the transactional rule's failure surface.
///
/// Every variant leaves the sequence state **unchanged** (the last committed
/// token/position stays authoritative). No variant is retryable without a
/// proven-deterministic replay from the last committed generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceCommitError {
    /// The commit named the wrong position (e.g. retry from an uncommitted
    /// token).
    OutOfOrderPosition {
        /// The sequence's current (last committed) position.
        expected: u32,
        /// The position the commit attempted.
        attempted: u32,
    },
    /// The commit named the wrong KV generation (KV would be written out of
    /// order).
    KvGenerationGap {
        /// The next generation (last committed + 1).
        expected: u64,
        /// The generation the commit attempted.
        attempted: u64,
    },
}

impl SequenceCommitError {
    /// The retry rule: `false` for every variant — retry is disabled unless
    /// replay from the last committed generation is proven deterministic
    /// (the transactional token-mutation rule, CTO S5; MD-A13 precedent).
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Invocation — one prefill or one-token decode with declared inputs/outputs
// ---------------------------------------------------------------------------

/// What an invocation declares as its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvocationOutput {
    /// Full-vocab raw logits `[49152]` (the pinned row's tied-head
    /// projection) — read back once, sampled host-side (GI4-3 default).
    Logits,
    /// A selected token (the committed visible output).
    SelectedToken,
}

/// The **`Invocation`** identity: one prefill step or one-token decode with
/// declared inputs and outputs (identity 4 of 4).
///
/// Per-invocation inputs are the token id and the absolute position the step
/// executes at; resident inputs (weights, KV) never appear on an invocation —
/// they live on the [`ExecutionSession`] under their [`InputUpdateCadence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    /// The workload mode of this step.
    pub mode: InvocationMode,
    /// The input token: the next prompt token at `position` for a prefill
    /// step, or the fed-back sampled token for a scalar-decode step.
    pub token: i64,
    /// The absolute sequence position this step executes at.
    pub position: u32,
    /// The declared output surface.
    pub output: InvocationOutput,
}

impl Invocation {
    /// Build a prefill invocation (workload mode `prefill`).
    #[must_use]
    pub const fn prefill(token: i64, position: u32) -> Self {
        Self {
            mode: InvocationMode::Prefill,
            token,
            position,
            output: InvocationOutput::Logits,
        }
    }

    /// Build a scalar-decode invocation (workload mode `scalar_decode`).
    #[must_use]
    pub const fn scalar_decode(token: i64, position: u32) -> Self {
        Self {
            mode: InvocationMode::ScalarDecode,
            token,
            position,
            output: InvocationOutput::Logits,
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionSession — one runtime session with resident weights + KV (2 of 4)
// ---------------------------------------------------------------------------

/// The **`ExecutionSession`** identity: one runtime session with resident
/// weights and KV (identity 2 of 4).
///
/// A session binds exactly one [`ModelInstance`] (load-once: the model's
/// resident inputs are uploaded once at session creation and never
/// re-copied), declares the typed [`KvCacheLayout`] under which the resident
/// KV is allocated, and carries the current [`SequenceState`] plus the
/// [`ReuseKey`] its resident resources were minted under. Per-invocation
/// inputs ride [`Invocation`]s — never the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSession {
    session_id: u64,
    model: ModelInstance,
    kv_layout: KvCacheLayout,
    epoch: u64,
    sequence: SequenceState,
}

impl ExecutionSession {
    /// Create one runtime session: the model + KV layout become resident.
    #[must_use]
    pub fn new(
        session_id: u64,
        model: ModelInstance,
        kv_layout: KvCacheLayout,
        epoch: u64,
        sequence: SequenceState,
    ) -> Self {
        Self {
            session_id,
            model,
            kv_layout,
            epoch,
            sequence,
        }
    }

    /// The session's opaque id.
    #[must_use]
    pub const fn session_id(&self) -> u64 {
        self.session_id
    }

    /// The load-once model identity bound to this session.
    #[must_use]
    pub fn model(&self) -> &ModelInstance {
        &self.model
    }

    /// The typed KV layout under which the resident KV is allocated.
    #[must_use]
    pub const fn kv_layout(&self) -> &KvCacheLayout {
        &self.kv_layout
    }

    /// The invalidation epoch the resident resources were minted under.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The session's current sequence state (GI4 is one sequence, one slot).
    #[must_use]
    pub fn sequence(&self) -> &SequenceState {
        &self.sequence
    }

    /// Mutable access to the sequence state (token commits only through
    /// [`SequenceState::commit`] — the transactional rule).
    pub fn sequence_mut(&mut self) -> &mut SequenceState {
        &mut self.sequence
    }

    /// The reuse key the session's resident resources were minted under.
    #[must_use]
    pub const fn reuse_key(&self) -> ReuseKey {
        ReuseKey::new(self.session_id, self.sequence_id_implicit(), self.epoch)
    }

    /// The implicit sequence component of the reuse key (one sequence, one
    /// slot in GI4 — the session's sequence id is its session id).
    #[must_use]
    const fn sequence_id_implicit(&self) -> u64 {
        self.session_id
    }
}
