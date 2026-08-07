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
//! transactional [`TokenCommit`] rule, and the **two residency keys**
//! (council `eb47fd88` item 1). **No wire edit here** — the carriage on the
//! wire is GI4-2's unit (nothing in this module touches
//! `radix-mir-fmir/src/schema/**`).
//!
//! ## Council corrections (head-cto `eb47fd88` item 1, folded before GI4-2
//! wire carriage)
//!
//! The frozen shape was corrected by the 2026-08-07 council before the wire
//! serialized it (the cheap-now window): **two separate invalidation
//! domains** — model weights belong to [`ModelInstance`] residency, KV state
//! belongs to a sequence + its epoch, so advancing/resetting/replacing a
//! sequence never invalidates otherwise reusable model weights (the original
//! `ReuseKey(session, sequence, epoch)` conflated the domains);
//! [`SequenceState`] is **independently identified** (one-active-sequence
//! admission is a policy limit, not a structural derivation from the session
//! identity); [`Invocation`] refers **explicitly** to the selected sequence;
//! and token advancement + KV publication flow **through the
//! [`ExecutionTransaction`](crate::execution_transaction::ExecutionTransaction)
//! publication boundary** — a failed/cancelled invocation publishes neither
//! a token advance nor partial KV writes, with the retry/idempotency rule
//! explicit ([`SequenceCommitOutcome::IdempotentReplay`]).

use crate::execution_transaction::TransactionCommitBoundary;
use crate::kv_cache::KvCacheLayout;
use std::fmt;

/// Version of this session contract freeze. Bumped only by a *contract*
/// revision; the GI4-2 wire carriage rides the accepted wire version and
/// never invents an unversioned fact. Rev 1 → 2: the council `eb47fd88` item 1
/// corrections (separate residency domains, independent sequence identity,
/// explicit sequence selection, ExecutionTransaction-boundary publication).
pub const SESSION_CONTRACT_VERSION: u32 = 2;

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
// Residency keys — two separate invalidation domains (council eb47fd88)
// ---------------------------------------------------------------------------

/// The **model-weights residency key**: `(session, model epoch)`.
///
/// Model weights belong to [`ModelInstance`] residency — resident on the
/// session, uploaded once at session creation. This key has **no sequence
/// component**: advancing, resetting, or replacing a sequence must never
/// invalidate otherwise reusable model weights. A new session or a bumped
/// model epoch invalidates the resident weights. The KV/sequence domain is a
/// **separate** key ([`SequenceResidencyKey`]) — the two domains never share
/// an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelResidencyKey {
    session_id: u64,
    model_epoch: u64,
}

impl ModelResidencyKey {
    /// Build the model-weights residency key from its two identity
    /// components.
    #[must_use]
    pub const fn new(session_id: u64, model_epoch: u64) -> Self {
        Self {
            session_id,
            model_epoch,
        }
    }

    /// The session component.
    #[must_use]
    pub const fn session_id(self) -> u64 {
        self.session_id
    }

    /// The model-weights residency epoch (bumping it invalidates the
    /// resident model weights — never the KV state).
    #[must_use]
    pub const fn model_epoch(self) -> u64 {
        self.model_epoch
    }
}

/// The **sequence/KV-residency key**: `(session, sequence, epoch)`.
///
/// KV state belongs to a sequence **and its epoch**. A new sequence, or a
/// bumped epoch, invalidates the resident KV state — and **only** the KV
/// state: the model weights live under their own key
/// ([`ModelResidencyKey`]), so a sequence advance/reset/replacement never
/// invalidates otherwise reusable model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceResidencyKey {
    session_id: u64,
    sequence_id: u64,
    epoch: u64,
}

impl SequenceResidencyKey {
    /// Build the sequence/KV-residency key from its three identity
    /// components.
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

    /// The sequence component (the sequence's **independent** identity).
    #[must_use]
    pub const fn sequence_id(self) -> u64 {
        self.sequence_id
    }

    /// The KV/sequence-residency epoch (bumping it invalidates the resident
    /// KV state — never the model weights).
    #[must_use]
    pub const fn epoch(self) -> u64 {
        self.epoch
    }
}

/// Whether the resident **model weights** stored under `stored` are reusable
/// by a request carrying `requested`. Both components (session, model epoch)
/// must match. The sequence domain is deliberately absent from this rule —
/// a sequence advance/reset/replacement never invalidates model weights.
#[must_use]
pub fn model_weights_reusable(stored: &ModelResidencyKey, requested: &ModelResidencyKey) -> bool {
    stored == requested
}

/// Whether the resident **KV state** stored under `stored` is reusable by a
/// request carrying `requested`. All three components (session, sequence,
/// epoch) must match. A new sequence or a bumped epoch invalidates only the
/// KV state — never the model weights ([`model_weights_reusable`] is the
/// separate domain rule).
#[must_use]
pub fn sequence_kv_reusable(stored: &SequenceResidencyKey, requested: &SequenceResidencyKey) -> bool {
    stored == requested
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

/// The **`SequenceState`** identity: sequence id, position, token history,
/// KV generations.
///
/// Holds the per-sequence inference state: the **independent sequence id**
/// (council `eb47fd88` item 1 — never structurally derived from the session
/// identity; an admission limit of one active sequence per session is a
/// policy fact, not a structural ABI limit), the next token position, the
/// token history (prompt then generated), and the committed KV generation
/// count. All of it advances **only** through a committed [`TokenCommit`]
/// (the transactional token-mutation rule) — nothing here moves on a failed
/// or uncommitted step, and a published token is published through the
/// [`ExecutionTransaction`](crate::execution_transaction::ExecutionTransaction)
/// publication boundary ([`TokenCommit::publication_boundary`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceState {
    sequence_id: u64,
    position: u32,
    token_history: Vec<i64>,
    kv_generations: u64,
}

impl SequenceState {
    /// The initial sequence state for a prompt: `sequence_id` is the
    /// sequence's independent identity, `position` = prompt length,
    /// `kv_generations` = prompt length (prefill writes one K/V generation
    /// per prompt position per layer), `token_history` = the prompt tokens.
    #[must_use]
    pub fn new(sequence_id: u64, prompt: Vec<i64>) -> Self {
        let len = prompt.len();
        Self {
            sequence_id,
            position: len as u32,
            token_history: prompt,
            kv_generations: len as u64,
        }
    }

    /// The sequence's **independent** identity (the component of the
    /// sequence/KV-residency key; never the session id by construction).
    #[must_use]
    pub const fn sequence_id(&self) -> u64 {
        self.sequence_id
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
    /// token-mutation rule; MD-A13 precedent), published **through the
    /// [`ExecutionTransaction`](crate::execution_transaction::ExecutionTransaction)
    /// publication boundary**.
    ///
    /// Publication preconditions, in order:
    /// 1. The commit declares a **non-empty publication boundary** — a token
    ///    that would advance outside a reached transaction boundary is
    ///    refused before any field moves ([`SequenceCommitError::
    ///    NoPublicationBoundary`]). A failed/cancelled invocation publishes
    ///    **neither** a token advance **nor** partial KV writes.
    /// 2. An **idempotent replay** — the exact last committed token
    ///    re-submitted (a lost-ack re-acknowledgment, same token/position/
    ///    KV generation) — is a **no-op success**
    ///    ([`SequenceCommitOutcome::IdempotentReplay`]): nothing advances,
    ///    so a committed token is never double-advanced. This is *not* a
    ///    retry — no uncommitted work re-executes.
    /// 3. Out of order (wrong position) or a KV-generation gap fails
    ///    **before** any field moves — the last committed token/position
    ///    stays authoritative.
    ///
    /// The token id, the sequence position, the KV generations, and the
    /// visible output advance **together**, and only after the token
    /// commits. Retry is otherwise disabled — the only resumption point is
    /// the last committed generation, and replay from it must be proven
    /// deterministic before a retry may be attempted
    /// ([`SequenceCommitError::is_retryable`] is `false` for every variant).
    ///
    /// The visible output of a greedy decode is the committed token itself
    /// (host-side argmax over the read-back logits, GI4-3 default); because
    /// the commit is all-or-nothing, the visible output can never advance
    /// past a failed commit.
    pub fn commit(&mut self, commit: &TokenCommit) -> Result<SequenceCommitOutcome, SequenceCommitError> {
        // Publication boundary precondition: token advancement + KV
        // publication flow through a reached ExecutionTransaction boundary.
        if commit.publication_boundary.is_empty() {
            return Err(SequenceCommitError::NoPublicationBoundary);
        }
        // Idempotent replay: the exact last committed token re-submitted —
        // a no-op, never a double advance.
        if commit.position == self.position.saturating_sub(1)
            && commit.kv_generation == self.kv_generations
            && self.token_history.last() == Some(&commit.token)
        {
            return Ok(SequenceCommitOutcome::IdempotentReplay);
        }
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
        Ok(SequenceCommitOutcome::Applied)
    }
}

/// A token that is about to commit against a [`SequenceState`].
///
/// The five mutation facts — token id, sequence position, KV generation, the
/// [`ExecutionTransaction`](crate::execution_transaction::ExecutionTransaction)
/// publication boundary the token's publication reached, and (for greedy
/// decode) the visible output — are declared together so the transactional
/// rule can advance them atomically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenCommit {
    /// The token id being committed.
    pub token: i64,
    /// The sequence position this token commits at (must equal the
    /// sequence's current position).
    pub position: u32,
    /// The KV generation this token writes (must equal the sequence's next
    /// generation — current + 1).
    pub kv_generation: u64,
    /// The ExecutionTransaction publication boundary the token and its KV
    /// writes published through — a **non-empty** boundary is required
    /// (nothing publishes outside a reached boundary; a failed/cancelled
    /// invocation publishes neither a token advance nor partial KV writes).
    pub publication_boundary: TransactionCommitBoundary,
}

/// The outcome of a [`TokenCommit`] against a [`SequenceState`] — the
/// retry/idempotency rule made explicit (council `eb47fd88` item 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceCommitOutcome {
    /// The commit applied: token, position, KV generations (and the visible
    /// output) advanced **together**.
    Applied,
    /// The commit was an **exact idempotent replay** of the last committed
    /// token (same token, position, KV generation): nothing moved — the
    /// invocation that already committed never double-advances. This is a
    /// lost-ack dedup, **not** a retry of uncommitted work.
    IdempotentReplay,
}

/// A refused token commit — the transactional rule's failure surface.
///
/// Every variant leaves the sequence state **unchanged** (the last committed
/// token/position stays authoritative). No variant is retryable without a
/// proven-deterministic replay from the last committed generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceCommitError {
    /// The commit declared an **empty publication boundary** — a token that
    /// would advance outside a reached
    /// [`ExecutionTransaction`](crate::execution_transaction::ExecutionTransaction)
    /// boundary. Nothing publishes outside a transaction.
    NoPublicationBoundary,
    /// The commit named the wrong position (e.g. a retry of a different
    /// token from an uncommitted position).
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
    /// (the transactional token-mutation rule, CTO S5; MD-A13 precedent). An
    /// idempotent replay is not a retry (nothing re-executes) — it is a
    /// no-op dedup handled by [`SequenceState::commit`].
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
/// Every invocation refers **explicitly** to the selected sequence (council
/// `eb47fd88` item 1) — `sequence_id` is a declared field, never derived
/// from the session. Per-invocation inputs are the selected sequence, the
/// token id, and the absolute position the step executes at; resident inputs
/// (weights, KV) never appear on an invocation — they live on the
/// [`ExecutionSession`] under their [`InputUpdateCadence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    /// The workload mode of this step.
    pub mode: InvocationMode,
    /// The selected sequence this invocation executes against (an explicit
    /// field — the invocation names its sequence).
    pub sequence_id: u64,
    /// The input token: the next prompt token at `position` for a prefill
    /// step, or the fed-back sampled token for a scalar-decode step.
    pub token: i64,
    /// The absolute sequence position this step executes at.
    pub position: u32,
    /// The declared output surface.
    pub output: InvocationOutput,
}

impl Invocation {
    /// Build a prefill invocation against the selected sequence (workload
    /// mode `prefill`).
    #[must_use]
    pub const fn prefill(sequence_id: u64, token: i64, position: u32) -> Self {
        Self {
            mode: InvocationMode::Prefill,
            sequence_id,
            token,
            position,
            output: InvocationOutput::Logits,
        }
    }

    /// Build a scalar-decode invocation against the selected sequence
    /// (workload mode `scalar_decode`).
    #[must_use]
    pub const fn scalar_decode(sequence_id: u64, token: i64, position: u32) -> Self {
        Self {
            mode: InvocationMode::ScalarDecode,
            sequence_id,
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
/// KV is allocated, and carries the current [`SequenceState`] plus **two
/// separate residency keys** (council `eb47fd88` item 1): the model weights
/// are resident under a model-weights key (`model_residency_key` —
/// `(session, model epoch)`), the KV state under a sequence/KV key
/// (`sequence_residency_key` — `(session, sequence, epoch)`). The two
/// domains never share an epoch: advancing/resetting/replacing a sequence
/// invalidates the KV residency only, never the model weights. Per-invocation
/// inputs ride [`Invocation`]s — never the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSession {
    session_id: u64,
    model: ModelInstance,
    kv_layout: KvCacheLayout,
    model_epoch: u64,
    sequence_epoch: u64,
    sequence: SequenceState,
}

impl ExecutionSession {
    /// Create one runtime session: the model + KV layout become resident.
    /// `model_epoch` is the model-weights residency invalidation epoch;
    /// `sequence_epoch` is the KV/sequence residency invalidation epoch
    /// (separate domains — a bump in one never invalidates the other).
    #[must_use]
    pub fn new(
        session_id: u64,
        model: ModelInstance,
        kv_layout: KvCacheLayout,
        model_epoch: u64,
        sequence_epoch: u64,
        sequence: SequenceState,
    ) -> Self {
        Self {
            session_id,
            model,
            kv_layout,
            model_epoch,
            sequence_epoch,
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

    /// The model-weights residency invalidation epoch (bumping it
    /// invalidates the resident weights — never the KV state).
    #[must_use]
    pub const fn model_epoch(&self) -> u64 {
        self.model_epoch
    }

    /// The KV/sequence residency invalidation epoch (bumping it invalidates
    /// the resident KV state — never the model weights).
    #[must_use]
    pub const fn sequence_epoch(&self) -> u64 {
        self.sequence_epoch
    }

    /// The session's current sequence state (GI4 is one sequence, one slot —
    /// an admission limit, not a structural identity derivation).
    #[must_use]
    pub fn sequence(&self) -> &SequenceState {
        &self.sequence
    }

    /// Mutable access to the sequence state (token commits only through
    /// [`SequenceState::commit`] — the transactional rule).
    pub fn sequence_mut(&mut self) -> &mut SequenceState {
        &mut self.sequence
    }

    /// The model-weights residency key the session's resident weights were
    /// minted under: `(session, model epoch)`. Sequence advances/resets/
    /// replacements never touch this key.
    #[must_use]
    pub const fn model_residency_key(&self) -> ModelResidencyKey {
        ModelResidencyKey::new(self.session_id, self.model_epoch)
    }

    /// The sequence/KV residency key the session's resident KV was minted
    /// under: `(session, sequence id, sequence epoch)` — the sequence id is
    /// the sequence's **independent** identity ([`SequenceState::
    /// sequence_id`]), never the session id by construction.
    #[must_use]
    pub const fn sequence_residency_key(&self) -> SequenceResidencyKey {
        SequenceResidencyKey::new(
            self.session_id,
            self.sequence.sequence_id(),
            self.sequence_epoch,
        )
    }
}
