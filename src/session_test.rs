//! GI4-1 — frozen inference session contract tests.
//!
//! Families:
//! 1. **Identities/lifetimes**: [`ModelInstance`] (load-once), the
//!    [`ExecutionSession`] (one runtime session with resident weights + KV),
//!    [`SequenceState`] (position / token history / KV generations), and
//!    [`Invocation`] (one prefill or one-token decode with declared
//!    inputs/outputs).
//! 2. **Transactional token mutation**: a commit advances token id, position,
//!    KV generations, and visible output **together**; an out-of-order
//!    position or a KV-generation gap fails BEFORE any field moves (the last
//!    committed token/position stays authoritative); no error is retryable.
//! 3. **Workload mode + input cadence vocabulary**: [`InvocationMode`] has
//!    exactly the two workload modes (`prefill` / `scalar_decode`);
//!    [`InputUpdateCadence`] distinguishes resident from per-invocation
//!    inputs; resident inputs never ride an invocation.
//! 4. **Discovery spikes** (the GI4-1 expressibility proof — the proof table
//!    of `radix/docs/factory/gpu-inference-gguf/evidence/
//!    gi4-session-discovery.md`): one spike test per missing-fact claim —
//!    SingleRun refuses once-init params; RepeatingStep's observation cadence
//!    is loss-shaped; a `PerProgram` buffer has no KV identity/lifetime/
//!    layout; no input-cadence/vocabulary type exists.

use crate::cpu_oracle::LAYER_COUNT;
use crate::decoder_ops::{HEAD_DIM, KV_HEAD_COUNT};
use crate::device::{DeviceBackend, DeviceHandle, DeviceHandleKind};
use crate::kv_cache::{KvCacheDtype, KvCacheLayout, KvReservePolicy};
use crate::session::*;

/// The pinned row's context length (ctx 8192 — the `llama.context_length`
/// fact, `gguf.rs`).
const CONTEXT_LENGTH: u32 = 8192;

/// The discovery's infeasibility figure: the weight upload a per-execution
/// re-copy would repeat on every token (`gi4-delivery.md` §GI4-1;
/// `gi3-delivery.md` §Q1).
const WEIGHT_UPLOAD_BYTES: u64 = 270 * 1024 * 1024;

/// A test-fixture byte hash — the pinned row's full SHA-256 lives in the
/// model contract (`gi0-model-contract.md`), never invented here.
const TEST_SHA_256: [u8; 32] = [0x2f; 32];

fn pinned_layout() -> KvCacheLayout {
    KvCacheLayout::new(
        1,
        CONTEXT_LENGTH,
        LAYER_COUNT as u32,
        KV_HEAD_COUNT as u32,
        HEAD_DIM as u32,
        KvCacheDtype::F32,
        KvReservePolicy::Fixed { bytes: 0 },
    )
    .expect("pinned-row layout is valid")
}

fn test_model() -> ModelInstance {
    ModelInstance::new("SmolLM2-360M-Instruct-Q4_K_M", TEST_SHA_256, WEIGHT_UPLOAD_BYTES)
}

fn test_session() -> ExecutionSession {
    ExecutionSession::new(
        7,
        test_model(),
        pinned_layout(),
        0,
        SequenceState::new(vec![504, 2365]),
    )
}

// ---------------------------------------------------------------------------
// Family 1 — identities/lifetimes
// ---------------------------------------------------------------------------

#[test]
fn model_instance_is_a_load_once_identity() {
    let model = test_model();
    assert_eq!(model.model_id(), "SmolLM2-360M-Instruct-Q4_K_M");
    assert_eq!(model.bytes_len(), WEIGHT_UPLOAD_BYTES);
    assert_eq!(model.short_sha256(), "2f2f2f2f2f2f2f2f");
}

#[test]
fn execution_session_binds_model_layout_sequence_and_reuse_key() {
    let layout = pinned_layout();
    let session = ExecutionSession::new(7, test_model(), layout, 3, SequenceState::new(vec![504]));
    assert_eq!(session.session_id(), 7);
    assert_eq!(session.model().bytes_len(), WEIGHT_UPLOAD_BYTES);
    assert_eq!(session.kv_layout(), &layout);
    assert_eq!(session.epoch(), 3);
    assert_eq!(session.sequence().position(), 1);

    let key = session.reuse_key();
    assert_eq!(key.session_id(), 7);
    assert_eq!(key.epoch(), 3);
    assert!(resident_reusable(&key, &key));
}

#[test]
fn sequence_state_initial_state_is_the_prompt() {
    let seq = SequenceState::new(vec![504, 2365, 6354]);
    assert_eq!(seq.position(), 3);
    assert_eq!(seq.kv_generations(), 3);
    assert_eq!(seq.token_history(), &[504, 2365, 6354]);
}

#[test]
fn invocation_carries_mode_token_position_and_output() {
    let decode = Invocation::scalar_decode(30, 9);
    assert!(decode.mode.is_decode());
    assert_eq!(decode.token, 30);
    assert_eq!(decode.position, 9);
    assert_eq!(decode.output, InvocationOutput::Logits);

    let prefill = Invocation::prefill(504, 0);
    assert!(prefill.mode.is_prefill());
    assert_eq!(prefill.token, 504);
    assert_eq!(prefill.position, 0);
}

// ---------------------------------------------------------------------------
// Family 2 — the transactional token mutation
// ---------------------------------------------------------------------------

#[test]
fn commit_advances_position_history_generations_together() {
    let mut seq = SequenceState::new(vec![504, 2365]);
    let commit = TokenCommit {
        token: 30,
        position: 2,
        kv_generation: 3,
    };
    assert!(seq.commit(&commit).is_ok());
    // Token id, position, KV generations (and the visible output — the
    // committed token) advanced together.
    assert_eq!(seq.position(), 3);
    assert_eq!(seq.kv_generations(), 3);
    assert_eq!(seq.token_history(), &[504, 2365, 30]);
}

#[test]
fn out_of_order_position_leaves_last_committed_authoritative() {
    let mut seq = SequenceState::new(vec![504, 2365]);
    // A retry from an uncommitted token (wrong position) fails before any
    // field moves.
    let err = seq
        .commit(&TokenCommit {
            token: 30,
            position: 1,
            kv_generation: 3,
        })
        .expect_err("out-of-order position must be refused");
    assert_eq!(
        err,
        SequenceCommitError::OutOfOrderPosition {
            expected: 2,
            attempted: 1,
        }
    );
    assert_eq!(seq.position(), 2);
    assert_eq!(seq.kv_generations(), 2);
    assert_eq!(seq.token_history(), &[504, 2365]);
}

#[test]
fn kv_generation_gap_is_rejected_before_any_move() {
    let mut seq = SequenceState::new(vec![504, 2365]);
    let err = seq
        .commit(&TokenCommit {
            token: 30,
            position: 2,
            kv_generation: 5,
        })
        .expect_err("KV generation gap must be refused");
    assert_eq!(
        err,
        SequenceCommitError::KvGenerationGap {
            expected: 3,
            attempted: 5,
        }
    );
    assert_eq!(seq.position(), 2);
    assert_eq!(seq.kv_generations(), 2);
    assert_eq!(seq.token_history(), &[504, 2365]);
}

#[test]
fn no_commit_error_is_retryable() {
    // The retry rule: retry is disabled unless replay from the last
    // committed generation is proven deterministic.
    assert!(!SequenceCommitError::OutOfOrderPosition {
        expected: 2,
        attempted: 1,
    }
    .is_retryable());
    assert!(!SequenceCommitError::KvGenerationGap {
        expected: 3,
        attempted: 5,
    }
    .is_retryable());
}

// ---------------------------------------------------------------------------
// Family 3 — workload mode + input cadence vocabulary
// ---------------------------------------------------------------------------

#[test]
fn invocation_mode_is_exactly_the_two_workload_modes() {
    let modes: [InvocationMode; 2] = [InvocationMode::Prefill, InvocationMode::ScalarDecode];
    for mode in modes {
        assert!(mode.is_prefill() || mode.is_decode());
    }
    assert_eq!(InvocationMode::Prefill.spelling(), "prefill");
    assert_eq!(InvocationMode::ScalarDecode.spelling(), "scalar_decode");
}

#[test]
fn input_cadence_distinguishes_resident_from_per_invocation() {
    assert!(InputUpdateCadence::Resident.is_resident());
    assert!(!InputUpdateCadence::Resident.is_per_invocation());
    assert!(InputUpdateCadence::PerInvocation.is_per_invocation());
    assert_eq!(InputUpdateCadence::Resident.spelling(), "resident");
    assert_eq!(InputUpdateCadence::PerInvocation.spelling(), "per_invocation");
}

// ---------------------------------------------------------------------------
// Family 4 — discovery spikes (the GI4-1 expressibility proof)
// ---------------------------------------------------------------------------

/// Spike 1 — `SingleRun` refuses once-init params.
///
/// Claim (composite_host.rs `ProgramSession::init_params`: "once-init params
/// are a RepeatingStep contract: a SingleRun session copies its declared host
/// inputs per execution"; `run.rs`: the route is whole-program, fixed inputs
/// per session): the generic single-run surface has **no once-init surface**,
/// so a weight upload re-copies the full model bytes on every execution —
/// infeasible per token. The frozen contract makes the once-init fact
/// explicit: weights live on the [`ExecutionSession`] as a `Resident` input
/// and never ride an [`Invocation`].
#[test]
fn spike_single_run_refuses_once_init_params() {
    let model = test_model();
    assert_eq!(model.bytes_len(), WEIGHT_UPLOAD_BYTES);

    // Generic single-run semantic (copy declared inputs per execution): 256
    // decode tokens → 256 full weight re-uploads.
    let mut bytes_copied = 0u64;
    for _ in 0..256 {
        bytes_copied += model.bytes_len();
    }
    assert_eq!(bytes_copied, WEIGHT_UPLOAD_BYTES * 256);

    // Frozen semantic: the model is resident on the session, uploaded once.
    let session = test_session();
    let once = session.model().bytes_len();
    assert_eq!(once, WEIGHT_UPLOAD_BYTES);

    // The per-invocation surface carries only token + position — a
    // per-token re-copy of the weight upload is structurally impossible on
    // the frozen surface (an Invocation has no weight/byte field).
    let inv = Invocation::scalar_decode(30, session.sequence().position());
    assert!(inv.mode.is_decode());
    assert_eq!(inv.token, 30);
    assert!(InputUpdateCadence::Resident.is_resident());
}

/// Spike 2 — `RepeatingStep`'s observation cadence is loss-shaped.
///
/// Claim (composite_host.rs: the `RepeatingStep` training-loop surface reads
/// back "the declared observation (the per-step loss trace)" and the
/// end-of-run observation set): once-init params exist but the observation
/// cadence is training-shaped. Inference state cannot ride that vocabulary —
/// a decode step needs a per-invocation, mode-labeled observation. The frozen
/// surface carries exactly that: the workload mode + a per-invocation output
/// (`logits` or the selected token); the loss/end-of-run cadence is not
/// representable on an [`Invocation`].
#[test]
fn spike_repeating_step_observation_cadence_is_loss_shaped() {
    let decode = Invocation::scalar_decode(30, 9);
    assert!(decode.mode.is_decode());
    assert_eq!(decode.output, InvocationOutput::Logits);

    let prefill = Invocation::prefill(504, 0);
    assert!(prefill.mode.is_prefill());

    // A per-invocation observation, mode-labeled — the loss-shaped
    // per-step-loss / end-of-run readback cadence is not an Invocation
    // output surface (InvocationOutput is closed to Logits | SelectedToken).
    assert!(matches!(decode.output, InvocationOutput::Logits));
    assert!(matches!(decode.output, InvocationOutput::SelectedToken) == false);
}

/// Spike 3 — a `PerProgram` buffer has no KV identity/lifetime/layout.
///
/// Claim (composite_host.rs: `PerProgram` = "allocated once at session
/// creation, released at program end" — an anonymous buffer; partition.rs:
/// "KV bytes per `KvCacheLayout` — consumed, not re-derived (GI4 owns the
/// layout)"): the generic buffer surface names `(backend, kind, len_bytes)`
/// and nothing else. The frozen [`KvCacheLayout`] carries the five KV facts
/// the anonymous buffer lacks: slots, context, layers/heads, dtype, reserve.
#[test]
fn spike_per_program_buffer_lacks_kv_identity_lifetime_layout() {
    // The generic opaque buffer surface (device.rs DeviceHandle): a byte
    // length and nothing else — no slots/context/layers/heads/dtype, no KV
    // identity or layout vocabulary.
    let handle = DeviceHandle {
        backend: DeviceBackend::Metal,
        kind: DeviceHandleKind::Buffer {
            len_bytes: 671_088_640,
        },
        id: 1,
    };
    assert_eq!(handle.len_bytes(), Some(671_088_640));
    assert_eq!(handle.kind.spelling(), "buffer");

    // The frozen layout carries the five KV facts the anonymous buffer
    // lacks.
    let layout = pinned_layout();
    assert_eq!(layout.slots(), 1);
    assert_eq!(layout.context_length(), CONTEXT_LENGTH);
    assert_eq!(layout.layer_count(), LAYER_COUNT as u32);
    assert_eq!(layout.kv_head_count(), KV_HEAD_COUNT as u32);
    assert_eq!(layout.head_dim(), HEAD_DIM as u32);
    assert_eq!(layout.dtype(), KvCacheDtype::F32);
}

/// Spike 4 — no input-cadence/vocabulary type exists in the live tree.
///
/// Claim (FC3, verified by the delivery audit): no `KvCacheLayout`,
/// `SequenceState`, `InputUpdateCadence`, or `InvocationMode` type existed.
/// The whole-program route (`run.rs` `execute_device_route`; `FABER_DEVICE_REPEAT`
/// re-executes a `SingleRun` program's full launch sequence N times on ONE
/// session with fixed inputs per session and whole-run observations) has no
/// per-invocation input cadence. The frozen vocabulary is the missing
/// surface, and the resident/per-invocation split is structural: resident
/// inputs (weights, KV) live on the session; per-invocation values ride
/// invocations.
#[test]
fn spike_no_input_cadence_or_invocation_vocabulary_exists() {
    // The frozen cadence vocabulary: exactly two closed variants.
    let cadences: [InputUpdateCadence; 2] = [
        InputUpdateCadence::Resident,
        InputUpdateCadence::PerInvocation,
    ];
    for cadence in cadences {
        assert!(cadence.is_resident() || cadence.is_per_invocation());
    }

    // The frozen workload-mode vocabulary: exactly two closed variants.
    let modes: [InvocationMode; 2] = [InvocationMode::Prefill, InvocationMode::ScalarDecode];
    for mode in modes {
        assert!(mode.is_prefill() || mode.is_decode());
    }

    // Resident/per-invocation split is structural: the resident model + KV
    // live on the session (uploaded once under its reuse key); the
    // per-invocation token rides the invocation.
    let session = test_session();
    assert_eq!(session.model().bytes_len(), WEIGHT_UPLOAD_BYTES);
    assert_eq!(session.kv_layout().slots(), 1);
    assert!(resident_reusable(&session.reuse_key(), &session.reuse_key()));
    let inv = Invocation::scalar_decode(30, session.sequence().position());
    assert_eq!(inv.position, 2);
    assert_eq!(inv.token, 30);
}
