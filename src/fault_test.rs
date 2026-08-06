//! MD3-F1 fault suite: the fake-device fault-injection backend
//! ([`FaultInjectingBackend`]) proving CAMPAIGN §MD3 exit-gate bullets 4 and
//! 5 at the virtual-fixture layer.
//!
//! - **Bullet 4 / MD-A13**: cancel, timeout, transfer error, kernel error,
//!   and device loss each release or retire **all affected resources with no
//!   partial commit** — the staged write-set is never partially published and
//!   the last committed state remains authoritative.
//! - **Bullet 5**: a degraded mesh (one partition's backend fails
//!   mid-execution) fails closed deterministically at the declared
//!   `TransactionCommitBoundary`, the degraded state is surfaced in the
//!   transaction receipt, and replanning is recorded as **MD5's surface**
//!   (MD3 never invents a replanner).
//!
//! The fixture is the T2 §5 **virtual** two-partition fixture over the
//! synthetic two-device snapshot (launch p0 → transfer 0→1 → broadcast →
//! barrier → launch p1, boundary on `barrier-main` + `launch-proj-b`).
//! Real independent device-loss stays **NOT ATTEMPTED** (single-device
//! acceptance host; lane queue §5b) — [`REAL_INDEPENDENT_DEVICE_LOSS_ROW`]
//! makes the absent real row explicit so it can never be mistaken for a pass.

use crate::bound_plan::{
    bind, AdmittedLogicalPlan, BoundDistributedPlan, DeclaredPlacementConstraint,
    LogicalPartitionId, PartitionBinding,
};
use crate::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use crate::device_set::DeviceSet;
use crate::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, P2pProbeState, ProbeProvenance,
};
use crate::execution_transaction::{
    BackendError, BarrierRef, CollectiveBroadcastMirror, CollectiveRef, CommitError,
    DeviceExecutionBackend, ExecuteError, ExecutionTransaction, LaunchRef, MirroredDtype,
    MirroredStorageLayout, OperationRef, PublicationOrdinal, TransactionCommitBoundary,
    TransactionDecision, TransactionFailure, TransactionId, TransactionOperation, TransactionState,
    TransferDirectionMirror, TransferOperationMirror, TransferRef, TransportPathMirror,
};
use crate::fake_device::{FaultClass, FaultInjectingBackend, FaultInjection, REPLANNING_SURFACE};
use crate::partition::{
    AdmissionRequest, FixtureIdentityClass, HardwareIsolationClaim, PartitionBudgetLedger,
    SafePhysicalLimit, TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use std::collections::{BTreeMap, BTreeSet};

/// The real independent device-loss row is **NOT ATTEMPTED**: the acceptance
/// host is single-device (lane queue §5b; md1-closeout §2), so a physical
/// second device dropping mid-transaction cannot be measured. The suite runs
/// the virtual mechanics row with honest fixture labels — the absent real row
/// must never be mistaken for a pass.
const REAL_INDEPENDENT_DEVICE_LOSS_ROW: &str = "NOT ATTEMPTED";

// T1 measured facts (pharos) reused for the synthetic snapshot shape.
const UUID_A: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const UUID_B: &str = "GPU-22222222-3333-4444-5555-666666666666";
const PROBE_TIME: u64 = 1_752_717_600_000_000_000; // fixed sample time
const LOGICAL_HASH: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

// CS-1 declared placement (md0-mode-fixtures.md §3): 2 virtual partitions @
// 160 MiB, forced 2-way split ≈129 MiB/device.
const CS1_SPLIT_BYTES: u64 = 135_266_304; // ≈129 MiB per partition
const CS1_LIMIT_BYTES: u64 = 167_772_160; // 160 MiB safe physical limit policy

// The synthetic two-partition transaction (T2 §5 fixture).
const LAUNCH_A_OUTPUT_BYTES: u64 = 4096;
const TRANSFER_BYTES: u64 = 8192;
const BROADCAST_BYTES: u64 = 2048;
const LAUNCH_B_OUTPUT_BYTES: u64 = 16_384;

fn device_a() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_A, None)
}

fn device_b() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_B, None)
}

fn partition_id(n: u32) -> LogicalPartitionId {
    LogicalPartitionId::new(format!("partition-{n}"))
}

/// A CS-1 weight ledger with the declared class 6/class 3 headroom.
fn ledger(class_six_bytes: u64, class_three_bytes: u64) -> PartitionBudgetLedger {
    PartitionBudgetLedger {
        weight_bytes: CS1_SPLIT_BYTES,
        kv_cache_bytes: 0,
        activation_scratch_bytes: class_three_bytes,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: class_six_bytes,
        concurrent_state_bytes: 0,
    }
}

/// An admitted virtual partition over `device` under the declared ledger.
fn vp(
    seed: u64,
    device: PhysicalDeviceId,
    ledger: PartitionBudgetLedger,
) -> VirtualDevicePartition {
    VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(seed), device, ledger),
        SafePhysicalLimit::new(CS1_LIMIT_BYTES),
    )
    .unwrap()
}

fn synthetic_entry(
    ordinal: u32,
    device: PhysicalDeviceId,
    generation: DeviceHealthGeneration,
) -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(ordinal),
        identity: device,
        device_model: Some("synthetic RTX 5070".to_owned()),
        capabilities: DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: DtypeSurface {
                f32: true,
                f64: true,
                f16: true,
                bf16: true,
                i8: true,
                i32: true,
            },
        },
        memory: DeviceMemory {
            tool_report_total_mib: Some(12_227),
            api_total_bytes: 12_343_705_600,
        },
        health: DeviceHealth::Healthy,
        health_generation: generation,
        probe_provenance: ProbeProvenance {
            probe: "synthetic two-partition fixture".to_owned(),
            tool_versions: "synthetic".to_owned(),
        },
    }
}

fn snapshot_with(
    entries: impl IntoIterator<Item = (u32, PhysicalDeviceId, DeviceHealthGeneration)>,
) -> DeviceDiscoverySnapshot {
    let devices: BTreeMap<_, _> = entries
        .into_iter()
        .map(|(ordinal, device, generation)| {
            (
                DeviceOrdinal::new(ordinal),
                synthetic_entry(ordinal, device, generation),
            )
        })
        .collect();
    DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted)
}

fn two_device_snapshot() -> DeviceDiscoverySnapshot {
    snapshot_with([
        (0, device_a(), DeviceHealthGeneration::initial()),
        (1, device_b(), DeviceHealthGeneration::initial()),
    ])
}

fn admitted_two_partition_plan() -> AdmittedLogicalPlan {
    AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0), partition_id(1)],
        [DeclaredPlacementConstraint::DistinctPhysicalDevices],
    )
    .expect("valid admitted two-partition plan")
}

/// The T2 §5 virtual-fixture bound plan: p0 → device A, p1 → device B, each
/// with an admitted virtual partition (p0: 10240 / 8448; p1: 10240 / 30976).
/// The plan receipt is labeled `FixtureIdentityClass::Synthetic`,
/// `TransportClass::HostStaged`, `HardwareIsolationClaim::NotClaimed` — the
/// honest virtual fixture labels.
fn fixture_plan() -> BoundDistributedPlan {
    let bindings = BTreeMap::from([
        (
            partition_id(0),
            PartitionBinding::with_virtual_partition(
                device_a(),
                vp(1, device_a(), ledger(10_240, 8_448)),
            ),
        ),
        (
            partition_id(1),
            PartitionBinding::with_virtual_partition(
                device_b(),
                vp(2, device_b(), ledger(10_240, 30_976)),
            ),
        ),
    ]);
    let snapshot = two_device_snapshot();
    bind(
        &admitted_two_partition_plan(),
        bindings,
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
        DeviceHealthGeneration::initial(),
        FixtureIdentityClass::Synthetic,
        TransportClass::HostStaged,
    )
    .expect("fixture plan binds")
}

fn transfer_op(id: &str, byte_count: u64) -> TransactionOperation {
    TransactionOperation::transfer(TransferOperationMirror::new(
        TransferRef::new(id),
        partition_id(0),
        partition_id(1),
        byte_count,
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        TransportPathMirror::HostStaged,
        0,
        1,
        TransactionCommitBoundary::default(),
    ))
}

/// The synthetic two-partition transaction operations in plan order.
fn fixture_operations() -> Vec<TransactionOperation> {
    vec![
        TransactionOperation::launch(
            partition_id(0),
            LaunchRef::new("launch-proj-a"),
            LAUNCH_A_OUTPUT_BYTES,
        ),
        transfer_op("t1", TRANSFER_BYTES),
        TransactionOperation::broadcast(CollectiveBroadcastMirror::broadcast(
            CollectiveRef::new("c1"),
            partition_id(0),
            BTreeSet::from([partition_id(0), partition_id(1)]),
            BROADCAST_BYTES,
        )),
        TransactionOperation::barrier(
            BarrierRef::new("barrier-main"),
            BTreeSet::from([partition_id(0), partition_id(1)]),
        ),
        TransactionOperation::launch(
            partition_id(1),
            LaunchRef::new("launch-proj-b"),
            LAUNCH_B_OUTPUT_BYTES,
        ),
    ]
}

fn fixture_boundary() -> TransactionCommitBoundary {
    TransactionCommitBoundary::new(
        [BarrierRef::new("barrier-main")],
        [LaunchRef::new("launch-proj-b")],
    )
}

fn fixture_transaction() -> ExecutionTransaction {
    ExecutionTransaction::new(
        TransactionId::new("txn-f1"),
        fixture_plan(),
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("fixture transaction constructs")
}

/// The declared write-set byte total over the fixture operations.
fn declared_write_bytes(operations: &[TransactionOperation]) -> u64 {
    operations
        .iter()
        .flat_map(|operation| operation.staged_writes())
        .map(|write| write.byte_count())
        .sum()
}

/// The transfer's stable snapshot key (`OperationRef::Transfer(t1)`).
fn transfer_key() -> OperationRef {
    OperationRef::Transfer(TransferRef::new("t1"))
}

/// The final launch's stable snapshot key
/// (`OperationRef::Launch(launch-proj-b)`).
fn launch_b_key() -> OperationRef {
    OperationRef::Launch(LaunchRef::new("launch-proj-b"))
}

// --- fault-injection delivery ----------------------------------------------

/// A fault on the t1 transfer of partition 1 (the mesh's second partition).
fn transfer_fault(partition: u32, class: FaultClass) -> FaultInjection {
    FaultInjection::new(transfer_key(), partition_id(partition), class)
}

/// Faults are one-shot, per-partition, and fire on the first matching
/// dispatch; the delivered record carries the class and the reclaimed
/// mid-copy bytes.
#[test]
fn fault_injection_delivers_on_first_matching_dispatch() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(transfer_fault(1, FaultClass::Timeout));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the t1 fault fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::Timeout { .. })
    ));

    let delivered = backend.delivered_faults();
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].operation, transfer_key());
    assert_eq!(delivered[0].partition, partition_id(1));
    assert_eq!(delivered[0].class, FaultClass::Timeout);
    assert_eq!(
        delivered[0].reclaimed_copy_bytes, TRANSFER_BYTES,
        "a mid-copy timeout reclaims the in-flight copy bytes"
    );
}

/// Per-partition targeting: a fault attributed to partition 1 fires on the
/// transfer (which involves both partitions) with partition 1 named in the
/// error.
#[test]
fn fault_injection_is_per_partition() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(transfer_fault(1, FaultClass::DeviceLoss));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the p1 device-loss fault fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::DeviceLoss { ref partition, .. })
            if *partition == partition_id(1)
    ));
}

// --- exit-gate bullet 4 / MD-A13: every fault class -------------------------

/// MD-A13 per-class: cancel, timeout, transfer error, kernel error, and
/// device loss each release or retire all affected resources with no partial
/// publication.
#[test]
fn every_fault_class_releases_or_retires_resources_without_partial_commit() {
    let classes = [
        FaultClass::Cancel,
        FaultClass::Timeout,
        FaultClass::TransferError,
        FaultClass::KernelError,
        FaultClass::DeviceLoss,
    ];
    for class in classes {
        let mut transaction = fixture_transaction();
        let mut backend = FaultInjectingBackend::new();
        // Cancel on launch-b (partition 1); the rest on the t1 transfer.
        let fault = if class == FaultClass::Cancel {
            FaultInjection::new(launch_b_key(), partition_id(1), class)
        } else {
            transfer_fault(1, class)
        };
        backend.inject(fault);
        transaction.prepare(&mut backend).expect("prepare succeeds");
        let execute_error = transaction
            .execute(&mut backend)
            .expect_err("the injected fault fails execute");
        assert!(
            matches!(execute_error, ExecuteError::Backend(_)),
            "{class:?} must surface as a backend error"
        );
        assert!(
            matches!(transaction.state(), TransactionState::Failed(_)),
            "{class:?} must move the machine to Failed"
        );

        // No partial publication is possible from the Failed state.
        assert!(
            matches!(
                transaction.commit(&mut backend, PublicationOrdinal::new(1)),
                Err(CommitError::InvalidState {
                    state: TransactionState::Failed(_)
                })
            ),
            "{class:?} must fail closed at the declared boundary"
        );
        assert_eq!(backend.published_bytes(), 0);

        let receipt = transaction
            .abort(&mut backend, "injected fault")
            .expect("abort completes teardown");
        assert!(
            matches!(transaction.state(), TransactionState::Aborted(_)),
            "{class:?} must end Aborted"
        );
        // All affected resources released or retired.
        assert!(
            backend.inner().reservations().is_empty(),
            "{class:?} must leave no reservation held"
        );
        assert_eq!(
            backend.staged_bytes(),
            0,
            "{class:?} must leave no staged bytes behind"
        );
        assert_eq!(
            backend.published_bytes(),
            0,
            "{class:?} must publish nothing"
        );
        assert!(
            receipt.publish_summary.is_none(),
            "{class:?} must record no publication"
        );
        assert!(
            !receipt.teardown.partial_publication,
            "{class:?} must never partially publish"
        );
        assert!(
            matches!(receipt.decision, TransactionDecision::Aborted { .. }),
            "{class:?} must record an abort decision"
        );
    }
}

/// Cancel before commit: the whole plan executes and stages, then the
/// coordinator-level cancel arrives before the boundary publication. Every
/// staged write is retired, every reservation released, nothing publishes.
#[test]
fn cancel_before_commit_retires_all_staged_state_and_publishes_nothing() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect("execute stages the whole plan");

    let receipt = transaction
        .abort(&mut backend, "user cancelled before commit")
        .expect("abort before commit succeeds");
    assert!(matches!(
        transaction.state(),
        TransactionState::Aborted(TransactionFailure::Cancelled { .. })
    ));

    // Both partitions staged writes, so both are retired — nothing released
    // as reservation-only, nothing published.
    assert_eq!(
        receipt.teardown.retired_partitions,
        BTreeSet::from([partition_id(0), partition_id(1)])
    );
    assert!(receipt.teardown.released_partitions.is_empty());
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.published_bytes(), 0);
    assert_eq!(backend.staged_bytes(), 0);
    assert!(backend.inner().reservations().is_empty());
}

/// A timeout mid-copy reclaims the in-flight copy and the coordinator retires
/// the partial staged state — the write-set never publishes partially.
#[test]
fn timeout_mid_copy_retires_partial_state_with_no_partial_commit() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(transfer_fault(1, FaultClass::Timeout));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the mid-copy timeout fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::Timeout { .. })
    ));

    // The mid-copy model: the in-flight copy bytes were staged and reclaimed
    // at fault delivery (a real runtime frees failed-copy staging), and never
    // reached the published write-set.
    let delivered = &backend.delivered_faults()[0];
    assert_eq!(delivered.class, FaultClass::Timeout);
    assert_eq!(delivered.reclaimed_copy_bytes, TRANSFER_BYTES);

    let receipt = transaction
        .abort(&mut backend, "timeout")
        .expect("abort completes teardown");
    // p0 held the launch-a staged write (retired); p1 held only the
    // reservation (released) — the faulted copy never staged in the
    // coordinator's write-set.
    assert_eq!(
        receipt.teardown.retired_partitions,
        BTreeSet::from([partition_id(0)])
    );
    assert_eq!(
        receipt.teardown.released_partitions,
        BTreeSet::from([partition_id(1)])
    );
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.staged_bytes(), 0);
    assert_eq!(backend.published_bytes(), 0);
    assert!(backend.inner().reservations().is_empty());
}

/// A transfer error fails the transfer operation; everything already staged
/// is retired, the reservation-only partition is released, nothing publishes.
#[test]
fn transfer_error_fails_closed_with_full_teardown() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(transfer_fault(0, FaultClass::TransferError));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the transfer error fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::Operation { ref detail, .. })
            if detail.contains("transfer-error") && detail.contains("t1")
    ));

    let receipt = transaction
        .abort(&mut backend, "transfer error")
        .expect("abort completes teardown");
    assert_eq!(
        receipt.teardown.retired_partitions,
        BTreeSet::from([partition_id(0)])
    );
    assert_eq!(
        receipt.teardown.released_partitions,
        BTreeSet::from([partition_id(1)])
    );
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.published_bytes(), 0);
    assert!(backend.inner().reservations().is_empty());
}

/// A kernel error fails a launch operation; the partially-staged write-set is
/// fully retired and never published.
#[test]
fn kernel_error_fails_closed_with_full_teardown() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(FaultInjection::new(
        launch_b_key(),
        partition_id(1),
        FaultClass::KernelError,
    ));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the kernel error fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::Operation { ref detail, .. })
            if detail.contains("kernel-error") && detail.contains("launch-proj-b")
    ));

    let receipt = transaction
        .abort(&mut backend, "kernel error")
        .expect("abort completes teardown");
    // The launch failed at the last operation: p0 and p1 both hold staged
    // writes, so both are retired.
    assert_eq!(
        receipt.teardown.retired_partitions,
        BTreeSet::from([partition_id(0), partition_id(1)])
    );
    assert!(receipt.teardown.released_partitions.is_empty());
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.published_bytes(), 0);
    assert_eq!(backend.staged_bytes(), 0);
    assert!(backend.inner().reservations().is_empty());
}

/// Device loss mid-transaction: the affected device's staged state is
/// retired, everything else released, nothing publishes. The virtual fixture
/// labels stay honest (`Synthetic`, `hardware_isolation_claimed=false`).
#[test]
fn device_loss_mid_transaction_fails_closed_with_honest_labels() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(transfer_fault(1, FaultClass::DeviceLoss));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the device-loss fault fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::DeviceLoss { ref partition, .. })
            if *partition == partition_id(1)
    ));

    let receipt = transaction
        .abort(&mut backend, "device loss")
        .expect("abort completes teardown");
    assert_eq!(
        receipt.teardown.retired_partitions,
        BTreeSet::from([partition_id(0)])
    );
    assert_eq!(
        receipt.teardown.released_partitions,
        BTreeSet::from([partition_id(1)])
    );
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.published_bytes(), 0);

    // Honest virtual fixture labels: never a real-device pass.
    assert_eq!(
        receipt.plan_receipt.fixture_identity_class(),
        FixtureIdentityClass::Synthetic
    );
    assert_eq!(
        receipt.plan_receipt.transport_class(),
        TransportClass::HostStaged
    );
    assert_eq!(
        receipt.plan_receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );
    assert_eq!(REAL_INDEPENDENT_DEVICE_LOSS_ROW, "NOT ATTEMPTED");
}

// --- exit-gate bullet 5: degraded mesh --------------------------------------

/// A degraded mesh (one partition's backend fails mid-execution) fails closed
/// deterministically at the declared boundary: the `Failed` state rejects
/// commit, nothing publishes, and the receipt surfaces the degraded state.
#[test]
fn degraded_mesh_fails_closed_at_declared_boundary_and_surfaces_degraded_state() {
    // The degraded mesh: partition 1's backend fails mid-execution on the t1
    // transfer (operation 2 of 5). Run the scenario three times to prove the
    // fail-closed outcome is deterministic.
    for round in 0..3 {
        let mut transaction = fixture_transaction();
        let mut backend = FaultInjectingBackend::new();
        backend.inject(transfer_fault(1, FaultClass::DeviceLoss));
        transaction.prepare(&mut backend).expect("prepare succeeds");
        let error = transaction
            .execute(&mut backend)
            .expect_err("the degraded partition's backend fails mid-execution");
        assert!(
            matches!(
                error,
                ExecuteError::Backend(BackendError::DeviceLoss { .. })
            ),
            "round {round}: the failure surfaces as device loss"
        );
        assert!(
            matches!(transaction.state(), TransactionState::Failed(_)),
            "round {round}: the machine is Failed after the fault"
        );

        // Fail closed: commit is rejected — the boundary can never be reached
        // through a Failed machine, so the staged write-set can never publish
        // (fully or partially).
        assert!(
            matches!(
                transaction.commit(&mut backend, PublicationOrdinal::new(1)),
                Err(CommitError::InvalidState {
                    state: TransactionState::Failed(_)
                })
            ),
            "round {round}: commit fails closed from the Failed state"
        );
        assert_eq!(
            backend.published_bytes(),
            0,
            "round {round}: nothing is published"
        );

        let receipt = transaction
            .abort(&mut backend, "degraded mesh")
            .expect("abort completes teardown");
        assert!(
            matches!(transaction.state(), TransactionState::Aborted(_)),
            "round {round}: the machine is Aborted after teardown"
        );

        // The degraded state is surfaced in the receipt: the abort decision
        // names the originating device-loss on partition 1.
        assert!(
            matches!(
                receipt.decision,
                TransactionDecision::Aborted {
                    failure:
                        TransactionFailure::Backend(BackendError::DeviceLoss {
                            ref partition,
                            ..
                        })
                } if *partition == partition_id(1)
            ),
            "round {round}: the receipt surfaces the degraded state"
        );

        // Teardown: p0's staged launch-a retired; p1 released reservation-only.
        assert_eq!(
            receipt.teardown.retired_partitions,
            BTreeSet::from([partition_id(0)]),
            "round {round}: the degraded partition's staged state is retired"
        );
        assert_eq!(
            receipt.teardown.released_partitions,
            BTreeSet::from([partition_id(1)]),
            "round {round}: the reservation-only partition is released"
        );
        assert!(
            !receipt.teardown.partial_publication,
            "round {round}: no partial publication"
        );
        assert!(backend.inner().reservations().is_empty());
    }
}

/// MD3 never invents a replanner: replanning a degraded mesh is MD5's
/// surface, recorded explicitly. The mesh fixture's labels are honest.
#[test]
fn degraded_mesh_records_replanning_as_md5_surface() {
    assert_eq!(
        REPLANNING_SURFACE, "MD5",
        "replanning is MD5's surface — MD3 fails closed and never invents a replanner"
    );

    // A degraded mesh abort completes without any replanning step: the only
    // terminal outcomes are commit or abort (retry is disabled, and there is
    // no replanning path in the transaction lifecycle).
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    backend.inject(transfer_fault(1, FaultClass::DeviceLoss));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect_err("the degraded mesh fails mid-execution");
    let receipt = transaction
        .abort(&mut backend, "degraded mesh — replanning is MD5's surface")
        .expect("abort completes teardown");
    assert!(matches!(
        receipt.decision,
        TransactionDecision::Aborted { .. }
    ));
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
}

// --- the last committed state remains authoritative -------------------------

/// MD-A13: a failed transaction never changes the last committed state. A
/// first transaction commits; a second transaction over the same plan faults
/// and aborts; the backend's committed write-set is still the first
/// transaction's.
#[test]
fn last_committed_state_remains_authoritative_after_failure() {
    let declared_total = declared_write_bytes(&fixture_operations());

    // Transaction 1 commits atomically.
    let mut first = ExecutionTransaction::new(
        TransactionId::new("txn-committed"),
        fixture_plan(),
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("first transaction constructs");
    let mut backend = FaultInjectingBackend::new();
    first.prepare(&mut backend).expect("first prepare succeeds");
    first.execute(&mut backend).expect("first execute succeeds");
    let commit_receipt = first
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("first commit publishes");
    assert!(matches!(
        commit_receipt.decision,
        TransactionDecision::Committed { .. }
    ));
    assert_eq!(backend.committed_bytes(), declared_total);
    assert_eq!(backend.committed_writes().len(), 4);

    // Transaction 2 faults mid-execution and aborts. The backend began a
    // fresh transaction at prepare; nothing is published.
    let mut second = ExecutionTransaction::new(
        TransactionId::new("txn-failing"),
        fixture_plan(),
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("second transaction constructs");
    backend.inject(transfer_fault(1, FaultClass::TransferError));
    second
        .prepare(&mut backend)
        .expect("second prepare succeeds");
    second
        .execute(&mut backend)
        .expect_err("the injected transfer error fails the second transaction");
    let abort_receipt = second
        .abort(&mut backend, "injected fault")
        .expect("abort completes teardown");
    assert!(matches!(
        abort_receipt.decision,
        TransactionDecision::Aborted { .. }
    ));

    // The last committed state is unchanged and authoritative.
    assert_eq!(backend.published_bytes(), 0);
    assert_eq!(backend.committed_bytes(), declared_total);
    assert_eq!(backend.committed_writes().len(), 4);
    assert_eq!(
        backend
            .committed_writes()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        first
            .declared_write_set()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        "the committed write-set is exactly the first transaction's declared write-set"
    );
}

/// The happy-path fake still drives the fixture: the fault-injecting wrapper
/// adds no behavior when no fault is configured.
#[test]
fn no_fault_configured_behaves_like_the_happy_path_fake() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect("execute succeeds without configured faults");
    assert!(backend.delivered_faults().is_empty());
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("commit publishes on the happy path");
    assert!(matches!(
        receipt.decision,
        TransactionDecision::Committed { .. }
    ));
    assert_eq!(
        backend.committed_bytes(),
        declared_write_bytes(&fixture_operations())
    );
}

/// The committed-state snapshot survives a backend reset between
/// transactions, and a fresh transaction stages from zero.
#[test]
fn committed_snapshot_persists_across_transactions() {
    let declared_total = declared_write_bytes(&fixture_operations());

    // Commit a first transaction, then drive a second failing transaction on
    // the same backend; the committed snapshot must persist.
    let mut first = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    first.prepare(&mut backend).expect("prepare succeeds");
    first.execute(&mut backend).expect("execute succeeds");
    first
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("commit publishes");
    assert_eq!(backend.committed_bytes(), declared_total);

    let mut second = fixture_transaction();
    backend.inject(FaultInjection::new(
        launch_b_key(),
        partition_id(1),
        FaultClass::KernelError,
    ));
    second.prepare(&mut backend).expect("prepare succeeds");
    second
        .execute(&mut backend)
        .expect_err("the kernel fault fails the second transaction");
    second
        .abort(&mut backend, "fault")
        .expect("abort completes teardown");

    assert_eq!(
        backend.committed_bytes(),
        declared_total,
        "the last committed state remains authoritative"
    );
    assert_eq!(backend.staged_bytes(), 0);
    assert_eq!(backend.published_bytes(), 0);
}

// A defensive sanity check on the fixture itself: the fixture's reservations
// fit the admitted budgets (the prepare path is the authority).
#[test]
fn fixture_prepare_reserves_within_admitted_budgets() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectingBackend::new();
    transaction
        .prepare(&mut backend)
        .expect("the fixture reservation fits the budgets");
    assert_eq!(
        transaction
            .reservation()
            .expect("reservation recorded")
            .len(),
        2
    );
}
