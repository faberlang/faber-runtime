//! MD3-F1: fault-injecting `DeviceExecutionBackend` (gpu-inference-multi-device).
//!
//! [`FaultInjectingBackend`] wraps the X1 happy-path fake
//! ([`FakeExecutionBackend`]) and adds **per-partition, per-operation fault
//! injection** in the MD3-F1 fault classes — cancel, timeout, transfer error,
//! kernel error, and device loss. It is the MD3-F1 evidence owner for
//! CAMPAIGN §MD3 exit-gate bullets 4 and 5:
//!
//! - **Bullet 4 / MD-A13** — every fault class releases or retires **all
//!   affected resources with no partial commit**: the staged write-set is
//!   never partially published and the last committed state remains
//!   authoritative (a failed transaction never changes what was published).
//! - **Bullet 5** — a degraded mesh (one partition's backend fails
//!   mid-execution) **fails closed deterministically at the declared
//!   [`TransactionCommitBoundary`](crate::execution_transaction::TransactionCommitBoundary)**:
//!   the coordinator's `Failed` state rejects `commit`, nothing publishes,
//!   `abort` completes teardown, and the degraded state is surfaced in the
//!   `TransactionReceipt`. **Replanning is MD5's surface** —
//!   [`REPLANNING_SURFACE`]; MD3 never invents a replanner.
//!
//! ## Honest fixture labels
//!
//! The fault suite runs on the T2 §5 **virtual** fixture (2 virtual
//! partitions, 1 directed host-staged link) with `fixture_identity_class =
//! synthetic` and `hardware_isolation_claimed = false`. Real independent
//! device-loss (a physical second device dropping mid-transaction) stays
//! **NOT ATTEMPTED** on the single-device acceptance host (lane queue §5b);
//! the virtual mechanics row never masquerades as a real-device pass.

use crate::bound_plan::LogicalPartitionId;
use crate::execution_transaction::{
    BackendError, DeviceExecutionBackend, FakeExecutionBackend, OperationEvent, OperationRef,
    OutputRef, ReservationRecord, StagedWrite, TransactionFailure, TransactionOperation,
};
use std::collections::BTreeMap;

/// Replanning a degraded mesh is **MD5's surface** (CAMPAIGN §MD3 bullet 5;
/// MD3-Q5). MD3 fails closed at the declared boundary and never invents a
/// replanner — this constant is the recorded boundary marker.
pub const REPLANNING_SURFACE: &str = "MD5";

/// The MD3-F1 fault classes (CAMPAIGN §MD3 bullet 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultClass {
    /// The operation was cancelled before completion (cancel before commit).
    Cancel,
    /// The operation timed out — a mid-copy timeout surfaces the in-flight
    /// copy being reclaimed and reported as `BackendError::Timeout`.
    Timeout,
    /// A transfer failed (surfaced as `BackendError::Operation`).
    TransferError,
    /// A kernel/launch failed (surfaced as `BackendError::Operation`).
    KernelError,
    /// The bound physical device failed or was removed (MD-A13, surfaced as
    /// `BackendError::DeviceLoss`).
    DeviceLoss,
}

impl FaultClass {
    /// Short diagnostic spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Cancel => "cancel",
            Self::Timeout => "timeout",
            Self::TransferError => "transfer-error",
            Self::KernelError => "kernel-error",
            Self::DeviceLoss => "device-loss",
        }
    }
}

/// One configured fault: fault the named operation on the named partition
/// with the named class. The partition must be one of the operation's
/// involved partitions (validated at delivery; a mismatch simply never
/// fires).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultInjection {
    operation: OperationRef,
    partition: LogicalPartitionId,
    class: FaultClass,
}

impl FaultInjection {
    /// Configure a fault on one operation of one partition.
    #[must_use]
    pub const fn new(
        operation: OperationRef,
        partition: LogicalPartitionId,
        class: FaultClass,
    ) -> Self {
        Self {
            operation,
            partition,
            class,
        }
    }

    /// The operation the fault targets.
    #[must_use]
    pub fn operation(&self) -> &OperationRef {
        &self.operation
    }

    /// The partition the fault is attributed to.
    #[must_use]
    pub fn partition(&self) -> &LogicalPartitionId {
        &self.partition
    }

    /// The fault class.
    #[must_use]
    pub const fn class(&self) -> FaultClass {
        self.class
    }

    /// The `BackendError` this fault surfaces when delivered.
    #[must_use]
    pub fn to_backend_error(&self) -> BackendError {
        let id = match &self.operation {
            OperationRef::Launch(reference) => reference.as_str(),
            OperationRef::Transfer(reference) => reference.as_str(),
            OperationRef::Collective(reference) => reference.as_str(),
            OperationRef::Barrier(reference) => reference.as_str(),
        };
        let detail = format!("injected {} on {id}", self.class.spelling());
        match self.class {
            FaultClass::Cancel => BackendError::cancelled(self.partition.clone(), detail),
            FaultClass::Timeout => BackendError::timeout(self.partition.clone(), detail),
            FaultClass::TransferError | FaultClass::KernelError => {
                BackendError::operation(self.partition.clone(), detail)
            }
            FaultClass::DeviceLoss => BackendError::device_loss(self.partition.clone(), detail),
        }
    }

    /// Bytes of the in-flight copy a mid-copy fault reclaims at delivery.
    ///
    /// A real runtime frees the staging of a failed/timed-out copy at
    /// failure; the fault suite models that reclaim honestly (the bytes never
    /// reach the published write-set) and records them for the test.
    #[must_use]
    pub fn reclaimed_copy_bytes(&self, operation: &TransactionOperation) -> u64 {
        match (self.class, operation) {
            (
                FaultClass::Timeout | FaultClass::TransferError,
                TransactionOperation::Transfer(transfer),
            ) => transfer.byte_count(),
            _ => 0,
        }
    }
}

/// A delivered fault — the evidence record the fault suite asserts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveredFault {
    /// The operation the fault fired on.
    pub operation: OperationRef,
    /// The partition the fault was attributed to.
    pub partition: LogicalPartitionId,
    /// The fault class.
    pub class: FaultClass,
    /// The backend error surfaced to the coordinator.
    pub error: BackendError,
    /// Bytes of the in-flight copy staged and reclaimed at fault delivery
    /// (mid-copy classes on a transfer; 0 otherwise).
    pub reclaimed_copy_bytes: u64,
}

/// A fault-injecting `DeviceExecutionBackend` wrapping the X1 happy-path fake
/// ([`FakeExecutionBackend`]).
///
/// Faults are **per-partition and per-operation**: a [`FaultInjection`]
/// names the operation identity and the affected partition, and fires on the
/// first matching dispatch (one-shot; the coordinator has no retry path).
/// The committed snapshot (the write-set of the last successful atomic
/// publication) is preserved across transactions — the last committed state
/// stays authoritative; a failed transaction never changes it (MD-A13).
///
/// A fresh transaction is detected at `reserve` when no reservation is held:
/// the inner fake's execution state is reset while the committed snapshot is
/// kept (a real runtime starts a new transaction with fresh staging).
#[derive(Debug, Clone)]
pub struct FaultInjectingBackend {
    inner: FakeExecutionBackend,
    faults: Vec<FaultInjection>,
    delivered: Vec<DeliveredFault>,
    committed_writes: BTreeMap<OutputRef, StagedWrite>,
    committed_bytes: u64,
}

impl FaultInjectingBackend {
    /// A fresh fault-injecting backend with no configured faults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: FakeExecutionBackend::new(),
            faults: Vec::new(),
            delivered: Vec::new(),
            committed_writes: BTreeMap::new(),
            committed_bytes: 0,
        }
    }

    /// Configure one fault injection.
    pub fn inject(&mut self, fault: FaultInjection) {
        self.faults.push(fault);
    }

    /// The faults delivered so far, in delivery order.
    #[must_use]
    pub fn delivered_faults(&self) -> &[DeliveredFault] {
        &self.delivered
    }

    /// The write-set of the last successful atomic publication — the last
    /// committed state. Never changed by a failed transaction (MD-A13).
    #[must_use]
    pub fn committed_writes(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.committed_writes
    }

    /// The byte total of the last committed state.
    #[must_use]
    pub const fn committed_bytes(&self) -> u64 {
        self.committed_bytes
    }

    /// The inner happy-path fake (observability: reservations, staged state,
    /// events).
    #[must_use]
    pub fn inner(&self) -> &FakeExecutionBackend {
        &self.inner
    }

    /// Begin a fresh transaction when none is in flight: the previous
    /// transaction released or retired every reservation, so the inner
    /// execution state resets while the committed snapshot persists.
    fn begin_fresh_transaction_if_needed(&mut self) {
        if self.inner.reservations().is_empty() {
            self.inner = FakeExecutionBackend::new();
        }
    }
}

impl Default for FaultInjectingBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceExecutionBackend for FaultInjectingBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        self.begin_fresh_transaction_if_needed();
        self.inner.reserve(partition, reservation)
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        let key = operation.operation_ref();
        let matched = self.faults.iter().position(|fault| {
            fault.operation == key && operation.partitions().contains(&fault.partition)
        });
        if let Some(index) = matched {
            let fault = self.faults.remove(index);
            let error = fault.to_backend_error();
            self.delivered.push(DeliveredFault {
                operation: fault.operation.clone(),
                partition: fault.partition.clone(),
                class: fault.class,
                error: error.clone(),
                reclaimed_copy_bytes: fault.reclaimed_copy_bytes(operation),
            });
            return Err(error);
        }
        self.inner.run_operation(operation)
    }

    fn event_completed(&self, event: &OperationEvent) -> bool {
        self.inner.event_completed(event)
    }

    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError> {
        self.inner.stage_write(write)
    }

    fn publish(&mut self) -> Result<(), BackendError> {
        let result = self.inner.publish();
        if result.is_ok() {
            self.committed_writes = self.inner.published_writes().clone();
            self.committed_bytes = self.inner.published_bytes();
        }
        result
    }

    fn staged_bytes(&self) -> u64 {
        self.inner.staged_bytes()
    }

    fn published_bytes(&self) -> u64 {
        self.inner.published_bytes()
    }

    fn release(&mut self, partition: &LogicalPartitionId) {
        self.inner.release(partition);
    }

    fn retire(&mut self, partition: &LogicalPartitionId, failure: &TransactionFailure) {
        self.inner.retire(partition, failure);
    }
}

#[cfg(test)]
#[path = "fault_test.rs"]
mod tests;
