//! Generic `ExecutionTransaction` state machine + staged write-set + atomic
//! publication (gpu-inference-multi-device, MD3-X1 — the serial gate).
//!
//! [`ExecutionTransaction`] coordinates one **abstract** execution of a
//! [`BoundDistributedPlan`] (MD2-B1) over a [`DeviceExecutionBackend`]:
//!
//! - [`ExecutionTransaction::prepare`] reserves typed transfer staging,
//!   output buffers, events, and transaction scratch against the **admitted**
//!   [`PartitionBudgetLedger`] class budgets (CTO `6badaa01` S3): staging
//!   against class 6 `transfer_staging_bytes`; outputs/events/scratch against
//!   class 3 `activation_scratch_bytes`. A reservation that exceeds the bound
//!   partition's admitted budget fails **before** execute, and the reservation
//!   is recorded (the prepare receipt).
//! - [`ExecutionTransaction::execute`] runs the accepted plan in order
//!   against the backend, reusing the reservation and **never silently
//!   growing the accepted plan** (the prepare snapshot *is* the accepted
//!   plan; an operation outside it fails — S3).
//! - [`ExecutionTransaction::commit`] publishes the staged write-set
//!   **atomically** only after every required device reaches the declared
//!   [`TransactionCommitBoundary`], releases the reservations, and records the
//!   **abstract publication ordinal** (a transaction-scoped publication
//!   counter — never the semantic `ValueGeneration`, naming contract §3) in
//!   the commit receipt.
//! - [`ExecutionTransaction::abort`] releases or retires every affected
//!   resource with **no partial publication** (MD-A13).
//!
//! ## Mirror vocabulary
//!
//! faber-runtime cannot import radix-mir (FC18), so the transaction consumes
//! the accepted plan as a **dependency-free mirror** — the
//! [`DeclaredPlacementConstraint`](crate::bound_plan::DeclaredPlacementConstraint)
//! pattern (FC8): [`TransactionOperation`] mirrors
//! `ExecutionOperation::{Launch,Transfer,Collective,Barrier}` and
//! [`TransactionCommitBoundary`] mirrors `ExecutionCommitBoundary` (barriers +
//! launches). Mirrors serialize to **stable canonical bytes** — the
//! `push_str`/`push_u64` discipline from `device_identity.rs`/`bound_plan.rs`
//! (faber-runtime has no serde, FC6).
//!
//! ## Generic (S5)
//!
//! The transaction introduces **no inference or training vocabulary** (the
//! inference binding is MD3I's surface, CTO `6badaa01` S5). It is not a
//! universal inference/training facade (MD-A16). **Retry is disabled**: the
//! state machine (`New → Prepared → Executing → Committed | Aborted`, with
//! `Failed → Aborted`) has no re-execution path.
//!
//! ## Constraint-tampering authority (MD2-C1 residual 2)
//!
//! `prepare` validates the accepted plan's declared placement against the
//! **bound device set/topology and the actual admitted partition budgets** —
//! the frozen `PartitionBudgetLedger` of the virtual partitions attached to
//! the bound plan's bindings. A tampered-but-internally-consistent declared
//! plan that over-commits the bound resources derives a reservation that
//! exceeds the admitted class budgets and fails at `prepare`, before any
//! execution.

use crate::bound_plan::{BoundDistributedPlan, LogicalPartitionId};
use crate::device_identity::{push_str, push_u64};
use crate::partition::{PartitionBudgetLedger, PartitionReceipt};
use crate::transport::TransportReceipt;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

// --- mirror vocabulary ------------------------------------------------------

/// Stable opaque reference to one semantic launch (mirror of radix-mir
/// `LaunchId`; faber-runtime cannot import radix-mir, FC18).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaunchRef(String);

impl LaunchRef {
    /// Build a launch reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for LaunchRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one transfer in the execution graph (mirror of
/// radix-mir `TransferId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransferRef(String);

impl TransferRef {
    /// Build a transfer reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransferRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one collective in the execution graph (mirror of
/// radix-mir `CollectiveId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CollectiveRef(String);

impl CollectiveRef {
    /// Build a collective reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CollectiveRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one barrier in the execution graph (mirror of
/// radix-mir `BarrierId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BarrierRef(String);

impl BarrierRef {
    /// Build a barrier reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BarrierRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one staged output buffer the transaction will
/// publish. A write is identified by the operation that produces it, so an
/// `OutputRef` is unique within a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputRef(String);

impl OutputRef {
    /// Build an output reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for OutputRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Direction of a host-staged transfer (T1 §2.2 measurement vocabulary).
///
/// `H2D` / `D2H` are the host-boundary half-moves; `BIDI` is the combined
/// device-to-device host-staged copy (concurrent H2D ∥ D2H — the way every
/// cross-partition move on the acceptance host traverses the admitted path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirectionMirror {
    /// Host → device (copy in to the destination partition).
    H2D,
    /// Device → host (copy out from the source partition).
    D2H,
    /// Both directions concurrently (the host-staged device-to-device move).
    BIDI,
}

impl TransferDirectionMirror {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::H2D => 0,
            Self::D2H => 1,
            Self::BIDI => 2,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::H2D => "h2d",
            Self::D2H => "d2h",
            Self::BIDI => "bidi",
        }
    }
}

/// Mirror of the element type of a transferred value — a **stable canonical
/// form** (FC6/FC18). The mapping from the logical `MirType` (radix-mir,
/// opaque) into this closed set is the translator's obligation at MD3-S1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirroredDtype {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 16-bit float.
    F16,
    /// bfloat16.
    BF16,
    /// Signed 8-bit integer.
    I8,
    /// Signed 32-bit integer.
    I32,
}

impl MirroredDtype {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::F32 => 0,
            Self::F64 => 1,
            Self::F16 => 2,
            Self::BF16 => 3,
            Self::I8 => 4,
            Self::I32 => 5,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::I32 => "i32",
        }
    }
}

/// Mirror of the storage layout of a transferred value — a stable canonical
/// form (the logical layout vocabulary is radix-mir's; this is the
/// dependency-free mirror the typed/ranged transfer checks consume).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirroredStorageLayout {
    /// Dense row-major element storage.
    Dense,
    /// Block-packed storage (quantized block layout).
    BlockPacked,
}

impl MirroredStorageLayout {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Dense => 0,
            Self::BlockPacked => 1,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::BlockPacked => "block-packed",
        }
    }
}

/// Transport path label of a transfer (T2 §7 — silent host staging
/// forbidden).
///
/// `host-staged` is the only admitted path on the acceptance host. The label
/// is a transport-**admissibility** constraint on the logical plan; the
/// **selected** transport records to the transaction receipt, never to the
/// portable logical plan (S4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportPathMirror {
    /// Pinned host memory ↔ device over PCIe — the admitted path.
    HostStaged,
}

impl TransportPathMirror {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::HostStaged => 0,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::HostStaged => "host-staged",
        }
    }
}

/// Mirror of `ExecutionCommitBoundary` (naming contract §3) — the declared
/// barrier/launch completion set that commits a value or a plan. **Never
/// called "execution generation"** (`ValueGeneration` is taken). [`commit`](
/// ExecutionTransaction::commit) publishes the staged write-set only after
/// every boundary barrier/launch completed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransactionCommitBoundary {
    barriers: BTreeSet<BarrierRef>,
    launches: BTreeSet<LaunchRef>,
}

impl TransactionCommitBoundary {
    /// Build a boundary from its barrier and launch reference sets.
    #[must_use]
    pub fn new(
        barriers: impl IntoIterator<Item = BarrierRef>,
        launches: impl IntoIterator<Item = LaunchRef>,
    ) -> Self {
        Self {
            barriers: barriers.into_iter().collect(),
            launches: launches.into_iter().collect(),
        }
    }

    /// The declared barrier references, in stable order.
    #[must_use]
    pub fn barriers(&self) -> &BTreeSet<BarrierRef> {
        &self.barriers
    }

    /// The declared launch references, in stable order.
    #[must_use]
    pub fn launches(&self) -> &BTreeSet<LaunchRef> {
        &self.launches
    }

    /// True when the boundary declares nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty() && self.launches.is_empty()
    }

    /// Deterministic canonical bytes: barrier references then launch
    /// references, each set in stable order.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u64(&mut out, self.barriers.len() as u64);
        for barrier in &self.barriers {
            push_str(&mut out, barrier.as_str());
        }
        push_u64(&mut out, self.launches.len() as u64);
        for launch in &self.launches {
            push_str(&mut out, launch.as_str());
        }
        out
    }
}

/// Mirror of radix-mir `TransferOperation`: one typed/ranged host-staged
/// cross-partition move, identified by byte count + logical
/// dtype/layout/generation, with the declared path label and per-transfer
/// completion boundary. The typed/ranged *validation before copy* is the
/// transport adapter's obligation (MD3-T1); this mirror carries the declared
/// facts in stable canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferOperationMirror {
    id: TransferRef,
    source: LogicalPartitionId,
    destination: LogicalPartitionId,
    byte_count: u64,
    direction: TransferDirectionMirror,
    element_dtype: MirroredDtype,
    layout: MirroredStorageLayout,
    path_label: TransportPathMirror,
    producer_generation: u64,
    consumer_generation: u64,
    completion_boundary: TransactionCommitBoundary,
}

impl TransferOperationMirror {
    /// Build a transfer mirror. `producer_generation` / `consumer_generation`
    /// are the content versions the producer wrote and the consumer reads —
    /// transfer-operation facts, never the semantic `ValueGeneration`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        id: TransferRef,
        source: LogicalPartitionId,
        destination: LogicalPartitionId,
        byte_count: u64,
        direction: TransferDirectionMirror,
        element_dtype: MirroredDtype,
        layout: MirroredStorageLayout,
        path_label: TransportPathMirror,
        producer_generation: u64,
        consumer_generation: u64,
        completion_boundary: TransactionCommitBoundary,
    ) -> Self {
        Self {
            id,
            source,
            destination,
            byte_count,
            direction,
            element_dtype,
            layout,
            path_label,
            producer_generation,
            consumer_generation,
            completion_boundary,
        }
    }

    /// The stable transfer identity.
    #[must_use]
    pub fn id(&self) -> &TransferRef {
        &self.id
    }

    /// The source partition (the value's owner for a read move).
    #[must_use]
    pub fn source(&self) -> &LogicalPartitionId {
        &self.source
    }

    /// The destination partition.
    #[must_use]
    pub fn destination(&self) -> &LogicalPartitionId {
        &self.destination
    }

    /// Byte count of the transferred value.
    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    /// The declared direction.
    #[must_use]
    pub const fn direction(&self) -> TransferDirectionMirror {
        self.direction
    }

    /// The declared element type.
    #[must_use]
    pub const fn element_dtype(&self) -> MirroredDtype {
        self.element_dtype
    }

    /// The declared storage layout.
    #[must_use]
    pub const fn layout(&self) -> MirroredStorageLayout {
        self.layout
    }

    /// The declared transport path label (admissibility constraint, v1 =
    /// `{host-staged}`).
    #[must_use]
    pub const fn path_label(&self) -> TransportPathMirror {
        self.path_label
    }

    /// The producer content version.
    #[must_use]
    pub const fn producer_generation(&self) -> u64 {
        self.producer_generation
    }

    /// The consumer content version.
    #[must_use]
    pub const fn consumer_generation(&self) -> u64 {
        self.consumer_generation
    }

    /// The per-transfer completion boundary (mirror fact; publication gates
    /// on the plan-level boundary).
    #[must_use]
    pub fn completion_boundary(&self) -> &TransactionCommitBoundary {
        &self.completion_boundary
    }

    /// Deterministic canonical bytes of every declared field.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, self.id.as_str());
        push_str(&mut out, self.source.as_str());
        push_str(&mut out, self.destination.as_str());
        push_u64(&mut out, self.byte_count);
        push_u64(&mut out, self.direction.tag());
        push_u64(&mut out, self.element_dtype.tag());
        push_u64(&mut out, self.layout.tag());
        push_u64(&mut out, self.path_label.tag());
        push_u64(&mut out, self.producer_generation);
        push_u64(&mut out, self.consumer_generation);
        out.extend_from_slice(&self.completion_boundary.canonical_bytes());
        out
    }
}

/// Mirror of a `Collective::Broadcast` (the only admitted collective, v1).
/// Broadcast is composed from labeled host-staged transfers + local kernels —
/// no collective library (md0-transport §8). The mirror carries the source,
/// the participant set (source plus every consumer), and the value's byte
/// contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveBroadcastMirror {
    id: CollectiveRef,
    source: LogicalPartitionId,
    participants: BTreeSet<LogicalPartitionId>,
    byte_count: u64,
}

impl CollectiveBroadcastMirror {
    /// Build a broadcast mirror. The source must be a participant; every
    /// participant is a declared partition (validated at prepare).
    #[must_use]
    pub fn broadcast(
        id: CollectiveRef,
        source: LogicalPartitionId,
        participants: BTreeSet<LogicalPartitionId>,
        byte_count: u64,
    ) -> Self {
        debug_assert!(
            participants.contains(&source),
            "the broadcast source must be a participant"
        );
        Self {
            id,
            source,
            participants,
            byte_count,
        }
    }

    /// The stable collective identity.
    #[must_use]
    pub fn id(&self) -> &CollectiveRef {
        &self.id
    }

    /// The source partition (owner / primary replica).
    #[must_use]
    pub fn source(&self) -> &LogicalPartitionId {
        &self.source
    }

    /// The participant set (source plus all consumers), in stable order.
    #[must_use]
    pub fn participants(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.participants
    }

    /// Byte count of the broadcast value (per participant copy).
    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    /// Deterministic canonical bytes of every declared field.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, self.id.as_str());
        push_str(&mut out, self.source.as_str());
        push_u64(&mut out, self.participants.len() as u64);
        for participant in &self.participants {
            push_str(&mut out, participant.as_str());
        }
        push_u64(&mut out, self.byte_count);
        out
    }
}

/// Stable reference of one operation within the prepare snapshot — the key
/// the no-silent-growth check runs on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperationRef {
    /// A launch, by its launch reference.
    Launch(LaunchRef),
    /// A transfer, by its transfer identity.
    Transfer(TransferRef),
    /// A broadcast collective, by its collective identity.
    Collective(CollectiveRef),
    /// A barrier, by its barrier reference.
    Barrier(BarrierRef),
}

/// A `BarrierRef` or `LaunchRef` that a [`TransactionCommitBoundary`] names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BoundaryRef {
    /// A barrier named by the boundary.
    Barrier(BarrierRef),
    /// A launch named by the boundary.
    Launch(LaunchRef),
}

/// One staged write the transaction will publish.
///
/// A `StagedWrite` is the atomic publication unit: reserved at prepare
/// (byte-counted against the partition's admitted class 3 budget), staged at
/// its partition during execute, and published **all-or-nothing** at commit.
/// A failure or cancel before commit publishes nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StagedWrite {
    partition: LogicalPartitionId,
    output_ref: OutputRef,
    byte_count: u64,
}

impl StagedWrite {
    /// Build a staged write for one partition's output buffer.
    #[must_use]
    pub const fn new(partition: LogicalPartitionId, output_ref: OutputRef, byte_count: u64) -> Self {
        Self {
            partition,
            output_ref,
            byte_count,
        }
    }

    /// The partition that stages this write.
    #[must_use]
    pub fn partition(&self) -> &LogicalPartitionId {
        &self.partition
    }

    /// The stable output reference.
    #[must_use]
    pub fn output_ref(&self) -> &OutputRef {
        &self.output_ref
    }

    /// The byte count of the staged output.
    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }
}

/// Mirror of one operation of the accepted plan (FC8 pattern).
///
/// `Launch` carries the declared output byte contract of the launch — the
/// write the launch stages — so `prepare` can reserve output buffers
/// (S3). `Transfer` / `CollectiveBroadcast` mirror the logical operation
/// facts in stable canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionOperation {
    /// A launch of the semantic program on one partition.
    Launch {
        /// The partition that runs the launch.
        partition: LogicalPartitionId,
        /// The stable reference of the launched program/launch.
        launch_ref: LaunchRef,
        /// The declared byte contract of the launch's output write.
        output_bytes: u64,
    },
    /// A host-staged cross-partition transfer.
    Transfer(TransferOperationMirror),
    /// A broadcast collective (source → participants).
    CollectiveBroadcast(CollectiveBroadcastMirror),
    /// A barrier synchronizing its participant partitions.
    Barrier {
        /// The stable barrier reference.
        barrier_ref: BarrierRef,
        /// The participant partition set.
        partitions: BTreeSet<LogicalPartitionId>,
    },
}

impl TransactionOperation {
    /// A launch operation.
    #[must_use]
    pub fn launch(partition: LogicalPartitionId, launch_ref: LaunchRef, output_bytes: u64) -> Self {
        Self::Launch {
            partition,
            launch_ref,
            output_bytes,
        }
    }

    /// A transfer operation.
    #[must_use]
    pub fn transfer(transfer: TransferOperationMirror) -> Self {
        Self::Transfer(transfer)
    }

    /// A broadcast operation.
    #[must_use]
    pub fn broadcast(broadcast: CollectiveBroadcastMirror) -> Self {
        Self::CollectiveBroadcast(broadcast)
    }

    /// A barrier operation.
    #[must_use]
    pub fn barrier(barrier_ref: BarrierRef, partitions: BTreeSet<LogicalPartitionId>) -> Self {
        Self::Barrier {
            barrier_ref,
            partitions,
        }
    }

    /// The operation's stable snapshot key.
    #[must_use]
    pub fn operation_ref(&self) -> OperationRef {
        match self {
            Self::Launch { launch_ref, .. } => OperationRef::Launch(launch_ref.clone()),
            Self::Transfer(transfer) => OperationRef::Transfer(transfer.id().clone()),
            Self::CollectiveBroadcast(broadcast) => {
                OperationRef::Collective(broadcast.id().clone())
            }
            Self::Barrier { barrier_ref, .. } => OperationRef::Barrier(barrier_ref.clone()),
        }
    }

    /// The partitions this operation involves, in stable order.
    #[must_use]
    pub fn partitions(&self) -> BTreeSet<LogicalPartitionId> {
        match self {
            Self::Launch { partition, .. } => BTreeSet::from([partition.clone()]),
            Self::Transfer(transfer) => BTreeSet::from([
                transfer.source().clone(),
                transfer.destination().clone(),
            ]),
            Self::CollectiveBroadcast(broadcast) => broadcast.participants().clone(),
            Self::Barrier { partitions, .. } => partitions.clone(),
        }
    }

    /// The exact byte contract of the operation: the bytes moved across the
    /// mesh (a transfer moves its byte count; a broadcast moves one copy per
    /// non-source participant; launches and barriers move nothing — their
    /// writes are accounted in the staged write-set).
    #[must_use]
    pub fn byte_count(&self) -> u64 {
        match self {
            Self::Launch { .. } | Self::Barrier { .. } => 0,
            Self::Transfer(transfer) => transfer.byte_count(),
            Self::CollectiveBroadcast(broadcast) => {
                broadcast.byte_count() * (broadcast.participants().len().saturating_sub(1) as u64)
            }
        }
    }

    /// The events this operation completes when it runs, one per involved
    /// partition. An operation is not complete until its declared events join
    /// the boundary or cancellation-safe reclamation reclaims them (S3).
    #[must_use]
    pub fn completed_events(&self) -> BTreeSet<OperationEvent> {
        match self {
            Self::Launch {
                partition,
                launch_ref,
                ..
            } => BTreeSet::from([OperationEvent::LaunchCompleted {
                partition: partition.clone(),
                launch_ref: launch_ref.clone(),
            }]),
            Self::Transfer(transfer) => BTreeSet::from([
                OperationEvent::TransferCompleted {
                    partition: transfer.source().clone(),
                    transfer_ref: transfer.id().clone(),
                },
                OperationEvent::TransferCompleted {
                    partition: transfer.destination().clone(),
                    transfer_ref: transfer.id().clone(),
                },
            ]),
            Self::CollectiveBroadcast(broadcast) => broadcast
                .participants()
                .iter()
                .map(|partition| OperationEvent::BroadcastCompleted {
                    partition: partition.clone(),
                    collective_ref: broadcast.id().clone(),
                })
                .collect(),
            Self::Barrier {
                barrier_ref,
                partitions,
            } => partitions
                .iter()
                .map(|partition| OperationEvent::BarrierCompleted {
                    partition: partition.clone(),
                    barrier_ref: barrier_ref.clone(),
                })
                .collect(),
        }
    }

    /// The staged writes this operation publishes, derived deterministically
    /// from the operation's declared facts (the atomic publication set is the
    /// union over the snapshot).
    #[must_use]
    pub fn staged_writes(&self) -> Vec<StagedWrite> {
        match self {
            Self::Launch {
                partition,
                launch_ref,
                output_bytes,
            } => vec![StagedWrite::new(
                partition.clone(),
                OutputRef::new(format!("launch:{launch_ref}:output")),
                *output_bytes,
            )],
            Self::Transfer(transfer) => vec![StagedWrite::new(
                transfer.destination().clone(),
                OutputRef::new(format!("transfer:{}:destination", transfer.id())),
                transfer.byte_count(),
            )],
            Self::CollectiveBroadcast(broadcast) => broadcast
                .participants()
                .iter()
                .filter(|participant| **participant != *broadcast.source())
                .map(|participant| {
                    StagedWrite::new(
                        participant.clone(),
                        OutputRef::new(format!("broadcast:{}:{participant}", broadcast.id())),
                        broadcast.byte_count(),
                    )
                })
                .collect(),
            Self::Barrier { .. } => Vec::new(),
        }
    }

    /// Deterministic canonical bytes of the operation — identical inputs
    /// produce identical bytes and different operations produce different
    /// bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Launch {
                partition,
                launch_ref,
                output_bytes,
            } => {
                out.push(0u8); // tag: Launch
                push_str(&mut out, partition.as_str());
                push_str(&mut out, launch_ref.as_str());
                push_u64(&mut out, *output_bytes);
            }
            Self::Transfer(transfer) => {
                out.push(1u8); // tag: Transfer
                out.extend_from_slice(&transfer.canonical_bytes());
            }
            Self::CollectiveBroadcast(broadcast) => {
                out.push(2u8); // tag: CollectiveBroadcast
                out.extend_from_slice(&broadcast.canonical_bytes());
            }
            Self::Barrier {
                barrier_ref,
                partitions,
            } => {
                out.push(3u8); // tag: Barrier
                push_str(&mut out, barrier_ref.as_str());
                push_u64(&mut out, partitions.len() as u64);
                for partition in partitions {
                    push_str(&mut out, partition.as_str());
                }
            }
        }
        out
    }
}

/// One synchronization event of the transaction — a partition reaching one
/// operation. Events join the declared boundary at commit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperationEvent {
    /// The partition completed the named launch.
    LaunchCompleted {
        /// The partition that ran the launch.
        partition: LogicalPartitionId,
        /// The completed launch reference.
        launch_ref: LaunchRef,
    },
    /// The partition completed the named transfer.
    TransferCompleted {
        /// The partition (source or destination).
        partition: LogicalPartitionId,
        /// The completed transfer identity.
        transfer_ref: TransferRef,
    },
    /// The partition completed the named broadcast.
    BroadcastCompleted {
        /// The participant partition.
        partition: LogicalPartitionId,
        /// The completed collective identity.
        collective_ref: CollectiveRef,
    },
    /// The partition reached the named barrier.
    BarrierCompleted {
        /// The participant partition.
        partition: LogicalPartitionId,
        /// The reached barrier reference.
        barrier_ref: BarrierRef,
    },
}

// --- reservation ------------------------------------------------------------

/// Declared per-operation event-object byte cost, charged to class 3
/// (`activation_scratch_bytes`). Each operation reserves one event object per
/// involved partition.
pub const EVENT_OBJECT_BYTES: u64 = 64;

/// Declared per-partition transaction-scratch byte cost, charged to class 3
/// (`activation_scratch_bytes`). Every participating partition reserves this
/// coordinator scratch for the transaction's lifetime.
pub const TRANSACTION_SCRATCH_BYTES_PER_PARTITION: u64 = 4096;

/// Which admitted ledger class a reservation field is charged against
/// (md0-closeout §3.2 item 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetClass {
    /// Ledger class 6 — `PartitionBudgetLedger::transfer_staging_bytes`.
    TransferStaging,
    /// Ledger class 3 — `PartitionBudgetLedger::activation_scratch_bytes`.
    ActivationScratch,
}

impl BudgetClass {
    /// The ledger class number (6 or 3).
    #[must_use]
    pub const fn ledger_class(self) -> u64 {
        match self {
            Self::TransferStaging => 6,
            Self::ActivationScratch => 3,
        }
    }
}

/// The per-partition reservation `prepare` derives from the accepted plan
/// (S3). Staging charges class 6 (`transfer_staging_bytes`); output buffers,
/// events, and transaction scratch charge class 3 (`activation_scratch_bytes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReservationRecord {
    /// Class 6: typed transfer staging buffers (in-flight copies at full
    /// size).
    transfer_staging_bytes: u64,
    /// Class 3: staged output buffer bytes.
    output_buffer_bytes: u64,
    /// Class 3: event objects.
    event_bytes: u64,
    /// Class 3: transaction scratch.
    transaction_scratch_bytes: u64,
}

impl ReservationRecord {
    /// Build a reservation record from its four charged fields.
    #[must_use]
    pub const fn new(
        transfer_staging_bytes: u64,
        output_buffer_bytes: u64,
        event_bytes: u64,
        transaction_scratch_bytes: u64,
    ) -> Self {
        Self {
            transfer_staging_bytes,
            output_buffer_bytes,
            event_bytes,
            transaction_scratch_bytes,
        }
    }

    /// The class 6 staging reservation.
    #[must_use]
    pub const fn transfer_staging_bytes(&self) -> u64 {
        self.transfer_staging_bytes
    }

    /// The class 3 output-buffer reservation.
    #[must_use]
    pub const fn output_buffer_bytes(&self) -> u64 {
        self.output_buffer_bytes
    }

    /// The class 3 event reservation.
    #[must_use]
    pub const fn event_bytes(&self) -> u64 {
        self.event_bytes
    }

    /// The class 3 transaction-scratch reservation.
    #[must_use]
    pub const fn transaction_scratch_bytes(&self) -> u64 {
        self.transaction_scratch_bytes
    }

    /// The class 6 charge (transfer staging).
    #[must_use]
    pub const fn class_six_bytes(&self) -> u64 {
        self.transfer_staging_bytes
    }

    /// The class 3 charge (output buffers + events + transaction scratch).
    #[must_use]
    pub const fn class_three_bytes(&self) -> u64 {
        self.output_buffer_bytes + self.event_bytes + self.transaction_scratch_bytes
    }
}

/// Deterministic reservation derivation over the accepted plan (S3).
///
/// Accounting rules (charged to the partition that holds the resource):
///
/// - A transfer reserves full `byte_count` staging at **both** endpoints
///   (the host-staged move stages both halves, class 6) and the destination's
///   staged output write (class 3).
/// - A broadcast reserves `byte_count` staging at **every** participant (the
///   value leaves the source and enters each destination through host
///   staging, class 6) and `byte_count` output at every non-source
///   participant (class 3).
/// - A launch reserves its declared `output_bytes` at its partition (class 3).
/// - Every operation reserves [`EVENT_OBJECT_BYTES`] per involved partition
///   and every participating partition reserves
///   [`TRANSACTION_SCRATCH_BYTES_PER_PARTITION`] (class 3).
fn derive_reservation(
    operations: &[TransactionOperation],
    referenced: &BTreeSet<LogicalPartitionId>,
) -> BTreeMap<LogicalPartitionId, ReservationRecord> {
    let mut staging: BTreeMap<LogicalPartitionId, u64> = BTreeMap::new();
    let mut outputs: BTreeMap<LogicalPartitionId, u64> = BTreeMap::new();
    let mut events: BTreeMap<LogicalPartitionId, u64> = BTreeMap::new();
    for operation in operations {
        for partition in operation.partitions() {
            *events.entry(partition).or_insert(0) += EVENT_OBJECT_BYTES;
        }
        match operation {
            TransactionOperation::Launch {
                partition,
                output_bytes,
                ..
            } => {
                *outputs.entry(partition.clone()).or_insert(0) += *output_bytes;
            }
            TransactionOperation::Transfer(transfer) => {
                let count = transfer.byte_count();
                *staging.entry(transfer.source().clone()).or_insert(0) += count;
                *staging.entry(transfer.destination().clone()).or_insert(0) += count;
                *outputs.entry(transfer.destination().clone()).or_insert(0) += count;
            }
            TransactionOperation::CollectiveBroadcast(broadcast) => {
                let count = broadcast.byte_count();
                for participant in broadcast.participants() {
                    *staging.entry(participant.clone()).or_insert(0) += count;
                    if participant != broadcast.source() {
                        *outputs.entry(participant.clone()).or_insert(0) += count;
                    }
                }
            }
            TransactionOperation::Barrier { .. } => {}
        }
    }
    referenced
        .iter()
        .map(|partition| {
            let record = ReservationRecord::new(
                staging.get(partition).copied().unwrap_or(0),
                outputs.get(partition).copied().unwrap_or(0),
                events.get(partition).copied().unwrap_or(0),
                TRANSACTION_SCRATCH_BYTES_PER_PARTITION,
            );
            (partition.clone(), record)
        })
        .collect()
}

/// The declared write-set of the accepted plan — the set of writes the
/// transaction will publish, derived deterministically at prepare (one
/// `OutputRef` per producing operation).
fn derive_declared_write_set(operations: &[TransactionOperation]) -> BTreeMap<OutputRef, StagedWrite> {
    let mut writes = BTreeMap::new();
    for operation in operations {
        for write in operation.staged_writes() {
            writes.insert(write.output_ref().clone(), write);
        }
    }
    writes
}

// --- backend abstraction ----------------------------------------------------

/// A backend failure. The fault classes mirror the MD3-F1 suite vocabulary
/// (cancel / timeout / transfer or kernel error / device loss / allocation);
/// the coordinator aborts on any backend failure with no partial publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// A reserve/allocate request could not be satisfied (physical pressure).
    Allocation {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// What failed, as reported by the runtime.
        detail: String,
    },
    /// An operation (transfer / kernel / barrier) failed.
    Operation {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// What failed, as reported by the runtime.
        detail: String,
    },
    /// The bound physical device failed or was removed (MD-A13).
    DeviceLoss {
        /// The lost partition's bound device, via its partition.
        partition: LogicalPartitionId,
        /// What happened, as reported by the runtime.
        detail: String,
    },
    /// The operation was cancelled before completion.
    Cancelled {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// Why it was cancelled.
        detail: String,
    },
    /// The operation timed out.
    Timeout {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// The timeout fact, as reported by the runtime.
        detail: String,
    },
}

impl BackendError {
    /// An allocation failure for one partition.
    #[must_use]
    pub fn allocation(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Allocation {
            partition,
            detail: detail.into(),
        }
    }

    /// An operation failure (transfer / kernel / barrier) for one partition.
    #[must_use]
    pub fn operation(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Operation {
            partition,
            detail: detail.into(),
        }
    }

    /// A device-loss failure (MD-A13) for one partition.
    #[must_use]
    pub fn device_loss(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::DeviceLoss {
            partition,
            detail: detail.into(),
        }
    }

    /// A cancellation for one partition.
    #[must_use]
    pub fn cancelled(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Cancelled {
            partition,
            detail: detail.into(),
        }
    }

    /// A timeout for one partition.
    #[must_use]
    pub fn timeout(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Timeout {
            partition,
            detail: detail.into(),
        }
    }

    /// The partition the failure names.
    #[must_use]
    pub fn partition(&self) -> &LogicalPartitionId {
        match self {
            Self::Allocation { partition, .. }
            | Self::Operation { partition, .. }
            | Self::DeviceLoss { partition, .. }
            | Self::Cancelled { partition, .. }
            | Self::Timeout { partition, .. } => partition,
        }
    }
}

/// The device-execution abstraction the transaction drives (MD3-X1; the real
/// implementation over MD1-H1's `DeviceRuntimeSet` is MD3-S1, QUEUED).
///
/// The transaction dispatches each snapshot operation via
/// [`DeviceExecutionBackend::run_operation`], stages the declared writes via
/// [`DeviceExecutionBackend::stage_write`], gates publication on the declared
/// boundary through [`DeviceExecutionBackend::event_completed`], publishes
/// the staged write-set atomically via [`DeviceExecutionBackend::publish`],
/// and tears down via [`DeviceExecutionBackend::release`] /
/// [`DeviceExecutionBackend::retire`].
pub trait DeviceExecutionBackend {
    /// Reserve the declared transaction resources for one partition.
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError>;

    /// Run one operation of the accepted plan.
    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError>;

    /// Whether a previously dispatched operation's event has completed
    /// (asynchronous join).
    fn event_completed(&self, event: &OperationEvent) -> bool;

    /// Stage one declared write into the staged write-set. Staging the same
    /// write twice is an error (the write-set is declared once).
    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError>;

    /// Publish every staged write atomically — all or nothing.
    fn publish(&mut self) -> Result<(), BackendError>;

    /// The total bytes currently staged.
    fn staged_bytes(&self) -> u64;

    /// The total bytes published by the last publish.
    fn published_bytes(&self) -> u64;

    /// Release the reservation held for one partition (commit path).
    fn release(&mut self, partition: &LogicalPartitionId);

    /// Retire a partition's staged state after a failure (no publication).
    fn retire(&mut self, partition: &LogicalPartitionId, failure: &TransactionFailure);
}

/// Minimal happy-path `DeviceExecutionBackend` driving the T2 §5 virtual
/// fixture. It records reservations, executes operations (completing their
/// events synchronously, or on demand when auto-complete is off), stages and
/// publishes the write-set atomically, and tracks byte accounting. MD3-F1's
/// fault-injecting backend wraps this fake; MD3-S1 implements the real
/// backend over `DeviceRuntimeSet`.
#[derive(Debug, Clone)]
pub struct FakeExecutionBackend {
    reservations: BTreeMap<LogicalPartitionId, ReservationRecord>,
    staged_writes: BTreeMap<OutputRef, StagedWrite>,
    published_writes: BTreeMap<OutputRef, StagedWrite>,
    staged_total_bytes: u64,
    published_total_bytes: u64,
    published_once: bool,
    executed_operations: Vec<TransactionOperation>,
    dispatched_events: BTreeSet<OperationEvent>,
    completed_events: BTreeSet<OperationEvent>,
    released_partitions: BTreeSet<LogicalPartitionId>,
    retired_partitions: BTreeSet<LogicalPartitionId>,
    auto_complete: bool,
}

impl FakeExecutionBackend {
    /// A fresh happy-path fake; events complete synchronously.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reservations: BTreeMap::new(),
            staged_writes: BTreeMap::new(),
            published_writes: BTreeMap::new(),
            staged_total_bytes: 0,
            published_total_bytes: 0,
            published_once: false,
            executed_operations: Vec::new(),
            dispatched_events: BTreeSet::new(),
            completed_events: BTreeSet::new(),
            released_partitions: BTreeSet::new(),
            retired_partitions: BTreeSet::new(),
            auto_complete: true,
        }
    }

    /// When on, every dispatched operation completes its events
    /// synchronously. When off, events stay pending until
    /// [`Self::complete_event`] — the asynchronous-join simulation the
    /// boundary gate test drives.
    pub fn set_auto_complete(&mut self, on: bool) {
        self.auto_complete = on;
    }

    /// Mark a dispatched event completed (asynchronous join). Lenient: an
    /// unknown event is recorded as completed anyway.
    pub fn complete_event(&mut self, event: OperationEvent) {
        self.dispatched_events.insert(event.clone());
        self.completed_events.insert(event);
    }

    /// The held reservations.
    #[must_use]
    pub fn reservations(&self) -> &BTreeMap<LogicalPartitionId, ReservationRecord> {
        &self.reservations
    }

    /// The staged write-set.
    #[must_use]
    pub fn staged_writes(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.staged_writes
    }

    /// The published write-set (populated by `publish`).
    #[must_use]
    pub fn published_writes(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.published_writes
    }

    /// The operations dispatched, in dispatch order.
    #[must_use]
    pub fn executed_operations(&self) -> &[TransactionOperation] {
        &self.executed_operations
    }

    /// The completed events.
    #[must_use]
    pub fn completed_events(&self) -> &BTreeSet<OperationEvent> {
        &self.completed_events
    }

    /// The dispatched-but-not-yet-completed events.
    #[must_use]
    pub fn pending_events(&self) -> BTreeSet<OperationEvent> {
        self.dispatched_events
            .difference(&self.completed_events)
            .cloned()
            .collect()
    }

    /// The partitions whose reservations were released.
    #[must_use]
    pub fn released_partitions(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.released_partitions
    }

    /// The partitions whose staged state was retired.
    #[must_use]
    pub fn retired_partitions(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.retired_partitions
    }
}

impl Default for FakeExecutionBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceExecutionBackend for FakeExecutionBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        if self.reservations.contains_key(partition) {
            return Err(BackendError::Allocation {
                partition: partition.clone(),
                detail: format!("partition {partition} already holds a reservation"),
            });
        }
        self.reservations.insert(partition.clone(), *reservation);
        Ok(())
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        let events = operation.completed_events();
        self.executed_operations.push(operation.clone());
        for event in events {
            self.dispatched_events.insert(event.clone());
            if self.auto_complete {
                self.completed_events.insert(event);
            }
        }
        Ok(())
    }

    fn event_completed(&self, event: &OperationEvent) -> bool {
        self.completed_events.contains(event)
    }

    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError> {
        if self.staged_writes.contains_key(write.output_ref()) {
            return Err(BackendError::Allocation {
                partition: write.partition().clone(),
                detail: format!("output {} staged twice", write.output_ref()),
            });
        }
        self.staged_total_bytes = self.staged_total_bytes.saturating_add(write.byte_count());
        self.staged_writes
            .insert(write.output_ref().clone(), write.clone());
        Ok(())
    }

    fn publish(&mut self) -> Result<(), BackendError> {
        if self.published_once {
            return Err(BackendError::Operation {
                partition: LogicalPartitionId::new("unknown"),
                detail: "publish called twice — publication is one-shot".to_owned(),
            });
        }
        // Atomic all-or-nothing: promote the whole staged write-set.
        self.published_writes = self.staged_writes.clone();
        self.published_total_bytes = self.staged_total_bytes;
        self.published_once = true;
        Ok(())
    }

    fn staged_bytes(&self) -> u64 {
        self.staged_total_bytes
    }

    fn published_bytes(&self) -> u64 {
        self.published_total_bytes
    }

    fn release(&mut self, partition: &LogicalPartitionId) {
        self.reservations.remove(partition);
        self.released_partitions.insert(partition.clone());
    }

    fn retire(&mut self, partition: &LogicalPartitionId, _failure: &TransactionFailure) {
        let retired: u64 = self
            .staged_writes
            .values()
            .filter(|write| write.partition() == partition)
            .map(|write| write.byte_count())
            .sum();
        self.staged_total_bytes = self.staged_total_bytes.saturating_sub(retired);
        self.staged_writes
            .retain(|_, write| write.partition() != partition);
        self.reservations.remove(partition);
        self.retired_partitions.insert(partition.clone());
    }
}

// --- state machine ----------------------------------------------------------

/// Machine-local opaque identity of one transaction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransactionId(String);

impl TransactionId {
    /// Build a transaction id from its stable string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The stable identity string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The **abstract publication ordinal** — the transaction-scoped publication
/// counter a committed transaction records in its receipt. This is the
/// "abstract execution generation ordinal" of the MD3 spec: a transaction-
/// scoped publication ordinal, **never the semantic `ValueGeneration`**
/// (naming contract §3). Minted by the coordinator that owns the publication
/// counter; `commit` records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PublicationOrdinal(u64);

impl PublicationOrdinal {
    /// Build a publication ordinal from a publication-counter value.
    #[must_use]
    pub const fn new(ordinal: u64) -> Self {
        Self(ordinal)
    }

    /// The raw ordinal value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for PublicationOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pub:{}", self.0)
    }
}

/// The deterministic state machine: `New → Prepared → Executing →
/// Committed | Aborted`, with `Failed → Aborted`. Retry is disabled — there
/// is no re-execution path once the machine leaves `Prepared`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionState {
    /// Constructed; nothing reserved.
    New,
    /// `prepare` succeeded: the reservation is recorded and held.
    Prepared,
    /// `execute` is running the accepted plan (or finished dispatching it,
    /// awaiting the boundary).
    Executing,
    /// `commit` published the staged write-set atomically.
    Committed,
    /// An operation or the publication failed; `abort` completes teardown.
    Failed(TransactionFailure),
    /// `abort` completed teardown; no publication happened.
    Aborted(TransactionFailure),
}

/// The recorded failure of the transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionFailure {
    /// A backend operation or reservation failed.
    Backend(BackendError),
    /// The atomic publication failed after the boundary was reached.
    PublishFailed {
        /// What failed, as reported by the backend.
        detail: String,
    },
    /// The transaction was cancelled.
    Cancelled {
        /// Why it was cancelled.
        reason: String,
    },
}

// --- errors -----------------------------------------------------------------

/// Why a transaction could not be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructError {
    /// The bound plan is the MD-A15 single-partition degenerate — one-device
    /// execution stays coordinator-free (no `ExecutionTransaction`).
    DegeneratePlan,
    /// The snapshot is empty — a distributed plan declares at least one
    /// operation.
    EmptySnapshot,
}

/// Why `prepare` rejected the accepted plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareError {
    /// The transaction is not in the `New` state.
    InvalidState {
        /// The current state.
        state: TransactionState,
    },
    /// An operation references a partition that is not declared by the bound
    /// plan (topology authority).
    UnknownPartition {
        /// The undeclared partition.
        partition: LogicalPartitionId,
        /// The index of the offending operation in the snapshot.
        operation_index: usize,
    },
    /// The snapshot contains two operations with the same stable identity.
    DuplicateOperation {
        /// The duplicated operation reference.
        operation_ref: OperationRef,
    },
    /// The declared boundary names a barrier the snapshot does not contain.
    UndeclaredBoundaryBarrier {
        /// The undeclared barrier.
        barrier: BarrierRef,
    },
    /// The declared boundary names a launch the snapshot does not contain.
    UndeclaredBoundaryLaunch {
        /// The undeclared launch.
        launch: LaunchRef,
    },
    /// A referenced partition's binding carries no admitted virtual partition,
    /// so there is no admitted budget the reservation can be checked against.
    MissingAdmittedBudget {
        /// The partition with no admitted budget.
        partition: LogicalPartitionId,
    },
    /// The derived reservation exceeds the partition's admitted budget for
    /// one ledger class (S3) — the focused over-commit diagnostic
    /// (MD2-C1 residual 2 closure).
    ReservationExceedsBudget {
        /// The over-committed partition.
        partition: LogicalPartitionId,
        /// The ledger class that was exceeded.
        class: BudgetClass,
        /// The derived reservation for that class.
        declared_bytes: u64,
        /// The admitted budget for that class.
        admitted_bytes: u64,
    },
    /// The backend could not hold the reservation.
    Backend(BackendError),
}

/// Why `execute` failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecuteError {
    /// The transaction is not in the `Prepared`/`Executing` state the call
    /// requires. Retry is disabled: after `Failed`/`Aborted`/`Committed`
    /// there is no re-execution path.
    InvalidState {
        /// The current state.
        state: TransactionState,
    },
    /// The operation is not part of the prepare snapshot (the accepted
    /// plan) — `execute` must never silently grow the accepted plan (S3).
    OperationOutsideSnapshot {
        /// The rejected operation reference.
        operation_ref: OperationRef,
    },
    /// The backend failed while running or staging an operation. The
    /// transaction recorded the failure and moved to `Failed`; no partial
    /// publication is possible.
    Backend(BackendError),
}

/// Why `commit` did not publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    /// The transaction is not in the `Executing` state.
    InvalidState {
        /// The current state.
        state: TransactionState,
    },
    /// The declared `TransactionCommitBoundary` was not reached — some
    /// boundary barriers/launches have not completed. Transient: nothing was
    /// published, the transaction stays `Executing`, and `commit` may be
    /// called again once the events join the boundary.
    BoundaryNotReached {
        /// The boundary references not yet completed.
        missing: BTreeSet<BoundaryRef>,
    },
    /// The atomic publication failed. Nothing was published; the transaction
    /// moved to `Failed` and must be aborted.
    PublishFailed(BackendError),
}

/// Why `abort` could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortError {
    /// The transaction already committed — a committed transaction is
    /// terminal.
    AlreadyCommitted,
}

// --- receipt ----------------------------------------------------------------

/// The commit/abort decision of the transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionDecision {
    /// The staged write-set was published atomically.
    Committed {
        /// The abstract publication ordinal (transaction-scoped).
        publication_ordinal: PublicationOrdinal,
    },
    /// The transaction was aborted; nothing was published.
    Aborted {
        /// The recorded failure (the cancel reason or the originating
        /// failure).
        failure: TransactionFailure,
    },
}

/// One executed operation of the snapshot, with its exact byte contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedOperationRecord {
    /// The operation as dispatched.
    pub operation: TransactionOperation,
    /// The operation's exact byte contract (bytes moved across the mesh).
    pub byte_count: u64,
}

/// The atomic-publication summary recorded in a commit receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishSummary {
    /// Total bytes staged before publication.
    pub staged_bytes: u64,
    /// Total bytes published.
    pub published_bytes: u64,
    /// Always true when present — publication is all-or-nothing.
    pub atomic: bool,
}

/// Teardown facts: which partitions were released and which were retired.
/// `partial_publication` is an invariant — always false.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TeardownFacts {
    /// Partitions whose reservations were released (commit, or abort without
    /// staged state).
    pub released_partitions: BTreeSet<LogicalPartitionId>,
    /// Partitions whose staged state was retired on abort/failure.
    pub retired_partitions: BTreeSet<LogicalPartitionId>,
    /// Invariant: a partial publication never happens.
    pub partial_publication: bool,
}

/// Wall-clock phase timings of the transaction (runtime evidence; not
/// canonical).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransactionTimings {
    /// `prepare` elapsed, nanoseconds.
    pub prepare_nanos: u64,
    /// `execute` elapsed, nanoseconds.
    pub execute_nanos: u64,
    /// `commit`/`abort` elapsed, nanoseconds.
    pub finalize_nanos: u64,
}

/// The base `TransactionReceipt` (exit-gate bullet 6; S4): transaction id,
/// both plan hashes, the device/virtual identities from the bound plan, the
/// per-partition reservation summary, the declared staged write-set, the
/// executed operations with exact bytes, the synchronization events, the
/// commit/abort decision + reason, the publication summary, teardown facts,
/// phase timings, and the S4 selected-transport section (the actual selected
/// transports: copy path/staging/events/timeout/bytes/timing + budget
/// accounting — the folded [`TransportReceipt`], CTO sanity-check amendment;
/// `None` when no transport adapter recorded transfers for this transaction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransactionReceipt {
    /// The transaction identity.
    pub transaction_id: TransactionId,
    /// The admitted logical plan hash (from the bound plan; never re-derived).
    pub logical_distributed_plan_hash: String,
    /// The bound-plan hash (from the bound plan).
    pub bound_distributed_plan_hash: String,
    /// Device/virtual identities from the bound plan (`hardware_isolation_claimed=false`).
    pub plan_receipt: PartitionReceipt,
    /// Per-partition reservation summary (the prepare receipt).
    pub reservation_summary: BTreeMap<LogicalPartitionId, ReservationRecord>,
    /// The declared staged write-set (reserved at prepare).
    pub declared_write_set: BTreeMap<OutputRef, StagedWrite>,
    /// The executed operations in plan order, with exact bytes.
    pub executed_operations: Vec<ExecutedOperationRecord>,
    /// The synchronization events completed when the transaction finalized.
    pub synchronization_events: BTreeSet<OperationEvent>,
    /// The commit/abort decision + reason.
    pub decision: TransactionDecision,
    /// The atomic-publication summary (`None` when nothing was published).
    pub publish_summary: Option<PublishSummary>,
    /// Teardown facts.
    pub teardown: TeardownFacts,
    /// Phase timings.
    pub timings: TransactionTimings,
    /// The S4 selected-transport section folded from the transport adapter
    /// used during the transaction (path/staging/events/timeout/bytes/timing
    /// + budget accounting at the measured rates). `None` when no transport
    /// adapter recorded transfers for this transaction.
    pub selected_transports: Option<TransportReceipt>,
}

// --- the transaction --------------------------------------------------------

/// The generic `ExecutionTransaction` coordinator (MD3-X1).
///
/// Consumes the accepted plan as a dependency-free mirror over a
/// [`BoundDistributedPlan`]. Lifecycle: [`new`](Self::new) → [`prepare`](
/// Self::prepare) → [`execute`](Self::execute) → [`commit`](Self::commit) |
/// [`abort`](Self::abort).
#[derive(Debug, Clone)]
pub struct ExecutionTransaction {
    id: TransactionId,
    bound_plan: BoundDistributedPlan,
    operations: Vec<TransactionOperation>,
    commit_boundary: TransactionCommitBoundary,
    operation_keys: BTreeMap<OperationRef, usize>,
    state: TransactionState,
    reservation: Option<BTreeMap<LogicalPartitionId, ReservationRecord>>,
    declared_write_set: BTreeMap<OutputRef, StagedWrite>,
    staged_write_set: BTreeMap<OutputRef, StagedWrite>,
    executed_operations: Vec<ExecutedOperationRecord>,
    completed_events: BTreeSet<OperationEvent>,
    failure: Option<TransactionFailure>,
    decision: Option<TransactionDecision>,
    publish_summary: Option<PublishSummary>,
    teardown: TeardownFacts,
    timings: TransactionTimings,
    transport_receipt: Option<TransportReceipt>,
    receipt: Option<TransactionReceipt>,
}

impl ExecutionTransaction {
    /// Construct a transaction over an accepted (already bound) distributed
    /// plan. The MD-A15 single-partition degenerate and an empty snapshot are
    /// rejected at construction. The snapshot is fixed here — it is the
    /// accepted plan and never grows.
    #[must_use]
    pub fn new(
        id: TransactionId,
        bound_plan: BoundDistributedPlan,
        operations: Vec<TransactionOperation>,
        commit_boundary: TransactionCommitBoundary,
    ) -> Result<Self, ConstructError> {
        if bound_plan.is_degenerate() {
            return Err(ConstructError::DegeneratePlan);
        }
        if operations.is_empty() {
            return Err(ConstructError::EmptySnapshot);
        }
        Ok(Self {
            id,
            bound_plan,
            operations,
            commit_boundary,
            operation_keys: BTreeMap::new(),
            state: TransactionState::New,
            reservation: None,
            declared_write_set: BTreeMap::new(),
            staged_write_set: BTreeMap::new(),
            executed_operations: Vec::new(),
            completed_events: BTreeSet::new(),
            failure: None,
            decision: None,
            publish_summary: None,
            teardown: TeardownFacts::default(),
            timings: TransactionTimings::default(),
            transport_receipt: None,
            receipt: None,
        })
    }

    /// The transaction identity.
    #[must_use]
    pub fn id(&self) -> &TransactionId {
        &self.id
    }

    /// The bound plan this transaction executes.
    #[must_use]
    pub fn bound_plan(&self) -> &BoundDistributedPlan {
        &self.bound_plan
    }

    /// The current state-machine state.
    #[must_use]
    pub fn state(&self) -> &TransactionState {
        &self.state
    }

    /// The accepted plan (the prepare snapshot), in plan order.
    #[must_use]
    pub fn operations(&self) -> &[TransactionOperation] {
        &self.operations
    }

    /// The declared commit boundary.
    #[must_use]
    pub fn commit_boundary(&self) -> &TransactionCommitBoundary {
        &self.commit_boundary
    }

    /// The per-partition reservation recorded at prepare (the prepare
    /// receipt's reservation summary); `None` before prepare.
    #[must_use]
    pub fn reservation(
        &self,
    ) -> Option<&BTreeMap<LogicalPartitionId, ReservationRecord>> {
        self.reservation.as_ref()
    }

    /// The declared staged write-set (reserved at prepare); empty before
    /// prepare.
    #[must_use]
    pub fn declared_write_set(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.declared_write_set
    }

    /// The executed operations in plan order (populated by execute).
    #[must_use]
    pub fn executed_operations(&self) -> &[ExecutedOperationRecord] {
        &self.executed_operations
    }

    /// The completed synchronization events.
    #[must_use]
    pub fn completed_events(&self) -> &BTreeSet<OperationEvent> {
        &self.completed_events
    }

    /// The recorded failure, when the transaction failed or was aborted.
    #[must_use]
    pub fn failure(&self) -> Option<&TransactionFailure> {
        self.failure.as_ref()
    }

    /// The final receipt, present after commit or abort.
    #[must_use]
    pub fn receipt(&self) -> Option<&TransactionReceipt> {
        self.receipt.as_ref()
    }

    /// The S4 selected-transport section recorded for this transaction, if
    /// the coordinator handed the transport adapter's `transport_receipt()`
    /// over ([`with_transport_receipt`](Self::with_transport_receipt)).
    #[must_use]
    pub fn selected_transports(&self) -> Option<&TransportReceipt> {
        self.transport_receipt.as_ref()
    }

    /// Record the S4 selected-transport section from the transport adapter
    /// used during this transaction. The coordinator folds the adapter's
    /// `transport_receipt()` over after execution (the actual selected
    /// transports: copy path/staging/events/timeout/bytes/timing + budget
    /// accounting at the measured rates); the commit/abort receipt carries it
    /// verbatim. Additive — a transaction that never touched a transport
    /// adapter records `None`. The portable logical plan is never touched
    /// (S4: the mirror carries only the admissibility `path_label`).
    pub fn with_transport_receipt(&mut self, receipt: TransportReceipt) -> &mut Self {
        self.transport_receipt = Some(receipt);
        self
    }

    /// Reserve the transaction's resources and validate the accepted plan
    /// (`New → Prepared`).
    ///
    /// Validation order (deterministic):
    ///
    /// 1. **Topology authority** — every operation's partitions must be
    ///    declared by the bound plan.
    /// 2. **Snapshot integrity** — no duplicate operation identities.
    /// 3. **Boundary declaration** — every boundary barrier/launch must be
    ///    declared by the snapshot.
    /// 4. **Admitted budgets** — every referenced partition must carry an
    ///    admitted virtual partition (the actual admitted ledger, frozen at
    ///    admission).
    /// 5. **Reservation (S3)** — the derived reservation is charged against
    ///    the admitted class budgets; an over-commit (a tampered-but-
    ///    internally-consistent plan) fails here, before execute (MD2-C1
    ///    residual 2 closure).
    ///
    /// On success the reservation is held on the backend and recorded (the
    /// prepare receipt). On any failure nothing is reserved and the
    /// transaction stays `New`.
    pub fn prepare(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
    ) -> Result<(), PrepareError> {
        if self.state != TransactionState::New {
            return Err(PrepareError::InvalidState {
                state: self.state.clone(),
            });
        }
        let start = Instant::now();
        let bindings = self
            .bound_plan
            .bindings()
            .expect("construction rejected the degenerate plan");
        let declared_partitions: BTreeSet<LogicalPartitionId> =
            bindings.keys().cloned().collect();

        // 1. Topology authority.
        for (index, operation) in self.operations.iter().enumerate() {
            for partition in operation.partitions() {
                if !declared_partitions.contains(&partition) {
                    return Err(PrepareError::UnknownPartition {
                        partition,
                        operation_index: index,
                    });
                }
            }
        }

        // 2. Snapshot integrity: unique operation identities.
        let mut operation_keys = BTreeMap::new();
        for (index, operation) in self.operations.iter().enumerate() {
            let key = operation.operation_ref();
            if operation_keys.insert(key.clone(), index).is_some() {
                return Err(PrepareError::DuplicateOperation {
                    operation_ref: key,
                });
            }
        }

        // 3. Boundary declaration.
        for barrier in &self.commit_boundary.barriers {
            let declared = self
                .operations
                .iter()
                .any(|operation| {
                    matches!(operation,
                        TransactionOperation::Barrier { barrier_ref, .. }
                            if barrier_ref == barrier)
                });
            if !declared {
                return Err(PrepareError::UndeclaredBoundaryBarrier {
                    barrier: barrier.clone(),
                });
            }
        }
        for launch in &self.commit_boundary.launches {
            let declared = self.operations.iter().any(|operation| {
                matches!(operation,
                    TransactionOperation::Launch { launch_ref, .. }
                        if launch_ref == launch)
            });
            if !declared {
                return Err(PrepareError::UndeclaredBoundaryLaunch {
                    launch: launch.clone(),
                });
            }
        }

        // 4. Admitted budgets: every referenced partition must carry the
        //    actual admitted ledger (frozen at admission).
        let referenced: BTreeSet<LogicalPartitionId> = self
            .operations
            .iter()
            .flat_map(|operation| operation.partitions())
            .collect();
        for partition in &referenced {
            let binding = bindings
                .get(partition)
                .expect("step 1 validated the partition is declared");
            if binding.virtual_partition().is_none() {
                return Err(PrepareError::MissingAdmittedBudget {
                    partition: partition.clone(),
                });
            }
        }

        // 5. Reservation derivation + budget check (S3).
        let reservation = derive_reservation(&self.operations, &referenced);
        for (partition, record) in &reservation {
            let binding = bindings
                .get(partition)
                .expect("step 4 validated the admitted budget");
            let ledger: &PartitionBudgetLedger = binding
                .virtual_partition()
                .expect("step 4 validated the admitted budget")
                .ledger();
            let class_six = record.class_six_bytes();
            if class_six > ledger.transfer_staging_bytes {
                return Err(PrepareError::ReservationExceedsBudget {
                    partition: partition.clone(),
                    class: BudgetClass::TransferStaging,
                    declared_bytes: class_six,
                    admitted_bytes: ledger.transfer_staging_bytes,
                });
            }
            let class_three = record.class_three_bytes();
            if class_three > ledger.activation_scratch_bytes {
                return Err(PrepareError::ReservationExceedsBudget {
                    partition: partition.clone(),
                    class: BudgetClass::ActivationScratch,
                    declared_bytes: class_three,
                    admitted_bytes: ledger.activation_scratch_bytes,
                });
            }
        }

        // Hold the reservation on the backend.
        for (partition, record) in &reservation {
            if let Err(error) = backend.reserve(partition, record) {
                return Err(PrepareError::Backend(error));
            }
        }

        self.operation_keys = operation_keys;
        self.reservation = Some(reservation);
        self.declared_write_set = derive_declared_write_set(&self.operations);
        self.timings.prepare_nanos = start.elapsed().as_nanos() as u64;
        self.state = TransactionState::Prepared;
        Ok(())
    }

    /// Run the accepted plan in order (`Prepared → Executing`).
    ///
    /// The plan's declared order is the dependency-ready order; the backend
    /// gates actual readiness. Each operation dispatches via
    /// [`execute_operation`](Self::execute_operation), so an operation
    /// outside the prepare snapshot can never run. On a backend failure the
    /// transaction moves to `Failed` (the recorded failure); no partial
    /// publication is possible and `abort` completes teardown. Retry is
    /// disabled — `execute` cannot run again.
    pub fn execute(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
    ) -> Result<(), ExecuteError> {
        if self.state != TransactionState::Prepared {
            return Err(ExecuteError::InvalidState {
                state: self.state.clone(),
            });
        }
        let start = Instant::now();
        self.state = TransactionState::Executing;
        let snapshot = self.operations.clone();
        for operation in &snapshot {
            if let Err(error) = self.execute_operation(backend, operation) {
                self.timings.execute_nanos = start.elapsed().as_nanos() as u64;
                return Err(error);
            }
        }
        self.timings.execute_nanos = start.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Run one operation of the accepted plan against the backend.
    ///
    /// The operation must be **exactly** an operation of the prepare snapshot
    /// (same identity *and* same declared facts) — an operation outside the
    /// snapshot fails with [`ExecuteError::OperationOutsideSnapshot`]
    /// (no silent growth, S3). On success the operation's staged writes are
    /// staged and its completed events are recorded.
    pub fn execute_operation(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
        operation: &TransactionOperation,
    ) -> Result<(), ExecuteError> {
        if self.state != TransactionState::Executing {
            return Err(ExecuteError::InvalidState {
                state: self.state.clone(),
            });
        }
        let key = operation.operation_ref();
        let index = match self.operation_keys.get(&key) {
            Some(index) => *index,
            None => {
                return Err(ExecuteError::OperationOutsideSnapshot {
                    operation_ref: key,
                });
            }
        };
        if &self.operations[index] != operation {
            return Err(ExecuteError::OperationOutsideSnapshot {
                operation_ref: key,
            });
        }

        if let Err(error) = backend.run_operation(operation) {
            self.fail(TransactionFailure::Backend(error.clone()));
            return Err(ExecuteError::Backend(error));
        }
        for write in operation.staged_writes() {
            if let Err(error) = backend.stage_write(&write) {
                self.fail(TransactionFailure::Backend(error.clone()));
                return Err(ExecuteError::Backend(error));
            }
            self.staged_write_set
                .insert(write.output_ref().clone(), write);
        }
        self.executed_operations.push(ExecutedOperationRecord {
            operation: operation.clone(),
            byte_count: operation.byte_count(),
        });
        for event in operation.completed_events() {
            if backend.event_completed(&event) {
                self.completed_events.insert(event);
            }
        }
        Ok(())
    }

    /// Publish the staged write-set **atomically** after the declared
    /// [`TransactionCommitBoundary`] is reached (`Executing → Committed`).
    ///
    /// Every boundary barrier/launch must have completed (the backend
    /// confirms the events). A missing boundary is transient: nothing was
    /// published and the transaction stays `Executing` so the events can join
    /// and `commit` can be called again. After the boundary is reached the
    /// whole staged write-set publishes atomically; on success every
    /// reservation is released and the receipt records the abstract
    /// publication ordinal. A publication failure moves the transaction to
    /// `Failed` (nothing published) — `abort` then completes teardown.
    pub fn commit(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
        ordinal: PublicationOrdinal,
    ) -> Result<TransactionReceipt, CommitError> {
        if self.state != TransactionState::Executing {
            return Err(CommitError::InvalidState {
                state: self.state.clone(),
            });
        }
        let start = Instant::now();

        // Re-check every snapshot event against the backend (async joins).
        self.refresh_completed_events(backend);

        let missing = self.missing_boundary_refs();
        if !missing.is_empty() {
            return Err(CommitError::BoundaryNotReached { missing });
        }

        if let Err(error) = backend.publish() {
            self.fail(TransactionFailure::PublishFailed {
                detail: format!("{error:?}"),
            });
            return Err(CommitError::PublishFailed(error));
        }

        // Release every held reservation (commit path).
        let reserved: BTreeSet<LogicalPartitionId> = self
            .reservation
            .as_ref()
            .map(|reservation| reservation.keys().cloned().collect())
            .unwrap_or_default();
        for partition in &reserved {
            backend.release(partition);
        }
        self.teardown.released_partitions.extend(reserved);

        self.publish_summary = Some(PublishSummary {
            staged_bytes: backend.staged_bytes(),
            published_bytes: backend.published_bytes(),
            atomic: true,
        });
        self.decision = Some(TransactionDecision::Committed {
            publication_ordinal: ordinal,
        });
        self.state = TransactionState::Committed;
        self.timings.finalize_nanos = start.elapsed().as_nanos() as u64;
        let receipt = self.build_receipt();
        self.receipt = Some(receipt.clone());
        Ok(receipt)
    }

    /// Release or retire every affected resource with no partial publication
    /// (`New | Prepared | Executing | Failed → Aborted`).
    ///
    /// Partitions holding staged writes are **retired** (staged state
    /// invalidated, reservation released); partitions only holding a
    /// reservation are **released**. The receipt records the abort decision
    /// with the reason. Idempotent: aborting an already-aborted transaction
    /// returns its receipt. A committed transaction cannot be aborted.
    pub fn abort(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
        reason: impl Into<String>,
    ) -> Result<TransactionReceipt, AbortError> {
        if matches!(self.state, TransactionState::Aborted(_)) {
            return Ok(self
                .receipt
                .clone()
                .expect("an aborted transaction always has its receipt"));
        }
        if self.state == TransactionState::Committed {
            return Err(AbortError::AlreadyCommitted);
        }
        let start = Instant::now();

        let staged_partitions: BTreeSet<LogicalPartitionId> = self
            .staged_write_set
            .values()
            .map(|write| write.partition().clone())
            .collect();
        let reserved: BTreeSet<LogicalPartitionId> = self
            .reservation
            .as_ref()
            .map(|reservation| reservation.keys().cloned().collect())
            .unwrap_or_default();

        let failure = match &self.state {
            TransactionState::Failed(failure) => failure.clone(),
            _ => TransactionFailure::Cancelled {
                reason: reason.into(),
            },
        };

        // Retire staged state, release the remaining reservations.
        for partition in &staged_partitions {
            backend.retire(partition, &failure);
        }
        for partition in reserved.difference(&staged_partitions) {
            backend.release(partition);
        }
        self.teardown
            .retired_partitions
            .extend(staged_partitions.iter().cloned());
        self.teardown
            .released_partitions
            .extend(reserved.difference(&staged_partitions).cloned());

        self.failure = Some(failure.clone());
        self.state = TransactionState::Aborted(failure.clone());
        self.decision = Some(TransactionDecision::Aborted { failure });
        self.timings.finalize_nanos = start.elapsed().as_nanos() as u64;
        let receipt = self.build_receipt();
        self.receipt = Some(receipt.clone());
        Ok(receipt)
    }

    /// Record a failure and move the state machine to `Failed` (first failure
    /// wins).
    fn fail(&mut self, failure: TransactionFailure) {
        if matches!(
            self.state,
            TransactionState::New | TransactionState::Prepared | TransactionState::Executing
        ) {
            self.failure = Some(failure.clone());
            self.state = TransactionState::Failed(failure);
        }
    }

    /// Re-query the backend for every snapshot event and record the completed
    /// ones (asynchronous joins).
    fn refresh_completed_events(&mut self, backend: &dyn DeviceExecutionBackend) {
        for operation in &self.operations {
            for event in operation.completed_events() {
                if backend.event_completed(&event) {
                    self.completed_events.insert(event);
                }
            }
        }
    }

    /// The boundary references whose events have not completed. Every
    /// boundary barrier must be completed by every participant partition;
    /// every boundary launch must be completed by its partition.
    fn missing_boundary_refs(&self) -> BTreeSet<BoundaryRef> {
        let mut missing = BTreeSet::new();
        for barrier in &self.commit_boundary.barriers {
            let operation = self.operations.iter().find(|operation| {
                matches!(operation,
                    TransactionOperation::Barrier { barrier_ref, .. }
                        if barrier_ref == barrier)
            });
            let Some(operation) = operation else {
                // Prepare validated boundary declaration; defensive only.
                missing.insert(BoundaryRef::Barrier(barrier.clone()));
                continue;
            };
            for partition in operation.partitions() {
                let event = OperationEvent::BarrierCompleted {
                    partition: partition.clone(),
                    barrier_ref: barrier.clone(),
                };
                if !self.completed_events.contains(&event) {
                    missing.insert(BoundaryRef::Barrier(barrier.clone()));
                }
            }
        }
        for launch in &self.commit_boundary.launches {
            let operation = self.operations.iter().find(|operation| {
                matches!(operation,
                    TransactionOperation::Launch { launch_ref, .. }
                        if launch_ref == launch)
            });
            let Some(operation) = operation else {
                // Prepare validated boundary declaration; defensive only.
                missing.insert(BoundaryRef::Launch(launch.clone()));
                continue;
            };
            for partition in operation.partitions() {
                let event = OperationEvent::LaunchCompleted {
                    partition: partition.clone(),
                    launch_ref: launch.clone(),
                };
                if !self.completed_events.contains(&event) {
                    missing.insert(BoundaryRef::Launch(launch.clone()));
                }
            }
        }
        missing
    }

    /// Assemble the final receipt from the recorded transaction state. Called
    /// at commit and abort only.
    fn build_receipt(&self) -> TransactionReceipt {
        TransactionReceipt {
            transaction_id: self.id.clone(),
            logical_distributed_plan_hash: self
                .bound_plan
                .logical_distributed_plan_hash()
                .to_owned(),
            bound_distributed_plan_hash: self
                .bound_plan
                .bound_distributed_plan_hash()
                .to_owned(),
            plan_receipt: self.bound_plan.receipt(),
            reservation_summary: self.reservation.clone().unwrap_or_default(),
            declared_write_set: self.declared_write_set.clone(),
            executed_operations: self.executed_operations.clone(),
            synchronization_events: self.completed_events.clone(),
            decision: self
                .decision
                .clone()
                .expect("the receipt is built only at commit or abort"),
            publish_summary: self.publish_summary,
            teardown: self.teardown.clone(),
            timings: self.timings,
            selected_transports: self.transport_receipt.clone(),
        }
    }
}

#[cfg(test)]
#[path = "execution_transaction_test.rs"]
mod tests;
