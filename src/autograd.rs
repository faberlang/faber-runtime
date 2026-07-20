//! Internal dense `Tensor<f32>` autograd graph boundary.
//!
//! This v0 is deliberately a tape/metadata scaffold, not a PyTorch-equivalent
//! runtime. It records contiguous/materialized leaf tensors and broadcast-aware
//! `add`, `sub`, `mul`, checked finite `divide`, tape-owned `neg`, rank-2
//! `matmul`, tape-owned scalar `scala`, `forma`, materialized `permute`,
//! axis-0 `sectio`, `summa`, and `media` forward operations, then runs a
//! scalar-loss backward pass for those same operations. Broadcasted binary
//! operations reduce parent gradients back to each original parent shape. Raw
//! `Tensor::sectio` views are rejected at the leaf boundary; tape-owned
//! `sectio` records parent identity and scatter-adds gradients back to the
//! parent. It does not implement sessions, optimizers, host ABI gradient
//! handles, or mutation semantics.

#![allow(dead_code)]

use crate::tensor::{tensor_flat_offset, tensor_shape_element_count};
use crate::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};

type BinaryForward = fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>, &'static str>;

static NEXT_AUTOGRAD_TAPE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AutogradTapeId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AutogradNodeId(usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AutogradOp {
    Leaf,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Matmul,
    Scala { factor: u32 },
    Forma,
    Permute { axes: Vec<i64> },
    Sectio { start: i64 },
    Summa,
    Media,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnsupportedAutogradOp {
    Mutation,
    View,
    HostAbi,
    Session,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AutogradError {
    Tensor(&'static str),
    ShapeMismatch,
    MissingNode,
    ForeignTapeValue,
    BackwardRequiresScalar,
    Unsupported(UnsupportedAutogradOp),
}

#[derive(Clone, Debug)]
pub(crate) struct AutogradValue {
    id: AutogradNodeId,
    tape_id: AutogradTapeId,
    tensor: Tensor<f32>,
}

impl AutogradValue {
    pub(crate) fn id(&self) -> AutogradNodeId {
        self.id
    }

    pub(crate) fn tape_id(&self) -> AutogradTapeId {
        self.tape_id
    }

    pub(crate) fn tensor(&self) -> &Tensor<f32> {
        &self.tensor
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AutogradNode {
    id: AutogradNodeId,
    op: AutogradOp,
    parents: Vec<AutogradNodeId>,
    shape: Vec<i64>,
}

impl AutogradNode {
    pub(crate) fn id(&self) -> AutogradNodeId {
        self.id
    }

    pub(crate) fn op(&self) -> AutogradOp {
        self.op.clone()
    }

    pub(crate) fn parents(&self) -> &[AutogradNodeId] {
        &self.parents
    }

    pub(crate) fn shape(&self) -> &[i64] {
        &self.shape
    }
}

#[derive(Debug)]
pub(crate) struct AutogradTape {
    id: AutogradTapeId,
    nodes: Vec<AutogradNode>,
    values: Vec<Tensor<f32>>,
}

impl Default for AutogradTape {
    fn default() -> Self {
        Self::new()
    }
}

impl AutogradTape {
    pub(crate) fn new() -> Self {
        Self {
            id: next_autograd_tape_id(),
            nodes: Vec::new(),
            values: Vec::new(),
        }
    }

    pub(crate) fn leaf(&mut self, tensor: Tensor<f32>) -> Result<AutogradValue, AutogradError> {
        if tensor.is_view() {
            return Err(AutogradError::Unsupported(UnsupportedAutogradOp::View));
        }
        Ok(self.record(AutogradOp::Leaf, Vec::new(), tensor))
    }

    pub(crate) fn add(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.binary(lhs, rhs, AutogradOp::Add, Tensor::addita)
    }

    pub(crate) fn sub(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.binary(lhs, rhs, AutogradOp::Sub, Tensor::subtrahe)
    }

    pub(crate) fn mul(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.binary(lhs, rhs, AutogradOp::Mul, Tensor::multiplica)
    }

    pub(crate) fn divide(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.binary(lhs, rhs, AutogradOp::Div, Tensor::divide)
    }

    pub(crate) fn neg(&mut self, value: &AutogradValue) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = self.value(value.id)?.neg();
        Ok(self.record(AutogradOp::Neg, vec![value.id], tensor))
    }

    pub(crate) fn matmul(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(lhs)?;
        self.ensure_member(rhs)?;
        let tensor = self
            .value(lhs.id)?
            .matmul(self.value(rhs.id)?)
            .map_err(AutogradError::Tensor)?;
        Ok(self.record(AutogradOp::Matmul, vec![lhs.id, rhs.id], tensor))
    }

    pub(crate) fn scala(
        &mut self,
        value: &AutogradValue,
        factor: f32,
    ) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = self.value(value.id)?.scala(factor);
        Ok(self.record(
            AutogradOp::Scala {
                factor: factor.to_bits(),
            },
            vec![value.id],
            tensor,
        ))
    }

    pub(crate) fn forma(
        &mut self,
        value: &AutogradValue,
        shape: &[i64],
    ) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = self
            .value(value.id)?
            .forma(shape)
            .map_err(AutogradError::Tensor)?;
        Ok(self.record(AutogradOp::Forma, vec![value.id], tensor))
    }

    pub(crate) fn permute(
        &mut self,
        value: &AutogradValue,
        axes: &[i64],
    ) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = self
            .value(value.id)?
            .permute(axes)
            .map_err(AutogradError::Tensor)?;
        Ok(self.record(
            AutogradOp::Permute {
                axes: axes.to_vec(),
            },
            vec![value.id],
            tensor,
        ))
    }

    pub(crate) fn sectio(
        &mut self,
        value: &AutogradValue,
        start: i64,
        end: i64,
    ) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = self
            .value(value.id)?
            .sectio(start, end)
            .map_err(AutogradError::Tensor)?
            .materialize();
        Ok(self.record(AutogradOp::Sectio { start }, vec![value.id], tensor))
    }

    pub(crate) fn summa(&mut self, value: &AutogradValue) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = Tensor::structa(vec![self.value(value.id)?.summa()], &[])
            .map_err(AutogradError::Tensor)?;
        Ok(self.record(AutogradOp::Summa, vec![value.id], tensor))
    }

    pub(crate) fn media(&mut self, value: &AutogradValue) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(value)?;
        let tensor = Tensor::structa(
            vec![self
                .value(value.id)?
                .media()
                .map_err(AutogradError::Tensor)?],
            &[],
        )
        .map_err(AutogradError::Tensor)?;
        Ok(self.record(AutogradOp::Media, vec![value.id], tensor))
    }

    pub(crate) fn reject_unsupported<T>(
        &self,
        op: UnsupportedAutogradOp,
    ) -> Result<T, AutogradError> {
        Err(AutogradError::Unsupported(op))
    }

    pub(crate) fn nodes(&self) -> &[AutogradNode] {
        &self.nodes
    }

    pub(crate) fn node(&self, id: AutogradNodeId) -> Option<&AutogradNode> {
        self.nodes.get(id.0)
    }

    pub(crate) fn id(&self) -> AutogradTapeId {
        self.id
    }

    pub(crate) fn backward(
        &self,
        loss: &AutogradValue,
    ) -> Result<AutogradGradients, AutogradError> {
        self.ensure_member(loss)?;
        let loss_node = self.node(loss.id).ok_or(AutogradError::MissingNode)?;
        if !loss_node.shape.is_empty() {
            return Err(AutogradError::BackwardRequiresScalar);
        }

        let mut gradients =
            AutogradGradients::new(self.nodes.iter().map(|node| node.shape.clone()).collect());
        gradients.accumulate(loss.id, scalar_tensor(1.0)?)?;

        for node in self.nodes.iter().rev() {
            let Some(upstream) = gradients.gradient(node.id).cloned() else {
                continue;
            };

            match node.op.clone() {
                AutogradOp::Leaf => {}
                AutogradOp::Add => {
                    let &[lhs, rhs] = parent_pair(node)?;
                    let lhs_shape = parent_shape(self, lhs)?;
                    let rhs_shape = parent_shape(self, rhs)?;
                    gradients.accumulate(lhs, reduce_broadcast_gradient(&upstream, &lhs_shape)?)?;
                    gradients.accumulate(rhs, reduce_broadcast_gradient(&upstream, &rhs_shape)?)?;
                }
                AutogradOp::Sub => {
                    let &[lhs, rhs] = parent_pair(node)?;
                    let lhs_shape = parent_shape(self, lhs)?;
                    let rhs_shape = parent_shape(self, rhs)?;
                    gradients.accumulate(lhs, reduce_broadcast_gradient(&upstream, &lhs_shape)?)?;
                    gradients.accumulate(
                        rhs,
                        reduce_broadcast_gradient(&scale_tensor(&upstream, -1.0)?, &rhs_shape)?,
                    )?;
                }
                AutogradOp::Mul => {
                    let &[lhs, rhs] = parent_pair(node)?;
                    let lhs_value = self.value(lhs)?;
                    let rhs_value = self.value(rhs)?;
                    let lhs_shape = parent_shape(self, lhs)?;
                    let rhs_shape = parent_shape(self, rhs)?;
                    gradients.accumulate(
                        lhs,
                        reduce_broadcast_gradient(
                            &upstream
                                .multiplica(rhs_value)
                                .map_err(AutogradError::Tensor)?,
                            &lhs_shape,
                        )?,
                    )?;
                    gradients.accumulate(
                        rhs,
                        reduce_broadcast_gradient(
                            &upstream
                                .multiplica(lhs_value)
                                .map_err(AutogradError::Tensor)?,
                            &rhs_shape,
                        )?,
                    )?;
                }
                AutogradOp::Div => {
                    let &[lhs, rhs] = parent_pair(node)?;
                    let lhs_value = self.value(lhs)?;
                    let rhs_value = self.value(rhs)?;
                    let lhs_shape = parent_shape(self, lhs)?;
                    let rhs_shape = parent_shape(self, rhs)?;
                    gradients.accumulate(
                        lhs,
                        reduce_broadcast_gradient(
                            &upstream.divide(rhs_value).map_err(AutogradError::Tensor)?,
                            &lhs_shape,
                        )?,
                    )?;
                    let quotient = lhs_value.divide(rhs_value).map_err(AutogradError::Tensor)?;
                    let numerator = upstream
                        .multiplica(&quotient)
                        .map_err(AutogradError::Tensor)?;
                    gradients.accumulate(
                        rhs,
                        reduce_broadcast_gradient(
                            &scale_tensor(&numerator, -1.0)?
                                .divide(rhs_value)
                                .map_err(AutogradError::Tensor)?,
                            &rhs_shape,
                        )?,
                    )?;
                }
                AutogradOp::Neg => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    gradients.accumulate(parent, upstream.neg())?;
                }
                AutogradOp::Matmul => {
                    let &[lhs, rhs] = parent_pair(node)?;
                    let lhs_value = self.value(lhs)?;
                    let rhs_value = self.value(rhs)?;
                    let rhs_transposed = transpose_rank2(rhs_value)?;
                    let lhs_transposed = transpose_rank2(lhs_value)?;
                    gradients.accumulate(
                        lhs,
                        upstream
                            .matmul(&rhs_transposed)
                            .map_err(AutogradError::Tensor)?,
                    )?;
                    gradients.accumulate(
                        rhs,
                        lhs_transposed
                            .matmul(&upstream)
                            .map_err(AutogradError::Tensor)?,
                    )?;
                }
                AutogradOp::Scala { factor } => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    gradients.accumulate(parent, upstream.scala(f32::from_bits(factor)))?;
                }
                AutogradOp::Forma => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    let parent_shape = parent_shape(self, parent)?;
                    gradients.accumulate(
                        parent,
                        upstream
                            .forma(&parent_shape)
                            .map_err(AutogradError::Tensor)?,
                    )?;
                }
                AutogradOp::Permute { axes } => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    let inverse_axes = inverse_permutation_axes(&axes)?;
                    gradients.accumulate(
                        parent,
                        upstream
                            .permute(&inverse_axes)
                            .map_err(AutogradError::Tensor)?,
                    )?;
                }
                AutogradOp::Sectio { start } => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    let parent_shape = parent_shape(self, parent)?;
                    gradients.accumulate(
                        parent,
                        scatter_axis0_gradient(&upstream, &parent_shape, start)?,
                    )?;
                }
                AutogradOp::Summa => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    let scalar = rank_zero_scalar(&upstream)?;
                    let parent_shape = self
                        .node(parent)
                        .ok_or(AutogradError::MissingNode)?
                        .shape
                        .clone();
                    gradients.accumulate(
                        parent,
                        Tensor::crea(&parent_shape, scalar).map_err(AutogradError::Tensor)?,
                    )?;
                }
                AutogradOp::Media => {
                    let &[parent] = node.parents.as_slice() else {
                        return Err(AutogradError::MissingNode);
                    };
                    let scalar = rank_zero_scalar(&upstream)?;
                    let parent_shape = parent_shape(self, parent)?;
                    let count = tensor_shape_element_count(&parent_shape)
                        .ok_or(AutogradError::ShapeMismatch)?;
                    if count == 0 {
                        return Err(AutogradError::Tensor(crate::tensor::ERR_MEDIA_EMPTY));
                    }
                    // SAFETY: element count is a tensor-size integer; f32 mean
                    // gradient uses IEEE precision intentionally.
                    #[allow(clippy::cast_precision_loss)]
                    let count_f = count as f32;
                    gradients.accumulate(
                        parent,
                        Tensor::crea(&parent_shape, scalar / count_f)
                            .map_err(AutogradError::Tensor)?,
                    )?;
                }
            }
        }

        Ok(gradients)
    }

    fn binary(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
        op: AutogradOp,
        forward: BinaryForward,
    ) -> Result<AutogradValue, AutogradError> {
        self.ensure_member(lhs)?;
        self.ensure_member(rhs)?;
        let tensor =
            forward(self.value(lhs.id)?, self.value(rhs.id)?).map_err(AutogradError::Tensor)?;
        Ok(self.record(op, vec![lhs.id, rhs.id], tensor))
    }

    fn record(
        &mut self,
        op: AutogradOp,
        parents: Vec<AutogradNodeId>,
        tensor: Tensor<f32>,
    ) -> AutogradValue {
        let id = AutogradNodeId(self.nodes.len());
        let saved_tensor = tensor.materialize();
        let value_tensor = saved_tensor.materialize();
        let shape = saved_tensor.magnitudines();
        self.values.push(saved_tensor);
        self.nodes.push(AutogradNode {
            id,
            op,
            parents,
            shape,
        });
        AutogradValue {
            id,
            tape_id: self.id,
            tensor: value_tensor,
        }
    }

    fn value(&self, id: AutogradNodeId) -> Result<&Tensor<f32>, AutogradError> {
        self.values.get(id.0).ok_or(AutogradError::MissingNode)
    }

    fn ensure_member(&self, value: &AutogradValue) -> Result<(), AutogradError> {
        if value.tape_id == self.id {
            Ok(())
        } else {
            Err(AutogradError::ForeignTapeValue)
        }
    }
}

fn next_autograd_tape_id() -> AutogradTapeId {
    AutogradTapeId(NEXT_AUTOGRAD_TAPE_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone, Debug)]
pub(crate) struct AutogradGradients {
    gradients: Vec<Option<Tensor<f32>>>,
    expected_shapes: Vec<Vec<i64>>,
}

impl AutogradGradients {
    fn new(expected_shapes: Vec<Vec<i64>>) -> Self {
        let len = expected_shapes.len();
        Self {
            gradients: vec![None; len],
            expected_shapes,
        }
    }

    pub(crate) fn gradient(&self, id: AutogradNodeId) -> Option<&Tensor<f32>> {
        self.gradients.get(id.0).and_then(Option::as_ref)
    }

    fn accumulate(
        &mut self,
        id: AutogradNodeId,
        gradient: Tensor<f32>,
    ) -> Result<(), AutogradError> {
        let expected_shape = self
            .expected_shapes
            .get(id.0)
            .ok_or(AutogradError::MissingNode)?;
        if gradient.magnitudines() != *expected_shape {
            return Err(AutogradError::ShapeMismatch);
        }
        let slot = self
            .gradients
            .get_mut(id.0)
            .ok_or(AutogradError::MissingNode)?;
        match slot {
            Some(existing) => {
                *existing = existing.addita(&gradient).map_err(AutogradError::Tensor)?;
            }
            None => *slot = Some(gradient),
        }
        Ok(())
    }
}

fn parent_pair(node: &AutogradNode) -> Result<&[AutogradNodeId; 2], AutogradError> {
    node.parents
        .as_slice()
        .try_into()
        .map_err(|_| AutogradError::MissingNode)
}

fn parent_shape(tape: &AutogradTape, id: AutogradNodeId) -> Result<Vec<i64>, AutogradError> {
    Ok(tape
        .node(id)
        .ok_or(AutogradError::MissingNode)?
        .shape
        .clone())
}

fn reduce_broadcast_gradient(
    upstream: &Tensor<f32>,
    target_shape: &[i64],
) -> Result<Tensor<f32>, AutogradError> {
    let upstream_shape = upstream.magnitudines();
    if broadcast_shape(target_shape, &upstream_shape)? != upstream_shape {
        return Err(AutogradError::ShapeMismatch);
    }

    let target_count =
        tensor_shape_element_count(target_shape).ok_or(AutogradError::ShapeMismatch)?;
    let mut data = vec![0.0_f32; target_count];

    for (ordinal, value) in upstream.planata().into_iter().enumerate() {
        let upstream_index = unravel_index(ordinal, &upstream_shape)?;
        let target_index = broadcast_parent_index(&upstream_index, target_shape)?;
        let offset =
            tensor_flat_offset(target_shape, &target_index).ok_or(AutogradError::ShapeMismatch)?;
        data[offset] += value;
    }

    Tensor::structa(data, target_shape).map_err(AutogradError::Tensor)
}

fn scatter_axis0_gradient(
    upstream: &Tensor<f32>,
    target_shape: &[i64],
    start: i64,
) -> Result<Tensor<f32>, AutogradError> {
    let upstream_shape = upstream.magnitudines();
    if start < 0
        || target_shape.is_empty()
        || upstream_shape.len() != target_shape.len()
        || upstream_shape
            .iter()
            .zip(target_shape.iter())
            .skip(1)
            .any(|(upstream_dim, target_dim)| upstream_dim != target_dim)
        || start
            .checked_add(*upstream_shape.first().ok_or(AutogradError::ShapeMismatch)?)
            .ok_or(AutogradError::ShapeMismatch)?
            > target_shape[0]
    {
        return Err(AutogradError::ShapeMismatch);
    }

    let target_count =
        tensor_shape_element_count(target_shape).ok_or(AutogradError::ShapeMismatch)?;
    let mut data = vec![0.0_f32; target_count];

    for (ordinal, value) in upstream.planata().into_iter().enumerate() {
        let mut target_index = unravel_index(ordinal, &upstream_shape)?;
        target_index[0] += start;
        let offset =
            tensor_flat_offset(target_shape, &target_index).ok_or(AutogradError::ShapeMismatch)?;
        data[offset] += value;
    }

    Tensor::structa(data, target_shape).map_err(AutogradError::Tensor)
}

fn broadcast_shape(lhs: &[i64], rhs: &[i64]) -> Result<Vec<i64>, AutogradError> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lhs_dim = broadcast_dim(lhs, rank, axis);
        let rhs_dim = broadcast_dim(rhs, rank, axis);
        let dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return Err(AutogradError::ShapeMismatch);
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn broadcast_dim(shape: &[i64], rank: usize, axis: usize) -> i64 {
    let pad = rank - shape.len();
    if axis < pad {
        1
    } else {
        shape[axis - pad]
    }
}

fn unravel_index(mut ordinal: usize, shape: &[i64]) -> Result<Vec<i64>, AutogradError> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let mut index = vec![0; shape.len()];
    for (axis, dim) in shape.iter().enumerate().rev() {
        let dim = usize::try_from(*dim).map_err(|_| AutogradError::ShapeMismatch)?;
        if dim == 0 {
            return Err(AutogradError::ShapeMismatch);
        }
        index[axis] = i64::try_from(ordinal % dim).map_err(|_| AutogradError::ShapeMismatch)?;
        ordinal /= dim;
    }
    Ok(index)
}

fn broadcast_parent_index(
    upstream_index: &[i64],
    target_shape: &[i64],
) -> Result<Vec<i64>, AutogradError> {
    if target_shape.len() > upstream_index.len() {
        return Err(AutogradError::ShapeMismatch);
    }
    let pad = upstream_index.len() - target_shape.len();
    Ok((0..target_shape.len())
        .map(|axis| {
            if target_shape[axis] == 1 {
                0
            } else {
                upstream_index[axis + pad]
            }
        })
        .collect())
}

fn scalar_tensor(value: f32) -> Result<Tensor<f32>, AutogradError> {
    Tensor::structa(vec![value], &[]).map_err(AutogradError::Tensor)
}

fn rank_zero_scalar(tensor: &Tensor<f32>) -> Result<f32, AutogradError> {
    if !tensor.magnitudines().is_empty() {
        return Err(AutogradError::BackwardRequiresScalar);
    }
    tensor
        .planata()
        .into_iter()
        .next()
        .ok_or(AutogradError::BackwardRequiresScalar)
}

fn scale_tensor(tensor: &Tensor<f32>, scalar: f32) -> Result<Tensor<f32>, AutogradError> {
    let data = tensor
        .planata()
        .into_iter()
        .map(|value| value * scalar)
        .collect();
    Tensor::structa(data, &tensor.magnitudines()).map_err(AutogradError::Tensor)
}

fn inverse_permutation_axes(axes: &[i64]) -> Result<Vec<i64>, AutogradError> {
    let rank = axes.len();
    let mut inverse = vec![0_i64; rank];
    let mut seen = vec![false; rank];
    for (output_axis, &input_axis) in axes.iter().enumerate() {
        let input_axis = usize::try_from(input_axis).map_err(|_| AutogradError::ShapeMismatch)?;
        if input_axis >= rank || seen[input_axis] {
            return Err(AutogradError::ShapeMismatch);
        }
        seen[input_axis] = true;
        inverse[input_axis] =
            i64::try_from(output_axis).map_err(|_| AutogradError::ShapeMismatch)?;
    }
    Ok(inverse)
}

fn transpose_rank2(tensor: &Tensor<f32>) -> Result<Tensor<f32>, AutogradError> {
    tensor.transpose_rank2().map_err(AutogradError::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32], shape: &[i64]) -> Tensor<f32> {
        Tensor::structa(values.to_vec(), shape).expect("test tensor shape matches")
    }

    fn leaf(tape: &mut AutogradTape, tensor: Tensor<f32>) -> AutogradValue {
        tape.leaf(tensor)
            .expect("materialized tensor is differentiable")
    }

    fn assert_tensor_close(actual: &Tensor<f32>, expected: &[f32], shape: &[i64]) {
        assert_eq!(actual.magnitudines(), shape);
        assert_eq!(actual.planata().len(), expected.len());
        for (index, (actual, expected)) in actual.planata().iter().zip(expected.iter()).enumerate()
        {
            let delta = (actual - expected).abs();
            assert!(
                delta <= 1.0e-5,
                "gradient[{index}] expected {expected}, got {actual}, delta {delta}"
            );
        }
    }

    #[test]
    fn autograd_leaf_ids_are_stable_and_record_shape() {
        let mut tape = AutogradTape::new();

        let x = leaf(&mut tape, tensor(&[1.0, 2.0], &[2]));
        let w = leaf(&mut tape, tensor(&[3.0, 4.0], &[2]));

        assert_eq!(x.id(), AutogradNodeId(0));
        assert_eq!(w.id(), AutogradNodeId(1));
        assert_eq!(tape.nodes().len(), 2);
        assert_eq!(tape.node(x.id()).expect("x node").op(), AutogradOp::Leaf);
        assert_eq!(tape.node(w.id()).expect("w node").shape(), &[2]);
        assert!(tape.node(w.id()).expect("w node").parents().is_empty());
    }

    #[test]
    fn autograd_values_are_stamped_with_their_tape_identity() {
        let mut lhs_tape = AutogradTape::new();
        let mut rhs_tape = AutogradTape::new();

        let lhs = leaf(&mut lhs_tape, tensor(&[1.0], &[]));
        let rhs = leaf(&mut rhs_tape, tensor(&[2.0], &[]));

        assert_eq!(lhs.tape_id(), lhs_tape.id());
        assert_eq!(rhs.tape_id(), rhs_tape.id());
        assert_ne!(lhs.tape_id(), rhs.tape_id());
    }

    #[test]
    fn gradient_accumulation_rejects_first_wrong_shape() {
        let mut gradients = AutogradGradients::new(vec![vec![2, 2]]);

        assert_eq!(
            gradients.accumulate(AutogradNodeId(0), tensor(&[1.0], &[1, 1])),
            Err(AutogradError::ShapeMismatch)
        );
        assert!(gradients.gradient(AutogradNodeId(0)).is_none());
    }

    #[test]
    fn gradient_accumulation_rejects_broadcast_compatible_wrong_shape() {
        let mut gradients = AutogradGradients::new(vec![vec![2, 2]]);

        gradients
            .accumulate(AutogradNodeId(0), tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
            .expect("initial exact-shape gradient");
        assert_eq!(
            gradients.accumulate(AutogradNodeId(0), tensor(&[10.0], &[1, 1])),
            Err(AutogradError::ShapeMismatch)
        );
        assert_tensor_close(
            gradients
                .gradient(AutogradNodeId(0))
                .expect("existing gradient remains unchanged"),
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
        );
    }

    #[test]
    fn autograd_rejects_cross_tape_binary_operands_without_recording_node() {
        let mut lhs_tape = AutogradTape::new();
        let mut rhs_tape = AutogradTape::new();
        let lhs = leaf(&mut lhs_tape, tensor(&[1.0, 2.0], &[2]));
        let rhs = leaf(&mut rhs_tape, tensor(&[3.0, 4.0], &[2]));

        let before = lhs_tape.nodes().len();
        assert_eq!(
            lhs_tape.add(&lhs, &rhs).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(
            lhs_tape.sub(&lhs, &rhs).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(
            lhs_tape.mul(&lhs, &rhs).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(lhs_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_rejects_cross_tape_matmul_operand_without_recording_node() {
        let mut lhs_tape = AutogradTape::new();
        let mut rhs_tape = AutogradTape::new();
        let lhs = leaf(&mut lhs_tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let rhs = leaf(&mut rhs_tape, tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

        let before = lhs_tape.nodes().len();
        assert_eq!(
            lhs_tape.matmul(&lhs, &rhs).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(lhs_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_rejects_cross_tape_summa_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.summa(&foreign).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn backward_rejects_cross_tape_loss_before_node_lookup() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let local = leaf(&mut local_tape, tensor(&[10.0], &[]));
        let _local_loss = local_tape.summa(&local).expect("local scalar loss");
        let foreign = leaf(&mut foreign_tape, tensor(&[2.0], &[]));
        let foreign_loss = foreign_tape.summa(&foreign).expect("foreign scalar loss");

        assert_eq!(
            local_tape.backward(&foreign_loss).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
    }

    #[test]
    fn autograd_records_same_shape_op_tags_parent_edges_and_forward_values() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[1.0, 2.0], &[2]));
        let w = leaf(&mut tape, tensor(&[3.0, 4.0], &[2]));

        let product = tape.mul(&x, &w).expect("same-shape mul records");
        let shifted = tape.add(&product, &x).expect("same-shape add records");
        let loss = tape.summa(&shifted).expect("summa records scalar");

        assert_eq!(product.tensor().planata(), vec![3.0, 8.0]);
        assert_eq!(shifted.tensor().planata(), vec![4.0, 10.0]);
        assert_eq!(loss.tensor().magnitudines(), Vec::<i64>::new());
        assert_eq!(loss.tensor().planata(), vec![14.0]);

        let product_node = tape.node(product.id()).expect("mul node");
        assert_eq!(product_node.op(), AutogradOp::Mul);
        assert_eq!(product_node.parents(), &[x.id(), w.id()]);

        let shifted_node = tape.node(shifted.id()).expect("add node");
        assert_eq!(shifted_node.op(), AutogradOp::Add);
        assert_eq!(shifted_node.parents(), &[product.id(), x.id()]);

        let loss_node = tape.node(loss.id()).expect("sum node");
        assert_eq!(loss_node.op(), AutogradOp::Summa);
        assert_eq!(loss_node.parents(), &[shifted.id()]);
    }

    #[test]
    fn autograd_records_subtraction_parent_order() {
        let mut tape = AutogradTape::new();
        let prediction = leaf(&mut tape, tensor(&[4.0, 8.0], &[2]));
        let target = leaf(&mut tape, tensor(&[1.0, 3.0], &[2]));

        let residual = tape
            .sub(&prediction, &target)
            .expect("same-shape sub records");

        assert_eq!(residual.tensor().planata(), vec![3.0, 5.0]);
        let node = tape.node(residual.id()).expect("sub node");
        assert_eq!(node.op(), AutogradOp::Sub);
        assert_eq!(node.parents(), &[prediction.id(), target.id()]);
    }

    #[test]
    fn autograd_records_broadcast_add_shape() {
        let mut tape = AutogradTape::new();
        let matrix = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let column = leaf(&mut tape, tensor(&[10.0, 20.0], &[2, 1]));

        let shifted = tape.add(&matrix, &column).expect("broadcast add records");

        assert_eq!(shifted.tensor().magnitudines(), vec![2, 2]);
        assert_eq!(shifted.tensor().planata(), vec![11.0, 12.0, 23.0, 24.0]);
        let node = tape.node(shifted.id()).expect("add node");
        assert_eq!(node.op(), AutogradOp::Add);
        assert_eq!(node.parents(), &[matrix.id(), column.id()]);
    }

    #[test]
    fn autograd_rejects_incompatible_broadcast_shape_without_recording_node() {
        let mut tape = AutogradTape::new();
        let matrix = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let vector = leaf(&mut tape, tensor(&[10.0, 20.0, 30.0], &[3]));

        let before = tape.nodes().len();
        let err = tape.add(&matrix, &vector).unwrap_err();

        assert_eq!(
            err,
            AutogradError::Tensor(crate::tensor::ERR_BROADCAST_SHAPE)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_records_rank2_matmul_op_tags_parent_edges_and_forward_values() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let rhs = leaf(
            &mut tape,
            tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
        );

        let product = tape.matmul(&lhs, &rhs).expect("rank-2 matmul records");

        assert_eq!(product.tensor().magnitudines(), vec![2, 2]);
        assert_eq!(product.tensor().planata(), vec![58.0, 64.0, 139.0, 154.0]);
        let node = tape.node(product.id()).expect("matmul node");
        assert_eq!(node.op(), AutogradOp::Matmul);
        assert_eq!(node.parents(), &[lhs.id(), rhs.id()]);
    }

    #[test]
    fn autograd_tape_owned_scala_records_parent_edge_factor_and_forward_value() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, -2.0, 3.5, 4.0], &[2, 2]));

        let scaled = tape.scala(&value, -0.5).expect("scala records");

        assert_eq!(scaled.tensor().magnitudines(), vec![2, 2]);
        assert_eq!(scaled.tensor().planata(), vec![-0.5, 1.0, -1.75, -2.0]);
        let node = tape.node(scaled.id()).expect("scala node");
        assert_eq!(
            node.op(),
            AutogradOp::Scala {
                factor: (-0.5_f32).to_bits()
            }
        );
        assert_eq!(node.parents(), &[value.id()]);
    }

    #[test]
    fn backward_scales_tape_owned_scala_gradient_by_factor() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, -2.0, 3.0], &[3]));
        let weights = leaf(&mut tape, tensor(&[5.0, 7.0, 11.0], &[3]));

        let scaled = tape.scala(&value, 0.25).expect("scala records");
        let product = tape.mul(&scaled, &weights).expect("same-shape product");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[1.25, 1.75, 2.75],
            &[3],
        );
        assert_tensor_close(
            gradients.gradient(weights.id()).expect("weights gradient"),
            &[0.25, -0.5, 0.75],
            &[3],
        );
    }

    #[test]
    fn backward_zero_scala_branch_does_not_mask_repeated_parent_gradient() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[2.0, -3.0], &[2]));
        let weights = leaf(&mut tape, tensor(&[5.0, 7.0], &[2]));

        let zero_scaled = tape.scala(&value, 0.0).expect("zero scala records");
        let direct = tape.mul(&value, &weights).expect("direct repeated use");
        let combined = tape.add(&zero_scaled, &direct).expect("same-shape add");
        let loss = tape.summa(&combined).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[5.0, 7.0],
            &[2],
        );
        assert_tensor_close(
            gradients.gradient(weights.id()).expect("weights gradient"),
            &[2.0, -3.0],
            &[2],
        );
    }

    #[test]
    fn autograd_tape_owned_scala_rejects_cross_tape_parent_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.scala(&foreign, 2.0).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_divide_records_parent_edges_and_forward_value() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[8.0, 18.0, -24.0, 40.0], &[2, 2]));
        let rhs = leaf(&mut tape, tensor(&[2.0, -4.0], &[2, 1]));

        let divided = tape.divide(&lhs, &rhs).expect("division records");

        assert_eq!(divided.tensor().magnitudines(), vec![2, 2]);
        assert_eq!(divided.tensor().planata(), vec![4.0, 9.0, 6.0, -10.0]);
        let node = tape.node(divided.id()).expect("divide node");
        assert_eq!(node.op(), AutogradOp::Div);
        assert_eq!(node.parents(), &[lhs.id(), rhs.id()]);
    }

    #[test]
    fn backward_divides_same_shape_gradients() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[8.0, -18.0, 24.0], &[3]));
        let rhs = leaf(&mut tape, tensor(&[2.0, -3.0, 4.0], &[3]));

        let divided = tape.divide(&lhs, &rhs).expect("same-shape division");
        let loss = tape.summa(&divided).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(lhs.id()).expect("lhs gradient"),
            &[0.5, -1.0 / 3.0, 0.25],
            &[3],
        );
        assert_tensor_close(
            gradients.gradient(rhs.id()).expect("rhs gradient"),
            &[-2.0, 2.0, -1.5],
            &[3],
        );
    }

    #[test]
    fn backward_reduces_broadcast_divide_denominator_gradient() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[8.0, 18.0, -24.0, 40.0], &[2, 2]));
        let rhs = leaf(&mut tape, tensor(&[2.0, -4.0], &[2, 1]));

        let divided = tape.divide(&lhs, &rhs).expect("row denominator broadcasts");
        let loss = tape.summa(&divided).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(lhs.id()).expect("lhs gradient"),
            &[0.5, 0.5, -0.25, -0.25],
            &[2, 2],
        );
        assert_tensor_close(
            gradients.gradient(rhs.id()).expect("rhs gradient"),
            &[-6.5, -1.0],
            &[2, 1],
        );
    }

    #[test]
    fn backward_divide_large_finite_denominator_avoids_rhs_square_overflow() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[1.0], &[]));
        let rhs = leaf(&mut tape, tensor(&[3.0e38], &[]));

        let divided = tape.divide(&lhs, &rhs).expect("finite forward division");
        assert!(divided.tensor().planata()[0].is_finite());
        let loss = tape.summa(&divided).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        let lhs_gradient = gradients
            .gradient(lhs.id())
            .expect("lhs gradient")
            .planata()[0];
        let rhs_gradient = gradients
            .gradient(rhs.id())
            .expect("rhs gradient")
            .planata()[0];
        assert!(lhs_gradient.is_finite());
        assert!(rhs_gradient.is_finite());
        assert_eq!(rhs_gradient, -0.0);
    }

    #[test]
    fn autograd_tape_owned_divide_rejects_forward_policy_failures_without_recording_node() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[1.0, 2.0], &[2]));
        let rhs = leaf(&mut tape, tensor(&[1.0, 0.0], &[2]));

        let before = tape.nodes().len();
        assert_eq!(
            tape.divide(&lhs, &rhs).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_DIVIDE_ZERO_DENOMINATOR)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_divide_rejects_cross_tape_operand_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let lhs = leaf(&mut local_tape, tensor(&[1.0, 2.0], &[2]));
        let foreign_rhs = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.divide(&lhs, &foreign_rhs).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_neg_records_parent_edge_and_forward_value() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, -2.0, 0.0, 4.5], &[2, 2]));

        let negated = tape.neg(&value).expect("neg records");

        assert_eq!(negated.tensor().magnitudines(), vec![2, 2]);
        assert_eq!(negated.tensor().planata(), vec![-1.0, 2.0, -0.0, -4.5]);
        let node = tape.node(negated.id()).expect("neg node");
        assert_eq!(node.op(), AutogradOp::Neg);
        assert_eq!(node.parents(), &[value.id()]);
    }

    #[test]
    fn backward_negates_tape_owned_neg_gradient() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, -2.0, 3.0], &[3]));
        let weights = leaf(&mut tape, tensor(&[5.0, 7.0, 11.0], &[3]));

        let negated = tape.neg(&value).expect("neg records");
        let product = tape.mul(&negated, &weights).expect("same-shape product");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[-5.0, -7.0, -11.0],
            &[3],
        );
        assert_tensor_close(
            gradients.gradient(weights.id()).expect("weights gradient"),
            &[-1.0, 2.0, -3.0],
            &[3],
        );
    }

    #[test]
    fn autograd_tape_owned_neg_rejects_cross_tape_parent_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.neg(&foreign).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_forma_records_parent_edge_and_forward_value() {
        let mut tape = AutogradTape::new();
        let matrix = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let vector = tape.forma(&matrix, &[4]).expect("reshape records");

        assert_eq!(vector.tensor().magnitudines(), vec![4]);
        assert_eq!(vector.tensor().planata(), vec![1.0, 2.0, 3.0, 4.0]);
        let node = tape.node(vector.id()).expect("forma node");
        assert_eq!(node.op(), AutogradOp::Forma);
        assert_eq!(node.parents(), &[matrix.id()]);
    }

    #[test]
    fn backward_reshapes_tape_owned_forma_gradient_into_parent_shape() {
        let mut tape = AutogradTape::new();
        let matrix = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let weights = leaf(&mut tape, tensor(&[5.0, 7.0, 11.0, 13.0], &[4]));

        let vector = tape.forma(&matrix, &[4]).expect("reshape records");
        let product = tape.mul(&vector, &weights).expect("same-shape product");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(matrix.id()).expect("matrix gradient"),
            &[5.0, 7.0, 11.0, 13.0],
            &[2, 2],
        );
        assert_tensor_close(
            gradients.gradient(weights.id()).expect("weights gradient"),
            &[1.0, 2.0, 3.0, 4.0],
            &[4],
        );
    }

    #[test]
    fn autograd_tape_owned_forma_rejects_invalid_shape_without_recording_node() {
        let mut tape = AutogradTape::new();
        let matrix = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let before = tape.nodes().len();
        assert_eq!(
            tape.forma(&matrix, &[3]).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_FORMA_RESHAPE_COUNT)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_forma_rejects_cross_tape_parent_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.forma(&foreign, &[2]).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_permute_records_parent_edge_axes_and_forward_value() {
        let mut tape = AutogradTape::new();
        let value = leaf(
            &mut tape,
            tensor(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                &[2, 3, 2],
            ),
        );

        let permuted = tape.permute(&value, &[2, 0, 1]).expect("permute records");

        assert_eq!(permuted.tensor().magnitudines(), vec![2, 2, 3]);
        assert_eq!(
            permuted.tensor().planata(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        );
        let node = tape.node(permuted.id()).expect("permute node");
        assert_eq!(
            node.op(),
            AutogradOp::Permute {
                axes: vec![2, 0, 1]
            }
        );
        assert_eq!(node.parents(), &[value.id()]);
    }

    #[test]
    fn backward_inverts_tape_owned_permute_gradient_into_parent_shape() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let weights = leaf(
            &mut tape,
            tensor(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]),
        );

        let permuted = tape.permute(&value, &[1, 0]).expect("permute records");
        let product = tape.mul(&permuted, &weights).expect("same-shape product");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[10.0, 30.0, 50.0, 20.0, 40.0, 60.0],
            &[2, 3],
        );
        assert_tensor_close(
            gradients.gradient(weights.id()).expect("weights gradient"),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            &[3, 2],
        );
    }

    #[test]
    fn autograd_tape_owned_permute_rejects_invalid_axes_without_recording_node() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let before = tape.nodes().len();
        assert_eq!(
            tape.permute(&value, &[0, 0]).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_PERMUTE_DUPLICATE_AXIS)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_permute_rejects_cross_tape_parent_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.permute(&foreign, &[0]).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_media_records_scalar_mean() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let mean = tape.media(&value).expect("mean records");

        assert_eq!(mean.tensor().magnitudines(), Vec::<i64>::new());
        assert_eq!(mean.tensor().planata(), vec![2.5]);
        let node = tape.node(mean.id()).expect("media node");
        assert_eq!(node.op(), AutogradOp::Media);
        assert_eq!(node.parents(), &[value.id()]);
    }

    #[test]
    fn backward_distributes_tape_owned_media_gradient_over_parent() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let mean = tape.media(&value).expect("mean records");
        let gradients = tape.backward(&mean).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[0.25, 0.25, 0.25, 0.25],
            &[2, 2],
        );
    }

    #[test]
    fn autograd_tape_owned_media_rejects_empty_without_recording_node() {
        let mut tape = AutogradTape::new();
        let empty = leaf(&mut tape, tensor(&[], &[0]));

        let before = tape.nodes().len();
        assert_eq!(
            tape.media(&empty).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_MEDIA_EMPTY)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_media_rejects_cross_tape_parent_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.media(&foreign).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_matmul_rejects_invalid_shapes_without_recording_node() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let rhs = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let before = tape.nodes().len();
        assert_eq!(
            tape.matmul(&lhs, &rhs).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_MATMUL_INNER_DIMENSION)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_v0_explicitly_rejects_out_of_scope_operations() {
        let tape = AutogradTape::new();

        assert_eq!(
            tape.reject_unsupported::<()>(UnsupportedAutogradOp::Mutation),
            Err(AutogradError::Unsupported(UnsupportedAutogradOp::Mutation))
        );
        assert_eq!(
            tape.reject_unsupported::<()>(UnsupportedAutogradOp::View),
            Err(AutogradError::Unsupported(UnsupportedAutogradOp::View))
        );
        assert_eq!(
            tape.reject_unsupported::<()>(UnsupportedAutogradOp::HostAbi),
            Err(AutogradError::Unsupported(UnsupportedAutogradOp::HostAbi))
        );
        assert_eq!(
            tape.reject_unsupported::<()>(UnsupportedAutogradOp::Session),
            Err(AutogradError::Unsupported(UnsupportedAutogradOp::Session))
        );
    }

    #[test]
    fn backward_matches_rank2_matmul_sum_vjp() {
        let mut tape = AutogradTape::new();
        let lhs = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let rhs = leaf(
            &mut tape,
            tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
        );

        let product = tape.matmul(&lhs, &rhs).expect("rank-2 product");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(lhs.id()).expect("lhs gradient"),
            &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0],
            &[2, 3],
        );
        assert_tensor_close(
            gradients.gradient(rhs.id()).expect("rhs gradient"),
            &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
            &[3, 2],
        );
    }

    #[test]
    fn backward_uses_mul_snapshots_after_post_capture_mutation() {
        let mut tape = AutogradTape::new();
        let mut x_source = tensor(&[2.0], &[]);
        let mut weight_source = tensor(&[3.0], &[]);
        let x = leaf(&mut tape, x_source.clone());
        let weight = leaf(&mut tape, weight_source.clone());

        let prediction = tape.mul(&x, &weight).expect("x * weight");
        let loss = tape.summa(&prediction).expect("scalar loss");
        x_source.reple(20.0);
        weight_source.reple(30.0);
        let mut x_value_alias = x.tensor().clone();
        x_value_alias.reple(200.0);

        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(gradients.gradient(x.id()).expect("x gradient"), &[3.0], &[]);
        assert_tensor_close(
            gradients.gradient(weight.id()).expect("weight gradient"),
            &[2.0],
            &[],
        );
    }

    #[test]
    fn autograd_ops_ignore_pre_op_mutation_of_exposed_value_tensor_clone() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[2.0], &[]));
        let weight = leaf(&mut tape, tensor(&[3.0], &[]));

        let mut x_value_alias = x.tensor().clone();
        x_value_alias.reple(20.0);
        let prediction = tape.mul(&x, &weight).expect("x * weight");
        let loss = tape.summa(&prediction).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(prediction.tensor(), &[6.0], &[]);
        assert_tensor_close(gradients.gradient(x.id()).expect("x gradient"), &[3.0], &[]);
        assert_tensor_close(
            gradients.gradient(weight.id()).expect("weight gradient"),
            &[2.0],
            &[],
        );
    }

    #[test]
    fn backward_uses_matmul_snapshots_after_post_capture_mutation() {
        let mut tape = AutogradTape::new();
        let mut lhs_source = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut rhs_source = tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let lhs = leaf(&mut tape, lhs_source.clone());
        let rhs = leaf(&mut tape, rhs_source.clone());

        let product = tape.matmul(&lhs, &rhs).expect("rank-2 product");
        let loss = tape.summa(&product).expect("scalar loss");
        lhs_source.reple(100.0);
        rhs_source.reple(200.0);
        let mut rhs_value_alias = rhs.tensor().clone();
        rhs_value_alias.reple(300.0);

        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(lhs.id()).expect("lhs gradient"),
            &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0],
            &[2, 3],
        );
        assert_tensor_close(
            gradients.gradient(rhs.id()).expect("rhs gradient"),
            &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
            &[3, 2],
        );
    }

    #[test]
    fn backward_accumulates_duplicate_parent_for_rank2_matmul() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let square = tape.matmul(&x, &x).expect("square matrix product");
        let loss = tape.summa(&square).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(x.id()).expect("x gradient"),
            &[7.0, 11.0, 9.0, 13.0],
            &[2, 2],
        );
    }

    #[test]
    fn backward_accumulates_duplicate_parent_for_rank_zero_square_plus_self() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[1.75], &[]));
        let square = tape.mul(&x, &x).expect("same-shape square");
        let shifted = tape.add(&square, &x).expect("same-shape add");
        let loss = tape.summa(&shifted).expect("scalar sum");

        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(x.id()).expect("x gradient"),
            &[2.0 * 1.75 + 1.0],
            &[],
        );
    }

    #[test]
    fn backward_matches_rung3_scalar_weight_gradient() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[2.0], &[]));
        let weight = leaf(&mut tape, tensor(&[3.0], &[]));
        let target = leaf(&mut tape, tensor(&[4.0], &[]));

        let prediction = tape.mul(&x, &weight).expect("x * weight");
        let residual = tape.sub(&prediction, &target).expect("prediction - target");
        let squared = tape.mul(&residual, &residual).expect("residual squared");
        let loss = tape.summa(&squared).expect("scalar loss");

        assert_eq!(loss.tensor().planata(), vec![4.0]);
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(weight.id()).expect("weight gradient"),
            &[8.0],
            &[],
        );
    }

    #[test]
    fn backward_matches_same_shape_vector_loss_gradients() {
        let mut tape = AutogradTape::new();
        let x_values = [0.5_f32, -1.0, 2.0];
        let w_values = [3.0_f32, -0.25, 0.75];
        let target_values = [1.0_f32, -2.0, 0.5];
        let x = leaf(&mut tape, tensor(&x_values, &[3]));
        let w = leaf(&mut tape, tensor(&w_values, &[3]));
        let target = leaf(&mut tape, tensor(&target_values, &[3]));

        let prediction = tape.mul(&x, &w).expect("x * w");
        let residual = tape.sub(&prediction, &target).expect("prediction - target");
        let squared = tape.mul(&residual, &residual).expect("residual squared");
        let loss = tape.summa(&squared).expect("vector loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        let residuals: Vec<f32> = x_values
            .iter()
            .zip(w_values.iter())
            .zip(target_values.iter())
            .map(|((x, w), target)| x * w - target)
            .collect();
        let expected_x: Vec<f32> = residuals
            .iter()
            .zip(w_values.iter())
            .map(|(residual, w)| 2.0 * residual * w)
            .collect();
        let expected_w: Vec<f32> = residuals
            .iter()
            .zip(x_values.iter())
            .map(|(residual, x)| 2.0 * residual * x)
            .collect();

        assert_tensor_close(
            gradients.gradient(x.id()).expect("x gradient"),
            &expected_x,
            &[3],
        );
        assert_tensor_close(
            gradients.gradient(w.id()).expect("w gradient"),
            &expected_w,
            &[3],
        );
    }

    #[test]
    fn backward_matches_broadcast_bias_oracle_gradient_reduction() {
        let mut tape = AutogradTape::new();
        let x_values = [0.5_f32, -1.0, 2.0, 4.0];
        let bias_values = [0.25_f32, -0.75];
        let target_values = [1.0_f32, -2.0, 0.5, 3.0];
        let x = leaf(&mut tape, tensor(&x_values, &[2, 2]));
        let bias = leaf(&mut tape, tensor(&bias_values, &[2, 1]));
        let target = leaf(&mut tape, tensor(&target_values, &[2, 2]));

        let prediction = tape.add(&x, &bias).expect("row bias broadcasts");
        let residual = tape.sub(&prediction, &target).expect("prediction - target");
        let squared = tape.mul(&residual, &residual).expect("residual squared");
        let loss = tape.summa(&squared).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        let residuals: Vec<f32> = x_values
            .chunks_exact(2)
            .zip(bias_values.iter())
            .flat_map(|(row, bias)| row.iter().map(move |x| x + bias))
            .zip(target_values.iter())
            .map(|(prediction, target)| prediction - target)
            .collect();
        let expected_x: Vec<f32> = residuals.iter().map(|residual| 2.0 * residual).collect();
        let expected_bias: Vec<f32> = residuals
            .chunks_exact(2)
            .map(|row| row.iter().map(|residual| 2.0 * residual).sum::<f32>())
            .collect();

        assert_tensor_close(
            gradients.gradient(x.id()).expect("x gradient"),
            &expected_x,
            &[2, 2],
        );
        assert_tensor_close(
            gradients.gradient(bias.id()).expect("bias gradient"),
            &expected_bias,
            &[2, 1],
        );
    }

    #[test]
    fn backward_reduces_broadcast_mul_with_opposite_operand() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let scale = leaf(&mut tape, tensor(&[10.0, -2.0], &[2, 1]));

        let product = tape.mul(&x, &scale).expect("row scale broadcasts");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(x.id()).expect("x gradient"),
            &[10.0, 10.0, -2.0, -2.0],
            &[2, 2],
        );
        assert_tensor_close(
            gradients.gradient(scale.id()).expect("scale gradient"),
            &[3.0, 7.0],
            &[2, 1],
        );
    }

    #[test]
    fn backward_reduces_broadcast_sub_with_rhs_negative_sign() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let bias = leaf(&mut tape, tensor(&[10.0, 20.0], &[2, 1]));

        let residual = tape.sub(&x, &bias).expect("row bias broadcasts");
        let loss = tape.summa(&residual).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(x.id()).expect("x gradient"),
            &[1.0, 1.0, 1.0, 1.0],
            &[2, 2],
        );
        assert_tensor_close(
            gradients.gradient(bias.id()).expect("bias gradient"),
            &[-2.0, -2.0],
            &[2, 1],
        );
    }

    #[test]
    fn backward_reduces_scalar_broadcast_after_permute() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let scalar = leaf(&mut tape, tensor(&[10.0], &[]));

        let permuted = tape.permute(&value, &[1, 0]).expect("permute records");
        let shifted = tape.add(&permuted, &scalar).expect("scalar broadcasts");
        let loss = tape.summa(&shifted).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[2, 3],
        );
        assert_tensor_close(
            gradients.gradient(scalar.id()).expect("scalar gradient"),
            &[6.0],
            &[],
        );
    }

    #[test]
    fn backward_reduces_vector_broadcast_mul_after_permute() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let scale = leaf(&mut tape, tensor(&[10.0, 20.0], &[2]));

        let permuted = tape.permute(&value, &[1, 0]).expect("permute records");
        let product = tape.mul(&permuted, &scale).expect("vector broadcasts");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            &[2, 3],
        );
        assert_tensor_close(
            gradients.gradient(scale.id()).expect("scale gradient"),
            &[6.0, 15.0],
            &[2],
        );
    }

    #[test]
    fn backward_reduces_tensor_broadcast_sub_after_permute() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let bias = leaf(&mut tape, tensor(&[10.0, 20.0, 30.0], &[3, 1]));

        let permuted = tape.permute(&value, &[1, 0]).expect("permute records");
        let residual = tape.sub(&permuted, &bias).expect("column broadcasts");
        let loss = tape.summa(&residual).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(value.id()).expect("value gradient"),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[2, 3],
        );
        assert_tensor_close(
            gradients.gradient(bias.id()).expect("bias gradient"),
            &[-2.0, -2.0, -2.0],
            &[3, 1],
        );
    }

    #[test]
    fn autograd_rejects_incompatible_broadcast_after_permute_without_recording_node() {
        let mut tape = AutogradTape::new();
        let value = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let incompatible = leaf(&mut tape, tensor(&[10.0, 20.0, 30.0, 40.0], &[4]));
        let permuted = tape.permute(&value, &[1, 0]).expect("permute records");

        let before = tape.nodes().len();
        assert_eq!(
            tape.add(&permuted, &incompatible).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_BROADCAST_SHAPE)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn broadcast_gradient_reduction_rejects_unsupported_shape_mismatch() {
        let upstream = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        assert_eq!(
            reduce_broadcast_gradient(&upstream, &[2, 3]).unwrap_err(),
            AutogradError::ShapeMismatch
        );
    }

    #[test]
    fn backward_rejects_non_scalar_output_seed() {
        let mut tape = AutogradTape::new();
        let x = leaf(&mut tape, tensor(&[1.0, 2.0], &[2]));
        let w = leaf(&mut tape, tensor(&[3.0, 4.0], &[2]));
        let product = tape.mul(&x, &w).expect("same-shape product");

        assert_eq!(
            tape.backward(&product).unwrap_err(),
            AutogradError::BackwardRequiresScalar
        );
    }

    #[test]
    fn autograd_tape_owned_sectio_records_parent_edge_and_forward_value() {
        let mut tape = AutogradTape::new();
        let base = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));

        let slice = tape.sectio(&base, 1, 3).expect("valid tape-owned slice");

        assert_eq!(slice.tensor().magnitudines(), vec![2, 2]);
        assert_eq!(slice.tensor().planata(), vec![3.0, 4.0, 5.0, 6.0]);
        let node = tape.node(slice.id()).expect("sectio node");
        assert_eq!(node.op(), AutogradOp::Sectio { start: 1 });
        assert_eq!(node.parents(), &[base.id()]);
    }

    #[test]
    fn backward_scatter_adds_tape_owned_sectio_gradient_into_parent() {
        let mut tape = AutogradTape::new();
        let base = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let weight = leaf(&mut tape, tensor(&[5.0, 7.0], &[1, 2]));

        let slice = tape.sectio(&base, 0, 1).expect("valid tape-owned slice");
        let product = tape.mul(&slice, &weight).expect("same-shape product");
        let loss = tape.summa(&product).expect("scalar loss");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(base.id()).expect("base gradient"),
            &[5.0, 7.0, 0.0, 0.0],
            &[2, 2],
        );
        assert_tensor_close(
            gradients.gradient(weight.id()).expect("weight gradient"),
            &[1.0, 2.0],
            &[1, 2],
        );
    }

    #[test]
    fn backward_accumulates_overlapping_tape_owned_sectio_gradients() {
        let mut tape = AutogradTape::new();
        let base = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let first_row = tape.sectio(&base, 0, 1).expect("first row slice");
        let all_rows = tape.sectio(&base, 0, 2).expect("all rows slice");
        let first_loss = tape.summa(&first_row).expect("first row sum");
        let all_loss = tape.summa(&all_rows).expect("all rows sum");
        let loss = tape.add(&first_loss, &all_loss).expect("scalar add");
        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(base.id()).expect("base gradient"),
            &[2.0, 2.0, 1.0, 1.0],
            &[2, 2],
        );
    }

    #[test]
    fn autograd_tape_owned_sectio_rejects_invalid_bounds_without_recording_node() {
        let mut tape = AutogradTape::new();
        let base = leaf(&mut tape, tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

        let before = tape.nodes().len();
        assert_eq!(
            tape.sectio(&base, -1, 1).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_NEGATIVE_SLICE)
        );
        assert_eq!(
            tape.sectio(&base, 0, 3).unwrap_err(),
            AutogradError::Tensor(crate::tensor::ERR_INDEX_OUT_OF_BOUNDS)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_tape_owned_sectio_rejects_cross_tape_parent_without_recording_node() {
        let mut local_tape = AutogradTape::new();
        let mut foreign_tape = AutogradTape::new();
        let foreign = leaf(&mut foreign_tape, tensor(&[1.0, 2.0], &[2]));

        let before = local_tape.nodes().len();
        assert_eq!(
            local_tape.sectio(&foreign, 0, 1).unwrap_err(),
            AutogradError::ForeignTapeValue
        );
        assert_eq!(local_tape.nodes().len(), before);
    }

    #[test]
    fn autograd_rejects_raw_sectio_view_leaf() {
        let mut tape = AutogradTape::new();
        let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let view = base.sectio(0, 1).expect("valid view");

        assert_eq!(
            tape.leaf(view).unwrap_err(),
            AutogradError::Unsupported(UnsupportedAutogradOp::View)
        );
        assert!(tape.nodes().is_empty());
    }

    #[test]
    fn autograd_accepts_materialized_copy_of_sectio_view() {
        let mut tape = AutogradTape::new();
        let base = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let materialized = base.sectio(0, 1).expect("valid view").materialize();

        let value = tape.leaf(materialized).expect("materialized copy is leaf");

        assert_eq!(value.tensor().magnitudines(), vec![1, 2]);
        assert_eq!(tape.nodes().len(), 1);
    }

    #[test]
    fn autograd_materialized_sectio_snapshot_ignores_parent_alias_mutation() {
        let mut tape = AutogradTape::new();
        let mut base = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let slice = leaf(
            &mut tape,
            base.sectio(0, 1).expect("valid view").materialize(),
        );
        let weight = leaf(&mut tape, tensor(&[5.0, 7.0], &[1, 2]));

        let product = tape.mul(&slice, &weight).expect("same-shape product");
        let loss = tape.summa(&product).expect("scalar loss");
        base.ponde(&[0, 0], 100.0).expect("parent write succeeds");
        base.ponde(&[0, 1], 200.0).expect("parent write succeeds");

        let gradients = tape.backward(&loss).expect("backward succeeds");

        assert_tensor_close(
            gradients.gradient(slice.id()).expect("slice gradient"),
            &[5.0, 7.0],
            &[1, 2],
        );
        assert_tensor_close(
            gradients.gradient(weight.id()).expect("weight gradient"),
            &[1.0, 2.0],
            &[1, 2],
        );
    }
}
