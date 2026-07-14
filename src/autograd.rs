//! Internal dense `Tensor<f32>` autograd graph boundary.
//!
//! This v0 is deliberately a tape/metadata scaffold, not a PyTorch-equivalent
//! runtime. It records contiguous/materialized leaf tensors and broadcast-aware
//! `add`, `sub`, `mul`, and `summa` forward operations, then runs a scalar-loss
//! backward pass for those same operations. Broadcasted binary operations reduce
//! parent gradients back to each original parent shape. It does not implement
//! sessions, optimizers, host ABI gradient handles, matmul, mutation, or view
//! semantics.

#![allow(dead_code)]

use crate::tensor::{tensor_flat_offset, tensor_shape_element_count};
use crate::Tensor;

type BinaryForward = fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AutogradNodeId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutogradOp {
    Leaf,
    Add,
    Sub,
    Mul,
    Summa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnsupportedAutogradOp {
    Matmul,
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
    BackwardRequiresScalar,
    Unsupported(UnsupportedAutogradOp),
}

#[derive(Clone, Debug)]
pub(crate) struct AutogradValue {
    id: AutogradNodeId,
    tensor: Tensor<f32>,
}

impl AutogradValue {
    pub(crate) fn id(&self) -> AutogradNodeId {
        self.id
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
        self.op
    }

    pub(crate) fn parents(&self) -> &[AutogradNodeId] {
        &self.parents
    }

    pub(crate) fn shape(&self) -> &[i64] {
        &self.shape
    }
}

#[derive(Default, Debug)]
pub(crate) struct AutogradTape {
    nodes: Vec<AutogradNode>,
    values: Vec<Tensor<f32>>,
}

impl AutogradTape {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            values: Vec::new(),
        }
    }

    pub(crate) fn leaf(&mut self, tensor: Tensor<f32>) -> AutogradValue {
        self.record(AutogradOp::Leaf, Vec::new(), tensor)
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

    pub(crate) fn summa(&mut self, value: &AutogradValue) -> Result<AutogradValue, AutogradError> {
        let tensor =
            Tensor::structa(vec![value.tensor.summa()], &[]).map_err(AutogradError::Tensor)?;
        Ok(self.record(AutogradOp::Summa, vec![value.id], tensor))
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

    pub(crate) fn backward(
        &self,
        loss: &AutogradValue,
    ) -> Result<AutogradGradients, AutogradError> {
        let loss_node = self.node(loss.id).ok_or(AutogradError::MissingNode)?;
        if !loss_node.shape.is_empty() {
            return Err(AutogradError::BackwardRequiresScalar);
        }

        let mut gradients = AutogradGradients::new(self.nodes.len());
        gradients.accumulate(loss.id, scalar_tensor(1.0)?)?;

        for node in self.nodes.iter().rev() {
            let Some(upstream) = gradients.gradient(node.id).cloned() else {
                continue;
            };

            match node.op {
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
        let tensor = forward(&lhs.tensor, &rhs.tensor).map_err(AutogradError::Tensor)?;
        Ok(self.record(op, vec![lhs.id, rhs.id], tensor))
    }

    fn record(
        &mut self,
        op: AutogradOp,
        parents: Vec<AutogradNodeId>,
        tensor: Tensor<f32>,
    ) -> AutogradValue {
        let id = AutogradNodeId(self.nodes.len());
        let shape = tensor.magnitudines();
        self.values.push(tensor.clone());
        self.nodes.push(AutogradNode {
            id,
            op,
            parents,
            shape,
        });
        AutogradValue { id, tensor }
    }

    fn value(&self, id: AutogradNodeId) -> Result<&Tensor<f32>, AutogradError> {
        self.values.get(id.0).ok_or(AutogradError::MissingNode)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AutogradGradients {
    gradients: Vec<Option<Tensor<f32>>>,
}

impl AutogradGradients {
    fn new(len: usize) -> Self {
        Self {
            gradients: vec![None; len],
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32], shape: &[i64]) -> Tensor<f32> {
        Tensor::structa(values.to_vec(), shape).expect("test tensor shape matches")
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

        let x = tape.leaf(tensor(&[1.0, 2.0], &[2]));
        let w = tape.leaf(tensor(&[3.0, 4.0], &[2]));

        assert_eq!(x.id(), AutogradNodeId(0));
        assert_eq!(w.id(), AutogradNodeId(1));
        assert_eq!(tape.nodes().len(), 2);
        assert_eq!(tape.node(x.id()).expect("x node").op(), AutogradOp::Leaf);
        assert_eq!(tape.node(w.id()).expect("w node").shape(), &[2]);
        assert!(tape.node(w.id()).expect("w node").parents().is_empty());
    }

    #[test]
    fn autograd_records_same_shape_op_tags_parent_edges_and_forward_values() {
        let mut tape = AutogradTape::new();
        let x = tape.leaf(tensor(&[1.0, 2.0], &[2]));
        let w = tape.leaf(tensor(&[3.0, 4.0], &[2]));

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
        let prediction = tape.leaf(tensor(&[4.0, 8.0], &[2]));
        let target = tape.leaf(tensor(&[1.0, 3.0], &[2]));

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
        let matrix = tape.leaf(tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let column = tape.leaf(tensor(&[10.0, 20.0], &[2, 1]));

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
        let matrix = tape.leaf(tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let vector = tape.leaf(tensor(&[10.0, 20.0, 30.0], &[3]));

        let before = tape.nodes().len();
        let err = tape.add(&matrix, &vector).unwrap_err();

        assert_eq!(
            err,
            AutogradError::Tensor(crate::tensor::ERR_BROADCAST_SHAPE)
        );
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_v0_explicitly_rejects_out_of_scope_operations() {
        let tape = AutogradTape::new();

        assert_eq!(
            tape.reject_unsupported::<AutogradValue>(UnsupportedAutogradOp::Matmul)
                .unwrap_err(),
            AutogradError::Unsupported(UnsupportedAutogradOp::Matmul)
        );
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
    fn backward_accumulates_duplicate_parent_for_rank_zero_square_plus_self() {
        let mut tape = AutogradTape::new();
        let x = tape.leaf(tensor(&[1.75], &[]));
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
        let x = tape.leaf(tensor(&[2.0], &[]));
        let weight = tape.leaf(tensor(&[3.0], &[]));
        let target = tape.leaf(tensor(&[4.0], &[]));

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
        let x = tape.leaf(tensor(&x_values, &[3]));
        let w = tape.leaf(tensor(&w_values, &[3]));
        let target = tape.leaf(tensor(&target_values, &[3]));

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
        let x = tape.leaf(tensor(&x_values, &[2, 2]));
        let bias = tape.leaf(tensor(&bias_values, &[2, 1]));
        let target = tape.leaf(tensor(&target_values, &[2, 2]));

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
        let x = tape.leaf(tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let scale = tape.leaf(tensor(&[10.0, -2.0], &[2, 1]));

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
        let x = tape.leaf(tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let bias = tape.leaf(tensor(&[10.0, 20.0], &[2, 1]));

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
        let x = tape.leaf(tensor(&[1.0, 2.0], &[2]));
        let w = tape.leaf(tensor(&[3.0, 4.0], &[2]));
        let product = tape.mul(&x, &w).expect("same-shape product");

        assert_eq!(
            tape.backward(&product).unwrap_err(),
            AutogradError::BackwardRequiresScalar
        );
    }
}
