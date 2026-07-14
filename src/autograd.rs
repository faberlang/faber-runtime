//! Internal dense `Tensor<f32>` autograd graph boundary.
//!
//! This v0 is deliberately a tape/metadata scaffold, not a PyTorch-equivalent
//! runtime. It records contiguous/materialized leaf tensors and same-shape
//! `add`, `sub`, `mul`, and `summa` forward operations. It does not implement
//! backward, sessions, optimizers, host ABI gradient handles, broadcasting,
//! matmul, mutation, or view semantics.

#![allow(dead_code)]

use crate::Tensor;

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
    Broadcast,
    Matmul,
    Mutation,
    View,
    Backward,
    HostAbi,
    Session,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AutogradError {
    Tensor(&'static str),
    ShapeMismatch,
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
}

impl AutogradTape {
    pub(crate) fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub(crate) fn leaf(&mut self, tensor: Tensor<f32>) -> AutogradValue {
        self.record(AutogradOp::Leaf, Vec::new(), tensor)
    }

    pub(crate) fn add(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.same_shape_binary(lhs, rhs, AutogradOp::Add, Tensor::addita)
    }

    pub(crate) fn sub(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.same_shape_binary(lhs, rhs, AutogradOp::Sub, Tensor::subtrahe)
    }

    pub(crate) fn mul(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
    ) -> Result<AutogradValue, AutogradError> {
        self.same_shape_binary(lhs, rhs, AutogradOp::Mul, Tensor::multiplica)
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

    fn same_shape_binary(
        &mut self,
        lhs: &AutogradValue,
        rhs: &AutogradValue,
        op: AutogradOp,
        forward: fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>, &'static str>,
    ) -> Result<AutogradValue, AutogradError> {
        if lhs.tensor.magnitudines() != rhs.tensor.magnitudines() {
            return Err(AutogradError::ShapeMismatch);
        }
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
        self.nodes.push(AutogradNode {
            id,
            op,
            parents,
            shape,
        });
        AutogradValue { id, tensor }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32], shape: &[i64]) -> Tensor<f32> {
        Tensor::structa(values.to_vec(), shape).expect("test tensor shape matches")
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
    fn autograd_rejects_broadcast_shape_without_recording_node() {
        let mut tape = AutogradTape::new();
        let matrix = tape.leaf(tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let column = tape.leaf(tensor(&[10.0, 20.0], &[2, 1]));

        let before = tape.nodes().len();
        let err = tape.add(&matrix, &column).unwrap_err();

        assert_eq!(err, AutogradError::ShapeMismatch);
        assert_eq!(tape.nodes().len(), before);
    }

    #[test]
    fn autograd_v0_explicitly_rejects_out_of_scope_operations() {
        let tape = AutogradTape::new();

        assert_eq!(
            tape.reject_unsupported::<AutogradValue>(UnsupportedAutogradOp::Broadcast)
                .unwrap_err(),
            AutogradError::Unsupported(UnsupportedAutogradOp::Broadcast)
        );
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
            tape.reject_unsupported::<()>(UnsupportedAutogradOp::Backward),
            Err(AutogradError::Unsupported(UnsupportedAutogradOp::Backward))
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
}
