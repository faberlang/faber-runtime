//! Dense numeric tensor runtime for generated Rust code.

use std::fmt::Debug;
use std::sync::{Arc, Mutex, MutexGuard};

/// Homogeneous numeric buffer with runtime shape metadata.
#[derive(Clone, Debug)]
pub struct Tensor<T> {
    data: Arc<Mutex<Vec<T>>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    view: bool,
}

pub const ERR_NEGATIVE_DIM: &str = "tensor shape dimension must be non-negative";
pub const ERR_NEGATIVE_INDEX: &str = "tensor index must be non-negative";
pub const ERR_NEGATIVE_SLICE: &str = "tensor slice bounds must be non-negative";
pub const ERR_INVALID_SLICE_RANGE: &str = "tensor slice end must be at least start";
pub const ERR_INDEX_OUT_OF_BOUNDS: &str = "tensor index out of bounds";
pub const ERR_ELEMENT_COUNT_OVERFLOW: &str = "tensor element count overflow";

pub const ERR_FORMA_RESHAPE_COUNT: &str = "tensor forma (reshape) element count mismatch";
pub const ERR_FORMA_ELEMENT_COUNT: &str = "tensor forma element count mismatch";
pub const ERR_ACCIPE_INVALID_INDEX: &str = "tensor accipe invalid index";
pub const ERR_PONDE_INVALID_INDEX: &str = "tensor ponde invalid index";
pub const ERR_CREA_INVALID_SHAPE: &str = "tensor crea invalid shape";
pub const ERR_SECTIO_INVALID_SLICE_BOUNDS: &str = "tensor sectio invalid slice bounds";
pub const ERR_BROADCAST_SHAPE: &str = "tensor broadcast shape mismatch";
pub const ERR_MATMUL_RECEIVER_RANK: &str = "tensor matmul requires rank-2 tensor receiver";
pub const ERR_MATMUL_ARGUMENT_RANK: &str = "tensor matmul requires rank-2 tensor argument";
pub const ERR_MATMUL_INNER_DIMENSION: &str = "tensor matmul inner dimension mismatch";
pub const ERR_TRANSPOSE_RANK: &str = "tensor transpose requires rank-2 tensor";
pub const ERR_PERMUTE_RANK: &str = "tensor permute axis count must equal tensor rank";
pub const ERR_PERMUTE_NEGATIVE_AXIS: &str = "tensor permute axis must be non-negative";
pub const ERR_PERMUTE_AXIS_OUT_OF_RANGE: &str = "tensor permute axis out of range";
pub const ERR_PERMUTE_DUPLICATE_AXIS: &str = "tensor permute axis must appear exactly once";
pub const ERR_MEDIA_EMPTY: &str = "tensor media (mean) requires at least one element";

pub fn tensor_dim_non_negative(value: i64) -> bool {
    value >= 0
}

pub fn tensor_shape_element_count(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1_usize, |acc, dim| {
        let dim = usize::try_from(*dim).ok()?;
        acc.checked_mul(dim)
    })
}

pub fn tensor_shape_has_element_count(shape: &[i64], actual: usize) -> bool {
    tensor_shape_element_count(shape) == Some(actual)
}

pub fn tensor_flat_offset(shape: &[i64], index: &[i64]) -> Option<usize> {
    if shape.len() != index.len() {
        return None;
    }
    let mut offset = 0_usize;
    let mut stride = 1_usize;
    for (dim, idx) in shape.iter().zip(index.iter()).rev() {
        let dim = usize::try_from(*dim).ok()?;
        let idx = usize::try_from(*idx).ok()?;
        if idx >= dim {
            return None;
        }
        offset = offset.checked_add(idx.checked_mul(stride)?)?;
        stride = stride.checked_mul(dim)?;
    }
    Some(offset)
}

fn shape_dims(shape: &[i64]) -> Result<Vec<usize>, &'static str> {
    shape
        .iter()
        .map(|&dim| parse_non_negative(dim, ERR_NEGATIVE_DIM))
        .collect()
}

fn shape_dims_and_count<T>(shape: &[i64]) -> Result<(Vec<usize>, usize), &'static str> {
    let dims = shape_dims(shape)?;
    let count = checked_allocation_count::<T>(&dims)?;
    Ok((dims, count))
}

fn index_dims(indices: &[i64]) -> Result<Vec<usize>, &'static str> {
    indices
        .iter()
        .map(|&index| parse_non_negative(index, ERR_NEGATIVE_INDEX))
        .collect()
}

fn parse_non_negative(value: i64, message: &'static str) -> Result<usize, &'static str> {
    if value < 0 {
        Err(message)
    } else {
        Ok(value as usize)
    }
}

fn slice_bounds(start: i64, end: i64) -> Result<(usize, usize), &'static str> {
    let start = parse_non_negative(start, ERR_NEGATIVE_SLICE)?;
    let end = parse_non_negative(end, ERR_NEGATIVE_SLICE)?;
    if end < start {
        return Err(ERR_INVALID_SLICE_RANGE);
    }
    Ok((start, end))
}

fn tensor_data<T>(data: &Arc<Mutex<Vec<T>>>) -> MutexGuard<'_, Vec<T>> {
    match data.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl<T: Clone + Default> Tensor<T> {
    pub(crate) fn linea(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::from_contiguous(data, shape)
    }

    /// Rank-0 tensor: one default-initialized element slot.
    pub fn vacua() -> Self {
        Self::from_contiguous(vec![T::default()], Vec::new())
    }

    pub fn longitudo(&self) -> i64 {
        self.shape.len() as i64
    }

    pub fn magnitudines(&self) -> Vec<i64> {
        self.shape.iter().map(|&d| d as i64).collect()
    }

    pub fn element_count(&self) -> usize {
        element_count_usize(&self.shape)
    }

    pub fn crea(shape: &[i64], fill: T) -> Result<Self, &'static str> {
        let (dims, count) = shape_dims_and_count::<T>(shape)?;
        Ok(Self::from_contiguous(vec![fill; count], dims))
    }

    pub fn structa(data: Vec<T>, shape: &[i64]) -> Result<Self, &'static str> {
        let dims = shape_dims(shape)?;
        if !tensor_shape_has_element_count(shape, data.len()) {
            return Err("tensor element count does not match shape");
        }
        Ok(Self::from_contiguous(data, dims))
    }

    pub fn planata(&self) -> Vec<T> {
        let data = tensor_data(&self.data);
        self.logical_offsets()
            .into_iter()
            .map(|offset| data[offset].clone())
            .collect()
    }

    pub fn forma(&self, shape: &[i64]) -> Result<Self, &'static str> {
        let dims = shape_dims(shape)?;
        if !tensor_shape_has_element_count(shape, self.element_count()) {
            return Err(ERR_FORMA_RESHAPE_COUNT);
        }
        Ok(Self::from_contiguous(self.planata(), dims))
    }

    pub fn accipe(&self, indices: &[i64]) -> Result<Option<T>, &'static str> {
        let index = index_dims(indices)?;
        let Some(offset) = self.offset_for_index(&index) else {
            return Ok(None);
        };
        Ok(tensor_data(&self.data).get(offset).cloned())
    }

    pub fn ponde(&mut self, indices: &[i64], value: T) -> Result<(), &'static str> {
        let index = index_dims(indices)?;
        let Some(offset) = self.offset_for_index(&index) else {
            return Err(ERR_INDEX_OUT_OF_BOUNDS);
        };
        tensor_data(&self.data)[offset] = value;
        Ok(())
    }

    pub fn reple(&mut self, value: T) {
        let offsets = self.logical_offsets();
        let mut data = tensor_data(&self.data);
        for offset in offsets {
            data[offset] = value.clone();
        }
    }

    /// Element-wise conversion preserving shape metadata.
    ///
    /// Codegen supplies the per-element map so tensor `↦` mirrors scalar conversio
    /// rules (widening casts, fractus→numerus truncation, and so on).
    pub fn convert_elements<B, F>(&self, map: F) -> Tensor<B>
    where
        B: Clone + Default,
        F: Fn(T) -> B,
    {
        let elems: Vec<B> = self.planata().into_iter().map(map).collect();
        Tensor::from_contiguous(elems, self.shape.clone())
    }

    /// View a contiguous slice along axis 0 from `start` (inclusive) to `end` (exclusive).
    pub fn sectio(&self, start: i64, end: i64) -> Result<Self, &'static str> {
        let (start, end) = slice_bounds(start, end)?;
        if self.shape.is_empty() || end > self.shape[0] {
            return Err(ERR_INDEX_OUT_OF_BOUNDS);
        }
        let mut shape = self.shape.clone();
        shape[0] = end - start;
        Ok(Self {
            data: Arc::clone(&self.data),
            shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[0],
            view: true,
        })
    }

    pub fn materialize(&self) -> Self {
        Self::from_contiguous(self.planata(), self.shape.clone())
    }

    /// Materialized rank-2 transpose.
    pub fn transpose_rank2(&self) -> Result<Self, &'static str> {
        if self.shape.len() != 2 {
            return Err(ERR_TRANSPOSE_RANK);
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let count = checked_allocation_count::<T>(&[cols, rows])?;
        let mut data = Vec::with_capacity(count);
        for col in 0..cols {
            for row in 0..rows {
                data.push(self.value_at_logical(&[row, col]));
            }
        }
        Ok(Self::from_contiguous(data, vec![cols, rows]))
    }

    /// Materialized axis permutation. The result is a copy with row-major strides.
    pub fn permute(&self, axes: &[i64]) -> Result<Self, &'static str> {
        let axes = permute_axes(axes, self.shape.len())?;
        let shape: Vec<usize> = axes.iter().map(|&axis| self.shape[axis]).collect();
        let count = checked_allocation_count::<T>(&shape)?;
        let mut data = Vec::with_capacity(count);
        for ordinal in 0..count {
            let output_index = unravel_index(ordinal, &shape);
            let mut input_index = vec![0; self.shape.len()];
            for (output_axis, &input_axis) in axes.iter().enumerate() {
                input_index[input_axis] = output_index[output_axis];
            }
            data.push(self.value_at_logical(&input_index));
        }
        Ok(Self::from_contiguous(data, shape))
    }

    pub(crate) fn is_view(&self) -> bool {
        self.view
    }

    fn from_contiguous(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            strides: row_major_strides(&shape),
            shape,
            offset: 0,
            view: false,
        }
    }

    fn offset_for_index(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut offset = self.offset;
        for ((idx, dim), stride) in index.iter().zip(self.shape.iter()).zip(self.strides.iter()) {
            if idx >= dim {
                return None;
            }
            offset = offset.checked_add(idx.checked_mul(*stride)?)?;
        }
        Some(offset)
    }

    fn logical_offsets(&self) -> Vec<usize> {
        let count = self.element_count();
        (0..count)
            .map(|ordinal| {
                let index = unravel_index(ordinal, &self.shape);
                self.logical_offset_for_index(&index)
            })
            .collect()
    }

    fn value_at_logical(&self, index: &[usize]) -> T {
        let offset = self.logical_offset_for_index(index);
        tensor_data(&self.data)[offset].clone()
    }

    fn logical_offset_for_index(&self, index: &[usize]) -> usize {
        self.offset
            + index
                .iter()
                .zip(self.strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>()
    }
}

fn element_count_usize(shape: &[usize]) -> usize {
    checked_element_count_usize(shape).expect("tensor shape has checked element count")
}

fn checked_element_count_usize(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
}

fn checked_allocation_count<T>(shape: &[usize]) -> Result<usize, &'static str> {
    let count = checked_element_count_usize(shape).ok_or(ERR_ELEMENT_COUNT_OVERFLOW)?;
    let element_size = std::mem::size_of::<T>();
    if element_size != 0 && count > (isize::MAX as usize) / element_size {
        return Err(ERR_ELEMENT_COUNT_OVERFLOW);
    }
    Ok(count)
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut next = 1_usize;
    for (idx, dim) in shape.iter().enumerate().rev() {
        strides[idx] = next;
        next = next.saturating_mul(*dim);
    }
    strides
}

fn unravel_index(mut ordinal: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut index = vec![0; shape.len()];
    for (axis, dim) in shape.iter().enumerate().rev() {
        index[axis] = ordinal % dim;
        ordinal /= dim;
    }
    index
}

fn permute_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>, &'static str> {
    if axes.len() != rank {
        return Err(ERR_PERMUTE_RANK);
    }
    let mut parsed = Vec::with_capacity(rank);
    let mut seen = vec![false; rank];
    for &axis in axes {
        let axis = parse_non_negative(axis, ERR_PERMUTE_NEGATIVE_AXIS)?;
        if axis >= rank {
            return Err(ERR_PERMUTE_AXIS_OUT_OF_RANGE);
        }
        if seen[axis] {
            return Err(ERR_PERMUTE_DUPLICATE_AXIS);
        }
        seen[axis] = true;
        parsed.push(axis);
    }
    Ok(parsed)
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, &'static str> {
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
            return Err(ERR_BROADCAST_SHAPE);
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn broadcast_dim(shape: &[usize], rank: usize, axis: usize) -> usize {
    let pad = rank - shape.len();
    if axis < pad {
        1
    } else {
        shape[axis - pad]
    }
}

fn broadcast_index(index: &[usize], shape: &[usize]) -> Vec<usize> {
    let pad = index.len() - shape.len();
    (0..shape.len())
        .map(|axis| {
            if shape[axis] == 1 {
                0
            } else {
                index[axis + pad]
            }
        })
        .collect()
}

fn tensor_elementwise<T, F>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    op: F,
) -> Result<Tensor<T>, &'static str>
where
    T: Clone + Default,
    F: Fn(T, T) -> T,
{
    let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
    let count = checked_allocation_count::<T>(&shape)?;
    let mut data = Vec::with_capacity(count);
    for ordinal in 0..count {
        let index = unravel_index(ordinal, &shape);
        let lhs_index = broadcast_index(&index, &lhs.shape);
        let rhs_index = broadcast_index(&index, &rhs.shape);
        data.push(op(
            lhs.value_at_logical(&lhs_index),
            rhs.value_at_logical(&rhs_index),
        ));
    }
    Ok(Tensor::from_contiguous(data, shape))
}

/// Elementwise broadcast arithmetic. Each op is its own `impl` block so the
/// `std::ops` bound is only required where the kernel actually needs it.
impl<T> Tensor<T>
where
    T: Clone + Default + std::ops::Add<Output = T>,
{
    /// Elementwise `self + other` after NumPy-style broadcast unification.
    pub fn addita(&self, other: &Tensor<T>) -> Result<Tensor<T>, &'static str> {
        tensor_elementwise(self, other, |lhs, rhs| lhs + rhs)
    }

    /// Sum of all elements. Integer overflow is the author's responsibility
    /// (per the tensor arithmetic goal non-goals); widen with `↦` first if needed.
    pub fn summa(&self) -> T {
        self.planata()
            .into_iter()
            .fold(T::default(), |acc, value| acc + value)
    }
}

impl<T> Tensor<T>
where
    T: Clone + Default + std::ops::Sub<Output = T>,
{
    /// Elementwise `self - other` after NumPy-style broadcast unification.
    pub fn subtrahe(&self, other: &Tensor<T>) -> Result<Tensor<T>, &'static str> {
        tensor_elementwise(self, other, |lhs, rhs| lhs - rhs)
    }
}

impl<T> Tensor<T>
where
    T: Clone + Default + std::ops::Mul<Output = T>,
{
    /// Elementwise `self * other` after NumPy-style broadcast unification.
    pub fn multiplica(&self, other: &Tensor<T>) -> Result<Tensor<T>, &'static str> {
        tensor_elementwise(self, other, |lhs, rhs| lhs * rhs)
    }
}

impl Tensor<f32> {
    /// Elementwise scalar multiplication preserving tensor shape.
    pub fn scala(&self, factor: f32) -> Tensor<f32> {
        Tensor::from_contiguous(
            self.planata()
                .into_iter()
                .map(|value| value * factor)
                .collect(),
            self.shape.clone(),
        )
    }

    /// Mean of all elements as an f32 scalar.
    pub fn media(&self) -> Result<f32, &'static str> {
        let count = self.element_count();
        if count == 0 {
            return Err(ERR_MEDIA_EMPTY);
        }
        Ok(self.summa() / count as f32)
    }
}

// WHY: matmul needs both `Add` and `Mul` trait bounds since the contraction
// sums products. Placing it in its own impl block keeps the `Add` bound
// scoped to matmul without polluting the elementwise `Mul` block.
impl<T> Tensor<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Rank-2 matrix multiply `self × other`.
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>, &'static str> {
        let dims = &self.shape;
        if dims.len() != 2 {
            return Err(ERR_MATMUL_RECEIVER_RANK);
        }
        let other_dims = &other.shape;
        if other_dims.len() != 2 {
            return Err(ERR_MATMUL_ARGUMENT_RANK);
        }
        let m = dims[0];
        let k1 = dims[1];
        let k2 = other_dims[0];
        let n = other_dims[1];
        if k1 != k2 {
            return Err(ERR_MATMUL_INNER_DIMENSION);
        }
        // WHY: explicit O(M*K*N) contraction loop keeps the kernel readable and
        // works for materialized tensors and views through descriptor offsets.
        let result_count = checked_allocation_count::<T>(&[m, n])?;
        let mut result = Vec::with_capacity(result_count);
        for i in 0..m {
            for j in 0..n {
                let mut acc = T::default();
                for k in 0..k1 {
                    let prod = self.value_at_logical(&[i, k]) * other.value_at_logical(&[k, j]);
                    acc = acc + prod;
                }
                result.push(acc);
            }
        }
        Ok(Tensor::from_contiguous(result, vec![m, n]))
    }
}

#[cfg(test)]
#[path = "tensor_test.rs"]
mod tests;
