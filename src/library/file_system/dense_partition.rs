use crate::library::file_system::Partition;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct DensePartition {
    /// The i-th entry is the identifier of the i-th vector in this shard.
    ids: Vec<usize>,

    /// The i-th row of the data matrix is the i-th vector in this shard.
    data: Array2<f32>,
}

impl DensePartition {
    pub fn new(ids: Vec<usize>, data: Array2<f32>) -> DensePartition {
        DensePartition { ids, data }
    }
}

#[typetag::serde]
impl Partition for DensePartition {
    fn points(&self) -> ArrayView2<f32> {
        self.data.view()
    }

    fn num_points(&self) -> usize {
        self.ids.len()
    }

    fn ids(&self) -> &[usize] {
        &self.ids
    }

    fn mean(&self) -> Array1<f32> {
        self.data.mean_axis(Axis(0)).unwrap()
    }

    fn covariance(&self) -> Array2<f32> {
        let centered_points = &self.data.view() - &self.mean().insert_axis(Axis(0));
        centered_points.t().dot(&centered_points) / self.data.nrows() as f32
    }
}
