use crate::library::file_system::Partitions;
use crate::library::routing::Router;
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering::Equal;

/// A router that simply ranks partitions by the inner product
/// between a query and partition centroids.
#[derive(Serialize, Deserialize)]
pub struct MeanRouter {
    centroids: Array2<f32>,
}

impl MeanRouter {
    pub fn new(centroids: Array2<f32>) -> MeanRouter {
        MeanRouter { centroids }
    }

    pub fn build(partitions: &Partitions) -> MeanRouter {
        let mut centroids = Array2::<f32>::zeros((partitions.len(), partitions.num_dimensions()));
        partitions
            .iter()
            .enumerate()
            .for_each(|(index, partition)| {
                centroids.row_mut(index).assign(&partition.mean());
            });

        MeanRouter { centroids }
    }
}

#[typetag::serde]
impl Router for MeanRouter {
    fn sort_partitions(
        &self,
        query: ArrayView1<f32>,
        _partitions: &Partitions,
    ) -> Vec<(usize, f32)> {
        // Compute inner product between centroids and the query vector.
        let scores = self.centroids.dot(&query);

        // Sort the partitions according to the computed scores.
        let mut results = scores.to_vec().into_iter().enumerate().collect::<Vec<_>>();
        results.par_sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Equal));
        results
    }

    fn size(&self) -> usize {
        self.centroids.len()
    }
}
