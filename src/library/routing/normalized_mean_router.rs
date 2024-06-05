use crate::library::file_system::Partitions;
use crate::library::routing::mean_router::MeanRouter;
use crate::library::routing::Router;
use linfa_linalg::norm::Norm;
use ndarray::{Array2, ArrayView1};
use serde::{Deserialize, Serialize};

/// A router that routes queries with respect to normalized means of partitions.
#[derive(Serialize, Deserialize)]
pub struct NormalizedMeanRouter {
    internal_router: MeanRouter,
}

impl NormalizedMeanRouter {
    pub fn build(partitions: &Partitions) -> NormalizedMeanRouter {
        let mut centroids = Array2::<f32>::zeros((partitions.len(), partitions.num_dimensions()));
        partitions
            .iter()
            .enumerate()
            .for_each(|(index, partition)| {
                let centroid = partition.mean();
                let centroid = &centroid / (centroid.norm_l2() + 1e-8);
                centroids.row_mut(index).assign(&centroid);
            });

        NormalizedMeanRouter {
            internal_router: MeanRouter::new(centroids),
        }
    }
}

#[typetag::serde]
impl Router for NormalizedMeanRouter {
    fn sort_partitions(
        &self,
        query: ArrayView1<f32>,
        partitions: &Partitions,
    ) -> Vec<(usize, f32)> {
        self.internal_router.sort_partitions(query, partitions)
    }

    fn size(&self) -> usize {
        self.internal_router.size()
    }
}
