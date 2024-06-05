use crate::library::file_system::Partitions;
use crate::library::routing::mean_router::MeanRouter;
use crate::library::routing::Router;
use crate::library::utility;
use linfa_linalg::cholesky::InverseC;
use linfa_linalg::norm::Norm;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ScannRouter {
    internal_router: MeanRouter,
}

impl ScannRouter {
    pub fn build(partitions: &Partitions, threshold: f32) -> ScannRouter {
        assert!(0.0 < threshold && threshold < 1.0);
        let mut means = Array2::<f32>::zeros((partitions.len(), partitions.num_dimensions()));

        let pb = utility::create_progress("Collecting stats", partitions.len());
        means
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, mut mean)| {
                let partition = partitions.partition(index).unwrap();
                // Normalize the points.
                let points = partition.points();
                let norms = Array1::from(
                    points
                        .axis_iter(Axis(0))
                        .map(|row| row.norm_l2())
                        .collect::<Vec<_>>(),
                )
                .insert_axis(Axis(1));
                let points = &points / norms;

                // Compute \sum_{x_i} (1 - h_\perp / h_\parallel) x_i x_i.t().
                let mut mat =
                    Array2::zeros((partitions.num_dimensions(), partitions.num_dimensions()));
                points
                    .axis_iter(Axis(0))
                    .for_each(|point| mat += point.t().dot(&point));
                mat *= 1_f32 - threshold;

                // Add (I \sum_{x_i} h_\perp / h_\parallel) to the diagonals.
                mat.diag_mut()
                    .iter_mut()
                    .for_each(|value| *value += threshold * points.nrows() as f32);

                let mat = mat.invc().unwrap();
                let sum = points.sum_axis(Axis(0));
                mean.assign(&mat.dot(&sum));
                pb.inc(1);
            });
        pb.finish_and_clear();

        ScannRouter {
            internal_router: MeanRouter::new(means),
        }
    }
}

#[typetag::serde]
impl Router for ScannRouter {
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
