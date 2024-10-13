use crate::library::file_system::Partitions;
use crate::library::routing::Router;
use crate::library::sketching::symmetric_psd_sketching::{
    SymmetricPSDSketch, SymmetricPSDSketcher,
};
use crate::library::utility;
use ndarray::{Array2, ArrayView1, Zip};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering::Equal;

#[derive(Serialize, Deserialize)]
pub struct OptimistRouter {
    means: Array2<f32>,
    covariances: Vec<Box<dyn SymmetricPSDSketch>>,
    delta: f32,
}

impl OptimistRouter {
    pub fn build(
        partitions: &Partitions,
        delta: f32,
        sketching: Box<dyn SymmetricPSDSketcher>,
    ) -> OptimistRouter {
        let mut means = Array2::<f32>::zeros((partitions.len(), partitions.num_dimensions()));

        Zip::indexed(means.rows_mut()).par_for_each(|partition_id, mut mean| {
            let partition = partitions.partition(partition_id).unwrap();
            mean.assign(&partition.mean());
        });

        let pb = utility::create_progress("Collecting stats", partitions.len());
        let covariances = partitions
            .par_iter()
            .map(|partition| {
                let covariance = partition.covariance();
                pb.inc(1);
                sketching.sketch(covariance.view())
            })
            .collect::<Vec<_>>();
        pb.finish_and_clear();

        OptimistRouter {
            means,
            covariances,
            delta,
        }
    }
}

#[typetag::serde]
impl Router for OptimistRouter {
    fn sort_partitions(
        &self,
        query: ArrayView1<f32>,
        _partitions: &Partitions,
    ) -> Vec<(usize, f32)> {
        let mut scores = self.means.dot(&query);
        let alpha = ((1_f32 + self.delta) / (1_f32 - self.delta)).sqrt();

        Zip::indexed(&mut scores).par_for_each(|index, score| {
            let std = self.covariances[index].dot(query).sqrt();
            *score += alpha * std;
        });

        let mut results = scores.to_vec().into_iter().enumerate().collect::<Vec<_>>();
        results.par_sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Equal));
        results
    }

    fn size(&self) -> usize {
        let covariance_size = self
            .covariances
            .iter()
            .map(|sketch| sketch.size())
            .sum::<usize>();
        self.means.len() + covariance_size
    }
}
