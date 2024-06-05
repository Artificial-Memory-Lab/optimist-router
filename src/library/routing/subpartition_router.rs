use crate::library::clustering::Clustering;
use crate::library::file_system::Partitions;
use crate::library::routing::Router;
use crate::library::utility;
use ndarray::{Array1, Array2, ArrayView1, Axis, Zip};
use rand::prelude::{SliceRandom, StdRng};
use rand::{thread_rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering::Equal;

#[derive(Serialize, Deserialize)]
struct SubPartition {
    points: Array2<f32>,
}

impl SubPartition {
    fn size(&self) -> usize {
        self.points.len()
    }

    fn score(&self, query: ArrayView1<f32>) -> f32 {
        *self
            .points
            .dot(&query)
            .iter()
            .max_by(|&a, &b| a.partial_cmp(b).unwrap_or(Equal))
            .unwrap()
    }
}

#[derive(Serialize, Deserialize)]
pub struct SubPartitionRouter {
    sub_partitions: Vec<SubPartition>,
}

impl SubPartitionRouter {
    pub fn build(
        seed: Option<u64>,
        partitions: &Partitions,
        clustering: Box<dyn Clustering>,
        num_clusters: usize,
    ) -> SubPartitionRouter {
        let pb = utility::create_progress("Collecting stats", partitions.len());
        let sub_partitions = partitions
            .par_iter()
            .map(|partition| {
                let (mut centroids, _) =
                    clustering.cluster(partition.points(), num_clusters, false);

                let mut rng = match seed {
                    None => StdRng::from_rng(thread_rng()).unwrap(),
                    Some(seed) => StdRng::seed_from_u64(seed),
                };
                let mut indexes: Vec<usize> = (0..partition.points().nrows()).collect();
                indexes.shuffle(&mut rng);
                let mut pointer = 0_usize;

                centroids.axis_iter_mut(Axis(0)).for_each(|mut centroid| {
                    if centroid.sum() == 0_f32 {
                        centroid.assign(&partition.points().row(pointer));
                        pointer += 1;
                    }
                });

                pb.inc(1);
                SubPartition { points: centroids }
            })
            .collect::<Vec<_>>();
        pb.finish_and_clear();

        SubPartitionRouter { sub_partitions }
    }
}

#[typetag::serde]
impl Router for SubPartitionRouter {
    fn sort_partitions(
        &self,
        query: ArrayView1<f32>,
        _partitions: &Partitions,
    ) -> Vec<(usize, f32)> {
        let mut scores = Array1::<f32>::zeros(self.sub_partitions.len());

        Zip::indexed(&mut scores).par_for_each(|partition_id, dot| {
            let sub_partition = &self.sub_partitions[partition_id];
            *dot = sub_partition.score(query);
        });

        let mut results = scores.to_vec().into_iter().enumerate().collect::<Vec<_>>();
        results.par_sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(Equal));
        results
    }

    fn size(&self) -> usize {
        self.sub_partitions.iter().map(|c| c.size()).sum::<usize>()
    }
}
