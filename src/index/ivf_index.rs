use ann_dataset::PointSet;
use anyhow::anyhow;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering::Equal;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;
use thousands::Separable;

use crate::implement_serialization;
use crate::library::clustering::Clustering;
use crate::library::file_system::dense_partition::DensePartition;
use crate::library::file_system::{Partition, Partitions};
use crate::library::routing::Router;
use crate::library::utility;
use crate::library::utility::search_result::{RetrievalResponse, SearchResult};
use crate::library::utility::serialization::FileSerializable;

#[derive(Clone)]
pub enum Query<'a> {
    Dense(ArrayView1<'a, f32>),
}

#[derive(Clone)]
pub enum Cutoff {
    NumPoints(Vec<usize>),
    NumClusters(Vec<usize>),
}

#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    partitions: Partitions,
}

// Takes care of serialization and deserialization.
impl FileSerializable for IvfIndex {
    implement_serialization!();
}

impl IvfIndex {
    pub fn new(
        data_points: &PointSet<f32>,
        clustering: Box<dyn Clustering>,
        num_clusters: usize,
    ) -> anyhow::Result<IvfIndex> {
        if data_points.get_sparse().is_some() {
            return Err(anyhow!("Sparse vectors are not yet supported."));
        }

        Ok(Self::new_dense_index(
            data_points.get_dense().unwrap().view(),
            clustering,
            num_clusters,
        ))
    }

    fn new_dense_index(
        points: ArrayView2<f32>,
        clustering: Box<dyn Clustering>,
        num_clusters: usize,
    ) -> IvfIndex {
        let (centroids, targets) = clustering.cluster(points.view(), num_clusters, true);

        // Split the data into separate partitions based on the given cluster assignments.
        let now = Instant::now();
        let pb = utility::create_progress("Creating partitions", centroids.nrows());
        let mut partitions: Vec<Box<dyn Partition>> = Vec::new();
        (0..centroids.nrows()).for_each(|partition_id| {
            let mut subset: Vec<usize> = Vec::new();
            targets.iter().enumerate().for_each(|(docid, &target)| {
                if target == partition_id {
                    subset.push(docid);
                }
            });

            pb.inc(1);
            if subset.is_empty() {
                return;
            }

            let partition_points = points.select(Axis(0), &subset);
            partitions.push(Box::new(DensePartition::new(subset, partition_points)));
        });
        pb.finish_and_clear();
        println!(
            "Created partitions in {} seconds",
            now.elapsed().as_secs().separate_with_commas()
        );

        IvfIndex {
            partitions: Partitions::new(partitions, points.ncols()),
        }
    }

    pub fn partitions(&self) -> &Partitions {
        &self.partitions
    }

    /// Performs top-k retrieval over the index for a given query.
    ///
    /// The algorithm probes as many partitions as necessary to examine at least `cutoff` points or
    /// clusters.
    pub fn retrieve(
        &self,
        router: &dyn Router,
        query: &Query,
        cutoffs: &Cutoff,
        top_k: usize,
    ) -> Vec<RetrievalResponse> {
        let now = Instant::now();
        // Ask the query router to sort partitions given the current query.
        let sorted_partitions = match &query {
            Query::Dense(q) => router.sort_partitions(q.view(), self.partitions()),
        };
        let routing_latency = now.elapsed();

        let cutoff_values = match cutoffs {
            Cutoff::NumPoints(c) => c,
            Cutoff::NumClusters(c) => c,
        };

        cutoff_values
            .iter()
            .map(|&cutoff| {
                // Find out how many top-ranking partitions we must probe so that at least
                // `min_vectors_probed` data points are eventually visited.
                let mut examined_docs = 0_usize;
                let mut partition = 0_usize;
                match cutoffs {
                    Cutoff::NumPoints(_) => {
                        while examined_docs < cutoff && partition < sorted_partitions.len() {
                            examined_docs += self
                                .partitions
                                .partition(sorted_partitions[partition].0)
                                .unwrap()
                                .num_points();
                            partition += 1;
                        }
                    }
                    Cutoff::NumClusters(_) => {
                        while partition < cutoff && partition < sorted_partitions.len() {
                            examined_docs += self
                                .partitions
                                .partition(sorted_partitions[partition].0)
                                .unwrap()
                                .num_points();
                            partition += 1;
                        }
                    }
                }

                // Accumulate prediction error.
                let mut prediction_error = 0_f32;
                // Compute scores for the data points in the top partitions and extract the top-k vectors.
                let mut heap: BinaryHeap<Reverse<SearchResult>> = BinaryHeap::new();
                // Keep track of the minimum score in the heap.
                let mut threshold = f32::MIN;
                sorted_partitions[..partition].iter().for_each(
                    |&(partition_id, predicted_score)| {
                        // This gives the inner product between query and every data point in the partition.
                        let scores = match &query {
                            Query::Dense(q) => self
                                .partitions
                                .partition(partition_id)
                                .unwrap()
                                .points()
                                .dot(&q.view()),
                        };

                        // Compute error.
                        let &max_score = scores
                            .iter()
                            .max_by(|x, y| x.partial_cmp(y).unwrap_or(Equal))
                            .unwrap();
                        prediction_error += (predicted_score / max_score - 1_f32).abs();

                        // Iterate over all scores from this partition and possibly update the heap.
                        scores.iter().enumerate().for_each(|(i, &score)| {
                            if score > threshold {
                                heap.push(Reverse(SearchResult {
                                    docid: self.partitions.partition(partition_id).unwrap().ids()[i]
                                        as u32,
                                    score,
                                }));
                                if heap.len() > top_k {
                                    threshold = heap.pop().unwrap().0.score;
                                }
                            }
                        });
                    },
                );

                // Extract the top vectors.
                let results = heap.into_sorted_vec().iter().map(|e| e.0).collect();
                RetrievalResponse {
                    results,
                    expected_number_of_docs_probed: examined_docs as u32,
                    actual_number_of_docs_probed: examined_docs as u32,
                    mean_relative_prediction_error: prediction_error / partition as f32,
                    routing_latency,
                }
            })
            .collect::<Vec<_>>()
    }
}
