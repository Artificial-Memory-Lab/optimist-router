pub mod dense_partition;

use linfa_linalg::eigh::{EigSort, Eigh};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::slice::Iter;

/// Represents a single partition.
#[typetag::serde(tag = "partition")]
pub trait Partition: Sync {
    /// Returns the points.
    fn points(&self) -> ArrayView2<f32>;

    /// Returns the number of points.
    fn num_points(&self) -> usize;

    /// Returns the ids of the points.
    fn ids(&self) -> &[usize];

    /// Returns the mean of the points.
    fn mean(&self) -> Array1<f32>;

    /// Returns the covariance of the points.
    fn covariance(&self) -> Array2<f32>;
}

#[derive(Serialize, Deserialize)]
pub struct Partitions {
    partitions: Vec<Box<dyn Partition>>,
    num_dimensions: usize,
}

impl Partitions {
    pub fn new(partitions: Vec<Box<dyn Partition>>, num_dimensions: usize) -> Partitions {
        Partitions {
            partitions,
            num_dimensions,
        }
    }

    /// Returns the total number of partitions.
    pub fn len(&self) -> usize {
        self.partitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dimensions(&self) -> usize {
        self.num_dimensions
    }

    pub fn partition(&self, index: usize) -> Result<&dyn Partition, String> {
        if index < self.partitions.len() {
            return Ok(self.partitions[index].as_ref());
        }
        Err(format!(
            "Partition {} requested, but there are only {} partitions.",
            index,
            self.partitions.len()
        ))
    }

    pub fn iter(&self) -> Iter<Box<dyn Partition>> {
        self.partitions.iter()
    }

    pub fn par_iter(&self) -> rayon::slice::Iter<Box<dyn Partition>> {
        self.partitions.par_iter()
    }

    /// Collects the t-th eigenvalue from D^{-1/2} \Sigma D^{-1/2} of each shard,
    /// where \Sigma is the covariance matrix and D is its diagonal.
    pub fn covariance_eigenvalues(&self, t: usize) -> Vec<f32> {
        self.partitions
            .par_iter()
            .map(|partition| {
                let sigma = partition.covariance();
                let mut d = Array2::<f32>::eye(sigma.ncols()) * sigma.diag();
                d.mapv_inplace(|v| if v != 0. { 1_f32 / v.sqrt() } else { 0_f32 });
                let m = d.dot(&sigma.dot(&d));

                let (eigen_values, _) = m.eigh().unwrap().sort_eig_desc();
                eigen_values[[t]]
            })
            .collect::<Vec<_>>()
    }
}
