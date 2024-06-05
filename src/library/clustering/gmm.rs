use crate::library::clustering::kmeans::{KMeansClustering, KMeansMode};
use crate::library::clustering::Clustering;
use indicatif::ProgressBar;
use itertools::Itertools;
use linfa_linalg::cholesky::Cholesky;
use ndarray::{Array, Array1, Array2, Array3, ArrayView2, Axis, Zip};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{thread_rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::cmp::Ordering::Equal;
use std::ops::IndexMut;
use std::time::Instant;
use thousands::Separable;

use crate::library::utility;
use crate::library::utility::linalg::{inverse_upper_triangular, softmax};

const DEFAULT_NUM_ITERATIONS: usize = 5_usize;
const DEFAULT_ASSIGNMENT_TOP_K: usize = 10_usize;

#[derive(PartialEq, Debug, Clone, Default, Serialize, Deserialize)]
pub enum GMMMode {
    #[default]
    Soft,

    Hard,
}

pub struct GMMClustering {
    seed: Option<u64>,
    mode: GMMMode,
    num_iterations: usize,
    samples_per_centroid: Option<usize>,
    hard_assignment_top_k: Option<usize>,
}

impl GMMClustering {
    pub fn new(
        seed: Option<u64>,
        mode: GMMMode,
        num_iterations: Option<usize>,
        samples_per_centroid: Option<usize>,
        hard_assignment_top_k: Option<usize>,
    ) -> GMMClustering {
        GMMClustering {
            seed,
            mode,
            num_iterations: num_iterations.unwrap_or(DEFAULT_NUM_ITERATIONS),
            samples_per_centroid,
            hard_assignment_top_k,
        }
    }
}

impl Clustering for GMMClustering {
    fn cluster(
        &self,
        points: ArrayView2<f32>,
        num_clusters: usize,
        verbose: bool,
    ) -> (Array2<f32>, Vec<usize>) {
        let mut rng = match self.seed {
            None => StdRng::from_rng(thread_rng()).unwrap(),
            Some(seed) => StdRng::seed_from_u64(seed),
        };

        let mut training_point_indexes: Vec<usize> = (0..points.nrows()).collect();
        training_point_indexes.shuffle(&mut rng);
        let training_point_indexes = match self.samples_per_centroid {
            None => &training_point_indexes,
            Some(samples_per_centroid) => {
                &training_point_indexes[..min(
                    num_clusters * samples_per_centroid,
                    training_point_indexes.len(),
                )]
            }
        };
        let training_points = points.select(Axis(0), training_point_indexes);

        // Initialize the model with Standard KMeans.
        let kmeans = KMeansClustering::new(
            self.seed,
            KMeansMode::Standard,
            Some(self.num_iterations),
            None,
        );
        let (means, targets) = kmeans.cluster(training_points.view(), num_clusters, verbose);

        let mut gmm: Box<dyn GMMPartitioner> = match self.mode {
            GMMMode::Soft => Box::new(SoftGMM::new(
                training_points.view(),
                means,
                &targets,
                num_clusters,
            )),
            GMMMode::Hard => Box::new(HardGMM::new(
                training_points.view(),
                means,
                &targets,
                num_clusters,
                self.hard_assignment_top_k
                    .unwrap_or(DEFAULT_ASSIGNMENT_TOP_K),
            )),
        };

        let mut num_iterations = self.num_iterations;
        let pb = if verbose {
            Some(utility::create_progress("Training GMM", num_iterations))
        } else {
            None
        };
        let now = Instant::now();
        while num_iterations > 0 {
            gmm.iterate();
            if let Some(pb) = &pb {
                pb.inc(1);
            }
            num_iterations -= 1;
        }
        if let Some(pb) = &pb {
            pb.finish_and_clear();
        }
        if verbose {
            println!(
                "Trained GMM in {} seconds",
                now.elapsed().as_secs().separate_with_commas()
            );
        }

        let now = Instant::now();
        let pb = if verbose {
            Some(utility::create_progress(
                "Assigning clusters",
                points.nrows(),
            ))
        } else {
            None
        };
        let targets = gmm.assign(points, pb.as_ref());
        if let Some(pb) = &pb {
            pb.finish_and_clear();
        }

        if verbose {
            println!(
                "Assigned clusters in {} seconds",
                now.elapsed().as_secs().separate_with_commas()
            );
        }

        (gmm.means(), targets)
    }
}

trait GMMPartitioner {
    /// Performs one iteration of training.
    fn iterate(&mut self);

    fn assign(&self, points: ArrayView2<f32>, pb: Option<&ProgressBar>) -> Vec<usize>;

    fn means(&self) -> Array2<f32>;
}

struct SoftGMM<'a> {
    points: ArrayView2<'a, f32>,
    weights: Array1<f32>,
    means: Array2<f32>,
    covariances: Array3<f32>,
    precisions_sqrt: Array3<f32>,
}

impl SoftGMM<'_> {
    fn new<'a>(
        points: ArrayView2<'a, f32>,
        means: Array2<f32>,
        targets: &[usize],
        num_clusters: usize,
    ) -> SoftGMM<'a> {
        let mut gmm = SoftGMM {
            points,
            weights: Array1::<f32>::ones(num_clusters),
            means,
            covariances: Array3::<f32>::zeros((num_clusters, points.ncols(), points.ncols())),
            precisions_sqrt: Array3::zeros((num_clusters, points.ncols(), points.ncols())),
        };

        let mut responsibilities = Array2::<f32>::zeros((points.nrows(), num_clusters));
        targets.iter().enumerate().for_each(|(row, &target)| {
            responsibilities[[row, target]] = 1_f32;
        });

        gmm.estimate_gaussian_parameters(responsibilities.view());
        gmm.compute_precision_matrix();
        gmm
    }

    fn estimate_gaussian_parameters(&mut self, responsibilities: ArrayView2<f32>) {
        let weights = responsibilities.sum_axis(Axis(0));
        self.means =
            responsibilities.t().dot(&self.points) / weights.to_owned().insert_axis(Axis(1));

        Zip::from(self.covariances.axis_iter_mut(Axis(0)))
            .and(self.means.outer_iter())
            .and(responsibilities.axis_iter(Axis(1)))
            .and(weights.view())
            .par_for_each(|mut covariance, mean, resp, &w| {
                let diff = &self.points - &mean;
                let m = &diff.t() * &resp;
                let mut cov_k = m.dot(&diff) / w;
                cov_k.diag_mut().mapv_inplace(|x| x + 1e-4);
                covariance.assign(&cov_k);
            });

        self.weights = weights / self.points.nrows() as f32;
    }

    /// Computes the square root of the Precision matrix (i.e., inverse of the covariance matrix).
    /// If the covariance matrix is S = L L.t(), then the matrix returned for each cluster is
    /// (L.t())^-1.
    fn compute_precision_matrix(&mut self) {
        Zip::from(self.covariances.outer_iter())
            .and(self.precisions_sqrt.outer_iter_mut())
            .par_for_each(|covariance, mut matrix| {
                let sol = {
                    let decomp = covariance.cholesky().unwrap();
                    inverse_upper_triangular(decomp.t())
                };
                matrix.assign(&sol);
            });
    }

    /// Performs one iteration of the expectation step and returns the log responsibilities.
    fn e_step(&self) -> Array2<f32> {
        self.estimate_log_responsibilities(self.points)
    }

    /// Updates model parameters given the provided log responsibilities.
    fn m_step(&mut self, log_responsibilities: ArrayView2<f32>) {
        self.estimate_gaussian_parameters(log_responsibilities.mapv(|x| x.exp()).view());
        self.compute_precision_matrix();
    }

    /// Estimates the log responsibilities for each sample.
    fn estimate_log_responsibilities(&self, points: ArrayView2<f32>) -> Array2<f32> {
        let mut output = self.estimate_log_gaussian_prob(points);
        output
            .rows_mut()
            .into_iter()
            .par_bridge()
            .for_each(|mut row| {
                row.assign(&softmax(row.view(), None));
                row.mapv_inplace(|x| x.ln());
            });
        output
    }

    /// Compute the log likelihood in case of the gaussian probabilities
    /// log(P(X|Mean, Precision)) = -0.5*(d*ln(2*PI)+ln(det(Precision))-(X-Mean)^t.Precision.(X-Mean)
    fn estimate_log_gaussian_prob(&self, points: ArrayView2<f32>) -> Array2<f32> {
        let n_samples = points.nrows();
        let n_dims = points.ncols();
        let n_clusters = self.means.nrows();

        let log_det = self.compute_log_det();
        let mut log_prob: Array2<f32> = Array::zeros((n_samples, n_clusters));
        Zip::from(self.means.rows())
            .and(self.precisions_sqrt.outer_iter())
            .and(log_prob.axis_iter_mut(Axis(1)))
            .par_for_each(|mu, precision, mut result| {
                let diff = (&points - &mu).dot(&precision);
                result.assign(&diff.mapv(|v| v * v).sum_axis(Axis(1)));
            });
        log_prob.mapv(|v| -0.5 * (v + n_dims as f32 * f32::ln(2_f32 * std::f32::consts::PI)))
            + log_det
            + self.weights.mapv(|x| x.ln())
    }

    /// Computes the log|L.t()^-1|, where S = L L.t() is the covariance matrix.
    fn compute_log_det(&self) -> Array1<f32> {
        Array1::<f32>::from(
            self.precisions_sqrt
                .axis_iter(Axis(0))
                .map(|matrix| matrix.diag().mapv(|x| x.ln()).sum())
                .collect::<Vec<f32>>(),
        )
    }
}

impl GMMPartitioner for SoftGMM<'_> {
    fn iterate(&mut self) {
        let log_responsibilities = self.e_step();
        self.m_step(log_responsibilities.view());
    }

    fn assign(&self, points: ArrayView2<f32>, pb: Option<&ProgressBar>) -> Vec<usize> {
        points
            .axis_chunks_iter(Axis(0), 100_000)
            .flat_map(|subset| {
                let log_responsibilities = self.estimate_log_responsibilities(subset);
                if let Some(bar) = pb {
                    bar.inc(subset.nrows() as u64);
                }
                log_responsibilities
                    .axis_iter(Axis(0))
                    .map(|row| {
                        row.iter()
                            .position_max_by(|&x, &y| x.partial_cmp(y).unwrap_or(Equal))
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn means(&self) -> Array2<f32> {
        self.means.to_owned()
    }
}

struct HardGMM<'a> {
    points: ArrayView2<'a, f32>,
    weights: Array1<f32>,
    means: Array2<f32>,
    covariances: Array3<f32>,
    precisions_sqrt: Array3<f32>,
    assignment_top_k: usize,
}

impl HardGMM<'_> {
    fn new<'a>(
        points: ArrayView2<'a, f32>,
        means: Array2<f32>,
        targets: &[usize],
        num_clusters: usize,
        assignment_top_k: usize,
    ) -> HardGMM<'a> {
        let mut gmm = HardGMM {
            points,
            weights: Array1::<f32>::ones(num_clusters),
            means,
            covariances: Array3::<f32>::zeros((num_clusters, points.ncols(), points.ncols())),
            precisions_sqrt: Array3::zeros((num_clusters, points.ncols(), points.ncols())),
            assignment_top_k,
        };

        gmm.estimate_gaussian_parameters(targets);
        gmm.compute_precision_matrix();
        gmm
    }

    fn estimate_gaussian_parameters(&mut self, assignments: &[usize]) {
        Zip::indexed(self.means.axis_iter_mut(Axis(0)))
            .and(self.covariances.axis_iter_mut(Axis(0)))
            .and(self.weights.view_mut())
            .par_for_each(|partition_id, mut mean, mut covariance, weight| {
                let point_ids = assignments
                    .iter()
                    .enumerate()
                    .filter_map(|(point_id, &target)| {
                        if target == partition_id {
                            Some(point_id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                let subset = self.points.select(Axis(0), &point_ids);
                mean.assign(&subset.mean_axis(Axis(0)).unwrap());
                *weight = subset.nrows() as f32;

                let centered = subset - mean.insert_axis(Axis(0));
                let mut cov = centered.t().dot(&centered) / *weight;
                cov.diag_mut().mapv_inplace(|x| x + 1e-4);
                covariance.assign(&cov);
            });

        self.weights /= self.points.nrows() as f32;
    }

    /// Computes the square root of the Precision matrix (i.e., inverse of the covariance matrix).
    /// If the covariance matrix is S = L L.t(), then the matrix returned for each cluster is
    /// (L.t())^-1.
    fn compute_precision_matrix(&mut self) {
        Zip::from(self.covariances.outer_iter())
            .and(self.precisions_sqrt.outer_iter_mut())
            .par_for_each(|covariance, mut matrix| {
                let sol = {
                    let decomp = covariance.cholesky().unwrap();
                    inverse_upper_triangular(decomp.t())
                };
                matrix.assign(&sol);
            });
    }

    /// Performs one iteration of the expectation step and returns the log responsibilities.
    fn e_step(&self) -> Vec<usize> {
        self.assign_points(self.points)
    }

    /// Updates model parameters given the provided log responsibilities.
    fn m_step(&mut self, targets: &[usize]) {
        self.estimate_gaussian_parameters(targets);
        self.compute_precision_matrix();
    }

    /// Compute shifted log likelihood in case of the gaussian probabilities
    /// log(P(X|Mean, Precision)) = -0.5*(ln(det(Covariance))-(X-Mean)^t.Precision.(X-Mean))
    fn assign_points(&self, points: ArrayView2<f32>) -> Vec<usize> {
        let n_samples = points.nrows();
        let n_clusters = self.means.nrows();

        let mut squared_norms = Array1::<f32>::zeros(n_clusters);
        Zip::from(&mut squared_norms)
            .and(self.means.rows())
            .par_for_each(|norm_squared, centroid| {
                *norm_squared = centroid.dot(&centroid);
            });

        let log_det = self.compute_log_det();
        let mut assignments: Array1<usize> = Array::zeros(n_samples);
        Zip::from(points.axis_iter(Axis(0)))
            .and(assignments.view_mut())
            .par_for_each(|point, assignment| {
                let mut distances: Array1<f32> = Array::zeros(n_clusters);
                // Find the top clusters based on the mean.
                Zip::from(distances.view_mut())
                    .and(self.means.rows())
                    .and(&squared_norms)
                    .for_each(|distances, centroid, &squared_norm| {
                        *distances = squared_norm - 2_f32 * centroid.dot(&point);
                    });
                let mut sorted = distances
                    .to_vec()
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, *v))
                    .collect::<Vec<_>>();
                sorted.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(Equal));

                // Of the top-k closest partitions by means, find the most likely partition by
                // the GMM parameters.
                distances.mapv_inplace(|_| f32::NEG_INFINITY);
                sorted[..min(self.assignment_top_k, sorted.len())]
                    .iter()
                    .for_each(|&(partition_id, _)| {
                        let diff = (&point - &self.means.row(partition_id))
                            .dot(&self.precisions_sqrt.index_axis(Axis(0), partition_id));
                        *distances.index_mut(partition_id) = diff.mapv(|v| -0.5 * v * v).sum()
                            + log_det[[partition_id]]
                            + self.weights[[partition_id]].ln();
                    });
                *assignment = distances
                    .iter()
                    .position_max_by(|&x, &y| x.partial_cmp(y).unwrap_or(Equal))
                    .unwrap();
            });
        assignments.to_vec()
    }

    /// Computes the log|L.t()^-1|, where S = L L.t() is the covariance matrix.
    fn compute_log_det(&self) -> Array1<f32> {
        Array1::<f32>::from(
            self.precisions_sqrt
                .axis_iter(Axis(0))
                .map(|matrix| matrix.diag().mapv(|x| x.ln()).sum())
                .collect::<Vec<f32>>(),
        )
    }
}

impl GMMPartitioner for HardGMM<'_> {
    fn iterate(&mut self) {
        let targets = self.e_step();
        self.m_step(&targets);
    }

    fn assign(&self, points: ArrayView2<f32>, pb: Option<&ProgressBar>) -> Vec<usize> {
        points
            .axis_chunks_iter(Axis(0), 100_000)
            .flat_map(|subset| {
                let batch_assignment = self.assign_points(subset);
                if let Some(bar) = pb {
                    bar.inc(subset.nrows() as u64);
                }
                batch_assignment
            })
            .collect::<Vec<_>>()
    }

    fn means(&self) -> Array2<f32> {
        self.means.to_owned()
    }
}
