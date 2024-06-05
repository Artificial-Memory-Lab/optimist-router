use crate::library::clustering::Clustering;
use indicatif::ProgressBar;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis, Zip};
use rand::prelude::{SliceRandom, StdRng};
use rand::thread_rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::min;
use std::cmp::Ordering::Equal;
use std::ops::AddAssign;
use std::time::Instant;
use thousands::Separable;

use crate::library::utility;

const DEFAULT_NUM_ITERATIONS: usize = 20_usize;

#[derive(PartialEq, Debug, Clone)]
pub enum KMeansMode {
    /// Standard KMeans
    Standard,

    /// Spherical KMeans
    Spherical,
}

pub struct KMeansClustering {
    seed: Option<u64>,
    mode: KMeansMode,
    num_iterations: usize,
    samples_per_centroid: Option<usize>,
}

impl KMeansClustering {
    pub fn new(
        seed: Option<u64>,
        mode: KMeansMode,
        num_iterations: Option<usize>,
        samples_per_centroid: Option<usize>,
    ) -> KMeansClustering {
        KMeansClustering {
            seed,
            mode,
            num_iterations: num_iterations.unwrap_or(DEFAULT_NUM_ITERATIONS),
            samples_per_centroid,
        }
    }
}

impl Clustering for KMeansClustering {
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

        // Initialize centroids.
        let mut centroid_indexes: Vec<usize> = (0..training_points.nrows()).collect();
        centroid_indexes.shuffle(&mut rng);
        let centroid_indexes = &centroid_indexes[..min(num_clusters, centroid_indexes.len())];
        let mut centroids = training_points.select(Axis(0), centroid_indexes);

        let mut num_iterations = self.num_iterations;
        let pb = if verbose {
            Some(utility::create_progress("Training KMeans", num_iterations))
        } else {
            None
        };
        let now = Instant::now();
        while num_iterations > 0 {
            let assignments = assign_clusters(training_points.view(), centroids.view(), None);

            compute_centroids(
                &mut centroids.view_mut(),
                &assignments,
                training_points.view(),
                &self.mode,
            );

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
                "Trained KMeans in {} seconds",
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
        let targets = assign_clusters(points.view(), centroids.view(), pb.as_ref());

        if let Some(pb) = &pb {
            pb.finish_and_clear();
        }
        if verbose {
            println!(
                "Assigned clusters in {} seconds",
                now.elapsed().as_secs().separate_with_commas()
            );
        }

        (centroids, targets)
    }
}

fn assign_clusters(
    points: ArrayView2<f32>,
    centroids: ArrayView2<f32>,
    pb: Option<&ProgressBar>,
) -> Vec<usize> {
    let mut squared_norms = Array1::<f32>::zeros(centroids.nrows());
    Zip::from(&mut squared_norms)
        .and(centroids.rows())
        .par_for_each(|norm_squared, centroid| {
            *norm_squared = centroid.dot(&centroid);
        });

    points
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|point| {
            if let Some(pb) = pb {
                pb.inc(1);
            }

            let mut distances = Array1::<f32>::zeros(centroids.nrows());
            Zip::from(&mut distances)
                .and(centroids.rows())
                .and(&squared_norms)
                .par_for_each(|distances, centroid, &squared_norm| {
                    *distances = squared_norm - 2_f32 * centroid.dot(&point);
                });
            distances
                .to_vec()
                .into_iter()
                .enumerate()
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(Equal))
                .unwrap()
                .0
        })
        .collect()
}

fn compute_centroids(
    centroids: &mut ArrayViewMut2<f32>,
    assignments: &[usize],
    points: ArrayView2<f32>,
    mode: &KMeansMode,
) {
    centroids.map_inplace(|v| *v = 0_f32);
    let mut counts = vec![0_usize; centroids.nrows()];

    assignments
        .iter()
        .enumerate()
        .for_each(|(point_id, &cluster_id)| {
            centroids
                .row_mut(cluster_id)
                .add_assign(&points.row(point_id));
            counts[cluster_id] += 1;
        });

    counts.iter().enumerate().for_each(|(cluster_id, &count)| {
        centroids.row_mut(cluster_id).map_inplace(|v| {
            if count != 0 {
                *v /= count as f32
            }
        });

        if mode == &KMeansMode::Spherical {
            let l2 = centroids
                .row(cluster_id)
                .dot(&centroids.row(cluster_id))
                .sqrt();
            centroids.row_mut(cluster_id).map_inplace(|v| *v /= l2);
        }
    });
}
