use ndarray::{Array2, ArrayView2};

pub mod gmm;
pub mod kmeans;

pub trait Clustering: Sync + Send {
    /// Clusters a set of points into a fixed number of clusters.
    /// Returns a set of cluster representatives with shape [num_clusters, num_dimensions]
    /// and cluster assignments for each point.
    fn cluster(
        &self,
        points: ArrayView2<f32>,
        num_clusters: usize,
        verbose: bool,
    ) -> (Array2<f32>, Vec<usize>);
}
