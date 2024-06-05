pub mod mean_router;
pub mod normalized_mean_router;
pub mod optimist_router;
pub mod scann_router;
pub mod subpartition_router;

use crate::library::file_system::Partitions;
use ndarray::ArrayView1;

#[typetag::serde(tag = "router")]
pub trait Router: Sync + Send {
    /// Conditioned on a query vector, a router sorts partitions by their likelihood of containing
    /// the nearest neighbor. The output of this particular function is a list of partition ids
    /// in that order along with their predicted scores.
    fn sort_partitions(&self, query: ArrayView1<f32>, partitions: &Partitions)
        -> Vec<(usize, f32)>;

    /// Returns the total amount of memory necessary to represent the router.
    fn size(&self) -> usize;
}
