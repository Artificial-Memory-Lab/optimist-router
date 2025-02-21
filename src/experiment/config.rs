use ann_dataset::Metric;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    /// Name of the experiment.
    pub name: Option<String>,

    /// Seed for randomized algorithms.
    pub seed: Option<u64>,

    /// Number of threads to use for indexing and retrieval.
    pub num_threads: Option<usize>,

    /// Path to a dataset in the `ann_dataset::AnnDataset` format.
    pub dataset: String,

    /// ANN search metric that will be used to query the index.
    /// This is also used to look up the ground-truth (i.e., exact nearest neighbors) from the
    /// dataset.
    pub metric: Metric,

    /// Indexing configuration.
    pub indexing: IndexingConfig,

    /// Routing configuration.
    pub routing: Option<RoutingConfig>,

    /// If provided, reports the specified eigenvalue of D^{-1/2} \Sigma D^{-1/2} for each shard,
    /// where \Sigma is the covariance matrix and D is its diagonal.
    pub report_covariance_eigenvalue: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum IndexingConfig {
    PreMade {
        /// Path to the serialized index.
        path: String,
    },
    New {
        /// Clustering configuration.
        clustering: ClusteringConfig,

        /// Path to file name to store the index.
        output: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ClusteringConfig {
    /// Clustering algorithm.
    pub algorithm: ClusteringAlgorithm,

    /// Number of clusters.
    pub num_clusters: usize,

    /// Number of iterations.
    pub num_iterations: Option<usize>,

    /// Number of samples per partition to use in training.
    /// This typically a number greater than 39.
    pub samples_per_centroid: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ClusteringAlgorithm {
    /// Standard KMeans using LLoyd's iterative algorithm.
    StandardKmeans,

    /// Similar to `StandardKMeans` but where cluster means are normalized
    /// every time they are computed.
    SphericalKmeans,

    /// Gaussian Mixture Model with hard assignment.
    HardGmm {
        /// When assigning a point to a cluster in the hard-gmm setting,
        /// we first find the `assignment-top-k` partitions whose means are closest to a point, and only
        /// then use the covariance of these top partitions to find the predicted partition for that point.
        /// This helps speed up clustering dramatically with minimal impact on partition quality.
        assignment_top_k: Option<usize>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RoutingConfig {
    /// Types of Routers to use.
    pub routers: Vec<RouterInstance>,

    /// Cutoff for search
    pub cutoff: Cutoff,

    /// Top-k.
    pub top_k: Vec<usize>,

    /// Percent queries to use for evaluation (between 0 and 100).
    /// The value 0 is a special case: If this is 0, we probed the top partition only.
    pub percent_queries: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Cutoff {
    PercentPoints {
        /// Percent data points to probe (between 0 and 100).
        percent_points_probed: Vec<f32>,
    },
    NumClusters {
        /// Number of clusters to probe
        num_clusters: Vec<usize>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RouterInstance {
    PreMade {
        /// Path from which the router will be read.
        path: String,
    },
    New {
        /// Type of router.
        router_type: RouterType,

        /// Path to a file where the router will be stored.
        output: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RouterType {
    Mean,

    NormalizedMean,

    Optimist {
        delta: f32,
        covariance_sketch: Option<CovarianceSketch>,
    },

    SubPartition {
        clustering_config: ClusteringConfig,
    },

    Scann {
        threshold: f32,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum CovarianceSketch {
    /// Produces a low-rank approximation of the covariance.
    LowRank { rank: usize },
    /// Keeps only the diagonal entries of the covariance.
    Diagonal,
}
