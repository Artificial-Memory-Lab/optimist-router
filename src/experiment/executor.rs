use crate::experiment::config::{
    ClusteringAlgorithm, ClusteringConfig, Config, CovarianceSketch, IndexingConfig,
    RouterInstance, RouterType, RoutingConfig,
};
use crate::implement_serialization;
use crate::index::ivf_index::Query::Dense;
use crate::index::ivf_index::{IvfIndex, Query};
use crate::library::clustering::gmm::{GMMClustering, GMMMode};
use crate::library::clustering::kmeans::{KMeansClustering, KMeansMode};
use crate::library::clustering::Clustering;
use crate::library::routing::mean_router::MeanRouter;
use crate::library::routing::normalized_mean_router::NormalizedMeanRouter;
use crate::library::routing::optimist_router::OptimistRouter;
use crate::library::routing::scann_router::ScannRouter;
use crate::library::routing::subpartition_router::SubPartitionRouter;
use crate::library::routing::Router;
use crate::library::sketching::symmetric_psd_sketching::{
    DiagonalSketcher, IdentitySketcher, LowRankApproximation, SymmetricPSDSketcher,
};
use crate::library::utility;
use crate::library::utility::search_result::RetrievalResponse;
use crate::library::utility::serialization::FileSerializable;
use crate::library::utility::{random_split_ndarray, random_split_vector};
use ann_dataset::{AnnDataset, GroundTruth, Hdf5File, InMemoryAnnDataset, Metric, PointSet};
use itertools::Itertools;
use ndarray::Axis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct SerializableDynamicRouter {
    router: Box<dyn Router>,
}

impl FileSerializable for SerializableDynamicRouter {
    implement_serialization!();
}

impl Config {
    pub fn execute(&self) {
        if let Some(num_threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();
        }

        if let Some(name) = self.name.as_ref() {
            println!("--------------------------------------");
            println!("Executing experiment: {}", name);
            println!("--------------------------------------");
        }
        println!("Configuration: {:?}", self);

        println!("Reading the dataset from {}", self.dataset);
        let mut dataset = InMemoryAnnDataset::<f32>::read(self.dataset.as_str())
            .expect("Failed to read the dataset.");

        let points = dataset.get_data_points_mut();
        match self.metric {
            Metric::Euclidean => {}
            Metric::Cosine => {
                println!(
                    "Normalizing the data points because metric is {}",
                    Metric::Cosine
                );
                points.l2_normalize_inplace();
            }
            Metric::InnerProduct => {}
            _ => {
                todo!()
            }
        }

        let index = match &self.indexing {
            IndexingConfig::PreMade { path } => {
                println!("Loading the pre-made index from {}", path);
                IvfIndex::read_from_file(path.as_str())
            }
            IndexingConfig::New { clustering, output } => {
                println!("Making a new index as requested...");
                let index = IvfIndex::new(
                    points,
                    self.compile_clustering(clustering),
                    clustering.num_clusters,
                )
                    .unwrap();
                if let Some(t) = self.report_covariance_eigenvalue {
                    let values = index.partitions().covariance_eigenvalues(t);
                    println!("    {}-th eigenvalues: {}", t, values.iter().format(","));
                }
                if let Some(output) = output.as_ref() {
                    index.save_to_file(output);
                }
                index
            }
        };

        if let Some(routing_config) = self.routing.as_ref() {
            println!("Executing routing experiments...");
            println!();

            let queries = self.compile_queries(&dataset, routing_config);
            let ground_truth = self.compile_ground_truth(&dataset, routing_config);
            let min_points_probed = routing_config
                .percent_points_probed
                .iter()
                .map(|&percent| {
                    assert!((0_f32..=100_f32).contains(&percent));
                    if percent == 0_f32 {
                        return 1_usize;
                    }
                    ((percent / 100_f32) * (dataset.get_data_points().num_points() as f32)).ceil()
                        as usize
                })
                .collect::<Vec<_>>();

            // Construct routers.
            let routers = routing_config
                .routers
                .iter()
                .map(|router_instance| {
                    println!("Compiling router {:?}", router_instance);
                    let router =
                        self.compile_router(dataset.get_data_points(), &index, router_instance);
                    println!(
                        "  Router has average size per partition of {} vectors",
                        router.size() as f32
                            / ((index.partitions().num_dimensions() * index.partitions().len())
                            as f32)
                    );
                    router
                })
                .collect::<Vec<_>>();

            // Execute routers for max_top_k.
            let &max_top_k = routing_config.top_k.iter().max().unwrap();

            type PredictionError = Vec<f32>;
            type PointsProbed = Vec<f32>;
            type Accuracies = Vec<f32>;
            type TopK = usize;
            type ResultSet = HashMap<TopK, (PointsProbed, Accuracies, PredictionError)>;
            type RouterId = usize;
            let mut all_results: HashMap<RouterId, ResultSet> = HashMap::new();
            println!();
            routers.iter().enumerate().for_each(|(router_id, router)| {
                println!("Executing router {:?}", routing_config.routers[router_id]);

                let pb = utility::create_progress("Searching", queries.len());
                let results: Vec<Vec<RetrievalResponse>> = queries
                    .par_iter()
                    .map(|query| {
                        pb.inc(1);
                        index.retrieve(router.as_ref(), query, &min_points_probed, max_top_k)
                    })
                    .collect();
                pb.finish_and_clear();

                let mut mean_prediction_error: Vec<f32> = vec![];
                let mut points_probed: Vec<f32> = vec![];
                let mut accuracies: HashMap<usize, Vec<f32>> = HashMap::new();
                (0..min_points_probed.len()).for_each(|i| {
                    let actual_docs_examined = results
                        .iter()
                        .map(|r| r[i].actual_number_of_docs_probed as usize)
                        .collect::<Vec<usize>>();
                    let actual_docs_examined =
                        actual_docs_examined.iter().sum::<usize>() as f32 / results.len() as f32;

                    let prediction_error = results
                        .iter()
                        .map(|r| r[i].mean_relative_prediction_error)
                        .collect::<Vec<f32>>();
                    let prediction_error =
                        prediction_error.iter().sum::<f32>() / results.len() as f32;

                    let retrieved_set = results
                        .iter()
                        .map(|r| {
                            r[i].results
                                .iter()
                                .map(|srs| srs.docid as usize)
                                .collect::<Vec<usize>>()
                        })
                        .collect::<Vec<Vec<usize>>>();

                    points_probed.push(actual_docs_examined);
                    mean_prediction_error.push(prediction_error);
                    routing_config.top_k.iter().for_each(|&k| {
                        let trimmed_set = retrieved_set
                            .iter()
                            .map(|set| set[..min(k, set.len())].to_owned())
                            .collect::<Vec<_>>();

                        let recall = ground_truth.mean_recall(&trimmed_set).unwrap();
                        accuracies.entry(k).or_default().push(recall);
                    });
                });

                routing_config.top_k.iter().for_each(|k| {
                    all_results.entry(router_id).or_default().insert(
                        *k,
                        (
                            points_probed.clone(),
                            accuracies.get(k).unwrap().clone(),
                            mean_prediction_error.clone(),
                        ),
                    );
                });
            });

            println!();
            println!("Results");
            routing_config.top_k.iter().for_each(|&k| {
                println!("Top-k = {}", k);

                routers.iter().enumerate().for_each(|(router_id, _)| {
                    let router_results = all_results.get(&router_id).unwrap().get(&k).unwrap();

                    println!("  Router {:?}", routing_config.routers[router_id]);
                    println!(
                        "    Mean relative prediction error: {}",
                        router_results.2.iter().format(",")
                    );
                    println!(
                        "    PointsProbed: {:3}",
                        router_results.0.iter().format(",")
                    );
                    println!("    Accuracies: {:3}", router_results.1.iter().format(","));
                    println!();
                });
            });
        }
    }

    fn compile_clustering(&self, clustering_config: &ClusteringConfig) -> Box<dyn Clustering> {
        match clustering_config.algorithm {
            ClusteringAlgorithm::StandardKmeans => Box::new(KMeansClustering::new(
                self.seed,
                KMeansMode::Standard,
                clustering_config.num_iterations,
                clustering_config.samples_per_centroid,
            )),
            ClusteringAlgorithm::SphericalKmeans => Box::new(KMeansClustering::new(
                self.seed,
                KMeansMode::Spherical,
                clustering_config.num_iterations,
                clustering_config.samples_per_centroid,
            )),
            ClusteringAlgorithm::HardGmm { assignment_top_k } => Box::new(GMMClustering::new(
                self.seed,
                GMMMode::Hard,
                clustering_config.num_iterations,
                clustering_config.samples_per_centroid,
                assignment_top_k,
            )),
        }
    }

    fn compile_queries<'a>(
        &'a self,
        dataset: &'a InMemoryAnnDataset<f32>,
        routing_config: &RoutingConfig,
    ) -> Vec<Query> {
        let test_set = dataset.get_test_query_set().unwrap().get_points();
        let mut queries = test_set
            .get_dense()
            .unwrap()
            .axis_iter(Axis(0))
            .map(Dense)
            .collect::<Vec<_>>();

        if let Some(percent) = routing_config.percent_queries {
            assert!((0_f32..=100_f32).contains(&percent));
            let ratio = percent / 100_f32;

            queries = random_split_vector(self.seed, &queries, ratio).0;
        }
        queries
    }

    fn compile_ground_truth(
        &self,
        dataset: &InMemoryAnnDataset<f32>,
        routing_config: &RoutingConfig,
    ) -> GroundTruth {
        let mut ground_truth = dataset
            .get_test_query_set()
            .unwrap()
            .get_ground_truth(&self.metric)
            .unwrap()
            .to_owned();

        if let Some(percent) = routing_config.percent_queries {
            assert!((0_f32..=100_f32).contains(&percent));
            let ratio = percent / 100_f32;
            let subset =
                random_split_ndarray(self.seed, ground_truth.get_neighbors().view(), ratio).0;
            ground_truth = GroundTruth::new(subset);
        }
        ground_truth
    }

    fn compile_router(
        &self,
        data_points: &PointSet<f32>,
        index: &IvfIndex,
        router: &RouterInstance,
    ) -> Box<dyn Router> {
        match router {
            RouterInstance::PreMade { path } => {
                println!("Loading the pre-made router from {}", path);
                let serialized_router = SerializableDynamicRouter::read_from_file(path.as_str());
                serialized_router.router
            }
            RouterInstance::New {
                router_type,
                output,
            } => {
                println!("Making a new router as requested...");

                let trained_router: Box<dyn Router> = match router_type {
                    RouterType::Mean => Box::new(MeanRouter::build(index.partitions())),
                    RouterType::NormalizedMean => {
                        Box::new(NormalizedMeanRouter::build(index.partitions()))
                    }
                    RouterType::Optimist {
                        delta,
                        covariance_sketch,
                    } => Box::new(OptimistRouter::build(
                        index.partitions(),
                        *delta,
                        self.compile_sketcher(covariance_sketch),
                    )),
                    RouterType::SubPartition { clustering_config } => {
                        if data_points.get_sparse().is_some() {
                            unimplemented!()
                        }

                        let clustering = self.compile_clustering(clustering_config);
                        Box::new(SubPartitionRouter::build(
                            self.seed,
                            index.partitions(),
                            clustering,
                            clustering_config.num_clusters,
                        ))
                    }
                    RouterType::Scann { threshold } => {
                        Box::new(ScannRouter::build(index.partitions(), *threshold))
                    }
                };

                if let Some(output) = output {
                    println!("Storing the new router at {}", output);

                    let serialized_router = SerializableDynamicRouter {
                        router: trained_router,
                    };
                    serialized_router.save_to_file(output);
                    return serialized_router.router;
                }

                trained_router
            }
        }
    }

    fn compile_sketcher(
        &self,
        covariance_sketch: &Option<CovarianceSketch>,
    ) -> Box<dyn SymmetricPSDSketcher> {
        if let Some(sketcher) = covariance_sketch {
            match sketcher {
                CovarianceSketch::LowRank { rank } => {
                    return Box::new(LowRankApproximation::new(*rank));
                }
                CovarianceSketch::Diagonal => return Box::<DiagonalSketcher>::default(),
            }
        }
        Box::<IdentitySketcher>::default()
    }
}
