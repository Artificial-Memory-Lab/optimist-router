# Optimist Router

## Obtaining the data

All datasets must be in the [AnnDataset](https://crates.io/crates/ann_dataset)
format. You can find datasets used in
[our paper](https://arxiv.org/abs/2405.12207) in this [bucket](https://console.cloud.google.com/storage/browser/ivf-partition-selection-data).

## Running experiments

You can run any number of experiments with the invocation of the following command line:

```bash
cargo run --release -- --config ${PATH_TO_YAML_CONFIG}
```

The YAML configuration file must contain a manifest of all experiments
you wish to run. For a detailed structure of the configuration, see
`crate::experiment::Config`. You can find an example configuration in
`data/experiment_config.yaml`.

You can find configs used in
[our paper](https://arxiv.org/abs/2405.12207) along with output logs in `experiments/`.

The degree of parallelism of the code can be set through the RAYON_NUM_THREADS env
variable. Here is an example setting the computation up to 64 threads:

```bash
export RAYON_NUM_THREADS=64
```

## Citation

Please use the following to cite our work:
```latex
@misc{bruch2024optimistic,
      title={Optimistic Query Routing in Clustering-based Approximate Maximum Inner Product Search}, 
      author={Sebastian Bruch and Aditya Krishnan and Franco Maria Nardini},
      year={2024},
      eprint={2405.12207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
