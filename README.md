# Optimist Router

## Obtaining the data

You can download all relevant datasets using the following links:
- [Deep](https://huggingface.co/datasets/artificial-memory-lab/ann-datasets/blob/main/deep-image-96.hdf5)
- [Glove](https://huggingface.co/datasets/artificial-memory-lab/ann-datasets/blob/main/glove-200.hdf5)
- [MsMarco](https://huggingface.co/datasets/artificial-memory-lab/ann-datasets/blob/main/msmarco-v1-passage-allMinniLML6V2.hdf5)
- [Music](https://huggingface.co/datasets/artificial-memory-lab/ann-datasets/blob/main/music-100.hdf5)
- [NQ](https://huggingface.co/datasets/artificial-memory-lab/ann-datasets/blob/main/nq-openai_ada002-1536.hdf5)
- [Text2Image](https://huggingface.co/datasets/artificial-memory-lab/ann-datasets/blob/main/text2image-10M.hdf5)

Copy the above datasets to `${PATH_TO_ROOT_DIR}/datasets/`.

## Running experiments

You can run any number of experiments with the invocation
of the following command line:

```bash
cargo run --release -- --config ${PATH_TO_YAML_CONFIG} --root ${PATH_TO_ROOT_DIR}
```

The YAML configuration file must contain a manifest of all experiments
you wish to run. For a detailed structure of the configuration, see
`crate::experiment::Config`. You can find configs for the experiments reported
in our paper in `experiments/`.

Note that, the degree of parallelism of the code can be set through
the `RAYON_NUM_THREADS` environment variable.
Here is an example setting the computation up to 4 threads:

```bash
export RAYON_NUM_THREADS=4
```

## Citation

Please use the following to cite our work:
```latex
@inproceedings{
  bruch2025optimistic,
  title={Optimistic Query Routing in Clustering-based Approximate Maximum Inner Product Search},
  author={Sebastian Bruch and Aditya Krishnan and Franco Maria Nardini},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=yzvpEHNL70}
}
```
