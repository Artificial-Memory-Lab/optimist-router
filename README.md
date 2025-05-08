# Optimist Router

## Obtaining the data

You can download all relevant datasets using the following links:
- [Deep](http://storage.googleapis.com/anonymized-ann-data/ann/deep-image-96.hdf5)
- [Glove](http://storage.googleapis.com/anonymized-ann-data/ann/glove-200.hdf5)
- [MsMarco](http://storage.googleapis.com/anonymized-ann-data/ann/msmarco-v1-passage-allMinniLML6V2.hdf5)
- [Music](http://storage.googleapis.com/anonymized-ann-data/ann/music-100.hdf5)
- [NQ](http://storage.googleapis.com/anonymized-ann-data/ann/nq-openai_ada002-1536.hdf5)
- [Text2Image](http://storage.googleapis.com/anonymized-ann-data/ann/text2image-10M.hdf5)

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
@misc{bruch2024optimistic,
      title={Optimistic Query Routing in Clustering-based Approximate Maximum Inner Product Search}, 
      author={Sebastian Bruch and Aditya Krishnan and Franco Maria Nardini},
      year={2024},
      eprint={2405.12207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
