# Running the two sampling studies on Kaggle (CPU)

`cutin_kernel.py` runs `cutin.study_cluster` (and, if you edit the list, `cutin.study_strata`) as a Kaggle script kernel.
Build a zip with `src/cutin/*.py`, `configs/default.toml`, `data/signals.npz` and `data/cutins.parquet` under a top folder
`cutin-atlas/`, upload it as the private dataset in `dataset-metadata.json`, then `kaggle kernels push -p kaggle`.
Both studies also run on a laptop (about 40 minutes each).
