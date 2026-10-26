# Reproducing the corrected v0.3.0 benchmark

## Frozen release artifacts

The v0.3.0 code release is archived at [Zenodo DOI 10.5281/zenodo.23661930](https://doi.org/10.5281/zenodo.23661930). The corrected inputs, graph bundles, model weights, predictions, metrics, and manifests are distributed as separate release archives. Verify each downloaded file against `inductive_v3_release.SHA256SUMS` before use.

The frozen PCA matrices are the authoritative model inputs. They allow predictions and reported metrics to be reproduced without rerunning preprocessing or training. The model weights and prediction files are in the extras archive.

## Rebuilding frozen preprocessing

The raw datasets are not redistributed here. Download the files listed in `data/census.json`, verify their SHA-256 checksums, and place them under `data/raw/` using the paths in each `raw_data_snapshot_id`. Then install the locked environment and run:

```bash
uv sync --frozen
uv run python scripts/run_preprocessing.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical
uv run python scripts/run_preprocessing.py --dataset openst_human_lymph_node --split section_held_out_canonical
uv run python scripts/run_preprocessing.py --dataset stereoseq_axolotl_telencephalon --split developmental_three_stage
uv run python scripts/run_preprocessing.py --dataset xenium_mouse_kidney --split replicate_held_out_canonical
```

Each generated bundle includes `preprocessing_model.json` and `preprocessing_parameters.npz`. Load the safe, pickle-free model and transform a new count matrix in the original gene order:

```python
from spatial_graph_bench.preprocessing.fitted import FittedPreprocessor

preprocessor = FittedPreprocessor.load("artifacts/preprocessed/DATASET/SPLIT")
X_pca = preprocessor.transform(count_matrix, feature_names=gene_names)
```

The run checks that the fitted object exactly reproduces the validation and test matrices it just generated. Training PCA outputs are retained as frozen arrays because scikit-learn's `PCA.fit_transform` output can differ from applying `transform` to the same training data.

## Current portability verification

Using the raw files in the original local checkout, the generated validation and test feature hashes match the frozen release hashes for MERFISH, Open-ST, and Xenium. Stereo-seq did not match: its regenerated PCA arrays differ from the frozen benchmark arrays even though the raw file checksums and split cell IDs match. The regenerated Stereo-seq fitted object is therefore not compatible with the released model weights and must not be used with them. Use the frozen Stereo-seq PCA arrays for the released predictions until the original fitted feature selection/PCA state is recovered.

The fitted-object supplement must be treated as partial until a Stereo-seq transform that matches the frozen feature hashes is recovered and verified. No model retraining is required if that preprocessing state can be recovered.

## Reproducibility boundaries

- Reproducing released predictions and metrics from frozen PCA inputs and saved weights does not require raw-data access or model training.
- Rebuilding raw-to-PCA inputs requires the matching fitted preprocessing state and raw dataset versions.
- Repeating model training may produce small numerical differences across platforms, BLAS implementations, and accelerator versions; the frozen weights and predictions remain the reference for the reported release.
- Source datasets retain their own terms. The code's MIT license does not relicense the source datasets or derived data.
