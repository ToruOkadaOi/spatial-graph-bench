# Reproducing benchmark preprocessing

This guide documents how to rebuild the frozen feature arrays from the source data, with fitted preprocessing objects for all four datasets.

## Stereo-seq feature construction

The Stereo-seq features use all 12,936 sorted genes shared across Stage44, Stage54, and Stage57. The preprocessing runner retains this complete gene panel and reproduces the released arrays exactly.

The frozen `feature_manifest.json` records `n_hvg: 2000` as a pipeline setting. The PCA arrays themselves were generated from the complete 12,936-gene panel. The fitted preprocessing object and the released arrays provide the exact feature construction used for this dataset.

## Rebuild and verify

Download and checksum the raw files listed in `data/census.json`, then place them at the paths specified there. Run:

```bash
uv sync --frozen
uv run python scripts/run_preprocessing.py --dataset stereoseq_axolotl_telencephalon --split developmental_three_stage
```

The runner checks that the saved fitted object reproduces validation and test features. On the locked release environment, the generated train, validation, and test PCA arrays also match the frozen arrays byte-for-byte:

| Partition | SHA-256 of PCA array |
| --- | --- |
| Train | `d5b00ed97bc3b135e8fba337bd68899b3b836cca9b82dbaff27fe7f51e3710e0` |
| Validation | `a30c79a9fbe6d6c9428d8438901d1ce63a6461d9f8abd19caa1b48533d58c9dd` |
| Test | `74f975dfb1e8e7debd09fa1649eff09086c382826a03790c44e8d52a573ba472` |

The v0.3.2 fitted-preprocessor supplement, `inductive_v3_preprocessors_v0.3.2.tar.gz`, contains all four datasets (SHA-256 `56a5c75139c984c2b3f3323f17b70033eef4039228e239e3b151f1590522032f`). Its Stereo-seq object uses the full ordered 12,936-gene panel, train-fitted scaler parameters, and train-fitted PCA parameters. Load it with `FittedPreprocessor.load(directory)` and transform counts in the recorded gene order.

## Reproducibility scope

Frozen feature arrays plus released model weights reproduce reported inference without raw data or retraining. Reconstructing raw-to-PCA features additionally requires the matching raw dataset snapshots and fitted preprocessing objects. Dataset terms remain those of their respective providers.
