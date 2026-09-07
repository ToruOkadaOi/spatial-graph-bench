# Benchmark Results & Release Artifacts Index

This index provides a verified catalog of all official artifact releases, frozen baseline benchmarks, and cryptographically signed delivery bundles in `spatial-graph-bench`.

---

## 1. Official GitHub Release Bundles

| Release Tag | Dataset | Split | Included Artifacts | Compressed Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`v0.1.0-merfish-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) | `merfish_mouse_spinal_cord` | `mouse_held_out_canonical` | PCA 50 features + 7 graph bundles (`spatial_knn_k6/12`, rewired controls, shuffled controls, bipartite ref) | 16.09 MB | `3f92e579fbc081cd17ea5d04d406f36c9dee1c20b06030584f3900abe8964e8b` |

### How to Fetch Release Assets
```bash
# Automated fetch & invariant verification:
uv run python scripts/manage_release_artifacts.py fetch --tag v0.1.0-merfish-inputs

# Or via GitHub CLI directly:
gh release download v0.1.0-merfish-inputs
shasum -a 256 -c sha256sums_merfish_mouse_spinal_cord_mouse_held_out_canonical.txt
tar -xzf artifacts_merfish_mouse_spinal_cord_mouse_held_out_canonical_inputs.tar.gz
```

---

## 2. Frozen Baseline Benchmark Snapshots

Baseline snapshots are cryptographically locked and stored in `audits/baselines_snapshot/` to prevent post-hoc moving target bias.

### Dataset 1: MERFISH Mouse Spinal Cord (`mouse_held_out_canonical`)
- **Snapshot Path**: `audits/baselines_snapshot/merfish_mouse_spinal_cord/mouse_held_out_canonical/baselines_summary.json`
- **Total Cells**: 41,267 (Train: 26,866, Val: 5,894, Test: 8,507)
- **Number of Classes**: 60 (100% evaluated, 0 excluded)
- **Features**: 50 Principal Components (fit strictly on train partition)

| Model | Evaluated Seeds | Test Macro-F1 (Mean $\pm$ Std) | Test Balanced Acc (Mean $\pm$ Std) | Parity Halfwidth ($\pm 2\sigma$) | Parity Band Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spatially Ignorant MLP** | 10 (seeds 42–51) | $\mathbf{0.5273 \pm 0.0035}$ | $\mathbf{0.5247 \pm 0.0036}$ | $\pm 0.0069$ | $[0.5204, 0.5342]$ |
| **Random Forest (200 trees)**| 1 (seed 42) | $0.4683$ | $0.4496$ | N/A | N/A |

> **TOST Equivalence Margin**: $\epsilon = \mathbf{0.0069}$. Any spatial GNN failing to score $> 0.5342$ Macro-F1 is statistically indistinguishable from, or worse than, gene expression alone.

---

## 3. Results Directory Layout

When running or ingesting benchmark runs, results are organized strictly as:
```text
artifacts/results/{dataset_name}/{split_id}/{run_id}/
├── metrics_summary.json     # Final test/val metrics (macro_f1, balanced_accuracy, label_coverage)
├── run_manifest.json        # Cryptographic lineage (model_name, hyperparameters, parent hashes)
├── test_preds.npy           # Per-cell test predictions
├── training_history.csv     # Epoch-by-epoch train/val loss and macro-F1
```
