# Benchmark Results & Release Artifacts Index

This index provides a verified catalog of all official artifact releases, frozen baseline benchmarks, and cryptographically signed delivery bundles in `spatial-graph-bench`.

---

## 1. Official GitHub Release Bundles

### 1.1 Input Bundles (Certified Features & Graph Topologies)

| Release Tag | Dataset | Split | Included Artifacts | Compressed Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`v0.1.0-merfish-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) | `merfish_mouse_spinal_cord` | `mouse_held_out_canonical` | PCA 50 features + 7 graph bundles | 16.09 MB | `3f92e579fbc081cd17ea5d04d406f36c9dee1c20b06030584f3900abe8964e8b` |
| [`v0.1.0-stereoseq-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-stereoseq-inputs) | `stereoseq_axolotl_telencephalon` | `developmental_three_stage` | PCA 50 features + 7 graph bundles | 3.14 MB | `0635daf89ba885e1c4421933522a36cb40ae62bdaa2acdf668f104c72137ffd5` |
| [`v0.1.0-openst-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-openst-inputs) | `openst_human_lymph_node` | `section_held_out_canonical` | PCA 50 features + 7 graph bundles | 27.32 MB | `83c8d2b416ce07cf3fff876fabd8695dbb8e5aa92b67fcfea55846f84779f115` |
| [`v0.1.0-xenium-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-xenium-inputs) | `xenium_mouse_kidney` | `replicate_held_out_canonical` | PCA 50 features + 7 graph bundles | 84.56 MB | `42e4512e7cbe077721537f4175367d3f3b41194b6f7f01036cc9dfefc1d5a33e` |

### 1.2 Results Bundles (Full 280-Run Audited Execution Bundles)

| Release Tag | Dataset | Split | Included Artifacts | Compressed Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`v0.2.0-merfish-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-merfish-results) | `merfish_mouse_spinal_cord` | `mouse_held_out_canonical` | 280 GNN runs + 11 baselines | 503.06 MB | `6fbc4b85c720414d2a36467d665b7ff92c1775bbc034c22e2c7d508c6e6ae3db` |
| [`v0.2.0-stereoseq-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-stereoseq-results) | `stereoseq_axolotl_telencephalon` | `developmental_three_stage` | 280 GNN runs + 11 baselines | 82.38 MB | `a2e1802e1d9cccdb4f245eab72cfa56f9cf01dc02d2657e56f3535cd6f3b45e8` |
| [`v0.2.0-openst-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-openst-results) | `openst_human_lymph_node` | `section_held_out_canonical` | 280 GNN runs + 11 baselines | 466.49 MB | `6a633727a5f00806a114a36788b16fcc6598f66166b5ed1c53da3a05aa2fcf41` |
| [`v0.2.0-xenium-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-xenium-results) | `xenium_mouse_kidney` | `replicate_held_out_canonical` | 280 GNN runs + 11 baselines | 1753.79 MB | `44c94734b8d5c3a8ef76a6ffebc6283aa82507ddde7da6ce7c0722d0615564b6` |

---

## 2. Frozen Baseline Benchmark Snapshots

Baseline snapshots are cryptographically locked and stored in `audits/baselines_snapshot/` to prevent post-hoc moving target bias.

| Dataset & Split | Test Cells | Evaluated Classes | MLP Test Macro-F1 ($N=10$) | Random Forest F1 | Parity Band ($\pm \epsilon$) | TOST Equivalence Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MERFISH** (`mouse_held_out_canonical`) | 8,507 | 60 | $0.5273 \pm 0.0035$ | $0.4683$ | $\pm 0.0069$ | $\epsilon = 0.0069$ |
| **Stereo-seq** (`developmental_three_stage`) | 24,008 | 16 | $0.6974 \pm 0.0107$ | $0.6698$ | $\pm 0.0215$ | $\epsilon = 0.0215$ |
| **Open-ST** (`section_held_out_canonical`) | 17,998 | 8 | $0.3946 \pm 0.0027$ | $0.3365$ | $\pm 0.0054$ | $\epsilon = 0.0054$ |
| **10x Xenium** (`replicate_held_out_canonical`) | 85,880 | 20 | $0.7635 \pm 0.0120$ | $0.7789$ | $\pm 0.0241$ | $\epsilon = 0.0241$ |

---

## 3. Detailed Empirical Reports

- [docs/results-synthesis.md](results-synthesis.md): Master Quad-Modality Comparative Synthesis Report
- [docs/results-merfish.md](results-merfish.md): MERFISH Mouse Spinal Cord Detailed Report
- [docs/results-stereoseq.md](results-stereoseq.md): Stereo-seq Axolotl Telencephalon Detailed Report
- [docs/results-openst.md](results-openst.md): Open-ST Human Lymph Node Detailed Report
- [docs/results-xenium.md](results-xenium.md): 10x Xenium Mouse Kidney Detailed Report
