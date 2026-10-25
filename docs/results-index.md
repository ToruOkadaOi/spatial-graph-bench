# Release artifacts and corrected analysis

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

MERFISH, Stereo-seq and Open-ST v1 result archives omit baselines; Xenium includes them. Four local Stereo-seq runs also differ from the old release. The exact missing/differing local historical metrics, manifests and predictions are preserved in `audits/reanalysis_recovery_v2.zip`, with reasons listed in its JSON manifest and bytes verified against the integrity index. Probabilities and histories are optional for prediction reanalysis; any present originals are still checked.

| Dataset | Kind | Release | Archive SHA-256 |
| --- | --- | --- | --- |
| merfish_mouse_spinal_cord | inputs | [v0.1.0-merfish-inputs](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) | `3f92e579fbc081cd17ea5d04d406f36c9dee1c20b06030584f3900abe8964e8b` |
| merfish_mouse_spinal_cord | results | [v0.2.0-merfish-results](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-merfish-results) | `6fbc4b85c720414d2a36467d665b7ff92c1775bbc034c22e2c7d508c6e6ae3db` |
| stereoseq_axolotl_telencephalon | inputs | [v0.1.0-stereoseq-inputs](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-stereoseq-inputs) | `0635daf89ba885e1c4421933522a36cb40ae62bdaa2acdf668f104c72137ffd5` |
| stereoseq_axolotl_telencephalon | results | [v0.2.0-stereoseq-results](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-stereoseq-results) | `a2e1802e1d9cccdb4f245eab72cfa56f9cf01dc02d2657e56f3535cd6f3b45e8` |
| openst_human_lymph_node | inputs | [v0.1.0-openst-inputs](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-openst-inputs) | `83c8d2b416ce07cf3fff876fabd8695dbb8e5aa92b67fcfea55846f84779f115` |
| openst_human_lymph_node | results | [v0.2.0-openst-results](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-openst-results) | `6a633727a5f00806a114a36788b16fcc6598f66166b5ed1c53da3a05aa2fcf41` |
| xenium_mouse_kidney | inputs | [v0.1.0-xenium-inputs](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-xenium-inputs) | `42e4512e7cbe077721537f4175367d3f3b41194b6f7f01036cc9dfefc1d5a33e` |
| xenium_mouse_kidney | results | [v0.2.0-xenium-results](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-xenium-results) | `44c94734b8d5c3a8ef76a6ffebc6283aa82507ddde7da6ce7c0722d0615564b6` |

Corrected scores and sample counts are generated from frozen arrays in the [README](../README.md) and [dataset reports](results-synthesis.md). Original baseline summaries remain historical records of their original metric convention.
