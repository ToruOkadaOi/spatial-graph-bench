# spatial-graph-bench

**When does a spatial graph help predict a cell’s type?**

A benchmark exploring whether cell–cell spatial graphs add useful information beyond gene expression in spatial transcriptomics. It compares four GNN architectures with expression-only MLP and random forest baselines across four platforms and distinct biological settings.

**1,164 runs · 4 platforms · 7 graph topologies · 10 random seeds**

## The experiment

Each cell has gene-expression features and a physical location. The benchmark asks whether connecting nearby cells improves cell-type classification—and how that depends on tissue context, graph construction, and model architecture.

The comparison spans:

- **Stereo-seq:** developing axolotl telencephalon, with a developmental-stage holdout.
- **MERFISH:** adult mouse spinal cord, with an animal holdout.
- **Open-ST:** human metastatic lymph node, with a tissue-section holdout.
- **Xenium:** mouse kidney, with a ShamL → ShamR replicate holdout.

GCN, GraphSAGE, GAT, and GIN share frozen 50-dimensional PCA features with the non-spatial baselines. Spatial k-NN graphs at two neighborhood sizes are compared with rewired graphs, shuffled coordinates, and expression-space reference graphs.

## Selected results

Mean test macro-F1 across ten seeds, comparing the MLP with GraphSAGE on spatial k-NN graphs (`k=12`):

| Platform | Evaluated cells | Cell types | MLP | GraphSAGE | Paired difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stereo-seq | 4,209 | 15 | 0.7282 | 0.7665 | +0.0383 |
| MERFISH | 8,507 | 60 | 0.5273 | 0.5020 | −0.0253 |
| Open-ST | 34,072 | 14 | 0.3946 | 0.3962 | +0.0016 |
| Xenium | 85,880 | 20 | 0.7635 | 0.7514 | −0.0121 |

Spatial proximity alone did not consistently improve classification. GraphSAGE reached a higher mean score in the developing brain dataset, while the other settings showed smaller differences or lower scores. The broader architecture and topology comparisons explore where neighborhood aggregation helps preserve—or loses—cell-type information.

[Explore all 112 comparisons →](results/reports/table_benchmark_synthesis.md)

## What’s in the project

- **Data pipeline:** acquisition manifests, frozen splits, train-fitted preprocessing, and feature bundles.
- **Graph construction:** spatial neighborhoods, topology controls, and bipartite reference connectivity.
- **Model evaluation:** paired seed comparisons, per-class metrics, boundary stratification, and multiple-testing correction.
- **Reproducibility tooling:** locked dependencies, SHA-256 artifact verification, GPU delivery checks, and report generation from saved predictions.

Built with **Python, PyTorch, PyTorch Geometric, Scanpy, scikit-learn, and uv**.

## Run it

```bash
uv sync --frozen --all-extras
./scripts/reproduce_all.sh
```

The workflow fetches the saved benchmark artifacts and regenerates the analysis, tables, and figures. With the artifacts already available locally:

```bash
./scripts/reproduce_all.sh --offline
```

## Explore further

- [Results by platform](docs/results-synthesis.md)
- [Methods and evaluation](docs/reanalysis-v2.md)
- [Reproduction guide](REPRODUCE.md)
- [Dataset and release artifacts](docs/results-index.md)
- [Related single-cell benchmark: scgraph-bench](https://github.com/ToruOkadaOi/scgraph-bench)

MIT license · [Citation](CITATION.md)
