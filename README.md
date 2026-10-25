# Spatial graph benchmark

Does spatial message passing improve cell-type classification over expression-only baselines?

**Status: historical results reanalyzed; strict-inductive GNN conclusions require corrected training.**

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

The preserved dataset contains 1,164 runs: four platforms × (280 GNN runs + ten MLP seeds + one random forest). Models use identical frozen PCA inputs within each split. These were fixed model configurations; a saved equal-budget tuning study is not available.

The table illustrates the fixed GraphSAGE spatial k12 configuration, rather than selecting a winner on the test set. Statistical decisions describe the historical fits only.

| Platform | Test / evaluated cells | Evaluated classes | Class coverage | MLP F1 ± sample SD | GraphSAGE k12 F1 | Paired lift | Holm decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stereo-seq | 4,410 / 4,209 | 15 | 88.24% | 0.7282 ± 0.0140 | 0.7665 | +0.0383 | Inconclusive |
| MERFISH | 8,507 / 8,507 | 60 | 100.00% | 0.5273 ± 0.0035 | 0.5020 | -0.0253 | Negative lift |
| Open-ST | 34,072 / 34,072 | 14 | 100.00% | 0.3946 ± 0.0027 | 0.3962 | +0.0016 | Inconclusive |
| Xenium | 85,880 / 85,880 | 20 | 100.00% | 0.7635 ± 0.0120 | 0.7514 | -0.0121 | Inconclusive |

Stereo-seq holds out Stage57 after training on Stage44 and Stage54. MERFISH holds out mouse F4. Open-ST holds out section 19 after training on section 6. Xenium evaluates ShamL → ShamR; this split does not test injury-stage generalization.

Test evaluation excludes true cells from unseen classes and averages over training/test intersection labels. Epsilon is max(0.005, twice the MLP sample SD). The reanalysis applies Holm adjustment to all 336 hypotheses across the four platforms. Equivalence and inconclusive evidence are separate outcomes. Boundary and per-class diagnostics are descriptive; they do not by themselves establish an oversmoothing mechanism.

## Reproduce the analysis without training

```bash
uv sync --frozen --all-extras
./scripts/reproduce_all.sh
```

The script fetches missing verified release bundles, validates the versioned byte index, recomputes all test metrics and statistics from predictions, then regenerates reports and figures. It never trains models or overwrites original run outputs. GitHub CLI access to the releases is needed when artifacts are absent. Runtime includes downloads and scales with network and disk speed; this is not a 30-second guarantee.

For an existing offline artifact checkout: `./scripts/reproduce_all.sh --offline`. For reports only: `./scripts/reproduce_all.sh --offline --no-figures`.

See [corrected analysis method](docs/reanalysis-v2.md), [all 112 comparisons](results/reports/table_benchmark_synthesis.md), [release index](docs/results-index.md), and [reproduction guide](REPRODUCE.md).

Original predictions, probabilities, metrics, histories and manifests remain immutable in `artifacts/results/`. Corrected metrics and analysis inputs are versioned under `results/reports/reanalysis_v2/`. The new integrity index preserves and records legacy PCA hash discrepancies instead of rewriting old manifests.

## Development

```bash
uv run --frozen pytest tests
uv run --frozen ruff check src tests scripts
uv run --frozen ruff format --check src tests scripts
```

MIT license. See [CITATION.md](CITATION.md) for citation information.
