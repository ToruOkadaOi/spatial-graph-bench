# Reproduce historical prediction analysis

The maintained non-training workflow is:

```bash
uv sync --frozen --all-extras
./scripts/reproduce_all.sh
```

It downloads missing frozen input/result releases, verifies the immutable byte index, recomputes all test metrics and global Holm decisions from predictions, and regenerates reports and figures. Existing original run files are retained. GitHub CLI access is required if archives are missing. Use `--offline` for a populated checkout or local release archives, and `--no-figures` to generate only reports. Runtime depends on downloads, environment installation and disk speed.

Corrected outputs: `results/reports/reanalysis_v2/`, the four `*_all_runs_summary.csv` files, the synthesis tables, README and dataset reports. No original model training is executed.

See [v2 method and known limitations](docs/reanalysis-v2.md). Historical GNNs used held-out features through pooled BatchNorm and incorrect checkpoint selection; reanalysis does not repair those fits. The original datasets, models and seed predictions are reproducible evidence of the historical implementation, not corrected strict-inductive results.

## Verify without changing derived reports

```bash
uv run --frozen python scripts/prepare_reanalysis_artifacts.py --offline
uv run --frozen pytest tests
```

## Training again

The training algorithms still need the fixes documented in the v2 note before new scientific claims. Original sweep scripts/configurations remain for provenance. Future runs refuse to overwrite existing results, and incompatible caches raise an error. Use an isolated results/project root for a new experiment. No weights or dirty-source diffs can be retroactively recovered from the historical bundles.

The prior handbook is available in Git history before the v2 correction. Claims of 30-second full reproduction, a saved tuning study, and biologically independent uncertainty across ten seeds do not apply.
