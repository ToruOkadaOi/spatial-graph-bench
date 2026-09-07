# Changelog

All notable changes to `spatial-graph-bench` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned (#todo)
- **Phase 6 GNN Sweeps on GPU**:
  - Execute GNN sweep on GPU for MERFISH canonical split (`configs/gpu_batch_merfish_canonical.yaml`).
  - Ingest delivery tarball with 4-layer verification and audit ledger logging.
  - Expand sweeps across Open-ST, Xenium, and Stereo-seq platforms.
- **Phase 7 Statistical Inference**:
  - Compute matched lift across GCN, GraphSAGE, GAT, and GIN against the frozen MLP baseline.
  - Execute Two One-Sided Tests (TOST) equivalence testing ($\epsilon = 0.0069$).
  - Perform boundary vs. interior cell margin stratification.
- **Phase 8 Cross-Platform Synthesis**:
  - Generate canonical markdown reports (`docs/results-*.md`).
  - Prepare Zenodo permanent DOI archival for research publication.

---

## [0.1.0] - 2026-09-04

### Added
- **Pre-Registration**: Formally pre-registered [`STUDY_PROTOCOL.md`](file:///Users/aman/Documents/spatial-graph-bench/STUDY_PROTOCOL.md) (commit `449a4d4`) defining hypotheses, closed model grid, and statistical decision rules.
- **Verification Gates G1–G6 (Phase 0)**:
  - Audited and anchored all 4 spatial transcriptomics platforms in [`data/census.json`](file:///Users/aman/Documents/spatial-graph-bench/data/census.json).
  - Verified MERFISH mouse spinal cord raw data (41,267 cells across 5 animals and 18 sections).
  - Established hierarchy constraints per platform (donor-held-out vs. section-held-out vs. replicate-held-out).
- **Partitioning & Disjointness Validation (Phase 2)**:
  - Generated and validated `mouse_held_out_canonical` split (60/60 classes covered, 0 excluded, zero cell ID overlap).
  - Generated exploratory spatial block split.
- **Feature Pipeline & Spatial Ignorance Audit (Phase 3)**:
  - Strict train-only fitting of library size normalization, highly variable gene selection, and 50-component PCA.
  - Coordinate-shuffle invariance audit passed with maximum absolute difference `0.0 <= 1e-6` (zero spatial leakage).
- **Spatial Graphs & Negative Controls (Phase 4)**:
  - Implemented section-own spatial $k$-NN graph builder ($k=6, 12$).
  - Implemented degree-preserving rewired controls ($k=6, 12$) and coordinate-shuffled controls ($k=6, 12$).
  - Implemented Variant A bipartite reference graph ($k=20$).
  - Verified 0 cross-partition and 0 cross-section edges on all 7 topologies.
- **Baseline Model Training & Parity Band (Phase 5)**:
  - Trained spatially ignorant MLP baseline across 10 independent random seeds (42–51).
  - Established empirical parity band ($\pm 2\sigma_{\text{MLP}} = \pm 0.0069$) with interval $[0.5204, 0.5342]$.
  - Pre-registered TOST equivalence margin $\epsilon = 0.0069$.
  - Trained Random Forest baseline (Macro-F1 $0.4683$).
  - Frozen snapshots archived in `audits/baselines_snapshot/merfish_mouse_spinal_cord/mouse_held_out_canonical/`.
- **Reproducibility Artifact Distribution (Phase 6 Scaffolding)**:
  - Published initial release asset bundle [`v0.1.0-merfish-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) containing 16.09 MB of verified preprocessed features and 7 graph bundles.
  - Implemented [`scripts/manage_release_artifacts.py`](file:///Users/aman/Documents/spatial-graph-bench/scripts/manage_release_artifacts.py) for automatic SHA-256 verification and safe unpacking.
- **Continuous Integration**: GitHub Actions CI workflow covering Python 3.11 and 3.12 with Ruff linter, Ruff formatter, Mypy type-checking, and Pytest suite.
