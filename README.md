# spatial-graph-bench: Benchmarking Graph Inductive Benefit in Spatial Transcriptomics

`spatial-graph-bench` is a benchmark evaluating whether cell–cell spatial graphs provide a genuine inductive benefit over strong non-graph baselines for cell-type annotation in spatial transcriptomics (ST).

---

## 1. Core Research Question

> **“When do cell–cell spatial graphs provide a genuine inductive benefit over tuned non-graph baselines for cell-type annotation in spatial transcriptomics, which graph constructions help, and which measurable spatial properties explain success or failure?”**

This benchmark isolates the effect of **spatial graph topology** from model architecture across 4 platforms (**MERFISH**, **Open-ST**, **Xenium**, **Stereo-seq**). A null or negative graph lift over an MLP baseline is treated as a valid scientific finding.

Standard transductive GNNs (GCN, GraphSAGE, GAT, GIN) are evaluated under an inductive, section-own protocol to test generalization to unseen biological slices and donors.

---

## 2. Scientific Principles

- **Donor / Section Held-Out Evaluation**: Strict inductive evaluation where donors and sections in validation and test partitions are disjoint from training tissue.
- **Section-Own Spatial Subgraphs**: Test-time message passing uses strictly internal spatial coordinates within the held-out tissue slice; zero cross-partition and zero cross-section edges.
- **Identical Fixed Features**: Every model (Random Forest, MLP, GNN) is evaluated on the exact same train-fitted 50 Principal Components.
- **Spatial Ignorance in Feature Extraction**: Verified via coordinate-shuffle audit (maximum absolute difference $0.0 \le 10^{-6}$; zero coordinate leakage).
- **Negative Controls**: Degree-preserving edge rewired controls and coordinate-shuffled controls to detect topological smoothing artifacts.
- **Reproducible Artifact Registry**: Splits, feature bundles, and graph topologies are frozen with cryptographic SHA-256 validation hashes and distributed via verified release bundles.

---

## 3. Quickstart

### Installation with `uv`

```bash
# Clone the repository
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# Synchronize exact dependencies from uv.lock
uv sync --frozen
```

### Running Verification & Tests

```bash
# Run standalone CPU dummy benchmark (< 1 second)
uv run python scripts/run_dummy_benchmark.py

# Run unit tests
uv run pytest -v tests

# Run linter and format checks
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

---

## 4. Architecture

```text
src/spatial_graph_bench/
├── config/         # Strict Pydantic benchmark configs (split, prep, graph, model)
├── splitting/      # Donor/section holdout partition assignment and split schemas
├── preprocessing/  # Train-fitted normalisation, HVG, PCA 50, and spatial ignorance audit
├── graph/          # Spatial k-NN, rewired controls, shuffled controls, bipartite
├── models/         # Tuned MLP, SpatialGNN (GCN, GraphSAGE, GAT, GIN), trainer loops
├── analysis/       # 4-layer delivery audit, batch manifests, ingestion ledger
├── tracking/       # Run manifests, matched graph lift, parity band evaluation
└── utils/          # Hashing, seed management, dual-stream logging, path resolution
```

---

## 5. Protocols and Documentation

- [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md): Pre-registered scientific contract, closed comparison grid, and statistical decision rules.
- [REPRODUCE.md](REPRODUCE.md): Step-by-step reproduction handbook with copy-paste commands and expected outputs.
- [TESTING.md](TESTING.md): 4-tier testing guide (unit, invariant validators, smoke runs, delivery audits).
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md): Current completion status and `#todo` roadmap across all phases and platforms.
- [DECISIONS_NEEDED.md](DECISIONS_NEEDED.md): Decision log tracking protocol options.
- [HANDOFF_TO_GPU.md](HANDOFF_TO_GPU.md): Remote GPU execution contract and batch manifests.
- [scripts/FLOW.md](scripts/FLOW.md): CLI script execution pipeline diagram.
- [src/FLOW.md](src/FLOW.md): Internal package architecture and data flow diagram.
- [docs/results-index.md](docs/results-index.md): Catalog of release artifacts, baseline snapshots, and SHA-256 checksums.
- [INSTALL.md](INSTALL.md): Extended installation recipes (uv, pip/venv, conda, Docker).
- [CONTRIBUTING.md](CONTRIBUTING.md): Contribution guidelines and topological invariant review checklist.
- [CITATION.md](CITATION.md): Academic citation format and BibTeX.
- [LICENSE](LICENSE): MIT License.

---

## 6. Baseline Reference & Parity Band (MERFISH Mouse Spinal Cord)

Computed across 10 independent random seeds (42–51) on `mouse_held_out_canonical` (41,267 cells across 5 animals and 18 sections):

- **Spatially Ignorant MLP**: Macro-F1 $\mathbf{0.5273 \pm 0.0035}$ (Balanced Accuracy: $0.5247 \pm 0.0036$).
- **Random Forest Baseline**: Macro-F1 $\mathbf{0.4683}$.
- **Empirical Parity Band ($\pm 2\sigma_{\text{MLP}}$)**: $\mathbf{\pm 0.0069}$, interval $[0.5204, 0.5342]$.
- **Pre-Registered TOST Equivalence Margin**: $\epsilon = \mathbf{0.0069}$.

Any spatial GNN failing to exceed $0.5342$ Macro-F1 provides no empirical benefit over gene expression alone.
