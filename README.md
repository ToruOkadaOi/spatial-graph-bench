# spatial-graph-bench

[![CI](https://github.com/ToruOkadaOi/spatial-graph-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/ToruOkadaOi/spatial-graph-bench/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ToruOkadaOi/spatial-graph-bench)](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Protocol: Pre-Registered](https://img.shields.io/badge/protocol-pre--registered-success.svg)](STUDY_PROTOCOL.md)

**Spatial Graph Inductive-Benefit Benchmark for Cell-Type Annotation**

A research-software benchmark investigating whether cell–cell spatial graphs provide a **genuine inductive benefit** over tuned *spatially ignorant* baselines for **cell-type annotation** in spatial transcriptomics (ST).

This project is a **pre-registered replication attempt** transplanting the rigorous single-cell evaluation methodology of `scgraph-bench` into the spatial domain across 4 major spatial platforms: **MERFISH**, **Open-ST**, **Xenium**, and **Stereo-seq**.

---

## Quick Navigation

| Resource | Purpose |
| :--- | :--- |
| 📖 **[`REPRODUCE.md`](REPRODUCE.md)** | **Step-by-step reproduction handbook** written for non-domain engineers (copy-paste commands, expected outputs). |
| 🧪 **[`TESTING.md`](TESTING.md)** | Complete testing guide covering Unit Tests, Invariant Validators, and CPU Smoke Runs. |
| 🗺️ **[`scripts/FLOW.md`](scripts/FLOW.md)** | Visual Mermaid diagram and reference of the CLI execution pipeline. |
| 🏗️ **[`src/FLOW.md`](src/FLOW.md)** | Internal architecture and immutable Pydantic data pipeline flow. |
| 📜 **[`STUDY_PROTOCOL.md`](STUDY_PROTOCOL.md)** | Pre-registered hypotheses, closed comparison grid, and statistical decision rules. |
| 📋 **[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)** | Live phase tracker (Phases 0–8) and `#todo` checklist across all 4 platforms. |
| 📦 **[`docs/results-index.md`](docs/results-index.md)** | Catalog of official release assets, SHA-256 hashes, and baseline snapshots. |
| ⚙️ **[`INSTALL.md`](INSTALL.md)** | Installation recipes for `uv`, standard `pip/venv`, `conda/mamba`, and Docker. |
| 🛡️ **[`SECURITY.md`](SECURITY.md)** | Responsible disclosure policy and cryptographic integrity model. |
| 📝 **[`CITATION.md`](CITATION.md)** | BibTeX entry and academic citation guidelines. |

---

## Core Invariants

1. **Labels never enter graph construction**: Graph topology is constructed strictly from spatial coordinates.
2. **Identical frozen features across all models**: Performance gains must be attributable to spatial message passing, not feature drift.
3. **Cryptographic parent-hashing**: Raw data snapshot $\to$ Split $\to$ Feature manifest $\to$ Construction manifest $\to$ Run manifest.
4. **The GPU is untrusted compute**: Every delivery package is audited locally on CPU through four integrity layers.
5. **No test data contamination**: Normalization, highly variable gene selection, and PCA are fit strictly on training partitions.
6. **Section-own-subgraph inductive evaluation**: Test-time message passing uses only the held-out section's internal spatial subgraph; strictly zero cross-partition and zero cross-section edges.

---

## 30-Second Quickstart

### 1. Bootstrap Environment
```bash
# Clone repository
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# Install pinned dependencies via uv
uv sync --frozen
```

### 2. Run the Instant CPU Dummy Benchmark (< 1 second)
Verify that the entire benchmark pipeline, schemas, training loops, delivery audits, and matched lift math work on your machine without downloading any data or using a GPU:
```bash
uv run python scripts/run_dummy_benchmark.py
```

### 3. Run Test Suite & Lint Checks
```bash
# Run pytest test suite (16 tests)
uv run pytest -v tests

# Run strict linter and formatter check
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

---

## Reproducing the Benchmark Results

To reproduce reported numbers on the canonical MERFISH mouse spinal cord dataset:

```bash
# 1. Download and verify pre-packaged release assets (16.09 MB)
uv run python scripts/manage_release_artifacts.py fetch --tag v0.1.0-merfish-inputs

# 2. Verify topological and feature invariants
uv run python scripts/validate_artifacts.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical

# 3. Train the 10-seed MLP baseline to verify the parity band (Macro-F1 0.5273 ± 0.0035)
uv run python scripts/train_baselines.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical --num-seeds 10
```

For the complete guide with hardware requirements, expected stdout logs, and from-scratch raw data execution, see **[`REPRODUCE.md`](REPRODUCE.md)**.

---

## Author & Citation

Maintained by **Aman Nalakath** ([@ToruOkadaOi](https://github.com/ToruOkadaOi)) and Spatial Graph Bench Contributors.

If you use this benchmark in your research, please cite:
```bibtex
@software{nalakath2026spatialgraphbench,
  author       = {Nalakath, Aman},
  title        = {spatial-graph-bench: A Rigorous Inductive-Benefit Benchmark for Spatial Graph Neural Networks in Cell-Type Annotation},
  year         = {2026},
  url          = {https://github.com/ToruOkadaOi/spatial-graph-bench}
}
```

Licensed under the [MIT License](LICENSE).
