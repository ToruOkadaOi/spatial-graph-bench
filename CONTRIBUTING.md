# Contributing to `spatial-graph-bench`

Thank you for your interest in contributing to `spatial-graph-bench`! We welcome contributions that expand spatial transcriptomics datasets, add novel graph topologies, improve benchmark efficiency, or strengthen statistical auditability.

---

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](file:///Users/aman/Documents/spatial-graph-bench/CODE_OF_CONDUCT.md). Please read it before participating.

---

## Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/spatial-graph-bench.git
   cd spatial-graph-bench
   ```

2. **Set up Environment**:
   ```bash
   uv sync --frozen
   ```

3. **Verify Existing Tests**:
   ```bash
   uv run pytest -v tests
   uv run python scripts/run_dummy_benchmark.py
   ```

---

## Coding Standards & Quality Gates

Before opening a pull request, verify that all quality gates pass:

1. **Ruff Linter & Formatter**:
   ```bash
   # Both linter and formatter are strictly enforced in CI:
   uv run ruff check src tests scripts
   uv run ruff format --check src tests scripts

   # Auto-format and fix:
   make format
   ```

2. **Type Annotations**:
   - Use Python 3.11+ type annotations (`int | None`, `list[str]`, etc.).
   - Run mypy: `uv run mypy src tests scripts`.

3. **Unit & Smoke Tests**:
   ```bash
   uv run pytest -v tests
   ```

---

## Topological Invariant Checklist for Pull Requests

Because `spatial-graph-bench` is a pre-registered benchmark testing the inductive benefit of spatial graphs, **data leakage and edge leakage are strictly prohibited**.

If your PR introduces or modifies dataset splitting, feature extraction, or graph construction, you **must certify the following invariants**:

- [ ] **Zero Cross-Partition Edges**: The graph builder must strictly guarantee that no edge connects a train cell to a val/test cell, or a val cell to a test cell. Run `scripts/validate_construction.py` on your graph bundle.
- [ ] **Zero Cross-Section Edges**: For section-own protocols, no edge may connect cells from different tissue slices.
- [ ] **Spatial Ignorance in Feature Extraction**: Preprocessed features (PCA components) must never leak spatial coordinate information. Verify via `verify_spatial_ignorance` (coordinate-shuffle test must show absolute diff $\le 10^{-6}$).
- [ ] **No Unseen Test Classes (§3.4)**: Every class in the test partition must be represented in the training partition. Verify via `scripts/validate_split.py`.
- [ ] **Closed Model Comparison Grid**: GNN evaluations must be benchmarked against the identical-seed spatially ignorant MLP baseline.

---

## Proposing New Datasets

To propose a new spatial transcriptomics platform (e.g. Visium HD, CosMx, Slide-seqV2):
1. Ensure the raw data has a public URL, open license, and raw count matrices.
2. Open a **New Dataset Proposal** issue using our template.
3. Anchor raw file metadata and SHA-256 hashes in `data/census.json`.
4. Implement split generation honoring donor/replicate boundaries.
