# spatial-graph-bench

**Spatial Graph Inductive-Benefit Benchmark for Cell-Type Annotation**

A research-software benchmark investigating whether cell–cell spatial graphs provide a
**genuine inductive benefit** over strong *spatially ignorant* baselines for **cell-type annotation**
in spatial transcriptomics (ST).

This project is a **pre-registered replication attempt** transplanting the single-cell evaluation
methodology of `scgraph-bench` (which found 0 of 45 positive comparisons) into the spatial domain.
It addresses the documented literature gap where spatial GNNs report gains over spatial clustering or
deconvolution tools but almost never against a tuned per-spot MLP or Random Forest baseline evaluated
on identical features and partitions.

---

## Core Invariants

1. **Labels never enter graph construction**: Graph topology is constructed strictly from spatial coordinates and features.
2. **Identical frozen features across all models**: Performance gains must be attributable to graph message passing, not feature drift.
3. **Cryptographic parent-hashing**: Raw data snapshot $\to$ Split $\to$ Feature manifest $\to$ Construction manifest $\to$ Run manifest.
4. **The GPU is untrusted compute**: Every result delivery is verified locally on CPU through four integrity layers.
5. **No test data contamination**: Hyperparameter tuning for every model is restricted strictly to training partitions.
6. **Section-own-subgraph inductive evaluation**: Test-time message passing uses only the held-out section's internal spatial subgraph; zero cross-partition edges.

---

## Quickstart

```bash
# Bootstrap virtual environment with uv
uv sync

# Run mandatory linter & test suite
make check

# Run end-to-end CPU smoke pipeline on fixture
uv run pytest -v tests/test_smoke_pipeline.py
```

See [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md) for pre-registration details and
[docs/provenance.md](docs/provenance.md) for the cryptographic audit trail.
