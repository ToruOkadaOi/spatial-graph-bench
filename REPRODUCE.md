# Reproducibility Handbook: The Zero-Domain-Knowledge Guide

This guide provides step-by-step instructions to reproduce all reported benchmark numbers, models, and statistical tests in `spatial-graph-bench`.

It is written so that **any machine learning practitioner or software engineer without a background in biology** can execute, understand, and verify the pipeline from zero.

---

## 1. Primer for Non-Domain Readers

### What Problem Are We Solving?
- **The Data**: We have data on cells measured inside biological tissue slices. Each cell has:
  1. **Gene Expression Features**: A vector of numbers representing activity across genes (reduced to 50 Principal Components).
  2. **Physical Coordinates**: A 2D spatial coordinate $(x, y)$ indicating where the cell is located in the tissue.
- **The Task**: Predict each cell's type (a standard multi-class classification problem; e.g. 60 distinct cell types in MERFISH mouse spinal cord).
- **The Research Question**:
  > *Does connecting cells with a spatial neighbor graph and using Graph Neural Networks (GNNs) actually improve classification accuracy over a simple feed-forward neural network (MLP) that only looks at gene expression?*
- **The Pre-Registered Decision Rule**:
  - We train a spatially ignorant Multi-Layer Perceptron (MLP) across 10 random seeds to establish an empirical **Parity Band** of $\pm 2\sigma = \pm 0.0069$ Macro-F1.
  - If a spatial GNN does not beat the MLP baseline by more than $+0.0069$ Macro-F1, the spatial graph provides **no empirical inductive benefit**.

### Why Macro-F1 Instead of Accuracy?
In biological tissue, some cell types represent 30% of all cells while others represent 0.1%. Standard classification accuracy would allow a model to score 95% simply by ignoring rare classes. **Macro-F1** computes the harmonic mean of precision and recall for each class independently and averages them equally, ensuring rare cell populations cannot be hidden.

---

## 2. Hardware & Environment Prerequisites

| Track | Minimal Hardware | Recommended Hardware | Execution Time |
| :--- | :--- | :--- | :--- |
| **Track A (Release Verification)** | 4 CPU cores, 16 GB RAM | Modern Laptop / Cloud VM | ~10–15 mins |
| **Track B (Full Raw Pipeline)** | 8 CPU cores, 32 GB RAM | 1x NVIDIA GPU (T4 / A10G / V100 / RTX 3090+) | ~45–60 mins |
| **Track C (CPU Dummy Smoke)** | 2 CPU cores, 4 GB RAM | Any modern computer | **< 5 seconds** |

### Software Setup
- **OS**: Linux (Ubuntu 22.04 / 24.04 recommended) or macOS.
- **Python**: 3.11 or 3.12 managed via `uv` (recommended).

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# 3. Synchronize exact pinned dependencies
uv sync --frozen
```

---

## 3. Choose Your Reproduction Track

### Track C: Ultra-Fast CPU Smoke Test (< 30 seconds)
If you want to immediately verify that the repository code, schemas, training loops, and math are functional on your machine without downloading any data:

```bash
uv run python scripts/run_dummy_benchmark.py
```

**Expected Output**:
```text
───────────────── SPATIAL-GRAPH-BENCH: DUMMY CPU BENCHMARK RUN ─────────────────
                       Dummy Benchmark Verification Steps                       
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Sta… ┃ Component                ┃ Invariant / Check                  ┃ Resu… ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 1.   │ ST Fixture Synthesis     │ 45 cells, 25 genes across 3 mice   │ PASS  │
│ 2.   │ Donor Holdout Split      │ Train: 20 | Val: 12 | Test: 13     │ PASS  │
│ 3.   │ Train-fit PCA & Audit    │ Coord shuffle max diff = 0.0 <= 1e-6│ PASS  │
│ 4.   │ Section-Own Spatial k-NN │ k=4, 218 edges, 0 cross-partition  │ PASS  │
│ 5.   │ MLP vs GCN (2 Epochs)    │ MLP F1: 0.1429 | GCN F1: 0.1569    │ PASS  │
│ 6.   │ 4-Layer Delivery Audit   │ Layer 1-4 checks verified on both  │ PASS  │
│ 7.   │ Matched Lift Math        │ Delta F1: +0.0140 | Status: pos    │ PASS  │
└──────┴──────────────────────────┴────────────────────────────────────┴───────┘
ALL CHECKS PASSED (Completed in 0.60s)
```

---

### Track A: Fast Benchmark Verification via GitHub Releases (10–15 mins, Recommended)

This track downloads the frozen preprocessed feature matrices and graph bundles directly from GitHub Releases (`v0.1.0-merfish-inputs`), avoiding the need to download 100MB+ raw biological `.h5ad` files.

#### Step A1: Fetch & Verify Cryptographic Release Assets
```bash
uv run python scripts/manage_release_artifacts.py fetch --tag v0.1.0-merfish-inputs
```
*What this does*: Downloads `artifacts_merfish_mouse_spinal_cord_mouse_held_out_canonical_inputs.tar.gz` (16.09 MB), checks its SHA-256 hash against `sha256sums_*.txt`, unpacks it safely, and runs topological invariant checks.

**Expected Verification Table**:
```text
              Checksum Verification: Release v0.1.0-merfish-inputs              
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ File                            ┃ Expected SHA-256                ┃  Status  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ artifacts_merfish_mouse_spinal… │ 3f92e579fbc081cd17ea5d04d406f3… │ VERIFIED │
└─────────────────────────────────┴─────────────────────────────────┴──────────┘
All benchmark artifacts verified and ready for compute.
```

#### Step A2: Inspect Frozen Baseline Benchmark
Inspect the frozen MLP baseline benchmark numbers computed across 10 random seeds (42–51):
```bash
cat audits/baselines_snapshot/merfish_mouse_spinal_cord/mouse_held_out_canonical/baselines_summary.json
```
- **MLP Mean Test Macro-F1**: $0.5273 \pm 0.0035$
- **MLP Mean Balanced Accuracy**: $0.5247 \pm 0.0036$
- **Parity Band Interval**: $[0.5204, 0.5342]$ ($\pm 0.0069$)

#### Step A3: Train or Reproduce the MLP Baseline Locally
```bash
uv run python scripts/train_baselines.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical --num-seeds 10
```

#### Step A4: Execute the GNN Sweep on GPU
```bash
PYTHONPATH=src uv run python scripts/run_gnn_sweep.py --batch-config configs/gpu_batch_merfish_canonical.yaml
```
*(On CPU-only machines, you can add `--device cpu` inside the YAML configuration).*

#### Step A5: Compute Matched Lift & Decision Thresholds
```bash
uv run python scripts/compute_matched_lift.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical
```
*What this does*: Pairs each GNN run with its identical-seed MLP baseline run, calculates $\Delta \text{Macro-F1} = \text{F1}_{\text{GNN}} - \text{F1}_{\text{MLP}}$, and evaluates whether the difference exceeds $+0.0069$.

---

### Track B: Full Raw-to-Result Pipeline from Scratch (45–60 mins)

This track runs the entire pipeline from raw data files to final evaluation.

#### Step B1: Verify Raw Data Anchors
```bash
uv run python scripts/audit_census.py
```
Asserts that `data/raw/merfish_mouse_spinal_cord/MERFISH_spinal_cord_resolved_0718.h5ad` matches SHA-256 hash `d5da08d873641ba432eb6ef3d4f128d9c02aa002a2498db257d07963d41f0a20`.

#### Step B2: Generate Partition Splits
```bash
uv run python scripts/generate_splits.py --dataset merfish_mouse_spinal_cord --hierarchy donor_held_out --seed 42
uv run python scripts/validate_split.py splits/merfish_mouse_spinal_cord/mouse_held_out_canonical.json
```
*Verification*: Asserts zero overlap between train (26,866 cells), val (5,894 cells), and test (8,507 cells), with 60/60 classes present in test.

#### Step B3: Run Feature Preprocessing & Spatial Ignorance Audit
```bash
uv run python scripts/run_preprocessing.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical
uv run python scripts/audit_spatial_ignorance.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical
```
*Verification*: The coordinate-shuffle test must output `Spatial ignorance audit PASSED (max diff = 0.000000e+00 <= 1e-6)`.

#### Step B4: Construct Spatial Graphs & Negative Controls
```bash
uv run python scripts/build_spatial_graphs.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical
```
*Verification*: Generates all 7 graph topologies:
1. `spatial_knn_k6` (290,052 edges)
2. `spatial_knn_k12` (565,698 edges)
3. `rewired_spatial_knn_k6` (290,052 edges)
4. `rewired_spatial_knn_k12` (565,698 edges)
5. `shuffled_spatial_knn_k6` (299,472 edges)
6. `shuffled_spatial_knn_k12` (584,424 edges)
7. `bipartite_ref_k20` (1,256,355 edges)
Every graph certifies **0 cross-partition and 0 cross-section edges**.

#### Step B5: Run Invariant Validator across All Artifacts
```bash
uv run python scripts/validate_artifacts.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical
```

---

## 4. Cryptographic Checksum Reference

All release artifacts are immutable and pinned by SHA-256 hashes:

| Artifact Name | Scope | Release Asset / Path | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **MERFISH Inputs Bundle** | Features + 7 Graphs | `v0.1.0-merfish-inputs` (16.09 MB) | `3f92e579fbc081cd17ea5d04d406f36c9dee1c20b06030584f3900abe8964e8b` |
| **Split Definition** | Canonical Split JSON | `splits/merfish_mouse_spinal_cord/mouse_held_out_canonical.json` | `a9ed50703ca8fa9cfc1a2eb1c1639b1ba284253c02b5e3556ab0b6242ceaf31a` |
| **Feature Manifest** | Train PCA 50 | `artifacts/preprocessed/.../feature_manifest.json` | `0ad75181c4f2c80b47d8c9b28d2d46fd09c3dc336480c02951abb653fb882e79` |
| **Graph Manifest (k=6)** | Real spatial $k$-NN | `artifacts/graphs/.../spatial_knn_k6/graph_manifest.json` | `1f3f0e36f26ba424603015ff90b8a2ee91d6c27664e27b5577831a2371a4243e` |

---

## 5. Troubleshooting & FAQ

### Q1: Can I run this without an NVIDIA GPU?
**Yes.** All baseline scripts (`train_baselines.py`), preprocessing scripts, graph generation, validators, and the dummy benchmark run on standard CPUs. For GNN sweeps, change `device: "cuda"` to `device: "cpu"` in the YAML batch config.

### Q2: What does `parity` classification mean in the results?
If a GNN model scores within $[0.5204, 0.5342]$ Macro-F1 ($\pm 0.0069$ of the MLP baseline), it is classified as `parity`. This proves statistically that adding the spatial graph did not improve prediction accuracy over gene expression alone.

### Q3: Where are execution logs stored?
Execution logs are streamed live to the console with Rich formatting and written to `logs/spatial_bench.log`. You can adjust logging verbosity by setting `SPATIAL_LOG_LEVEL=DEBUG`.
