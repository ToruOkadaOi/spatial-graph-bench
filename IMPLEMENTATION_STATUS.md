# IMPLEMENTATION_STATUS.md: Phase Tracker & Test Coverage

This document tracks execution progress, gate outcomes, and test coverage across the 9 phases of `spatial-graph-bench`.

---

## Overall Status Summary

| Phase | Description | Status | Exit Criterion | Notes |
|---|---|---|---|---|
| **0** | Reference Mining & Data Verification Gates G1–G6 | **COMPLETED** | `census.json` complete, hierarchy settled, mining checklist reviewed | Gates G1–G6 audited; reference mined; Dataset 1 downloaded & verified. |
| **1** | Scaffolding, Pre-Registration, CI, Porting | **COMPLETED** | CI green on fixture, protocol commit hash recorded, reference clone destroyed | CI green (16/16 tests pass), commit `449a4d4` recorded, reference clone destroyed. |
| **2** | Split Protocol & Disjointness Validators | **COMPLETED** | `validate_split.py` green on all datasets | Canonical mouse-held-out (`00408e4741f5`) & spatial-block splits generated & validated. |
| **3** | Feature Pipelines A/B & Spatial Ignorance Audit | **COMPLETED** | Coordinate-shuffle test clean, manifests issued | Dual pipelines (strict vs platform-default); shuffle audit passed (max diff 0.0); MERFISH manifests issued. |
| **4** | Graph Constructions, Controls & Validators | PENDING | Construction manifests chained, connectivity verified per variant | Spatial k-NN, rewired & shuffled controls, section-own topology. |
| **5** | Baseline Model Training (MLP $\ge 10$ seeds, RF) | PENDING | Canonical baselines frozen in `audits/baselines_snapshot/` | Tuned spatially ignorant models on train partitions only. |
| **6** | GNN Sweeps on GPU via Handoff & Ingestion | PENDING | All runs PASS or quarantined with notes | Audited delivery with 4-layer verification. |
| **7** | Post-Hoc Analysis & Anti-Circularity Audit | PENDING | Circular metrics demoted per §3.3 | Matched lift, TOST equivalence test, boundary stratification. |
| **8** | Reports & Cross-Dataset Synthesis | PENDING | Generated docs pass stale-reference guard | Canonical tables exported to markdown. |

---

## Phase 0: Verification Gates Detailed Status

- [x] **Reference Mining**: Cloned `scgraph-bench` to scratch `~/reference/scgraph-bench`, mined tracking schemas, packaging/receive scripts, split validators, bipartite vs. section-own connectivity, CI, and bugfix history.
- [x] **Gate G1 (Raw-Data Root)**: Opened Dataset 1's h5ad (`MERFISH_spinal_cord_resolved_0718.h5ad`); confirmed raw uint32 counts in `adata.X` and spatial coordinates in `obsm['spatial']` and `obs[['center_x', 'center_y']]`.
- [x] **Gate G2 (MERFISH Hierarchy)**: Confirmed `obs['Mouse ID']` in h5ad. 5 adult mice (F3, F4, F5, M4, M5) across 18 sections (41,267 cells). Canonical split is **donor-held-out** (or **mouse-held-out**).
- [x] **Gate G3 (Xenium Hierarchy)**: Pulled GSE269719 sample metadata (GSM8325615–GSM8325626). Confirmed absence of explicit animal IDs. Split hierarchy is formally locked to **replicate-held-out**, never "donor-held-out".
- [x] **Gate G4 (Open-ST Serial Stack)**: Examined GSE251926 3D stack. Confirmed upstream convention: section #6 as reference (GSM7990104) and section #19 as query (GSM7990110), spaced ~234 µm apart in tissue depth. Split hierarchy is **section-held-out**.
- [x] **Gate G5 (Stereo-seq Axolotl Access)**: Verified public ARTISTA download links for all 5 benchmark h5ad files (`Stage44.h5ad`, `Stage54.h5ad`, `5DPI_1.h5ad`, `5DPI_2.h5ad`, `5DPI_3.h5ad`). Split hierarchy is **replicate-held-out**.
- [x] **Gate G6 (Census Manifest & Script)**: Authored `data/census.json` capturing URLs, file sizes, licenses, raw snapshot paths, and SHA-256 anchors. Built `scripts/audit_census.py` and passed verification audit.

---

## Phase 1 Execution Checklist

- [x] Repository layout scaffolded (.github, audits, configs, data, docs, scripts, splits, src, tests).
- [x] Root `.gitignore` configured to keep heavy binaries out of git while tracking audit ledgers.
- [x] `STUDY_PROTOCOL.md` pre-registration document authored.
- [x] `DECISIONS_NEEDED.md` decision log authored.
- [x] `docs/PORTING_LOG.md` authored.
- [x] Core package `src/spatial_graph_bench` ported and unit-tested.
- [x] Fixture dataset created and CPU smoke pipeline tested.
- [x] `pyproject.toml`, `uv.lock`, `Makefile`, `Dockerfile`, and GitHub Actions CI workflow implemented.
- [x] Protocol commit hash recorded; scratch reference clone `~/reference/scgraph-bench` destroyed.

---

## Phase 2 Execution Checklist

- [x] `scripts/generate_splits.py` implemented.
- [x] `scripts/validate_split.py` implemented with disjointness and §3.4 unseen-class coverage validator.
- [x] MERFISH `mouse_held_out_canonical` generated and validated (`00408e4741f5681a...`, 60/60 classes covered, 0 excluded).
- [x] MERFISH `spatial_block_exploratory` generated and validated (`8abde4e16b30955f...`).

---

## Phase 3 Execution Checklist

- [x] `src/spatial_graph_bench/preprocessing/pipeline.py` implemented (train-only fitting of scalers, HVG, PCA).
- [x] `src/spatial_graph_bench/preprocessing/audit.py` implemented (coordinate-shuffle test).
- [x] `scripts/run_preprocessing.py` implemented.
- [x] Coordinate-shuffle audit passed (max diff 0.0 <= 1e-6).
- [x] Version A feature manifests and bundles issued for MERFISH canonical and block splits.
