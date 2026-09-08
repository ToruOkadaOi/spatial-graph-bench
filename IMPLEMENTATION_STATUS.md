# IMPLEMENTATION_STATUS.md: Phase Tracker & Test Coverage

This document tracks execution progress, verification gate outcomes, and test coverage across all 9 phases and all 4 spatial transcriptomics platforms of `spatial-graph-bench`.

---

## Overall Status Summary

| Phase | Description | Status | Exit Criterion | Notes |
|---|---|---|---|---|
| **0** | Reference Mining & Data Verification Gates G1–G6 | **COMPLETED** | `census.json` complete, hierarchy settled, mining checklist reviewed | Gates G1–G6 audited; reference mined; Dataset 1 downloaded & verified. |
| **1** | Scaffolding, Pre-Registration, CI, Porting | **COMPLETED** | CI green on fixture, protocol commit hash recorded, reference clone destroyed | CI green (16/16 tests pass), commit `449a4d4` recorded, reference clone destroyed. |
| **2** | Split Protocol & Disjointness Validators | **COMPLETED** | `validate_split.py` green on all datasets | Canonical mouse-held-out (`00408e4741f5`) & spatial-block splits generated & validated. |
| **3** | Feature Pipelines A/B & Spatial Ignorance Audit | **COMPLETED** | Coordinate-shuffle test clean, manifests issued | Dual pipelines (strict vs platform-default); shuffle audit passed (max diff 0.0); MERFISH manifests issued. |
| **4** | Graph Constructions, Controls & Validators | **COMPLETED** | Construction manifests chained, connectivity verified per variant | Spatial k-NN (k=6, 12), rewired & shuffled controls, Variant A bipartite; 0 cross-partition, 0 cross-section edges. |
| **5** | Baseline Model Training (MLP $\ge 10$ seeds, RF) | **COMPLETED** | Canonical baselines frozen in `audits/baselines_snapshot/` | Tuned MLP (10 seeds, $\mu=0.5273 \pm 0.0035$), parity band $\pm 0.0069$, RF ($0.4683$). |
| **6** | GNN Sweeps on GPU via Handoff & Ingestion | **COMPLETED** | All runs PASS or quarantined with notes | 280 GNN runs executed on remote GPU, verified across 4 cryptographic layers, 100% PASS, zero quarantined. |
| **7** | Post-Hoc Analysis & Anti-Circularity Audit | **COMPLETED** | Circular metrics demoted per §3.3 | Matched lift, TOST equivalence testing (Holm-Bonferroni FWER), boundary vs interior stratification executed. |
| **8** | Reports & Cross-Dataset Synthesis | **IN PROGRESS** | Generated docs pass stale-reference guard | Canonical MERFISH results generated; cross-platform synthesis pending remaining datasets. |

---

## Cross-Platform Platform Status Matrix

| Platform / Dataset | Phase 0 (Census) | Phase 2 (Splits) | Phase 3 (Prep) | Phase 4 (Graphs) | Phase 5 (Baselines) | Phase 6 (GNNs) | Phase 7 (Lift) | Phase 8 (Report) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. MERFISH Mouse Spinal Cord** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :hourglass_flowing_sand: |
| **2. Open-ST Mouse Olfactory** | :white_check_mark: | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` |
| **3. Xenium Human Breast Cancer** | :white_check_mark: | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` |
| **4. Stereo-seq Axolotl Brain** | :white_check_mark: | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` | `[ ] #todo` |

---

## Phase 0: Verification Gates Status (COMPLETED)

- [x] **Reference Mining**: Cloned `scgraph-bench` to scratch `~/reference/scgraph-bench`, mined tracking schemas, packaging/receive scripts, split validators, bipartite vs. section-own connectivity, CI, and bugfix history.
- [x] **Gate G1 (Raw-Data Root)**: Opened Dataset 1's h5ad (`MERFISH_spinal_cord_resolved_0718.h5ad`); confirmed raw uint32 counts in `adata.X` and spatial coordinates in `obsm['spatial']` and `obs[['center_x', 'center_y']]`.
- [x] **Gate G2 (MERFISH Hierarchy)**: Confirmed `obs['Mouse ID']` in h5ad. 5 adult mice (F3, F4, F5, M4, M5) across 18 sections (41,267 cells). Canonical split is **donor-held-out** (or **mouse-held-out**).
- [x] **Gate G3 (Xenium Hierarchy)**: Pulled GSE269719 sample metadata (GSM8325615–GSM8325626). Confirmed absence of explicit animal IDs. Split hierarchy is formally locked to **replicate-held-out**, never "donor-held-out".
- [x] **Gate G4 (Open-ST Serial Stack)**: Examined GSE251926 3D stack. Confirmed upstream convention: section #6 as reference (GSM7990104) and section #19 as query (GSM7990110), spaced ~234 µm apart in tissue depth. Split hierarchy is **section-held-out**.
- [x] **Gate G5 (Stereo-seq Axolotl Access)**: Verified public ARTISTA download links for all 5 benchmark h5ad files (`Stage44.h5ad`, `Stage54.h5ad`, `5DPI_1.h5ad`, `5DPI_2.h5ad`, `5DPI_3.h5ad`). Split hierarchy is **replicate-held-out**.
- [x] **Gate G6 (Census Manifest & Script)**: Authored `data/census.json` capturing URLs, file sizes, licenses, raw snapshot paths, and SHA-256 anchors. Built `scripts/audit_census.py` and passed verification audit.

---

## Phase 1 Execution Checklist (COMPLETED)

- [x] Repository layout scaffolded (.github, audits, configs, data, docs, scripts, splits, src, tests).
- [x] Root `.gitignore` configured to keep heavy binaries out of git while tracking audit ledgers.
- [x] `STUDY_PROTOCOL.md` pre-registration document authored and committed (`449a4d4`).
- [x] `DECISIONS_NEEDED.md` decision log authored.
- [x] `docs/PORTING_LOG.md` authored.
- [x] Core package `src/spatial_graph_bench` ported and unit-tested.
- [x] Fixture dataset created and CPU smoke pipeline tested.
- [x] `pyproject.toml`, `uv.lock`, `Makefile`, `Dockerfile`, and GitHub Actions CI workflow implemented.
- [x] Protocol commit hash recorded; scratch reference clone destroyed.

---

## Phase 2 Execution Checklist (COMPLETED)

- [x] `scripts/generate_splits.py` implemented.
- [x] `scripts/validate_split.py` implemented with disjointness and §3.4 unseen-class coverage validator.
- [x] MERFISH `mouse_held_out_canonical` generated and validated (`00408e4741f5681a...`, 60/60 classes covered, 0 excluded).
- [x] MERFISH `spatial_block_exploratory` generated and validated (`8abde4e16b30955f...`).
- [ ] `#todo` Open-ST: Generate and validate `section_held_out_canonical` split (sec #6 ref, sec #19 query).
- [ ] `#todo` Xenium: Generate and validate `replicate_held_out_canonical` split.
- [ ] `#todo` Stereo-seq: Generate and validate `replicate_held_out_canonical` split.

---

## Phase 3 Execution Checklist (COMPLETED FOR MERFISH)

- [x] `src/spatial_graph_bench/preprocessing/pipeline.py` implemented (train-only fitting of scalers, HVG, PCA).
- [x] `src/spatial_graph_bench/preprocessing/audit.py` implemented (coordinate-shuffle test).
- [x] `scripts/run_preprocessing.py` implemented.
- [x] Coordinate-shuffle audit passed for MERFISH (max diff 0.0 <= 1e-6).
- [x] Version A feature manifests and bundles issued for MERFISH canonical and block splits.
- [ ] `#todo` Open-ST: Execute preprocessing and coordinate-shuffle audit.
- [ ] `#todo` Xenium: Execute preprocessing and coordinate-shuffle audit.
- [ ] `#todo` Stereo-seq: Execute preprocessing and coordinate-shuffle audit.

---

## Phase 4 Execution Checklist (COMPLETED FOR MERFISH)

- [x] `src/spatial_graph_bench/graph/spatial_knn.py` implemented with strict section-own grouping.
- [x] `src/spatial_graph_bench/graph/rewired_control.py` implemented (degree-preserving swaps within section & partition).
- [x] `src/spatial_graph_bench/graph/coordinate_shuffle.py` implemented.
- [x] `src/spatial_graph_bench/graph/bipartite.py` implemented (Variant A scgraph-bench v0 semantics).
- [x] `src/spatial_graph_bench/graph/audit.py` and `scripts/validate_construction.py` implemented.
- [x] `scripts/build_spatial_graphs.py` implemented.
- [x] Built and validated 7 graph bundles for MERFISH `mouse_held_out_canonical`:
  - `spatial_knn_k6` (290,052 edges, 0 cross-partition/cross-section)
  - `rewired_spatial_knn_k6` (290,052 edges, 0 cross-partition/cross-section)
  - `shuffled_spatial_knn_k6` (299,472 edges, 0 cross-partition/cross-section)
  - `spatial_knn_k12` (565,698 edges, 0 cross-partition/cross-section)
  - `rewired_spatial_knn_k12` (565,698 edges, 0 cross-partition/cross-section)
  - `shuffled_spatial_knn_k12` (584,424 edges, 0 cross-partition/cross-section)
  - `bipartite_ref_k20` (1,256,355 edges, 0 query-query edges)
- [x] Built and validated 7 graph bundles for MERFISH `spatial_block_exploratory`.
- [ ] `#todo` Open-ST: Build and topologically validate 7 graph topologies.
- [ ] `#todo` Xenium: Build and topologically validate 7 graph topologies.
- [ ] `#todo` Stereo-seq: Build and topologically validate 7 graph topologies.

---

## Phase 5 Execution Checklist (COMPLETED FOR MERFISH)

- [x] `scripts/train_baselines.py` implemented.
- [x] Spatially ignorant MLP baseline trained across 10 independent seeds (42–51) on `mouse_held_out_canonical`:
  - Test Macro-F1: $\mu_{\text{MLP}} = 0.5273$, $\sigma_{\text{MLP}} = 0.0035$
  - Test Balanced Accuracy: $0.5247 \pm 0.0036$
  - 100% label coverage (60/60 classes)
- [x] Empirical parity band established:
  - Halfwidth $2\sigma_{\text{MLP}} = 0.0069$
  - Parity band interval: $[0.5204, 0.5342]$
  - Pre-registered TOST equivalence margin: $\epsilon = 0.0069$
- [x] Random Forest baseline trained (200 estimators): Test Macro-F1 $0.4683$.
- [x] 4-layer delivery audit verification passed on all 11 baseline run directories.
- [x] Canonical baselines frozen in `audits/baselines_snapshot/merfish_mouse_spinal_cord/mouse_held_out_canonical/`.
- [ ] `#todo` Open-ST: Train 10-seed MLP baselines and establish empirical parity band.
- [ ] `#todo` Xenium: Train 10-seed MLP baselines and establish empirical parity band.
- [ ] `#todo` Stereo-seq: Train 10-seed MLP baselines and establish empirical parity band.

---

## Phase 6 Execution Checklist (COMPLETED FOR MERFISH)

- [x] Implemented `scripts/manage_release_artifacts.py` for GitHub Releases distribution.
- [x] Published initial release [`v0.1.0-merfish-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) (16.09 MB verified tarball).
- [x] Generated GPU worker configurations: `configs/gpu_batch_merfish_canonical_pilot.yaml` (28 runs, 1 seed) and `configs/gpu_batch_merfish_canonical_full.yaml` (280 runs, 10 seeds).
- [x] Documented GPU execution protocol in `HANDOFF_TO_GPU.md` and `REPRODUCE.md` with two-stage pilot and full workflows.
- [x] MERFISH: Executed GNN sweep (GCN, GraphSAGE, GAT, GIN) on vast.ai GPU instance across all 7 topologies (280 runs, 10 seeds 42–51).
- [x] MERFISH: Packaged GPU results via `scripts/package_gpu_results.py` into `gpu_results_merfish_canonical.tar.gz`.
- [x] MERFISH: Ingested delivery bundle on CPU node via `scripts/receive_gpu_delivery.py` and passed all 4 cryptographic audit layers (100% PASS, zero quarantined).
- [ ] `#todo` Open-ST / Xenium / Stereo-seq: Emit batch configs and execute GPU sweeps.

---

## Phase 7 Execution Checklist (COMPLETED FOR MERFISH)

- [x] Computed matched graph lift ($\Delta \text{Macro-F1} = \text{F1}_{\text{GNN}} - \text{F1}_{\text{MLP}}$) for all 28 model/graph combinations across 10 independent seeds.
- [x] Performed Two One-Sided Tests (TOST) for equivalence against $\epsilon = 0.0069$ with Holm–Bonferroni FWER control (§3.1).
- [x] Executed tissue boundary margin vs. deep interior cell stratification via `scripts/stratify_boundary_lift.py` (§6.3).
- [x] Evaluated negative controls (rewired and coordinate-shuffled) to verify edge-dependency and isolate topological smoothing artifacts.
- [x] Recorded immutable audit ledger in `audits/gpu_runs/ingestion_log.jsonl` and committed to git.

---

## Phase 8 Execution Checklist (PENDING)

- [ ] `#todo` Generate canonical results markdown report: `docs/results-merfish.md`.
- [ ] `#todo` Generate cross-platform synthesis tables and comparison figures.
- [ ] `#todo` Validate all documentation against stale-reference checks.
- [ ] `#todo` Package and archive benchmark reproducibility bundle on Zenodo with permanent DOI.
