# PORTING_LOG.md: Deliberate Porting & Spatial Adaptation Record

This log documents every component mined and ported from the reference engineering
repository (`github.com/ToruOkadaOi/scgraph-bench`) into `spatial-graph-bench`. No bulk
copy-pasting is permitted; every component is explicitly adapted to the spatial transcriptomics
setting.

---

## 1. Cryptographic Hashing Utilities
- **Source**: `scgraph-bench:src/scgraph_bench/utils/hashing.py`
- **Destination**: `src/spatial_graph_bench/utils/hashing.py`
- **What was taken**: Deterministic SHA-256 computation for bytes, strings, dictionaries, Pydantic models, disk files, and numpy arrays.
- **What changed**: Added support for spatial coordinate tensors and structured arrays.
- **Why**: Cryptographic parent-hashing is an immutable invariant across the pipeline; byte-deterministic hashing is universal.

## 2. Tracking Schemas & Run Manifest
- **Source**: `scgraph-bench:src/scgraph_bench/tracking/schema.py`
- **Destination**: `src/spatial_graph_bench/tracking/schema.py`
- **What was taken**: `RunStatus`, `FailureMetadata`, `RunManifest`, `MetricRecord`, `GraphLiftRecord`, `TidyResultsCollection`.
- **What changed**:
  - `RunManifest` now explicitly tracks `spatial_construction_manifest_hash`, `coordinate_hash`, and `protocol_variant` (`canonical_section_own`, `variant_a_bipartite`, `variant_b_pooled_transductive`).
  - `MetricRecord` adds `section_id`, `axial_level`, `distance_to_boundary`, and spatial strata.
  - Adapted `GraphLiftRecord` to record matched same-seed lift ($GNN - MLP$) stratified by interior vs boundary cells, and per-section breakdowns.
- **Why**: Spatial graphs evaluate section-own topology and require stratification across spatial domains and boundary margins.

## 3. Split Schema & Disjointness Enforcement
- **Source**: `scgraph-bench:src/scgraph_bench/splitting/schema.py`
- **Destination**: `src/spatial_graph_bench/splitting/schema.py`
- **What was taken**: `SplitDefinition` with strict disjointness validation (`validate_disjointness`), serialization to JSON, and deterministic split hashing.
- **What changed**:
  - Expanded from donor-only partitioning to multi-level hierarchy: `donor_id`, `mouse_id`, `section_id`, and `spatial_block_id`.
  - Added support for `unseen_classes_handling` (§3.4): explicitly tracking per-fold label space intersection and excluded classes.
- **Why**: In spatial transcriptomics, holdouts follow a strict hierarchy (donor > section > block). Cells belonging to classes absent from training must be logged and uniformly excluded from evaluation across all models.

## 4. Graph Construction & Topological Container
- **Source**: `scgraph-bench:src/scgraph_bench/graph/schema.py`, `config/graph.py`, `pca_knn.py`
- **Destination**: `src/spatial_graph_bench/graph/schema.py`, `config/graph.py`, `spatial_knn.py`, `rewired_control.py`, `coordinate_shuffle.py`
- **What was taken**: `GraphBundle` container, `GraphManifest`, degree-preserving rewiring algorithm.
- **What changed**:
  - **Canonical protocol changed completely**: `scgraph-bench` used `strict_bipartite` (test/val connect only to training reference via feature similarity). `spatial-graph-bench` canonical protocol enforces **section-own-subgraph inductive evaluation**: message passing occurs strictly within each section's own spatial coordinates; no cross-partition edges, no cross-section edges.
  - Input changed from expression PCA to physical spatial coordinates (`(x, y)` or `(x, y, z)`).
  - Graph builders strictly forbid cross-partition edges and assert intra-section locality.
  - Added coordinate-shuffled control builder (`coordinate_shuffle.py`) to test spatial specificity.
  - Preserved bipartite reference connectivity as a labeled secondary variant (`variant_a_bipartite`) for cross-project comparison.
- **Why**: The object of study in spatial transcriptomics is the tissue's own spatial architecture.

## 5. GPU Result Delivery Verification & Ingestion
- **Source**: `scgraph-bench:scripts/package_gpu_results.py`, `scripts/receive_gpu_delivery.py`, `src/scgraph_bench/analysis/delivery.py`
- **Destination**: `scripts/package_gpu_results.py`, `scripts/receive_gpu_delivery.py`, `src/spatial_graph_bench/analysis/delivery.py`
- **What was taken**: 4-layer cryptographic delivery verification:
  1. Per-file SHA-256 vs batch manifest.
  2. Batch aggregate hash tamper check.
  3. Provenance hash-chain verification against local frozen artifacts.
  4. Independent metric recomputation from frozen labels.
- **What changed**:
  - Layer 4 recomputation now strictly enforces the §3.4 label-space intersection rule (calculating macro-F1 only on classes present in the training fold).
  - Added checks for spatial construction manifests and protocol variants.
- **Why**: Zero trust in GPU compute; audit logs and quarantine ledgers must verify spatial invariants and reproducible metrics.

## 6. Preprocessing & Spatial Ignorance Audit
- **Source**: `scgraph-bench:src/scgraph_bench/preprocessing/`
- **Destination**: `src/spatial_graph_bench/preprocessing/`
- **What was taken**: Frozen feature pipeline pattern, feature manifests, and training-fit-only transforms.
- **What changed**:
  - Implemented dual baseline feature pipelines:
    - **Version A (strict)**: Raw counts $\to$ library log-normalization $\to$ HVG $\to$ PCA, strictly sans spatial/neighborhood operations.
    - **Version B (platform-default)**: Deposited matrices as-is.
  - Added mandatory coordinate-shuffle feature audit: any feature that shifts under coordinate shuffling is flagged as "spatially smuggled" and removed from Version A.
- **Why**: Prevents "spatially ignorant" baselines from secretly benefiting from local smoothing or density-based QC.

## 7. Model Implementations & Baselines
- **Source**: `scgraph-bench:src/scgraph_bench/models/`
- **Destination**: `src/spatial_graph_bench/models/`
- **What was taken**: MLP training loop, PyG model wrappers, early stopping, and seed determinism.
- **What changed**:
  - Added Random Forest (RF) tabular baseline.
  - Harmonized GNN suite to the co-equal pre-registered set: **{GCN, GraphSAGE, GAT, GIN}**.
  - All architectures run on identical frozen feature representations under identical early stopping and tuning budgets.
- **Why**: The pre-registered comparison grid treats all four GNN architectures as co-equal peers against strong spatially ignorant baselines.
