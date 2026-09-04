# DECISIONS_NEEDED.md: Owner Decision Log & Protocol Specifications

This document tracks scientific ambiguities, protocol choices, design alternatives, and the owner's explicit decisions for `spatial-graph-bench`.

---

## 1. Resolved & Approved Decisions

### Decision 1: Canonical Evaluation Protocol (Section-Own Subgraph Inductive)
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: The primary benchmark protocol is **section-own-subgraph inductive evaluation**.
- **Specification**: The model is trained on reference partitions (training donors/sections and their spatial subgraphs) and evaluated on held-out test partitions using exclusively each test partition's own spatial coordinates and internal subgraph.
- **Invariants**: No cross-partition edges, no cross-section edges. Message passing at test time strictly aggregates from observed cells within the same held-out physical section.

### Decision 2: Secondary Protocol Variants (Bipartite & Pooled Transductive)
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: Two secondary comparison variants are maintained:
  - **Variant A (Bipartite Reference Connectivity)**: Test/val cells connect only to training-reference cells via feature similarity; test–test edges disabled ($0$ test-test edges). Preserves `scgraph-bench` v0 semantics for single-cell vs spatial cross-comparison.
  - **Variant B (Pooled Transductive)**: Single unified graph across all partitions with masked labels, serving as an upper-bound diagnostic of transductive feature leakage.
- **Reporting Rule**: Numbers from canonical and variant protocols shall never be mixed or averaged in single summary tables.

### Decision 3: Co-Equal Status of Pre-Registered GNN Architectures
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: The four architectures **{GCN, GraphSAGE, GAT, GIN}** are co-equal members of the pre-registered comparison grid.
- **Specification**: There is no primary vs. secondary hierarchy among them. Anti-cherry-picking discipline is enforced by pre-registering the complete grid and requiring family-wise error control (Holm–Bonferroni) across all tests.

### Decision 4: Tolerance Band & Equivalence Margin
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: The parity band is defined as $\pm 2\sigma_{\text{MLP}}$ calculated across $\ge 10$ seeds of the frozen MLP baseline. Formal equivalence testing uses TOST with an equivalence margin $\epsilon = 0.005$ macro-F1 (or $2\sigma_{\text{MLP}}$).
- **Specification**: Every delta is reported with its parity classification and band width.

### Decision 5: Anti-Circularity Metric Validation Protocol
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: Metrics based on spatial label consistency are validated against degree-preserving rewired and coordinate-shuffled controls. Any metric improving under both real and control graphs is disqualified from primary evidence and relegated to diagnostics.

### Decision 6: Unseen-Class Handling Policy
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: Macro-F1 is computed strictly over the intersection of the training-fold and test-fold label space ($\mathcal{Y}_{\text{train}} \cap \mathcal{Y}_{\text{test}}$). Cells belonging to classes absent from training are uniformly excluded from evaluation across all models, and per-fold coverage is reported.

### Decision 7: Platform Hierarchy & Holdout Language (Phase 0 Gates)
- **Status**: **APPROVED** by Phase 0 Audit
- **Decision**:
  - **MERFISH mouse spinal cord**: Gate G2 confirmed explicit `Mouse ID` in `obs`. Hierarchy is **donor-held-out** (or **mouse-held-out**) across 5 adult mice (F3, F4, F5, M4, M5) across 18 sections (41,267 cells).
  - **Open-ST human metastatic lymph node**: Gate G4 confirmed 19 serial sections in 3D stack. Hierarchy is **section-held-out** (reference section #6, query section #19).
  - **Xenium mouse kidney IRI**: Gate G3 confirmed absence of animal IDs in GEO metadata. Hierarchy is **replicate-held-out** (rep1 Left vs rep2 Right), never "donor-held-out".
  - **Stereo-seq axolotl telencephalon**: Gate G5 confirmed absence of animal IDs. Hierarchy is **replicate-held-out** (5DPI rep1+rep2 reference vs rep3 query; Stage54 reference vs Stage44 query).

### Decision 8: Dual Baseline Feature Pipelines & Coordinate-Shuffle Audit
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: Maintain Version A (strict: raw counts $\to$ library log-normalization $\to$ HVG $\to$ PCA fit on train only, sans spatial operations) and Version B (platform-default). Baseline features must pass the coordinate-shuffle test before freezing.

### Decision 9: Boundary & Degree Confound Stratification
- **Status**: **APPROVED** by Protocol Specification
- **Decision**: Spatial degree and distance to tissue boundary are precomputed for every spot. Results must report matched lift stratified into interior vs. boundary-margin cells.

---

## 2. Open Technical Decisions (Scheduled for Phase 4–6)

### Open Decision 1: Spatial k-NN Neighbor Count ($k$) & Kernel Bandwidth
- **Context**: Canonical spatial graphs require fixing $k$ for physical coordinate $k$-NN graphs.
- **Proposed Default**: $k \in \{6, 12, 18\}$ representing first- and second-order spatial rings in 2D tissue, with $k=6$ as canonical.
- **Owner Action**: Confirm $k=6$ as primary physical neighborhood or specify alternative grid.

### Open Decision 2: GNN Hidden Dimensions and Dropout Bounds for GPU Handoff
- **Context**: Hyperparameter tuning grid for GCN, GraphSAGE, GAT, GIN during GPU sweep.
- **Proposed Default**: Hidden dim $\in \{64, 128, 256\}$, layers $\in \{2, 3\}$, dropout $\in \{0.1, 0.3, 0.5\}$, learning rate $\in [10^{-4}, 10^{-2}]$, weight decay $\in [10^{-5}, 10^{-3}]$, 15-epoch early stopping.
- **Owner Action**: Approve parameter bounds before emitting `HANDOFF_TO_GPU.md`.
