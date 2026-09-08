# STUDY_PROTOCOL.md: Pre-Registration Protocol for Spatial Graph Inductive-Benefit Benchmark

**Project**: `spatial-graph-bench`  
**Registration Date**: 2026-09-04  
**Pre-Registration Commit Hash**: `449a4d42cab2eb7a09e8c301c540659d8888cabe`  
**Primary Question**: Do cell–cell spatial graphs provide a **genuine inductive benefit** over strong
*spatially ignorant* baselines for **cell-type annotation** in spatial transcriptomics (ST)?  
**Prior Framing**: Pre-registered replication and domain transplant of `scgraph-bench`, which observed
0/45 positive comparisons (GCN lift −0.017…−0.030 macro-F1 across 25 seeds; GraphSAGE near-parity;
no graph construction outperforming a tuned MLP baseline).

---

## 1. Pre-Registered Comparison Set & Evaluation Grid

### 1.1 Closed Comparison Grid
All comparisons are fixed prior to any model training. The primary pre-registered grid is closed:
$$\{\text{MLP, RF}\} \times \{\text{GCN, GraphSAGE, GAT, GIN}\} \times \{\text{Real Spatial } k\text{-NN, Rewired Control}\} \times 4 \text{ Datasets}$$

- **All four GNN architectures (GCN, GraphSAGE, GAT, GIN) are co-equal members of the pre-registered set.** There is no primary or secondary hierarchy among them.
- Every cell of this grid must be executed, evaluated, and reported. No cell may be omitted or selectively suppressed.
- **Hypothesis Testing & Significance**: Any claim of significant positive lift over the tuned baseline must survive:
  1. Family-wise error rate (FWER) control via Holm–Bonferroni correction over the complete grid of tests.
  2. The strict non-inferiority / parity band definition (§3).
- **Exploratory Status**: Any model, graph construction (e.g., Delaunay triangulation, histology-informed graphs), or split variant (e.g., spatial-block holdout) outside this pre-registered grid is explicitly designated as **exploratory**. A positive exploratory finding shall never constitute the headline conclusion of this study.

### 1.2 Benchmark Platform Datasets

| # | Dataset ID | Platform | Biological System | Split Stratum | Canonical Holdout Language |
|---|---|---|---|---|---|
| 1 | `merfish_mouse_spinal_cord` | MERFISH (500 genes) | Adult mouse spinal cord (18 sections, 41,267 cells) | Animal (`Mouse ID`: F3, F4, F5, M4, M5) | **donor-held-out** |
| 2 | `openst_human_lymph_node` | Open-ST (3D stack) | Human metastatic lymph node (19 serial sections, 350 µm stack) | Depth stack section (Ref #6 vs Query #19) | **section-held-out** |
| 3 | `xenium_mouse_kidney` | Xenium (300 genes) | Mouse kidney IRI / repair (12 sections, 6 stages) | Technical replicate per stage (Left vs Right kidney) | **replicate-held-out** |
| 4 | `stereoseq_axolotl_telencephalon` | Stereo-seq | Axolotl telencephalon (development & regeneration) | Biological slice replicate (Stage44 + Stage54 vs Stage57; Stage54 vs 44; 5DPI) | **replicate-held-out** |

---

## 2. Protocol Variants & Connectivity Semantics

### 2.1 Canonical Protocol: Section-Own-Subgraph Inductive Evaluation
- **Definition**: The model is trained on reference partitions (training donors/sections and their respective subgraphs). At inference time, the model is applied to the held-out test partition using **exclusively that partition's own spatial subgraph**.
- **Connectivity Rules**:
  - Test-time message passing aggregates information strictly from other observed spots/cells within the same held-out physical section.
  - Zero cross-partition edges ($\text{train} \leftrightarrow \text{val}$, $\text{train} \leftrightarrow \text{test}$, $\text{val} \leftrightarrow \text{test}$).
  - Zero cross-section edges (even in registered 3D stacks, sections are isolated unless a 3D protocol is explicitly invoked).
- **Rationale**: Reflects real-world spatial annotation where an unannotated query tissue section is profiled, and its own spatial coordinate topology is available for local message aggregation.

### 2.2 Secondary Variant A: Bipartite Reference Connectivity (scgraph-bench v0 Semantics)
- **Definition**: Test and validation cells connect exclusively to training-reference cells via feature-similarity edges. Test-to-test and val-to-val edges are explicitly disabled ($0\text{ test-test edges}$).
- **Purpose**: Direct backwards compatibility with the single-cell benchmark (`scgraph-bench`). Disentangles "graph as reference-feature smoothing" from "graph as true spatial tissue architecture."

### 2.3 Secondary Variant B: Pooled Transductive Evaluation
- **Definition**: A single unified graph constructed over all cells simultaneously (training, validation, and test nodes present in one adjacency matrix; labels masked on validation and test).
- **Purpose**: Diagnostic control quantifying how much apparent GNN benefit in prior literature is attributable to transductive test-feature access during message passing.

*Invariant*: Metrics from canonical and secondary protocols shall never be combined or averaged in the same summary table.

---

## 3. Parity Band & Equivalence Testing Specification

To prevent declaring statistical noise as substantive "lift", we establish a formal tolerance band:

1. **Empirical Parity Band**:
   $$\text{Parity Band} = \left[ -2\sigma_{\text{MLP}}, +2\sigma_{\text{MLP}} \right]$$
   where $\sigma_{\text{MLP}}$ is the sample standard deviation of test macro-F1 across $\ge 10$ independently seeded runs of the frozen MLP baseline.
   - Any observed lift ($\Delta = \text{macro-F1}_{\text{GNN}} - \text{macro-F1}_{\text{MLP}}$) falling within this band is formally reported as **parity**, not positive or negative.

2. **Formal Equivalence Test (TOST)**:
   - Two One-Sided Tests (TOST) conducted against a pre-registered equivalence margin $\epsilon = 0.005$ macro-F1 (or $2\sigma_{\text{MLP}}$, whichever is larger).
   - Equivalence is accepted at $\alpha = 0.05$ if the 90% confidence interval of $\Delta$ lies entirely within $(-\epsilon, +\epsilon)$.

3. **Classification Taxonomy**:
   Every model comparison is categorized into exactly one of:
   - **Positive Lift**: $\Delta > 0$ and $p_{\text{superiority}} < 0.05$ (after FWER correction) and $\Delta > +\text{band}$.
   - **Parity**: 90% CI within $[-\epsilon, +\epsilon]$ or $|\Delta| \le \text{band}$.
   - **Negative Lift**: $\Delta < 0$ and $p_{\text{inferiority}} < 0.05$ and $\Delta < -\text{band}$.

---

## 4. Metric Validity Rule (Anti-Circularity Protocol)

In spatial transcriptomics, tissue domains are spatially clustered. A naive metric that rewards adjacent cells having identical labels is mathematically isomorphic to GCN message passing and circular.

### 4.1 Metric Validation Protocol
Before any metric enters headline evidence, it must undergo negative-control calibration:
1. Model evaluated on:
   - (a) Real spatial $k$-NN graph
   - (b) Degree-preserving rewired graph control
   - (c) Coordinate-shuffled graph control
2. **Decision Rule**:
   - If the metric improves under (a) but remains flat under (b) and (c), it is certified as measuring genuine spatial inductive benefit.
   - If the metric improves under both (a) and (b)/(c), it is confounded by topological smoothing or label spatial autocorrelation. It is immediately disqualified from headline reporting and demoted to diagnostic-only.

### 4.2 Metric Tiers
- **Tier 1: Primary Evidence**:
  - Matched same-seed macro-F1 lift vs. tuned MLP on the held-out test partition.
- **Tier 2: Spatially Independent Secondary Evidence**:
  - Marker-gene concordance via single-sample GSEA (ssGSEA AUC-ROC) against ground-truth cell-type marker gene sets.
  - Transcriptional silhouette coefficient in uncorrupted raw expression space.
- **Tier 3: Structure-Aware Diagnostics (Non-Primary)**:
  - AvgBIO, ECS, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Adjusted Mutual Information (AMI), Average Silhouette Width (ASW).

---

## 5. Unseen-Class Handling & Label Coverage Policy

Because biological donor and section holdouts inevitably exclude rare or spatially restricted subtypes:
1. **Intersection Label Space**:
   For any split fold, evaluation metrics (macro-F1, balanced accuracy) are computed strictly over the **intersection of the training fold and test fold label spaces**:
   $$\mathcal{Y}_{\text{eval}} = \mathcal{Y}_{\text{train}} \cap \mathcal{Y}_{\text{test}}$$
2. **Uniformity Across Models**:
   - Cells belonging to excluded classes ($\mathcal{Y}_{\text{test}} \setminus \mathcal{Y}_{\text{train}}$) are completely removed from evaluation for all models (MLP, RF, GCN, GraphSAGE, GAT, GIN).
   - No model-specific exclusions are permitted.
3. **Auditability**:
   - Every excluded class and cell count is recorded in the split manifest (`splits/{dataset}/{split_id}.json`).
   - Per-fold class coverage ($\frac{|\mathcal{Y}_{\text{eval}}|}{|\mathcal{Y}_{\text{test}}|}$) must be reported in all result tables.

---

## 6. Spatial-Ignorance Audit & Feature Invariants

To guarantee that baseline models are genuinely *spatially ignorant* and that GNNs do not benefit from "spatially smuggled" features:
1. **Dual Feature Pipelines**:
   - **Version A (Strict Spatial Ignorance)**: Raw count matrix $\to$ library size log-normalization ($\text{target\_sum}=10^4$, $\log(1+p)$) $\to$ Highly Variable Genes (HVG, Seurat flavor fit on train only) $\to$ PCA (50 components fit on train only). Zero neighborhood, radius, or smoothing operations.
   - **Version B (Platform-Default)**: Deposited preprocessed expression matrix as provided by authors.
2. **Coordinate-Shuffle Feature Test**:
   - The preprocessing pipeline is executed on coordinate-shuffled inputs. Any feature whose numerical value changes under spatial permutation is rejected from Version A and logged.
3. **Boundary / Degree Confound Stratification**:
   - Every cell's spatial degree and distance to tissue boundary are precomputed.
   - All evaluation scripts report matched lift stratified into **Interior cells** vs. **Boundary-margin cells** to determine if graph benefits are localized or confounded by tissue edge artifacts.

---

## 7. Hyperparameter Budget & Reproducibility Contract

1. **Zero Test-Data Contamination**:
   - Hyperparameter tuning for every model (MLP, RF, GNNs) is restricted entirely to training partitions via internal validation splits or cross-validation.
   - Test partitions are evaluated exactly once by the frozen, selected model checkpoints.
2. **Harmonized Tuning Budgets**:
   - Baselines receive an equal or greater number of random hyperparameter search draws compared to GNN architectures.
   - Early stopping patience (15 epochs) and learning rate scheduling are identical across neural models.
3. **Cryptographic Provenance Chain**:
   $$\text{Census Raw Hash} \longrightarrow \text{Split Hash} \longrightarrow \text{Feature Manifest Hash} \longrightarrow \text{Construction Manifest Hash} \longrightarrow \text{Run Manifest Hash}$$
   No result is ingested or reported unless all hashes in this chain match canonical frozen records.
