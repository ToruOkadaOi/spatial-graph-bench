# spatial-graph-bench

> **Q**: Does message passing over cell–cell spatial graphs improve (task: cell-type classification) over strong non-spatial baselines trained on identical features?

---

## 1. Rule

A graph architecture $M$ provides inductive benefit over cellular gene expression alone if and only if:

$$\text{Macro-F1}(M) > \text{Macro-F1}(\text{MLP}) + \epsilon, \quad \text{where } \epsilon = \max(0.005, 2\sigma_{\text{MLP}})$$

evaluated under paired superiority/inferiority tests and Two One-Sided Tests (TOST), with Holm-Bonferroni FWER control ($\alpha = 0.05$). The corrected sweep applies this correction across all 336 hypotheses (three tests for each of 112 matched model/graph/dataset comparisons).

- A positive lift is reported when the paired superiority test supports $\Delta > \epsilon$ after correction; a negative lift when the paired inferiority test supports $\Delta < -\epsilon$.
- Parity is reported when neither positive nor negative lift is established after correction. It means no resolved lift in this analysis; it does not claim formal equivalence unless TOST supports equivalence within the band.

---

## 2. Findings

Across 4 spatial transcriptomics platforms (1,164 corrected benchmark runs, 10 random seeds):

1. Across 112 matched comparisons, 78 show negative lift and 34 are parity after the global correction. None shows a statistically supported positive lift.
2. For physical spatial k-NN graphs alone, 20 of 32 comparisons show negative lift and 12 are parity.
3. In fine-grained cellular mixtures (MERFISH adult mouse spinal cord), all four architectures show negative lift on the physical k=12 graph; GraphSAGE is 2.2 percentage points below the MLP.
4. Across sharp histological boundaries (Open-ST metastatic carcinoma), GCN, GAT, and GIN show negative lift of 6.3–7.3 percentage points on the physical k=12 graph. GraphSAGE is near the MLP baseline and is classified as parity.
5. In structurally imbalanced tissues (10x Xenium mouse kidney), GCN, GAT, and GIN show negative lift of 33.2–39.4 percentage points on the ShamL → ShamR split. GraphSAGE performs much closer to the MLP and is classified as parity. Expression-space bipartite reference graphs do not show statistically supported positive lift under the corrected analysis.

---

## 3. Benchmark Results

All neural models evaluated across $N=10$ independent random seeds (42–51) on the same frozen 50-dimensional PCA features under stage/animal/section/replicate held-out evaluation. The GNNs were retrained with train-only optimization and post-update validation checkpoint selection; test labels were not used for model selection. Test macro-F1 uses labels shared between the training and held-out partitions.

| Platform & Dataset | Biological Context | Evaluated Test Cells (Classes) | Spatially Ignorant MLP Macro-F1 | Parity Band ($\pm \epsilon$) | Spatial GNN | GNN Macro-F1 | Matched Lift ($\Delta$) | Empirical Verdict |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| **Stereo-seq**<br>`axolotl_telencephalon` | Developing brain<br>*(Stage held-out)* | 4,209 (15) | $0.7286 \pm 0.0106$ | $\pm 0.0212$ | GraphSAGE ($k=12$) | $0.7446 \pm 0.0184$ | $\mathbf{+0.0160 \pm 0.0214}$ | **Parity** |
| **MERFISH**<br>`mouse_spinal_cord` | Adult spinal cord<br>*(Animal held-out)* | 8,507 (60) | $0.5285 \pm 0.0023$ | $\pm 0.0050$ | GraphSAGE ($k=12$) | $0.5061 \pm 0.0049$ | $-0.0224 \pm 0.0049$ | **Negative Lift** |
| **Open-ST**<br>`human_lymph_node` | Metastatic carcinoma<br>*(Section held-out)* | 34,072 (14) | $0.3942 \pm 0.0032$ | $\pm 0.0064$ | GraphSAGE ($k=12$) | $0.3959 \pm 0.0067$ | $+0.0017 \pm 0.0066$ | **Parity** |
| **10x Xenium**<br>`mouse_kidney` | Sham adult kidney<br>*(Replicate held-out)* | 85,880 (20) | $0.7680 \pm 0.0036$ | $\pm 0.0072$ | GraphSAGE ($k=12$) | $0.7576 \pm 0.0077$ | $-0.0104 \pm 0.0078$ | **Parity** |

Model scores show mean $\pm$ standard deviation across seeds. Matched lift is paired by seed. Verdicts use the global 336-hypothesis Holm correction.

### Architectural Comparison on Canonical Spatial Graph ($k=12$)

| Platform & Dataset | Spatially Ignorant MLP Baseline | GCN ($k=12$) | GAT ($k=12$) | GIN ($k=12$) | GraphSAGE ($k=12$) | Architectural Finding |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Stereo-seq**<br>*(Axolotl Brain)* | $0.7286$ | $0.6887$ ($-0.0398$) | $0.6652$ ($-0.0633$) | $0.6983$ ($-0.0302$) | $\mathbf{0.7446}$ ($\mathbf{+0.0160}$) | **GraphSAGE has the highest mean**, but every architecture is parity after global correction. |
| **MERFISH**<br>*(Mouse Spinal Cord)* | $0.5285$ | $0.1621$ ($-0.3664$) | $0.1845$ ($-0.3440$) | $0.1615$ ($-0.3670$) | $\mathbf{0.5061}$ ($\mathbf{-0.0224}$) | **All four architectures show negative lift.** |
| **Open-ST**<br>*(Human Lymph Node)* | $0.3942$ | $0.3302$ ($-0.0639$) | $0.3215$ ($-0.0727$) | $0.3309$ ($-0.0632$) | $\mathbf{0.3959}$ ($\mathbf{+0.0017}$) | **GCN/GAT/GIN show negative lift**; GraphSAGE is parity. |
| **10x Xenium**<br>*(Mouse Kidney)* | $0.7680$ | $0.3795$ ($-0.3885$) | $0.4361$ ($-0.3319$) | $0.3744$ ($-0.3936$) | $\mathbf{0.7576}$ ($\mathbf{-0.0104}$) | **GCN/GAT/GIN show large negative lift**; GraphSAGE is parity. |

The corrected sweep contains 112 matched comparisons across four datasets, four architectures, and seven graph conditions. Full run instructions: [docs/RETRAINING.md](docs/RETRAINING.md). Raw-to-PCA reproduction: [docs/reproducibility-v0.3.2.md](docs/reproducibility-v0.3.2.md).

---

## 4. Open Questions [W.I.P]

#### 1. Which graph designs helped?

- No tested graph design showed a statistically supported positive lift after the corrected global analysis. GraphSAGE on the Stereo-seq physical k=12 graph had the highest mean lift (+1.6 percentage points), but is classified as parity.
- Connecting cells to a reference set in gene-expression space produced positive mean lifts in some MERFISH runs, but none passed the corrected positive-lift criterion.

#### 2. Why aren't simple physical-neighbor graphs good enough, and what would work better?

Possible limitations to test include:

- They can connect cells that are physically close even if there is biological separation.
- They may force every cell to have the same number of neighbors, even when real tissue density might be different.
- Cells near the tissue's outer edge have neighbors on one side only; whether this skews their results remains to be tested.

***Yes there are/should be better approaches!*** [see end]

#### 3. Would connecting cells by gene expression (as opp. to physical location) work better?

- This remains a direction to explore; the corrected benchmark did not establish a positive lift for expression-space reference graphs.
- A hypothesis is that expression-similar references could reduce cross-boundary mixing, regardless of physical proximity.
- Physical damage in the tissue may affect location more than the underlying gene expression.

#### 4. Why do some GNN types collapse in mixed-neighbor tissue, while others don't?

In MERFISH and Xenium, GCN, GAT, and GIN show drops of roughly 33–39 percentage points on the physical k=12 graph. One possible explanation is that neighborhood aggregation mixes in misleading cells; this remains a hypothesis about why these models performed poorly here.

GraphSAGE performed better than the other GNNs in those settings; the corrected verdict is parity for Xenium and negative for MERFISH.
#### 5. Misc: Do cells near tissue boundaries perform worse than cells in the interior?

This was not evaluated in the corrected sweep and remains an open question.

- One possibility is that cells near the outer edge have fewer neighbors to draw on, since there's nothing beyond the boundary.

---

## 5. Design Invariants

All benchmarks enforce four controlled experimental variables:

- **Zero cross-partition edges**: Strict donor- and section-held-out splits with section-own subgraphs; test-time message passing operates strictly on internal spatial coordinates within held-out tissue slices (zero cross-section or cross-partition edges).
- **Identical fixed features with audited spatial ignorance**: Random Forest, MLP, and all GNNs evaluate on identical 50-dimensional Principal Components fitted strictly on training gene expression. Coordinate-shuffle audits verify zero spatial leakage into features ($\max |\Delta| = 0.0 \le 10^{-6}$).
- **Controlled model comparison**: Standard inductive GNN architectures (GCN, GraphSAGE, GAT, GIN) evaluated against a regularized non-spatial MLP on identical train/val/test partitions across 10 random seeds (42–51). GNN optimization uses training nodes only; checkpoint selection uses fresh post-update validation predictions.
- **Topological negative controls**: Coordinate shuffling and undirected degree-preserving edge rewiring are performed within each partition and section and benchmarked across every condition.

*see - [scgraph-bench](https://github.com/ToruOkadaOi/scgraph-bench), my [ml_gist](https://gist.github.com/ToruOkadaOi/a20725a1d13a83d6884d551f53953b20#file-mlgist-md)*

---

## 6. Reproduction & Extended Documentation

<details>
<summary><b>…</b></summary>

### Reproducing the corrected benchmark

The corrected 1,164-run sweep uses frozen PCA features, split definitions, and graph inputs. Re-run it using the setup and GPU instructions in [docs/RETRAINING.md](docs/RETRAINING.md). Rebuild raw-to-PCA features with the matching source data versions using [docs/reproducibility-v0.3.2.md](docs/reproducibility-v0.3.2.md). Release archives and checksums will be linked here when the corrected result package is published.

### Developer Environment Setup

```bash
# Synchronize exact dependencies from uv.lock
uv sync --frozen --all-extras

# Run unit tests and invariant validators
uv run pytest -v tests

# Run linter and formatting checks
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

### Protocol Documentation

- [docs/RETRAINING.md](docs/RETRAINING.md): Corrected strict-inductive training protocol and run instructions.
- [docs/reproducibility-v0.3.2.md](docs/reproducibility-v0.3.2.md): Raw-to-PCA reproduction and fitted preprocessing.
- [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md): Study protocol and statistical decision rules.
- [REPRODUCE.md](REPRODUCE.md): Step-by-step manual reproduction handbook.
- [TESTING.md](TESTING.md): 4-tier testing guide (unit, invariant validators, smoke runs, delivery audits).
- [CITATION.md](CITATION.md): Academic citation format and BibTeX.
- [LICENSE](LICENSE): MIT License.

</details>

---

## 7. Next Steps [W.I.P]

Extend to relation-specific message passing
