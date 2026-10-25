# spatial-graph-bench

> **Q**: Does message passing over cell–cell spatial graphs improve (task: cell-type classification) over strong non-spatial baselines trained on identical features?

---

## 1. Rule

A graph architecture $M$ provides inductive benefit over cellular gene expression alone if and only if:

$$\text{Macro-F1}(M) > \text{Macro-F1}(\text{MLP}) + \epsilon, \quad \text{where } \epsilon = \max(0.005, 2\sigma_{\text{MLP}})$$

evaluated under superiority/inferiority tests and Two One-Sided Tests (TOST), with Holm-Bonferroni FWER control ($\alpha = 0.05$).

- If $|\Delta| \le \epsilon$, the model falls within the pre-registered empirical parity band (equal to twice the MLP's own standard deviation $\sigma$ across 10 seed runs).
- If $\Delta < -\epsilon$, message passing degrades classification accuracy.

---

## 2. Findings

Across 4 spatial transcriptomics platforms (1,164 benchmark runs, 10 random seeds):

1. Cell–cell spatial graphs exceed the empirical parity band in only 1 of 4 biological contexts (Stereo-seq developing axolotl brain, +3.8 percentage points; statistical verdict inconclusive).
2. In fine-grained cellular mixtures (MERFISH adult mouse spinal cord), spatial graphs fall below the empirical parity band relative to MLP baseline (−2.5 percentage points for GraphSAGE k=12).
3. Across sharp histological boundaries (Open-ST metastatic carcinoma), isotropic neighbor-averaging shows negative lift of 6.4–7.3 percentage points.
4. In structurally imbalanced tissues (10x Xenium mouse kidney), spatial message passing shows catastrophic performance loss, with negative lift ranging −33.0 to −39.2 percentage points across GCN, GAT, and GIN on the ShamL → ShamR split.
5. Physical spatial k-NN graphs fail to provide lift in MERFISH and Xenium, but non-physical expression-space bipartite reference graphs help in both cases. Perfomance restored to 0.5640 in MERFISH (+0.0367 lift) and 0.7560 in Xenium (within the empirical parity band, recovering from a −37.9 percentage-point collapse).

---

## 3. Benchmark Results

All neural models evaluated across $N=10$ independent random seeds (42–51) on frozen 50-dimensional PCA features under stage/animal/section/replicate held-out evaluation.

| Platform & Dataset | Biological Context | Evaluated Test Cells (Classes) | Spatially Ignorant MLP Macro-F1 | Parity Band ($\pm \epsilon$) | Best Spatial GNN Architecture | Best GNN Macro-F1 | Matched Lift ($\Delta$) | Empirical Verdict |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| **Stereo-seq**<br>`axolotl_telencephalon` | Developing brain<br>*(Stage held-out)* | 4,209 (15) | $0.7282 \pm 0.0140$ | $\pm 0.0280$ | GraphSAGE ($k=12$) | $0.7665 \pm 0.0106$ | $\mathbf{+0.0383 \pm 0.0230}$<br>*(peak $+0.0702$)* | **Inconclusive** |
| **MERFISH**<br>`mouse_spinal_cord` | Adult spinal cord<br>*(Animal held-out)* | 8,507 (60) | $0.5273 \pm 0.0035$ | $\pm 0.0069$ | GraphSAGE ($k=12$) | $0.5020 \pm 0.0068$ | $-0.0253 \pm 0.0082$ | **Negative Lift** |
| **Open-ST**<br>`human_lymph_node` | Metastatic carcinoma<br>*(Section held-out)* | 34,072 (14) | $0.3946 \pm 0.0027$ | $\pm 0.0054$ | GraphSAGE ($k=12$) | $0.3962 \pm 0.0102$ | $+0.0016 \pm 0.0092$ | **Inconclusive** |
| **10x Xenium**<br>`mouse_kidney` | Sham adult kidney<br>*(Replicate held-out)* | 85,880 (20) | $0.7635 \pm 0.0120$ | $\pm 0.0241$ | GraphSAGE ($k=12$) | $0.7514 \pm 0.0139$ | $-0.0121 \pm 0.0213$ | **Inconclusive** |

### Architectural Comparison on Canonical Spatial Graph ($k=12$)

| Platform & Dataset | Spatially Ignorant MLP Baseline | GCN ($k=12$) | GAT ($k=12$) | GIN ($k=12$) | GraphSAGE ($k=12$) | Architectural Finding |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Stereo-seq**<br>*(Axolotl Brain)* | $0.7282 \pm 0.0140$ | $0.7198$ ($-0.0084$) | $0.7145$ ($-0.0136$) | $0.6590$ ($-0.0692$) | $\mathbf{0.7665}$ ($\mathbf{+0.0383}$) | **GraphSAGE provides sole positive mean lift**; GCN/GAT stay within the empirical parity band. |
| **MERFISH**<br>*(Mouse Spinal Cord)* | $0.5273 \pm 0.0035$ | $0.1577$ ($-0.3696$) | $0.1810$ ($-0.3462$) | $0.1587$ ($-0.3686$) | $\mathbf{0.5020}$ ($\mathbf{-0.0253}$) | **Isotropic GNNs collapse by $-35\%$ to $-37\%$**; GraphSAGE reduces the loss to 2.5 percentage points. |
| **Open-ST**<br>*(Human Lymph Node)* | $0.3946 \pm 0.0027$ | $0.3216$ ($-0.0730$) | $0.3257$ ($-0.0689$) | $0.3303$ ($-0.0643$) | $\mathbf{0.3962}$ ($\mathbf{+0.0016}$) | **Isotropic GNNs lose 6–7 percentage points**; GraphSAGE remains within the empirical parity band. |
| **10x Xenium**<br>*(Mouse Kidney)* | $0.7635 \pm 0.0120$ | $0.3711$ ($-0.3925$) | $0.4338$ ($-0.3297$) | $0.3846$ ($-0.3789$) | $\mathbf{0.7514}$ ($\mathbf{-0.0121}$) | **Catastrophic $-33\%$ to $-39\%$ collapse under GCN/GAT/GIN**; GraphSAGE recovers most of the loss. |

Complete statistical tables, CI, and run details: [docs/results-synthesis.md](docs/results-synthesis.md) and [results/reports/table_benchmark_synthesis.md](results/reports/table_benchmark_synthesis.md).

---

## 4. Open Questions [W.I.P]

#### 1. Which graph designs helped?

- A moderate K number of neighbors (12 instead of 6), in tissues where neighboring cells really do share the same type. In those cases, wider neighborhoods filter out noise more effectively.
-  Connecting cells to a reference set of cells in gene-expression space.

#### 2. Why aren't simple physical-neighbor graphs good enough, and what would work better?

Spatial graphs have three problems:

- They connect cells that are physically close even if there is biological separation
- They force every cell to have the same number of neighbors, even when real tissue density might be different; 
- Cells near the tissue's outer edge only have neighbors on one side, which skews their results.

***Yes there are/should be better approaches!*** [see end]

#### 3. Why did connecting cells by gene expression (as opp. to physical location) work better?

- No cross-boundary mixing. A cell only connects to reference cells of similar type, regardless of what's physically next to it in the tissue.
- Pushing to an average version of its true type reduces the noise
- Physical damage in the tissue  doesn't affect the underlying gene expression

#### 4. Why do some GNN types collapse in mixed-neighbor tissue, while others don't?

Averaging (GCN, GIN, and GAT's weighted sum) mixes self and neighbors into one shared signal before transformation, so self can get overwhelmed when neighbors are misleading.  This causes macro-F1 drops (in my case down 35–39 percentage points in two datasets).

GraphSAGE keeps separate transformations for the cell itself and its neighbors, giving a "skip-connection" for the cell’s own signal
#### 5. Misc: Do cells near tissue boundaries perform worse than cells in the interior?

Across all the datasets tested (boundary-margin macro-F1 was lower than interior macro-F1 for GraphSAGE k=12)

- Leaving aside sample prep. related artifacts, cells near the outer edge of a tissue sample simply have fewer neighbors to draw on, since there's nothing beyond the boundary

---

## 5. Design Invariants

All benchmarks enforce four controlled experimental variables:

- **Zero cross-partition edges**: Strict donor- and section-held-out splits with section-own subgraphs; test-time message passing operates strictly on internal spatial coordinates within held-out tissue slices (zero cross-section or cross-partition edges).
- **Identical fixed features with audited spatial ignorance**: Random Forest, MLP, and all GNNs evaluate on identical 50-dimensional Principal Components fitted strictly on training gene expression. Coordinate-shuffle audits verify zero spatial leakage into features ($\max |\Delta| = 0.0 \le 10^{-6}$).
- **Controlled model comparison**: Standard inductive GNN architectures (GCN, GraphSAGE, GAT, GIN) evaluated against a regularized non-spatial MLP on identical train/val/test partitions across 10 random seeds (42–51).
- **Topological negative controls**: Directed degree-preserving edge rewiring and coordinate shuffling benchmarked across every condition to verify models respond to authentic tissue topology rather than unspecific smoothing artifacts.

*see - [scgraph-bench](https://github.com/ToruOkadaOi/scgraph-bench), my [ml_gist](https://gist.github.com/ToruOkadaOi/a20725a1d13a83d6884d551f53953b20#file-mlgist-md)*

---

## 6. Reproduction & Extended Documentation

<details>
<summary><b>…</b></summary>

### Turnkey Reproduction

Verify environment invariants, cryptographic hashes, and regenerate all publication figures and LaTeX tables locally:

```bash
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# One-click reproduction pipeline
./scripts/reproduce_all.sh
```

### Verified Release Artifacts

All preprocessed input datasets and benchmark sweeps are distributed with SHA-256 checksums:

| Modality | Input Release Package | Result Release Package | Test Cells | SHA-256 Status |
|:---|:---|:---|:---:|:---:|
| **Stereo-seq** | [`v0.1.0-stereoseq-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-stereoseq-inputs) | [`v0.2.0-stereoseq-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-stereoseq-results) | 4,410 | Verified PASS |
| **MERFISH** | [`v0.1.0-merfish-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) | [`v0.2.0-merfish-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-merfish-results) | 8,507 | Verified PASS |
| **Open-ST** | [`v0.1.0-openst-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-openst-inputs) | [`v0.2.0-openst-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-openst-results) | 34,072 | Verified PASS |
| **10x Xenium** | [`v0.1.0-xenium-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-xenium-inputs) | [`v0.2.0-xenium-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-xenium-results) | 85,880 | Verified PASS |

See [docs/results-index.md](docs/results-index.md) for full archive manifests and checksum verification instructions.

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

- [docs/results-synthesis.md](docs/results-synthesis.md): Comprehensive 4-modality synthesis and empirical analysis.
- [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md): Pre-registered scientific contract and statistical decision rules.
- [REPRODUCE.md](REPRODUCE.md): Step-by-step manual reproduction handbook.
- [TESTING.md](TESTING.md): 4-tier testing guide (unit, invariant validators, smoke runs, delivery audits).
- [docs/results-index.md](docs/results-index.md): Catalog of release artifacts and baseline snapshots.
- [CITATION.md](CITATION.md): Academic citation format and BibTeX.
- [LICENSE](LICENSE): MIT License.

</details>

---

## 7. Next Steps [W.I.P]

Extend to relation-specific message passing