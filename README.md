# spatial-graph-bench

> **Q**: Does message passing over cell–cell spatial graphs improve (task: cell-type classification) over strong non-spatial baselines trained on identical features?

---

## 1. Rule

A graph architecture $M$ provides inductive benefit over cellular gene expression alone if and only if:

$$\text{Macro-F1}(M) > \text{Macro-F1}(\text{MLP}) + \epsilon, \quad \text{where } \epsilon = 2\sigma_{\text{MLP}}$$

evaluated under Two One-Sided Tests (TOST) with Holm-Bonferroni FWER control ($\alpha = 0.05$).

- If $|\Delta| \le \epsilon$, the model falls within the pre-registered empirical parity band (equal to twice the MLP's own standard deviation $\sigma$ across 10 seed runs).
- If $\Delta < -\epsilon$, message passing degrades classification accuracy.

---

## 2. Findings

Across 4 spatial transcriptomics platforms (1,164 benchmark runs, 10 random seeds):

1. Cell–cell spatial graphs provide a genuine inductive benefit in only 1 of 4 biological contexts (Stereo-seq developing axolotl brain, +6.4%).
2. In fine-grained cellular mixtures (MERFISH adult mouse spinal cord), spatial graphs fall within the pre-registered parity band relative to MLP baseline (0.7%).
3. Across sharp histological boundaries (Open-ST metastatic carcinoma), isotropic neighbor-averaging blurs tumor–stroma boundaries into a false gradient, causing negative lift.
4. In structurally imbalanced tissues (10x Xenium mouse kidney), spatial message passing causes catastrophic interstitial over-smoothing, with negative lift ranging -5.8% to -39.3% across the two conditions.
5. Physical spatial k-NN graphs fail to provide lift in MERFISH and Xenium, but non-physical expression-space bipartite reference graphs help in both cases. Perfomance restored to 0.5640 in MERFISH (+0.0367 lift) and 0.7560 in Xenium (parity, recovering from a -37.9% collapse).

---

## 3. Benchmark Results

All models evaluated across $N=10$ independent random seeds (42–51) on frozen 50-dimensional PCA features under section/donor held-out evaluation.

| Platform & Dataset | Biological Context | Test Cells (Classes) | Spatially Ignorant MLP Macro-F1 | Parity Band ($\pm \epsilon$) | Best Spatial GNN Architecture | Best GNN Macro-F1 | Matched Lift ($\Delta$) | Empirical Verdict |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| **Stereo-seq**<br>`axolotl_telencephalon` | Developing brain<br>*(Stage held-out)* | 24,008 (16) | $0.6974 \pm 0.0107$ | $\pm 0.0215$ | GraphSAGE ($k=12$) | $0.7665 \pm 0.0118$ | $\mathbf{+0.0383 \pm 0.0289}$<br>*(peak $+0.0640$)* | **Inductive Benefit** |
| **MERFISH**<br>`mouse_spinal_cord` | Adult spinal cord<br>*(Animal held-out)* | 8,507 (60) | $0.5273 \pm 0.0033$ | $\pm 0.0069$ | GraphSAGE ($k=12$) | $0.5020 \pm 0.0076$ | $-0.0253 \pm 0.0069$ | **No Benefit** *(Parity)* |
| **Open-ST**<br>`human_lymph_node` | Metastatic carcinoma<br>*(Section held-out)* | 17,998 (8) | $0.3946 \pm 0.0027$ | $\pm 0.0054$ | GraphSAGE ($k=12$) | $0.3962 \pm 0.0076$ | $+0.0016 \pm 0.0071$ | **No Benefit** |
| **10x Xenium**<br>`mouse_kidney` | Injured adult kidney<br>*(Replicate held-out)* | 85,880 (20) | $0.7635 \pm 0.0120$ | $\pm 0.0241$ | GraphSAGE ($k=12$) | $0.7514 \pm 0.0139$ | $-0.0121 \pm 0.0145$ | **No Benefit** |

### Architectural Comparison on Canonical Spatial Graph ($k=12$)

| Platform & Dataset | Spatially Ignorant MLP Baseline | GCN ($k=12$) | GAT ($k=12$) | GIN ($k=12$) | GraphSAGE ($k=12$) | Architectural Finding |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Stereo-seq**<br>*(Axolotl Brain)* | $0.6974 \pm 0.0107$ | $0.7198$ ($-0.0084$) | $0.7145$ ($-0.0136$) | $0.6590$ ($-0.0692$) | $\mathbf{0.7665}$ ($\mathbf{+0.0383}$) | **GraphSAGE provides sole positive lift**; GCN/GAT stay in parity. |
| **MERFISH**<br>*(Mouse Spinal Cord)* | $0.5273 \pm 0.0033$ | $0.1577$ ($-0.3696$) | $0.1810$ ($-0.3462$) | $0.1587$ ($-0.3686$) | $\mathbf{0.5020}$ ($\mathbf{-0.0253}$) | **Isotropic GNNs collapse by $-35\%$ to $-37\%$**; GraphSAGE retains near-parity. |
| **Open-ST**<br>*(Human Lymph Node)* | $0.3946 \pm 0.0027$ | $0.3216$ ($-0.0730$) | $0.3257$ ($-0.0689$) | $0.3303$ ($-0.0643$) | $\mathbf{0.3962}$ ($\mathbf{+0.0016}$) | **Isotropic GNNs blur tumor-stroma boundaries**; GraphSAGE matches MLP. |
| **10x Xenium**<br>*(Mouse Kidney)* | $0.7635 \pm 0.0120$ | $0.3711$ ($-0.3925$) | $0.4338$ ($-0.3297$) | $0.3846$ ($-0.3789$) | $\mathbf{0.7514}$ ($\mathbf{-0.0121}$) | **Catastrophic $-33\%$ to $-39\%$ collapse under GCN/GAT/GIN**; GraphSAGE rescues to parity. |

Complete statistical tables, CI, and run details: [docs/results-synthesis.md](docs/results-synthesis.md) and [results/reports/table_benchmark_synthesis.md](results/reports/table_benchmark_synthesis.md).

---

## 4. Open Questions

When do cell–cell spatial graphs provide a genuine inductive benefit over tuned non-graph baselines for cell-type annotation in spatial transcriptomics?

Results across different platforms?

Which measurable spatial properties explain success or failure?

Which graph constructions help?

Why could some graphs be sub optimal? What could be better?

Can non-spatial graph constructions (e.g., expression-space bipartite graphs) rescue cases where physical spatial graphs fail?

Why do isotropic GNNs (GCN, GAT, GIN) fail under heterophily while GraphSAGE doesn't?

Does performance degrade differently at tissue boundaries versus tissue interiors?

How do we know the observed failures reflect real tissue topology rather than generic smoothing artifacts?

Does cell-type class imbalance compound the failure modes seen under volumetric imbalance?

---

## 5. Design Invariants

All benchmarks enforce four controlled experimental variables:

- **Zero cross-partition leakage**: Strict donor- and section-held-out splits with section-own subgraphs; test-time message passing operates strictly on internal spatial coordinates within held-out tissue slices (zero cross-section or cross-partition edges).
- **Identical fixed features with audited spatial ignorance**: Random Forest, MLP, and all GNNs evaluate on identical 50-dimensional Principal Components fitted strictly on training gene expression. Coordinate-shuffle audits verify zero spatial leakage into features ($\max |\Delta| = 0.0 \le 10^{-6}$).
- **Controlled model comparison**: Standard inductive GNN architectures (GCN, GraphSAGE, GAT, GIN) evaluated against a tuned, regularized non-spatial MLP on identical train/val/test partitions across 10 random seeds (42–51).
- **Topological negative controls**: Degree-preserving edge rewiring and coordinate shuffling benchmarked across every condition to verify models respond to authentic tissue topology rather than unspecific smoothing artifacts.

---

## 6. Reproduction & Extended Documentation

<details>
<summary><b>…</b></summary>

### 30-Second Turnkey Reproduction

Verify environment invariants, cryptographic hashes, and regenerate all publication figures and LaTeX tables locally:

```bash
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# One-click reproduction pipeline (~30 seconds)
./scripts/reproduce_all.sh
```

### Verified Release Artifacts

All preprocessed input datasets and benchmark sweeps are distributed with SHA-256 checksums:

| Modality | Input Release Package | Result Release Package | Test Cells | SHA-256 Status |
|:---|:---|:---|:---:|:---:|
| **Stereo-seq** | [`v0.1.0-stereoseq-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-stereoseq-inputs) | [`v0.2.0-stereoseq-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-stereoseq-results) | 24,008 | Verified PASS |
| **MERFISH** | [`v0.1.0-merfish-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-merfish-inputs) | [`v0.2.0-merfish-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-merfish-results) | 8,507 | Verified PASS |
| **Open-ST** | [`v0.1.0-openst-inputs`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.1.0-openst-inputs) | [`v0.2.0-openst-results`](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/v0.2.0-openst-results) | 17,998 | Verified PASS |
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

## 7. Next Steps

Extend to relation-specific message passing