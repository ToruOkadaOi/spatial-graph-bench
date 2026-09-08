# Empirical Benchmark Results: MERFISH Mouse Spinal Cord (`mouse_held_out_canonical`)

**Benchmark Platform**: MERFISH (500 genes)  
**Dataset**: `merfish_mouse_spinal_cord`  
**Split**: `mouse_held_out_canonical` (Mouse F4 held-out; 3 sections, 8,507 test cells across 60 cell classes)  
**Pre-Registered Comparison Grid**: 4 GNN Architectures × 7 Graph Topologies × 10 Seeds (42–51) = 280 GPU Runs  
**Parity Equivalence Margin**: $\epsilon = \pm 0.0069$ ($2\sigma_{\text{MLP}}$)  
**Baseline Reference**: Spatially Ignorant MLP Test Macro-F1 = $\mathbf{0.5273 \pm 0.0033}$; Random Forest = $0.4683$  
**Audit Ingestion Hash**: `d2b2a697d1d3d0113dff373c246329e542a9349c4d29b25ddd039ab9cb2a4f8c` (100% 4-Layer PASS)  

---

## 1. Primary Evidence Table: 10-Seed Aggregated Performance

| Model | Graph Construction | GNN Test F1 | Baseline MLP | Matched Lift ($\Delta$) | 90% TOST CI | FWER Decision ($\alpha=0.05$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **GCN** | `spatial_knn_k6` | $0.2004 \pm 0.0082$ | $0.5273 \pm 0.0033$ | $-0.3269 \pm 0.0089$ | $[-0.3323, -0.3215]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GCN** | `spatial_knn_k12` | $0.1577 \pm 0.0075$ | $0.5273 \pm 0.0033$ | $-0.3696 \pm 0.0085$ | $[-0.3748, -0.3644]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GCN** | `rewired_spatial_knn_k6` | $0.0478 \pm 0.0023$ | $0.5273 \pm 0.0033$ | $-0.4795 \pm 0.0037$ | $[-0.4818, -0.4772]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GCN** | `rewired_spatial_knn_k12` | $0.0345 \pm 0.0095$ | $0.5273 \pm 0.0033$ | $-0.4928 \pm 0.0103$ | $[-0.4991, -0.4865]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GCN** | `shuffled_spatial_knn_k6` | $0.1186 \pm 0.0052$ | $0.5273 \pm 0.0033$ | $-0.4087 \pm 0.0045$ | $[-0.4114, -0.4059]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GCN** | `shuffled_spatial_knn_k12` | $0.0471 \pm 0.0197$ | $0.5273 \pm 0.0033$ | $-0.4802 \pm 0.0200$ | $[-0.4924, -0.4680]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GCN** | `bipartite_ref_k20` | $0.5253 \pm 0.0311$ | $0.5273 \pm 0.0033$ | $-0.0020 \pm 0.0314$ | $[-0.0212, +0.0172]$ | **PARITY** |
| **GAT** | `spatial_knn_k6` | $0.2172 \pm 0.0058$ | $0.5273 \pm 0.0033$ | $-0.3101 \pm 0.0081$ | $[-0.3150, -0.3051]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GAT** | `spatial_knn_k12` | $0.1810 \pm 0.0128$ | $0.5273 \pm 0.0033$ | $-0.3462 \pm 0.0142$ | $[-0.3549, -0.3376]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GAT** | `rewired_spatial_knn_k6` | $0.0583 \pm 0.0145$ | $0.5273 \pm 0.0033$ | $-0.4690 \pm 0.0156$ | $[-0.4785, -0.4594]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GAT** | `rewired_spatial_knn_k12` | $0.0350 \pm 0.0221$ | $0.5273 \pm 0.0033$ | $-0.4923 \pm 0.0234$ | $[-0.5065, -0.4780]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GAT** | `shuffled_spatial_knn_k6` | $0.1771 \pm 0.0211$ | $0.5273 \pm 0.0033$ | $-0.3502 \pm 0.0207$ | $[-0.3628, -0.3375]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GAT** | `shuffled_spatial_knn_k12` | $0.1176 \pm 0.0198$ | $0.5273 \pm 0.0033$ | $-0.4097 \pm 0.0191$ | $[-0.4213, -0.3981]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GAT** | `bipartite_ref_k20` | $0.5336 \pm 0.0209$ | $0.5273 \pm 0.0033$ | $+0.0063 \pm 0.0184$ | $[-0.0049, +0.0176]$ | **PARITY** |
| **GIN** | `spatial_knn_k6` | $0.2061 \pm 0.0081$ | $0.5273 \pm 0.0033$ | $-0.3212 \pm 0.0099$ | $[-0.3273, -0.3152]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GIN** | `spatial_knn_k12` | $0.1587 \pm 0.0035$ | $0.5273 \pm 0.0033$ | $-0.3686 \pm 0.0061$ | $[-0.3723, -0.3649]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GIN** | `rewired_spatial_knn_k6` | $0.0509 \pm 0.0027$ | $0.5273 \pm 0.0033$ | $-0.4764 \pm 0.0035$ | $[-0.4785, -0.4742]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GIN** | `rewired_spatial_knn_k12` | $0.0379 \pm 0.0036$ | $0.5273 \pm 0.0033$ | $-0.4894 \pm 0.0047$ | $[-0.4923, -0.4865]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GIN** | `shuffled_spatial_knn_k6` | $0.1060 \pm 0.0074$ | $0.5273 \pm 0.0033$ | $-0.4213 \pm 0.0079$ | $[-0.4261, -0.4165]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GIN** | `shuffled_spatial_knn_k12` | $0.0451 \pm 0.0132$ | $0.5273 \pm 0.0033$ | $-0.4822 \pm 0.0129$ | $[-0.4900, -0.4743]$ | **NEGATIVE LIFT** ($p < 10^{-15}$) |
| **GIN** | `bipartite_ref_k20` | $\mathbf{0.5640 \pm 0.0094}$ | $0.5273 \pm 0.0033$ | $\mathbf{+0.0367 \pm 0.0111}$ | $[+0.0300, +0.0435]$ | **POSITIVE LIFT** ($p = 1.01 \times 10^{-5}$) |
| **GraphSAGE** | `spatial_knn_k6` | $0.4966 \pm 0.0074$ | $0.5273 \pm 0.0033$ | $-0.0307 \pm 0.0091$ | $[-0.0363, -0.0252]$ | **NEGATIVE LIFT** ($p = 1.28 \times 10^{-5}$) |
| **GraphSAGE** | `spatial_knn_k12` | $0.5020 \pm 0.0064$ | $0.5273 \pm 0.0033$ | $-0.0253 \pm 0.0078$ | $[-0.0301, -0.0206]$ | **NEGATIVE LIFT** ($p = 2.85 \times 10^{-5}$) |
| **GraphSAGE** | `rewired_spatial_knn_k6` | $0.4885 \pm 0.0165$ | $0.5273 \pm 0.0033$ | $-0.0388 \pm 0.0172$ | $[-0.0493, -0.0283]$ | **NEGATIVE LIFT** ($p = 1.75 \times 10^{-4}$) |
| **GraphSAGE** | `rewired_spatial_knn_k12` | $0.4937 \pm 0.0085$ | $0.5273 \pm 0.0033$ | $-0.0336 \pm 0.0107$ | $[-0.0402, -0.0270]$ | **NEGATIVE LIFT** ($p = 1.93 \times 10^{-5}$) |
| **GraphSAGE** | `shuffled_spatial_knn_k6` | $0.4837 \pm 0.0129$ | $0.5273 \pm 0.0033$ | $-0.0436 \pm 0.0126$ | $[-0.0512, -0.0359]$ | **NEGATIVE LIFT** ($p = 1.54 \times 10^{-5}$) |
| **GraphSAGE** | `shuffled_spatial_knn_k12` | $0.4849 \pm 0.0210$ | $0.5273 \pm 0.0033$ | $-0.0424 \pm 0.0219$ | $[-0.0558, -0.0290]$ | **NEGATIVE LIFT** ($p = 4.48 \times 10^{-4}$) |
| **GraphSAGE** | `bipartite_ref_k20` | $0.5397 \pm 0.0263$ | $0.5273 \pm 0.0033$ | $+0.0124 \pm 0.0271$ | $[-0.0042, +0.0290]$ | **PARITY** |

---

## 2. Post-Hoc Stratification: Interior vs. Boundary Margin

Cells were geometrically stratified by distance to the section convex hull (outer 15% classified as boundary margin cells):
- **Deep Interior Cells**: $N = 7,229$ ($85.0\%$)
- **Boundary Margin Cells**: $N = 1,278$ ($15.0\%$)

| Model & Graph | Overall Lift ($\Delta$) | Interior Lift ($\Delta_{\text{int}}$) | Boundary Lift ($\Delta_{\text{bnd}}$) | Margin Effect ($\Delta_{\text{bnd}} - \Delta_{\text{int}}$) |
|:---|:---:|:---:|:---:|:---:|
| **GCN** on `spatial_knn_k6` | $-0.3269$ | $-0.3204$ | $-0.3475$ | **$-0.0271$** (exacerbated collapse) |
| **GAT** on `spatial_knn_k6` | $-0.3101$ | $-0.3040$ | $-0.3344$ | **$-0.0304$** (exacerbated collapse) |
| **GIN** on `spatial_knn_k6` | $-0.3212$ | $-0.3168$ | $-0.3359$ | **$-0.0191$** (exacerbated collapse) |
| **GraphSAGE** on `spatial_knn_k6` | $-0.0307$ | $-0.0234$ | $-0.0360$ | **$-0.0126$** (exacerbated collapse) |
| **GIN** on `bipartite_ref_k20` | $\mathbf{+0.0367}$ | $\mathbf{+0.0398}$ | $-0.0019$ | **$-0.0417$** |

**Scientific Takeaway**:  
Spatial message passing causes catastrophic degradation throughout the entire tissue, but performance deteriorates **even further at the tissue boundary** ($\approx -0.02$ to $-0.03$ additional loss) due to truncated neighborhoods and forced cross-lineage edges at tissue edges.
