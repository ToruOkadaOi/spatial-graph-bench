# Stereo-seq: historical prediction reanalysis

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

Dataset: `stereoseq_axolotl_telencephalon`; split: `developmental_three_stage`.
Training groups: Stage44, Stage54; held-out test groups: Stage57.
Test cells/spots: 4,410; evaluated: 4,209; intersection classes: 15; class coverage: 88.24%.
Excluded test classes: MSN, nptxEX.
MLP test F1: 0.7282 ± 0.0140; RF: 0.6763; epsilon = max(0.005, 2 × sample SD) = 0.0280.

Boundary diagnostics use the outer 15% by distance to each section's convex hull; this is not an internal histological boundary annotation.

| Model | Topology | GNN F1 | Paired lift | 90% CI (unadjusted) | Holm decision | Interior lift | Boundary lift |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| GCN | spatial_knn_k6 | 0.7043 | -0.0239 | [-0.0970, +0.0492] | Inconclusive | -0.0319 | -0.0374 |
| GCN | spatial_knn_k12 | 0.7198 | -0.0084 | [-0.0235, +0.0067] | Inconclusive | -0.0154 | -0.0492 |
| GCN | rewired_spatial_knn_k6 | 0.0542 | -0.6739 | [-0.6889, -0.6590] | Negative lift | -0.6544 | -0.3087 |
| GCN | rewired_spatial_knn_k12 | 0.0181 | -0.7101 | [-0.7183, -0.7019] | Negative lift | -0.6899 | -0.3206 |
| GCN | shuffled_spatial_knn_k6 | 0.2204 | -0.5077 | [-0.5233, -0.4922] | Negative lift | -0.4930 | -0.2384 |
| GCN | shuffled_spatial_knn_k12 | 0.1448 | -0.5834 | [-0.6017, -0.5651] | Negative lift | -0.5645 | -0.2715 |
| GCN | bipartite_ref_k20 | 0.7047 | -0.0234 | [-0.0312, -0.0156] | Inconclusive | -0.0232 | +0.0068 |
| GAT | spatial_knn_k6 | 0.7012 | -0.0269 | [-0.0995, +0.0456] | Inconclusive | -0.0277 | -0.0442 |
| GAT | spatial_knn_k12 | 0.7145 | -0.0136 | [-0.0312, +0.0040] | Inconclusive | -0.0125 | -0.0453 |
| GAT | rewired_spatial_knn_k6 | 0.0340 | -0.6942 | [-0.7096, -0.6788] | Negative lift | -0.6759 | -0.3135 |
| GAT | rewired_spatial_knn_k12 | 0.0187 | -0.7095 | [-0.7189, -0.7001] | Negative lift | -0.6882 | -0.3276 |
| GAT | shuffled_spatial_knn_k6 | 0.2167 | -0.5114 | [-0.5286, -0.4942] | Negative lift | -0.5023 | -0.2274 |
| GAT | shuffled_spatial_knn_k12 | 0.0555 | -0.6726 | [-0.6887, -0.6565] | Negative lift | -0.6561 | -0.3017 |
| GAT | bipartite_ref_k20 | 0.7080 | -0.0202 | [-0.0309, -0.0094] | Inconclusive | -0.0190 | -0.0034 |
| GIN | spatial_knn_k6 | 0.6679 | -0.0602 | [-0.0955, -0.0250] | Inconclusive | -0.0654 | -0.0482 |
| GIN | spatial_knn_k12 | 0.6590 | -0.0692 | [-0.0954, -0.0429] | Inconclusive | -0.0773 | -0.0587 |
| GIN | rewired_spatial_knn_k6 | 0.0481 | -0.6800 | [-0.6935, -0.6666] | Negative lift | -0.6621 | -0.3079 |
| GIN | rewired_spatial_knn_k12 | 0.0298 | -0.6984 | [-0.7139, -0.6829] | Negative lift | -0.6786 | -0.3213 |
| GIN | shuffled_spatial_knn_k6 | 0.1865 | -0.5416 | [-0.5591, -0.5242] | Negative lift | -0.5299 | -0.2448 |
| GIN | shuffled_spatial_knn_k12 | 0.1040 | -0.6241 | [-0.6462, -0.6021] | Negative lift | -0.6089 | -0.2785 |
| GIN | bipartite_ref_k20 | 0.7046 | -0.0236 | [-0.0348, -0.0124] | Inconclusive | -0.0223 | -0.0025 |
| GRAPHSAGE | spatial_knn_k6 | 0.7369 | +0.0088 | [-0.0559, +0.0735] | Inconclusive | +0.0120 | -0.0410 |
| GRAPHSAGE | spatial_knn_k12 | 0.7665 | +0.0383 | [+0.0250, +0.0516] | Inconclusive | +0.0414 | -0.0233 |
| GRAPHSAGE | rewired_spatial_knn_k6 | 0.7213 | -0.0069 | [-0.0215, +0.0078] | Inconclusive | +0.0006 | -0.0058 |
| GRAPHSAGE | rewired_spatial_knn_k12 | 0.7267 | -0.0014 | [-0.0120, +0.0091] | Inconclusive | +0.0065 | -0.0183 |
| GRAPHSAGE | shuffled_spatial_knn_k6 | 0.7154 | -0.0128 | [-0.0253, -0.0003] | Inconclusive | -0.0057 | -0.0155 |
| GRAPHSAGE | shuffled_spatial_knn_k12 | 0.7230 | -0.0052 | [-0.0170, +0.0066] | Inconclusive | +0.0021 | -0.0168 |
| GRAPHSAGE | bipartite_ref_k20 | 0.7257 | -0.0025 | [-0.0113, +0.0063] | Inconclusive | +0.0015 | -0.0027 |
