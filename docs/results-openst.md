# Open-ST: historical prediction reanalysis

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

Dataset: `openst_human_lymph_node`; split: `section_held_out_canonical`.
Training groups: 6; held-out test groups: 19.
Test cells/spots: 34,072; evaluated: 34,072; intersection classes: 14; class coverage: 100.00%.
Excluded test classes: none.
MLP test F1: 0.3946 ± 0.0027; RF: 0.3365; epsilon = max(0.005, 2 × sample SD) = 0.0054.

Boundary diagnostics use the outer 15% by distance to each section's convex hull; this is not an internal histological boundary annotation.

| Model | Topology | GNN F1 | Paired lift | 90% CI (unadjusted) | Holm decision | Interior lift | Boundary lift |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| GCN | spatial_knn_k6 | 0.3428 | -0.0517 | [-0.0576, -0.0458] | Negative lift | -0.0507 | -0.1147 |
| GCN | spatial_knn_k12 | 0.3216 | -0.0730 | [-0.0841, -0.0618] | Negative lift | -0.0722 | -0.1299 |
| GCN | rewired_spatial_knn_k6 | 0.0270 | -0.3675 | [-0.3729, -0.3622] | Negative lift | -0.3769 | -0.3161 |
| GCN | rewired_spatial_knn_k12 | 0.0195 | -0.3750 | [-0.3800, -0.3701] | Negative lift | -0.3845 | -0.3226 |
| GCN | shuffled_spatial_knn_k6 | 0.1264 | -0.2682 | [-0.2755, -0.2609] | Negative lift | -0.2744 | -0.2375 |
| GCN | shuffled_spatial_knn_k12 | 0.0984 | -0.2962 | [-0.3038, -0.2886] | Negative lift | -0.3038 | -0.2572 |
| GCN | bipartite_ref_k20 | 0.3521 | -0.0424 | [-0.0451, -0.0398] | Negative lift | -0.0421 | -0.0867 |
| GAT | spatial_knn_k6 | 0.3400 | -0.0546 | [-0.0595, -0.0496] | Negative lift | -0.0536 | -0.1140 |
| GAT | spatial_knn_k12 | 0.3257 | -0.0689 | [-0.0752, -0.0625] | Negative lift | -0.0684 | -0.1243 |
| GAT | rewired_spatial_knn_k6 | 0.0545 | -0.3400 | [-0.3558, -0.3242] | Negative lift | -0.3489 | -0.2925 |
| GAT | rewired_spatial_knn_k12 | 0.0258 | -0.3687 | [-0.3827, -0.3547] | Negative lift | -0.3781 | -0.3171 |
| GAT | shuffled_spatial_knn_k6 | 0.1984 | -0.1961 | [-0.2024, -0.1899] | Negative lift | -0.2004 | -0.1932 |
| GAT | shuffled_spatial_knn_k12 | 0.1537 | -0.2409 | [-0.2588, -0.2230] | Negative lift | -0.2460 | -0.2249 |
| GAT | bipartite_ref_k20 | 0.3577 | -0.0369 | [-0.0421, -0.0317] | Negative lift | -0.0367 | -0.0829 |
| GIN | spatial_knn_k6 | 0.3506 | -0.0439 | [-0.0467, -0.0411] | Negative lift | -0.0425 | -0.1128 |
| GIN | spatial_knn_k12 | 0.3303 | -0.0643 | [-0.0690, -0.0596] | Negative lift | -0.0627 | -0.1269 |
| GIN | rewired_spatial_knn_k6 | 0.0195 | -0.3750 | [-0.3859, -0.3641] | Negative lift | -0.3843 | -0.3230 |
| GIN | rewired_spatial_knn_k12 | 0.0086 | -0.3859 | [-0.3918, -0.3801] | Negative lift | -0.3957 | -0.3318 |
| GIN | shuffled_spatial_knn_k6 | 0.1199 | -0.2746 | [-0.2782, -0.2711] | Negative lift | -0.2809 | -0.2458 |
| GIN | shuffled_spatial_knn_k12 | 0.0805 | -0.3141 | [-0.3190, -0.3091] | Negative lift | -0.3224 | -0.2710 |
| GIN | bipartite_ref_k20 | 0.3677 | -0.0269 | [-0.0298, -0.0239] | Negative lift | -0.0263 | -0.0739 |
| GRAPHSAGE | spatial_knn_k6 | 0.3937 | -0.0009 | [-0.0047, +0.0028] | Inconclusive | +0.0008 | -0.0701 |
| GRAPHSAGE | spatial_knn_k12 | 0.3962 | +0.0016 | [-0.0037, +0.0070] | Inconclusive | +0.0035 | -0.0674 |
| GRAPHSAGE | rewired_spatial_knn_k6 | 0.3966 | +0.0020 | [-0.0017, +0.0057] | Inconclusive | +0.0026 | -0.0318 |
| GRAPHSAGE | rewired_spatial_knn_k12 | 0.3927 | -0.0019 | [-0.0073, +0.0036] | Inconclusive | -0.0011 | -0.0353 |
| GRAPHSAGE | shuffled_spatial_knn_k6 | 0.3966 | +0.0021 | [-0.0009, +0.0051] | Inconclusive | +0.0022 | -0.0161 |
| GRAPHSAGE | shuffled_spatial_knn_k12 | 0.3961 | +0.0016 | [-0.0018, +0.0050] | Inconclusive | +0.0023 | -0.0308 |
| GRAPHSAGE | bipartite_ref_k20 | 0.3804 | -0.0141 | [-0.0181, -0.0101] | Inconclusive | -0.0134 | -0.0562 |
