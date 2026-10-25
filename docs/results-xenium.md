# Xenium: historical prediction reanalysis

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

Dataset: `xenium_mouse_kidney`; split: `replicate_held_out_canonical`.
Training groups: ShamL; held-out test groups: ShamR.
Test cells/spots: 85,880; evaluated: 85,880; intersection classes: 20; class coverage: 100.00%.
Excluded test classes: none.
MLP test F1: 0.7635 ± 0.0120; RF: 0.7789; epsilon = max(0.005, 2 × sample SD) = 0.0241.

Boundary diagnostics use the outer 15% by distance to each section's convex hull; this is not an internal histological boundary annotation.

| Model | Topology | GNN F1 | Paired lift | 90% CI (unadjusted) | Holm decision | Interior lift | Boundary lift |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| GCN | spatial_knn_k6 | 0.4440 | -0.3196 | [-0.3254, -0.3138] | Negative lift | -0.3182 | -0.3051 |
| GCN | spatial_knn_k12 | 0.3711 | -0.3925 | [-0.4027, -0.3822] | Negative lift | -0.3917 | -0.3554 |
| GCN | rewired_spatial_knn_k6 | 0.0443 | -0.7192 | [-0.7258, -0.7127] | Negative lift | -0.7221 | -0.5748 |
| GCN | rewired_spatial_knn_k12 | 0.0212 | -0.7423 | [-0.7503, -0.7344] | Negative lift | -0.7453 | -0.5920 |
| GCN | shuffled_spatial_knn_k6 | 0.3530 | -0.4105 | [-0.4203, -0.4007] | Negative lift | -0.4131 | -0.3444 |
| GCN | shuffled_spatial_knn_k12 | 0.2159 | -0.5476 | [-0.5589, -0.5363] | Negative lift | -0.5521 | -0.4383 |
| GCN | bipartite_ref_k20 | 0.7306 | -0.0329 | [-0.0439, -0.0219] | Inconclusive | -0.0310 | -0.0453 |
| GAT | spatial_knn_k6 | 0.4871 | -0.2764 | [-0.2839, -0.2689] | Negative lift | -0.2762 | -0.2562 |
| GAT | spatial_knn_k12 | 0.4338 | -0.3297 | [-0.3376, -0.3218] | Negative lift | -0.3298 | -0.2999 |
| GAT | rewired_spatial_knn_k6 | 0.0786 | -0.6849 | [-0.7063, -0.6636] | Negative lift | -0.6875 | -0.5517 |
| GAT | rewired_spatial_knn_k12 | 0.0395 | -0.7240 | [-0.7319, -0.7162] | Negative lift | -0.7265 | -0.5805 |
| GAT | shuffled_spatial_knn_k6 | 0.4385 | -0.3251 | [-0.3432, -0.3069] | Negative lift | -0.3268 | -0.2764 |
| GAT | shuffled_spatial_knn_k12 | 0.3388 | -0.4247 | [-0.4471, -0.4023] | Negative lift | -0.4272 | -0.3481 |
| GAT | bipartite_ref_k20 | 0.7415 | -0.0220 | [-0.0351, -0.0088] | Inconclusive | -0.0211 | -0.0309 |
| GIN | spatial_knn_k6 | 0.4754 | -0.2881 | [-0.2964, -0.2799] | Negative lift | -0.2868 | -0.2777 |
| GIN | spatial_knn_k12 | 0.3846 | -0.3789 | [-0.4086, -0.3492] | Negative lift | -0.3782 | -0.3407 |
| GIN | rewired_spatial_knn_k6 | 0.0303 | -0.7332 | [-0.7422, -0.7242] | Negative lift | -0.7359 | -0.5875 |
| GIN | rewired_spatial_knn_k12 | 0.0136 | -0.7499 | [-0.7552, -0.7446] | Negative lift | -0.7531 | -0.5966 |
| GIN | shuffled_spatial_knn_k6 | 0.4281 | -0.3354 | [-0.3424, -0.3283] | Negative lift | -0.3373 | -0.2886 |
| GIN | shuffled_spatial_knn_k12 | 0.2795 | -0.4840 | [-0.4899, -0.4781] | Negative lift | -0.4866 | -0.3958 |
| GIN | bipartite_ref_k20 | 0.7560 | -0.0075 | [-0.0149, -0.0001] | Inconclusive | -0.0073 | -0.0100 |
| GRAPHSAGE | spatial_knn_k6 | 0.7493 | -0.0142 | [-0.0257, -0.0027] | Inconclusive | -0.0142 | -0.0265 |
| GRAPHSAGE | spatial_knn_k12 | 0.7514 | -0.0121 | [-0.0245, +0.0003] | Inconclusive | -0.0121 | -0.0253 |
| GRAPHSAGE | rewired_spatial_knn_k6 | 0.7402 | -0.0233 | [-0.0316, -0.0150] | Inconclusive | -0.0228 | -0.0256 |
| GRAPHSAGE | rewired_spatial_knn_k12 | 0.7414 | -0.0221 | [-0.0338, -0.0104] | Inconclusive | -0.0216 | -0.0240 |
| GRAPHSAGE | shuffled_spatial_knn_k6 | 0.7403 | -0.0232 | [-0.0323, -0.0140] | Inconclusive | -0.0228 | -0.0266 |
| GRAPHSAGE | shuffled_spatial_knn_k12 | 0.7387 | -0.0248 | [-0.0368, -0.0129] | Inconclusive | -0.0244 | -0.0277 |
| GRAPHSAGE | bipartite_ref_k20 | 0.7418 | -0.0217 | [-0.0352, -0.0082] | Inconclusive | -0.0208 | -0.0298 |
