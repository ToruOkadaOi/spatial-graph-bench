# MERFISH: historical prediction reanalysis

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

Dataset: `merfish_mouse_spinal_cord`; split: `mouse_held_out_canonical`.
Training groups: F3, M4, F5; held-out test groups: F4.
Test cells/spots: 8,507; evaluated: 8,507; intersection classes: 60; class coverage: 100.00%.
Excluded test classes: none.
MLP test F1: 0.5273 ± 0.0035; RF: 0.4683; epsilon = max(0.005, 2 × sample SD) = 0.0069.

Boundary diagnostics use the outer 15% by distance to each section's convex hull; this is not an internal histological boundary annotation.

| Model | Topology | GNN F1 | Paired lift | 90% CI (unadjusted) | Holm decision | Interior lift | Boundary lift |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| GCN | spatial_knn_k6 | 0.2004 | -0.3269 | [-0.3323, -0.3215] | Negative lift | -0.3163 | -0.2937 |
| GCN | spatial_knn_k12 | 0.1577 | -0.3696 | [-0.3748, -0.3644] | Negative lift | -0.3621 | -0.3274 |
| GCN | rewired_spatial_knn_k6 | 0.0478 | -0.4795 | [-0.4818, -0.4772] | Negative lift | -0.4731 | -0.3961 |
| GCN | rewired_spatial_knn_k12 | 0.0345 | -0.4928 | [-0.4991, -0.4865] | Negative lift | -0.4844 | -0.4123 |
| GCN | shuffled_spatial_knn_k6 | 0.1186 | -0.4087 | [-0.4114, -0.4059] | Negative lift | -0.3948 | -0.3575 |
| GCN | shuffled_spatial_knn_k12 | 0.0471 | -0.4802 | [-0.4924, -0.4680] | Negative lift | -0.4685 | -0.4000 |
| GCN | bipartite_ref_k20 | 0.5253 | -0.0020 | [-0.0212, +0.0172] | Inconclusive | -0.0018 | -0.0398 |
| GAT | spatial_knn_k6 | 0.2172 | -0.3101 | [-0.3150, -0.3051] | Negative lift | -0.2980 | -0.2868 |
| GAT | spatial_knn_k12 | 0.1810 | -0.3462 | [-0.3549, -0.3376] | Negative lift | -0.3355 | -0.3134 |
| GAT | rewired_spatial_knn_k6 | 0.0583 | -0.4690 | [-0.4785, -0.4594] | Negative lift | -0.4633 | -0.3854 |
| GAT | rewired_spatial_knn_k12 | 0.0350 | -0.4923 | [-0.5065, -0.4780] | Negative lift | -0.4825 | -0.4083 |
| GAT | shuffled_spatial_knn_k6 | 0.1771 | -0.3502 | [-0.3628, -0.3375] | Negative lift | -0.3467 | -0.2976 |
| GAT | shuffled_spatial_knn_k12 | 0.1176 | -0.4097 | [-0.4213, -0.3981] | Negative lift | -0.4011 | -0.3460 |
| GAT | bipartite_ref_k20 | 0.5336 | +0.0063 | [-0.0049, +0.0176] | Inconclusive | +0.0094 | -0.0296 |
| GIN | spatial_knn_k6 | 0.2061 | -0.3212 | [-0.3273, -0.3152] | Negative lift | -0.3111 | -0.2876 |
| GIN | spatial_knn_k12 | 0.1587 | -0.3686 | [-0.3723, -0.3649] | Negative lift | -0.3597 | -0.3293 |
| GIN | rewired_spatial_knn_k6 | 0.0509 | -0.4764 | [-0.4785, -0.4742] | Negative lift | -0.4699 | -0.3933 |
| GIN | rewired_spatial_knn_k12 | 0.0379 | -0.4894 | [-0.4923, -0.4865] | Negative lift | -0.4814 | -0.4081 |
| GIN | shuffled_spatial_knn_k6 | 0.1060 | -0.4213 | [-0.4261, -0.4165] | Negative lift | -0.4098 | -0.3577 |
| GIN | shuffled_spatial_knn_k12 | 0.0451 | -0.4822 | [-0.4900, -0.4743] | Negative lift | -0.4699 | -0.3993 |
| GIN | bipartite_ref_k20 | 0.5640 | +0.0367 | [+0.0300, +0.0435] | Positive lift | +0.0408 | -0.0115 |
| GRAPHSAGE | spatial_knn_k6 | 0.4966 | -0.0307 | [-0.0363, -0.0252] | Negative lift | -0.0234 | -0.0360 |
| GRAPHSAGE | spatial_knn_k12 | 0.5020 | -0.0253 | [-0.0301, -0.0206] | Negative lift | -0.0184 | -0.0313 |
| GRAPHSAGE | rewired_spatial_knn_k6 | 0.4885 | -0.0388 | [-0.0493, -0.0283] | Negative lift | -0.0341 | -0.0468 |
| GRAPHSAGE | rewired_spatial_knn_k12 | 0.4937 | -0.0336 | [-0.0402, -0.0270] | Negative lift | -0.0294 | -0.0331 |
| GRAPHSAGE | shuffled_spatial_knn_k6 | 0.4837 | -0.0436 | [-0.0512, -0.0359] | Negative lift | -0.0404 | -0.0388 |
| GRAPHSAGE | shuffled_spatial_knn_k12 | 0.4849 | -0.0424 | [-0.0558, -0.0290] | Inconclusive | -0.0393 | -0.0378 |
| GRAPHSAGE | bipartite_ref_k20 | 0.5397 | +0.0124 | [-0.0042, +0.0290] | Inconclusive | +0.0162 | -0.0184 |
