# Historical prediction reanalysis (v2)

Retrospective evaluation of historical fits. GNN training used pooled BatchNorm over train/validation/test features and stale training-mode validation predictions. These results do not establish strict-inductive benefit; corrected training is required. Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. Seed uncertainty describes optimization on one fixed holdout, not biological replication.

All reported test scores use the train/test label intersection; unseen true cells are excluded uniformly.
Paired seeds 42–51; sample SD. Unadjusted 90% CIs are descriptive. Holm adjusted p-values cover all 336 hypotheses (112 configurations × superiority, inferiority, equivalence). This is a retrospective analysis amendment.

| Dataset | Model | Topology | GNN F1 | MLP F1 | Matched lift | 90% CI | Adjusted p (sup / inf / equiv) | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Stereo-seq | GCN | spatial_knn_k6 | 0.7043 | 0.7282 | -0.0239 | [-0.0970, +0.0492] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GCN | spatial_knn_k12 | 0.7198 | 0.7282 | -0.0084 | [-0.0235, +0.0067] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GCN | rewired_spatial_knn_k6 | 0.0542 | 0.7282 | -0.6739 | [-0.6889, -0.6590] | 1 / 6.23e-12 / 1 | Negative lift |
| Stereo-seq | GCN | rewired_spatial_knn_k12 | 0.0181 | 0.7282 | -0.7101 | [-0.7183, -0.7019] | 1 / 1.95e-14 / 1 | Negative lift |
| Stereo-seq | GCN | shuffled_spatial_knn_k6 | 0.2204 | 0.7282 | -0.5077 | [-0.5233, -0.4922] | 1 / 1.23e-10 / 1 | Negative lift |
| Stereo-seq | GCN | shuffled_spatial_knn_k12 | 0.1448 | 0.7282 | -0.5834 | [-0.6017, -0.5651] | 1 / 1.39e-10 / 1 | Negative lift |
| Stereo-seq | GCN | bipartite_ref_k20 | 0.7047 | 0.7282 | -0.0234 | [-0.0312, -0.0156] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GAT | spatial_knn_k6 | 0.7012 | 0.7282 | -0.0269 | [-0.0995, +0.0456] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GAT | spatial_knn_k12 | 0.7145 | 0.7282 | -0.0136 | [-0.0312, +0.0040] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GAT | rewired_spatial_knn_k6 | 0.0340 | 0.7282 | -0.6942 | [-0.7096, -0.6788] | 1 / 6.26e-12 / 1 | Negative lift |
| Stereo-seq | GAT | rewired_spatial_knn_k12 | 0.0187 | 0.7282 | -0.7095 | [-0.7189, -0.7001] | 1 / 6.21e-14 / 1 | Negative lift |
| Stereo-seq | GAT | shuffled_spatial_knn_k6 | 0.2167 | 0.7282 | -0.5114 | [-0.5286, -0.4942] | 1 / 2.77e-10 / 1 | Negative lift |
| Stereo-seq | GAT | shuffled_spatial_knn_k12 | 0.0555 | 0.7282 | -0.6726 | [-0.6887, -0.6565] | 1 / 1.24e-11 / 1 | Negative lift |
| Stereo-seq | GAT | bipartite_ref_k20 | 0.7080 | 0.7282 | -0.0202 | [-0.0309, -0.0094] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GIN | spatial_knn_k6 | 0.6679 | 0.7282 | -0.0602 | [-0.0955, -0.0250] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GIN | spatial_knn_k12 | 0.6590 | 0.7282 | -0.0692 | [-0.0954, -0.0429] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GIN | rewired_spatial_knn_k6 | 0.0481 | 0.7282 | -0.6800 | [-0.6935, -0.6666] | 1 / 2.25e-12 / 1 | Negative lift |
| Stereo-seq | GIN | rewired_spatial_knn_k12 | 0.0298 | 0.7282 | -0.6984 | [-0.7139, -0.6829] | 1 / 6.3e-12 / 1 | Negative lift |
| Stereo-seq | GIN | shuffled_spatial_knn_k6 | 0.1865 | 0.7282 | -0.5416 | [-0.5591, -0.5242] | 1 / 1.85e-10 / 1 | Negative lift |
| Stereo-seq | GIN | shuffled_spatial_knn_k12 | 0.1040 | 0.7282 | -0.6241 | [-0.6462, -0.6021] | 1 / 3.91e-10 / 1 | Negative lift |
| Stereo-seq | GIN | bipartite_ref_k20 | 0.7046 | 0.7282 | -0.0236 | [-0.0348, -0.0124] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GRAPHSAGE | spatial_knn_k6 | 0.7369 | 0.7282 | +0.0088 | [-0.0559, +0.0735] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GRAPHSAGE | spatial_knn_k12 | 0.7665 | 0.7282 | +0.0383 | [+0.0250, +0.0516] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GRAPHSAGE | rewired_spatial_knn_k6 | 0.7213 | 0.7282 | -0.0069 | [-0.0215, +0.0078] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GRAPHSAGE | rewired_spatial_knn_k12 | 0.7267 | 0.7282 | -0.0014 | [-0.0120, +0.0091] | 1 / 1 / 0.167 | Inconclusive |
| Stereo-seq | GRAPHSAGE | shuffled_spatial_knn_k6 | 0.7154 | 0.7282 | -0.0128 | [-0.0253, -0.0003] | 1 / 1 / 1 | Inconclusive |
| Stereo-seq | GRAPHSAGE | shuffled_spatial_knn_k12 | 0.7230 | 0.7282 | -0.0052 | [-0.0170, +0.0066] | 1 / 1 / 0.803 | Inconclusive |
| Stereo-seq | GRAPHSAGE | bipartite_ref_k20 | 0.7257 | 0.7282 | -0.0025 | [-0.0113, +0.0063] | 1 / 1 / 0.0637 | Inconclusive |
| MERFISH | GCN | spatial_knn_k6 | 0.2004 | 0.5273 | -0.3269 | [-0.3323, -0.3215] | 1 / 3.93e-13 / 1 | Negative lift |
| MERFISH | GCN | spatial_knn_k12 | 0.1577 | 0.5273 | -0.3696 | [-0.3748, -0.3644] | 1 / 9.23e-14 / 1 | Negative lift |
| MERFISH | GCN | rewired_spatial_knn_k6 | 0.0478 | 0.5273 | -0.4795 | [-0.4818, -0.4772] | 1 / 5e-18 / 1 | Negative lift |
| MERFISH | GCN | rewired_spatial_knn_k12 | 0.0345 | 0.5273 | -0.4928 | [-0.4991, -0.4865] | 1 / 3.5e-14 / 1 | Negative lift |
| MERFISH | GCN | shuffled_spatial_knn_k6 | 0.1186 | 0.5273 | -0.4087 | [-0.4114, -0.4059] | 1 / 1.22e-16 / 1 | Negative lift |
| MERFISH | GCN | shuffled_spatial_knn_k12 | 0.0471 | 0.5273 | -0.4802 | [-0.4924, -0.4680] | 1 / 1.65e-11 / 1 | Negative lift |
| MERFISH | GCN | bipartite_ref_k20 | 0.5253 | 0.5273 | -0.0020 | [-0.0212, +0.0172] | 1 / 1 / 1 | Inconclusive |
| MERFISH | GAT | spatial_knn_k6 | 0.2172 | 0.5273 | -0.3101 | [-0.3150, -0.3051] | 1 / 2.76e-13 / 1 | Negative lift |
| MERFISH | GAT | spatial_knn_k12 | 0.1810 | 0.5273 | -0.3462 | [-0.3549, -0.3376] | 1 / 1.53e-11 / 1 | Negative lift |
| MERFISH | GAT | rewired_spatial_knn_k6 | 0.0583 | 0.5273 | -0.4690 | [-0.4785, -0.4594] | 1 / 2.29e-12 / 1 | Negative lift |
| MERFISH | GAT | rewired_spatial_knn_k12 | 0.0350 | 0.5273 | -0.4923 | [-0.5065, -0.4780] | 1 / 5.2e-11 / 1 | Negative lift |
| MERFISH | GAT | shuffled_spatial_knn_k6 | 0.1771 | 0.5273 | -0.3502 | [-0.3628, -0.3375] | 1 / 3.83e-10 / 1 | Negative lift |
| MERFISH | GAT | shuffled_spatial_knn_k12 | 0.1176 | 0.5273 | -0.4097 | [-0.4213, -0.3981] | 1 / 4.48e-11 / 1 | Negative lift |
| MERFISH | GAT | bipartite_ref_k20 | 0.5336 | 0.5273 | +0.0063 | [-0.0049, +0.0176] | 1 / 1 / 1 | Inconclusive |
| MERFISH | GIN | spatial_knn_k6 | 0.2061 | 0.5273 | -0.3212 | [-0.3273, -0.3152] | 1 / 1.28e-12 / 1 | Negative lift |
| MERFISH | GIN | spatial_knn_k12 | 0.1587 | 0.5273 | -0.3686 | [-0.3723, -0.3649] | 1 / 4.44e-15 / 1 | Negative lift |
| MERFISH | GIN | rewired_spatial_knn_k6 | 0.0509 | 0.5273 | -0.4764 | [-0.4785, -0.4742] | 1 / 3.34e-18 / 1 | Negative lift |
| MERFISH | GIN | rewired_spatial_knn_k12 | 0.0379 | 0.5273 | -0.4894 | [-0.4923, -0.4865] | 1 / 3.64e-17 / 1 | Negative lift |
| MERFISH | GIN | shuffled_spatial_knn_k6 | 0.1060 | 0.5273 | -0.4213 | [-0.4261, -0.4165] | 1 / 1.36e-14 / 1 | Negative lift |
| MERFISH | GIN | shuffled_spatial_knn_k12 | 0.0451 | 0.5273 | -0.4822 | [-0.4900, -0.4743] | 1 / 3.23e-13 / 1 | Negative lift |
| MERFISH | GIN | bipartite_ref_k20 | 0.5640 | 0.5273 | +0.0367 | [+0.0300, +0.0435] | 0.00271 / 1 / 1 | Positive lift |
| MERFISH | GRAPHSAGE | spatial_knn_k6 | 0.4966 | 0.5273 | -0.0307 | [-0.0363, -0.0252] | 1 / 0.00341 / 1 | Negative lift |
| MERFISH | GRAPHSAGE | spatial_knn_k12 | 0.5020 | 0.5273 | -0.0253 | [-0.0301, -0.0206] | 1 / 0.00756 / 1 | Negative lift |
| MERFISH | GRAPHSAGE | rewired_spatial_knn_k6 | 0.4885 | 0.5273 | -0.0388 | [-0.0493, -0.0283] | 1 / 0.0461 / 1 | Negative lift |
| MERFISH | GRAPHSAGE | rewired_spatial_knn_k12 | 0.4937 | 0.5273 | -0.0336 | [-0.0402, -0.0270] | 1 / 0.00513 / 1 | Negative lift |
| MERFISH | GRAPHSAGE | shuffled_spatial_knn_k6 | 0.4837 | 0.5273 | -0.0436 | [-0.0512, -0.0359] | 1 / 0.00147 / 1 | Negative lift |
| MERFISH | GRAPHSAGE | shuffled_spatial_knn_k12 | 0.4849 | 0.5273 | -0.0424 | [-0.0558, -0.0290] | 1 / 0.117 / 1 | Inconclusive |
| MERFISH | GRAPHSAGE | bipartite_ref_k20 | 0.5397 | 0.5273 | +0.0124 | [-0.0042, +0.0290] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GCN | spatial_knn_k6 | 0.3428 | 0.3946 | -0.0517 | [-0.0576, -0.0458] | 1 / 2.17e-05 / 1 | Negative lift |
| Open-ST | GCN | spatial_knn_k12 | 0.3216 | 0.3946 | -0.0730 | [-0.0841, -0.0618] | 1 / 0.000196 / 1 | Negative lift |
| Open-ST | GCN | rewired_spatial_knn_k6 | 0.0270 | 0.3946 | -0.3675 | [-0.3729, -0.3622] | 1 / 1.15e-13 / 1 | Negative lift |
| Open-ST | GCN | rewired_spatial_knn_k12 | 0.0195 | 0.3946 | -0.3750 | [-0.3800, -0.3701] | 1 / 4.68e-14 / 1 | Negative lift |
| Open-ST | GCN | shuffled_spatial_knn_k6 | 0.1264 | 0.3946 | -0.2682 | [-0.2755, -0.2609] | 1 / 3.16e-11 / 1 | Negative lift |
| Open-ST | GCN | shuffled_spatial_knn_k12 | 0.0984 | 0.3946 | -0.2962 | [-0.3038, -0.2886] | 1 / 1.88e-11 / 1 | Negative lift |
| Open-ST | GCN | bipartite_ref_k20 | 0.3521 | 0.3946 | -0.0424 | [-0.0451, -0.0398] | 1 / 1.3e-07 / 1 | Negative lift |
| Open-ST | GAT | spatial_knn_k6 | 0.3400 | 0.3946 | -0.0546 | [-0.0595, -0.0496] | 1 / 2.99e-06 / 1 | Negative lift |
| Open-ST | GAT | spatial_knn_k12 | 0.3257 | 0.3946 | -0.0689 | [-0.0752, -0.0625] | 1 / 2.57e-06 / 1 | Negative lift |
| Open-ST | GAT | rewired_spatial_knn_k6 | 0.0545 | 0.3946 | -0.3400 | [-0.3558, -0.3242] | 1 / 3.51e-09 / 1 | Negative lift |
| Open-ST | GAT | rewired_spatial_knn_k12 | 0.0258 | 0.3946 | -0.3687 | [-0.3827, -0.3547] | 1 / 5.76e-10 / 1 | Negative lift |
| Open-ST | GAT | shuffled_spatial_knn_k6 | 0.1984 | 0.3946 | -0.1961 | [-0.2024, -0.1899] | 1 / 1.33e-10 / 1 | Negative lift |
| Open-ST | GAT | shuffled_spatial_knn_k12 | 0.1537 | 0.3946 | -0.2409 | [-0.2588, -0.2230] | 1 / 2.35e-07 / 1 | Negative lift |
| Open-ST | GAT | bipartite_ref_k20 | 0.3577 | 0.3946 | -0.0369 | [-0.0421, -0.0317] | 1 / 0.000202 / 1 | Negative lift |
| Open-ST | GIN | spatial_knn_k6 | 0.3506 | 0.3946 | -0.0439 | [-0.0467, -0.0411] | 1 / 1.6e-07 / 1 | Negative lift |
| Open-ST | GIN | spatial_knn_k12 | 0.3303 | 0.3946 | -0.0643 | [-0.0690, -0.0596] | 1 / 3.74e-07 / 1 | Negative lift |
| Open-ST | GIN | rewired_spatial_knn_k6 | 0.0195 | 0.3946 | -0.3750 | [-0.3859, -0.3641] | 1 / 5.44e-11 / 1 | Negative lift |
| Open-ST | GIN | rewired_spatial_knn_k12 | 0.0086 | 0.3946 | -0.3859 | [-0.3918, -0.3801] | 1 / 1.68e-13 / 1 | Negative lift |
| Open-ST | GIN | shuffled_spatial_knn_k6 | 0.1199 | 0.3946 | -0.2746 | [-0.2782, -0.2711] | 1 / 4.33e-14 / 1 | Negative lift |
| Open-ST | GIN | shuffled_spatial_knn_k12 | 0.0805 | 0.3946 | -0.3141 | [-0.3190, -0.3091] | 1 / 2.29e-13 / 1 | Negative lift |
| Open-ST | GIN | bipartite_ref_k20 | 0.3677 | 0.3946 | -0.0269 | [-0.0298, -0.0239] | 1 / 4.07e-05 / 1 | Negative lift |
| Open-ST | GRAPHSAGE | spatial_knn_k6 | 0.3937 | 0.3946 | -0.0009 | [-0.0047, +0.0028] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GRAPHSAGE | spatial_knn_k12 | 0.3962 | 0.3946 | +0.0016 | [-0.0037, +0.0070] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GRAPHSAGE | rewired_spatial_knn_k6 | 0.3966 | 0.3946 | +0.0020 | [-0.0017, +0.0057] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GRAPHSAGE | rewired_spatial_knn_k12 | 0.3927 | 0.3946 | -0.0019 | [-0.0073, +0.0036] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GRAPHSAGE | shuffled_spatial_knn_k6 | 0.3966 | 0.3946 | +0.0021 | [-0.0009, +0.0051] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GRAPHSAGE | shuffled_spatial_knn_k12 | 0.3961 | 0.3946 | +0.0016 | [-0.0018, +0.0050] | 1 / 1 / 1 | Inconclusive |
| Open-ST | GRAPHSAGE | bipartite_ref_k20 | 0.3804 | 0.3946 | -0.0141 | [-0.0181, -0.0101] | 1 / 0.381 / 1 | Inconclusive |
| Xenium | GCN | spatial_knn_k6 | 0.4440 | 0.7635 | -0.3196 | [-0.3254, -0.3138] | 1 / 1.49e-12 / 1 | Negative lift |
| Xenium | GCN | spatial_knn_k12 | 0.3711 | 0.7635 | -0.3925 | [-0.4027, -0.3822] | 1 / 3.16e-11 / 1 | Negative lift |
| Xenium | GCN | rewired_spatial_knn_k6 | 0.0443 | 0.7635 | -0.7192 | [-0.7258, -0.7127] | 1 / 2.08e-15 / 1 | Negative lift |
| Xenium | GCN | rewired_spatial_knn_k12 | 0.0212 | 0.7635 | -0.7423 | [-0.7503, -0.7344] | 1 / 8.91e-15 / 1 | Negative lift |
| Xenium | GCN | shuffled_spatial_knn_k6 | 0.3530 | 0.7635 | -0.4105 | [-0.4203, -0.4007] | 1 / 1.44e-11 / 1 | Negative lift |
| Xenium | GCN | shuffled_spatial_knn_k12 | 0.2159 | 0.7635 | -0.5476 | [-0.5589, -0.5363] | 1 / 3.37e-12 / 1 | Negative lift |
| Xenium | GCN | bipartite_ref_k20 | 0.7306 | 0.7635 | -0.0329 | [-0.0439, -0.0219] | 1 / 1 / 1 | Inconclusive |
| Xenium | GAT | spatial_knn_k6 | 0.4871 | 0.7635 | -0.2764 | [-0.2839, -0.2689] | 1 / 5.6e-11 / 1 | Negative lift |
| Xenium | GAT | spatial_knn_k12 | 0.4338 | 0.7635 | -0.3297 | [-0.3376, -0.3218] | 1 / 1.65e-11 / 1 | Negative lift |
| Xenium | GAT | rewired_spatial_knn_k6 | 0.0786 | 0.7635 | -0.6849 | [-0.7063, -0.6636] | 1 / 1.17e-10 / 1 | Negative lift |
| Xenium | GAT | rewired_spatial_knn_k12 | 0.0395 | 0.7635 | -0.7240 | [-0.7319, -0.7162] | 1 / 1.02e-14 / 1 | Negative lift |
| Xenium | GAT | shuffled_spatial_knn_k6 | 0.4385 | 0.7635 | -0.3251 | [-0.3432, -0.3069] | 1 / 3.06e-08 / 1 | Negative lift |
| Xenium | GAT | shuffled_spatial_knn_k12 | 0.3388 | 0.7635 | -0.4247 | [-0.4471, -0.4023] | 1 / 1.61e-08 / 1 | Negative lift |
| Xenium | GAT | bipartite_ref_k20 | 0.7415 | 0.7635 | -0.0220 | [-0.0351, -0.0088] | 1 / 1 / 1 | Inconclusive |
| Xenium | GIN | spatial_knn_k6 | 0.4754 | 0.7635 | -0.2881 | [-0.2964, -0.2799] | 1 / 8.65e-11 / 1 | Negative lift |
| Xenium | GIN | spatial_knn_k12 | 0.3846 | 0.7635 | -0.3789 | [-0.4086, -0.3492] | 1 / 5.62e-07 / 1 | Negative lift |
| Xenium | GIN | rewired_spatial_knn_k6 | 0.0303 | 0.7635 | -0.7332 | [-0.7422, -0.7242] | 1 / 2.99e-14 / 1 | Negative lift |
| Xenium | GIN | rewired_spatial_knn_k12 | 0.0136 | 0.7635 | -0.7499 | [-0.7552, -0.7446] | 1 / 2.26e-16 / 1 | Negative lift |
| Xenium | GIN | shuffled_spatial_knn_k6 | 0.4281 | 0.7635 | -0.3354 | [-0.3424, -0.3283] | 1 / 5.24e-12 / 1 | Negative lift |
| Xenium | GIN | shuffled_spatial_knn_k12 | 0.2795 | 0.7635 | -0.4840 | [-0.4899, -0.4781] | 1 / 3.18e-14 / 1 | Negative lift |
| Xenium | GIN | bipartite_ref_k20 | 0.7560 | 0.7635 | -0.0075 | [-0.0149, -0.0001] | 1 / 1 / 0.347 | Inconclusive |
| Xenium | GRAPHSAGE | spatial_knn_k6 | 0.7493 | 0.7635 | -0.0142 | [-0.0257, -0.0027] | 1 / 1 / 1 | Inconclusive |
| Xenium | GRAPHSAGE | spatial_knn_k12 | 0.7514 | 0.7635 | -0.0121 | [-0.0245, +0.0003] | 1 / 1 / 1 | Inconclusive |
| Xenium | GRAPHSAGE | rewired_spatial_knn_k6 | 0.7402 | 0.7635 | -0.0233 | [-0.0316, -0.0150] | 1 / 1 / 1 | Inconclusive |
| Xenium | GRAPHSAGE | rewired_spatial_knn_k12 | 0.7414 | 0.7635 | -0.0221 | [-0.0338, -0.0104] | 1 / 1 / 1 | Inconclusive |
| Xenium | GRAPHSAGE | shuffled_spatial_knn_k6 | 0.7403 | 0.7635 | -0.0232 | [-0.0323, -0.0140] | 1 / 1 / 1 | Inconclusive |
| Xenium | GRAPHSAGE | shuffled_spatial_knn_k12 | 0.7387 | 0.7635 | -0.0248 | [-0.0368, -0.0129] | 1 / 1 / 1 | Inconclusive |
| Xenium | GRAPHSAGE | bipartite_ref_k20 | 0.7418 | 0.7635 | -0.0217 | [-0.0352, -0.0082] | 1 / 1 / 1 | Inconclusive |
