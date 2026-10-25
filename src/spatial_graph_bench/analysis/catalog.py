"""The complete frozen four-platform analysis grid."""

DATASETS = (
    ("stereoseq_axolotl_telencephalon", "developmental_three_stage", "Stereo-seq"),
    ("merfish_mouse_spinal_cord", "mouse_held_out_canonical", "MERFISH"),
    ("openst_human_lymph_node", "section_held_out_canonical", "Open-ST"),
    ("xenium_mouse_kidney", "replicate_held_out_canonical", "Xenium"),
)
MODELS = ("gcn", "gat", "gin", "graphsage")
GRAPHS = (
    "spatial_knn_k6",
    "spatial_knn_k12",
    "rewired_spatial_knn_k6",
    "rewired_spatial_knn_k12",
    "shuffled_spatial_knn_k6",
    "shuffled_spatial_knn_k12",
    "bipartite_ref_k20",
)
SEEDS = tuple(range(42, 52))
ANALYSIS_VERSION = "prediction-reanalysis-v2"
LIMITATION = (
    "Retrospective evaluation of historical fits. GNN training used pooled BatchNorm "
    "over train/validation/test features and stale training-mode validation predictions. "
    "These results do not establish strict-inductive benefit; corrected training is required. "
    "Legacy shuffled controls mix sections and rewired controls lose edge reciprocity. "
    "Seed uncertainty describes optimization on one fixed holdout, not biological replication."
)
