#!/usr/bin/env bash
# Reconstruct and reanalyze immutable historical artifacts. Never trains models.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE=0
FIGURES=1
for arg in "$@"; do
    case "$arg" in
        --offline) OFFLINE=1 ;;
        --no-figures) FIGURES=0 ;;
        *) echo "Usage: $0 [--offline] [--no-figures]" >&2; exit 2 ;;
    esac
done
if [ "$OFFLINE" -eq 1 ]; then
    uv sync --frozen --all-extras --offline
    uv run --frozen python scripts/prepare_reanalysis_artifacts.py --offline
else
    uv sync --frozen --all-extras
    uv run --frozen python scripts/prepare_reanalysis_artifacts.py
fi
uv run --frozen python scripts/reanalyze_results.py
if [ "$FIGURES" -eq 1 ]; then
    export MPLBACKEND=Agg
    uv run --frozen python scripts/plotting/plot_cross_modality_benchmark.py
    plot_dataset() {
        uv run --frozen python scripts/plotting/plot_topological_controls.py --dataset "$1" --split "$2" --suffix "$3"
        uv run --frozen python scripts/plotting/plot_boundary_stratification.py --dataset "$1" --split "$2" --graph spatial_knn_k12 --title "$4" --suffix "$3"
    }
    plot_dataset stereoseq_axolotl_telencephalon developmental_three_stage "" "Stereo-seq Stage57 (historical fits)"
    plot_dataset merfish_mouse_spinal_cord mouse_held_out_canonical _merfish "MERFISH mouse F4 (historical fits)"
    plot_dataset openst_human_lymph_node section_held_out_canonical _openst "Open-ST section 19 (historical fits)"
    plot_dataset xenium_mouse_kidney replicate_held_out_canonical _xenium "Xenium ShamR (historical fits)"
    uv run --frozen python scripts/plotting/plot_kidney_interstitial_oversmoothing.py
    uv run --frozen python scripts/plotting/plot_tumor_stroma_over_smoothing.py
fi
echo "Historical prediction reanalysis complete: results/reports/reanalysis_v2/"
echo "GNN training validity issues remain; see docs/reanalysis-v2.md."
