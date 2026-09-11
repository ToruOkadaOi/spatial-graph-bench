#!/usr/bin/env bash
# ==============================================================================
# Turnkey Reproduction Script for spatial-graph-bench
# Verifies environment, tests, checksums, and regenerates all figures and tables.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "========================================================================"
echo " spatial-graph-bench: One-Click Benchmark Reproduction Pipeline"
echo "========================================================================"

# 1. Environment & Invariant Tests
echo ">>> [1/5] Verifying environment dependencies..."
uv sync --frozen --all-extras

echo ">>> [2/5] Running pytest invariant test suite..."
PYTHONPATH=src uv run pytest tests -q

echo ">>> [3/5] Verifying cryptographic SHA-256 checksums of release assets..."
cd dist
for cs in sha256sums_*_results.txt; do
    if [ -f "${cs}" ]; then
        if command -v shasum &> /dev/null; then
            shasum -a 256 -c "${cs}"
        else
            sha256sum -c "${cs}"
        fi
    fi
done
cd "${REPO_ROOT}"

# 4. Generate LaTeX Synthesis Table
echo ">>> [4/5] Regenerating Quad-Modality Synthesis LaTeX Table..."
PYTHONPATH=src uv run python scripts/export_latex_tables.py

# 5. Regenerate Publication Figures
echo ">>> [5/5] Regenerating publication-ready figures..."
PYTHONPATH=src uv run python scripts/plotting/plot_cross_modality_benchmark.py
PYTHONPATH=src uv run python scripts/plotting/plot_topological_controls.py --dataset xenium_mouse_kidney --split replicate_held_out_canonical --suffix _xenium
PYTHONPATH=src uv run python scripts/plotting/plot_boundary_stratification.py --dataset xenium_mouse_kidney --split replicate_held_out_canonical --graph spatial_knn_k12 --title "10x Xenium (Mouse Kidney Replicate Split)" --suffix _xenium
PYTHONPATH=src uv run python scripts/plotting/plot_kidney_interstitial_oversmoothing.py
PYTHONPATH=src uv run python scripts/plotting/plot_tumor_stroma_over_smoothing.py

echo "========================================================================"
echo " SUCCESS: All benchmark deliverables, figures, and tables regenerated!"
echo " Results Directory: results/reports/"
echo " Figures Directory: results/figures/"
echo "========================================================================"
