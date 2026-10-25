#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRAIN_WORK="${RETRAIN_ROOT:-experiments/inductive-v3}"
RETRAIN_INPUT="${RETRAIN_INPUTS:-${RETRAIN_WORK}/inputs}"
MODE="${1:-}"
case "$MODE" in
    smoke)
        uv run --frozen python scripts/retrain.py check-gpu
        exec uv run --frozen python -u scripts/retrain.py run --inputs "$RETRAIN_INPUT" --results "${RETRAIN_WORK}/cuda-smoke" --pilot --datasets xenium_mouse_kidney --models gat --graphs spatial_knn_k12 --seeds 42 --epochs 2
        ;;
    pilot)
        uv run --frozen python scripts/retrain.py check-gpu
        exec uv run --frozen python -u scripts/retrain.py run --inputs "$RETRAIN_INPUT" --results "${RETRAIN_WORK}/pilot" --pilot
        ;;
    full)
        uv run --frozen python scripts/retrain.py check-gpu
        exec uv run --frozen python -u scripts/retrain.py run --inputs "$RETRAIN_INPUT" --results "${RETRAIN_WORK}/full"
        ;;
    *)
        echo "Usage: $0 smoke|pilot|full" >&2
        exit 2
        ;;
esac
