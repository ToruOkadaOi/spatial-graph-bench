#!/usr/bin/env bash
# ==============================================================================
# Turnkey Execution Script: 280-Run Stereo-seq Developmental 3-Stage Sweep
# Dataset: stereoseq_axolotl_telencephalon (Scope A: Stage 44+54 -> Stage 57)
# Architectures: GCN, GraphSAGE, GAT, GIN (4 models × 7 graphs × 10 seeds = 280 runs)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="stereoseq_axolotl_telencephalon"
SPLIT="developmental_three_stage"
INPUT_TAG="v0.1.0-stereoseq-inputs"
RESULTS_TAG="v0.2.0-stereoseq-results"
CONFIG_FILE="configs/gpu_batch_stereoseq_axolotl_three_stage.yaml"
OUTPUT_ARCHIVE="dist/gpu_results_${DATASET}_${SPLIT}.tar.gz"
CHECKSUM_FILE="dist/sha256sums_${DATASET}_${SPLIT}_results.txt"

echo "========================================================================"
echo " Starting Stereo-seq Axolotl Neurogenesis Benchmark Sweep (280 runs)"
echo " Dataset: ${DATASET}"
echo " Split:   ${SPLIT}"
echo " Config:  ${CONFIG_FILE}"
echo "========================================================================"

# 1. Environment & Hardware Verification
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' package manager not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo ">>> Verifying environment dependencies..."
uv sync

echo ">>> Checking CUDA GPU acceleration..."
uv run python -c "
import torch
cuda_ok = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_ok else 'CPU only'
print(f'  CUDA Available: {cuda_ok} ({device_name})')
if not cuda_ok:
    print('  [WARNING] CUDA is not available. Execution will run on CPU and take significantly longer.')
"

# 2. Fetch Frozen Input Bundle from GitHub Releases or Local Dist
if [ ! -d "artifacts/preprocessed/${DATASET}/${SPLIT}" ] || [ ! -d "artifacts/graphs/${DATASET}/${SPLIT}" ]; then
    LOCAL_TAR="dist/artifacts_${DATASET}_${SPLIT}_inputs.tar.gz"
    if [ -f "${LOCAL_TAR}" ]; then
        echo ">>> Found local input bundle (${LOCAL_TAR}). Unpacking..."
        tar -xzf "${LOCAL_TAR}"
    elif command -v gh &> /dev/null; then
        echo ">>> Fetching certified input bundle (${INPUT_TAG}) via GitHub CLI..."
        PYTHONPATH=src:. uv run python scripts/manage_release_artifacts.py fetch \
            --tag "${INPUT_TAG}" \
            --dataset "${DATASET}" \
            --split "${SPLIT}"
    else
        echo "========================================================================"
        echo " ERROR: Input artifacts missing under artifacts/ and 'gh' CLI is not found."
        echo ""
        echo " Because this repository is private, you need the input bundle (3.14 MB)."
        echo " Choose ONE of the following options:"
        echo ""
        echo " Option 1 (SCP from local machine - fastest, no GitHub login needed):"
        echo "   scp dist/artifacts_${DATASET}_${SPLIT}_inputs.tar.gz user@this-gpu:$(pwd)/dist/"
        echo "   Then re-run this script."
        echo ""
        echo " Option 2 (Authenticate GitHub CLI on this machine):"
        echo "   gh auth login"
        echo "   Then re-run this script."
        echo "========================================================================"
        exit 1
    fi
else
    echo ">>> Input artifacts found locally."
fi

# 3. Cryptographic Invariant Validation
echo ">>> Running invariant validation on input artifacts..."
PYTHONPATH=src uv run python scripts/validate_artifacts.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}"

# 4. Execute Full 280-Run GPU Sweep
echo ">>> Launching 280-run GNN sweep..."
START_TIME=$(date +%s)
PYTHONPATH=src uv run python scripts/run_gnn_sweep.py \
    --batch-config "${CONFIG_FILE}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ">>> Sweep completed in ${ELAPSED} seconds (~$((ELAPSED / 60)) minutes)."

# 5. Cryptographic Packaging & Audit Verification
echo ">>> Auditing and packaging results delivery bundle..."
mkdir -p dist
PYTHONPATH=src uv run python scripts/package_gpu_results.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output "${OUTPUT_ARCHIVE}"

# 6. Generate Checksum File
ARCHIVE_FILENAME=$(basename "${OUTPUT_ARCHIVE}")
cd dist
if command -v shasum &> /dev/null; then
    shasum -a 256 "${ARCHIVE_FILENAME}" > "$(basename "${CHECKSUM_FILE}")"
else
    sha256sum "${ARCHIVE_FILENAME}" > "$(basename "${CHECKSUM_FILE}")"
fi
cd "${REPO_ROOT}"

echo "========================================================================"
echo " SUCCESS: Stereo-seq Sweep Complete & Audited!"
echo " Results Archive:  ${OUTPUT_ARCHIVE}"
echo " Checksum File:    ${CHECKSUM_FILE}"
echo ""
echo " To publish these results to GitHub Releases, run:"
echo "   PYTHONPATH=src uv run python scripts/manage_release_artifacts.py publish-results \\"
echo "     --tag ${RESULTS_TAG} \\"
echo "     --archive ${OUTPUT_ARCHIVE} \\"
echo "     --checksums ${CHECKSUM_FILE} \\"
echo "     --dataset ${DATASET} \\"
echo "     --split ${SPLIT}"
echo "========================================================================"
