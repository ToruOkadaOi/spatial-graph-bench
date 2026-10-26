# Training workflow: local CPU → Vast GPU → local analysis

All work for this workflow belongs to `fix/inductive-retraining` in the separate worktree `/Users/aman/Documents/spatial-graph-bench-retrain`. The original checkout and README remain unchanged.

## What changed

- GNN optimization forwards contain training nodes only, including every BatchNorm operation. Validation forwards contain training references and validation queries in evaluation mode; test nodes never enter these forwards.
- Epoch selection uses fresh post-update validation predictions. Validation metrics use the intersection of training/validation labels; test metrics use the training/test intersection. Test labels do not select checkpoints. The frozen dataset ontology/output dimension is shared across models.
- Coordinates are shuffled independently within each partition and section. Rewiring swaps undirected edge pairs within each partition/section, preserving both directed degrees and reciprocal edges. Graph randomness is fixed at seed 42, with ten attempted swaps per undirected edge.
- Every neural epoch atomically saves model state, optimizer state, RNG states, history, selected weights and stopping state. Resume requires the same source, configuration, input bytes and environment. Completed runs are hashed and skipped; changed or incompatible files cause an error. One sweep process may own a results directory at a time.
- Model selection, graph/control fixes and resume are regression-tested on CPU. CUDA behavior must pass the worker's smoke test; GPU bitwise determinism is not asserted.

The full experiment fits 1,120 GNNs, 40 MLPs and four random forests. Fresh baselines ensure a consistent validation policy across models. Features/PCA are reused; no raw-data preprocessing is repeated. The default command runs all fits on the instance (RF uses its CPU). An optional local-baseline path is provided below.

## 1. Local inputs (already prepared for this handoff)

From the new worktree, this is the preparation command for a **new** destination:

```bash
uv sync --frozen --all-extras
uv run --frozen python scripts/retrain.py prepare \
  --source /Users/aman/Documents/spatial-graph-bench \
  --output experiments/inductive-v3/inputs \
  --archive dist/inductive_v3_inputs.tar.gz
```

Preparation retains the original PCA/label/coordinate arrays, copies the frozen splits and existing real/reference graph edges, creates manifests, and rebuilds the four controls per dataset. It refuses to overwrite an existing input directory or archive. Verify an existing handoff with:

```bash
uv run --frozen python scripts/retrain.py verify-inputs \
  --inputs experiments/inductive-v3/inputs
```

The code handoff is a Git bundle containing the committed branch. This avoids requiring GitHub credentials on the rented machine. Its SHA-256 sidecar and the input archive sidecar must travel with the files.

## 2. Copy to the instance from your laptop

Use the SSH host and port shown by Vast's Connect button:

```bash
cd /Users/aman/Documents/spatial-graph-bench-retrain
export RETRAIN_HOST='REPLACE_WITH_VAST_SSH_HOST'
export RETRAIN_PORT='REPLACE_WITH_VAST_SSH_PORT'
ssh -p "$RETRAIN_PORT" "root@$RETRAIN_HOST" 'mkdir -p /workspace/retraining-transfer'
scp -P "$RETRAIN_PORT" \
  dist/inductive_v3_code.bundle dist/inductive_v3_code.bundle.sha256 \
  dist/inductive_v3_inputs.tar.gz dist/inductive_v3_inputs.tar.gz.sha256 \
  "root@$RETRAIN_HOST:/workspace/retraining-transfer/"
```

## 3. Instance setup

Choose a Linux x86-64 SSH instance with an NVIDIA GPU. The current lock resolves PyTorch 2.14 with CUDA 13 packages on Linux; select a compatible driver. NVIDIA lists driver branch 580 or newer for CUDA 13 minor compatibility, with feature/PTX restrictions possible on older compatible drivers. The following real CUDA/GAT smoke tests are the execution gate. See [NVIDIA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/latest/595/minor-version-compatibility.html).

Inside the instance:

```bash
set -euo pipefail
apt-get update
apt-get install -y git curl ca-certificates tmux libgomp1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/retraining-transfer
sha256sum -c inductive_v3_code.bundle.sha256
sha256sum -c inductive_v3_inputs.tar.gz.sha256

git clone --branch fix/inductive-retraining \
  /workspace/retraining-transfer/inductive_v3_code.bundle \
  /workspace/spatial-graph-bench-retrain
cd /workspace/spatial-graph-bench-retrain
# Record this commit with your run; the bundle freezes the supplied branch.
git rev-parse HEAD
uv sync --python 3.11 --frozen --all-extras
mkdir -p experiments/inductive-v3
tar -xzf /workspace/retraining-transfer/inductive_v3_inputs.tar.gz \
  -C experiments/inductive-v3

nvidia-smi
uv run --frozen python scripts/retrain.py check-gpu
uv run --frozen python scripts/retrain.py verify-inputs \
  --inputs experiments/inductive-v3/inputs
```

Use a separate instance working directory if one already exists. Keep the committed code unchanged during a sweep; changing it deliberately invalidates checkpoint/cache compatibility. There is no CPU fallback when CUDA is requested.

## 4. Smoke, pilot and full runs

Use tmux so an SSH disconnection does not end the session:

```bash
tmux new -s inductive-retrain
```

Inside tmux:

```bash
cd /workspace/spatial-graph-bench-retrain
export PATH="$HOME/.local/bin:$PATH"
set -o pipefail
mkdir -p experiments/inductive-v3/logs

# Largest-dataset GAT memory/runtime probe: one MLP and one GAT, two epochs.
./scripts/run_retraining.sh smoke 2>&1 | tee -a experiments/inductive-v3/logs/smoke.log

# Inspect GPU memory recorded for the GAT run.
cat experiments/inductive-v3/cuda-smoke/xenium_mouse_kidney/replicate_held_out_canonical/gat_spatial_knn_k12_seed42/performance.json

# 12 GNNs + six matched MLPs: Stereo-seq/MERFISH, GCN/GraphSAGE, k12, seeds 42–44.
./scripts/run_retraining.sh pilot 2>&1 | tee -a experiments/inductive-v3/logs/pilot.log

# After reviewing validation histories and runtime, run the full frozen grid.
./scripts/run_retraining.sh full 2>&1 | tee -a experiments/inductive-v3/logs/full.log
```

Detach with `Ctrl-b`, then `d`. Reconnect using `tmux attach -t inductive-retrain`. If a run is interrupted, rerun its identical command: valid completed runs are skipped and incomplete neural runs resume from their last epoch checkpoint. Failed RF fits restart from initialization. A compact result export is for local analysis; it is not a full interruption backup.

Do not select hyperparameters from pilot test scores. Inspect training/validation histories for debugging. Pilot and smoke outputs live outside `full/` and cannot be mistaken for final runs. Partial grids receive descriptive summaries only; global Holm decisions require the full 112 configurations × ten seeds.

## 5. Package and download results

On the instance:

```bash
cd /workspace/spatial-graph-bench-retrain
uv run --frozen python scripts/retrain.py summarize \
  --results experiments/inductive-v3/full
uv run --frozen python scripts/retrain.py pack \
  --results experiments/inductive-v3/full \
  --archive dist/inductive_v3_results.tar.gz
```

The default compact package includes selected model weights, all partition predictions, histories, environment/configuration records and metrics. It omits large probability arrays and interruption checkpoints. Use `pack --full` with a distinct archive name for a complete backup including these files; do not delete the sole copy if you need them.

On the laptop:

```bash
cd /Users/aman/Documents/spatial-graph-bench-retrain
mkdir -p dist returned/inductive-v3
scp -P "$RETRAIN_PORT" \
  "root@$RETRAIN_HOST:/workspace/spatial-graph-bench-retrain/dist/inductive_v3_results.tar.gz" \
  "root@$RETRAIN_HOST:/workspace/spatial-graph-bench-retrain/dist/inductive_v3_results.tar.gz.sha256" \
  dist/
(cd dist && shasum -a 256 -c inductive_v3_results.tar.gz.sha256)
tar -xzf dist/inductive_v3_results.tar.gz -C returned/inductive-v3
uv run --frozen python scripts/retrain.py verify-results \
  --results returned/inductive-v3/results \
  --inputs experiments/inductive-v3/inputs
```

The return check independently recalculates train/validation/test metrics from the frozen labels and saved predictions.

Read `returned/inductive-v3/results/summary.csv` and `summary.json`. These commands leave your README and existing run outputs untouched. Returned compact exports can regenerate statistics without the omitted probabilities. Copy and verify the data you want to retain before destroying the rented instance.

## Optional: keep baseline fitting on the local CPU

Instead of running the full `all` phase on Vast, run this locally after committing the branch:

```bash
uv sync --python 3.11 --frozen --all-extras
uv run --frozen python scripts/retrain.py run \
  --inputs experiments/inductive-v3/inputs \
  --results experiments/inductive-v3/full \
  --phase baselines --device cpu
```

Copy that `full/` directory to the **same relative experiment location** on Vast, then run:

```bash
uv run --frozen python -u scripts/retrain.py run \
  --inputs experiments/inductive-v3/inputs \
  --results experiments/inductive-v3/full \
  --phase gnn --device cuda:0
```

Keep seeds, epoch budget, thread count, inputs and source commit identical across the two stages. The stored baseline device may be CPU while the GNN device is CUDA. Use `--phase gnn` when following this path; the default `all` phase intentionally rejects a cached baseline with a different requested device configuration.
