# HANDOFF_TO_GPU.md: GPU Training Session Handoff Ledger & Contract

This document provides the exact, copy-pasteable execution contract for ephemeral GPU training sessions.
No GPU training command is ever assembled by hand; commands are deterministically emitted by batch configuration.

---

## GPU Execution Protocol

Every GPU worker session follows a strict five-step lifecycle:
1. **Setup**: Pull exact pinned Git commit, synchronize environment from `uv.lock`, verify CUDA device.
2. **Input Verification**: Re-verify SHA-256 hashes of frozen splits, feature bundles, and graph construction manifests before running compute.
3. **Deterministic Run**: Execute single canonical batch entry point (`python -m spatial_graph_bench.train ...` or `make train-batch`).
4. **Pack Delivery**: Execute `scripts/package_gpu_results.py` to index all run files, verify pack-time integrity, generate `batch_manifest.json`, and output `gpu_results_<batch>_<hash>.tar.gz` and batch fingerprint.
5. **Receive & Ingest (CPU Side)**: Local machine receives tarball and runs `python scripts/receive_gpu_delivery.py`, which executes 4-layer validation (file hashes, aggregate hash, provenance match, metric recomputation) and appends to `audits/gpu_runs/ingestion_log.jsonl`.

---

## Batch Handoff Template

```markdown
### Batch: <batch_id>
- **Emitted At**: <ISO8601>
- **Target Platform / Split**: <dataset_id> / <split_id>
- **Protocol Variant**: <canonical_section_own | variant_a_bipartite | variant_b_pooled>
- **Commit SHA**: <git_sha>
- **Lockfile SHA**: <uv_lock_sha>

#### 1. Setup Commands
```bash
git clone https://github.com/<owner>/spatial-graph-bench.git
cd spatial-graph-bench
git checkout <git_sha>
uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA device:', torch.cuda.get_device_name(0))"
```

#### 2. Input Artifacts & Hashes
- Split JSON: `splits/<dataset_id>/<split_id>.json` (SHA-256: `<split_hash>`)
- Feature Manifest: `artifacts/preprocessed/<dataset_id>/<split_id>/feature_manifest.json` (SHA-256: `<feat_hash>`)
- Construction Manifest: `artifacts/graphs/<dataset_id>/<split_id>/<graph_id>/graph_manifest.json` (SHA-256: `<graph_hash>`)

#### 3. Execution Command
```bash
uv run python -m spatial_graph_bench.models.runner --batch-config configs/gpu_batch_<batch_id>.yaml
```

#### 4. Packaging & Ship-Back
```bash
uv run python scripts/package_gpu_results.py --dataset <dataset_id> --split <split_id> --output gpu_results_<batch_id>.tar.gz
```

#### 5. Stop Conditions & Failure Policy
- In the event of NaN loss, out-of-memory (OOM), or early divergence, **do not discard artifacts**.
- Package and ship the failed run directory with `run_manifest.json` recording `status: "failed"` and `failure_metadata`. Partial evidence is audited and quarantined upon delivery.

#### 6. Ingestion & Audit Verdict (Completed CPU-Side)
- **Delivered Archive**: `gpu_results_<batch_id>_<hash>.tar.gz`
- **Batch Fingerprint**: `<batch_fingerprint>`
- **Verdict**: `PENDING`
- **Ingestion Log Record**: `audits/gpu_runs/ingestion_log.jsonl`
```

---

## Active & Historical GPU Batches

### Batch: `merfish_canonical_gnn_sweep`
- **Emitted At**: 2026-09-04T15:52:00Z
- **Target Platform / Split**: `merfish_mouse_spinal_cord` / `mouse_held_out_canonical`
- **Protocol Variant**: `canonical_section_own` (and `variant_a_bipartite`)
- **Commit SHA**: `127a9a29c4cebb3c0382011b81e05b8cc57138e3`
- **Lockfile SHA**: `4c1a3104f659b4f13cd99524bab75ea8bb0f82b556ff70dcddf8482dc6bd9cdf`

#### 1. Setup Commands
```bash
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench
git checkout 127a9a29c4cebb3c0382011b81e05b8cc57138e3
uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA device:', torch.cuda.get_device_name(0))"
```

#### 2. Input Artifacts & Hashes
- Split JSON: `splits/merfish_mouse_spinal_cord/mouse_held_out_canonical.json` (SHA-256: `a9ed50703ca8fa9cfc1a2eb1c1639b1ba284253c02b5e3556ab0b6242ceaf31a`)
- Feature Manifest: `artifacts/preprocessed/merfish_mouse_spinal_cord/mouse_held_out_canonical/feature_manifest.json` (SHA-256: `0ad75181c4f2c80b47d8c9b28d2d46fd09c3dc336480c02951abb653fb882e79`)
- Graph Manifest (`spatial_knn_k6`): `artifacts/graphs/merfish_mouse_spinal_cord/mouse_held_out_canonical/spatial_knn_k6/graph_manifest.json` (SHA-256: `1f3f0e36f26ba424603015ff90b8a2ee91d6c27664e27b5577831a2371a4243e`)
- Frozen MLP Baselines: `audits/baselines_snapshot/merfish_mouse_spinal_cord/mouse_held_out_canonical/baselines_summary.json` (Parity Band: $\pm 0.0069$)

#### 3. Execution Command
```bash
PYTHONPATH=src uv run python scripts/run_gnn_sweep.py --batch-config configs/gpu_batch_merfish_canonical.yaml
```

#### 4. Packaging & Delivery
```bash
uv run python scripts/package_gpu_results.py --dataset merfish_mouse_spinal_cord --split mouse_held_out_canonical --output gpu_results_merfish_canonical.tar.gz
```

#### 5. Local CPU Ingestion
```bash
uv run python scripts/receive_gpu_delivery.py gpu_results_merfish_canonical.tar.gz
```
- **Verdict**: `PENDING_DELIVERY`
- **Ingestion Log Record**: `audits/gpu_runs/ingestion_log.jsonl`

