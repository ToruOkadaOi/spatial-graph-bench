# Installation Guide

This guide covers installing `spatial-graph-bench` using multiple environment managers: **`uv`** (strongly recommended for speed and reproducibility), standard **`pip` + `venv`**, **`conda` / `mamba`**, and **`Docker`**.

---

## Prerequisites
- **Python**: Version `3.11` or `3.12`.
- **Operating System**: Linux (Ubuntu 22.04 / 24.04 recommended), macOS (Apple Silicon or Intel), or Windows (via WSL2).
- **GPU Acceleration (Optional)**: NVIDIA GPU with CUDA 12.1+ for large GNN sweeps. Standard CPUs are fully supported for all baselines, data preparation, graph construction, and smoke tests.

---

## Method 1: Using `uv` (Recommended)

`uv` provides instant dependency resolution and exact lockfile reproduction.

```bash
# 1. Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# 3. Synchronize exact environment from uv.lock
uv sync --frozen

# 4. Run sanity check (< 1 second)
uv run python scripts/run_dummy_benchmark.py
```

---

## Method 2: Standard `pip` and Virtual Environment

```bash
# 1. Clone repository
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench

# 2. Create and activate a clean virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package with development tools
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Verify installation
python scripts/run_dummy_benchmark.py
```

---

## Method 3: Using `conda` / `mamba`

```bash
# 1. Create a conda environment with Python 3.11
conda create -n spatial-bench python=3.11 -y
conda activate spatial-bench

# 2. Clone and install
git clone https://github.com/ToruOkadaOi/spatial-graph-bench.git
cd spatial-graph-bench
pip install -e ".[dev]"

# 3. Verify installation
python scripts/run_dummy_benchmark.py
```

---

## Method 4: Docker Container

A multi-stage `Dockerfile` is provided in the repository root.

```bash
# 1. Build container image
docker build -t spatial-graph-bench:latest .

# 2. Run the dummy benchmark inside container
docker run --rm spatial-graph-bench:latest uv run python scripts/run_dummy_benchmark.py

# 3. Run interactive shell with GPU support (requires nvidia-container-toolkit)
docker run --gpus all -it --rm spatial-graph-bench:latest /bin/bash
```

---

## Sanity Check

Once installed, run the standalone dummy benchmark to verify that PyTorch, PyTorch Geometric, Scanpy, and scikit-learn are working properly together:

```bash
uv run python scripts/run_dummy_benchmark.py
# Or with standard activated venv:
python scripts/run_dummy_benchmark.py
```

You should see:
```text
ALL CHECKS PASSED (Completed in < 1s)
The benchmark pipeline, schemas, training loops, and audits are fully functional on CPU.
```
