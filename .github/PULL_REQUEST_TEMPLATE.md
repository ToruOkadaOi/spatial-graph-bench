## Description
Briefly describe the purpose of this PR and what changes were made.

## Invariant Verification Checklist
Please certify the following pre-registered benchmark invariants:
- [ ] **Topological Integrity**: Zero cross-partition and zero cross-section edges certified (`scripts/validate_construction.py`).
- [ ] **Spatial Ignorance**: Feature pipeline produces identical PCA features under coordinate-shuffle test (`verify_spatial_ignorance`).
- [ ] **Label Coverage**: Disjoint train/val/test splits with 100% test class representation (`scripts/validate_split.py`).
- [ ] **Baselines**: Spatially ignorant baseline comparison preserved across identical seeds.
- [ ] **Tests Pass**: `uv run pytest -v tests` and `uv run python scripts/run_dummy_benchmark.py` pass cleanly.
- [ ] **Code Quality**: `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` pass.

## Related Issues
Closes #
