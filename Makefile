.PHONY: install install-dev lint format test typecheck check clean smoke-test

PYTHON ?= python3
UV ?= uv

install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev]"

lint:
	$(UV) run ruff check src tests scripts

format:
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

test:
	$(UV) run pytest -v tests

typecheck:
	$(UV) run mypy --explicit-package-bases src tests scripts

smoke-test:
	$(UV) run pytest -v tests/test_smoke_pipeline.py

test-dummy:
	$(UV) run python scripts/run_dummy_benchmark.py

check: lint test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
