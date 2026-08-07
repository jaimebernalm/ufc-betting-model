.DEFAULT_GOAL := help
.PHONY: help setup test lint format check clean data features train backtest preview

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and dev tooling into the active environment
	python -m pip install --upgrade pip
	pip install -e ".[dev,notebooks]"
	pre-commit install

test:  ## Run the test suite
	pytest -q

lint:  ## Check style and imports
	ruff check .
	ruff format --check .

format:  ## Auto-fix style and reformat
	ruff check . --fix
	ruff format .

check: lint test  ## Everything CI runs

clean:  ## Remove caches and build artifacts
	find . -path ./.conda -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info

# ── Pipeline ───────────────────────────────────────────────────────────────
data:  ## Refresh the UFCStats fight history
	ufc-update

features:  ## Rebuild the Bayesian skill features (slow: ~30 min CPU)
	python scripts/tools/build_skill_features.py

train:  ## Train the deployed 10-seed ensemble
	python -m ufc_pred.models.baseline_v7_1

backtest:  ## Evaluate the strategy grid on the validation window
	python scripts/research/hybrid_strategies_backtest.py

preview:  ## Read-only preview of the next Kalshi card
	ufc-preview
