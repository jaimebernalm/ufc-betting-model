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

# Corrected, isolated bundle. Existing production model files are preserved.
.PHONY: corrected-history corrected-skill corrected-train corrected-evaluate shadow-check
corrected-history:
	PYTHONPATH=src .conda/bin/python scripts/research/validate_result_envelope.py

corrected-skill: corrected-history
	PYTHONPATH=src .conda/bin/python -c "from pathlib import Path; from ufc_pred.features.skill_v3_pipeline import build; p=Path('artifacts/corrected_2026_09_14'); print(build(output_path=p/'skill_features.parquet', history_path=p/'verified_history.parquet'))"

corrected-train:
	PYTHONPATH=src .conda/bin/python scripts/research/rebuild_corrected_system.py

corrected-evaluate:
	PYTHONPATH=src .conda/bin/python scripts/research/evaluate_corrected_system.py

shadow-check:
	PYTHONPATH=src .conda/bin/python -m ufc_pred.cli.shadow_runner --check
