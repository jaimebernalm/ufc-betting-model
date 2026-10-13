#!/usr/bin/env bash
# Single holdout look (2024-2026) for the frozen finalist, fixed ablations and controls.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="/Users/jaimebernal/Desktop/pred_project/UFC pred/.conda/bin/python"
export PYTHONPATH=src
RUN="scripts/research/r2026_10/claude_dev_run.py"
MKT="artifacts/r2026_10/claude/mkt_extra.parquet"
B1="artifacts/r2026_10/codex/perf_features.parquet"
B2="artifacts/r2026_10/codex/perf_adjusted.parquet"

"$PY" -W ignore "$RUN" --holdout --cands R6 --seeds 10 --threads 8 --perf "$MKT" --tag _mkt
"$PY" -W ignore "$RUN" --holdout --cands R3,MF --seeds 10 --threads 8
"$PY" -W ignore "$RUN" --holdout --cands R6 --seeds 10 --threads 8 --perf "$MKT,$B1" --tag _mkt_B1v2
"$PY" -W ignore "$RUN" --holdout --cands R6 --seeds 10 --threads 8 --perf "$MKT,$B1,$B2" --tag _mkt_B1v2B2
echo HOLDOUT_DONE
