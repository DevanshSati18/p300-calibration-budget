#!/usr/bin/env bash
# Full re-run from cached features (sweep from scratch) for a reproducibility check.
set -e
cd "$(dirname "$0")/.."
LOG=outputs/logs/rerun_check.log; mkdir -p outputs/logs; : > "$LOG"
run() { echo ">>> $(date +%H:%M:%S) $*" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1; }
run python3 scripts/04_run_sweep.py --jobs 3 --overwrite
run python3 scripts/05_compute_budget.py
run python3 scripts/05_compute_budget.py --metric acc_r_secondary
run python3 scripts/05_compute_budget.py --metric auc
run python3 scripts/05_compute_budget.py --metric auc --threshold 0.70
run python3 scripts/05_compute_budget.py --metric auc --threshold 0.80
run python3 scripts/06_run_stats.py --metric acc
run python3 scripts/06_run_stats.py --metric acc --subset all
run python3 scripts/06_run_stats.py --metric auc
run python3 scripts/06b_robustness.py
run python3 scripts/08_verify_results.py
echo "RERUN DONE $(date +%H:%M:%S)" | tee -a "$LOG"
