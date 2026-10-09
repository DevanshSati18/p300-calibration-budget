#!/usr/bin/env bash
# Phase B: AUC co-primary budget, threshold sensitivity, stats on both metrics.
set -e
cd "$(dirname "$0")/.."
LOG=outputs/logs/phaseB.log; mkdir -p outputs/logs; : > "$LOG"
run() { echo ">>> $*" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1; }
run python scripts/05_compute_budget.py                      # B80, character accuracy
run python scripts/05_compute_budget.py --metric auc         # AUC budget, threshold 0.75
run python scripts/05_compute_budget.py --metric auc --threshold 0.70
run python scripts/05_compute_budget.py --metric auc --threshold 0.80
run python scripts/06_run_stats.py --metric acc              # valid-decoding subset
run python scripts/06_run_stats.py --metric acc --subset all
run python scripts/06_run_stats.py --metric auc              # all cohorts
echo "PHASE B DONE" | tee -a "$LOG"
