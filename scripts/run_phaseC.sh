#!/usr/bin/env bash
# Phase C: stats with artefact-rejection moderator, robustness, figures.
set -e
cd "$(dirname "$0")/.."
LOG=outputs/logs/phaseC.log; mkdir -p outputs/logs; : > "$LOG"
run() { echo ">>> $*" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1; }
run python scripts/06_run_stats.py --metric acc
run python scripts/06_run_stats.py --metric acc --subset all
run python scripts/06_run_stats.py --metric auc
run python scripts/06b_robustness.py
run python scripts/07_make_figures.py
echo "PHASE C DONE" | tee -a "$LOG"
