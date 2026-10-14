#!/usr/bin/env bash
# Every statistic, table and figure in the paper from the saved sweep
# (outputs/tables/sweep_results.csv is in the repository) — no EEG data needed. ~5 min.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOG=outputs/logs/run_analysis.log; mkdir -p outputs/logs; : > "$LOG"
run() { echo ">>> $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; }
run scripts/05_compute_budget.py                        # B80 (character accuracy)
run scripts/05_compute_budget.py --metric auc           # AUC budget (co-primary)
run scripts/05_compute_budget.py --metric acc_r_secondary   # pre-specified secondary (R = 1)
run scripts/06_run_stats.py --metric acc                # heterogeneity, moderators, LOCO (valid cohorts)
run scripts/06_run_stats.py --metric auc                # same, all cohorts
run scripts/06b_robustness.py                           # Table II
run scripts/09_transport_pilot.py                       # within-cohort reference, local pilot
run scripts/07_make_figures.py --paper                  # Figs 1-4
run scripts/08_verify_results.py                        # independent checks + paper numbers
echo "DONE — see $LOG and outputs/tables/verification_report.csv"
