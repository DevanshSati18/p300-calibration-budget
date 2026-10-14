#!/usr/bin/env bash
# Full pipeline from the raw bigP3BCI EDF files (data/raw/bigP3BCI-data). Several hours.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; JOBS=${JOBS:-4}
LOG=outputs/logs/run_all.log; mkdir -p outputs/logs; : > "$LOG"
run() { echo ">>> $(date +%H:%M:%S) $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; }
run scripts/01_audit_archive.py --jobs "$JOBS"          # file-level audit of the archive
run scripts/02_build_inventory.py --jobs "$JOBS"        # records and eligibility
run scripts/03_extract_epochs.py --jobs "$JOBS"         # filtering, epochs, features
run scripts/04_run_sweep.py --jobs "$JOBS" --overwrite  # 6,488 decoders + shuffled controls
run scripts/10_participant_overlap.py                   # cross-study duplicate screen
bash scripts/run_analysis.sh
