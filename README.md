# p300-calibration-budget

**Does a P300 speller's calibration budget transport across protocols?**
A cross-cohort analysis of calibration requirements in the public bigP3BCI archive.

For a fixed shrinkage-LDA decoder, we estimate in each bigP3BCI cohort the calibration
budget **B80**: the smallest number of calibration characters at which 80% of participants
reach 70% offline character accuracy. We then test how much B80 varies between cohorts,
whether protocol characteristics explain the variation, and whether a budget estimated on
other cohorts predicts a withheld cohort's budget. See [PROPOSAL.md](PROPOSAL.md) (frozen v1.0).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Data

bigP3BCI v1.0.0 (PhysioNet, doi:10.13026/0byy-ry86) is not redistributed here.
Download it and link it into the project:

```bash
ln -s /path/to/bigP3BCI-data data/raw/bigP3BCI-data
```

## Pipeline (run in order)

| Step | Script | Output |
| --- | --- | --- |
| 01 | `scripts/01_audit_archive.py` | `outputs/tables/archive_files.csv`, `archive_summary.csv` |
| 02 | `scripts/02_build_inventory.py` | `inventory_records.csv`, `eligibility_cohorts.csv` |
| 03 | `scripts/03_extract_epochs.py` | cached features per record in `data/processed/` |
| 04 | `scripts/04_run_sweep.py` | accuracy per record × calibration size |
| 05 | `scripts/05_compute_budget.py` | learning curves, B80 with bootstrap CIs |
| 06 | `scripts/06_run_stats.py --metric acc\|auc` | heterogeneity, moderators, leave-one-cohort-out |
| 06b | `scripts/06b_robustness.py` | robustness variants (first session, low artefact, thresholds) |
| 07 | `scripts/07_make_figures.py [--paper]` | figures in `outputs/figures/` (`--paper`: print versions) |
| 08 | `scripts/08_verify_results.py` | independent verification → `outputs/tables/verification_report.csv` |

Shortcuts: `bash scripts/run_phaseB.sh` (budgets + stats) and `bash scripts/run_phaseC.sh` (stats, robustness, figures).
Any step accepts `--dev` for a quick run on a small subset.

All settings live in `config/frozen.yaml`. Every output records the config hash that produced it.

## Layout

```
config/       frozen analysis settings
src/p300budget/  io_edf, events, preprocess, decoder, sweep, budget, stats_models, plots
scripts/      numbered pipeline steps
tests/        unit tests (synthetic data)
outputs/      tables, figures, logs
paper/        manuscript
```

## Citation

Please cite the bigP3BCI archive (Mainsah et al., PhysioNet 2025) and this repository.
