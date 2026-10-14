# p300-calibration-budget

Code and results for the paper **"Calibration Data Requirements of Machine Learning Classifiers for EEG-Based P300 Spellers: A Multi-Study Learning-Curve Analysis"** (D. Sati and A. Shah, Rashtriya Raksha University; submitted to MoSICom 2026).

P300 spellers need a calibration phase before each new user can spell. Methods that shorten calibration are usually tested on one dataset, which assumes the calibration a decoder needs is portable. This repository measures how much calibration a fixed decoder needs in 14 cohorts of the public [bigP3BCI](https://doi.org/10.13026/0byy-ry86) archive (224 participants, 566 records), and tests whether that amount transports between studies.

## Main results

| Finding | Value | Source table |
| --- | --- | --- |
| Calibration budget (80% of participants usable) | 2–30 characters; 15-fold (bootstrap 95% CI 6–18-fold) | `b80.csv`, `b80_auc.csv`, `transport_summary.json` |
| Heterogeneity of learning-curve slopes | I² = 0.88 (τ = 0.20, prediction interval 0.55–1.56) | `heterogeneity_acc_valid.json` |
| Error of a budget borrowed from other cohorts | factor 2.5 (B80) and 3.1 (AUC budget) | `transport_reference_*.csv` |
| Error between two halves of the same cohort | factor 1.5 and 1.3 | `transport_reference_*.csv` |
| Five-participant local pilot (secondary, post-hoc) | factor 1.45 (AUC) and 1.5 (B80) | `pilot_*.csv` |
| Protocol descriptors (slope and intercept) | none significant after Holm correction | `moderators_*.csv`, `intercept_moderators_*.csv` |

Decoder: shrinkage LDA (scikit-learn, Ledoit–Wolf) trained on the first *k* calibration characters in recording order. Budget B80 is the smallest *k* at which ≥ 80% of a cohort's participants reach ≥ 70% character accuracy. The AUC budget uses flash-level AUC ≥ 0.75 instead.

## Reproduce

### 1. Install

```bash
git clone https://github.com/DevanshSati18/p300-calibration-budget
cd p300-calibration-budget
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions used for outputs/tables
pip install -e ".[dev]"
pytest -q                              # 10 unit tests on synthetic data
```

### 2. Reproduce every number in the paper without the EEG data (~5 min)

The calibration sweep (6,488 trained decoders plus 562 shuffled-label controls) is saved in `outputs/tables/sweep_results.csv`. Everything downstream can be re-run from it:

```bash
bash scripts/run_analysis.sh           # budgets, statistics, robustness, transport, figures, verification
```

The last step, `08_verify_results.py`, re-implements truncation, decoding and the budget rule independently. It also checks that every number quoted in `paper/main.tex` matches the output tables. Expected output: `16/16 checks passed`.

### 3. Full pipeline from the raw EEG (several hours)

Download bigP3BCI v1.0.0 from PhysioNet (about 45 GB; it is not redistributed here) and link it in:

```bash
ln -s /path/to/bigP3BCI-data data/raw/bigP3BCI-data
JOBS=4 bash scripts/run_all.sh
```

Six studies are excluded by design: C and P (online outcomes not recoverable), and D, F, G and H (calibration and test recorded in different conditions). The processed features (`data/processed/`, about 11 GB) contain only the 16 shared EEG channels, 0–800 ms after each flash, at 64 Hz.

## Pipeline

| Step | Script | Produces (in `outputs/tables/` unless noted) |
| --- | --- | --- |
| 01 | `01_audit_archive.py` | `archive_files.csv`, `archive_summary.csv` (file-level audit) |
| 02 | `02_build_inventory.py` | `inventory_files.csv`, `inventory_records.csv`, `eligibility_cohorts.csv` |
| 03 | `03_extract_epochs.py` | features per record in `data/processed/epochs/`; `extraction_log.csv` |
| 04 | `04_run_sweep.py` | `sweep_results.csv` (accuracy and AUC per record × calibration size) |
| 05 | `05_compute_budget.py --metric acc\|auc\|acc_r_secondary` | `participant_curves*.csv`, `learning_curves*.csv`, `b80*.csv` |
| 06 | `06_run_stats.py --metric acc\|auc` | `cohort_fits_*`, `heterogeneity_*`, `moderators_*`, `b80_moderators_*`, `loco_*`, `decoding_validity.csv` |
| 06b | `06b_robustness.py` | `robustness_budgets.csv`, `robustness_summary.csv` (Table II) |
| 07 | `07_make_figures.py --paper` | `outputs/figures/fig1–fig4_*_paper.{pdf,png}` |
| 08 | `08_verify_results.py` | `verification_report.csv` |
| 09 | `09_transport_pilot.py` | `transport_reference_*.csv`, `pilot_*.csv`, `intercept_moderators_*.csv`, `transport_summary.json` |
| 10 | `10_participant_overlap.py` | `participant_overlap.csv` (screen for the same person in several studies) |

Every step reads its settings from `config/frozen.yaml`, and every output records the hash of that file. Steps 03 and 04 accept `--dev` for a quick run on a small subset (outputs get a `_dev` suffix).

## Repository layout

```
config/frozen.yaml     all analysis settings (frozen before the full run; later changes logged in PROPOSAL.md)
src/p300budget/        io_edf, events, preprocess, decoder, sweep, budget, stats_models, config
scripts/               numbered pipeline steps + run_analysis.sh / run_all.sh
tests/                 unit tests on synthetic data
outputs/tables/        all result tables used in the paper
outputs/figures/       paper figures
outputs/audit/         sensitivity checks from the internal audit (13 Oct 2026)
paper/                 manuscript (IEEEtran LaTeX) and figures
docs/                  reproducibility checks
PROPOSAL.md            study plan (v1.0) and dated change log of every deviation
```

## Transparency notes

- The analysis plan was frozen in `PROPOSAL.md` and `config/frozen.yaml` before the full run. This is an internal freeze, not a public pre-registration. Every later change is dated in the PROPOSAL change log, including:
  - the AUC co-primary budget;
  - the decoding-validity rule;
  - the exploratory artefact moderator;
  - the post-hoc transport reference and local-pilot analyses.
- Re-running on a different machine changes results only at floating-point level (see `docs/reproducibility_check.md`).
- Code and manuscript drafting were assisted by an AI system (Claude, Anthropic). All analyses were re-run and verified.

## Citation

If you use this code, please cite the paper and the bigP3BCI archive:

> B. Mainsah, C. Fleeting, T. Balmat, E. Sellers, and L. Collins, "bigP3BCI: an open, diverse and machine learning ready P300-based brain-computer interface dataset (v1.0.0)," PhysioNet, 2025, doi:10.13026/0byy-ry86.

License: see `LICENSE`.
