# Reproducibility check (11 Oct 2026)

The full analysis was re-run from scratch and compared with the published results.

**What was re-run**
- Raw EDF → features (step 03) for 3 records (B, O, R), compared with the cached features.
- Calibration sweep (step 04) for all 566 records with `--overwrite` (≈7,100 models), then
  budgets (05), statistics (06), robustness (06b) and independent verification (08).
- Environment: Linux (OpenBLAS); the published results were produced on macOS (Apple Accelerate),
  same package versions of the analysis code.

**Results**
| Check | Outcome |
|---|---|
| Feature extraction (3 records) | 11/13 arrays bit-identical (labels, order, flash membership, targets); features differ by ≤ 2×10⁻⁶ µV (float32 rounding) |
| Calibration budgets (B80, AUC budget, 0.70/0.80 sensitivity) | **Identical** |
| Leave-one-cohort-out tables, robustness summary, decoding validity | **Identical** |
| Pooled slope, τ, I², prediction interval | Identical to 3 decimals |
| Artefact moderator | p 0.0020 → 0.0019, Holm p 0.0156 → 0.0156 (unchanged conclusion) |
| Character decisions | 8 of 188,912 decoded characters flipped (0.004%) |
| Flash-level AUC | median change 0; only 2 rows > 0.01, both degenerate fits (3 training epochs, 1 target, AUC ≈ chance) |
| Independent verification (08) | 13/13 checks passed; all manuscript numbers match |

**Interpretation:** the residual differences are floating-point effects of different linear-algebra
libraries on near-singular fits at k = 1; no reported number or conclusion changes.

Published (macOS) outputs: `outputs/tables/`. Linux re-run: `outputs/tables_rerun_linux/`.
