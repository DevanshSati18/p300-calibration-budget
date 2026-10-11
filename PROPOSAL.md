# PROPOSAL — FROZEN v1.0 (9 Oct 2026)

> Status: **FROZEN.** Changes only via the kill/pivot criteria in §10, logged in §12.

## 1. Title
**Does a P300 Speller's Calibration Budget Transport Across Protocols? A Cross-Cohort Analysis of Calibration Requirements in bigP3BCI**

## 2. Problem statement
Practitioners need to know how much subject-specific calibration a P300 speller requires before a new user reaches usable accuracy. It is unknown whether this calibration budget is a stable property of the decoder or varies across recording protocols and cohorts. This study estimates, for a fixed conventional decoder (shrinkage LDA), the calibration budget required in each eligible cohort of the public bigP3BCI archive; quantifies its between-cohort variation; tests whether protocol characteristics explain that variation; and evaluates how accurately a budget estimated from other cohorts predicts the budget of a withheld cohort.

## 3. Gap and positioning
- SpellerSSL (Hong et al., NeurIPS 2025): reports 60% calibration reduction, single legacy dataset (BCI Comp. II-B). Method paper.
- Gao et al. (TNSRE 2023), Chen et al. (arXiv 2602.15955), Song et al. (JNE 2024): calibration-reduction / transfer methods. Not budget variability.
- **Gorenshtein et al. (medRxiv 2026, 10.64898/2026.07.30.26359353):** 18 bigP3BCI cohorts; shows the *calibration-score → accuracy mapping* is cohort-specific. Always uses the **full** calibration block; never varies calibration amount (verified in their code, 9 Oct 2026).
- **This study:** tests whether the *calibration budget* itself transports. Direct, complementary extension.

## 4. Research questions
- **RQ1** How does offline character accuracy change with calibration size k, within each cohort?
- **RQ2** How much does the required budget B80 vary between cohorts?
- **RQ3** Do protocol characteristics (time per selection, flashes per sequence, grid size, paradigm) explain B80 variation?
- **RQ4** How accurately does a pooled B80 predict a withheld cohort's B80?

## 5. Hypotheses
- **H1** Accuracy rises and saturates with k; the saturation point differs between cohorts.
- **H2** B80 varies across cohorts by ≥ 2× (max/min over cohorts with defined B80).
- **H3** Time per selection / stimulus repetitions explains a substantial share of between-cohort B80 variance.
- **H4** Leave-one-cohort-out B80 prediction errs by ≥ 2 characters in ≥ 1/3 of cohorts.
Rejection of any hypothesis is a reportable result.

## 6. Operational definitions
| Term | Definition |
|---|---|
| Calibration budget k | First k calibration characters in recording order. Also reported: target epochs, seconds. |
| Grid | k ∈ {1, 2, 3, 4, 6, 8, 10, 12, 15, 20, …, max}; truncated to what each session has. |
| Test data | Test-phase recordings of the same session, decoded offline. Never overlaps calibration. |
| Decoder | Shrinkage LDA (Ledoit–Wolf). 16 ch, 0.5–30 Hz 4th-order zero-phase Butterworth, epoch 0–800 ms, baseline −200–0 ms, decimate to 64 Hz. Fixed for all cohorts. |
| Character decoding | Sum LDA scores over the flashes containing each candidate character across R sequences; argmax. |
| Repetitions R | One common fixed value (largest supported by all eligible cohorts; set in Day 2–3 audit). Secondary sweep over R. |
| Usable accuracy | ≥ 70% offline character accuracy. |
| **B80 (primary)** | Smallest k at which ≥ 80% of a cohort's participants reach usable accuracy. Participant-bootstrap 95% CI. Right-censored if not reached. |
| Cohort | One bigP3BCI source study. |
| Eligible cohort | Has Train + Test phases, recoverable target and stimulus codes, ≥ 5 participants, ≥ 4 evaluable k levels. |

## 7. Data
bigP3BCI v1.0.0, PhysioNet, doi:10.13026/0byy-ry86. 20 source studies, shared 16-channel montage at 256 Hz. Candidate cohorts: the 18 with recoverable outcomes (all except Studies C and P), subject to §6 eligibility.

## 8. Analysis
1. Per-participant accuracy at each k (and target/non-target AUC as secondary).
2. Per-cohort learning curves; B80 per cohort with participant-cluster bootstrap (2,000 reps).
3. Heterogeneity: mixed-effects logistic, `correct ~ log(k) + (1 | participant) + (1 + log k | cohort)`; report cohort-level SD of intercept and slope; B80 range and ratio.
4. Moderators: protocol descriptors added as cohort-level fixed effects interacting with log(k); Holm correction.
5. Transportability: leave-one-cohort-out — fit pooled curve on 17, predict withheld B80, report absolute error.
6. Secondary: sweep R; EEGNet replication only if Days 1–11 are on schedule.

## 9. Scope
In: offline bigP3BCI analysis, one fixed classical decoder, within-session budgets, cross-cohort comparison.
Out: new algorithms; online testing; reproducing original dynamic-stopping decoders; causal ALS claims (cohort type confounded with protocol — descriptive only); subject-transfer learning.

## 10. Validity safeguards & kill criteria
Safeguards: chronological calibration truncation only; all fitting on calibration data only; hyperparameters frozen on 2 pilot cohorts before full run; participant-clustered CIs; cohort as unit of replication.
- **Day 2:** published calibration-size sweep on bigP3BCI found → stop and reframe.
- **Day 4:** offline decoding not plausible on 2 pilot cohorts → primary outcome becomes AUC curves.
- **Day 7:** < 6 eligible cohorts → multi-cohort case study, drop RQ4.

## 11. Timeline & deliverables
| Days | Work |
|---|---|
| 1–3 | Data download, archive audit, event parsing, fix R, eligibility table |
| 4–7 | Pipeline: epoching, LDA, character decoding; pilot on 2 cohorts; freeze hyperparameters |
| 8–11 | Full budget sweep, learning curves, B80 + bootstrap |
| 12–15 | Mixed-effects, moderators, LOCO transport, figures |
| 16–21 | Paper draft (6–8 pp), repo cleanup, arXiv preprint |

Deliverables: code repo · learning-curve and B80 figures · heterogeneity/moderator tables · paper draft · preprint.

## 12. Change log
- v1.0 (9 Oct 2026): frozen. Reference repo of Gorenshtein et al. has no licence → used as read-only reference; all pipeline code written independently.
- v1.0 clarifications (9 Oct 2026, implementation details only; objective, RQs, hypotheses and outcomes unchanged):
  - **Record unit** = participant × session × condition that has its *own* Train and Test files. Test-only conditions (e.g. dynamic-stopping conditions in D, F, G, H) are reported in the inventory but not analysed.
  - **Character decoding** uses the *mean* (not sum) classifier score over the flashes that lit each candidate, because checkerboard variants light characters an unequal number of times.
  - **R** is defined as the number of times every grid character has been lit; decoding stops at the first flash where all characters reach R.
  - Archive facts confirmed on pilots A_01 and B_01: per-character flag channels give flash membership for every paradigm; StimulusType matches "intended character lit" on 100% of flashes; offline full-calibration accuracy reproduces online accuracy (A_01 0.94 vs 0.92; B_01 SE004 0.93 vs 0.93). Day-4 decoding checkpoint passed on pilots.

- v1.0 checkpoint decisions (9 Oct 2026, after Step 4 inventory; objective, RQs, hypotheses and B80 unchanged):
  - **Eligible cohorts: 14** (A, B, E, I, J, K, L, M, N, O, Q, R, S1, S2; 224 participants — corrected 10 Oct 2026, earlier 228 was an arithmetic slip). Day-7 cohort checkpoint passed; RQ4 kept.
  - **D, F, G, H excluded**: calibration recorded in one condition, tests only in other conditions, so no record has its own Train and Test (record-unit rule). Reported as excluded with reason.
  - **R rule replaced.** The frozen rule ("largest R every eligible cohort's Test data supports") gave R = 0 because E, I, J, K, M, N, O used dynamic stopping in their test phase. Primary analysis now decodes each test character with all flashes the original protocol presented (as-run test length), which reproduces online accuracy on pilots. Test length (median flashes / time per selection) is added as an RQ3 moderator. Secondary analysis: fixed R = 1 across all cohorts.
- Analysis implementation notes (9 Oct 2026, Step 8; RQs, hypotheses and outcomes unchanged):
  - **Heterogeneity model:** the pre-specified two-stage approach is used as primary (per-cohort binomial GLM of correct ~ log k with participant-clustered SEs, then DerSimonian–Laird random effects with Hartung–Knapp CIs and a prediction interval). It is robust with 14 cohorts; validated on synthetic data (recovers a known moderator effect, null for a random one).
  - **RQ4 predictors:** primary = B80 of all other cohorts' participants pooled (as frozen); secondary = median B80 of the other cohorts, added because the pooled budget is dominated by the slowest cohorts.
  - **Secondary R = 1 analysis:** no participant reaches 70% with a single flash per character, so B80 at R = 1 is fully censored; it will be reported via AUC / a lower threshold (decided at the figures step).
- Phase A decision (9 Oct 2026): in the dynamic-stopping cohorts (I, J, K, M, N, and E) offline character accuracy at full calibration falls well below online accuracy, because their tests present few flashes per character. **B80 (character accuracy) stays primary but is interpreted on the valid-decoding subset** (|offline − online| ≤ 0.10). **Co-primary: an AUC budget** (smallest k at which ≥ 80% of participants reach flash-level AUC ≥ 0.75; sensitivity 0.70 / 0.80) across all 14 cohorts, since AUC does not depend on test length. Seconds per calibration character is now derived from the sweep.
- Exploratory analyses (disclosed 10 Oct 2026 after external audit): (1) **artefact rejection rate** was added as a moderator in Phase C, after the pre-specified protocol descriptors were tested and none survived Holm correction; it is reported in the paper as exploratory, and **H3 as pre-specified is not supported**. (2) The **decoding-tolerance restriction** for B80 (|offline − online| ≤ 0.10) was decided after inspecting full-calibration accuracy; including all 14 cohorts leaves the B80 range unchanged (2–30 characters).

