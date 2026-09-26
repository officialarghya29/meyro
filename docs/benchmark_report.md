# MEYRO — Master Benchmark Report

**Investigator:** Arghya Bose (`officialarghya29`)
**Artifact repository:** [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)
**Source of truth:** `experiments/master_benchmark/results.json` (regenerate with
`python scripts/run_full_evaluation.py`)

> **Correction notice.** An earlier version of this report compared trained statistical
> baselines against **randomly initialised** neural networks, and its published numbers were
> not reproducible from the code. Both defects are fixed here. Every neural model is now
> trained on the same calibration windows the statistical baselines are fitted on, and every
> number below is copied from the committed artifact.

---

## 1. Protocol

- **Cohort:** 15 synthetic subjects × 60 days (`SyntheticBenchmarkGenerator`, seed 42).
- **Split:** 21-day calibration per subject, then causal evaluation. 495 evaluation windows,
  14.3% deviation prevalence, window size 7 days.
- **Leakage controls:** subject-isolated splits; backward-looking windows only; population
  standardisation fitted on calibration windows only; no test-set tuning; seed fixed before
  every model construction for reproducible initialisation.
- **Operating point:** precision/recall/F1/FPR reported at the threshold achieving 85%
  sensitivity, so all models are compared at equal recall.
- **Training:** reconstruction autoencoders trained on calibration windows; MEYRO-V1/V2
  trained self-supervised on calibration windows only (memories are per-subject and
  leave-one-window-out during training).

---

## 2. Results

```text
Model                              | AUROC  | AUPRC  | F1     | FPR @ 85% Sens
--------------------------------------------------------------------------------
Personal Baseline (Static)         | 0.9949 | 0.9658 | 0.8971 | 0.0094
Personal Baseline (Adaptive)       | 0.9929 | 0.9534 | 0.8857 | 0.0165
One-Class SVM (per subject)        | 0.9944 | 0.9536 | 0.8905 | 0.0118
MEYRO-V1 (trained, streamed)       | 0.9693 | 0.8441 | 0.8052 | 0.0495
Isolation Forest (per subject)     | 0.9426 | 0.7349 | 0.6321 | 0.1439
Population Baseline (control)      | 0.9394 | 0.7792 | 0.5701 | 0.1934
LSTM Autoencoder (trained)         | 0.9183 | 0.5163 | 0.6740 | 0.1156
TCN Autoencoder (trained)          | 0.9164 | 0.5765 | 0.6667 | 0.1250
Transformer Autoencoder (trained)  | 0.9119 | 0.5438 | 0.5894 | 0.1769
GRU Autoencoder (trained)          | 0.8956 | 0.4438 | 0.6524 | 0.1297
MEYRO-V2 (trained, streamed)       | 0.8649 | 0.6589 | 0.5571 | 0.2052
MEYRO-V2 (untrained reference)     | 0.6562 | 0.2600 | 0.3202 | 0.5873
```

---

## 3. Findings

1. **Personalization is the dominant effect.** At matched 85% sensitivity the population
   baseline's false-alarm rate is `0.1934` versus `0.0094` for the static personal baseline —
   a ~20.6× reduction. This supports the core hypothesis that personal baselines reduce false
   positives relative to population thresholds.
2. **Training is decisive for neural models.** MEYRO-V2 rises from `0.6562` (untrained) to
   `0.8649` (trained). Any comparison against an untrained network measures initialization,
   not architecture.
3. **Reconstruction-error baselines are the weakest family** (`0.8956`–`0.9183`), consistent
   with the expectation that a global reconstruction manifold conflates inter-subject
   variation with intra-subject deviation.
4. **The learned MEYRO models do not beat the statistical personal baseline.** MEYRO-V1
   reaches `0.9693` versus `0.9949` for the robust static baseline, and MEYRO-V2 `0.8649`.
   This is a negative result for the neural architecture on this cohort and is reported as
   such. A plausible explanation is that the synthetic deviation is a sustained mean shift,
   which robust per-subject statistics model almost optimally, leaving little for a learned
   memory to add.
5. **Dual-timescale memory gives no static-detection gain** (see
   `experiments/ablation/results.json`: single-timescale `0.8981` vs full `0.8913`). Its
   demonstrated contribution is drift adaptation (`experiments/drift/results.json`:
   persistent false alarms `61.5%` static → `2.2%` MEYRO-V2).

---

## 4. Threats to validity

- **Synthetic data.** Deviation structure is injected by the generator; real physiological
  drift is unlikely to be this well-behaved.
- **Single seed, one cohort size.** Deltas below roughly 0.01 AUROC should be treated as
  within noise.
- **Cross-sectional generalization is untested.** The benchmark measures within-subject
  detection only; there is no held-out-subject evaluation yet.
- **No confidence intervals or significance tests** are reported in this iteration.

---

## 5. Reproduction

```bash
python scripts/run_full_evaluation.py      # writes experiments/*/results.json
python scripts/generate_figures.py         # writes assets/*.png from those artifacts
```
