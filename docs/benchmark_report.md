# MEYRO — Master Benchmark Report

**Investigator:** Arghya Bose (`officialarghya29`)
**Source of truth:** `experiments/*/results.json` (regenerate with `python scripts/run_full_evaluation.py`)

> **Correction notice.** Two defects invalidated earlier versions of this report.
> (1) The comparison evaluated MEYRO and the deep baselines at **random initialization**
> against trained statistical baselines. (2) The drift suite varied between identical runs
> because multi-threaded CPU recurrent kernels reduce nondeterministically — the same
> configuration produced a 2.2% and a 93.0% persistent-false-alarm rate on separate runs.
>
> Both are fixed. Every neural model is trained on the same calibration windows the
> statistical baselines use, determinism is enforced, and every number below is read from a
> committed artifact. The corrected drift result **reverses** the earlier claim.

---

## 1. Protocol

- **Cohort:** 15 synthetic subjects × 60 days (seed 42); 495 evaluation windows; 14.34% prevalence.
- **Split:** 21-day calibration per subject, then causal evaluation; 7-day backward windows.
- **Leakage controls:** subject-isolated splits; backward-only windows; population
  standardization fitted on calibration only; no test-set tuning.
- **Training:** all neural models trained on calibration windows; 80 epochs applied identically
  to every neural model (budget chosen from the diagnosis below, not tuned on the test set).
- **Operating point:** threshold-dependent metrics at 85% sensitivity, so models are compared
  at equal recall.
- **Determinism:** `torch.set_num_threads(1)` + seeded construction; a regression test asserts
  bit-for-bit reproducibility of the drift suite.

---

## 2. Master comparison

```text
Model                              | AUROC  | AUPRC  | F1     | FPR @ 85% Sens
--------------------------------------------------------------------------------
Personal Baseline (Static)         | 0.9949 | 0.9658 | 0.8971 | 0.0094
One-Class SVM (per subject)        | 0.9944 | 0.9536 | 0.8905 | 0.0118
Personal Baseline (Adaptive)       | 0.9929 | 0.9534 | 0.8857 | 0.0165
MEYRO-V1 (trained, streamed)       | 0.9680 | 0.8169 | 0.7722 | 0.0613
Isolation Forest (per subject)     | 0.9426 | 0.7349 | 0.6321 | 0.1439
Population Baseline (control)      | 0.9394 | 0.7792 | 0.5701 | 0.1934
MEYRO-V2 (trained, streamed)       | 0.9179 | 0.6483 | 0.6778 | 0.1132
TCN Autoencoder (trained)          | 0.9172 | 0.5561 | 0.6354 | 0.1415
LSTM Autoencoder (trained)         | 0.9166 | 0.5281 | 0.6813 | 0.1156
Transformer Autoencoder (trained)  | 0.9123 | 0.5379 | 0.6100 | 0.1604
GRU Autoencoder (trained)          | 0.9107 | 0.5096 | 0.6739 | 0.1203
MEYRO-V2 (untrained reference)     | 0.6142 | 0.2658 | 0.2857 | 0.6958
```

---

## 3. Findings

1. **Personalization is the dominant effect.** `0.1934 → 0.0094` false-alarm rate at matched
   sensitivity (~20.6×). This supports the primary hypothesis.
2. **The result is statistically significant per subject.** Across 3 seeds (25 subject-seeds):
   population `0.9356` `[0.8877, 0.9716]` → personalized `0.9995` `[0.9985, 1.0000]`;
   paired Wilcoxon **p = 6.55e-04**, win rate `0.60`, and personalization never loses.
3. **Training is decisive.** MEYRO-V2: `0.6142` untrained → `0.9179` trained.
4. **Reconstruction baselines are weakest** (`0.9107`–`0.9172`), consistent with a global
   reconstruction manifold conflating between-person variation with within-person deviation.
5. **The learned models do not beat the statistical control** (`0.9680`/`0.9179` vs `0.9949`).
6. **Held-out subjects generalize:** MEYRO-V2 scores `0.9095` on 6 unseen subjects versus
   `0.8697` within-subject (gap `−0.0398`), so the deficit is not subject memorization.

---

## 4. Ablation (trained MEYRO-V2, streamed)

| Variant | AUROC | Δ vs Full | F1 |
|---|---|---|---|
| Full | `0.8621` | — | `0.6497` |
| w/o quality conditioning | `0.8621` | `0.0000` | `0.6582` |
| Single-timescale memory | `0.8584` | `−0.0037` | `0.6415` |
| w/o context encoder | `0.8503` | `−0.0118` | `0.6415` |
| + persistence gating | `0.8404` | `−0.0217` | `0.6375` |
| Naive Euclidean deviation | `0.8193` | `−0.0428` | `0.4615` |
| w/o personal memory | `0.8003` | `−0.0618` | `0.4064` |
| Untrained reference | `0.6760` | `−0.1861` | `0.4416` |

Personal memory is the largest architectural contribution (`+0.062` AUROC, `+0.24` F1) and the
learned relational deviation module beats Euclidean distance (`+0.043`). Persistence gating
*degrades* detection and is therefore reported rather than applied to the score.

---

## 5. Diagnosing the neural gap

| Hypothesis | Test | Verdict |
|---|---|---|
| Memory adaptation absorbs deviations | frozen vs streaming memory | **Falsified** (`+0.0008` AUROC) |
| Under-training | 30 / 60 / 120 epochs | **Confirmed**: `+0.0397` by 120 epochs |
| Subject overfitting | held-out subjects | **Falsified** (unseen ≥ within) |
| Residual cause | — | **Unresolved** (`0.0903` AUROC) |

The most plausible remaining explanation is that the injected deviation is a sustained mean
shift, which robust per-subject statistics model almost optimally. Testing this requires real
longitudinal data, not a new architecture.

---

## 6. Baseline drift — failure

| Method | Persistent false alarms | Days to final alert |
|---|---|---|
| Personal Baseline (Adaptive) | `30.0%` | `41.0` |
| Personal Baseline (Static) | `61.5%` | `41.0` |
| MEYRO-V2 (streamed) | **`93.0%`** | `41.0` |

MEYRO-V2 is the **worst** performer. Its gate `exp(−4·A_t)` suppresses adaptation exactly when a
deviation is flagged, so a sustained change is alerted on indefinitely. This is the mirror image
of the failure the gate was designed to prevent. The statistical adaptive baseline (a `z < 2.5`
criterion) adapts correctly. Fix proposed, not implemented.

---

## 7. Threats to validity

- **Synthetic data with one deviation type** (sustained mean shift). Real physiology is messier.
- **One architecture size and window length**; no capacity sweep.
- **Subgroup/fairness analysis impossible** with the available (synthetic) metadata.
- **Latency is single-threaded CPU** and not a deployment figure.

---

## 8. Reproduction

```bash
python scripts/run_full_evaluation.py      # 9 suites -> experiments/*/results.json
python scripts/generate_figures.py         # figures  -> assets/*.png
pytest -q                                  # 63 tests
```
