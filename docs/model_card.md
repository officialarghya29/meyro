# Model Card — MEYRO-V2

**Model name:** MEYRO (Personalized Longitudinal Health Baseline Modeling)
**Versions:** `MEYRO-V1` (single-memory reference) and `MEYRO-V2` (current: dual-timescale memory + persistence)
**Author:** Arghya Bose (`officialarghya29`)
**Repository:** [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)
**License:** MIT

> Every metric in this card is copied from `experiments/*/results.json`, produced by
> `scripts/run_full_evaluation.py`. Rerun that command to reproduce them. No metric here
> was typed by hand or estimated.

---

## 1. Overview

MEYRO models an individual's own long-term behavioural/physiological normal and reports
deviations from *that* baseline, rather than comparing a person against population
thresholds. It outputs a deviation score, an uncertainty estimate, a persistence score and
a data-quality figure — never a diagnosis.

---

## 2. Intended use

- **Intended:** longitudinal deviation monitoring against a personal baseline; change-point
  observation; data quality and baseline-support reporting.
- **Out of scope:** disease diagnosis, clinical decision-making, or any statement of
  health status. MEYRO never outputs "you have disease X" or "you are healthy."

---

## 3. Architectures

**MEYRO-V1** — context-conditioned causal GRU encoder; single latent personal baseline
memory; relational deviation module over `[E, B, E−B, E⊙B, Q]`; anomaly and uncertainty
heads; anomaly-gated memory update.

**MEYRO-V2** — adds two mechanisms addressing gaps in the baselines:

1. **Dual-timescale personal memory.** A *fast* memory (`fast_rate = 0.35`) tracks recent
   personal state; a *slow* memory (`slow_rate = 0.03`) holds the established baseline.
   Both update with `α = rate · exp(−4·A_t) · mean(Q_t)`, so a high-scoring observation
   barely moves either memory.
2. **Persistence module.** A causal GRU over the within-window deviation trajectory,
   separating a one-off excursion from a sustained shift.

---

## 4. Evaluation protocol

- **Data:** synthetic longitudinal cohort (`SyntheticBenchmarkGenerator`), 15 subjects ×
  60 days for the benchmark; 495 evaluation windows; 14.3% deviation prevalence.
- **Splits:** strictly causal — a 21-day calibration window per subject, then evaluation.
  Windows are backward-looking; calibration and evaluation never overlap.
- **Normalization:** population-level z-score standardisation fitted on **calibration
  windows only**. Deliberately not per-subject: per-subject statistics would encode the
  personal baseline into the input and confound every personalization ablation.
- **Operating point:** threshold-dependent metrics are reported at the threshold reaching
  85% sensitivity, so models are compared at equal recall.
- **Training:** all neural models are trained on calibration windows on which the
  statistical baselines are fitted — no model is evaluated at random initialization.

---

## 5. Results

### 5.1 Master model comparison

| Model | AUROC | AUPRC | F1 | FPR @ 85% Sens |
|---|---|---|---|---|
| Personal Baseline (Static) | `0.9949` | `0.9658` | `0.8971` | `0.0094` |
| Personal Baseline (Adaptive) | `0.9929` | `0.9534` | `0.8857` | `0.0165` |
| One-Class SVM (per subject) | `0.9944` | `0.9536` | `0.8905` | `0.0118` |
| MEYRO-V1 (trained, streamed) | `0.9693` | `0.8441` | `0.8052` | `0.0495` |
| Isolation Forest (per subject) | `0.9426` | `0.7349` | `0.6321` | `0.1439` |
| Population Baseline (control) | `0.9394` | `0.7792` | `0.5701` | `0.1934` |
| LSTM Autoencoder (trained) | `0.9183` | `0.5163` | `0.6740` | `0.1156` |
| TCN Autoencoder (trained) | `0.9164` | `0.5765` | `0.6667` | `0.1250` |
| Transformer Autoencoder (trained) | `0.9119` | `0.5438` | `0.5894` | `0.1769` |
| GRU Autoencoder (trained) | `0.8956` | `0.4438` | `0.6524` | `0.1297` |
| MEYRO-V2 (trained, streamed) | `0.8649` | `0.6589` | `0.5571` | `0.2052` |
| MEYRO-V2 (untrained reference) | `0.6562` | `0.2600` | `0.3202` | `0.5873` |

### 5.2 Ablation (streamed, trained MEYRO-V2)

| Variant | AUROC |
|---|---|
| Full model | `0.8913` |
| + persistence gating | `0.9016` |
| Single-timescale memory | `0.8981` |
| w/o quality conditioning | `0.8911` |
| w/o personal memory | `0.8855` |
| Naive Euclidean deviation | `0.8854` |
| w/o context encoder | `0.8589` |
| Untrained reference | `0.7202` |

### 5.3 Robustness (personal statistical baseline, AUROC)

| Axis | Levels | AUROC |
|---|---|---|
| Missing data | 0% → 30% | `0.9968` → `0.9750` |
| Sensor noise | 1× → 4× | `0.9974` → `0.9859` |
| Spike outliers | 1% → 5% | `0.9907` → `0.9551` |
| History length | 7 → 28 days | `0.9813` → `0.9982` |
| Sampling rate | 1× → 1/3× | `0.9968` → `0.9983` |
| Device bias | 0% → 15% | `0.9968` → `0.9968` |

### 5.4 Cold start

| History | Population | Personalized | Advantage |
|---|---|---|---|
| 3 days | `0.9256` | `0.9335` | +0.0079 |
| 7 days | `0.9395` | `0.9417` | +0.0022 |
| 14 days | `0.9431` | `0.9828` | +0.0397 |
| 21 days | `0.9522` | `0.9946` | +0.0424 |
| 28 days | `0.9494` | `0.9954` | +0.0460 |

Minimum recommended baseline collection: **~14 days**.

### 5.5 Baseline drift

| Method | Persistent false alarms post-drift | Days to final alert |
|---|---|---|
| Personal Baseline (Static) | `61.5%` | `41.0` |
| Personal Baseline (Adaptive) | `30.0%` | `41.0` |
| MEYRO-V2 (streamed) | `2.2%` | `39.0` |

Acute-outlier probe: max baseline displacement `0.43 σ` from a single 3× observation.

### 5.6 Efficiency (CPU, 64-window batch)

| Model | Parameters | Latency (ms) | Size (KB) |
|---|---|---|---|
| MEYRO-V1 | `5045` | `0.3609` | `19.71` |
| MEYRO-V2 | `8966` | `0.9252` | `35.02` |

---

## 6. Known limitations & failure cases

1. **Synthetic data only.** No claim transfers to real physiology until run on a licensed
   longitudinal dataset. GLOBEM requires credentialed access.
2. **Learned models do not beat the statistical personal baseline** on this cohort
   (`0.9693` for V1 vs `0.9949` for the robust static baseline). The statistical control is
   strong and is not treated as a straw man.
3. **Dual-timescale memory gives no static-detection gain** (`0.8981` single-timescale vs
   `0.8913` full, i.e. slightly negative). Its measured benefit is drift adaptation.
4. **Single seed, small cohorts.** 6–15 subjects per suite, one generator seed. No
   confidence intervals or multi-seed variance yet. Treat all deltas below ~0.01 AUROC as
   within noise.
5. **Latency unoptimised.** MEYRO-V2 is the slowest model measured (~2.6× MEYRO-V1).
6. **No fairness/subgroup analysis** has been performed; subgroup claims would be
   unsupported.

---

## 7. Ethical & privacy notes

- MEYRO must not output diagnostic statements; the safety layer enforces
  deviation-language only.
- No personally identifiable or health data is committed to the repository
  (`.gitignore` blocks `data/**`).
- Model checkpoints and result artifacts exclude any personal data; checkpoints store
  provenance metadata (config, seed, git commit) but no subject data.
- Regulatory compliance (HIPAA/GDPR/DPDP) has **not** been formally assessed and is not
  claimed.
