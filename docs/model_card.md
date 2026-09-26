# Model Card — MEYRO

**Model family:** MEYRO (Personalized Longitudinal Health Baseline Modeling)
**Versions:** `MEYRO-V1` (single-memory reference), `MEYRO-V2` (current: dual-timescale memory + persistence)
**Author:** Arghya Bose (`officialarghya29`)
**Repository:** [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)
**License:** MIT

> Every metric is copied from `experiments/*/results.json`, written by
> `scripts/run_full_evaluation.py`. Rerun it to reproduce. No metric was typed by hand.
>
> **Revision note.** An earlier version of this card reported MEYRO-V2 achieving a 2.2%
> drift false-alarm rate and beating the statistical baseline. Both were artefacts: the
> benchmark evaluated untrained networks, and the drift suite varied between identical runs
> because multi-threaded CPU recurrent kernels are nondeterministic. Determinism is now
> enforced and the results below are reproducible; the corrected drift result is a failure.

---

## 1. Intended use

- **Intended:** longitudinal deviation monitoring against a personal baseline; change-point
  observation; baseline-support and data-quality reporting.
- **Out of scope:** diagnosis, clinical decision-making, or any statement of health status.
  MEYRO never outputs "you have disease X" or "you are healthy".

---

## 2. Architectures

**MEYRO-V1** — context-conditioned causal GRU encoder; single latent personal baseline memory;
relational deviation module; anomaly and uncertainty heads; anomaly-gated memory update.

**MEYRO-V2** — adds a **dual-timescale personal memory** (fast `η_f=0.35`, slow `η_s=0.03`)
and a **persistence module** (causal GRU over the within-window deviation trajectory).

Memory update: `gate = exp(−4·A_t)·mean(Q_t)`, `B ← (1 − η·gate)·B + η·gate·E_t`.
A single outlier barely moves the baseline — and, as the drift experiment shows, neither is
a sustained change absorbed.

---

## 3. Evaluation protocol

- **Data:** seeded synthetic cohort, 15 subjects × 60 days; 495 evaluation windows; 14.34%
  prevalence; 7-day windows.
- **Splits:** strictly causal. 21-day calibration per subject, then evaluation.
- **Normalization:** population z-score fitted on **calibration windows only**. Deliberately
  not per-subject — per-subject statistics encode the personal baseline into the input and
  confound every personalization ablation.
- **Training:** all neural models trained on the same calibration windows the statistical
  baselines are fitted on; 80 epochs applied identically to every neural model.
- **Operating point:** threshold-dependent metrics at the threshold reaching 85% sensitivity.
- **Determinism:** single-threaded torch, seeded before each model construction.

---

## 4. Results

### 4.1 Master comparison (15 subjects × 60 days, 495 windows)

| Model | AUROC | AUPRC | F1 | FPR @ 85% Sens |
|---|---|---|---|---|
| Personal Baseline (Static) | `0.9949` | `0.9658` | `0.8971` | `0.0094` |
| One-Class SVM (per subject) | `0.9944` | `0.9536` | `0.8905` | `0.0118` |
| Personal Baseline (Adaptive) | `0.9929` | `0.9534` | `0.8857` | `0.0165` |
| MEYRO-V1 (trained, streamed) | `0.9680` | `0.8169` | `0.7722` | `0.0613` |
| Isolation Forest (per subject) | `0.9426` | `0.7349` | `0.6321` | `0.1439` |
| Population Baseline (control) | `0.9394` | `0.7792` | `0.5701` | `0.1934` |
| MEYRO-V2 (trained, streamed) | `0.9179` | `0.6483` | `0.6778` | `0.1132` |
| TCN Autoencoder (trained) | `0.9172` | `0.5561` | `0.6354` | `0.1415` |
| LSTM Autoencoder (trained) | `0.9166` | `0.5281` | `0.6813` | `0.1156` |
| Transformer Autoencoder (trained) | `0.9123` | `0.5379` | `0.6100` | `0.1604` |
| GRU Autoencoder (trained) | `0.9107` | `0.5096` | `0.6739` | `0.1203` |
| MEYRO-V2 (untrained reference) | `0.6142` | `0.2658` | `0.2857` | `0.6958` |

Personalization reduces the false-alarm rate ~**20.6×** at matched sensitivity.

### 4.2 Statistical significance (subject as unit; 3 seeds, 25 subject-seeds)

| Method | Per-subject AUROC (95% CI) | Pooled FPR @ 85% |
|---|---|---|
| Personal Baseline (Adaptive) | `1.0000` `[1.0000, 1.0000]` | `0.0085` |
| Personal Baseline (Static) | `0.9995` `[0.9985, 1.0000]` | `0.0061` |
| Population Baseline | `0.9356` `[0.8877, 0.9716]` | `0.1586` |

Paired Wilcoxon, personalized vs population: **p = 6.55e-04**, median Δ `+0.0185`, win rate `0.60`.
Adaptive vs static (both personalized): p = `0.3173` — a null result.

### 4.3 Ablation (streamed, trained MEYRO-V2, 60 epochs)

| Variant | AUROC | Δ | F1 |
|---|---|---|---|
| Full (trained) | `0.8621` | — | `0.6497` |
| w/o quality conditioning | `0.8621` | `0.0000` | `0.6582` |
| Single-timescale memory | `0.8584` | `−0.0037` | `0.6415` |
| w/o context encoder | `0.8503` | `−0.0118` | `0.6415` |
| + persistence gating | `0.8404` | `−0.0217` | `0.6375` |
| Naive Euclidean deviation | `0.8193` | `−0.0428` | `0.4615` |
| w/o personal memory | `0.8003` | `−0.0618` | `0.4064` |
| Untrained reference | `0.6760` | `−0.1861` | `0.4416` |

Personal memory is the largest architectural contributor (`+0.062`); training is worth `+0.186`.

### 4.4 Diagnosing the gap to the statistical control

Freezing the memory changes AUROC by `+0.0008`, which **falsifies** the hypothesis that memory
adaptation blurs detection. Training budget does matter (`+0.0397` from 30 → 120 epochs), so the
benchmark budget was raised to 80 for all neural models. A residual gap of `0.0903` AUROC remains
and is unresolved.

### 4.5 Held-out subjects (14 train / 6 held out)

| Configuration | AUROC | F1 | FPR @ 85% |
|---|---|---|---|
| MEYRO-V2 within-subject | `0.8697` | — | — |
| MEYRO-V2 unseen subjects | `0.9095` | `0.6444` | `0.1697` |
| Statistical baseline unseen subjects | `0.9993` | `0.9355` | `0.0000` |

Generalization gap `−0.0398` AUROC: no subject overfitting.

### 4.6 Baseline drift — **failure**

| Method | Persistent false alarms | Days to final alert |
|---|---|---|
| Personal Baseline (Adaptive) | **`30.0%`** | `41.0` |
| Personal Baseline (Static) | `61.5%` | `41.0` |
| MEYRO-V2 (streamed) | **`93.0%`** — worst | `41.0` |

Cause: the gate `exp(−4·A_t)` suppresses adaptation exactly when a deviation is flagged, so a
sustained change is alerted on indefinitely. Acute-outlier probe: max baseline displacement
`0.43 σ` from one 3× observation — MEYRO under-adapts to sustained change, it does not over-adapt to spikes.

### 4.7 Robustness (personal statistical baseline, AUROC)

| Axis | Levels | AUROC |
|---|---|---|
| Missing data | 0% → 30% | `0.9968` → `0.9750` |
| Sensor noise | 1× → 4× | `0.9974` → `0.9859` |
| Spike outliers | 1% → 5% | `0.9907` → `0.9551` |
| History length | 7 → 28 days | `0.9813` → `0.9982` |
| Sampling rate | 1× → 1/3× | `0.9968` → `0.9983` |
| Device bias | 0% → 15% | `0.9968` → `0.9968` |

### 4.8 Cold start

| History | Population | Personalized | Advantage |
|---|---|---|---|
| 3 days | `0.9256` | `0.9335` | +0.0079 |
| 7 days | `0.9395` | `0.9417` | +0.0022 |
| 14 days | `0.9431` | `0.9828` | +0.0397 |
| 21 days | `0.9522` | `0.9946` | +0.0424 |
| 28 days | `0.9494` | `0.9954` | +0.0460 |

Recommended minimum baseline collection: **~14 days**.

### 4.9 Efficiency (CPU, 64-window batch, single-threaded)

| Model | Parameters | Latency (ms) | Size (KB) |
|---|---|---|---|
| MEYRO-V1 | `5045` | `0.3938` | `19.71` |
| MEYRO-V2 | `8966` | `0.8710` | `35.02` |

---

## 5. Known limitations & failure cases

1. **Synthetic data only.** No claim transfers to real physiology until evaluated on a licensed
   longitudinal dataset (GLOBEM requires credentialed access).
2. **Learned models lose to the statistical personal baseline** (`0.9680`/`0.9179` vs `0.9949`).
   Diagnosed as not adaptation and not under-training; residual `0.0903` gap unexplained.
3. **Drift adaptation fails** (`93.0%` persistent false alarms). Mechanism identified, fix not implemented.
4. **Persistence gating degrades detection** (`−0.0217`) and is reported rather than applied.
5. **Dual-timescale memory gains only `+0.0037` AUROC** — within noise here.
6. **Single configuration family:** one hidden width, window size, and deviation type.
7. **No fairness/subgroup analysis** is possible or claimed with the available metadata.
8. **Latency unoptimised** and single-threaded for determinism.

---

## 6. Ethical & privacy notes

- Output is restricted to deviation language; diagnostic statements are prohibited by design.
- No personal or health data is committed (`.gitignore` blocks `data/**`).
- Checkpoints store provenance metadata (config, seed, git commit) but no subject data.
- Regulatory compliance (HIPAA/GDPR/DPDP) has **not** been assessed and is not claimed.
