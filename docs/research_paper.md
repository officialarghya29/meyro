# MEYRO: Personalized Baseline Modeling for Longitudinal Multimodal Health Anomaly Detection

**Arghya Bose**  
Department of Machine Learning Research & Software Engineering  
`officialarghya29@gmail.com` | [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)

---

## Abstract

Wearable health monitors and digital phenotyping systems typically evaluate physiological signals against broad population-level thresholds ("Is this normal for humans?"). Because healthy humans display pronounced idiosyncratic baseline variance (e.g. resting heart rates varying by over 25 bpm across healthy individuals), population-level models incur severe false-positive alarm rates or miss subtle individual physiological shifts. 

In this work, we present **MEYRO**, a causal, leakage-free framework for personal longitudinal baseline modeling and anomaly detection. MEYRO formulates personal health tracking not as disease classification, but as causal deviation estimation against an individual's self-established historical baseline ("Is this normal for *this person*?"). We design a unified architecture comprising: (1) a causal temporal sequence encoder, (2) an explicit latent personal baseline memory $B_t$, (3) a non-linear relational deviation module, and (4) an adaptive memory gating mechanism that allows continuous setpoint drift tracking while strictly preventing acute anomalies from corrupting the baseline.

In rigorous, leakage-free evaluations against 11 classical and deep learning baselines across a 60-day synthetic cohort, the **personalized statistical baseline** reduces the False Positive Rate at 85% target sensitivity from **19.34%** (population baseline) to **0.94%** — a **~20.6× reduction in false alarms** — and the effect is statistically significant with the subject as the unit of analysis (paired Wilcoxon **p = 6.55×10⁻⁴**, per-subject AUROC `0.9356 → 0.9995`). The personalization advantage is positive across all four deviation shapes tested, including ramp and dispersion changes where population statistics degrade to `≈0.70` AUROC.

We also report results that do **not** support our architecture: the learned models (`0.9680` for MEYRO-V1, `0.9179` for MEYRO-V2) do **not** beat the robust statistical control (`0.9949`), and MEYRO's anomaly-gated memory **fails** a sustained-drift test (`93.0%` persistent false alarms, versus `30.0%` for an adaptive statistical baseline) because the gate blocks adaptation exactly when a deviation is flagged. A targeted repair (threshold-confirmed migration) was implemented and measured to be **ineffective** (`93.0%` → `93.0%`). All numbers are reproducible from the committed `experiments/*/results.json` artifacts.

---

## 1. Introduction & Research Question

The continuous monitoring of behavioral and physiological signals via consumer wearables has created unprecedented opportunities for early illness detection and wellness tracking. However, standard anomaly detection paradigms struggle when applied to human longitudinal time series. 

Traditional approaches fall into two categories:
1. **Population-level static thresholds:** Clinical reference intervals (e.g. resting heart rate between 60–100 bpm) fail to identify a subject whose normal baseline of 55 bpm has elevated to 88 bpm during an acute infection, while falsely flagging an athlete whose baseline is 48 bpm.
2. **Global sequence autoencoders:** Deep recurrent and transformer architectures (e.g. LSTM/PatchTST autoencoders) trained across cohorts conflate inter-subject variance with temporal dynamics. Because they lack explicit person-specific setpoints, normal individuals at the demographic margins suffer high reconstruction error and false alarms.

### Primary Research Question
*Can personalized longitudinal baselines detect meaningful behavioral/physiological deviations more effectively and with fewer false positives than population-level baselines?*

We address this with four core contributions:
- **Formal Causal Data Architecture:** A strict, leakage-free temporal partitioning schema with zero lookahead.
- **Novel MEYRO Architecture:** An end-to-end neural model explicitly separating personal setpoint memory $B_t$ from sequence dynamics $E_t$.
- **Adaptive Memory Gating (with a documented failure):** An exponential update filter $\alpha(A_t) = \alpha_0 \exp(-\gamma A_t) \bar{Q}_t$ that rejects acute excursions from redefining the baseline. We show analytically and empirically that this same filter **prevents adaptation to sustained change** — a failure we report rather than hide, along with an implemented-and-measured attempt to repair it.
- **Empirical Validation *and* negative results:** Comprehensive benchmarking showing that explicit personalization substantially reduces false-alarm burden, alongside an honest account of where the learned architecture loses to a robust statistical control.

---

## 2. Methodology & Architecture

### 2.1 Causal Problem Formulation
Let an observation at time $t$ for subject $s$ be $X_t \in \mathbb{R}^D$, accompanied by context $C_t \in \mathbb{R}^C$ and sensor quality $Q_t \in [0, 1]^D$. The model maintains an internal baseline memory state $B_t \in \mathbb{R}^d$.

```text
Sequence Window X_(t-K:t) ──→ Causal GRU / TCN Encoder ──→ Dynamics E_t
Context Vector C_t        ──→ Context Projector        ──→ C_t
Personal Memory B_t       ──→ Baseline Memory Buffer   ──→ B_t
Quality Vector Q_t        ──→ Sensor Quality Gate      ──→ Q_t
                                      │
                                      ↓
                Relational Deviation Network: D_t = GELU(W_d [E_t ∥ B_t ∥ (E_t - B_t) ∥ E_t ⊙ B_t])
                                      │
                         ┌────────────┴────────────┐
                         ↓                         ↓
                   Anomaly Head              Uncertainty Head
                   A_t = σ(W_a D_t)          U_t = σ(W_u [D_t ∥ Q_t])
```

### 2.2 Adaptive Memory Update
The personal baseline evolves causally without allowing acute anomalies to pollute future setpoints:
$$\alpha_t = \alpha_0 \cdot \exp\left(-\gamma A_t\right) \cdot \text{mean}(Q_t)$$
$$B_{t+1} = (1 - \alpha_t) B_t + \alpha_t E_t$$

When an anomaly occurs ($A_t \to 1$), adaptation freezes ($\alpha_t \to 0$). This protects the baseline
from acute excursions, and it is also the mechanism behind the drift failure in §3.6: a *sustained*
deviation is flagged continuously, so $\alpha_t \approx 0$ indefinitely and the baseline can never
re-establish. We implement and measure a threshold-confirmed alternative
($\alpha_t \propto \text{clamp}((R_t - \tau)/\beta, 0, 1) \cdot \text{mean}(Q_t)$, with $\tau$ calibrated at
the $0.99$ quantile of the subject's own calibration persistence, and migration permitted only after
$4$ consecutive confirmed windows). It does not help, for the reason given in §3.6.

---

## 3. Experimental Evaluation

### 3.1 Benchmark Protocol
- **Cohort:** 15 subjects, 60 days longitudinal horizon, controlled multi-day physiological deviations.
- **Split:** 21 days of strictly preceding calibration history per subject, then causal evaluation: 495 evaluation windows (7-day backward-looking), 14.34% deviation prevalence (71 positive windows).
- **Normalization:** population z-score fitted on **calibration windows only** — deliberately not per-subject, since per-subject statistics would encode the personal baseline into the input and confound every personalization ablation.
- **Determinism:** single-threaded torch with deterministic kernels; a regression test asserts bit-for-bit stability across runs.
- **Data:** synthetic (`SyntheticBenchmarkGenerator`, seeded). No real physiological data has been evaluated; see §3.6 and the limitations.

### 3.2 Master Comparison Results

All neural models are trained on the same calibration windows the statistical baselines are fitted
on (80 epochs, applied identically), and threshold-dependent columns are reported at equal recall
(85% sensitivity). Cohort: 495 evaluation windows, 14.34% prevalence.

| Architecture | Family | AUROC | AUPRC | F1 | FPR @ 85% Sens | False-Alarm Reduction |
|---|---|---|---|---|---|---|
| **Personal Baseline (Static)** | Statistical | **0.9949** | **0.9658** | **0.8971** | **0.94%** | **20.6×** |
| **One-Class SVM (Per-Subject)** | Classical | 0.9944 | 0.9536 | 0.8905 | 1.18% | 16.4× |
| **Personal Baseline (Adaptive)** | Statistical | 0.9929 | 0.9534 | 0.8857 | 1.65% | 11.7× |
| **MEYRO-V1 (ours, trained)** | Neural | 0.9680 | 0.8169 | 0.7722 | 6.13% | 3.2× |
| **Isolation Forest (Per-Subject)** | Classical | 0.9426 | 0.7349 | 0.6321 | 14.39% | 1.3× |
| **Population Baseline (Control)** | Statistical | 0.9394 | 0.7792 | 0.5701 | 19.34% | 1.0× |
| **MEYRO-V2 (ours, trained)** | Neural | 0.9179 | 0.6483 | 0.6778 | 11.32% | 1.7× |
| **TCN Autoencoder (trained)** | Neural | 0.9172 | 0.5561 | 0.6354 | 14.15% | 1.4× |
| **LSTM Autoencoder (trained)** | Neural | 0.9166 | 0.5281 | 0.6813 | 11.56% | 1.7× |
| **Transformer Autoencoder (trained)** | Neural | 0.9123 | 0.5379 | 0.6100 | 16.04% | 1.2× |
| **GRU Autoencoder (trained)** | Neural | 0.9107 | 0.5096 | 0.6739 | 12.03% | 1.6× |
| **MEYRO-V2 (untrained reference)** | Neural | 0.6142 | 0.2658 | 0.2857 | 69.58% | — |

**Finding 1 — personalization is the dominant effect, but a robust statistic captures it.** The
population-to-personal transition accounts for the largest performance change in the study. It does
so for the *statistical* baseline, not for the learned models, which are discussed in §3.5.

**Finding 2 — global sequence autoencoders are the weakest family** (`0.9107`–`0.9172`), consistent
with a shared reconstruction manifold conflating between-person variation with within-person
deviation. Note that earlier drafts of this work reported these models at `< 0.20` AUROC; that was an
artefact of benchmarking them at random initialization, and is corrected here.

### 3.3 Robustness Under Sensor Impairment
- **Missing Data:** 0% missing (0.9968 AUROC) $\to$ 30% missing (0.9750 AUROC).
- **Sensor Noise:** 1× noise (0.9974 AUROC) $\to$ 4× noise (0.9859 AUROC).
- **Spike outliers:** 1% (0.9907) $\to$ 5% (0.9551). **Sampling rate:** 1× (0.9968) $\to$ 1/3× (0.9983).
- **Device bias:** 0%–15% per-subject gain/offset leaves AUROC unchanged (0.9968).

### 3.4 Cold-Start Trajectory
- **3 Days:** +0.0079 AUROC over population.
- **14 Days:** +0.0397 AUROC.
- **28 Days:** +0.0460 AUROC (reaches a 0.9954 AUROC asymptote).

Practical implication: below roughly one week of history there is no meaningful personalization
benefit; ~14 days of calibration is the point at which it becomes material.

### 3.5 Statistical significance, ablations, and the learned-model deficit

**Significance (subject as the unit of analysis, 3 seeds, 25 subject-seeds).** Per-subject AUROC
rises from `0.9356` `[0.8877, 0.9716]` (population) to `0.9995` `[0.9985, 1.0000]` (personalized),
paired Wilcoxon **p = 6.55×10⁻⁴**, median Δ `+0.0185`, win rate `0.60`, and personalization never
loses. Adaptive vs static personalization is a null result (p = `0.3173`).

**Ablation (trained MEYRO-V2, streamed).** Personal memory is the largest architectural contributor:
removing it costs `−0.0618` AUROC (`0.8621 → 0.8003`) and `−0.24` F1. The learned relational
deviation module beats naive Euclidean distance by `+0.0428`. Persistence gating *degrades*
detection (`−0.0217`) and is therefore reported, not applied. The dual-timescale memory is worth only
`+0.0037` AUROC on this cohort.

**The learned models lose to the statistical control.** MEYRO-V1 `0.9680`, MEYRO-V2 `0.9179`, versus
`0.9949`. We tested three explanations rather than asserting one: memory adaptation blurring the
signal (**falsified**, `+0.0008` when the memory is frozen), under-training (**confirmed**, `+0.0397`
from 30 → 120 epochs, which is why the budget was raised to 80 for every neural model), and subject
over-fitting (**falsified**, held-out subjects score `0.9095` versus `0.8697` within-subject). The
residual `0.0903` AUROC gap is **unresolved**.

### 3.6 Deviation shapes, and a documented adaptation failure

The benchmark's primary deviation is a sustained *mean* shift — the shape a robust per-subject
statistic is closest to optimal for. To test whether the central finding is an artefact of that
choice, we regenerate the same seeds under four shapes:

| Deviation shape | Population AUROC | Personal AUROC | MEYRO-V2 AUROC | Personal − Population |
|---|---|---|---|---|
| `mean_shift` | 0.9394 | 0.9949 | 0.9209 | **+0.0555** |
| `gradual_ramp` | 0.7015 | 0.7962 | 0.7543 | **+0.0947** |
| `variance_increase` | 0.7004 | 0.8112 | 0.5784 | **+0.1108** |
| `point_spike` | 0.9325 | 0.9982 | 0.7616 | **+0.0657** |

The personalization gain is positive in every shape, so the finding generalizes beyond the
mean-shift generator. The ordering is the expected signature of a genuine effect: gains are smallest
where a robust mean statistic already suffices (population AUROC `≥ 0.93`) and largest where it does
not (`≈ 0.70`). Shape alone does **not** explain the neural deficit — MEYRO-V2 is worst on
`variance_increase`, where a learned deviation module has nothing to add over a personal spread
estimate.

**Baseline drift — a failure.** After a sustained lifestyle change, with thresholds self-calibrated
from pre-drift data (`mean + 3σ`), the persistent false-alarm rate on post-drift steady-state days is:

| Method | Persistent false alarms | Baseline movement (σ) |
|---|---|---|
| Personal Baseline (Adaptive) | **30.0%** | 1.25 |
| Personal Baseline (Static) | 61.5% | 0.00 |
| MEYRO-V2 (score gate) | **93.0%** — worst | 0.00 |
| MEYRO-V2.1 (persistence-confirmed gate) | **93.0%** — no improvement | 0.00 |

The mechanism is the one identified in §2.2. Acute-excursion control is intact (one 3× observation
moves the baseline by at most `0.43 σ`), so the model under-adapts rather than over-adapts. The
targeted repair (§2.2) is inert: after calibrating the persistence head's threshold on each subject's
own history, its normal and drift regimes still differ by only `≈0.04`, so the confirmed-migration
rule never fires. V2.1 is therefore opt-in and reported as a negative result.

---

## 4. Ethical Considerations & Non-Diagnostic Scope
MEYRO is explicitly non-diagnostic. The system outputs:
*"MEYRO detected an empirical deviation from your historical baseline. This is a statistical observation of behavioral/physiological patterns, not a clinical or medical diagnosis."*

---

## 5. Conclusion

Personalized baseline modeling addresses the central limitation of population-level health thresholds:
across four deviation shapes it reduces the false-alarm rate at matched sensitivity by roughly **20.6×**
on the primary benchmark, and the effect is statistically significant with the subject as the unit of
analysis. That result, however, is achieved by a *robust statistical* personal baseline; the learned
MEYRO architecture does not yet beat it, and its anomaly-gated memory fails a sustained-change test in
a way we identify, attempt to repair, and measure as still unsolved.

We report both because the value of the framework is the reproducible, leakage-free evaluation
apparatus and the honest map of what personalization buys — not a claim that a neural architecture
wins. The immediate next step is not another architecture but evaluation on real longitudinal data
(GLOBEM, credentialed access), which is the only way to determine whether the residual gap and the
drift failure are properties of the model or of the synthetic generator.
