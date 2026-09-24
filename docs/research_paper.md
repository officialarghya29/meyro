# MEYRO: Personalized Baseline Modeling for Longitudinal Multimodal Health Anomaly Detection

**Arghya Bose**  
Department of Machine Learning Research & Software Engineering  
`officialarghya29@gmail.com` | [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)

---

## Abstract

Wearable health monitors and digital phenotyping systems typically evaluate physiological signals against broad population-level thresholds ("Is this normal for humans?"). Because healthy humans display pronounced idiosyncratic baseline variance (e.g. resting heart rates varying by over 25 bpm across healthy individuals), population-level models incur severe false-positive alarm rates or miss subtle individual physiological shifts. 

In this work, we present **MEYRO**, a causal, leakage-free framework for personal longitudinal baseline modeling and anomaly detection. MEYRO formulates personal health tracking not as disease classification, but as causal deviation estimation against an individual's self-established historical baseline ("Is this normal for *this person*?"). We design a unified architecture comprising: (1) a causal temporal sequence encoder, (2) an explicit latent personal baseline memory $B_t$, (3) a non-linear relational deviation module, and (4) an adaptive memory gating mechanism that allows continuous setpoint drift tracking while strictly preventing acute anomalies from corrupting the baseline.

In rigorous, leakage-free evaluations against 9 classical and deep learning baselines across a 60-day longitudinal cohort, MEYRO's personalized baseline reduces the False Positive Rate at 85% target sensitivity from **19.07%** (population baseline) to **0.78%**—representing a **>24× reduction in false alarms**. Furthermore, MEYRO demonstrates remarkable resilience to real-world sensor degradation, maintaining >0.975 AUROC under 30% missing data and 4× measurement noise.

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
- **Adaptive Drift Gating:** An exponential update filter $\alpha(A_t) = \alpha_0 \exp(-\gamma A_t) \bar{Q}_t$ that tracks gradual lifestyle changes while rejecting acute multi-day anomalies.
- **Empirical Validation:** Comprehensive benchmarking showing that explicit personalization is mathematically required to avoid false alarm fatigue.

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

When an anomaly occurs ($A_t \to 1$), adaptation freezes ($\alpha_t \to 0$), preserving baseline integrity.

---

## 3. Experimental Evaluation

### 3.1 Benchmark Protocol
- **Cohort:** 15 subjects, 60 days longitudinal horizon, controlled multi-day physiological deviations.
- **Split:** 21 days historical calibration (normal setpoint establishment), 39 days evaluation (60 positive longitudinal anomaly windows).

### 3.2 Master Comparison Results

| Architecture | AUROC | AUPRC | FPR @ 85% Sens | False Alarm Reduction |
|---|---|---|---|---|
| **Population Baseline (Control)** | 0.9417 | 0.7604 | 19.07% | 1.0× (Baseline) |
| **Isolation Forest (Per-Subject)** | 0.9442 | 0.7095 | 14.40% | 1.3× |
| **Global Deep Autoencoders (GRU/TCN)** | < 0.20 | < 0.10 | > 95.0% | Failed (Conflates subjects) |
| **One-Class SVM (Per-Subject)** | 0.9938 | 0.9380 | 1.56% | 12.2× |
| **Personal Baseline (Adaptive)** | 0.9935 | 0.9461 | 1.36% | 14.0× |
| **MEYRO Personal Baseline (Static)** | **0.9955** | **0.9598** | **0.78%** | **24.4×** |

### 3.3 Robustness Under Sensor Impairment
- **Missing Data:** 0% missing (0.9968 AUROC) $\to$ 30% missing (0.9750 AUROC).
- **Sensor Noise:** 1× noise (0.9974 AUROC) $\to$ 4× noise (0.9859 AUROC).

### 3.4 Cold-Start Trajectory
- **3 Days:** +0.0079 AUROC over population.
- **14 Days:** +0.0397 AUROC.
- **28 Days:** +0.0460 AUROC (reaches >0.995 AUROC asymptote).

---

## 4. Ethical Considerations & Non-Diagnostic Scope
MEYRO is explicitly non-diagnostic. The system outputs:
*"MEYRO detected an empirical deviation from your historical baseline. This is a statistical observation of behavioral/physiological patterns, not a clinical or medical diagnosis."*

---

## 5. Conclusion
Personalized baseline modeling resolves the fundamental limitation of population-level health thresholds, reducing false alarm rates by over 24× while preserving sensitivity to subtle individual physiological shifts. MEYRO provides an open, reproducible, leakage-free foundation for next-generation personalized health intelligence.
