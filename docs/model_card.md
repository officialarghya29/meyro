# Model Card — MEYRO V1

**Model Name:** MEYRO V1 (Personalized Longitudinal Health Baseline Modeling)  
**Version:** 1.0.0-rc1  
**Author:** Arghya Bose (`officialarghya29`)  
**Repository:** [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)  
**License:** MIT  

---

## 1. Model Overview

MEYRO V1 is a specialized neural and statistical architecture designed for **longitudinal personalized anomaly detection** in continuous physiological and behavioral wearable time series. 

Rather than comparing an individual's vital signs against population-level diagnostic thresholds ("Is this normal for humans?"), MEYRO explicitly models and tracks an individual's idiosyncratic baseline normal ("Is this normal for THIS PERSON?").

---

## 2. Intended Use

- **Intended:** Longitudinal behavioral pattern tracking, detection of sustained multi-day deviations from personal setpoints (e.g. elevated resting heart rate combined with reduced mobility indicating illness/fatigue), and objective change-point monitoring.
- **Out of Scope / Non-Goals:** **NOT** a disease diagnosis system. MEYRO never outputs "You have disease X", nor should it replace professional medical judgment.

---

## 3. Architecture & Key Mechanisms

1. **Context-Conditioned Temporal Encoder:** Causal sequence modeling (GRU/TCN) conditioned on diurnal and weekly cyclic covariates.
2. **Personal Baseline Memory ($B_t$):** Latent subject-specific setpoint memory fitted strictly on historical calibration intervals.
3. **Relational Deviation Module ($D_t$):** Non-linear relational network contrasting current sequence dynamics against personal memory.
4. **Adaptive Memory Gating:** Dynamic update factor $\alpha(A_t) = \alpha_0 \exp(-\gamma A_t) \bar{Q}_t$ that enables benign drift tracking while blocking acute deviations from polluting the baseline.
5. **Quality-Conditioned Uncertainty Head:** Explicit prediction of confidence and sensor noise indicators.

---

## 4. Empirical Evaluation Summary

### 4.1 Master Benchmark (vs. 9 Baselines)
- **Population Baseline (Control):** AUROC 0.9417, AUPRC 0.7604, FPR @ 85% Sens: **19.07%**
- **Personalized Baseline (Ours):** AUROC **0.9955**, AUPRC **0.9598**, FPR @ 85% Sens: **0.78%**
- **Advantage:** **15× to 24× reduction in false alarm rate** under idiosyncratic population variance.

### 4.2 Robustness Under Sensing Impairments
- **Missing Data:** 0% missing (0.9968 AUROC) $\to$ 30% missing (0.9750 AUROC) — minimal degradation due to causal imputation.
- **Sensor Noise:** 1× noise (0.9974 AUROC) $\to$ 4× severe sensor noise (0.9859 AUROC).

### 4.3 Few-Shot Cold-Start Convergence
- **3 Days History:** +0.0079 AUROC advantage over population.
- **14 Days History:** +0.0397 AUROC advantage.
- **28 Days History:** +0.0460 AUROC advantage (converges to >0.995 AUROC).
- **Usability Recommendation:** Minimum 14 days of baseline collection recommended before enabling full personal alerting.
