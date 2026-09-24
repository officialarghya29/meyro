# MEYRO — Experimental Protocol & Benchmark Results (Phase 11)

**Status:** Completed (Phase 11 Primary Research Experiment)  
**Experiment Lead:** Arghya Bose (`officialarghya29`)  
**Repository:** [github.com/officialarghya29/meyro](https://github.com/officialarghya29/meyro)

---

## 1. Research Question & Hypothesis

- **Primary Research Question:** Can personalized longitudinal baselines detect meaningful behavioral/physiological deviations more effectively and with fewer false positives than population-level baselines?
- **Hypothesis:** Idiosyncratic inter-individual physiological variations (e.g. basal heart rate differences of 20+ bpm between healthy subjects) inflate population-level variance, resulting in high false-positive rates when clinical population thresholds are applied. A causal, personalized baseline will achieve superior AUPRC and significantly lower False Positive Rates (FPR) at matched high sensitivity.

---

## 2. Experimental Setup & Leakage Prevention

1. **Cohort Specification:**
   - 20 synthetic longitudinal subjects with idiosyncratic normal baselines (steps, resting HR, sleep duration).
   - 90 days monitoring horizon per subject (total 1,800 subject-days).
   - Sustained multi-day physiological deviation periods (illness/fatigue patterns) injected at controlled rates.
2. **Causal Temporal Splitting (Zero Lookahead Leakage):**
   - **Calibration Window:** Days 1–28 (used exclusively to construct individual and population reference distributions).
   - **Evaluation Window:** Days 29–90 (1,240 evaluation windows, containing 85 ground-truth positive anomaly windows).
   - Parameter estimation (medians, IQR scales, EWMA centers) is strictly backwards-looking.

---

## 3. Evaluated Conditions

1. **Population Baseline (Control Condition):**
   - Cohort-level robust median and IQR calculated across all subjects in the calibration period.
   - Fixed population Z-score deviation threshold.
2. **Personalized Baseline (Static Historical):**
   - Subject-specific median and IQR estimated solely from each individual's 28-day calibration history.
3. **Personalized Baseline (Adaptive / Drift-Aware):**
   - Subject-specific baseline updating via slow causal exponential tracking, with anomaly rejection guards preventing acute anomalies from distorting the baseline.

---

## 4. Empirical Results (Frozen Benchmark Run)

```text
Cohort Size: 20 subjects, 90 days longitudinal horizon
Evaluation Windows: 1240 total (85 positive anomaly states)

Condition                 | AUROC    | AUPRC    | FPR @ 85% Sens
-----------------------------------------------------------------
Population Baseline       | 0.9822   | 0.8139   | 0.0390
Personalized (Static)     | 0.9982   | 0.9738   | 0.0026
Personalized (Adaptive)   | 0.9983   | 0.9729   | 0.0026
```

### Key Scientific Findings:
- **False Positive Rate Reduction:** At 85% target sensitivity, the personalized baseline achieves an **FPR of 0.26%**, compared to **3.90%** for the population baseline — a **15× reduction in false alarm rate**.
- **Precision-Recall Superiority:** Precision-Recall AUC increases from **0.8139** (population) to **0.9738** (personalized), demonstrating that when anomalies are rare events in longitudinal monitoring, personalized baselines maintain vastly superior precision.
- **Hypothesis Supported:** Under idiosyncratic baseline heterogeneity, personalized longitudinal modeling provides statistically superior deviation detection compared to fixed population baselines.
