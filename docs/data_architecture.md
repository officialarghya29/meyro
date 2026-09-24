# MEYRO — Data Architecture Specification

**Status:** Stable (Phase 4 Specification)  
**Applicability:** Longitudinal Physiological & Activity Time Series  
**Reference Format:** Synthetic benchmark fixture & PMData/LifeSnaps aligned schema

---

## 1. Objective

This specification defines the canonical data architecture for MEYRO. The primary goal is to ensure:
1. **Schema enforcement** and type validation for longitudinal multimodal observations.
2. **Strict temporal ordering** and subject indexing to make temporal and subject leakage structurally impossible.
3. **Reproducible synthetic benchmarks** with parameterized ground truth (subject-specific normal baselines, diurnal cycles, measurement noise, and known deviation periods) allowing rigorous testing before private/external datasets are integrated.

---

## 2. Canonical Data Models

### 2.1 Schema Definition

Observations are represented at regular resampled intervals (default: 1-hour or 1-day bins for daily phenotyping).

```text
Record Schema:
├── subject_id: str (Pseudonymized subject identifier)
├── timestamp: datetime (UTC timestamp of the observation start)
├── modality: str ("activity", "physiology", "sleep")
├── features: Dict[str, float] (Metric values, e.g., step_count, resting_heart_rate, hrv_rmssd)
├── quality_score: float (Signal reliability in [0.0, 1.0])
└── ground_truth_label: Optional[int] (0 = normal, 1 = true deviation from personal normal; benchmark only)
```

### 2.2 Feature Specifications

| Feature Name | Modality | Physical Unit | Valid Range | Missing Strategy |
|---|---|---|---|---|
| `step_count` | activity | steps / bin | [0, 50000] | Zero imputation with missingness indicator |
| `active_minutes` | activity | minutes / bin | [0, 1440] | Zero imputation with missingness indicator |
| `resting_hr` | physiology | beats / min | [30.0, 220.0] | Forward fill up to 3 bins, then linear interpolation |
| `hrv_rmssd` | physiology | milliseconds | [5.0, 300.0] | Forward fill up to 3 bins, then personal median |
| `sleep_duration_min`| sleep | minutes | [0, 1000] | Personal historical median |
| `sleep_efficiency` | sleep | percentage | [0.0, 1.0] | Personal historical median |

---

## 3. Data Integrity & Leakage Prevention Rules

1. **Subject Isolation:** All normalization parameters, scalers, and personalized baselines are fitted exclusively on an individual subject's historical window. Inter-subject information exchange is only permitted within population baseline models.
2. **Causal (Historical) Windowing:** For any time point $t_i$, features and baselines may only utilize observations from $\{t_k \mid t_k < t_i\}$. Strictly no lookahead, centered rolling windows, or bidirectional smoothing filters.
3. **Partition Scheme:**
   - **Calibration / Warmup Period:** First $N_{\text{calib}}$ days (e.g., 14 to 30 days) used to construct the initial personal baseline.
   - **Evaluation Period:** Subsequent longitudinal window where personal deviations are evaluated against the frozen or adaptively updated historical baseline.
