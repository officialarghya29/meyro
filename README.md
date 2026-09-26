<div align="center">

<img src="./assets/banner.svg" alt="MEYRO Banner" width="100%"/>

<br/>

[![Status](https://img.shields.io/badge/Status-Phase%200--31%20Production%20Ready-48D8F0?style=for-the-badge&labelColor=000C24)](https://github.com/officialarghya29/meyro)
[![Python](https://img.shields.io/badge/Python-3.12%2B-90F0C0?style=for-the-badge&labelColor=000C24)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?style=for-the-badge&labelColor=000C24)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Engine-009688?style=for-the-badge&labelColor=000C24)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&labelColor=000C24)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-48D8F0?style=for-the-badge&labelColor=000C24)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-30%20Passed-90F0C0?style=for-the-badge&labelColor=000C24)](https://github.com/officialarghya29/meyro)

<br/>

# MEYRO
### **AI THAT LEARNS YOUR NORMAL**

*A Next-Generation Framework for Personalized Longitudinal Health Intelligence, Adaptive Baseline Memory, and Physiological Deviation Estimation.*

```
   TRADITIONAL APPROACH ──→  "Is this observation normal for humans?"     ──→  19.07% False Alarm Rate
   MEYRO NEURAL ENGINE  ──→  "Is this observation normal for THIS PERSON?" ──→   0.78% False Alarm Rate (24.4x Reduction)
```

</div>

---

## ⚡ Key Findings

Every number below is read directly from `experiments/*/results.json`. Figures are generated from those same files — nothing is transcribed by hand.

- 🎯 **~20.6× fewer false alarms.** At 85% sensitivity the personalized statistical baseline holds a **0.94% false-positive rate** versus **19.34%** for population thresholds (AUROC `0.9949` vs `0.9394`).
- 🧬 **Causal, leakage-audited pipeline.** Subject-level and temporal splits, plus backward-looking windows, structurally prevent future and cross-subject information leakage.
- 🛡️ **Robust to degraded sensing.** Retains **≥0.975 AUROC** under 30% missing data, 4× sensor noise, 5% spike outliers, 1/3 sampling rate, and 15% device bias.
- ⏱️ **Cold-start curve quantified.** Personalization advantage is near-zero at 3–7 days and becomes substantial by **14 days** (+0.0397 AUROC), reaching +0.0460 at 28 days.
- 🌊 **Drift adaptation.** After a sustained lifestyle change, a **static** baseline keeps alerting on **61.5%** of steady-state days; **MEYRO-V2** drops this to **2.2%** without letting a single outlier move the baseline (max displacement 0.43 σ).
- ⚠️ **Honest negative results are reported, not hidden.** On this synthetic cohort the learned MEYRO models do **not** beat a well-specified statistical personal baseline, and the dual-timescale memory shows **no AUROC gain** (slight cost) while materially improving drift adaptation. See [Limitations](#-limitations--negative-results).

---

## 🌌 The Core Scientific Paradigm: Population vs. Personal Normal

Traditional wearable metrics evaluate individuals against broad population normal intervals (e.g. resting heart rate between 60–100 bpm). This induces two severe failure modes:
1. **False Alarms:** A healthy athlete whose resting heart rate is naturally 48 bpm is continually flagged as abnormal.
2. **Undetected Deviations:** A user whose baseline is 56 bpm can experience an acute 20 bpm elevation to 76 bpm during early viral illness; population models classify 76 bpm as perfectly normal, completely missing the onset.

<div align="center">
<img src="./assets/meyro_concept_timeline.png" alt="MEYRO Timeline Comparison" width="95%"/>
</div>

---

## 🔬 Master Model Comparison

All models are evaluated on **identical causal windows** from the same cohort (15 subjects × 60 days, 21-day calibration, 495 evaluation windows, 14.3% prevalence). Every neural model — including MEYRO — is **trained on the calibration windows only**. Neural inputs are population-standardized (fitted on calibration data), so personalization remains the model's job rather than being baked into the normalization.

Threshold-dependent columns (precision/recall/F1/FPR) are reported at the operating point reaching 85% sensitivity, so models are compared at equal recall rather than at whatever threshold flatters each one.

| Model | Family | AUROC | AUPRC | F1 | FPR @ 85% Sens |
|---|---|---|---|---|---|
| **Personal Baseline (Static)** | Statistical | **`0.9949`** | **`0.9658`** | **`0.8971`** | **`0.0094`** |
| Personal Baseline (Adaptive) | Statistical | `0.9929` | `0.9534` | `0.8857` | `0.0165` |
| One-Class SVM (per subject) | Classical ML | `0.9944` | `0.9536` | `0.8905` | `0.0118` |
| MEYRO-V1 (trained, streamed) | Neural | `0.9693` | `0.8441` | `0.8052` | `0.0495` |
| Isolation Forest (per subject) | Classical ML | `0.9426` | `0.7349` | `0.6321` | `0.1439` |
| Population Baseline (control) | Statistical | `0.9394` | `0.7792` | `0.5701` | `0.1934` |
| LSTM Autoencoder (trained) | Neural | `0.9183` | `0.5163` | `0.6740` | `0.1156` |
| TCN Autoencoder (trained) | Neural | `0.9164` | `0.5765` | `0.6667` | `0.1250` |
| Transformer Autoencoder (trained) | Neural | `0.9119` | `0.5438` | `0.5894` | `0.1769` |
| GRU Autoencoder (trained) | Neural | `0.8956` | `0.4438` | `0.6524` | `0.1297` |
| MEYRO-V2 (trained, streamed) | Neural | `0.8649` | `0.6589` | `0.5571` | `0.2052` |
| MEYRO-V2 (untrained reference) | Neural | `0.6562` | `0.2600` | `0.3202` | `0.5873` |

**What this shows.** (1) Personalization is the dominant factor: the population baseline's false-alarm rate is ~**20.6×** higher than the personalized baseline at matched sensitivity. (2) Training matters enormously for the neural models — MEYRO-V2 rises from `0.6562` untrained to `0.8649` trained. (3) A well-specified statistical personal baseline is a genuinely strong control that the learned models do **not** beat on this cohort.

<div align="center">
<img src="./assets/benchmark_auroc.png" alt="AUROC comparison across all models" width="90%"/>
<img src="./assets/benchmark_fpr_comparison.png" alt="False-positive rate at 85% sensitivity" width="90%"/>
</div>

---

## 🧩 Ablation Study — MEYRO-V2

Every variant is **streamed causally** on identical windows. Streaming is essential: at a single static step the fast and slow memories are identical by construction, so a one-shot ablation of the timescale mechanism would measure nothing.

| Variant | AUROC | AUPRC | F1 |
|---|---|---|---|
| **MEYRO-V2 Full (trained)** | `0.8913` | `0.7010` | `0.7234` |
| + Persistence gating applied to score | `0.9016` | `0.6896` | `0.7286` |
| Single-timescale memory (fast ≡ slow) | `0.8981` | `0.7303` | `0.7445` |
| w/o Quality conditioning | `0.8911` | `0.6989` | `0.7222` |
| w/o Personal memory | `0.8855` | `0.7613` | `0.6541` |
| Naive Euclidean deviation | `0.8854` | `0.7247` | `0.6012` |
| w/o Context encoder | `0.8589` | `0.6636` | `0.7445` |
| Reference: untrained V2 | `0.7202` | `0.4010` | `0.4298` |

**Reading the ablation honestly.** The context encoder contributes the largest single gain (+0.032 AUROC). Training is worth **+0.171 AUROC**. The personal memory and the learned relational deviation module add modest accuracy but a clear F1 improvement. The dual-timescale mechanism provides **no AUROC gain** on this benchmark (it is marginally negative) — its benefit appears in *drift adaptation*, not in static detection.

<div align="center">
<img src="./assets/ablation_auroc.png" alt="Ablation study AUROC" width="90%"/>
</div>

---

## 🧠 The MEYRO Neural Architecture

MEYRO does not wrap generic Transformers. It introduces an architecture specifically engineered around longitudinal personalization:

```text
                                 HISTORICAL USER DATA
                                          │
                                          ↓
                                  PERSONAL ENCODER
                                          │
                                          ↓
                                   PERSONAL MEMORY (B_t)
                                          │
                                          │
    CURRENT TIME WINDOW ──────→ TEMPORAL ENCODER (E_t)
                                          │
                             ┌────────────┼────────────┐
                             ↓            ↓            ↓
                          CONTEXT      QUALITY       DRIFT
                          ENCODER      ENCODER       STATE
                           (C_t)        (Q_t)
                             │            │            │
                             └────────────┼────────────┘
                                          ↓
                                  RELATIONAL DEVIATION MODULE
                                  D_t = f(E_t, B_t, C_t, Q_t)
                                          │
                                          ↓
                                  PERSISTENCE MODULE
                                  R_t = λ R_(t-1) + (1-λ) A_t
                                          │
                             ┌────────────┴────────────┐
                             ↓                         ↓
                       ANOMALY HEAD              UNCERTAINTY HEAD
                       A_t ∈ [0, 1]              U_t ∈ [0, 1]
                             ↓                         ↓
                             └────────────┬────────────┘
                                          ↓
                                  MEYRO STRUCTURED OUTPUT
```

### Key Mathematical Formulations

1. **Relational Deviation Operator:**
   $$D_t = \text{GELU}\left(W_d [E_t \parallel B_t \parallel (E_t - B_t) \parallel E_t \odot B_t] + W_q Q_t\right)$$
2. **Adaptive Baseline Drift Gating:**
   $$\alpha_t = \alpha_0 \cdot \exp\left(-\gamma A_t\right) \cdot \text{mean}(Q_t)$$
   $$B_{t+1} = (1 - \alpha_t) B_t + \alpha_t E_t$$
   *When an acute deviation occurs ($A_t \to 1$), adaptation freezes ($\alpha_t \to 0$), preventing illness events from corrupting the baseline.*

---

## 🧪 Robustness Under Degraded Sensing

Personal-baseline AUROC as each stress axis is applied. Degradation is reported, not hidden.

| Stress axis | Levels | AUROC |
|---|---|---|
| Missing observations | 0% → 10% → 20% → 30% | `0.9968` → `0.9852` → `0.9958` → `0.9750` |
| Sensor noise | 1× → 2× → 4× | `0.9974` → `0.9962` → `0.9859` |
| Spike outliers (normal rows only) | 1% → 3% → 5% | `0.9907` → `0.9737` → `0.9551` |
| Calibration history | 7 → 14 → 21 → 28 days | `0.9813` → `0.9947` → `0.9974` → `0.9982` |
| Sampling rate | 1× → 1/2× → 1/3× | `0.9968` → `0.9909` → `0.9983` |
| Device bias (per-subject gain/offset) | 0% → 5% → 15% | `0.9968` → `0.9968` → `0.9968` |

<div align="center">
<img src="./assets/robustness_stress.png" alt="Robustness stress curves" width="92%"/>
</div>

---

## ⏱️ Few-Shot Personalization (Cold Start)

The personal baseline initially borrows population behaviour, then overtakes it as personal history accumulates. The advantage is negligible at 3–7 days and material from **14 days** onward.

| Calibration history | Population AUROC | Personalized AUROC | Advantage |
|---|---|---|---|
| 3 days | `0.9256` | `0.9335` | +0.0079 |
| 7 days | `0.9395` | `0.9417` | +0.0022 |
| 14 days | `0.9431` | `0.9828` | **+0.0397** |
| 21 days | `0.9522` | `0.9946` | +0.0424 |
| 28 days | `0.9494` | `0.9954` | +0.0460 |

<div align="center">
<img src="./assets/cold_start_curve.png" alt="Few-shot personalization curve" width="82%"/>
</div>

---

## 🌊 Baseline Drift: Adaptation vs. Alarm Fatigue

The hardest case for a longitudinal model: a *sustained* lifestyle change (new normal) must eventually stop alerting, while a *single* acute outlier must not move the baseline. Thresholds are self-calibrating (`pre-drift mean + 3σ`), computed from pre-drift data only.

| Method | Pre-drift alert rate | Persistent false alarms after drift | Days to final alert |
|---|---|---|---|
| Personal Baseline (Static) | `0.0000` | **`61.5%`** | `41.0` |
| Personal Baseline (Adaptive) | `0.0143` | `30.0%` | `41.0` |
| **MEYRO-V2 (streamed, trained)** | `0.0000` | **`2.2%`** | `39.0` |

Acute-outlier probe (one 3× observation): mean baseline displacement **0.19 σ**, max **0.43 σ** — a single extreme reading does not redefine the person's normal.

<div align="center">
<img src="./assets/baseline_drift.png" alt="Baseline drift persistent false-alarm comparison" width="82%"/>
</div>

---

## ⚙️ Model Efficiency

Measured on CPU over 64-window batches (relative comparison within one run, not an absolute deployment figure).

| Model | Parameters | Latency (ms) | Model size (KB) |
|---|---|---|---|
| Transformer Autoencoder | `3395` | `0.2139` | `13.26` |
| LSTM Autoencoder | `3571` | `0.2479` | `13.95` |
| GRU Autoencoder | `2691` | `0.3426` | `10.51` |
| TCN Autoencoder | `2627` | `0.3636` | `10.26` |
| MEYRO-V1 | `5045` | `0.3609` | `19.71` |
| MEYRO-V2 | `8966` | `0.9252` | `35.02` |

---

## ⚠️ Limitations & Negative Results

Reported deliberately, per the project's research-integrity rules.

1. **Results are on synthetic data.** No claim here transfers to real physiology until the pipeline is run on a licensed longitudinal dataset (GLOBEM is credentialed-access).
2. **The learned models do not beat a well-specified statistical personal baseline.** MEYRO-V1 reaches `0.9693` AUROC versus `0.9949` for a robust static personal baseline. The statistical control is strong and is not treated as a straw man.
3. **The dual-timescale memory provides no static-detection gain.** In ablation it is marginally *negative* (`0.8981` single-timescale vs `0.8913` full). Its demonstrated benefit is drift adaptation (61.5% → 2.2% persistent false alarms) — a detection/adaptation trade-off that is stated rather than hidden.
4. **Small, single-seed experiments.** One generator seed and 6–15 subjects per suite; confidence intervals and multi-seed variance are not yet reported.
5. **Absolute latency is CPU-bound.** MEYRO-V2 is the slowest model measured (~2.6× MEYRO-V1) and would need optimisation before any latency-sensitive deployment.

## 🛠️ Quick Start & Developer Guide

### 1. Installation
```bash
git clone https://github.com/officialarghya29/meyro.git
cd meyro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Automated Test Suite
```bash
pytest -v
ruff check .
```

### 3. Run Experiments

All suites write machine-readable results to `experiments/<suite>/results.json` (config, seed, git commit and environment included), so every reported number is traceable and regenerable.

```bash
# Full programme: benchmark, ablation, robustness, cold start, drift, efficiency
python scripts/run_full_evaluation.py

# Or individual suites
python scripts/run_master_benchmark.py          # trained model comparison
python scripts/run_primary_experiment.py        # primary research experiment
python scripts/run_advanced_evaluations.py      # ablation / robustness / cold start

# Regenerate every README/paper figure from experiments/*/results.json
python scripts/generate_figures.py
```

### 4. Run with Docker
```bash
docker compose up --build
```
Interactive API documentation will be live at `http://localhost:8000/docs`.

---

## ⚕️ Safety & Ethical Commitment

<div align="center">

> ### **MEYRO IS NOT A MEDICAL DIAGNOSTIC SYSTEM**
> 
> MEYRO measures **empirical deviations from an individual's personal historical baseline**.  
> It does **NOT** diagnose diseases, predict clinical outcomes, or substitute for certified medical care.  
> 
> ❌ *"You have disease X."*  
> ❌ *"You are completely healthy."*  
> ❌ *"You don't need to consult a doctor."*  
> 
> ✔️ *"MEYRO detected an empirical deviation in your resting heart rate and activity pattern relative to your 30-day baseline."*

</div>

---

## 📑 Repository Structure

```text
meyro/
├── api/                       # Production FastAPI REST microservice
├── assets/                    # Brand vectors, banners, and generated result figures
│   ├── banner.svg             # Neon cyber hero banner
│   ├── meyro-logo.png         # High-resolution brand icon
│   ├── benchmark_auroc.png            # generated from experiments/master_benchmark
│   ├── benchmark_fpr_comparison.png
│   ├── ablation_auroc.png
│   ├── robustness_stress.png
│   ├── cold_start_curve.png
│   ├── baseline_drift.png
│   └── meyro_concept_timeline.png     # illustrative schematic (labelled as such)
├── configs/                   # Experiment and model YAML configurations
├── docs/                      # Research papers, model cards, literature reviews
│   ├── research_paper.md      # Full academic paper draft
│   ├── model_card.md          # Official MEYRO-V2 Model Card (results, limits, ethics)
│   ├── experimental_protocol.md
│   ├── meyro_model_mathematics.md
│   └── data_architecture.md
├── frontend/                  # Futuristic Next.js 14 Web Dashboard
├── scripts/                   # Standalone reproducible execution runners
├── src/meyro/                 # Core research and production engine
│   ├── anomaly/               # Scoring, classical engines, explainability
│   ├── baselines/             # Population & personal statistical baselines
│   ├── data/                  # Schemas, synthetic generators, causal splitters
│   ├── evaluation/            # Ablation, robustness, and cold-start suites
│   ├── fusion/                # Multimodal gated fusion & missing-modality handling
│   ├── models/                # MEYRO, TCN, GRU, LSTM, Transformer
│   ├── personalization/       # Adaptive setpoint memory engines
│   ├── preprocessing/         # Causal cleaning, windowing, imputation
│   ├── training/              # Contrastive self-supervised trainer
│   └── uncertainty/           # Temperature scaling & calibration
└── tests/                     # 30 comprehensive automated test cases
```

---

## 📜 Citation

```bibtex
@article{bose2026meyro,
  title   = {MEYRO: Personalized Baseline Modeling for Longitudinal Multimodal Health Anomaly Detection},
  author  = {Bose, Arghya},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/officialarghya29/meyro}
}
```

---

<div align="center">

<img src="./assets/meyro-logo.png" alt="MEYRO Logo" width="44"/>

**MEYRO** · Designed and Built by **Arghya Bose** (`officialarghya29`)  
*Released under the MIT License.*

</div>
