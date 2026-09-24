# MEYRO — Model Literature & Comparative Architecture Analysis

**Author:** Arghya Bose (`officialarghya29`)  
**Status:** Frozen (Phase 1 Specification)  
**Target:** Longitudinal Personal Baseline Modeling & Wearable Anomaly Detection

---

## 1. Architectural Survey

| Architecture Class | Exemplar Model | Input Formulation | Personalization Mechanism | Temporal Mechanism | Adaptation Mechanism | Uncertainty Mechanism | Primary Limitation for MEYRO |
|---|---|---|---|---|---|---|---|
| **Classical Population Statistics** | Robust Z-Score / IQR | Flat feature vector $X_t$ | None (Cohort pooled) | None (Pointwise) | Static | None | High false alarms due to inter-individual variance |
| **Statistical Personal Baseline** | Rolling EWMA / Median | Flat feature vector $X_t$ | Subject-specific running window | Causal rolling statistics | Exponential decay | IQR scale | Ignores multi-step temporal dynamics and complex context |
| **Classical Distance / Density ML** | Isolation Forest, One-Class SVM, LOF | Window or instantaneous vector $X_t$ | Independent per-subject models | Minimal (via lag features) | Static (requires complete retraining) | Decision margin heuristic | High memory scaling ($O(N)$ models), cannot model continuous drift |
| **Recurrent Temporal Deep Learning** | LSTM / GRU Autoencoders (Malhotra et al. 2016) | Sliding window $X_{t-k:t}$ | Per-subject finetuning or global | Recurrent gating ($h_t$) | Gradient updates | Reconstruction error | Lacks explicit personal baseline memory $B_t$; treats deviations as reconstruction failure |
| **Temporal Convolutional Networks** | TCN (Bai et al. 2018) | Causal dilated convolutions | Shared parameters | Dilated causal receptive fields | Static | None | No native identity encoding or adaptive baseline memory |
| **Temporal Transformers** | PatchTST / TimeSeriesTransformer | Patch/token embeddings | Positional encodings only | Multi-head self-attention | Static | None | Quadratic attention complexity; no explicit distinction between persistent drift and acute anomaly |
| **MEYRO Architecture (Target)** | **MEYRO V1/V2** | Observation $X_t$, Context $C_t$, Quality $Q_t$, History $H_t$ | Dedicated Personal Memory $B_t$ + Gated Adaptation | Causal Temporal Encoder + Attention | Gated Adaptive Memory Update ($\alpha(z)$) | Dedicated Uncertainty Head (Epistemic + Aleatoric) | Unified architecture designed specifically around individual baseline modeling |

---

## 2. Critical Research Gap: What Existing Architectures Fail to Represent

Existing time-series and anomaly models fail along four fundamental axes:

1. **Failure to Explicitly Decouple Personal Normal from Population Dynamics:**
   Standard deep architectures conflate temporal dynamics with personal setpoints. If subject A has a resting heart rate of 55 bpm and subject B has 75 bpm, standard models require distinct training or struggle with unnormalized variance.
2. **Instantaneous Anomaly vs. Persistent Drift:**
   Existing models flag any unfamiliar pattern as an anomaly. In longitudinal physiology, lifestyle shifts (e.g. improved cardiovascular fitness) represent benign drift ($B_t \to B_{t+1}$), whereas fever/illness represents acute persistent deviation that should *not* pollute the baseline.
3. **Absence of Data Quality & Uncertainty Conditioning:**
   Wearable sensors constantly experience motion artifacts and missing intervals. Existing models output deterministic anomaly scores regardless of sensor reliability.
4. **Lack of Persistence Modeling:**
   A single atypical movement or heart rate spike is frequently transient sensor noise. Meaningful physiological deviations are sustained across consecutive temporal windows.
