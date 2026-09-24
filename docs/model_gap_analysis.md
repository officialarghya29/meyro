# MEYRO — Model Gap Analysis & Design Justification

**Author:** Arghya Bose (`officialarghya29`)  
**Phase:** Phase 7 Synthesis

---

## 1. Experimental Baseline Findings

From benchmarking population models, personal statistical baselines, classical detectors (Isolation Forest, LOF, OCSVM), and temporal deep learning models (GRU, LSTM, TCN, Transformer):

1. **Population & Global Temporal Models Suffer from Baseline Incommensurability:**
   Global neural architectures (standard GRU/LSTM/Transformer autoencoders) fit a single global manifold. They treat individual setpoint differences (e.g. resting HR 52 vs 76) as reconstruction residuals, yielding false anomaly triggers.
2. **Per-Subject Classical Models Lack Sequence & Context Reasoning:**
   Isolation Forests and One-Class SVMs trained per subject capture personal distributions, but treat observations as independent draws, missing multi-day temporal degradation patterns.
3. **No Baseline Disambiguates Transient Spikes from Persistent Drift:**
   Standard architectures have no mechanism to slowly adapt to benign drift (e.g. fitness shifts) while remaining robust against acute Multi-day illness events.

---

## 2. The Architectural Solution: MEYRO V1/V2

MEYRO resolves these limitations through four novel coupled mechanisms:
- **Explicit Personal Memory State ($B_t$):** Separately track individual setpoints in latent space.
- **Context & Quality-Gated Temporal Encoding ($E_t$):** Condition causal sequence encoding on diurnal cycles and sensor reliability.
- **Contrastive Learned Deviation Module ($D_t$):** Compute deviation representations directly contrasting $E_t$ against $B_t$.
- **Gated Persistence & Uncertainty Heads:** Continuous temporal accumulator for anomaly persistence and confidence estimation.
