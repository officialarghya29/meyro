# MEYRO — Mathematical Formulation of the MEYRO Architecture

**Author:** Arghya Bose (`officialarghya29`)  
**Specification:** Phase 9 Mathematical Rigor

---

## 1. System Variables

Let an observation window at time $t$ for subject $s$ be defined by:
- $X_t \in \mathbb{R}^{K \times D}$: temporal window of $K$ historical steps up to time $t$.
- $C_t \in \mathbb{R}^{C}$: contextual covariates (time of day, weekend indicator).
- $Q_t \in [0, 1]^{D}$: sensor quality / measurement confidence vector.
- $B_t \in \mathbb{R}^{d}$: personal baseline memory state at time $t$.

---

## 2. Component Formulations

### 2.1 Context-Conditioned Temporal Encoder
$$E_t = \text{TemporalEncoder}(X_t, C_t) \in \mathbb{R}^{d}$$
Combines causal sequence dynamics via GRU/TCN with context projection:
$$E_t = \text{LayerNorm}\left(\text{GRU}(X_t) + W_c C_t\right)$$

### 2.2 Learned Deviation Module
Instead of a naive Euclidean distance $\|E_t - B_t\|_2$, the deviation representation is parameterized by a multi-layer relational network conditioned on quality $Q_t$:
$$D_t = \text{GELU}\left(W_d [E_t \parallel B_t \parallel (E_t - B_t) \parallel E_t \odot B_t] + W_q Q_t\right) \in \mathbb{R}^{d}$$

### 2.3 Anomaly Head
The scalar anomaly score $A_t \in [0, 1]$ is computed as:
$$A_t = \sigma\left(W_a D_t + b_a\right)$$

### 2.4 Uncertainty Head
Predicts epistemic + aleatoric uncertainty $U_t \in [0, 1]$ conditioned on sensor quality and latent deviation:
$$U_t = \sigma\left(W_u [D_t \parallel Q_t] + b_u\right)$$

### 2.5 Persistence Module
Maintains an exponential persistence accumulator $R_t$:
$$R_t = \lambda_p R_{t-1} + (1 - \lambda_p) A_t$$
where $\lambda_p \in [0, 1]$ is the persistence momentum factor (e.g. 0.75).

### 2.6 Adaptive Baseline Memory Update
The personal baseline evolves causally without allowing acute anomalies to pollute $B_{t+1}$:
$$\alpha_t = \alpha_0 \cdot \exp\left(-\gamma A_t\right) \cdot \text{mean}(Q_t)$$
$$B_{t+1} = (1 - \alpha_t) B_t + \alpha_t E_t$$
When an anomaly is detected ($A_t \to 1$), adaptation gating shuts down ($\alpha_t \to 0$), preserving baseline integrity.
