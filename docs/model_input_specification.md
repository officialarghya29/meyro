# MEYRO — Model Input Specification & Data Contracts

**Author:** Arghya Bose (`officialarghya29`)  
**Specification Level:** Phase 2 Data Contract

---

## 1. Input Contract Formulation

For an individual subject $s$ at time index $t$, the input to the MEYRO model is defined as the tuple:
$$\mathcal{I}_t = (X_t, H_t, C_t, B_t, Q_t)$$

Where:
- **$X_t \in \mathbb{R}^{D}$:** Current observation vector across $D$ physiological/activity channels (e.g., step count, resting heart rate, sleep duration).
- **$H_t \in \mathbb{R}^{K \times D}$:** Recent causal temporal history window covering time steps $\{t-K, \dots, t-1\}$ strictly prior to $t$.
- **$C_t \in \mathbb{R}^{C}$:** Contextual metadata vector (e.g., cyclical hour of day $\sin/\cos$, day of week, active vs. sedentary state).
- **$B_t \in \mathbb{R}^{M}$:** Personal baseline memory state representing the established historical normal distribution for subject $s$.
- **$Q_t \in [0, 1]^{D}$:** Sensor data quality vector indicating measurement fidelity and missingness indicators for each channel.

---

## 2. Output Contract Formulation

The MEYRO model computes a structured prediction tuple:
$$\mathcal{O}_t = (P_t, E_t, D_t, A_t, U_t, R_t, B_{t+1})$$

Where:
- **$P_t \in \mathbb{R}^{d_m}$:** Personal identity/pattern embedding.
- **$E_t \in \mathbb{R}^{d_m}$:** Encoded temporal dynamics representation of the current observation and window.
- **$D_t \in \mathbb{R}^{d_m}$:** Learned latent deviation representation ($E_t$ contrasted against $B_t$ and conditioned on $C_t$).
- **$A_t \in [0, 1]$:** Scaled scalar anomaly score (higher = greater deviation from personal normal).
- **$U_t \in [0, 1]$:** Calibrated uncertainty score (epistemic + aleatoric, conditioned on $Q_t$).
- **$R_t \in [0, 1]$:** Persistence score tracking sustained longitudinal deviation over consecutive windows.
- **$B_{t+1} \in \mathbb{R}^{M}$:** Updated personal baseline memory for step $t+1$.
