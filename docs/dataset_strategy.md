# MEYRO — Dataset Strategy

> **Phase 3 artifact.** Candidate datasets are recorded here **before** any data
> is downloaded. No dataset file is fetched until its license and suitability
> have been reviewed below, and no dataset ever enters Git.
>
> ⚠️ **Licence fields marked `TO VERIFY` have not been confirmed at the source
> of record.** They must be confirmed before the corresponding download is
> written into `scripts/download_data.py`. Nothing here has been downloaded yet.

| | |
|---|---|
| **Status** | Draft v0.1 — Phase 3 |
| **Selected primary** | **LifeSnaps** (open, 71 participants × ~4 months) |
| **Selected secondary** | **PMData** (CC BY 4.0 confirmed, 16 × 5 months) |
| **Selected scale-up** | **GLOBEM** (497 participants, 705 person-years) — ⚠️ credentialed access |
| **Decision finalised in** | Phase 4 (data architecture) / Phase 5 (pipeline), after licence confirmation |

---

## 1. Selection criteria

Because MEYRO's entire premise is *within-person* history, the criteria are
ordered by how hard a dataset is to substitute:

| Priority | Criterion | Why it is non-negotiable |
|---|---|---|
| **C1** | **Repeated observations from the same subjects over months** | Without per-subject history there is no personal baseline. A single session per subject cannot test H1. |
| **C2** | **Sufficient history per subject** (~weeks minimum, months preferred) | RQ-S3 asks how much history is needed; datasets with 10-week windows can only answer the low end. |
| **C3** | **Behavioural or physiological time series** | The master specification's recommended first direction. |
| **C4** | **A workable operationalisation of "deviation"** | See `problem_definition.md` §7.1. Labels do **not** have to be clinical. |
| **C5** | **Permissive, confirmed licence** | Reproducibility is a stated project objective; a credentialed dataset that others cannot obtain undermines it. |
| **C6** | **Enough subjects to evaluate per-individual performance** | RQ-S4 needs a distribution across people, not an aggregate. |

**Explicitly *not* a criterion:** clinical richness, disease labels, or
diagnostic value. MEYRO is not a diagnostic system, and seeking clinical labels
would push the project toward exactly the framing it must avoid.

---

## 2. Candidate datasets

### 2.1 LifeSnaps — ⭐ SELECTED PRIMARY

| Field | Value |
|---|---|
| **Name** | LifeSnaps |
| **Source** | Data descriptor: *Scientific Data* (Nature), 2022 — `https://www.nature.com/articles/s41597-022-01764-x`; dataset on Zenodo record 7229547 |
| **Licence** | `TO VERIFY` — Scientific Data data descriptors are typically CC BY 4.0; **the Zenodo record-level licence must be confirmed before download** |
| **Subjects** | 71 participants |
| **Samples** | >71,000,000 rows across >35 distinct data types |
| **Sampling rate** | From **second-level to daily** granularity |
| **Longitudinal availability** | ~4 months, continuous ("in the wild", geographically distributed) |
| **Features** | Fitbit Sense signals including heart rate, temperature, oxygen saturation, activity, sleep, plus anthropological self-reports |
| **Labels** | Self-report instruments; **no deviation labels** — see §3 |
| **Missing data** | Expected to be substantial and non-uniform (real-world wear behaviour); the descriptor publishes data-availability figures |
| **Demographics** | Geographically distributed; must be read from the descriptor |
| **Potential leakage** | Subject leakage (71 subjects must be split, not mixed); temporal leakage (the most likely serious error — see §4); device/firmware changes over 4 months; survey windows overlapping signal windows |
| **Limitations** | 71 subjects is modest; 4 months limits the long-history end of RQ-S3; Fitbit-derived features are vendor-processed, so raw-signal control is limited |
| **MEYRO suitability** | **High.** Open, multi-modal, genuinely longitudinal, and big enough for per-individual analysis. The best available balance of C1–C6. |

### 2.2 PMData — ⭐ SELECTED SECONDARY (licence-clean)

| Field | Value |
|---|---|
| **Name** | PMData (a sports/logging dataset) |
| **Source** | `https://datasets.simula.no/pmdata/`; descriptor: Thambawita et al. (2020) |
| **Licence** | **CC BY 4.0** — confirmed at source: *"PMData is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License"* |
| **Subjects** | 16 participants |
| **Samples** | ~1.4 GB; 252 files (json, csv, xlsx) |
| **Sampling rate** | Mixed: Fitbit-derived daily and intraday, plus event-level logs |
| **Longitudinal availability** | **5 months** per participant |
| **Features** | Fitbit Versa 2 smartwatch data (activity, heart rate, sleep, steps, calories), Google Forms self-reports, and PMSys sports-logging entries |
| **Labels** | Self-reported wellness/sport logs; **no deviation labels** |
| **Missing data** | Known real-world sparsity and inconsistent logging across participants |
| **Demographics** | 16 participants; verify participant-level detail in the descriptor |
| **Potential leakage** | Small-N makes subject leakage catastrophic (already a leakage outlier vs a 71- or 497-subject cohort); temporal leakage; self-report/signal window overlap |
| **Limitations** | Only 16 participants — **insufficient alone for RQ-S4 (heterogeneity)**; vendor-processed features |
| **MEYRO suitability** | **High as a replication and pipeline-validation set.** Its licence is confirmed and unconditional, so it is the cleanest dataset to build and test the end-to-end pipeline on. It is *not* large enough to carry the primary result on its own. |

### 2.3 GLOBEM — ⭐ SELECTED SCALE-UP

| Field | Value |
|---|---|
| **Name** | GLOBEM Dataset: Multi-Year Datasets for Longitudinal Human Behavior Modeling Generalization (v1.1) |
| **Source** | PhysioNet — `https://physionet.org/content/globem/1.1/`; DOI 10.13026/r9s1-s711 |
| **Licence** | ⚠️ **Credentialed Access** on PhysioNet (data use agreement + credentialing required). **Not openly downloadable.** |
| **Subjects** | **497 unique participants** |
| **Samples** | **705 person-years** across four datasets |
| **Sampling rate** | Predominantly daily-level behavioural features derived from mobile and wearable sensing |
| **Longitudinal availability** | **2018–2021, multi-year, two institutions** — the strongest longitudinal depth of any candidate |
| **Features** | Mobile and wearable passive sensing: activity, sleep, location, phone usage and related behavioural signals |
| **Labels** | Repeated mental-health survey instruments (e.g. depression/anxiety scales). **Used only as secondary research constructs — never as MEYRO's detection target** (see §5) |
| **Missing data** | Documented in the dataset paper as a central challenge of multi-year passive sensing |
| **Demographics** | Described as racially, ability- and immigrant-diverse across 497 participants |
| **Potential leakage** | **Cohort leakage** (four sub-datasets from two institutions — must not be mixed naively); subject leakage; temporal leakage; survey/instrument version changes |
| **Limitations** | Credentialed access blocks full third-party reproduction; features are mostly daily aggregates rather than raw signals; multi-institution construction complicates pooling |
| **MEYRO suitability** | **Very high scientifically, problematic operationally.** It is the dataset that can actually test multi-year personal baselines and RQ-S3 properly. Its credentialing is the single biggest threat to MEYRO's reproducibility objective (C5), so the primary result must not depend on it alone. |

### 2.4 StudentLife

| Field | Value |
|---|---|
| **Name** | StudentLife |
| **Source** | Dartmouth — `https://studentlife.cs.dartmouth.edu/`; Wang et al. (2014), UbiComp |
| **Licence** | `TO VERIFY` |
| **Subjects** | 48 students |
| **Samples** | Continuous passive sensing over a 10-week academic term |
| **Sampling rate** | Continuous sensor streams plus periodic EMA self-reports |
| **Longitudinal availability** | 10 weeks |
| **Features** | Accelerometer, GPS, microphone-derived audio features, call/SMS logs, screen state, light |
| **Labels** | EMA self-reports; PHQ-9 and related instruments |
| **Missing data** | Documented; sensing dropouts |
| **Demographics** | Undergraduate students at a single institution |
| **Potential leakage** | Subject and temporal; EMA/signal overlap |
| **Limitations** | 10 weeks only; single-institution student cohort; hand-crafted features |
| **MEYRO suitability** | **Validation only.** Valuable as a historically important longitudinal sensing precedent, but 10 weeks is too short to carry H1. |

### 2.5 WESAD

| Field | Value |
|---|---|
| **Name** | WESAD (Wearable Stress and Affect Detection) |
| **Source** | UCI ML Repository id 465; Schmidt et al. (2018), ICMI |
| **Licence** | `TO VERIFY` at repository level |
| **Subjects** | 15 |
| **Samples** | One controlled laboratory session per subject |
| **Sampling rate** | High-frequency physiological (Hz-scale: ECG/EDA/EMG/respiration/temperature/accelerometer) |
| **Longitudinal availability** | **None across days — one session per subject** |
| **Features** | Chest and wrist: ECG, EDA, EMG, respiration, skin temperature, BVP, heart rate, accelerometer |
| **Labels** | Condition labels within the lab protocol (baseline, stress, amusement…) |
| **Missing data** | Low (controlled lab) |
| **Demographics** | 15 subjects; verify in descriptor |
| **Potential leakage** | Subject leakage; window overlap within a session |
| **Limitations** | **Fundamentally unsuitable for personal longitudinal baselines** — no per-subject history of the kind MEYRO needs |
| **MEYRO suitability** | **Low for the primary question; useful later for signal-processing and modality plumbing** (Phases 15–16) because the raw physiology is rich and clean. Explicitly **not** a candidate for Phase 11. |

### 2.6 PAMAP2

| Field | Value |
|---|---|
| **Name** | PAMAP2 Physical Activity Monitoring |
| **Source** | UCI ML Repository; Reiss & Stricker (2012), ISWC |
| **Licence** | `TO VERIFY` |
| **Subjects** | 9 (main protocol; an optional additional protocol exists) |
| **Samples** | 18 scripted activity classes |
| **Sampling rate** | IMU-rate (Hz-scale) |
| **Longitudinal availability** | Approximately one collection session per subject |
| **Features** | 3 IMUs + heart-rate monitor: accelerometer, gyroscope, magnetometer, temperature, heart rate |
| **Labels** | Activity class labels |
| **Missing data** | Some sensor dropouts documented |
| **Demographics** | Small adult cohort |
| **Potential leakage** | Subject leakage; window overlap |
| **Limitations** | Protocol-driven, short, small-N |
| **MEYRO suitability** | **Robustness and signal-quality work only** (Phase 27). Not a longitudinal baseline dataset. |

### 2.7 HHAR

| Field | Value |
|---|---|
| **Name** | Heterogeneity Human Activity Recognition |
| **Source** | UCI ML Repository id 344; Stisen et al. (2015), SenSys |
| **Licence** | `TO VERIFY` |
| **Subjects** | 9 users, 8 smartphones, 4 smartwatches |
| **Samples** | Short collection periods with scripted activities |
| **Sampling rate** | Device-dependent (this heterogeneity is the point of the dataset) |
| **Longitudinal availability** | Short; not longitudinal in the MEYRO sense |
| **Features** | Accelerometer, gyroscope |
| **Labels** | Activity classes |
| **Missing data** | Device-dependent |
| **Demographics** | 9 users |
| **Potential leakage** | Device leakage if devices are split across folds carelessly |
| **Limitations** | Very short; tiny cohort |
| **MEYRO suitability** | **Device-heterogeneity robustness only** (Phase 27, decision D7). Does not support H1. |

### 2.8 Sleep-EDF Database Expanded

| Field | Value |
|---|---|
| **Name** | Sleep-EDF Database Expanded v1.0.0 |
| **Source** | PhysioNet — `https://physionet.org/content/sleep-edfx/1.0.0/` |
| **Licence** | **Open Data Commons Attribution License v1.0 (ODC-BY 1.0)** — confirmed on the PhysioNet licence page |
| **Subjects** | Expanded set spans healthy subjects and subjects with mild difficulty falling asleep |
| **Samples** | **197 whole-night** polysomnographic recordings (version-dependent; some earlier counts cite 61/153) |
| **Sampling rate** | Polysomnography-standard (Hz-scale: EEG, EOG, EMG) |
| **Longitudinal availability** | **Multiple nights per subject**, but nights are discontinuous |
| **Features** | EEG, EOG, chin EMG, event markers |
| **Labels** | Expert sleep-stage annotations |
| **Missing data** | Low, but per-subject night counts vary |
| **Demographics** | Mixed; verify in the record |
| **Potential leakage** | **Subject leakage is the classic error here** — nights from the same subject must not straddle splits; temporal leakage across nights |
| **Limitations** | Nights, not continuous days; sleep-only; discontinuous coverage |
| **MEYRO suitability** | **Good secondary modality** (sleep) with a clean open licence. Supports a per-night personal baseline, but discontinuous coverage weakens the "continuous personal history" argument. |

### 2.9 UK Biobank accelerometer

| Field | Value |
|---|---|
| **Name** | UK Biobank wrist-worn accelerometer dataset |
| **Source** | UK Biobank (application and access fees required) |
| **Licence** | ⚠️ **Access-controlled research access** — not open |
| **Subjects** | ~103,000 participants with accelerometer data (~92,480 in some analysed subsets) |
| **Samples** | Very large; ~700,000 person-days cited in downstream self-supervised work |
| **Sampling rate** | High-frequency raw triaxial accelerometry |
| **Longitudinal availability** | **~7 days per participant** — cross-sectional at the individual level |
| **Features** | Raw triaxial wrist accelerometry; derived activity metrics |
| **Labels** | Mostly none for anomaly detection; extensive health-record linkage |
| **Missing data** | Quality-control fields provided |
| **Demographics** | Adults aged ~40–80; well documented |
| **Potential leakage** | Subject leakage; care needed with the temporal ordering of the single week |
| **Limitations** | **One week per person ⇒ cannot test within-person change over months.** Access is gated and costly |
| **MEYRO suitability** | **Not suitable for H1.** Excellent for *population baseline* estimation (Phase 6) and for large-scale pretraining, which is a genuinely useful complementary role. |

### 2.10 Rejected / noted

| Dataset | Reason |
|---|---|
| WESAD (as a primary), PAMAP2, HHAR | No per-subject longitudinal history ⇒ cannot test H1 |
| UK Biobank (as a primary) | One week per participant ⇒ cannot test within-person change |
| Commercial vendor exports | Licence and methodological transparency unacceptable |
| Datasets requiring clinical labels as the target | Would push MEYRO toward diagnostic framing it must avoid (E1) |

---

## 3. The labelling problem, per dataset

This is the crux identified in `problem_definition.md` §7.1 and it constrains
dataset choice more than any other factor. **None** of the candidates ship with
"meaningful deviation" labels, because that concept is defined relative to a
personal history that the dataset authors did not compute.

Candidate operationalisations and which datasets can support them:

| Operationalisation | Support | Notes |
|---|---|---|
| **Induced deviations** (perturb held-out segments; onset known by construction) | All candidates | Fully controllable, fully reproducible, and **the least clinically meaningful**. Must be labelled as synthetic wherever reported (A6, E6). |
| **Held-out natural change points** | GLOBEM, LifeSnaps, PMData, Sleep-EDF | Annotated or detectable transitions. Measures *change*, not clinical meaning. |
| **Self-reported events** | PMData (Google Forms, PMSys), LifeSnaps, StudentLife | Noisy, sparse, subjective. Secondary signal only. |
| **Distributional shift proxies** (statistical change points) | All candidates | Truong et al. formalism. Explicitly **statistical**, explicitly **not** medical. |

**Decision deferred to Phase 11**, where the protocol is frozen before test
evaluation. Phase 11 must state the chosen operationalisation in every figure
and table, and must accompany induced-deviation results with an explicit
external-validity caveat.

---

## 4. Leakage risks (mandatory, per master specification rule 11–12)

Every dataset above shares the same failure modes. These are recorded now so
that `docs/leakage_audit.md` (Phase 25) has something concrete to audit:

| Risk | Mechanism | Mitigation to implement in Phase 5 |
|---|---|---|
| **Subject leakage** | Same subject in train and test inflates performance dramatically, because a personal baseline is person-specific by construction. **This is the single most dangerous error in MEYRO.** | Split **by `subject_id`**, enforced by an assertion in split construction. |
| **Temporal leakage** | A personal baseline built from data *after* the evaluated observation. | Baselines may consume only `t' < t`. Enforced by an ordering assertion, not by convention. |
| **Normalization leakage** | Global standardization statistics (mean/σ) computed over the full dataset, including test subjects. | Fit scalers on training data only; **prefer per-subject normalization where the design calls for it**, and document the choice. |
| **Cohort leakage** | GLOBEM contains four sub-datasets from two institutions. | Treat sub-datasets as separate cohorts; never mix before splitting. |
| **Device / firmware drift** | LifeSnaps and PMData span months of vendor updates; HHAR is *about* device differences. | Detect and model device/firmware as a covariate; robustness test in Phase 27. |
| **Label leakage** | Self-report windows overlapping the signal window they label. | Enforce temporal separation between report and window; document the gap. |
| **Hyperparameter leakage / test-set tuning** | Selecting thresholds or adaptation rates using test performance. | All model and threshold selection on validation data only (Phase 11 decision rule). |

---

## 5. Ethical constraints on dataset use

- **No clinical target.** GLOBEM's and StudentLife's mental-health instruments
  (PHQ-9 etc.) are available and are genuinely useful for *research
  comparison*, but MEYRO must **not** use them as its detection target. Doing so
  would convert a deviation-detection study into a depression-detection study,
  which is neither the research question nor something a non-clinical project
  should claim. If used at all, they are secondary constructs, reported with
  explicit limitations.
- **No redistribution.** Nothing from these datasets is committed to Git, at
  any stage of processing, in any anonymised form.
- **Attribution.** CC BY 4.0 (PMData) and ODC-BY 1.0 (Sleep-EDF) require
  attribution; dataset citations will be recorded in `CITATION.cff` and
  `docs/reproduction.md`.
- **Credentialed data.** GLOBEM requires credentialing; MEYRO must not describe
  it as openly reproducible, and its results must be reported separately from
  open-dataset results.
- **Consent scope.** Some datasets restrict use; the permitted scope must be
  read and recorded before download.
- **No disease claims.** A statistically detected deviation is not a clinical
  finding (§3). This must appear on every figure produced from these datasets.

---

## 6. Recommendation

**Staged, licence-aware selection:**

1. **Primary — LifeSnaps.** 71 participants × ~4 months, >35 data types,
   second-to-daily granularity. Best balance of longitudinal depth, cohort
   size, and openness. **Licence must be confirmed first.**
2. **Secondary / pipeline validation — PMData.** Licence confirmed as
   CC BY 4.0 and unconditional, so it is the correct dataset to build and test
   the end-to-end pipeline on before touching anything with unresolved terms.
   16 participants ⇒ replication only, never the sole basis of H1.
3. **Scale-up / long-history replication — GLOBEM.** 497 participants and 705
   person-years is the only candidate that can genuinely test multi-year
   personal baselines and the high end of RQ-S3. Its credentialed access means
   MEYRO's headline result must remain reproducible without it.
4. **Complementary — Sleep-EDF Expanded** (ODC-BY 1.0) for a second modality
   with a clean open licence, and **UK Biobank** purely for population-baseline
   estimation and large-scale pretraining — never for H1.

**Modality for the first end-to-end experiment (master specification Phase 4):
activity + physiological time series**, taken from LifeSnaps or PMData, on a
single modality with daily-or-finer resolution. Multimodality is deferred to
Phase 15.

### 6.1 Blocking tasks before any download

- [ ] Confirm the Zenodo record-level licence for LifeSnaps.
- [ ] Confirm the licence and consent scope for StudentLife and WESAD.
- [ ] Confirm UCI repository licences for PAMAP2 and HHAR.
- [ ] Read the GLOBEM data use agreement and the credentialing requirements.
- [ ] Confirm the permitted use scope for UK Biobank accelerometry.
- [ ] Record attribution requirements for CC BY 4.0 and ODC-BY 1.0 sources.

**No `scripts/download_data.py` is written until these are resolved.** The
Phase 5 pipeline will be developed and tested against a **small synthetic
fixture**, so that no phase is blocked on licensing and no real health data is
needed to validate the code.

---

## 7. Change log

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-09-23 | Phase 3 first pass: 9 candidates reviewed, LifeSnaps/PMData/GLOBEM selected, licences confirmed where verifiable and marked `TO VERIFY` where not, leakage risks enumerated, no data downloaded. |
