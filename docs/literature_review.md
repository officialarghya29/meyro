# MEYRO — Literature Review

> **Phase 2 artifact.** Companion machine-readable matrix:
> [`literature_matrix.csv`](./literature_matrix.csv).
>
> ⚠️ **Read the methodology section before citing anything here.** This is a
> **scoping review conducted with a general web search engine**, not a
> PRISMA-style systematic review of bibliographic databases. It is sufficient
> to orient the project and to test the working gap statement; it is **not**
> sufficient to support a novelty claim in a paper. Section 8 states exactly
> what still has to be done.

| | |
|---|---|
| **Status** | Draft v0.1 — Phase 2, first pass |
| **Sources captured** | 19 entries in the matrix |
| **Verification** | Every entry carries a `source_verification` column; **no citation in this document was invented** |
| **Revises** | The provisional gap statement in `problem_definition.md` §4 |
| **Must be revised by** | A database-backed search (Scopus/PubMed/IEEE Xplore/ACM DL) before any paper submission |

---

## 1. Objective

Answer four questions, in order:

1. **What already exists** in personalized longitudinal health monitoring and
   anomaly detection?
2. **What do existing methods do well?**
3. **What do they fail to address?**
4. **Is the gap MEYRO proposed in Phase 1 real, narrower than claimed, or
   already closed?**

Question 4 is the important one. The master specification is explicit:
*"Do not call MEYRO novel simply because it sounds novel."*

---

## 2. Methodology

### 2.1 Search strategy (as actually performed)

Queries issued against a general web search engine, deliberately spanning the
eleven areas named in the master specification:

| # | Area | Representative query |
|---|------|---------------------|
| 1 | Digital phenotyping | `digital phenotyping smartphone Jain Onnela 2015` |
| 2 | Personalized health monitoring | `personalized baseline anomaly detection wearable longitudinal individual normal` |
| 3 | Personalized anomaly detection | `personalized anomaly detection wearables individual baseline paper` |
| 4 | Concept drift | `Gama 2014 survey concept drift adaptation machine learning` |
| 5 | Anomaly detection foundations | `Chandola 2009 Anomaly Detection A Survey ACM Computing Surveys` |
| 6 | Classical detectors | `Liu Ting Zhou Isolation Forest 2008 ICDM` |
| 7 | Temporal deep models | `Malhotra LSTM encoder-decoder anomaly detection time series 2016` |
| 8 | Self-supervised learning | `TS2Vec universal representation learning time series 2022 AAAI`; `self-supervised learning physiological time series healthcare representation survey` |
| 9 | Change point detection | `Truong Oudre Vayatis selective review offline change point detection 2020` |
| 10 | Uncertainty estimation | `Lakshminarayanan deep ensembles uncertainty estimation 2017 NIPS` |
| 11 | Individual variability / longitudinal sensing | `studentlife smartphone sensing dataset longitudinal mental health Dartmouth` |

### 2.2 Inclusion

Included if the source addresses at least one of: anomaly/outlier detection
theory, temporal anomaly detection, change point detection, concept drift,
uncertainty estimation, self-supervised representation learning for time
series, digital phenotyping, wearable/passive sensing, personalized health
monitoring, or a candidate dataset.

### 2.3 Exclusion

- Sources with no identifiable author, venue, or year.
- Vendored/commercial product pages with no method disclosure.
- Sources whose only accessible text was an abstract that could not be
  cross-checked, unless marked as such in the matrix.

### 2.4 What this review is *not* — stated plainly

1. **Not systematic.** No database query strings, no PRISMA flow, no
   dual-reviewer screening, no formal risk-of-bias assessment.
2. **Not exhaustive.** Nineteen entries cannot represent a field this broad.
3. **Not independently verified.** Metadata was confirmed against
   publisher/DOI/repository records where reachable; several author lists and
   venue details (flagged in the matrix) remain unconfirmed and are marked
   accordingly rather than guessed.
4. **Potentially biased by the search engine's ranking**, which favours
   highly-cited and recent work.

These are real weaknesses. Phase 2 is therefore marked **incomplete**: it
establishes the landscape and tests the gap hypothesis, and it must be
upgraded before any publication claim.

---

## 3. Findings by theme

### 3.1 Anomaly detection: the foundations are settled

Chandola, Banerjee & Kumar (2009) remains the canonical taxonomy: anomalies as
**point / contextual / collective**, with detection framed as
**supervised / semi-supervised / unsupervised**. The survey's own list of
standing difficulties — *rarity of labelled anomalies, noise, the boundary
between normal and abnormal being ill-defined, and the fact that "normal"
changes over time* — reads almost as a specification for MEYRO.

**Implication:** MEYRO does not need to invent anomaly-detection machinery. It
needs to choose a reference frame and evaluate it honestly. This is a
*reduction* in claimed novelty, and a useful one.

### 3.2 Classical detectors are cheap, strong, and must not be skipped

Liu, Ting & Zhou (2008) introduced Isolation Forest: an ensemble that *isolates*
points rather than profiling normal regions, with linear time complexity and an
explicit claim of robustness and low memory cost. Its descendants
(Extended Isolation Forest, and the standard implementations shipped in
scikit-learn) are the default unsupervised baseline in applied work.

**Implication:** a temporal neural model that cannot beat Isolation Forest on
matched data is not contributing. Phase 8 exists to make this comparison
mandatory, and Phase 26's ablation must include it.

### 3.3 Temporal deep models: reconstruction error is the default, and it is not automatically better

Malhotra et al. (2016, EncDec-AD) trained a stacked LSTM encoder-decoder on
normal data and flagged anomalies by reconstruction error, showing robustness
across predictable, unpredictable, periodic, aperiodic, and quasi-periodic
series, and from windows as short as 30 steps.

**Implication:** the reconstruction-error deviation score is a mature, well-
understood mechanism and is the right first temporal model for MEYRO (Phase 9).
But the modern literature also documents that reconstruction-based detectors
can be *worse* than simple baselines on real benchmarks, so MEYRO must test
rather than assume. The architecture in the master specification (encoder →
temporal encoder → personal context → anomaly head) is a hypothesis, not a
result.

### 3.4 Self-supervision: strong, generic, and unproven for *personal* baselines

Yue et al. (2022, TS2Vec) learn hierarchical contrastive representations with
**timestamp-level** granularity, transferring across classification,
forecasting, and anomaly-detection tasks. The physiological-signal
self-supervision literature is active and, importantly, has begun to raise its
own critique: generic pretraining objectives can *obscure the clinical
semantics* needed for downstream transfer.

**Implication:** self-supervised pretraining is a plausible component of MEYRO
(Phase 12), but the honest position is that **a generic, population-level
pretrained representation may be exactly the wrong thing for a personalized
baseline** — it encodes what is typical *across* people. Phase 12 must
therefore include a personalization-aware comparator, not just TS2Vec, and
must be allowed to conclude "not worth it."

### 3.5 Concept drift: the vocabulary MEYRO needed, and a warning

Gama et al. (2014) formalise drift as **sudden, gradual, incremental, or
recurring**, and catalogue adaptation strategies: sliding windows, instance
weighting, ensembles, and explicit drift detectors.

Two things are directly load-bearing for MEYRO:

1. **The adaptation/detection conflict is a known hazard.** The survey's framing
   of adaptation as *necessary and hazardous* matches MEYRO's problem
   definition §2.2 exactly. MEYRO is not inventing this tension; it is
   inheriting it.
2. **The framing is supervised.** Concept drift is usually defined over
   `P(X, y)`. MEYRO's setting is *unsupervised* — there is no label stream to
   detect drift against. Adapting drift semantics to unsupervised personal
   baselines is a genuine, if modest, open problem.

### 3.6 Change point detection: how to define "when it changed"

Truong, Oudre & Vayatis (2020) organise offline change point detection into
*cost function + search method + constraint*, reviewing 140+ articles. This is
the formal machinery behind MEYRO's operational definition of deviation onset
and detection delay.

**Implication:** the "distributional shift proxy" label option in
`problem_definition.md` §7.1 is well-founded — but it measures **statistical
change**, not clinical meaning, and must be labelled as such everywhere.

### 3.7 Uncertainty: solvable with unglamorous methods

Lakshminarayanan, Pritzel & Blundell (2017) showed deep ensembles with
adversarial training and proper scoring rules give well-calibrated predictive
uncertainty, competitive with Bayesian approaches and far simpler.

**Implication:** MEYRO's Phase 13 does not require a novel uncertainty
mechanism. Deep ensembles plus calibration reporting is defensible, and
calibration curves (Phase 33, figure 6) are the honest way to present it.

### 3.8 Digital phenotyping and passive sensing: individual-scale data is real

Jain et al. (2015) coined *digital phenotype*; Onnela & Rauch (2016) argued
smartphone sensing enables continuous behavioural measurement at individual
scale. Empirically, Wang et al. (2014, StudentLife) sensed 48 students over a
10-week term with passive smartphone data and correlated the results with
PHQ-9 depression scores and academic outcomes. Nepal et al. (2024) extended
this to **two cohorts tracked across four years**, described as the longest
longitudinal mobile sensing study to date.

**Implication:** this is the strongest evidence in the review that MEYRO's
premise — repeated, individual-level observation over months to years — is
practically available. It also directly supports two of MEYRO's secondary
research questions: how much history a personal baseline needs (RQ-S3), and
how benefit varies across individuals (RQ-S4).

### 3.9 Applied personalized wearable monitoring: personalization is *pursued*, but how well is it *compared*?

Sunny et al. (2022) review wearables anomaly detection and explicitly note the
gap between wearable data volume and clinically meaningful anomaly detection.
Olyanasab & Annabestani (2024) review machine learning for **personalized**
wearable devices, treating personalization as a central design goal.
Gabrielli et al. (2025, *AI on the Pulse*) present a real-world wearable
health anomaly detection system.

**Implication:** personalization is mainstream in *intent*. The question this
review was built to answer is whether it is mainstream in *controlled
comparison* — see §4.

---

## 4. Testing the Phase 1 gap hypothesis

Phase 1 advanced three candidate gap components (G1–G3). Verdicts from this
first pass:

### G1 — "The population control condition is often untuned or a strawman"
**Status: plausible, partially supported.**

The applied works reviewed either omit a population baseline entirely (Sunny
et al. focus on detection methods, not reference frames) or treat it as
normalisation. None of the sources reviewed present a *tuned, information-
matched* population baseline as an explicit experimental control against which
personalization is adjudicated.

**Caveat:** absence in this scoping review is weak evidence. A database search
may well surface work that does exactly this. **Not yet claimed.**

### G2 — "Metric choice hides the false-positive mechanism"
**Status: weakly supported.**

The motivating mechanism (population thresholds over-alert on atypical-but-
stable individuals) is discussed qualitatively, and per-individual performance
is rarely reported in the sources reviewed. But the review did not run a
metric-by-metric audit of any paper's tables, which is what this claim requires.

**Action required:** before this is claimed, extract the reported metrics from
each candidate comparison paper and show that false-positive rate at matched
sensitivity is not the headline metric. That is a Phase 2 revision task.

### G3 — "Adaptation is under-examined as a confound"
**Status: supported as a *conceptual* gap, unverified as a *literature* gap.**

The concept-drift literature (Gama et al.) establishes the hazard rigorously,
but it does so in a supervised streaming setting. Nothing reviewed treats the
adaptation rate of a *personal* baseline as an explicit axis of the
personalized-vs-population comparison.

**Caveat:** the drift literature may already contain the needed analysis under
different terminology. **Not yet claimed.**

### Net effect on the Phase 1 gap statement

> The broad claim — *"personalized baselines are rarely compared to population
> baselines"* — is **too strong and is hereby narrowed.** Personalization is
> well-established in intent, and directly on-topic work exists (§5).
>
> The surviving, narrower, still-unverified candidate contribution is:
>
> **MEYRO would contribute a controlled, leakage-audited, per-individual
> comparison of a *tuned* population baseline against a personalized baseline
> on identical longitudinal data, with the adaptation/detection trade-off and
> false-positive rate at matched sensitivity reported as first-class outcomes
> rather than aggregate accuracy.**
>
> That is a **methodological and evaluative** contribution. MEYRO should not
> claim to have invented personalized baselines.

This narrowing is the single most important output of Phase 2.

---

## 5. Direct threats to MEYRO's novelty (stated openly)

These sources overlap MEYRO's stated idea closely enough that ignoring them
would be misconduct.

| Source | Overlap | Consequence for MEYRO |
|---|---|---|
| **"Personalized Baseline Modeling Using Machine Learning to Detect Anomalies in Longitudinal Wearable Sensor Data"** (2025, ResearchGate) | Uses the *exact* phrase "personalized baseline modeling" and "longitudinal wearable sensor data"; LSTM autoencoders | **The name and the basic idea are taken.** MEYRO cannot claim the concept. Its claim must rest on the *controlled comparison*, not the concept. This source must be obtained in full text and read before Phase 11. |
| **"Adaptive Baseline Modeling for Personalized [monitoring]"** (Zenodo, 2026) | Adaptive baselines + anomaly detection for personal monitoring | Overlaps Phase 10 (adaptive personal baseline). Must be read in full before claiming anything about adaptive baselining. |
| Valerio et al. (2024) | *Personalized* anomaly detection for wearable motion-artifact detection | Overlaps, but scoped to signal-quality/artifact detection rather than deviation-from-normal. Distinguish carefully. |
| Singh (2025) | Unsupervised anomaly detection as a personal "early warning" system | Very close in framing; published in a venue of unclear rigour and should be assessed on the full text, not the snippet. |

**Rule adopted:** whichever of these turns out to already contain the controlled
comparison, MEYRO's contribution statement is revised to exclude that
component. A smaller, true claim is acceptable; a large, false one is not.

---

## 6. Implications for MEYRO's design

Adopted from the literature (each traceable to a source in the matrix):

| # | Design decision | Justification |
|---|---|---|
| D1 | Reconstruction-error deviation score as the first temporal model | Malhotra et al. 2016 |
| D2 | Isolation Forest (and robust statistics) as mandatory non-neural comparators | Liu et al. 2008; Chandola et al. 2009 |
| D3 | Uncertainty via deep ensembles + explicit calibration reporting | Lakshminarayanan et al. 2017 |
| D4 | Drift taxonomy (sudden/gradual/incremental/recurring) adopted as the vocabulary for baseline adaptation | Gama et al. 2014 |
| D5 | Detection delay defined via change point detection formalisation | Truong et al. 2020 |
| D6 | Self-supervision treated as a *testable* component, not a default; must include a personalization-aware comparator | Yue et al. 2022 + the clinical-semantics critique |
| D7 | Robustness to device heterogeneity is a required evaluation axis, not a footnote | Stisen et al. 2015 (HHAR) |
| D8 | History-length sensitivity (cold start) is a first-class experiment | Wang et al. 2014; Nepal et al. 2024 |

**Explicitly *not* adopted:**
- No novel anomaly-detection algorithm is claimed.
- No novel uncertainty method is claimed.
- No claim that deep learning beats classical methods.
- No medical or diagnostic framing, in any source reviewed.

---

## 7. Limitations of this review

1. **Scoping, not systematic** (§2.4). No PRISMA, no dual screening, no
   risk-of-bias assessment.
2. **Only 19 sources**, captured from a search engine whose ranking is
   citation- and recency-biased. Long-tail relevant work is likely missed.
3. **Snippet-level evidence.** Several entries rest on search-result snippets
   plus repository/metadata pages, not full texts. Where a claim required the
   full text to substantiate, it is marked **not claimed** rather than asserted.
4. **Incomplete metadata.** Some author lists and venues are marked unverified
   in the matrix; nothing was filled in from memory and presented as verified.
5. **Publication bias.** Negative results about personalization are unlikely to
   be surfaced by this method, which systematically inflates the apparent
   strength of personalized approaches — and therefore *understates* the value
   of the comparison MEYRO proposes.
6. **No quantitative synthesis.** Metrics are not pooled; evaluation protocols
   across sources are heterogeneous.
7. **G2/G3 are not yet tested to the standard they need.** Both require metric-
   level extraction from primary sources, which has not been done.

---

## 8. Phase 2 revision plan (required before any publication claim)

- [ ] Re-run the search against **Scopus, Web of Science, PubMed, IEEE Xplore,
      ACM Digital Library** with documented query strings.
- [ ] Adopt a PRISMA-style flow diagram and record counts at each stage.
- [ ] Obtain and read in full the four overlapping works in §5.
- [ ] Extract reported metrics from each personalized-vs-population comparison
      paper, to actually test G2.
- [ ] Verify and complete all `source_verification` entries in the matrix.
- [ ] Add primary sources for One-Class SVM (Schölkopf et al. 2001) and Local
      Outlier Factor (Breunig et al. 2000), which Phase 8 requires and which
      were **not** verified in this pass.
- [ ] Re-test G1–G3 and rewrite §4 and the contribution statement accordingly.

Until these are done, the gap statement remains **provisional** and MEYRO is
**not** described as novel.

---

## 9. References

Metadata as verified in this pass; see `literature_matrix.csv` for the
verification status of every field. Ordered by year.

1. Chandola, V., Banerjee, A., Kumar, V. (2009). *Anomaly detection: A survey.* ACM Computing Surveys 41(3), Article 15. DOI 10.1145/1541880.1541882.
2. Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). *Isolation Forest.* IEEE ICDM 2008, 413–422. DOI 10.1109/ICDM.2008.17.
3. Reiss, A., Stricker, D. (2012). *Introducing a New Benchmarked Dataset for Activity Monitoring.* ISWC 2012. (PAMAP2)
4. Wang, R., et al. (2014). *StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students using Smartphones.* UbiComp 2014.
5. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). *A survey on concept drift adaptation.* ACM Computing Surveys 46(4). DOI 10.1145/2523813.
6. Jain, S.H., Brownstein, J.S. (2015). *The digital phenotype.* Nature Biotechnology.
7. Stisen, A., et al. (2015). *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition.* SenSys 2015. (HHAR)
8. Onnela, J.-P., Rauch, S.L. (2016). *Harnessing Smartphone-Based Digital Phenotyping to Enhance Behavioral and Mental Health.* Neuropsychopharmacology. PMC4869063.
9. Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., Shroff, G. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.* arXiv:1607.00148.
10. Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. arXiv:1612.01474.
11. Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018. (UCI id 465)
12. Truong, C., Oudre, L., Vayatis, N. (2020). *Selective review of offline change point detection methods.* Signal Processing 167:107299.
13. Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., Xu, B. (2022). *TS2Vec: Towards Universal Representation of Time Series.* AAAI-22. arXiv:2106.10466.
14. Sunny, J.S., et al. (2022). *Anomaly Detection Framework for Wearables Data: A Perspective Review on Data Types, Machine Learning, and Implementation.* Sensors 22(3):756.
15. Olyanasab, A., Annabestani, M. (2024). *Leveraging Machine Learning for Personalized Wearable Biomedical Devices: A Review.* PMC10890129.
16. Valerio, A., et al. (2024). *Development of a Personalized Anomaly Detection Model to [mitigate motion artifacts].* (institutional repository, UCC)
17. Nepal, S., et al. (2024). *A Four-Year Mobile Sensing Study of Mental Health …* ACM. DOI 10.1145/3643501.
18. Gabrielli, D., et al. (2025). *AI on the Pulse: Real-Time Health Anomaly Detection with wearable sensors.* arXiv:2508.03436.
19. *Personalized Baseline Modeling Using Machine Learning to Detect Anomalies in Longitudinal Wearable Sensor Data.* (2025) — **direct novelty threat; full text required.**

---

## 10. Change log

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-09-23 | First pass: 19 sources, gap hypothesis narrowed, §5 novelty threats documented, Phase 2 marked incomplete. |
