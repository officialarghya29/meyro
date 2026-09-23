# MEYRO — Problem Definition

> **Phase 1 artifact.** This document defines *what MEYRO is investigating* and
> *what would count as an answer*. It contains **no results and no claims of
> proven benefit.** Every statement about existing literature is marked
> provisional and is to be verified or revised in the Phase 2 literature
> review. Every objective below must be falsifiable.

| | |
|---|---|
| **Project** | MEYRO — *AI that learns your normal.* |
| **Document status** | Draft v0.1 — Phase 1 |
| **Depends on** | — |
| **Superseded by** | Phase 2 (`literature_review.md`) will validate or refute the research gap; Phase 11 (`experimental_protocol.md`) will freeze the evaluation. |

---

## 1. Core statement

> **MEYRO investigates whether personalized longitudinal baselines can identify
> meaningful deviations in individual behavioral/physiological signals more
> effectively, or with fewer false positives, than population-level baselines.**

This is a **question**, not a finding. MEYRO does **not** assume that
personalization wins.

---

## 2. The problem

Most deployed health and behavior analytics are built on **population
reference ranges**. A value is judged by where it falls in a distribution
estimated across a large, heterogeneous cohort. This framing answers:

> *"Is this value typical for a human?"*

but the question a longitudinal monitoring system actually needs to answer is:

> *"Is this value typical for **this** human, given **their** own history?"*

These are different questions, and they can disagree in both directions.

### 2.1 Two failure modes of population baselining

**Failure mode A — false alarms on atypical-but-healthy individuals.**
Inter-individual variation in behavioral and physiological signals is large.
A person whose resting physiology, sleep architecture, or activity profile sits
naturally at the tail of the population distribution is *permanently flagged*
against a population threshold, while their own trajectory is entirely stable.
The alert carries no information about **change**.

**Failure mode B — missed personal change.**
A slow, sustained drift — the kind longitudinal monitoring is ostensibly for —
may never cross a population boundary, because the person's starting point was
not near that boundary. The population baseline is *insensitive to the
individual's own reference frame*, so a meaningful personal transition stays
below threshold.

Both failure modes share one root cause: **the reference distribution is the
wrong reference distribution.** Population baselines encode between-person
variation, but the quantity of interest for a single monitored individual is
within-person variation over time.

### 2.2 Why this is not merely a preprocessing choice

Choosing a reference distribution is not a hyperparameter. It changes:

- **what counts as an event** (a deviation from *what*?),
- **the achievable false-positive rate** (calibrated against between-person or
  within-person spread),
- **what the system's output can mean** (a population percentile is a statement
  about a cohort; a personal deviation is a statement about a trajectory),
- **how the system must behave over time** (population statistics are static;
  a personal baseline must adapt — and must be prevented from adapting to the
  very change it is meant to detect).

That last point is the central technical tension, and MEYRO takes it as the
heart of the problem rather than an implementation detail:

> **Adaptation/detection conflict.** A baseline that tracks recent data too
> aggressively absorbs a genuine deviation as "the new normal" and never
> reports it. A baseline that is too rigid cannot accommodate ordinary
> non-stationarity (seasonality, ageing, recovery, lifestyle change) and
> produces persistent false positives. MEYRO must **distinguish temporary
> deviation, persistent deviation, gradual drift, and noise** — see §7.

---

## 3. Motivation

1. **Monitoring is inherently longitudinal.** Wearables, phones, and home
   devices produce repeated observations of the same person over months to
   years. Population cross-sections discard precisely the structure that this
   data modality makes available.
2. **Inter-individual variation is large and structured.** The same nominal
   signal can differ by person for reasons that are stable and benign
   (body composition, fitness, device placement, chronotype). Stable
   between-person differences are a poor basis for within-person change.
3. **False positives are not harmless.** In longitudinal monitoring, a system
   that fires constantly is ignored, and an ignored system detects nothing.
   False-positive rate is therefore not a secondary metric — for a monitoring
   tool it is close to the product.
4. **Rare labels are the norm.** True physiological/behavioral change events are
   sparse and often unlabelled. A method that depends on abundant positive
   labels is a poor fit. This motivates label-efficient and
   self-supervised directions (Phases 12+) — but only if experiments justify
   keeping them.
5. **The idea is testable now.** Answering the question requires public
   longitudinal data, explicit baselines, leakage-controlled evaluation, and
   standard metrics. It does **not** require clinical endpoints, new sensing
   hardware, or regulatory approval.

---

## 4. Research gap

> ⚠️ **PROVISIONAL.** The gap below is the working hypothesis of this document.
> It is **not** a verified claim about the literature. Phase 2 exists to test
> it, and Phase 2 is permitted — expected — to narrow, relocate, or falsify it.
> MEYRO must not be described as novel until Phase 2 supports that description.

The working gap statement:

> Personalized baselines are widely *used* in applied health monitoring, yet
> the **comparison between a personalized baseline and a population-level
> baseline is rarely treated as the primary, controlled object of study.**
> Population baselines frequently appear as a data-normalization step or as a
> strawman rather than as a tuned, fairly-evaluated control condition.

Concretely, the gap has three candidate components to be checked in Phase 2:

- **G1 — Control condition is often untuned.** When personalization is
  reported to outperform population thresholds, it is unclear whether the
  population baseline received comparable tuning effort and information.
- **G2 — Metric choice hides the mechanism.** Population baselines are claimed
  to be worse primarily via aggregate accuracy/AUROC; the *false-positive*
  mechanism motivating personalization is less often quantified directly, and
  per-individual performance is infrequently reported.
- **G3 — Adaptation is under-examined as a confound.** Adaptive/personal
  baselines risk absorbing true change. The adaptation/detection trade-off is
  rarely evaluated as an explicit axis of the comparison.

**Rule:** if Phase 2 shows any of G1–G3 is already well addressed, that
component is removed from MEYRO's claimed contribution and the gap statement
is rewritten. A narrower, true gap beats a broad, false one.

---

## 5. Hypothesis

### H1 (alternative)
Personalized longitudinal baselines detect meaningful deviations in an
individual's behavioral/physiological signals **more effectively** and/or
**with a lower false-positive rate** than population-level baselines, under a
matched, leakage-free evaluation protocol.

### H0 (null)
There is no difference between personalized and population-level baselines on
the primary outcome(s), beyond what is attributable to chance.

### Outcomes of the study

The experiment must be capable of producing **any** of these, and all are
legitimate results to report:

| Outcome | Interpretation |
|---|---|
| **H1 supported** | Personalization improves detection and/or false-positive behaviour under the frozen protocol. |
| **H0 not rejected** | No detectable difference; personalization's value is not established by this design. |
| **H1 contradicted** | Population baselines match or beat personalization; personalization adds cost without benefit. |
| **Inconclusive** | Effect present but underpowered / confounded (e.g. too little history per subject to build a baseline). |

An inconclusive or negative result is a **finding**, and will be reported with
the same prominence as a positive one. The project's success criteria (§50 of
the master specification) do **not** include "the hypothesis is confirmed".

---

## 6. Research questions

**Primary**

- **RQ-P.** Are personalized longitudinal baselines more effective than
  population-level baselines at identifying meaningful deviations in an
  individual's behavioral/physiological signals?

**Secondary**

- **RQ-S1 (false positives).** Do personalized baselines reduce the
  false-positive rate relative to a tuned population baseline, at matched
  sensitivity?
- **RQ-S2 (adaptation).** How do adaptation rate and window length trade off
  detection sensitivity against absorption of true change? Where is the
  operating point, and does it differ across subjects?
- **RQ-S3 (history).** How much personal history is required before a
  personalized baseline matches or exceeds the population baseline? Is there a
  measurable cold-start penalty?
- **RQ-S4 (heterogeneity).** Does the benefit of personalization vary
  systematically across individuals, and can that variation be predicted from
  observable properties (baseline stability, signal-to-noise, data density)?
- **RQ-S5 (complexity).** Do temporal deep models and/or self-supervised
  representations add measurable benefit over simple statistical personal
  baselines, once both are fairly tuned?
- **RQ-S6 (robustness).** How does the comparison degrade under missing
  modalities, sampling-rate changes, and device differences?
- **RQ-S7 (uncertainty).** Can the system distinguish "no deviation" from
  "insufficient data to judge"? Is its stated confidence calibrated?

**Deliberately deferred** (not asked until the single-modality study is
stable): multimodal fusion, vision, and voice questions (Phases 15–18).

---

## 7. Operational definitions

Ambiguity here would make the experiment unfalsifiable, so terms are fixed.

| Term | Operational definition |
|---|---|
| **Signal** | A time-indexed, numeric series derived from one modality for one pseudonymous subject. |
| **Baseline** | An estimate of the subject's *expected* signal behaviour, computed **only** from observations strictly earlier than the observation being evaluated. |
| **Population baseline** | A baseline estimated by pooling subjects, with no subject-specific parameters. **The control condition.** |
| **Personalized baseline** | A baseline whose parameters are estimated from that subject's own prior observations only. |
| **Adaptive baseline** | A personalized baseline whose parameters update over time according to an explicit, configurable rule. |
| **Deviation score** | A scalar produced by comparing an observation to the applicable baseline. Higher = more unusual for the reference frame. |
| **Deviation event** | A deviation score exceeding a threshold, where the threshold is chosen by a protocol fixed in advance. |
| **Meaningful deviation** | A deviation event that is **both** (a) statistically unusual relative to the reference frame **and** (b) persistent beyond a pre-specified minimum duration. Persistence is required to separate genuine change from single-sample noise. |
| **Persistence** | The number of consecutive observations over which a deviation condition holds. |
| **Detection delay** | The number of observations between true onset and the first correctly attributed deviation event. |
| **Cold start** | The period during which a subject has insufficient history to form a personalized baseline. |

### 7.1 The labelling problem (central, and stated honestly)

"Meaningful deviation" is only measurable if some notion of the true onset or
presence of a deviation exists. MEYRO does **not** have access to disease
labels and will not use them as the primary target. Candidate
operationalizations, to be evaluated in Phases 3–5 and frozen in Phase 11:

1. **Held-out natural change points** — onsets annotated by the dataset
   (e.g. transitions, condition-change dates, intervention start/stop).
2. **Induced deviations** — controlled perturbations applied to held-out
   segments, with ground-truth onset and duration known by construction.
3. **Self-reported events** — subject-reported labels where a dataset provides
   them (e.g. illness episodes), used as a *secondary*, noisy signal.
4. **Distributional shift proxies** — statistically defined change points in
   held-out data, acknowledging that these measure change, **not** clinical
   meaning.

> **Honesty constraint.** A statistically defined change point is not a medical
> event. If MEYRO reports performance against an induced or statistical
> definition of deviation, that must be stated in every paper, figure, and
> table. MEYRO must never present deviation-detection performance as
> diagnostic performance.

This is the single largest threat to the validity of the study and is
escalated to a first-class research decision rather than a footnote.

---

## 8. Assumptions

Stated so they can be attacked.

- **A1.** Public longitudinal datasets exist that contain repeated observations
  from the same individuals at a usable cadence, with licenses permitting
  research use.
- **A2.** Within-person variation is, for the signals studied, sufficiently
  structured that a personal baseline estimated from history has predictive
  value for that person's future values.
- **A3.** "Deviation" can be operationalized as in §7.1 without resorting to
  disease labels.
- **A4.** Sufficient history exists per subject to estimate a personal baseline
  for a non-trivial fraction of the cohort (otherwise RQ-P is untestable on
  that dataset).
- **A5.** Signals are comparable within a subject over time (device and
  processing stay stable enough that change is attributable to the subject,
  not the instrument) — or such changes can be detected and controlled.
- **A6.** Metrics computed on simulated/induced deviations transfer, at least
  directionally, to naturally occurring deviations. **This is a real
  assumption and will be tested, not presumed.**

---

## 9. Ethical constraints

These constrain the design, not just the write-up.

- **E1 — No diagnosis.** MEYRO must never state or imply that a user has a
  disease, is healthy, is safe, or does not need care. Output is framed as an
  observation about a personal pattern.
- **E2 — No unsupported medical claims.** No claim of clinical utility,
  diagnostic accuracy, or regulatory compliance without evidence and formal
  assessment. No HIPAA/GDPR/DPDP compliance claim unless formally assessed.
- **E3 — Uncertainty is surfaced, never hidden.** Confidence, data quality,
  and historical support are part of the output, not optional metadata.
- **E4 — No real health data in the repository.** Development, tests, and
  demos use public, licensed, or synthetic data only.
- **E5 — Minimum data, consent, deletion.** Collect the least data the
  question requires; support consent, export, and deletion as product
  features, not afterthoughts.
- **E6 — Honest reporting.** Negative, inconclusive, and inconvenient results
  are reported. Failure analysis is a deliverable, not something to hide.
- **E7 — No fabrication.** No invented datasets, metrics, citations,
  participant counts, or significance.
- **E8 — Fairness.** Subgroup performance is examined where metadata permits,
  and poor subgroup performance is reported rather than buried (Phase 28).
- **E9 — Staging.** No public-facing demo built on a non-existent model; no
  claim made before the corresponding experiment exists.

---

## 10. Non-goals

MEYRO is explicitly **not**:

- ❌ a diagnostic system or disease predictor,
- ❌ a symptom checker,
- ❌ a generic health/calorie/fitness tracker,
- ❌ a chatbot or LLM wrapper (an LLM may never substitute for the model),
- ❌ a generic dashboard,
- ❌ a medical device, and claims no regulatory status,
- ❌ a production clinical product,
- ❌ a claim that personalization *is* better — that is the open question,
- ❌ a platform for collecting users' real health data at this stage.

---

## 11. Measurable objectives

Each is falsifiable and maps to a phase. "Demonstrated" means measured under
the frozen protocol, on leakage-free splits, with confidence intervals.

| # | Objective | Phase | Falsified if |
|---|---|---|---|
| **O1** | Build a leakage-free pipeline with subject- and time-aware splits | 5 | any subject or future information reaches training |
| **O2** | Implement a **tuned** population baseline as control | 6 | control is untuned or mis-specified relative to treatment |
| **O3** | Implement a personalized baseline using only prior history | 7 | baseline uses the evaluated observation or later data |
| **O4** | Establish classical anomaly-detection references | 8 | no baseline comparison exists per model |
| **O5** | Freeze the primary protocol **before** touching test data | 11 | protocol changes after test results are seen |
| **O6** | Compare personal vs population on AUROC, AUPRC, precision, recall, F1, FPR, sensitivity, specificity, detection delay, calibration, individual-level performance, with CIs | 11 | any metric reported without CI where appropriate, or subset-cherry-picked |
| **O7** | Quantify the adaptation/detection trade-off (RQ-S2) | 10–11 | adaptivity claimed beneficial without measurement |
| **O8** | Measure the cold-start penalty and required history (RQ-S3) | 10–11 | history requirement asserted rather than measured |
| **O9** | Decompose performance by individual and identify predictors of benefit (RQ-S4) | 11, 28 | only aggregate means are reported |
| **O10** | Ablate every component: personalization, adaptivity, temporal model, context, modality, self-supervision, uncertainty | 26 | a component's value is assumed rather than measured |
| **O11** | Test robustness: noise, missing data, sampling rate, device, short/long history, drift, shift, outliers | 27 | failures are not recorded and reported |
| **O12** | Audit for subject, temporal, feature, label, normalization, hyperparameter, and augmentation leakage | 25 | the audit is skipped or its findings omitted |
| **O13** | Make the primary experiment reproducible from a commit + config + seed | 29–30 | a second researcher cannot reproduce the headline result |
| **O14** | Report uncertainty and data quality alongside every deviation | 13 | a bare score is presented as if it were a conclusion |
| **O15** | Explain every deviation: what, when, compared with what, how unusual, persistence, contributing signals | 14 | output cannot answer these from stored artifacts |

### 11.1 Decision rule set in advance

To prevent post-hoc rationalisation, the following is fixed **now**:

- **Primary metric:** to be chosen in Phase 11 and frozen in
  `experimental_protocol.md` **before** test evaluation. The candidate primary
  metric is **AUPRC** on deviation events (positives are rare), with AUROC and
  **FPR at matched sensitivity** reported as co-primary.
- **Significance:** appropriate statistical test per design (paired comparison
  across subjects where applicable), with confidence intervals; the test is
  chosen in Phase 11 and not changed afterwards.
- **No test-set tuning.** Model/threshold selection happens on validation data.
- **No cherry-picking.** All pre-registered metrics are reported for all
  conditions, including the ones that lose.
- **Negative results ship.** If personalization does not help, that is the
  headline.

---

## 12. Falsification criteria

MEYRO's central claim is **disconfirmed** if, under the frozen protocol with a
tuned population control:

1. Personalized and population baselines are statistically indistinguishable on
   the primary metric **and** on FPR at matched sensitivity; or
2. The population baseline is superior on the primary metric across datasets
   and subjects, with no subgroup in which personalization reliably helps; or
3. Any apparent advantage of personalization is fully explained by leakage
   (O12) or by unequal tuning between conditions (O2), so the comparison is
   invalid.

Any of these is a reportable, publishable outcome. Reporting it correctly is
worth more to the project than obscuring it.

---

## 13. Open questions carried into later phases

- **Q1.** Is deviation detection even well-posed without clinical labels — and
  how much does the induced-deviation assumption (A6) inflate apparent
  performance?
- **Q2.** What is the correct way to compare adaptation rates between
  personalized and population baselines fairly? (Population baselines can also
  be made rolling/adaptive; a strawman control is a validity threat — see G1.)
- **Q3.** How should per-individual results be aggregated without hiding
  catastrophic failures on a minority of subjects?
- **Q4.** How much of the personalization benefit is explained by simple
  robust statistics versus learned representations? (Motivates RQ-S5 and the
  ablation in O10.)
- **Q5.** Which modality gives the strongest, most honest first test of the
  hypothesis? (Phase 3/4 — likely activity/physiological time series, chosen
  for longitudinal availability, not novelty.)

---

## 14. Change log

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-09-23 | Initial Phase 1 problem definition. Gap statement provisional pending Phase 2. |

---

## Appendix — one-paragraph summary

Population health thresholds answer a question about people in general, while
longitudinal monitoring needs an answer about one person in particular. MEYRO
asks whether replacing the population reference distribution with a
personalized, history-only baseline improves the detection of meaningful
deviations — or at least reduces the false alarms that make monitoring systems
behave like noise. The question is answered by a controlled comparison against
a *tuned* population baseline on public longitudinal data, with leakage-free
subject- and time-aware splits, explicit pre-registered metrics, reported
uncertainty, and an honest account of the adaptation/detection trade-off.
MEYRO does not assume personalization wins, does not diagnose disease, and will
report a negative result if that is what the evidence shows.
