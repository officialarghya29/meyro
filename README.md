<div align="center">

<img src="./assets/banner.svg" alt="MEYRO — AI that learns your normal" width="100%"/>

<br/>

<img src="./assets/meyro-logo.png" alt="MEYRO logo" width="96"/>

# MEYRO

### AI THAT LEARNS YOUR NORMAL

**A personalized longitudinal intelligence system that learns one person's<br/>
historical behavioral & physiological patterns — and detects meaningful<br/>
deviations from *that person's* baseline.**

<br/>

[![Status](https://img.shields.io/badge/status-phase%200%20%C2%B7%20initialization-48d8f0?style=flat-square&labelColor=000c24)](#project-status)
[![Research](https://img.shields.io/badge/research-in%20progress-90f0c0?style=flat-square&labelColor=000c24)](#research-question)
[![Not a diagnosis](https://img.shields.io/badge/MEYRO-not%20a%20medical%20diagnosis-e8b655?style=flat-square&labelColor=000c24)](#-safety-statement)
[![Python](https://img.shields.io/badge/python-3.12-48d8f0?style=flat-square&labelColor=000c24)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-90f0c0?style=flat-square&labelColor=000c24)](./LICENSE)

</div>

---

> **This repository is under active construction and follows a strictly gated,
> phase-by-phase research workflow.** No results, benchmarks, or claims exist yet.
> Everything published here will be reproducible, traceable to real artifacts,
> and honest about what has *not* been established. See [Project Status](#project-status).

---

## Table of Contents

- [The Problem](#the-problem)
- [Research Question](#research-question)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Status](#project-status)
- [Dataset Strategy](#dataset-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Limitations](#limitations)
- [Privacy](#privacy)
- [Ethics & Safety](#ethics--safety)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## The Problem

Conventional health and behavior analytics ask a **population** question:

> *"Is this value normal for a human?"*

Population thresholds are built from averages across thousands of people. They
are blind to the one thing that matters most for an individual: **you are not
average.** A resting heart rate, a sleep rhythm, a gait signature, or a daily
activity curve can sit comfortably inside population norms while being a
genuine, sustained departure from *that specific person's* own history — or
sit outside population norms every day of their life while being perfectly
normal *for them*.

This mismatch produces two failure modes:

| Failure mode | What happens |
|---|---|
| **False alarms** | Population thresholds flag healthy individuals whose physiology is simply atypical. |
| **Missed deviations** | A slow, personal drift never crosses a one-size-fits-all line, so it is never surfaced. |

MEYRO investigates whether reversing the reference frame — comparing a person
**to their own past** rather than to a crowd — detects meaningful longitudinal
change more reliably.

---

## Research Question

> **Can personalized baselines detect meaningful longitudinal deviations more
> effectively than population-level baselines?**

**Primary hypothesis.** A personalized baseline may reduce false positives
and/or improve detection of meaningful deviations compared with
population-level thresholds.

**This is a hypothesis, not a claim.** The experimental design is explicitly
built so that the evidence can **support it, refute it, or fail to establish
it.** The negative result is a legitimate, publishable outcome.

**Measurable objectives**

1. Implement a leakage-free population baseline (control) and a personalized
   baseline (treatment) over the same longitudinal signal.
2. Quantify detection quality across AUROC, AUPRC, precision, recall, F1,
   false-positive rate, sensitivity, specificity, detection delay, and
   calibration — with confidence intervals.
3. Report per-individual performance, not only aggregate averages.
4. Determine, by ablation, which components (personalization, adaptivity,
   temporal modeling, self-supervision, multimodality, uncertainty) actually
   contribute measured benefit.

---

## How It Works

MEYRO is **not** a health tracker, calorie counter, chatbot, symptom checker,
disease predictor, generic dashboard, or LLM wrapper. It does not diagnose.

The insight it tests is small and specific:

```
POPULATION BASELINE                 PERSONALIZED BASELINE
───────────────────                 ─────────────────────
   "normal for humans"                 "normal for THIS person"

      observation                         observation
          │                                   │
          ▼                                   ▼
   ┌─────────────┐                     ┌─────────────┐
   │  NORMAL     │  ← missed drift     │  DEVIATION  │  ← surfaced
   └─────────────┘                     └─────────────┘
```

Every deviation MEYRO reports must answer:

- **What** changed?
- **When** did it change?
- Compared with **what** (the personal baseline, over what window)?
- How **unusual** is it?
- How **long** has it persisted?
- **Which** signals contributed?
- How **confident** is MEYRO?
- How much **historical data** supports the baseline?

…and must always be framed as an observation about a personal pattern, never
as a medical conclusion.

---

## Architecture

Target architecture (built incrementally — **not** all implemented yet):

```
                              MEYRO
                                │
                          USER CONSENT
                                │
                                ▼
                       DATA COLLECTION
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
        Activity              Sleep              Physiology
           ▼                    ▼                    ▼
        Encoder              Encoder              Encoder
           │                    │                    │
           └────────────────────┼────────────────────┘
                                ▼
                       PERSONAL CONTEXT
                                ▼
                        TEMPORAL MODEL
                                ▼
                      PERSONAL BASELINE
                                ▼
                       DEVIATION ENGINE
                                ▼
                  UNCERTAINTY ESTIMATION
                                ▼
                     EXPLANATION ENGINE
                                ▼
                        SAFETY LAYER
                                ▼
                            API
                                ▼
                          MEYRO UI
```

The deliverable of each phase is a working, tested slice of this stack — never
a fabricated placeholder for a later phase.

---

## Project Status

MEYRO is built in gated phases. A phase does not begin until the previous one
is implemented, tested, and committed.

| # | Phase | Status |
|---|-------|:------:|
| 0 | Project initialization | ✅ **complete** |
| 1 | Research & problem definition | ⬜ pending |
| 2 | Literature review | ⬜ pending |
| 3 | Dataset discovery & selection | ⬜ pending |
| 4 | Data architecture | ⬜ pending |
| 5 | Data preprocessing pipeline | ⬜ pending |
| 6 | Population baseline (control) | ⬜ pending |
| 7 | Personalized baseline | ⬜ pending |
| 8 | Classical anomaly detection | ⬜ pending |
| 9 | Temporal deep-learning models | ⬜ pending |
| 10 | Adaptive personal baseline | ⬜ pending |
| 11 | **Primary research experiment** | ⬜ pending |
| 12 | Self-supervised learning | ⬜ pending |
| 13 | Uncertainty estimation | ⬜ pending |
| 14 | Explainability | ⬜ pending |
| 15 | Multimodal learning | ⬜ pending |
| 16 | Missing-modality robustness | ⬜ pending |
| 17 | Computer-vision signals | ⬜ pending |
| 18 | Voice / audio signals | ⬜ pending |
| 19 | Backend / API | ⬜ pending |
| 20 | Database | ⬜ pending |
| 21 | Security & privacy | ⬜ pending |
| 22 | Frontend | ⬜ pending |
| 23 | Integration | ⬜ pending |
| 24 | Testing | ⬜ pending |
| 25 | Data-leakage audit | ⬜ pending |
| 26 | Ablation study | ⬜ pending |
| 27 | Robustness | ⬜ pending |
| 28 | Fairness / subgroup analysis | ⬜ pending |
| 29 | MLOps & reproducibility | ⬜ pending |
| 30 | Deployment | ⬜ pending |
| 31 | Monitoring | ⬜ pending |
| 32 | Research results | ⬜ pending |
| 33 | Research paper | ⬜ pending |
| 34 | GitHub / public release | ⬜ pending |
| 35 | Final review | ⬜ pending |

---

## Dataset Strategy

**No datasets are bundled or downloaded in this repository.**

MEYRO prioritizes **public datasets with repeated observations from the same
subjects** — longitudinal structure is a hard requirement, because a
personalized baseline is meaningless without a personal history to learn from.

Each candidate dataset is documented in `docs/dataset_strategy.md` with its
license, cohort size, modalities, sampling rate, longitudinal availability,
missingness, leakage risks, and suitability. Data is only downloaded **after**
licensing and suitability review, and **sensitive health data is never placed
inside Git.**

---

## Installation

Requires **Python 3.12+**. Docker is optional and used only from the
deployment phase onward.

```bash
git clone https://github.com/officialarghya29/meyro.git
cd meyro

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

cp .env.example .env           # then edit locally — never commit .env
```

---

## Usage

```bash
# Run the test suite
pytest

# Lint and type-check
ruff check .
mypy src
```

The end-to-end research commands (`download_data` → `validate_data` →
`preprocess` → `create_splits` → `train` → `evaluate`) will be added as their
phases are implemented. **This README will always document the real commands
that actually work — never a wishlist.**

---

## Experiments

The primary experiment (Phase 11) compares a **population baseline (control)**
against a **personalized baseline (treatment)** on identical, leakage-free
splits. Its protocol is frozen in `docs/experimental_protocol.md` **before**
final evaluation and is not altered after seeing test results.

Guarantees that will hold for every reported experiment:

- Splits are by **subject and time** — no subject leakage, no temporal leakage.
- The test set is never trained on, normalized with, or tuned against.
- Every model has a documented baseline for comparison.
- Every experiment is reproducible from a recorded config + seed.

*No experiments have been run yet.*

---

## Results

> **There are no results yet.**

This section is intentionally empty. MEYRO will not publish numbers it has not
produced, and it will not soften or omit a result that contradicts its
hypothesis. When experiments exist, this section will contain real figures,
real tables, and real statistical tests generated from
`results/figures/`, `results/tables/`, and `results/statistical_tests/`.

---

## Limitations

MEYRO is honest about what it is not and what it cannot yet do:

- **Not a medical device.** No regulatory clearance or compliance is claimed.
- **Not a diagnostic tool.** It describes deviations from a personal pattern,
  not disease.
- **No clinical validation.** Any future health relevance requires independent
  clinical study, which this project does not claim to provide.
- **Personalization needs history.** A new user has no baseline; cold-start
  behaviour is a genuine open problem.
- **Signal ≠ meaning.** A detected deviation is a statistical observation, and
  may be benign, artefactual, or noise.
- **Dataset-dependent.** Conclusions will be bounded by the populations,
  devices, and conditions of the datasets actually used.

---

## Privacy

MEYRO treats behavioral and physiological data as **sensitive by default**.

- No personal or health data is ever committed to this repository.
- Secrets live only in local `.env` files, never in source.
- The system is designed around **minimum data collection**, pseudonymization,
  consent records, user-initiated export, and user-initiated deletion.
- Security and privacy architecture (threat model, consent, encryption,
  audit logs) is specified in `SECURITY.md`, `docs/privacy.md`, and
  `docs/threat_model.md` as its phase is implemented.

No claim of HIPAA, GDPR, DPDP, or other regulatory compliance is made unless
and until a formal assessment says so.

---

## Ethics & Safety

<div align="center">

### ⚕️ SAFETY STATEMENT

</div>

MEYRO will **never** output:

- ❌ *"You have disease X."*
- ❌ *"You're definitely healthy."*
- ❌ *"You are safe."*
- ❌ *"You don't need a doctor."*

MEYRO **will** say things like:

- ✅ *"MEYRO detected a deviation from your historical baseline."*
- ✅ *"This result is not a medical diagnosis."*
- ✅ *"Consider this in context and seek professional medical advice when appropriate."*

The exact production wording is reviewed before any public deployment.

---

## Reproducibility

Another researcher should be able to clone MEYRO and reproduce the primary
experiment from a recorded commit, config, dataset version, and seed.

Reproducibility machinery (experiment tracking, dataset/model versioning,
containerized environments, frozen protocols) is introduced in its own phase
and documented in `docs/reproduction.md`. Until then, this README states
plainly: **no experiment is reproducible yet, because no experiment has run.**

---

## Roadmap

```
NOW   ── Phase 0   project initialization (skeleton, tooling, governance)
NEXT  ── Phases 1–5   research definition, literature, data architecture,
                       preprocessing & leakage-safe splits
THEN  ── Phases 6–11  population baseline → personalized baseline →
                       anomaly detection → adaptive baselines →
                       PRIMARY EXPERIMENT
LATER ── Phases 12–18 self-supervised learning, uncertainty, explainability,
                       multimodality, robustness, vision, voice
SHIP  ── Phases 19–35 API, database, security, frontend, integration,
                       testing, audits, MLOps, deployment, paper, release
```

---

## Citation

If you reference MEYRO, please cite it as below (see [`CITATION.cff`](./CITATION.cff) for the machine-readable form):

```bibtex
@software{meyro,
  title        = {MEYRO: Personalized Baseline Modeling for Longitudinal
                  Multimodal Health Anomaly Detection},
  author       = {Bose, Arghya},
  year         = {2026},
  url          = {https://github.com/officialarghya29/meyro},
  note         = {Research in progress — no results published yet}
}
```

> The citation will only claim novelty that the Phase 2 literature review
> actually supports.

---

## License

Released under the **MIT License** — see [`LICENSE`](./LICENSE).

---

<div align="center">

<img src="./assets/meyro-logo.png" alt="MEYRO" width="44"/>

**MEYRO** · *AI that learns your normal.*

Built as reproducible research — and honest about what has not been proven yet.

</div>
