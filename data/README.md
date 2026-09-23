# MEYRO Data Policy

**No personal or health data is stored in this repository. Ever.**

This directory holds *placeholders only* so the pipeline has stable paths.
Its contents are git-ignored except for this file and `.gitkeep` markers.

```
data/
├── raw/        # immutable source data, exactly as downloaded. Never edited.
├── interim/    # partially processed, reproducible from raw/
├── processed/  # model-ready outputs, reproducible from raw/ + configs
└── external/   # third-party reference data (e.g. population statistics)
```

## Rules

1. **Licensing first.** No dataset is downloaded until its license and
   suitability have been reviewed and recorded in `docs/dataset_strategy.md`.
2. **Never committed.** Raw, interim, and processed data never enter Git, even
   in "anonymized" form. Only `data/README.md` and `.gitkeep` are tracked.
3. **Raw is immutable.** `raw/` is written once and never modified by code.
   Every later stage is derived from it, so the pipeline stays reproducible.
4. **Reproducible by construction.** `interim/` and `processed/` must be
   regenerable from `raw/` plus a recorded config and seed.
5. **Pseudonymize on ingest.** Subject identifiers are replaced with stable
   pseudonyms before data reaches any modeling stage.
6. **Minimum necessary.** Collect and retain the least data the research
   question requires — nothing more.

## Reproducing the pipeline

The exact commands for a real, end-to-end run will be documented here and in
`README.md` as their phases are implemented. **Until then, this file describes
intent, not a working pipeline.**
