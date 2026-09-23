# Contributing to MEYRO

Thanks for your interest. MEYRO is **research software**, so contributions are
judged first on scientific validity and reproducibility, then on code quality —
and only then on appearance.

## The one rule that matters most

> **Never fabricate, fake, or overstate anything.**

That applies to data, results, metrics, benchmarks, citations, medical claims,
dataset sizes, and participant counts. A negative or failing result that is
reported honestly is a valuable contribution. A fabricated positive result is
grounds for rejection.

## Development workflow

MEYRO is built **phase by phase**. Each phase must be:

1. Explained (objective, why it is needed, files touched).
2. Implemented in isolation — no premature future-phase work.
3. Tested (`pytest`) and validated.
4. Documented.
5. Committed as a coherent, working state.

Do not bundle unrelated phases into one pull request. Do not leave `main` in a
broken state.

## Getting set up

```bash
git clone https://github.com/officialarghya29/meyro.git
cd meyro
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

## Before you open a PR

```bash
ruff check .        # lint
ruff format --check .
mypy src            # types
pytest              # tests
```

All four must pass. Critical test failures block merge.

## Engineering standards

- **Modular architecture**, small focused modules.
- **Python** for ML/research; **type hints** everywhere.
- **Meaningful names**; no `data1`, `tmp`, `x2`.
- **Tests** for new logic, including edge cases.
- **Configuration files** instead of hardcoded parameters.
- **Deterministic seeds** wherever randomness exists.
- **Log important operations**; never silently swallow errors.
- Document major architectural decisions.

## Scientific standards

- Prevent **subject leakage** and **temporal leakage** in every split.
- Never train on test data. Never tune on the final test set.
- Ensure the baseline is built **only** from data available *before* the
  observation being evaluated.
- Every model needs a documented baseline for comparison.
- Every claimed improvement must be experimentally demonstrated, not asserted.
- Freeze the evaluation protocol **before** looking at test results.

## Privacy

- Never commit health or personal data — not even "anonymized" examples.
- Use synthetic or public, licensed data in tests.
- Keep secrets in `.env` (git-ignored), never in source.

## Style

- `ruff` is the source of truth for linting and formatting.
- Conventional-commit style messages, e.g.
  `feat: implement personalized baseline engine`,
  `fix: correct temporal split boundary`,
  `docs: add dataset leakage notes`.

## Reporting problems

Bugs and research-design concerns → GitHub issues.
Security vulnerabilities → see [`SECURITY.md`](./SECURITY.md). Do not file a
public issue for security problems.

## Code of conduct

Be rigorous, be kind, and be honest. Critique ideas, not people. Assume good
faith. Prioritize the integrity of the research over the appearance of
success.
