# Security Policy — MEYRO

MEYRO processes behavioral and physiological signals. Even in research form,
**treat all data flowing through this project as sensitive health data.**

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately to:

- **Email:** officialarghya29@gmail.com
- **GitHub:** open a private security advisory via the repository's
  *Security → Advisories → Report a vulnerability* tab.

Include: affected component, reproduction steps, impact, and any suggested
fix. You will receive an acknowledgement, and a disclosure timeline will be
agreed once the issue is confirmed.

## Supported versions

MEYRO is pre-release research software (`0.0.x`). Only the latest `main` is
supported. There are no backports.

## Security principles for this project

These are enforced throughout development:

1. **No secrets in source.** Configuration lives in local `.env` files, which
   are git-ignored. `.env.example` contains placeholders only.
2. **No health or personal data in Git.** `data/**` and `models/**` are
   ignored; only `.gitkeep` placeholders are tracked.
3. **Authentication & authorization** are implemented for every non-public
   endpoint (Phase 19–21).
4. **Least privilege** for database users and service accounts.
5. **Input validation** on all external input; no unchecked deserialization.
6. **Encryption in transit** and, where appropriate, **at rest**.
7. **Audit logging** of access to sensitive resources — without logging the
   sensitive values themselves.
8. **Rate limiting** on authentication and data-ingestion endpoints.
9. **Consent, export, and deletion** flows are first-class product features.
10. **Dependency auditing** before deployment.

## Out of scope

- MEYRO is **not** a medical device and makes **no** clinical claims.
- No claim of HIPAA, GDPR, DPDP, or other regulatory compliance is made unless
  a formal assessment has been completed and documented.

## Known security debt

Phase 0 contains no network services, no authentication, and no data
collection. This file will be expanded as those phases land.
