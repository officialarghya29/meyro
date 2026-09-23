"""MEYRO — AI that learns your normal.

Personalized longitudinal baselines for meaningful deviation detection.

MEYRO compares an individual's observations against *their own* historical
baseline rather than against population-level thresholds.

SAFETY SCOPE
------------
MEYRO is research software. It is **not** a medical device, it does **not**
diagnose disease, and its outputs must never be represented as clinical
advice. Reported deviations are statistical observations about a personal
pattern, not medical conclusions.
"""

from __future__ import annotations

__all__ = ["TAGLINE", "__version__"]

__version__ = "0.0.1"

TAGLINE = "AI that learns your normal."
