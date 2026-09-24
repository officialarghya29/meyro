"""Explainability engine for MEYRO (Phase 26).

Translates latent deviation vectors and per-feature z-scores into structured,
clinically safe explanations that never formulate unsupported medical diagnoses.
"""

from __future__ import annotations

from typing import Any


class DeviationExplainer:
    """Generates structured, non-diagnostic longitudinal deviation explanations."""

    @staticmethod
    def explain(
        subject_id: str,
        current_features: dict[str, float],
        baseline_stats: dict[str, dict[str, float]],
        anomaly_score: float,
        uncertainty: float,
        persistence_count: int,
        data_quality: float,
    ) -> dict[str, Any]:
        """Produces a structured explanation of what signals deviated from personal normal."""
        signal_breakdowns = []
        for feature, val in current_features.items():
            base = baseline_stats.get(feature, {"center": val, "scale": 1.0})
            center = base["center"]
            scale = base["scale"] if base["scale"] > 1e-4 else 1.0
            z_dev = (val - center) / scale
            diff_pct = ((val - center) / center * 100.0) if abs(center) > 1e-4 else 0.0

            signal_breakdowns.append({
                "signal": feature,
                "current_value": round(val, 2),
                "personal_baseline": round(center, 2),
                "z_deviation": round(z_dev, 2),
                "relative_change_pct": round(diff_pct, 1),
                "primary_contributor": bool(abs(z_dev) >= 2.0),
            })

        # Categorize magnitude
        if anomaly_score >= 0.75:
            magnitude = "high"
        elif anomaly_score >= 0.40:
            magnitude = "moderate"
        else:
            magnitude = "low"

        # Categorize confidence
        confidence_val = 1.0 - uncertainty
        if confidence_val >= 0.80 and data_quality >= 0.80:
            confidence_level = "high"
        elif confidence_val >= 0.50:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        return {
            "subject_id": subject_id,
            "is_meaningful_deviation": bool(anomaly_score >= 0.50),
            "deviation_magnitude": magnitude,
            "anomaly_score": round(anomaly_score, 4),
            "confidence_level": confidence_level,
            "confidence_score": round(confidence_val, 4),
            "sensor_quality": round(data_quality, 4),
            "persistence_observations": persistence_count,
            "signals": signal_breakdowns,
            "safety_disclaimer": (
                "MEYRO detected an empirical deviation from your historical baseline. "
                "This is a statistical observation of behavioral/physiological patterns, "
                "not a clinical or medical diagnosis. Seek professional medical guidance when appropriate."
            ),
        }
