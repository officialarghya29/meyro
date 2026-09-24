"""FastAPI application for MEYRO (Phase 19).

Provides endpoints for:
- Health check
- Ingesting observations
- Querying personal baseline state
- Scoring causal deviations
- Explaining detected deviations with safety disclaimers
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from meyro.anomaly.explanation import DeviationExplainer
from meyro.personalization.adaptive import AdaptivePersonalBaseline

app = FastAPI(
    title="MEYRO Intelligence Engine",
    description="Personalized Longitudinal Health Baseline & Anomaly Detection API",
    version="1.0.0",
)

# In-memory baseline state store for demonstration/API serving
adaptive_engine = AdaptivePersonalBaseline(
    feature_cols=["step_count", "resting_hr", "sleep_duration_min"],
    adaptation_rate=0.05,
    anomaly_rejection_threshold=2.5,
)

# Initialize sample subject baselines
adaptive_engine.state["SUBJ_001"] = {
    "step_count": {"center": 9200.0, "scale": 1100.0},
    "resting_hr": {"center": 62.0, "scale": 3.8},
    "sleep_duration_min": {"center": 460.0, "scale": 35.0},
}


class MeasurementPayload(BaseModel):
    subject_id: str = Field(...)
    timestamp: str = Field(...)
    features: dict[str, float] = Field(...)
    quality_score: float = Field(0.95, ge=0.0, le=1.0)


class DeviationResponse(BaseModel):
    subject_id: str
    deviation_score: float
    is_meaningful_deviation: bool
    deviation_magnitude: str
    confidence_level: str
    explanation: dict[str, Any]


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "system": "MEYRO Longitudinal Engine",
        "version": "1.0.0",
        "active_subjects": len(adaptive_engine.state),
    }


@app.get("/baseline/{subject_id}")
def get_baseline(subject_id: str):
    if subject_id not in adaptive_engine.state:
        raise HTTPException(status_code=404, detail="Subject baseline not found")
    return {
        "subject_id": subject_id,
        "baseline_setpoints": adaptive_engine.state[subject_id],
        "status": "calibrated",
    }


@app.post("/deviations/score", response_model=DeviationResponse)
def score_deviation(payload: MeasurementPayload):
    subj_id = payload.subject_id
    if subj_id not in adaptive_engine.state:
        # Default initialization for cold-start subject
        adaptive_engine.state[subj_id] = {
            col: {"center": payload.features.get(col, 0.0), "scale": 1.0}
            for col in adaptive_engine.feature_cols
        }

    res = adaptive_engine.step(
        subject_id=subj_id,
        features=payload.features,
        quality_score=payload.quality_score,
    )

    dev_score = res["deviation_score"]
    explanation = DeviationExplainer.explain(
        subject_id=subj_id,
        current_features=payload.features,
        baseline_stats=adaptive_engine.state[subj_id],
        anomaly_score=min(1.0, dev_score / 3.0),
        uncertainty=1.0 - payload.quality_score,
        persistence_count=1 if dev_score >= 1.5 else 0,
        data_quality=payload.quality_score,
    )

    return DeviationResponse(
        subject_id=subj_id,
        deviation_score=round(dev_score, 4),
        is_meaningful_deviation=explanation["is_meaningful_deviation"],
        deviation_magnitude=explanation["deviation_magnitude"],
        confidence_level=explanation["confidence_level"],
        explanation=explanation,
    )
