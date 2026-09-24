"""Data models and synthetic benchmark generator for MEYRO (Phase 4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class ObservationRecord(BaseModel):
    """Schema for a single time-series observation in MEYRO."""

    subject_id: str = Field(..., description="Pseudonymized subject identifier")
    timestamp: datetime = Field(..., description="Observation UTC timestamp")
    modality: str = Field(..., description="Signal modality: activity, physiology, sleep")
    features: dict[str, float] = Field(..., description="Numeric feature mapping")
    quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Signal quality in [0, 1]")
    ground_truth_label: int | None = Field(
        None, description="0 = normal, 1 = true deviation; benchmark evaluation only"
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class SyntheticBenchmarkGenerator:
    """Generates synthetic longitudinal physiological/activity data with ground truth personal deviations.

    Models:
    - Subject-specific idiosyncratic normal baselines (different means, variances)
    - Weekly periodicity (weekday vs weekend)
    - Controlled noise and sensor quality degradation
    - Injected longitudinal deviations (sustained shift, abrupt outlier, drift)
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_subject_series(
        self,
        subject_id: str,
        n_days: int = 90,
        start_date: datetime | None = None,
        base_step_mean: float | None = None,
        base_hr_mean: float | None = None,
        base_sleep_mean: float | None = None,
        anomaly_spans: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        if start_date is None:
            start_date = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

        # Subject-specific baseline distributions if not provided
        step_mean = base_step_mean or float(self.rng.uniform(6000, 12000))
        hr_mean = base_hr_mean or float(self.rng.uniform(55, 78))
        sleep_mean = base_sleep_mean or float(self.rng.uniform(400, 520))

        records = []
        for day in range(n_days):
            current_date = start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5

            # Diurnal/weekly pattern
            weekend_step_mult = 1.15 if is_weekend else 1.0
            weekend_sleep_add = 35.0 if is_weekend else 0.0

            # Daily normal variation
            daily_steps = max(500, self.rng.normal(step_mean * weekend_step_mult, 1200))
            daily_hr = max(40.0, self.rng.normal(hr_mean, 3.5))
            daily_sleep = max(180.0, self.rng.normal(sleep_mean + weekend_sleep_add, 40))
            quality = float(np.clip(self.rng.beta(20, 1), 0.1, 1.0))
            label = 0

            # Inject controlled anomalies if within specified spans
            if anomaly_spans:
                for span in anomaly_spans:
                    if span["start_day"] <= day <= span["end_day"]:
                        label = 1
                        if "step_multiplier" in span:
                            daily_steps *= span["step_multiplier"]
                        if "hr_shift" in span:
                            daily_hr += span["hr_shift"]
                        if "sleep_shift" in span:
                            daily_sleep += span["sleep_shift"]

            records.append({
                "subject_id": subject_id,
                "timestamp": current_date,
                "step_count": float(daily_steps),
                "resting_hr": float(daily_hr),
                "sleep_duration_min": float(daily_sleep),
                "quality_score": quality,
                "ground_truth_label": label,
            })

        df = pd.DataFrame(records)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def generate_cohort(
        self,
        n_subjects: int = 10,
        n_days: int = 90,
        anomaly_rate_per_subject: float = 0.5,
    ) -> pd.DataFrame:
        all_dfs = []
        for i in range(n_subjects):
            subj_id = f"SUBJ_{i+1:03d}"
            spans = []
            if self.rng.uniform() < anomaly_rate_per_subject:
                # Inject a sustained deviation of 5-10 days
                start_day = int(self.rng.integers(35, max(36, n_days - 15)))
                length = int(self.rng.integers(4, 9))
                # Reduced activity & elevated heart rate (e.g. malaise/illness pattern)
                spans.append({
                    "start_day": start_day,
                    "end_day": start_day + length,
                    "step_multiplier": 0.4,
                    "hr_shift": 12.0,
                    "sleep_shift": 60.0,
                })
            df = self.generate_subject_series(
                subject_id=subj_id, n_days=n_days, anomaly_spans=spans
            )
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)
