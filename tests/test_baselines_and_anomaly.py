"""Tests for Classical Baselines and Anomaly Detectors."""

from meyro.anomaly.classical import ClassicalAnomalyEngine
from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.preprocessing.pipeline import PreprocessingPipeline


def test_baselines_and_anomaly_engines():
    gen = SyntheticBenchmarkGenerator(seed=42)
    cohort = gen.generate_cohort(n_subjects=4, n_days=50, anomaly_rate_per_subject=0.5)
    features = ["step_count", "resting_hr", "sleep_duration_min"]

    pipe = PreprocessingPipeline(feature_cols=features)
    cleaned = pipe.clean_and_impute(cohort)
    calib, eval_df = PreprocessingPipeline.create_subject_temporal_splits(cleaned, calibration_days=25)

    # 1. Population baseline
    pop = PopulationBaseline(feature_cols=features)
    pop.fit(calib)
    scored_pop = pop.score(eval_df)
    assert "deviation_score" in scored_pop.columns
    assert len(scored_pop) == len(eval_df)

    # 2. Personalized baseline
    pers = PersonalizedBaseline(feature_cols=features)
    pers.fit_calibration(calib)
    scored_pers = pers.score_causal(eval_df)
    assert "deviation_score" in scored_pers.columns

    # 3. Isolation Forest per subject
    iso = ClassicalAnomalyEngine(algorithm="isolation_forest", random_state=42)
    iso.fit_per_subject(calib, feature_cols=features)
    scored_iso = iso.score_per_subject(eval_df, feature_cols=features)
    assert "anomaly_score" in scored_iso.columns
    assert not scored_iso["anomaly_score"].isna().any()
