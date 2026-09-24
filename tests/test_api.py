"""Tests for FastAPI backend."""

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["system"] == "MEYRO Longitudinal Engine"


def test_get_baseline_existing():
    response = client.get("/baseline/SUBJ_001")
    assert response.status_code == 200
    data = response.json()
    assert data["subject_id"] == "SUBJ_001"
    assert "resting_hr" in data["baseline_setpoints"]


def test_get_baseline_not_found():
    response = client.get("/baseline/SUBJ_NON_EXISTENT")
    assert response.status_code == 404


def test_score_deviation_endpoint():
    payload = {
        "subject_id": "SUBJ_001",
        "timestamp": "2026-03-25T12:00:00Z",
        "features": {
            "step_count": 2100.0,
            "resting_hr": 84.0,
            "sleep_duration_min": 560.0,
        },
        "quality_score": 0.95,
    }
    response = client.post("/deviations/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["subject_id"] == "SUBJ_001"
    assert "deviation_score" in data
    assert "explanation" in data
    assert "safety_disclaimer" in data["explanation"]
