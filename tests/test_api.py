
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_diagnose_endpoint():
    reading = {
        "timestamp": "2026-10-01T14:30:00Z",
        "asset_id": "Chiller-01",
        "power_kw": 285.5,
        "flow_rate": 450.0,
        "supply_temp_c": 7.2,
        "return_temp_c": 12.8,
        "pressure_bar": 4.2,
        "status": "RUNNING"
    }

    response = client.post(
        "/api/v1/diagnose",
        json=reading
    )

    assert response.status_code == 200

    result = response.json()

    assert "anomaly_score" in result
    assert "classification" in result
    assert result["classification"] in [
        "NORMAL",
        "WARNING",
        "CRITICAL"
    ]
