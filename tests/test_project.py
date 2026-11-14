
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import pandas as pd

from data_processor import clean_data, engineer_features


def sample_reading():
    return pd.DataFrame([{
        "timestamp": "2026-10-01T08:00:00Z",
        "asset_id": "Chiller-01",
        "power_kw": 280.0,
        "flow_rate": 450.0,
        "supply_temp_c": 7.0,
        "return_temp_c": 13.0,
        "pressure_bar": 4.2,
        "status": "RUNNING"
    }])


def test_clean_data_keeps_valid_power():
    cleaned = clean_data(sample_reading())

    assert cleaned["power_kw"].iloc[0] >= 0


def test_feature_engineering_creates_required_features():
    cleaned = clean_data(sample_reading())
    featured = engineer_features(cleaned)

    assert "delta_t" in featured.columns
    assert "specific_energy" in featured.columns
    assert "rolling_power_mean_1h" in featured.columns
    assert "load_ratio" in featured.columns

    assert round(featured["delta_t"].iloc[0], 2) == 6.0


def test_feature_values_are_numeric():
    cleaned = clean_data(sample_reading())
    featured = engineer_features(cleaned)

    assert pd.notna(featured["specific_energy"].iloc[0])
    assert featured["specific_energy"].iloc[0] > 0
