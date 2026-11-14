
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from train import add_anomaly_scores
from data_processor import clean_data, engineer_features
from insights import generate_insights

app = FastAPI(
    title="Industrial Asset Energy & Maintenance Analytics",
    version="1.0"
)

MODEL_PATH = Path("models/anomaly_detector.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]


class TelemetryReading(BaseModel):
    timestamp: str
    asset_id: str
    power_kw: float
    flow_rate: float
    supply_temp_c: float
    return_temp_c: float
    pressure_bar: float
    status: str


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model_loaded": model is not None
    }


@app.post("/api/v1/diagnose")
def diagnose(reading: TelemetryReading):
    record = pd.DataFrame([reading.model_dump()])

    if reading.asset_id not in [
        "Chiller-01",
        "Compressor-02",
        "Pump-03"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Unknown asset_id"
        )

    cleaned = clean_data(record)
    featured = engineer_features(cleaned)

    scored = add_anomaly_scores(
        featured,
        model,
        scaler
    )

    score = float(scored.iloc[0]["anomaly_score"])
    classification = str(
        scored.iloc[0]["classification"]
    )

    result = generate_insights(
        scored.assign(
            frozen_sensor=False
        )
    )

    diagnostic = "No unusual operating behaviour detected."

    if classification == "WARNING":
        diagnostic = "Warning: inspect the machine operating conditions."

    if classification == "CRITICAL":
        diagnostic = "Critical anomaly detected. Maintenance inspection recommended."

    if bool(scored.iloc[0].get("frozen_sensor", False)):
        diagnostic = (
            "Sensor freeze detected on temperature probe. "
            "Possible thermocouple failure or communication loop lockup."
        )

    return {
        "asset_id": reading.asset_id,
        "anomaly_score": round(score, 4),
        "classification": classification,
        "diagnostic": diagnostic
    }


@app.post("/api/v1/batch-audit")
async def batch_audit(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        data = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the CSV file."
        )

    required_columns = [
        "timestamp",
        "asset_id",
        "power_kw",
        "flow_rate",
        "supply_temp_c",
        "return_temp_c",
        "pressure_bar",
        "status"
    ]

    missing = set(required_columns) - set(data.columns)

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {sorted(missing)}"
        )

    cleaned = clean_data(data)
    featured = engineer_features(cleaned)

    scored = add_anomaly_scores(
        featured,
        model,
        scaler
    )

    insights = generate_insights(
        scored.assign(frozen_sensor=False)
    )

    flagged = scored[
        scored["classification"] != "NORMAL"
    ]

    intervals = []

    for asset_id, group in flagged.groupby("asset_id"):
        intervals.append({
            "asset_id": asset_id,
            "start": str(group["timestamp"].min()),
            "end": str(group["timestamp"].max()),
            "records": int(len(group))
        })

    return {
        "total_records": int(len(scored)),
        "anomaly_count": int(len(flagged)),
        "excess_energy_kwh": insights["excess_energy_kwh"],
        "estimated_cost_loss_usd": insights["estimated_cost_loss_usd"],
        "flagged_intervals": intervals
    }
