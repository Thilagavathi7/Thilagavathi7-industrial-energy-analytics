
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "power_kw",
    "flow_rate",
    "supply_temp_c",
    "return_temp_c",
    "pressure_bar",
    "delta_t",
    "specific_energy",
    "load_ratio",
    "rolling_power_mean_1h",
]


def train_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(
        "data/cleaned_telemetry.csv",
        parse_dates=["timestamp"],
    )

    # The model should learn normal machine behaviour while running.
    running_data = df[
        (df["status"] == "RUNNING")
    ].copy()

    training_data = running_data.dropna(
        subset=FEATURE_COLUMNS
    )

    X = training_data[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
    )

    model.fit(X_scaled)

    # Higher decision scores indicate more unusual observations.
    raw_scores = -model.decision_function(X_scaled)

    # Convert scores to a consistent 0-1 range.
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score == min_score:
        anomaly_scores = np.zeros(len(raw_scores))
    else:
        anomaly_scores = (
            (raw_scores - min_score) /
            (max_score - min_score)
        )

    training_data["anomaly_score"] = anomaly_scores

    training_data["classification"] = np.select(
        [
            training_data["anomaly_score"] < 0.50,
            training_data["anomaly_score"] < 0.75,
        ],
        [
            "NORMAL",
            "WARNING",
        ],
        default="CRITICAL",
    )

    # Save the model and scaler together.
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURE_COLUMNS,
        },
        "models/anomaly_detector.joblib",
    )

    # Save scored records for visualization and diagnostics.
    training_data.to_csv(
        "outputs/anomaly_results.csv",
        index=False,
    )

    print("Model training completed.")
    print(f"Training records: {len(training_data)}")
    print("\nClassification summary:")
    print(training_data["classification"].value_counts())

    print("\nModel saved to models/anomaly_detector.joblib")


if __name__ == "__main__":
    train_model()
