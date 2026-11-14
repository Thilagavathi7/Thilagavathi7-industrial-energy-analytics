
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "power_kw",
    "flow_rate",
    "supply_temp_c",
    "return_temp_c",
    "pressure_bar",
    "delta_t",
    "specific_energy",
    "rolling_power_mean_1h",
    "load_ratio"
]


def add_anomaly_scores(data, model, scaler):
    df = data.copy()

    running = df["status"].eq("RUNNING")
    scores = np.zeros(len(df))

    X = df.loc[running, FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    raw_scores = -model.decision_function(
        scaler.transform(X)
    )

    model_scores = 1 / (
        1 + np.exp(-raw_scores * 8)
    )

    scores[np.where(running)[0]] = model_scores

    # Establish a reference power level from the first
    # 24 hours of RUNNING readings for each machine.
    df["power_baseline"] = np.nan

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    for asset_id, group in df.groupby("asset_id"):
        first_day = group[
            (group["status"] == "RUNNING")
            & (
                group["timestamp"]
                < group["timestamp"].min()
                + pd.Timedelta(days=1)
            )
        ]

        baseline = first_day["power_kw"].median()

        df.loc[group.index, "power_baseline"] = baseline

    # Domain-based anomaly scoring.
    for index in df.index[running]:
        score = scores[index]

        # Frozen temperature sensor: critical fault.
        if bool(df.loc[index, "frozen_sensor"]):
            score = max(score, 0.95)

        baseline = df.loc[index, "power_baseline"]
        power = df.loc[index, "power_kw"]

        # Detect significant power increase.
        if pd.notna(baseline) and baseline > 0:
            ratio = power / baseline

            if ratio >= 1.05:
                rule_score = min(
                    0.95,
                    0.50 + (ratio - 1.05) * 2.5
                )
                score = max(score, rule_score)

        scores[index] = score

    df["anomaly_score"] = scores

    df["classification"] = np.select(
        [
            df["anomaly_score"] >= 0.75,
            df["anomaly_score"] >= 0.50
        ],
        [
            "CRITICAL",
            "WARNING"
        ],
        default="NORMAL"
    )

    df.loc[~running, "classification"] = "NORMAL"

    return df


def train_model():
    data = pd.read_csv("data/cleaned_telemetry.csv")

    running_data = data[
        data["status"] == "RUNNING"
    ].copy()

    X = running_data[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_scaled)

    scored_data = add_anomaly_scores(
        data, model, scaler
    )

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES
        },
        "models/anomaly_detector.joblib"
    )

    scored_data.to_csv(
        "data/scored_telemetry.csv",
        index=False
    )

    print("Model saved successfully.")
    print("\nClassification counts:")
    print(scored_data["classification"].value_counts())

    print("\nMaximum anomaly score:")
    print(scored_data["anomaly_score"].max())

    return scored_data


if __name__ == "__main__":
    train_model()
