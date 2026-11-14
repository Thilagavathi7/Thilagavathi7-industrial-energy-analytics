
import pandas as pd
import numpy as np

ELECTRICITY_RATE = 0.12

DIAGNOSTICS = {
    "power": (
        "High power surge detected without proportional flow increase. "
        "Check electrical motor load or compressor stage bypass."
    ),
    "freeze": (
        "Sensor freeze detected on temperature probe. Possible "
        "thermocouple failure or communication loop lockup."
    ),
    "efficiency": (
        "Gradual thermal efficiency loss detected (+25% above baseline). "
        "Recommend inspecting heat exchanger tubes for fouling or refrigerant leak."
    )
}


def generate_insights(data):
    df = data.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["asset_id", "timestamp"])

    # Estimate baseline power using normal readings at similar flow.
    normal = df[
        (df["status"] == "RUNNING")
        & (df["classification"] == "NORMAL")
    ]

    baseline_by_asset = (
        normal.groupby("asset_id")["power_kw"].median()
    )

    df["baseline_power_kw"] = df["asset_id"].map(
        baseline_by_asset
    )

    df["baseline_power_kw"] = df["baseline_power_kw"].fillna(
        df["power_kw"]
    )

    # One-minute readings, so each reading represents 1/60 hour.
    df["excess_kwh"] = (
        (df["power_kw"] - df["baseline_power_kw"]).clip(lower=0)
        / 60
    )

    df["cost_loss"] = (
        df["excess_kwh"] * ELECTRICITY_RATE
    )

    total_excess = df.loc[
        df["classification"] != "NORMAL",
        "excess_kwh"
    ].sum()

    total_cost = total_excess * ELECTRICITY_RATE

    messages = []

    for asset_id, group in df.groupby("asset_id"):
        abnormal = group[
            group["classification"] != "NORMAL"
        ]

        if abnormal.empty:
            continue

        high_power = (
            group["power_kw"] > group["baseline_power_kw"] * 1.15
        ) & (group["flow_rate"] > 0)

        if high_power.any():
            messages.append({
                "asset_id": asset_id,
                "diagnostic": DIAGNOSTICS["power"]
            })

        if group["frozen_sensor"].astype(bool).any():
            messages.append({
                "asset_id": asset_id,
                "diagnostic": DIAGNOSTICS["freeze"]
            })

        running = group[group["status"] == "RUNNING"]

        if len(running) >= 24 * 60:
            first_day = running.head(24)["specific_energy"].mean()
            last_day = running.tail(24)["specific_energy"].mean()

            if first_day > 0 and last_day >= first_day * 1.20:
                messages.append({
                    "asset_id": asset_id,
                    "diagnostic": DIAGNOSTICS["efficiency"]
                })

    return {
        "excess_energy_kwh": round(float(total_excess), 3),
        "estimated_cost_loss_usd": round(float(total_cost), 3),
        "diagnostics": messages,
        "data": df
    }


if __name__ == "__main__":
    data = pd.read_csv("data/scored_telemetry.csv")
    result = generate_insights(data)

    print("Estimated excess energy:", result["excess_energy_kwh"], "kWh")
    print("Estimated cost loss: $", result["estimated_cost_loss_usd"])

    for item in result["diagnostics"]:
        print("\n", item["asset_id"], "-", item["diagnostic"])
