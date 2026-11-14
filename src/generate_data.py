
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MACHINES = {
    "Chiller-01": {
        "power": 280,
        "flow": 450,
        "supply": 7,
        "return": 13,
        "pressure": 4.2,
        "capacity": 400,
        "flow_unit": "GPM"
    },
    "Compressor-02": {
        "power": 180,
        "flow": 900,
        "supply": 25,
        "return": 38,
        "pressure": 7.5,
        "capacity": 250,
        "flow_unit": "CFM"
    },
    "Pump-03": {
        "power": 90,
        "flow": 300,
        "supply": 18,
        "return": 24,
        "pressure": 3.5,
        "capacity": 130,
        "flow_unit": "GPM"
    }
}


def generate_machine_data(asset_id, spec, start_time):
    timestamps = pd.date_range(
        start=start_time,
        periods=7 * 24 * 60,
        freq="min",
        tz="UTC"
    )

    df = pd.DataFrame({"timestamp": timestamps})

    rng = np.random.default_rng(
        sum(ord(char) for char in asset_id)
    )

    minute_of_day = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
    )

    running = (
        (minute_of_day >= 6 * 60)
        & (minute_of_day < 22 * 60)
    )

    standby = (
        (minute_of_day >= 5 * 60)
        & (minute_of_day < 6 * 60)
    ) | (
        (minute_of_day >= 22 * 60)
        & (minute_of_day < 23 * 60)
    )

    df["status"] = np.select(
        [running, standby],
        ["RUNNING", "STANDBY"],
        default="OFF"
    )

    df["power_kw"] = np.where(
        running,
        spec["power"] + rng.normal(0, 4, len(df)),
        np.where(standby, spec["power"] * 0.12, 0)
    )

    df["flow_rate"] = np.where(
        running,
        spec["flow"] + rng.normal(0, 5, len(df)),
        np.where(standby, spec["flow"] * 0.1, 0)
    )

    df["supply_temp_c"] = (
        spec["supply"] + rng.normal(0, 0.15, len(df))
    )

    df["return_temp_c"] = (
        spec["return"] + rng.normal(0, 0.2, len(df))
    )

    df["pressure_bar"] = (
        spec["pressure"] + rng.normal(0, 0.08, len(df))
    )

    # OFF machines have zero power and flow.
    off = df["status"] == "OFF"
    df.loc[off, "power_kw"] = 0
    df.loc[off, "flow_rate"] = 0

    # Communication dropout: missing sensor readings.
    df.loc[1000:1019, [
        "power_kw",
        "flow_rate",
        "supply_temp_c",
        "return_temp_c",
        "pressure_bar"
    ]] = np.nan

    # Impossible sensor readings.
    df.loc[2000, "power_kw"] = -50
    df.loc[2500, "supply_temp_c"] = 999

    # Temperature sensor freeze for 50 minutes while running.
    freeze_start = 3 * 24 * 60 + 12 * 60
    freeze_end = freeze_start + 50

    frozen_value = float(spec["supply"])

    df.loc[freeze_start:freeze_end, "supply_temp_c"] = frozen_value

    # Gradual power degradation over 48 hours at constant flow.
    if asset_id == "Chiller-01":
        degradation_start = 4 * 24 * 60 + 6 * 60
        degradation_end = degradation_start + 48 * 60

        positions = np.arange(
            degradation_start,
            degradation_end
        )

        progress = np.linspace(0, 0.25, len(positions))

        for position, increase in zip(positions, progress):
            if df.loc[position, "status"] == "RUNNING":
                df.loc[position, "flow_rate"] = spec["flow"]
                df.loc[position, "power_kw"] = (
                    spec["power"] * (1 + increase)
                )

    df["asset_id"] = asset_id

    return df[[
        "timestamp",
        "asset_id",
        "power_kw",
        "flow_rate",
        "supply_temp_c",
        "return_temp_c",
        "pressure_bar",
        "status"
    ]]


def generate_dataset():
    start_time = "2026-10-01 00:00:00"

    frames = []

    for asset_id, spec in MACHINES.items():
        machine_df = generate_machine_data(
            asset_id,
            spec,
            start_time
        )
        frames.append(machine_df)

    data = pd.concat(frames, ignore_index=True)

    data["timestamp"] = data["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    output_path = OUTPUT_DIR / "raw_telemetry.csv"
    data.to_csv(output_path, index=False)

    print("Raw dataset generated:", output_path)
    print("Total rows:", len(data))
    print(data.groupby("asset_id").size())


if __name__ == "__main__":
    generate_dataset()
