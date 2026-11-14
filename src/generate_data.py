
import os

import numpy as np
import pandas as pd


# Each machine has its own normal operating values.
MACHINE_SPECS = {
    "Chiller-01": {
        "power": 220,
        "flow": 450,
        "supply_temp": 7,
        "return_temp": 13,
        "pressure": 4.2,
        "rated_capacity": 300,
    },
    "Compressor-02": {
        "power": 110,
        "flow": 320,
        "supply_temp": 25,
        "return_temp": 35,
        "pressure": 7.5,
        "rated_capacity": 150,
    },
    "Pump-03": {
        "power": 55,
        "flow": 180,
        "supply_temp": 15,
        "return_temp": 22,
        "pressure": 3.5,
        "rated_capacity": 75,
    },
}


def generate_machine_data(asset_id, specs):
    # Seven days of readings, collected once every minute.
    timestamps = pd.date_range(
        start="2026-10-01 00:00:00",
        periods=7 * 24 * 60,
        freq="min",
        tz="UTC",
    )

    data = pd.DataFrame({"timestamp": timestamps})

    data["asset_id"] = asset_id

    # Machines operate from 6 AM to 10 PM.
    hour = data["timestamp"].dt.hour

    data["status"] = np.where(
        (hour >= 6) & (hour < 22),
        "RUNNING",
        "STANDBY",
    )

    data.loc[data["status"] == "STANDBY", "status"] = "OFF"

    running = data["status"] == "RUNNING"

    # Generate normal sensor readings with small variations.
    data["power_kw"] = np.where(
        running,
        np.random.normal(specs["power"], specs["power"] * 0.025, len(data)),
        np.random.normal(2, 0.3, len(data)),
    )

    data["flow_rate"] = np.where(
        running,
        np.random.normal(specs["flow"], specs["flow"] * 0.02, len(data)),
        0,
    )

    data["supply_temp_c"] = np.random.normal(
        specs["supply_temp"], 0.15, len(data)
    )

    data["return_temp_c"] = np.random.normal(
        specs["return_temp"], 0.2, len(data)
    )

    data["pressure_bar"] = np.random.normal(
        specs["pressure"], 0.08, len(data)
    )

    # Keep readings realistic when the machine is not running.
    data.loc[~running, "power_kw"] = 2
    data.loc[~running, "flow_rate"] = 0

    # Inject a 20-minute communication dropout.
    dropout_start = 1000
    dropout_end = dropout_start + 20

    sensor_columns = [
        "power_kw",
        "flow_rate",
        "supply_temp_c",
        "return_temp_c",
        "pressure_bar",
    ]

    data.loc[dropout_start:dropout_end - 1, sensor_columns] = np.nan

    # Inject physically impossible sensor readings.
    data.loc[500, "power_kw"] = -25
    data.loc[600, "supply_temp_c"] = 999

    # Freeze the supply temperature for 50 minutes while running.
    freeze_start = 3 * 24 * 60 + 12 * 60
    freeze_end = freeze_start + 50

    data.loc[
        freeze_start:freeze_end - 1, "supply_temp_c"
    ] = specs["supply_temp"]

    # Simulate gradual power degradation for Chiller-01.
    # Flow remains constant while power increases over 48 hours.
    if asset_id == "Chiller-01":
        degradation_start = 4 * 24 * 60 + 6 * 60
        degradation_end = degradation_start + 48 * 60

        data.loc[
            degradation_start:degradation_end - 1, "status"
        ] = "RUNNING"

        data.loc[
            degradation_start:degradation_end - 1, "flow_rate"
        ] = specs["flow"]

        power_increase = np.linspace(
            0,
            0.25,
            degradation_end - degradation_start,
        )

        data.loc[
            degradation_start:degradation_end - 1, "power_kw"
        ] = specs["power"] * (1 + power_increase)

    return data


def generate_dataset():
    np.random.seed(42)

    all_machines = []

    for asset_id, specs in MACHINE_SPECS.items():
        machine_data = generate_machine_data(asset_id, specs)
        all_machines.append(machine_data)

    final_data = pd.concat(all_machines, ignore_index=True)

    os.makedirs("data", exist_ok=True)

    output_path = "data/raw_telemetry.csv"
    final_data.to_csv(output_path, index=False)

    print("Synthetic telemetry dataset generated successfully.")
    print(f"Saved to: {output_path}")
    print(f"Total records: {len(final_data)}")

    print("\nRecords per machine:")
    print(final_data.groupby("asset_id").size())

    print("\nMachine status distribution:")
    print(final_data.groupby(["asset_id", "status"]).size())


if __name__ == "__main__":
    generate_dataset()
