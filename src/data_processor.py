
import pandas as pd
import numpy as np
from pathlib import Path

RATED_CAPACITY = {
    "Chiller-01": 400,
    "Compressor-02": 250,
    "Pump-03": 130
}

SENSOR_COLUMNS = [
    "power_kw",
    "flow_rate",
    "supply_temp_c",
    "return_temp_c",
    "pressure_bar"
]


def clean_data(data):
    df = data.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        ["asset_id", "timestamp"]
    ).reset_index(drop=True)

    df["frozen_sensor"] = False

    for asset_id, group in df.groupby("asset_id"):
        index = group.index

        for column in [
            "supply_temp_c",
            "return_temp_c"
        ]:
            frozen = (
                group[column]
                .rolling(window=30, min_periods=30)
                .std()
                .eq(0)
            )

            # Expand each detected flat window backwards
            # to include the beginning of the frozen interval.
            frozen_values = frozen.to_numpy()
            expanded = np.zeros(len(group), dtype=bool)

            detected_positions = np.flatnonzero(frozen_values)

            for position in detected_positions:
                start_position = max(0, position - 29)
                expanded[start_position:position + 1] = True

            running = group["status"].eq("RUNNING").to_numpy()

            df.loc[index, "frozen_sensor"] |= (
                expanded & running
            )

    # Remove physically impossible values.
    df.loc[df["power_kw"] < 0, "power_kw"] = np.nan

    for column in ["supply_temp_c", "return_temp_c"]:
        invalid = (
            (df[column] < -5)
            | (df[column] > 110)
        )
        df.loc[invalid, column] = np.nan

    for column in SENSOR_COLUMNS:
        df[column] = (
            df.groupby("asset_id")[column]
            .transform(
                lambda series: series.interpolate(
                    limit=30,
                    limit_direction="both"
                )
            )
        )

    # Any remaining edge gaps are filled within each machine.
    for column in SENSOR_COLUMNS:
        df[column] = (
            df.groupby("asset_id")[column]
            .transform(lambda series: series.ffill().bfill())
        )

    return df


def engineer_features(data):
    df = data.copy()

    df["delta_t"] = (
        df["return_temp_c"] - df["supply_temp_c"]
    )

    df["specific_energy"] = (
        df["power_kw"] / (df["flow_rate"] + 1e-5)
    )

    df["rolling_power_mean_1h"] = (
        df.groupby("asset_id")["power_kw"]
        .transform(
            lambda series: series.rolling(
                window=60,
                min_periods=1
            ).mean()
        )
    )

    df["rated_capacity"] = df["asset_id"].map(
        RATED_CAPACITY
    )

    df["load_ratio"] = (
        df["power_kw"] / df["rated_capacity"]
    )

    return df


def process_data(input_path="data/raw_telemetry.csv"):
    raw = pd.read_csv(input_path)

    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)

    output_path = Path("data/cleaned_telemetry.csv")
    featured.to_csv(output_path, index=False)

    print("Cleaned dataset saved:", output_path)
    print("Rows:", len(featured))

    return featured


if __name__ == "__main__":
    process_data()
