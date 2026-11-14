
import os

import numpy as np
import pandas as pd


RATED_CAPACITY = {
    "Chiller-01": 300,
    "Compressor-02": 150,
    "Pump-03": 75,
}

SENSOR_COLUMNS = [
    "power_kw",
    "flow_rate",
    "supply_temp_c",
    "return_temp_c",
    "pressure_bar",
]


def load_data(file_path="data/raw_telemetry.csv"):
    return pd.read_csv(file_path, parse_dates=["timestamp"])


def clean_data(df):
    df = df.copy()

    df = df.sort_values(
        ["asset_id", "timestamp"]
    ).reset_index(drop=True)

    # Record missing readings before filling them.
    df["missing_flag"] = df[SENSOR_COLUMNS].isna().any(axis=1)

    # Replace physically impossible sensor values with NaN.
    df.loc[df["power_kw"] < 0, "power_kw"] = np.nan

    for column in ["supply_temp_c", "return_temp_c"]:
        invalid = (df[column] < -5) | (df[column] > 110)
        df.loc[invalid, column] = np.nan

    # Detect frozen temperature sensors before interpolation.
    df["frozen_sensor_flag"] = False

    for asset_id, group in df.groupby("asset_id"):
        rolling_std = (
            group["supply_temp_c"]
            .rolling(window=30, min_periods=30)
            .std()
        )

        frozen = (
            (rolling_std == 0) &
            (group["status"] == "RUNNING")
        )

        df.loc[group.index, "frozen_sensor_flag"] = (
            frozen.fillna(False).to_numpy()
        )

    # Fill short gaps separately for each machine.
    for asset_id, group in df.groupby("asset_id"):
        for column in SENSOR_COLUMNS:
            df.loc[group.index, column] = (
                group[column]
                .interpolate(
                    method="linear",
                    limit=30,
                    limit_area="inside",
                )
            )

    return df


def engineer_features(df):
    df = df.copy()

    df["delta_t"] = (
        df["return_temp_c"] - df["supply_temp_c"]
    )

    df["specific_energy"] = (
        df["power_kw"] / (df["flow_rate"] + 1e-5)
    )

    df["rated_capacity_kw"] = df["asset_id"].map(
        RATED_CAPACITY
    )

    df["load_ratio"] = (
        df["power_kw"] / df["rated_capacity_kw"]
    )

    df["rolling_power_mean_1h"] = (
        df.groupby("asset_id")["power_kw"]
        .transform(
            lambda values: values.rolling(
                window=60,
                min_periods=1,
            ).mean()
        )
    )

    return df


def process_data():
    os.makedirs("data", exist_ok=True)

    raw_df = load_data()

    cleaned_df = clean_data(raw_df)
    cleaned_df = engineer_features(cleaned_df)

    output_path = "data/cleaned_telemetry.csv"
    cleaned_df.to_csv(output_path, index=False)

    print("Data processing completed.")
    print(f"Total records: {len(cleaned_df)}")
    print(
        "Missing readings flagged:",
        int(cleaned_df["missing_flag"].sum()),
    )
    print(
        "Frozen sensor readings flagged:",
        int(cleaned_df["frozen_sensor_flag"].sum()),
    )
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    process_data()
