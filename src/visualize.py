
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)


def create_visualizations():
    data = pd.read_csv("data/scored_telemetry.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Use Chiller-01 for the time-series charts.
    chiller = data[data["asset_id"] == "Chiller-01"].copy()

    # 1. Power and flow trends
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    axes[0].plot(
        chiller["timestamp"],
        chiller["power_kw"],
        linewidth=0.8,
        label="Power (kW)"
    )
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title("Chiller-01 Power Trend")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        chiller["timestamp"],
        chiller["flow_rate"],
        linewidth=0.8,
        label="Flow Rate"
    )
    axes[1].set_ylabel("Flow Rate (GPM)")
    axes[1].set_title("Chiller-01 Flow Trend")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    abnormal = chiller["classification"] != "NORMAL"

    axes[0].scatter(
        chiller.loc[abnormal, "timestamp"],
        chiller.loc[abnormal, "power_kw"],
        color="red",
        s=8,
        label="Abnormal"
    )

    axes[0].legend()
    plt.tight_layout()
    plt.savefig("outputs/trends.png", dpi=150)
    plt.close()

    # 2. Power vs flow, colored by operating status
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=data.sample(min(5000, len(data)), random_state=42),
        x="flow_rate",
        y="power_kw",
        hue="status",
        alpha=0.6
    )

    plt.title("Power vs Flow Rate by Machine Status")
    plt.xlabel("Flow Rate")
    plt.ylabel("Power (kW)")
    plt.tight_layout()
    plt.savefig("outputs/efficiency_correlation.png", dpi=150)
    plt.close()

    # 3. Anomaly scores over time
    plt.figure(figsize=(15, 5))

    plt.plot(
        chiller["timestamp"],
        chiller["anomaly_score"],
        label="Anomaly Score",
        linewidth=0.8
    )

    plt.axhline(0.50, linestyle="--", label="Warning threshold")
    plt.axhline(0.75, linestyle="--", label="Critical threshold")

    plt.fill_between(
        chiller["timestamp"],
        0,
        chiller["anomaly_score"],
        where=chiller["anomaly_score"] >= 0.50,
        alpha=0.25
    )

    plt.title("Chiller-01 Anomaly Detection Timeline")
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/anomalies_detected.png", dpi=150)
    plt.close()

    print("All three visualization files created.")


if __name__ == "__main__":
    create_visualizations()
