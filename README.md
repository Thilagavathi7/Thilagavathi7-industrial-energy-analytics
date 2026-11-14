# Industrial Asset Energy & Maintenance Analytics

## 1. Project Overview

This project analyses industrial machine telemetry to identify unusual
operating behaviour, estimate energy waste and support maintenance teams.

The project uses synthetic sensor readings for three industrial assets.

## 2. Machines

- Chiller-01 – Central Cooling Plant
- Compressor-02 – Compressed Air System
- Pump-03 – Cooling Water Circulation

The dataset contains seven days of one-minute readings for each machine.

## 3. Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Isolation Forest
- FastAPI
- Pytest

## 4. Project Workflow

1. Generate synthetic industrial telemetry.
2. Inject missing values, sensor outliers, frozen readings and power degradation.
3. Clean and validate sensor readings.
4. Engineer operational and energy-efficiency features.
5. Train an Isolation Forest anomaly detector.
6. Generate visualizations and anomaly scores.
7. Estimate excess energy consumption and financial loss.
8. Expose diagnosis and batch audit through FastAPI.

## 5. Anomaly Classification

| Classification | Anomaly Score |
|---|---|
| NORMAL | Below 0.50 |
| WARNING | 0.50 to below 0.75 |
| CRITICAL | 0.75 or higher |

The model is trained using RUNNING records to reduce false alarms
from normal standby and off periods.

## 6. Energy Waste Calculation

Excess energy is estimated using:

Excess kWh = sum((Actual kW - Baseline kW) * minutes / 60)

Estimated electricity cost uses $0.12 per kWh.

## 7. API Endpoints

- GET /health
- POST /api/v1/diagnose
- POST /api/v1/batch-audit

## 8. Run Locally

Install the dependencies:

    pip install -r requirements.txt

Generate data:

    python src/generate_data.py

Clean data:

    python src/data_processor.py

Train the anomaly detector:

    python src/train.py

Generate charts:

    python src/visualize.py

Run the API:

    bash run.sh

Open the API documentation:

    http://127.0.0.1:8000/docs

Run tests:

    pytest -v tests/

## 9. Model Evaluation and Trade-offs

Isolation Forest is an unsupervised model that identifies unusual
operating patterns without requiring a large labelled failure dataset.

A high recall helps identify more unusual machine behaviour. However,
too many false alarms can cause unnecessary maintenance inspections
and operational disruption.

The WARNING and CRITICAL thresholds are selected to separate
increasing levels of operational concern. These thresholds should be
validated against labelled anomalies before production use.

The synthetic dataset includes power degradation, sensor freezes
and impossible sensor values. The model's detection results should
be compared against these known injected events.

## 10. Limitations

- The dataset is synthetic and does not represent actual factory readings.
- Anomaly scores are model-based indicators, not confirmed equipment failures.
- Baseline power is an estimate and should be improved using real operating
  conditions before production deployment.
