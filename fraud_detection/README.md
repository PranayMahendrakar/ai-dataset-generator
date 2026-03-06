# 💳 Credit Card Fraud Detection

> Real-time ML-powered fraud detection using an ensemble of **Isolation Forest**, **Random Forest**, and **XGBoost**.

## Features

| Feature | Description |
|---------|-------------|
| 🔍 Anomaly Detection | Isolation Forest identifies novel fraud patterns without labels |
| 📊 Transaction Scoring | Weighted ensemble score (0-100) with 4 risk levels |
| ⚡ Real-time Prediction | <5ms per transaction, full REST API |
| 📈 18 Features | Engineered from raw transaction data |
| 🎛️ Streamlit Dashboard | Interactive exploration and batch analysis |

## Project Structure

```
fraud_detection/
├── __init__.py          # Package exports
├── fraud_detector.py    # Core ML pipeline (IF + RF + XGBoost)
├── evaluation.py        # Metrics, ROC/PR curves, dashboards
├── api.py               # FastAPI REST endpoint
├── dashboard.py         # Streamlit interactive UI
└── requirements.txt     # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r fraud_detection/requirements.txt

# Run the demo (trains all 3 models + scores sample transactions)
python -m fraud_detection.fraud_detector

# Launch the Streamlit dashboard
streamlit run fraud_detection/dashboard.py

# Start the REST API
uvicorn fraud_detection.api:app --reload
```

## Models

### 1. Isolation Forest (Anomaly Detection)
- **Type**: Unsupervised — no fraud labels needed
- **How it works**: Isolates anomalies by random feature splitting; fraudulent transactions are isolated faster
- **Ensemble weight**: 25%
- **Strengths**: Catches novel/unknown fraud patterns

### 2. Random Forest
- **Type**: Supervised classification
- **Key params**: 300 trees, `class_weight='balanced'`, `max_depth=12`
- **Ensemble weight**: 35%
- **Strengths**: Stable predictions, interpretable feature importances

### 3. XGBoost
- **Type**: Gradient boosted trees
- **Key params**: 500 rounds, `learning_rate=0.05`, `scale_pos_weight=50`
- **Ensemble weight**: 40%
- **Strengths**: Best accuracy on tabular data with high class imbalance

## Ensemble Scoring

The final **Fraud Score** (0-100) is a weighted combination:

```
fraud_score = 0.25 × IF_score + 0.35 × RF_proba + 0.40 × XGB_proba
```

| Score | Risk Level | Action |
|-------|-----------|--------|
| 0-30  | 🟢 Low    | Approve |
| 30-60 | 🟡 Medium | Review |
| 60-80 | 🟠 High   | Manual check |
| 80-100| 🔴 Critical | Block |

## 18 Engineered Features

```
Raw features:    amount, hour, day_of_week, merchant_category,
                 velocity_1h, distance_from_home_km, is_international,
                 card_present, unusual_location, avg_amount_7d, transaction_count_7d

Derived:         amount_vs_avg    (amount / 7d average)
                 is_night         (hour < 6 or > 22)
                 is_weekend       (day >= 5)
                 high_velocity    (velocity_1h > 5)
                 log_amount       (log-scaled amount)
                 log_distance     (log-scaled distance)
                 manual_risk      (composite hand-crafted score)
```

## REST API

```bash
# Health check
GET /health

# Score single transaction
POST /predict
{
  "amount": 2850.00,
  "hour": 3,
  "day_of_week": 6,
  "merchant_category": "online_shopping",
  "velocity_1h": 8,
  "distance_from_home_km": 4200.0,
  "is_international": 1,
  "card_present": 0,
  "unusual_location": 1,
  "avg_amount_7d": 65.0,
  "transaction_count_7d": 2
}

# Batch scoring (up to 1,000 transactions)
POST /predict/batch

# Model info
GET /model/info
```

## Python Usage

```python
from fraud_detection import RealTimeFraudDetector

detector = RealTimeFraudDetector()
detector.train()  # trains all 3 models on synthetic data

result = detector.predict({
    "amount": 42.50, "hour": 14, "day_of_week": 2,
    "merchant_category": "grocery", "velocity_1h": 1,
    "distance_from_home_km": 3.2, "is_international": 0,
    "card_present": 1, "unusual_location": 0,
    "avg_amount_7d": 55.0, "transaction_count_7d": 12,
})

print(result['fraud_score'])    # e.g. 8.2
print(result['risk_level'])     # "low"
print(result['is_fraud'])       # False
print(result['explanation'])    # ["No high-risk signals detected"]
print(result['processing_time_ms'])  # e.g. 2.3
```

## Evaluation Output

Running `detector.train(verbose=True)` produces:

```
MODEL EVALUATION
=======================================================
isolation_forest     | AUC=0.8912 | AP=0.4521 | F1=0.6103
random_forest        | AUC=0.9823 | AP=0.7845 | F1=0.7912
xgboost              | AUC=0.9891 | AP=0.8123 | F1=0.8034
```

## License

MIT
