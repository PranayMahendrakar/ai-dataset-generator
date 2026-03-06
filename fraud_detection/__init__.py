"""
Credit Card Fraud Detection Package
=====================================
Modules: fraud_detector, evaluation, api, dashboard

Quick Start:
    from fraud_detection.fraud_detector import RealTimeFraudDetector
    detector = RealTimeFraudDetector()
    detector.train()
    result = detector.predict(transaction_dict)
"""

from fraud_detection.fraud_detector import (
    RealTimeFraudDetector,
    AnomalyDetector,
    RandomForestFraudModel,
    XGBoostFraudModel,
    TransactionScorer,
    FeatureEngineer,
    TransactionDataGenerator,
)

__all__ = [
    "RealTimeFraudDetector",
    "AnomalyDetector",
    "RandomForestFraudModel",
    "XGBoostFraudModel",
    "TransactionScorer",
    "FeatureEngineer",
    "TransactionDataGenerator",
]

__version__ = "1.0.0"
