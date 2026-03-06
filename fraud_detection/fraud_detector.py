"""
Credit Card Fraud Detection System
====================================
Features: Anomaly Detection, Transaction Scoring, Real-time Prediction
Models: Isolation Forest, Random Forest, XGBoost
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score
)
import xgboost as xgb


class TransactionDataGenerator:
    """Generate synthetic credit card transaction data."""

    MERCHANT_CATEGORIES = [
        'grocery', 'restaurant', 'gas_station', 'online_shopping',
        'entertainment', 'travel', 'healthcare', 'electronics',
        'clothing', 'atm_withdrawal'
    ]

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_samples=10000, fraud_rate=0.02):
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud
        legit = self._generate_legitimate(n_legit)
        fraud = self._generate_fraudulent(n_fraud)
        df = pd.concat([legit, fraud], ignore_index=True)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def _generate_legitimate(self, n):
        hours = self.rng.integers(8, 22, n)
        amounts = np.clip(self.rng.lognormal(3.5, 1.0, n), 1, 2000)
        return self._build_df(n, hours, amounts, label=0)

    def _generate_fraudulent(self, n):
        hours = self.rng.integers(0, 24, n)
        amounts = np.clip(self.rng.lognormal(5.0, 1.5, n), 50, 10000)
        return self._build_df(n, hours, amounts, label=1, fraud=True)

    def _build_df(self, n, hours, amounts, label, fraud=False):
        rng = self.rng
        merchants = [self.MERCHANT_CATEGORIES[i]
                     for i in rng.integers(0, len(self.MERCHANT_CATEGORIES), n)]
        velocity = rng.poisson(8 if fraud else 2, n)
        distance = rng.exponential(500 if fraud else 20, n)
        is_intl = (rng.random(n) > (0.3 if fraud else 0.95)).astype(int)
        card_present = (~(rng.random(n) > (0.6 if fraud else 0.05))).astype(int)
        unusual_loc = (rng.random(n) > (0.3 if fraud else 0.9)).astype(int)
        return pd.DataFrame({
            'amount': np.round(amounts, 2),
            'hour': hours,
            'day_of_week': rng.integers(0, 7, n),
            'merchant_category': merchants,
            'velocity_1h': velocity,
            'distance_from_home_km': np.round(distance, 1),
            'is_international': is_intl,
            'card_present': card_present,
            'unusual_location': unusual_loc,
            'avg_amount_7d': np.round(rng.lognormal(3.2, 0.8, n), 2),
            'transaction_count_7d': rng.integers(1, 50, n),
            'is_fraud': label
        })


class FeatureEngineer:
    """Feature engineering for fraud detection."""

    def __init__(self):
        self.le = LabelEncoder()
        self._fitted = False

    def fit_transform(self, df):
        df = df.copy()
        df['merchant_category_enc'] = self.le.fit_transform(df['merchant_category'])
        self._fitted = True
        return self._add_features(df)

    def transform(self, df):
        df = df.copy()
        df['merchant_category_enc'] = self.le.transform(df['merchant_category'])
        return self._add_features(df)

    def _add_features(self, df):
        df['amount_vs_avg'] = df['amount'] / (df['avg_amount_7d'] + 1e-6)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['high_velocity'] = (df['velocity_1h'] > 5).astype(int)
        df['log_amount'] = np.log1p(df['amount'])
        df['log_distance'] = np.log1p(df['distance_from_home_km'])
        df['manual_risk'] = (
            df['is_international'] * 3 +
            df['unusual_location'] * 2 +
            df['high_velocity'] * 2 +
            (~df['card_present'].astype(bool)).astype(int) * 2 +
            df['is_night']
        )
        return df

    @property
    def feature_columns(self):
        return [
            'amount', 'hour', 'day_of_week', 'merchant_category_enc',
            'velocity_1h', 'distance_from_home_km', 'is_international',
            'card_present', 'unusual_location', 'avg_amount_7d',
            'transaction_count_7d', 'amount_vs_avg', 'is_night',
            'is_weekend', 'high_velocity', 'log_amount', 'log_distance',
            'manual_risk'
        ]


class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest."""

    def __init__(self, contamination=0.02, n_estimators=200):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X):
        self.model.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X):
        raw = self.model.score_samples(self.scaler.transform(X))
        lo, hi = raw.min(), raw.max()
        return 1 - (raw - lo) / (hi - lo + 1e-9)

    def predict(self, X):
        preds = self.model.predict(self.scaler.transform(X))
        return (preds == -1).astype(int)


class RandomForestFraudModel:
    """Supervised fraud classifier using Random Forest."""

    def __init__(self, n_estimators=300):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class XGBoostFraudModel:
    """Supervised fraud classifier using XGBoost."""

    def __init__(self, scale_pos_weight=50):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def fit(self, X, y, X_val=None, y_val=None):
        X_s = self.scaler.fit_transform(X)
        eval_set = [(self.scaler.transform(X_val), y_val)] if X_val is not None else None
        self.model.fit(X_s, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class TransactionScorer:
    """Combine model scores into a unified fraud risk score 0-100."""

    RISK_THRESHOLDS = {
        'low': (0, 30), 'medium': (30, 60),
        'high': (60, 80), 'critical': (80, 101)
    }

    def __init__(self, w_anomaly=0.25, w_rf=0.35, w_xgb=0.40):
        total = w_anomaly + w_rf + w_xgb
        self.w_anom = w_anomaly / total
        self.w_rf   = w_rf / total
        self.w_xgb  = w_xgb / total

    def score(self, anomaly_score, rf_proba, xgb_proba):
        raw = (self.w_anom * anomaly_score + self.w_rf * rf_proba + self.w_xgb * xgb_proba) * 100
        risk = next(k for k, (lo, hi) in self.RISK_THRESHOLDS.items() if lo <= raw < hi)
        return {
            'fraud_score': round(raw, 2),
            'risk_level': risk,
            'is_fraud': raw >= 50,
            'anomaly_score': round(anomaly_score * 100, 2),
            'rf_probability': round(rf_proba * 100, 2),
            'xgb_probability': round(xgb_proba * 100, 2),
        }


class RealTimeFraudDetector:
    """End-to-end real-time fraud detection pipeline."""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.anomaly_detector  = AnomalyDetector(contamination=0.02)
        self.rf_model          = RandomForestFraudModel()
        self.xgb_model         = XGBoostFraudModel()
        self.scorer            = TransactionScorer()
        self._is_trained       = False

    def train(self, df=None, verbose=True):
        if df is None:
            if verbose: print("Generating synthetic transaction data...")
            df = TransactionDataGenerator().generate(n_samples=15000, fraud_rate=0.02)

        if verbose:
            n_fraud = df['is_fraud'].sum()
            print(f"Dataset: {len(df):,} rows | {n_fraud:,} fraud ({n_fraud/len(df)*100:.2f}%)")

        df_feat = self.feature_engineer.fit_transform(df)
        X = df_feat[self.feature_engineer.feature_columns].values
        y = df_feat['is_fraud'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

        if verbose: print("Training Isolation Forest...")
        self.anomaly_detector.fit(X_train)
        if verbose: print("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        if verbose: print("Training XGBoost...")
        self.xgb_model.fit(X_tr, y_tr, X_val, y_val)

        self._is_trained = True
        return self._evaluate(X_test, y_test, verbose)

    def _evaluate(self, X_test, y_test, verbose=True):
        rf_p  = self.rf_model.predict_proba(X_test)
        xgb_p = self.xgb_model.predict_proba(X_test)
        anom  = self.anomaly_detector.score(X_test)

        metrics = {}
        for name, proba, preds in [
            ('random_forest',    rf_p,  self.rf_model.predict(X_test)),
            ('xgboost',          xgb_p, self.xgb_model.predict(X_test)),
            ('isolation_forest', anom,  self.anomaly_detector.predict(X_test)),
        ]:
            metrics[name] = {
                'roc_auc':       round(roc_auc_score(y_test, proba), 4),
                'avg_precision': round(average_precision_score(y_test, proba), 4),
                'report':        classification_report(y_test, preds, output_dict=True),
            }

        if verbose:
            print("\n" + "="*55)
            print("MODEL EVALUATION")
            print("="*55)
            for m_name, m in metrics.items():
                cr = m['report']
                f1 = cr.get('1', cr.get(1, {})).get('f1-score', 0)
                print(f"{m_name:20s} | AUC={m['roc_auc']:.4f} | AP={m['avg_precision']:.4f} | F1={f1:.4f}")
        return metrics

    def predict(self, transaction):
        if not self._is_trained:
            raise RuntimeError("Call .train() first.")
        t0 = datetime.now()
        df_row = pd.DataFrame([transaction])
        df_feat = self.feature_engineer.transform(df_row)
        X = df_feat[self.feature_engineer.feature_columns].values

        result = self.scorer.score(
            float(self.anomaly_detector.score(X)[0]),
            float(self.rf_model.predict_proba(X)[0]),
            float(self.xgb_model.predict_proba(X)[0]),
        )
        result['processing_time_ms'] = round((datetime.now() - t0).total_seconds() * 1000, 2)
        result['timestamp']          = datetime.now().isoformat()
        result['explanation']        = self._explain(transaction, df_feat, result)
        return result

    def predict_batch(self, df):
        if not self._is_trained:
            raise RuntimeError("Call .train() first.")
        df_feat = self.feature_engineer.transform(df.copy())
        X = df_feat[self.feature_engineer.feature_columns].values
        anom  = self.anomaly_detector.score(X)
        rf_p  = self.rf_model.predict_proba(X)
        xgb_p = self.xgb_model.predict_proba(X)
        results = [self.scorer.score(anom[i], rf_p[i], xgb_p[i]) for i in range(len(df))]
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    def _explain(self, txn, feat_df, result):
        reasons = []
        if txn.get('amount', 0) > 500:
            reasons.append(f"High amount: ${txn['amount']:.2f}")
        if txn.get('velocity_1h', 0) > 5:
            reasons.append(f"High velocity: {txn['velocity_1h']} txns/hr")
        if txn.get('is_international', 0):
            reasons.append("International transaction")
        if txn.get('unusual_location', 0):
            reasons.append("Unusual location")
        if not txn.get('card_present', 1):
            reasons.append("Card not present (CNP)")
        if txn.get('distance_from_home_km', 0) > 100:
            reasons.append(f"Distance from home: {txn['distance_from_home_km']:.0f}km")
        hour = txn.get('hour', 12)
        if hour < 6 or hour > 22:
            reasons.append(f"Night transaction (hour={hour})")
        ratio = feat_df['amount_vs_avg'].iloc[0]
        if ratio > 3:
            reasons.append(f"Amount {ratio:.1f}x above 7d average")
        return reasons or ["No high-risk signals detected"]


def demo():
    print("=" * 55)
    print("  CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 55)

    detector = RealTimeFraudDetector()
    detector.train(verbose=True)

    print("\n--- Real-Time Predictions ---")
    tests = [
        {"name": "Normal grocery",
         "amount": 42.50, "hour": 14, "day_of_week": 2,
         "merchant_category": "grocery", "velocity_1h": 1,
         "distance_from_home_km": 3.2, "is_international": 0,
         "card_present": 1, "unusual_location": 0,
         "avg_amount_7d": 55.0, "transaction_count_7d": 12},
        {"name": "Suspicious large CNP",
         "amount": 2850.00, "hour": 3, "day_of_week": 6,
         "merchant_category": "online_shopping", "velocity_1h": 8,
         "distance_from_home_km": 4200.0, "is_international": 1,
         "card_present": 0, "unusual_location": 1,
         "avg_amount_7d": 65.0, "transaction_count_7d": 2},
    ]

    for txn in tests:
        name = txn.pop("name")
        r = detector.predict(txn)
        print(f"\n{name}:")
        print(f"  Score={r['fraud_score']:.1f}/100  Risk={r['risk_level'].upper()}")
        print(f"  Fraud={'YES' if r['is_fraud'] else 'NO'}  Latency={r['processing_time_ms']:.1f}ms")
        for reason in r['explanation']:
            print(f"  >> {reason}")


if __name__ == '__main__':
    demo()
