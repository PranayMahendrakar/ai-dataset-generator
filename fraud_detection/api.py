"""
Fraud Detection REST API
========================
FastAPI-based REST endpoint for real-time transaction scoring.
Run: uvicorn fraud_detection.api:app --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import asyncio
import logging

from fraud_detection.fraud_detector import RealTimeFraudDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud scoring with Isolation Forest, Random Forest & XGBoost",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton detector (loaded once at startup)
detector = RealTimeFraudDetector()


@app.on_event("startup")
async def startup_event():
    logger.info("Training fraud detection models...")
    detector.train(verbose=True)
    logger.info("Models ready.")


# ─── Schemas ─────────────────────────────────────────────────────────────────
MERCHANT_CATEGORIES = [
    'grocery', 'restaurant', 'gas_station', 'online_shopping',
    'entertainment', 'travel', 'healthcare', 'electronics',
    'clothing', 'atm_withdrawal',
]


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon)")
    merchant_category: str = Field(..., description="Merchant category")
    velocity_1h: int = Field(..., ge=0, description="Transactions in last hour")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from home (km)")
    is_international: int = Field(..., ge=0, le=1, description="1 if international")
    card_present: int = Field(..., ge=0, le=1, description="1 if card physically present")
    unusual_location: int = Field(..., ge=0, le=1, description="1 if unusual location")
    avg_amount_7d: float = Field(..., gt=0, description="Avg transaction amount (7 days)")
    transaction_count_7d: int = Field(..., ge=0, description="Transaction count (7 days)")

    @validator('merchant_category')
    def validate_merchant(cls, v):
        if v not in MERCHANT_CATEGORIES:
            raise ValueError(f"merchant_category must be one of {MERCHANT_CATEGORIES}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "amount": 299.99,
                "hour": 14,
                "day_of_week": 2,
                "merchant_category": "online_shopping",
                "velocity_1h": 2,
                "distance_from_home_km": 5.0,
                "is_international": 0,
                "card_present": 1,
                "unusual_location": 0,
                "avg_amount_7d": 85.0,
                "transaction_count_7d": 18,
            }
        }


class FraudPrediction(BaseModel):
    fraud_score: float
    risk_level: str
    is_fraud: bool
    anomaly_score: float
    rf_probability: float
    xgb_probability: float
    processing_time_ms: float
    timestamp: str
    explanation: List[str]


class BatchTransaction(BaseModel):
    transactions: List[Transaction]


class BatchPrediction(BaseModel):
    results: List[FraudPrediction]
    total: int
    flagged_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    version: str
    timestamp: str


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_trained=detector._is_trained,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=FraudPrediction, tags=["Fraud Detection"])
async def predict_fraud(transaction: Transaction):
    """
    Score a single transaction for fraud risk.

    Returns a fraud score (0-100), risk level, and explanation.
    """
    if not detector._is_trained:
        raise HTTPException(status_code=503, detail="Model not yet trained.")
    try:
        result = detector.predict(transaction.dict())
        return FraudPrediction(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Fraud Detection"])
async def predict_fraud_batch(batch: BatchTransaction):
    """
    Score multiple transactions in a single request.

    Efficient for bulk processing up to 1,000 transactions.
    """
    if not detector._is_trained:
        raise HTTPException(status_code=503, detail="Model not yet trained.")
    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1,000 transactions per batch.")

    t0 = datetime.now()
    try:
        results = []
        for txn in batch.transactions:
            r = detector.predict(txn.dict())
            results.append(FraudPrediction(**r))

        total_ms = (datetime.now() - t0).total_seconds() * 1000
        flagged  = sum(1 for r in results if r.is_fraud)

        return BatchPrediction(
            results=results,
            total=len(results),
            flagged_count=flagged,
            processing_time_ms=round(total_ms, 2),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/merchants", tags=["Reference"])
async def list_merchant_categories():
    """List all valid merchant category values."""
    return {"merchant_categories": MERCHANT_CATEGORIES}


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return model configuration details."""
    return {
        "models": ["IsolationForest", "RandomForest", "XGBoost"],
        "features": 18,
        "fraud_score_range": "0-100",
        "risk_levels": ["low", "medium", "high", "critical"],
        "score_weights": {
            "isolation_forest": "25%",
            "random_forest":    "35%",
            "xgboost":          "40%",
        },
    }
