"""
inference/inference_service.py
==============================
Production-grade inference service for the churn prediction ANN.

Provides:
  - Single-record synchronous prediction
  - Batch prediction
  - Model loading with health checking
  - Pre/post-processing pipeline integration
  - Latency tracking
  - Input validation via Pydantic
  - FastAPI application with OpenAPI docs
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Input / Output Schemas
# ─────────────────────────────────────────────


class CustomerFeatures(BaseModel):
    """
    Raw customer features for inference.
    Accepts both raw strings (Geography, Gender) and pre-encoded integers.
    """

    CreditScore: int = Field(..., ge=300, le=900, description="Customer credit score (300–900)")
    Geography: str = Field(..., description="Customer country: France | Germany | Spain")
    Gender: str = Field(..., description="Customer gender: Male | Female")
    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Tenure: int = Field(..., ge=0, le=10, description="Years as bank customer")
    Balance: float = Field(..., ge=0.0, description="Account balance in EUR")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products held")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card: 1=yes, 0=no")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Active member: 1=yes, 0=no")
    EstimatedSalary: float = Field(..., ge=0.0, description="Estimated annual salary in EUR")

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, v: str) -> str:
        valid = {"France", "Germany", "Spain"}
        if v not in valid:
            raise ValueError(f"Geography must be one of {valid}, got '{v}'")
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        valid = {"Male", "Female"}
        if v not in valid:
            raise ValueError(f"Gender must be one of {valid}, got '{v}'")
        return v


class PredictionResponse(BaseModel):
    """Single prediction response."""

    churn_probability: float = Field(..., description="Model output probability ∈ [0, 1]")
    churn_prediction: int = Field(..., description="Binary label (1=churn, 0=retain)")
    threshold: float = Field(..., description="Decision threshold used")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: str = Field(..., description="Loaded model identifier")


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


# ─────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────

GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}
FEATURE_ORDER = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


class InferenceEngine:
    """
    Stateful inference engine that holds the loaded model and scaler.

    This is a singleton in the FastAPI app. It is initialised once at startup
    and reused across all requests.
    """

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: Optional[str | Path] = None,
        threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.threshold = threshold
        self.model_version = self.model_path.stem
        self._model = None
        self._scaler = None
        self._start_time = time.time()

    def load(self) -> None:
        """Load model and scaler into memory."""
        import tensorflow as tf
        import pickle

        logger.info("Loading model from %s", self.model_path)
        self._model = tf.keras.models.load_model(str(self.model_path))
        logger.info("Model loaded successfully: %s", self.model_version)

        if self.scaler_path and self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("Scaler loaded from %s", self.scaler_path)
        else:
            logger.warning(
                "Scaler not found at %s. Predictions will use unscaled features.",
                self.scaler_path,
            )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict_single(
        self,
        customer: CustomerFeatures,
        threshold: Optional[float] = None,
    ) -> PredictionResponse:
        """Run inference for a single customer record."""
        t0 = time.perf_counter()
        features = self._preprocess(customer)
        prob = float(self._model.predict(features, verbose=0)[0][0])
        latency_ms = (time.perf_counter() - t0) * 1000

        thresh = threshold if threshold is not None else self.threshold
        label = int(prob >= thresh)

        return PredictionResponse(
            churn_probability=round(prob, 6),
            churn_prediction=label,
            threshold=thresh,
            latency_ms=round(latency_ms, 3),
            model_version=self.model_version,
        )

    def predict_batch(
        self,
        customers: List[CustomerFeatures],
        threshold: Optional[float] = None,
    ) -> BatchPredictionResponse:
        """Run vectorised batch inference."""
        t0 = time.perf_counter()
        features = np.vstack([self._preprocess(c) for c in customers])
        probs = self._model.predict(features, verbose=0).ravel()
        total_ms = (time.perf_counter() - t0) * 1000

        thresh = threshold if threshold is not None else self.threshold
        per_ms = total_ms / len(customers)

        predictions = [
            PredictionResponse(
                churn_probability=round(float(p), 6),
                churn_prediction=int(p >= thresh),
                threshold=thresh,
                latency_ms=round(per_ms, 3),
                model_version=self.model_version,
            )
            for p in probs
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(customers),
            total_latency_ms=round(total_ms, 3),
        )

    def _preprocess(self, customer: CustomerFeatures) -> np.ndarray:
        """Encode and scale a single CustomerFeatures record."""
        row = np.array([[
            customer.CreditScore,
            GEOGRAPHY_MAP[customer.Geography],
            GENDER_MAP[customer.Gender],
            customer.Age,
            customer.Tenure,
            customer.Balance,
            customer.NumOfProducts,
            customer.HasCrCard,
            customer.IsActiveMember,
            customer.EstimatedSalary,
        ]], dtype=np.float64)

        if self._scaler is not None:
            row = self._scaler.transform(row)

        return row

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok" if self.is_ready else "loading",
            model_loaded=self.is_ready,
            model_version=self.model_version,
            uptime_seconds=round(time.time() - self._start_time, 1),
        )


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────


def create_app(
    model_path: str = "artifacts/checkpoints/best_model.keras",
    scaler_path: str = "artifacts/pipeline/scaler.pkl",
    threshold: float = 0.5,
):
    """
    Factory function that creates and configures the FastAPI application.

    Usage (uvicorn)
    ---------------
    uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000

    Or via Docker:
    docker run -p 8000:8000 churn-platform:latest
    """
    try:
        from fastapi import FastAPI, HTTPException, Depends
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    engine = InferenceEngine(model_path=model_path, scaler_path=scaler_path, threshold=threshold)

    app = FastAPI(
        title="Customer Churn Prediction API",
        description=(
            "Enterprise-grade REST API for real-time customer churn scoring. "
            "Powered by a feedforward ANN trained on European retail banking data."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        engine.load()
        logger.info("Inference service started. Model: %s", engine.model_version)

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """System health check. Returns model status and uptime."""
        return engine.health()

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    async def predict(customer: CustomerFeatures, threshold: float = 0.5):
        """
        Predict churn probability for a single customer.

        Returns a binary prediction and the raw sigmoid probability.
        The `threshold` parameter controls the decision boundary (default 0.5).
        """
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model not yet loaded")
        return engine.predict_single(customer, threshold=threshold)

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
    async def predict_batch(request: BatchPredictionRequest):
        """
        Predict churn for a batch of customers in a single request.

        Efficiently processes up to 1,000 records via vectorised model inference.
        """
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model not yet loaded")
        if len(request.customers) > 1000:
            raise HTTPException(status_code=400, detail="Batch size must not exceed 1,000")
        return engine.predict_batch(request.customers, threshold=request.threshold)

    @app.get("/model/info", tags=["Model"])
    async def model_info():
        """Return model architecture metadata."""
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded")
        model = engine._model
        return {
            "name": model.name,
            "version": engine.model_version,
            "total_parameters": int(model.count_params()),
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "layers": [
                {"name": l.name, "type": type(l).__name__, "output_shape": str(l.output_shape)}
                for l in model.layers
            ],
        }

    return app


# Expose default app instance for uvicorn
app = create_app()
