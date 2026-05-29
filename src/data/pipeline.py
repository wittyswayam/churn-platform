"""
data/pipeline.py
================
Enterprise data pipeline for customer churn prediction.

Implements a reproducible, leak-free preprocessing pipeline with:
  - Schema validation
  - Ordinal and one-hot encoding strategies
  - Stratified train/validation/test splitting
  - StandardScaler fit on train only
  - Artifact serialisation (scaler, encoder maps)
  - Feature-level provenance tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

RAW_SCHEMA: Dict[str, str] = {
    "CreditScore": "int64",
    "Geography": "object",
    "Gender": "object",
    "Age": "int64",
    "Tenure": "int64",
    "Balance": "float64",
    "NumOfProducts": "int64",
    "HasCrCard": "int64",
    "IsActiveMember": "int64",
    "EstimatedSalary": "float64",
    "Exited": "int64",
}

GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}

FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["CustomerId", "Surname"]
INDEX_COLUMN = "RowNumber"


# ─────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────


@dataclass
class DataSplit:
    """Typed container for a train/val/test split."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    data_hash: str = ""
    scaler: Optional[StandardScaler] = None

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    def class_distribution(self, split: str = "train") -> Dict[int, float]:
        """Return class proportions for a given split."""
        y = {"train": self.y_train, "val": self.y_val, "test": self.y_test}[split]
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {int(k): round(v / total, 4) for k, v in zip(unique, counts)}


@dataclass
class PipelineConfig:
    """All hyperparameters for the data pipeline."""

    test_size: float = 0.20
    val_size: float = 0.10          # fraction of train set used for validation
    random_state: int = 42
    stratify: bool = True
    geography_encoding: str = "ordinal"   # 'ordinal' | 'onehot'
    scale_features: bool = True
    save_artifacts: bool = True
    artifacts_dir: str = "artifacts/pipeline"


# ─────────────────────────────────────────────
# Core Pipeline
# ─────────────────────────────────────────────


class ChurnDataPipeline:
    """
    End-to-end data preprocessing pipeline.

    Usage
    -----
    >>> pipeline = ChurnDataPipeline(config)
    >>> split = pipeline.run("data/raw/ChurnPrediction.csv")
    >>> X_train, y_train = split.X_train, split.y_train
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._scaler: Optional[StandardScaler] = None
        self._encoding_maps: Dict[str, Dict] = {}
        self._feature_names: list[str] = []

    # ── Public API ──────────────────────────────

    def run(self, csv_path: str | Path) -> DataSplit:
        """Execute the full pipeline and return a typed DataSplit."""
        logger.info("Starting ChurnDataPipeline on %s", csv_path)

        df_raw = self._load(csv_path)
        self._validate_schema(df_raw)

        df_clean = self._clean(df_raw)
        df_encoded = self._encode(df_clean)

        X, y = self._split_features_target(df_encoded)
        self._feature_names = list(X.columns)

        data_hash = self._hash_dataframe(df_encoded)
        logger.info("Dataset fingerprint (SHA-256): %s", data_hash)

        X_trainval, X_test, y_trainval, y_test = self._split(X, y, self.config.test_size)
        val_frac = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = self._split(X_trainval, y_trainval, val_frac)

        X_train_sc, X_val_sc, X_test_sc = self._scale(X_train, X_val, X_test)

        split = DataSplit(
            X_train=X_train_sc,
            X_val=X_val_sc,
            X_test=X_test_sc,
            y_train=y_train.values,
            y_val=y_val.values,
            y_test=y_test.values,
            feature_names=self._feature_names,
            data_hash=data_hash,
            scaler=self._scaler,
        )

        self._log_split_summary(split)

        if self.config.save_artifacts:
            self._save_artifacts()

        return split

    def transform_single(self, raw_features: Dict) -> np.ndarray:
        """
        Transform a single raw customer record for inference.

        Parameters
        ----------
        raw_features : dict
            Keys matching FEATURE_COLUMNS with raw (pre-encoding) values.

        Returns
        -------
        np.ndarray of shape (1, n_features), scaled.
        """
        if self._scaler is None:
            raise RuntimeError("Pipeline has not been fitted. Call run() first.")

        row = {k: [v] for k, v in raw_features.items()}
        df = pd.DataFrame(row)

        # Apply same encoding
        for col, mapping in self._encoding_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        df = df[self._feature_names]
        return self._scaler.transform(df.values)

    # ── Private Helpers ──────────────────────────

    def _load(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=INDEX_COLUMN)
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in RAW_SCHEMA if c not in df.columns and c not in DROP_COLUMNS]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        null_counts = df[list(RAW_SCHEMA.keys())].isnull().sum()
        if null_counts.any():
            logger.warning("Null values detected:\n%s", null_counts[null_counts > 0])

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
        # Remove exact duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            logger.warning("Removed %d duplicate rows.", before - after)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.config.geography_encoding == "ordinal":
            df["Geography"] = df["Geography"].map(GEOGRAPHY_MAP)
            self._encoding_maps["Geography"] = GEOGRAPHY_MAP
        elif self.config.geography_encoding == "onehot":
            geo_dummies = pd.get_dummies(df["Geography"], prefix="geo", drop_first=True)
            df = pd.concat([df.drop("Geography", axis=1), geo_dummies], axis=1)

        df["Gender"] = df["Gender"].map(GENDER_MAP)
        self._encoding_maps["Gender"] = GENDER_MAP

        logger.debug("Encoding complete. Shape after encode: %s", df.shape)
        return df

    def _split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        return df[feature_cols], df[TARGET_COLUMN]

    def _split(
        self, X: pd.DataFrame, y: pd.Series, test_size: float
    ) -> Tuple:
        strat = y if self.config.stratify else None
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=strat,
        )

    def _scale(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.config.scale_features:
            return X_train.values, X_val.values, X_test.values

        self._scaler = StandardScaler()
        X_train_sc = self._scaler.fit_transform(X_train)   # fit ONLY on train
        X_val_sc = self._scaler.transform(X_val)
        X_test_sc = self._scaler.transform(X_test)
        logger.info("StandardScaler fitted on %d training samples.", len(X_train))
        return X_train_sc, X_val_sc, X_test_sc

    def _save_artifacts(self) -> None:
        out_dir = Path(self.config.artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._scaler:
            with open(out_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self._scaler, f)

        with open(out_dir / "encoding_maps.json", "w") as f:
            json.dump(self._encoding_maps, f, indent=2)

        with open(out_dir / "feature_names.json", "w") as f:
            json.dump(self._feature_names, f, indent=2)

        logger.info("Pipeline artifacts saved to %s", out_dir)

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """Deterministic SHA-256 fingerprint of a DataFrame."""
        h = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        )
        return h.hexdigest()

    @staticmethod
    def _log_split_summary(split: DataSplit) -> None:
        logger.info(
            "Split summary — Train: %d | Val: %d | Test: %d",
            split.train_size, split.val_size, split.test_size,
        )
        for s in ("train", "val", "test"):
            dist = split.class_distribution(s)
            logger.info("  %s class distribution: %s", s, dist)


# ─────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────


def load_pipeline_from_artifacts(artifacts_dir: str | Path) -> ChurnDataPipeline:
    """Reconstruct a fitted pipeline from saved artifacts (for inference)."""
    artifacts_dir = Path(artifacts_dir)
    pipeline = ChurnDataPipeline()

    scaler_path = artifacts_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            pipeline._scaler = pickle.load(f)

    enc_path = artifacts_dir / "encoding_maps.json"
    if enc_path.exists():
        with open(enc_path) as f:
            pipeline._encoding_maps = json.load(f)

    feat_path = artifacts_dir / "feature_names.json"
    if feat_path.exists():
        with open(feat_path) as f:
            pipeline._feature_names = json.load(f)

    logger.info("Pipeline artifacts loaded from %s", artifacts_dir)
    return pipeline
