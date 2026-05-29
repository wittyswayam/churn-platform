"""
training/trainer.py
===================
Enterprise-grade model trainer for the churn ANN.

Features
--------
- Configurable training loop via TrainingConfig
- Early stopping with patience
- Learning rate reduction on plateau
- TensorBoard callback integration
- MLflow experiment tracking (optional)
- Checkpoint management
- Class weight computation for imbalanced datasets
- History serialisation for reproducibility
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """All hyperparameters governing the training loop."""

    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.0     # 0 = use explicit val set
    early_stopping_patience: int = 15
    early_stopping_monitor: str = "val_loss"
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    reduce_lr_min: float = 1e-6
    use_class_weights: bool = True
    checkpoint_dir: str = "artifacts/checkpoints"
    tensorboard_log_dir: str = "artifacts/tensorboard"
    enable_tensorboard: bool = True
    enable_mlflow: bool = False
    mlflow_experiment: str = "churn-ann-baseline"
    mlflow_run_name: Optional[str] = None
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────
# Training Result
# ─────────────────────────────────────────────


@dataclass
class TrainingResult:
    """Container for training outputs."""

    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    best_val_auc: float
    train_duration_seconds: float
    model_path: str
    config: TrainingConfig

    @property
    def final_train_accuracy(self) -> float:
        return self.history.get("accuracy", [0.0])[-1]

    @property
    def final_val_accuracy(self) -> float:
        return self.history.get("val_accuracy", [0.0])[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_auc": self.best_val_auc,
            "train_duration_seconds": self.train_duration_seconds,
            "final_train_accuracy": self.final_train_accuracy,
            "final_val_accuracy": self.final_val_accuracy,
            "model_path": self.model_path,
        }


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────


class ChurnModelTrainer:
    """
    Orchestrates model training with enterprise-grade callbacks
    and optional MLflow experiment tracking.

    Usage
    -----
    >>> trainer = ChurnModelTrainer(training_config)
    >>> result = trainer.train(model, split.X_train, split.y_train,
    ...                        split.X_val, split.y_val)
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._mlflow_run = None

    def train(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_config: Optional[Dict] = None,
    ) -> TrainingResult:
        """
        Train `model` on the provided data.

        Parameters
        ----------
        model : keras.Model
            Compiled Keras model.
        X_train, y_train : arrays for training.
        X_val, y_val : arrays for validation.
        model_config : optional dict logged to MLflow.

        Returns
        -------
        TrainingResult
        """
        cfg = self.config
        callbacks = self._build_callbacks()

        class_weights = None
        if cfg.use_class_weights:
            class_weights = self._compute_class_weights(y_train)
            logger.info("Class weights: %s", class_weights)

        if cfg.enable_mlflow:
            self._start_mlflow(cfg, model_config)

        logger.info(
            "Training started — epochs=%d, batch_size=%d, train_size=%d, val_size=%d",
            cfg.epochs, cfg.batch_size, len(X_train), len(X_val),
        )

        t0 = time.perf_counter()
        history_obj = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=cfg.verbose,
        )
        duration = time.perf_counter() - t0

        history = history_obj.history
        best_epoch = int(np.argmin(history.get("val_loss", [0])))
        best_val_loss = float(min(history.get("val_loss", [float("inf")])))
        best_val_auc = float(max(history.get("val_auc", [0.0])))

        model_path = self._save_best_model(model)

        if cfg.enable_mlflow:
            self._log_mlflow_metrics(history, best_epoch, best_val_loss, best_val_auc, duration)
            self._end_mlflow()

        result = TrainingResult(
            history=history,
            best_epoch=best_epoch + 1,
            best_val_loss=best_val_loss,
            best_val_auc=best_val_auc,
            train_duration_seconds=round(duration, 2),
            model_path=model_path,
            config=cfg,
        )

        self._save_history(history, model_path)

        logger.info(
            "Training complete — best epoch: %d | val_loss: %.4f | val_auc: %.4f | time: %.1fs",
            result.best_epoch, best_val_loss, best_val_auc, duration,
        )
        return result

    # ── Private Helpers ──────────────────────────

    def _build_callbacks(self) -> list:
        """Assemble the Keras callback stack."""
        from tensorflow.keras import callbacks as cb

        cfg = self.config
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callback_list = [
            cb.EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            cb.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.reduce_lr_factor,
                patience=cfg.reduce_lr_patience,
                min_lr=cfg.reduce_lr_min,
                verbose=1,
            ),
            cb.ModelCheckpoint(
                filepath=str(checkpoint_dir / "checkpoint_{epoch:03d}_{val_loss:.4f}.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            cb.CSVLogger(
                filename=str(checkpoint_dir / "training_log.csv"),
                append=True,
            ),
        ]

        if cfg.enable_tensorboard:
            tb_dir = Path(cfg.tensorboard_log_dir)
            tb_dir.mkdir(parents=True, exist_ok=True)
            callback_list.append(
                cb.TensorBoard(
                    log_dir=str(tb_dir),
                    histogram_freq=1,
                    write_graph=True,
                )
            )

        return callback_list

    @staticmethod
    def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
        """Balanced class weights: n_samples / (n_classes * class_count)."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes.astype(int), weights))

    def _save_best_model(self, model) -> str:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        model_path = str(checkpoint_dir / "best_model.keras")
        model.save(model_path)
        logger.info("Best model saved: %s", model_path)
        return model_path

    @staticmethod
    def _save_history(history: Dict, model_path: str) -> None:
        history_path = Path(model_path).parent / "training_history.json"
        serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Training history saved: %s", history_path)

    # ── MLflow Integration ───────────────────────

    def _start_mlflow(self, cfg: TrainingConfig, model_config: Optional[Dict]) -> None:
        try:
            import mlflow
            mlflow.set_experiment(cfg.mlflow_experiment)
            self._mlflow_run = mlflow.start_run(run_name=cfg.mlflow_run_name)
            mlflow.log_params(cfg.to_dict())
            if model_config:
                mlflow.log_params({f"model_{k}": v for k, v in model_config.items()})
            logger.info("MLflow run started: %s", self._mlflow_run.info.run_id)
        except ImportError:
            logger.warning("MLflow not installed; skipping experiment tracking.")
            cfg.enable_mlflow = False

    def _log_mlflow_metrics(
        self,
        history: Dict,
        best_epoch: int,
        best_val_loss: float,
        best_val_auc: float,
        duration: float,
    ) -> None:
        try:
            import mlflow
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_val_auc", best_val_auc)
            mlflow.log_metric("train_duration_seconds", duration)
            # Log per-epoch metrics
            for metric_name, values in history.items():
                for epoch, val in enumerate(values):
                    mlflow.log_metric(metric_name, val, step=epoch)
        except Exception as e:
            logger.warning("MLflow metric logging failed: %s", e)

    def _end_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass
