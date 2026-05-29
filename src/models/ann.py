"""
models/ann.py
=============
Enterprise ANN model definition for binary churn classification.

Design goals:
  - Configurable architecture via ModelConfig dataclass
  - Clean separation of architecture definition, compilation, and training
  - Support for regularisation (dropout, L2 weight decay)
  - TensorFlow/Keras backend
  - Reproducible random seeding
  - Model summary and parameter count utilities
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Reproducibility ──────────────────────────


def set_global_seed(seed: int = 42) -> None:
    """Fix all RNG seeds for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────


@dataclass
class ModelConfig:
    """
    Declarative configuration for the ANN architecture and compilation.

    Attributes
    ----------
    input_dim : int
        Number of input features.
    hidden_units : list[int]
        Number of neurons in each hidden layer. Default is [6, 6].
    hidden_activation : str
        Activation function for hidden layers. Default 'relu'.
    output_activation : str
        Activation for the output neuron. Default 'sigmoid'.
    kernel_initializer : str
        Weight initialisation strategy.
    dropout_rate : float
        Dropout probability applied after each hidden layer (0 = disabled).
    l2_lambda : float
        L2 regularisation coefficient (0 = disabled).
    optimizer : str
        Keras optimizer name or config dict.
    learning_rate : float
        Learning rate (used when optimizer is a string name).
    loss : str
        Loss function name.
    metrics : list[str]
        Metrics to track during training.
    seed : int
        Random seed for reproducibility.
    """

    input_dim: int = 10
    hidden_units: List[int] = field(default_factory=lambda: [6, 6])
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    kernel_initializer: str = "glorot_uniform"
    dropout_rate: float = 0.0
    l2_lambda: float = 0.0
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = "binary_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "AUC"])
    seed: int = 42

    @property
    def n_hidden_layers(self) -> int:
        return len(self.hidden_units)

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "kernel_initializer": self.kernel_initializer,
            "dropout_rate": self.dropout_rate,
            "l2_lambda": self.l2_lambda,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "metrics": self.metrics,
            "seed": self.seed,
        }


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────


class ChurnANN:
    """
    Builder and wrapper for the churn-prediction ANN.

    Usage
    -----
    >>> config = ModelConfig(input_dim=10, hidden_units=[32, 16], dropout_rate=0.3)
    >>> ann = ChurnANN(config)
    >>> model = ann.build()
    >>> model.summary()
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")
        return self._model

    def build(self):
        """Construct, compile and return the Keras model."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        set_global_seed(self.config.seed)
        cfg = self.config

        reg = (
            regularizers.l2(cfg.l2_lambda)
            if cfg.l2_lambda > 0
            else None
        )

        inputs = keras.Input(shape=(cfg.input_dim,), name="features")
        x = inputs

        for i, units in enumerate(cfg.hidden_units):
            x = layers.Dense(
                units=units,
                activation=cfg.hidden_activation,
                kernel_initializer=cfg.kernel_initializer,
                kernel_regularizer=reg,
                name=f"hidden_{i + 1}",
            )(x)
            if cfg.dropout_rate > 0:
                x = layers.Dropout(rate=cfg.dropout_rate, name=f"dropout_{i + 1}")(x)

        outputs = layers.Dense(
            units=1,
            activation=cfg.output_activation,
            kernel_initializer=cfg.kernel_initializer,
            name="output",
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="ChurnANN")

        optimizer = self._build_optimizer(cfg)
        model.compile(
            optimizer=optimizer,
            loss=cfg.loss,
            metrics=self._resolve_metrics(cfg.metrics),
        )

        self._model = model
        self._log_summary(model)
        return model

    # ── Private ──────────────────────────────────

    def _build_optimizer(self, cfg: ModelConfig):
        """Create the Keras optimizer from config."""
        from tensorflow.keras import optimizers

        opt_map = {
            "adam": optimizers.Adam,
            "sgd": optimizers.SGD,
            "rmsprop": optimizers.RMSprop,
            "adamw": optimizers.AdamW,
        }
        if isinstance(cfg.optimizer, dict):
            name = cfg.optimizer.pop("name")
            return opt_map[name](learning_rate=cfg.learning_rate, **cfg.optimizer)

        cls = opt_map.get(cfg.optimizer.lower())
        if cls is None:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
        return cls(learning_rate=cfg.learning_rate)

    @staticmethod
    def _resolve_metrics(metric_names: List[str]):
        """Convert metric name strings to Keras metric objects."""
        from tensorflow.keras import metrics as km

        resolved = []
        for name in metric_names:
            if name == "AUC":
                resolved.append(km.AUC(name="auc", curve="ROC"))
            elif name == "PR_AUC":
                resolved.append(km.AUC(name="pr_auc", curve="PR"))
            elif name == "precision":
                resolved.append(km.Precision(name="precision"))
            elif name == "recall":
                resolved.append(km.Recall(name="recall"))
            else:
                resolved.append(name)
        return resolved

    @staticmethod
    def _log_summary(model) -> None:
        n_params = model.count_params()
        logger.info(
            "Model built: %s | Parameters: %d | Layers: %d",
            model.name,
            n_params,
            len(model.layers),
        )


# ─────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────


def baseline_config(input_dim: int = 10) -> ModelConfig:
    """Reproduce the original tutorial architecture exactly."""
    return ModelConfig(
        input_dim=input_dim,
        hidden_units=[6, 6],
        kernel_initializer="uniform",
        dropout_rate=0.0,
        l2_lambda=0.0,
        optimizer="adam",
        learning_rate=1e-3,
    )


def regularised_config(input_dim: int = 10) -> ModelConfig:
    """Deeper network with dropout and L2 regularisation."""
    return ModelConfig(
        input_dim=input_dim,
        hidden_units=[64, 32, 16],
        dropout_rate=0.3,
        l2_lambda=1e-4,
        optimizer="adam",
        learning_rate=5e-4,
        metrics=["accuracy", "AUC", "PR_AUC", "precision", "recall"],
    )


def wide_config(input_dim: int = 10) -> ModelConfig:
    """Wide shallow network for ablation studies."""
    return ModelConfig(
        input_dim=input_dim,
        hidden_units=[128, 64],
        dropout_rate=0.2,
        optimizer="adam",
        learning_rate=1e-3,
    )


# ─────────────────────────────────────────────
# Model I/O utilities
# ─────────────────────────────────────────────


def save_model(model, path: str | Path, config: Optional[ModelConfig] = None) -> None:
    """Save model weights and optionally its config."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info("Model saved to %s", path)

    if config:
        config_path = path.parent / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info("Model config saved to %s", config_path)


def load_model(path: str | Path):
    """Load a saved Keras model."""
    import tensorflow as tf
    model = tf.keras.models.load_model(str(path))
    logger.info("Model loaded from %s", path)
    return model
