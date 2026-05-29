"""
training/hyperparameter_search.py
==================================
Automated hyperparameter optimisation using Optuna.

Search space covers:
  - Network width (hidden unit counts per layer)
  - Network depth (number of hidden layers: 1–4)
  - Dropout rate
  - L2 regularisation coefficient
  - Learning rate
  - Batch size
  - Optimizer choice

Supports:
  - TPE sampler (default) — tree-structured Parzen estimator
  - Median pruner for early stopping of unpromising trials
  - MLflow callback for per-trial logging
  - Best-trial reproduction via saved config
  - Parallel execution across CPU cores
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Search Configuration
# ─────────────────────────────────────────────


@dataclass
class SearchConfig:
    """Configuration for the hyperparameter search."""

    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1                        # parallel trials (set -1 for all CPUs)
    direction: str = "maximize"            # 'maximize' val_auc or 'minimize' val_loss
    metric: str = "val_auc"
    sampler: str = "tpe"                   # 'tpe' | 'random' | 'cmaes'
    pruner: str = "median"                 # 'median' | 'hyperband' | 'none'
    study_name: str = "churn-ann-hpo"
    storage: Optional[str] = None         # e.g. "sqlite:///hpo.db" for persistence
    output_dir: str = "artifacts/hpo"
    enable_mlflow: bool = False

    # Fixed training config (not tuned)
    max_epochs: int = 50
    early_stopping_patience: int = 10


# ─────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────


class ChurnHPOObjective:
    """
    Optuna objective for churn ANN hyperparameter search.

    Each trial:
    1. Samples a configuration from the search space
    2. Builds and trains a model
    3. Returns the validation metric (AUC or loss)
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Optional[SearchConfig] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config or SearchConfig()

    def __call__(self, trial) -> float:
        """Called by Optuna for each trial."""
        model_params = self._sample_model_params(trial)
        train_params = self._sample_train_params(trial)

        model = self._build_model(model_params)
        val_metric = self._train_and_evaluate(model, trial, train_params)

        return val_metric

    def _sample_model_params(self, trial) -> Dict[str, Any]:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        units_per_layer = [
            trial.suggest_categorical(f"units_l{i}", [16, 32, 64, 128, 256])
            for i in range(n_layers)
        ]
        return {
            "hidden_units": units_per_layer,
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "l2_lambda": trial.suggest_float("l2_lambda", 1e-6, 1e-2, log=True),
            "kernel_initializer": trial.suggest_categorical(
                "kernel_initializer", ["glorot_uniform", "he_normal", "uniform"]
            ),
        }

    def _sample_train_params(self, trial) -> Dict[str, Any]:
        return {
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def _build_model(self, params: Dict) -> Any:
        """Build and compile a Keras model from trial params."""
        from src.models.ann import ChurnANN, ModelConfig

        config = ModelConfig(
            input_dim=self.X_train.shape[1],
            hidden_units=params["hidden_units"],
            dropout_rate=params["dropout_rate"],
            l2_lambda=params["l2_lambda"],
            kernel_initializer=params["kernel_initializer"],
        )
        return ChurnANN(config).build()

    def _train_and_evaluate(self, model, trial, params: Dict) -> float:
        """Train model and return the validation metric."""
        from tensorflow.keras.callbacks import EarlyStopping

        try:
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=self.config.max_epochs,
                batch_size=params["batch_size"],
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=self.config.early_stopping_patience,
                        restore_best_weights=True,
                    ),
                    _OptunaPruningCallback(trial, monitor="val_loss"),
                ],
                verbose=0,
            )

            if self.config.metric == "val_auc":
                # AUC key may be 'val_auc' or 'val_auc_1' depending on Keras version
                auc_keys = [k for k in history.history if k.startswith("val_auc")]
                if auc_keys:
                    return float(max(history.history[auc_keys[0]]))
                return 0.0
            else:
                return float(min(history.history.get("val_loss", [1.0])))

        except Exception as e:
            logger.warning("Trial failed: %s", e)
            raise


# ─────────────────────────────────────────────
# Pruning Callback
# ─────────────────────────────────────────────


class _OptunaPruningCallback:
    """Keras callback that reports intermediate values to Optuna for pruning."""

    def __init__(self, trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if logs is None:
            return
        value = logs.get(self.monitor)
        if value is not None:
            self.trial.report(float(value), step=epoch)
            if self.trial.should_prune():
                import optuna
                raise optuna.TrialPruned()


# ─────────────────────────────────────────────
# Search Orchestrator
# ─────────────────────────────────────────────


class HyperparameterSearch:
    """
    Runs Optuna-based hyperparameter search and produces a best-trial config.

    Usage
    -----
    >>> search = HyperparameterSearch(config)
    >>> best_params = search.run(X_train, y_train, X_val, y_val)
    >>> model = search.build_best_model(best_params, input_dim=10)
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self._study = None
        self._best_params: Optional[Dict] = None

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute the full hyperparameter search."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna")

        cfg = self.config
        sampler = self._build_sampler(cfg)
        pruner = self._build_pruner(cfg)

        study = optuna.create_study(
            study_name=cfg.study_name,
            direction=cfg.direction,
            sampler=sampler,
            pruner=pruner,
            storage=cfg.storage,
            load_if_exists=True,
        )
        self._study = study

        objective = ChurnHPOObjective(X_train, y_train, X_val, y_val, config=cfg)

        logger.info(
            "Starting HPO search: %d trials, metric=%s, direction=%s",
            cfg.n_trials, cfg.metric, cfg.direction,
        )

        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout_seconds,
            n_jobs=cfg.n_jobs,
            show_progress_bar=True,
        )

        self._best_params = study.best_params
        self._save_results(study)

        logger.info(
            "HPO complete. Best trial: #%d | %s=%.4f | params=%s",
            study.best_trial.number,
            cfg.metric,
            study.best_value,
            self._best_params,
        )

        return self._best_params

    def build_best_model(self, best_params: Dict, input_dim: int = 10):
        """Reconstruct the best model from the search results."""
        from src.models.ann import ChurnANN, ModelConfig

        n_layers = best_params["n_layers"]
        hidden_units = [best_params[f"units_l{i}"] for i in range(n_layers)]

        config = ModelConfig(
            input_dim=input_dim,
            hidden_units=hidden_units,
            dropout_rate=best_params.get("dropout_rate", 0.0),
            l2_lambda=best_params.get("l2_lambda", 0.0),
            kernel_initializer=best_params.get("kernel_initializer", "glorot_uniform"),
            optimizer=best_params.get("optimizer", "adam"),
            learning_rate=best_params.get("learning_rate", 1e-3),
        )
        return ChurnANN(config).build(), config

    def plot_optimisation(self) -> None:
        """Generate Optuna visualisation plots."""
        if self._study is None:
            raise RuntimeError("No study available. Run .run() first.")
        try:
            import optuna.visualization as vis
            import matplotlib
            matplotlib.use("Agg")

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            fig = vis.plot_optimization_history(self._study)
            fig.write_image(str(output_dir / "optimization_history.png"))

            fig = vis.plot_param_importances(self._study)
            fig.write_image(str(output_dir / "param_importances.png"))

            fig = vis.plot_parallel_coordinate(self._study)
            fig.write_image(str(output_dir / "parallel_coordinate.png"))

            logger.info("HPO plots saved to %s", output_dir)
        except Exception as e:
            logger.warning("HPO plotting failed: %s", e)

    def _save_results(self, study) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }

        with open(output_dir / "hpo_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Per-trial CSV
        df = study.trials_dataframe()
        df.to_csv(output_dir / "all_trials.csv", index=False)
        logger.info("HPO results saved to %s", output_dir)

    @staticmethod
    def _build_sampler(cfg: SearchConfig):
        import optuna
        if cfg.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=42)
        elif cfg.sampler == "random":
            return optuna.samplers.RandomSampler(seed=42)
        elif cfg.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        raise ValueError(f"Unknown sampler: {cfg.sampler}")

    @staticmethod
    def _build_pruner(cfg: SearchConfig):
        import optuna
        if cfg.pruner == "median":
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif cfg.pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif cfg.pruner == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(f"Unknown pruner: {cfg.pruner}")
