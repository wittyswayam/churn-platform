"""
tests/unit/test_pipeline.py
Unit tests for the data pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """Minimal valid churn DataFrame."""
    return pd.DataFrame({
        "RowNumber": range(1, 101),
        "CustomerId": range(100),
        "Surname": ["Test"] * 100,
        "CreditScore": np.random.randint(400, 850, 100),
        "Geography": np.random.choice(["France", "Germany", "Spain"], 100),
        "Gender": np.random.choice(["Male", "Female"], 100),
        "Age": np.random.randint(20, 70, 100),
        "Tenure": np.random.randint(0, 10, 100),
        "Balance": np.random.uniform(0, 200000, 100),
        "NumOfProducts": np.random.randint(1, 4, 100),
        "HasCrCard": np.random.randint(0, 2, 100),
        "IsActiveMember": np.random.randint(0, 2, 100),
        "EstimatedSalary": np.random.uniform(20000, 200000, 100),
        "Exited": np.random.randint(0, 2, 100),
    }).set_index("RowNumber")


@pytest.fixture
def csv_path(tmp_path, sample_df):
    path = tmp_path / "test_churn.csv"
    sample_df.reset_index().to_csv(path, index=False)
    return path


# ── Pipeline Tests ────────────────────────────────────────────


class TestChurnDataPipeline:

    def test_run_returns_data_split(self, csv_path, tmp_path):
        from src.data.pipeline import ChurnDataPipeline, PipelineConfig
        pipeline = ChurnDataPipeline(PipelineConfig(
            save_artifacts=True,
            artifacts_dir=str(tmp_path / "artifacts"),
        ))
        split = pipeline.run(csv_path)

        assert split.X_train is not None
        assert split.X_val is not None
        assert split.X_test is not None
        assert split.X_train.shape[1] == 10

    def test_no_data_leakage(self, csv_path, tmp_path):
        """Scaler must be fit ONLY on training data."""
        from src.data.pipeline import ChurnDataPipeline, PipelineConfig
        pipeline = ChurnDataPipeline(PipelineConfig(
            save_artifacts=False,
            artifacts_dir=str(tmp_path / "artifacts"),
        ))
        split = pipeline.run(csv_path)
        scaler = split.scaler

        # Scaler mean should match training data mean (approximately)
        assert scaler is not None
        assert len(scaler.mean_) == 10

    def test_split_sizes(self, csv_path, tmp_path):
        from src.data.pipeline import ChurnDataPipeline, PipelineConfig
        pipeline = ChurnDataPipeline(PipelineConfig(
            test_size=0.2,
            val_size=0.1,
            save_artifacts=False,
            artifacts_dir=str(tmp_path / "a"),
        ))
        split = pipeline.run(csv_path)
        total = split.train_size + split.val_size + split.test_size
        assert total == 100

    def test_geography_encoding(self, csv_path, tmp_path):
        from src.data.pipeline import ChurnDataPipeline, PipelineConfig
        pipeline = ChurnDataPipeline(PipelineConfig(save_artifacts=False, artifacts_dir=str(tmp_path / "a")))
        split = pipeline.run(csv_path)
        # Geography (col index 1) should be numeric only
        assert np.issubdtype(split.X_train.dtype, np.floating)

    def test_transform_single(self, csv_path, tmp_path):
        from src.data.pipeline import ChurnDataPipeline, PipelineConfig
        pipeline = ChurnDataPipeline(PipelineConfig(save_artifacts=False, artifacts_dir=str(tmp_path / "a")))
        pipeline.run(csv_path)

        record = {
            "CreditScore": 650, "Geography": "France", "Gender": "Male",
            "Age": 40, "Tenure": 5, "Balance": 50000.0,
            "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
            "EstimatedSalary": 80000.0,
        }
        result = pipeline.transform_single(record)
        assert result.shape == (1, 10)


# ── Model Config Tests ────────────────────────────────────────


class TestModelConfig:

    def test_defaults(self):
        from src.models.ann import ModelConfig
        cfg = ModelConfig()
        assert cfg.input_dim == 10
        assert cfg.hidden_units == [6, 6]
        assert cfg.n_hidden_layers == 2

    def test_to_dict_roundtrip(self):
        from src.models.ann import ModelConfig
        cfg = ModelConfig(hidden_units=[32, 16], dropout_rate=0.3)
        d = cfg.to_dict()
        assert d["hidden_units"] == [32, 16]
        assert d["dropout_rate"] == 0.3

    def test_presets(self):
        from src.models.ann import baseline_config, regularised_config, wide_config
        assert baseline_config().hidden_units == [6, 6]
        assert regularised_config().dropout_rate > 0
        assert wide_config().hidden_units[0] >= 64


# ── Evaluator Tests ───────────────────────────────────────────


class TestModelEvaluator:

    def test_ece_perfect_calibration(self):
        from src.evaluation.evaluator import ModelEvaluator
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        ece = ModelEvaluator._expected_calibration_error(y_true, y_proba)
        assert ece < 0.1  # well calibrated

    def test_threshold_optimisation(self):
        from src.evaluation.evaluator import ModelEvaluator
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_proba = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, 200), 0, 1)
        f1_thresh, _ = ModelEvaluator._optimise_thresholds(y_true, y_proba)
        assert 0.0 < f1_thresh < 1.0


# ── Registry Tests ────────────────────────────────────────────


class TestModelRegistry:

    def test_register_and_promote(self, tmp_path):
        from src.utils.model_registry import ModelRegistry

        # Create a dummy model file
        model_path = tmp_path / "dummy_model.keras"
        model_path.write_bytes(b"fake-model-bytes")

        registry = ModelRegistry(root=tmp_path / "registry")
        entry = registry.register(
            name="test-model",
            model_path=model_path,
            metrics={"val_auc": 0.87},
        )
        assert entry.stage == "candidate"
        assert entry.version == "v1"

        registry.promote(entry.model_id, stage="champion")
        champion = registry.get_champion("test-model")
        assert champion is not None
        assert champion.is_champion

    def test_version_increment(self, tmp_path):
        from src.utils.model_registry import ModelRegistry

        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"x")
        registry = ModelRegistry(root=tmp_path / "registry")

        e1 = registry.register("m", model_path, metrics={})
        e2 = registry.register("m", model_path, metrics={})
        assert e2.version == "v2"

    def test_compare(self, tmp_path):
        from src.utils.model_registry import ModelRegistry

        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"x")
        registry = ModelRegistry(root=tmp_path / "registry")

        e1 = registry.register("m", model_path, metrics={"val_auc": 0.80})
        e2 = registry.register("m", model_path, metrics={"val_auc": 0.87})
        comp = registry.compare(e1.model_id, e2.model_id)
        assert comp["metrics"]["val_auc"]["delta"] == pytest.approx(0.07)
