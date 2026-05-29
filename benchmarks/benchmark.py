"""
benchmarks/benchmark.py
========================
Systematic benchmarking framework for the churn prediction platform.

Benchmarks:
  1. Inference throughput  — single-record and batch (latency / RPS)
  2. Model comparison      — ANN vs Logistic Regression vs Random Forest vs XGBoost
  3. Scalability analysis  — performance across dataset sizes
  4. Preprocessing speed   — pipeline throughput
  5. Memory profiling      — model memory footprint

Outputs:
  - JSON benchmark reports
  - Console summary tables
  - matplotlib charts
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
# Result Containers
# ─────────────────────────────────────────────


@dataclass
class InferenceBenchmarkResult:
    model_name: str
    n_requests: int
    batch_size: int
    total_time_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_mb: float = 0.0

    def to_dict(self) -> Dict:
        return self.__dict__


@dataclass
class ModelComparisonResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    train_time_s: float
    inference_latency_ms: float
    n_parameters: int = 0

    def to_dict(self) -> Dict:
        return self.__dict__


# ─────────────────────────────────────────────
# Benchmark Suites
# ─────────────────────────────────────────────


class InferenceBenchmark:
    """
    Measures inference latency and throughput for a fitted Keras model.

    Usage
    -----
    >>> bench = InferenceBenchmark(model, n_warmup=50)
    >>> result = bench.run(X_test, batch_size=1, n_requests=1000)
    >>> bench.print_summary(result)
    """

    def __init__(self, model, n_warmup: int = 20):
        self.model = model
        self.n_warmup = n_warmup

    def run(
        self,
        X: np.ndarray,
        batch_size: int = 1,
        n_requests: int = 500,
    ) -> InferenceBenchmarkResult:
        """Benchmark inference with given batch size."""

        # Warmup — allow XLA compilation, caching
        logger.info("Warming up (%d requests, batch=%d)…", self.n_warmup, batch_size)
        for _ in range(self.n_warmup):
            batch = X[:batch_size]
            self.model.predict(batch, verbose=0)

        latencies: List[float] = []
        logger.info("Benchmarking %d requests…", n_requests)

        for i in range(n_requests):
            # Rotate through available data
            start_idx = (i * batch_size) % max(1, len(X) - batch_size)
            batch = X[start_idx: start_idx + batch_size]

            t0 = time.perf_counter()
            self.model.predict(batch, verbose=0)
            latencies.append((time.perf_counter() - t0) * 1000)

        total_time = sum(latencies) / 1000  # seconds
        throughput = (n_requests * batch_size) / total_time

        result = InferenceBenchmarkResult(
            model_name=getattr(self.model, "name", "unknown"),
            n_requests=n_requests,
            batch_size=batch_size,
            total_time_s=round(total_time, 3),
            mean_latency_ms=round(float(np.mean(latencies)), 3),
            p50_latency_ms=round(float(np.percentile(latencies, 50)), 3),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 3),
            p99_latency_ms=round(float(np.percentile(latencies, 99)), 3),
            throughput_rps=round(throughput, 1),
        )

        logger.info(
            "Benchmark [batch=%d]: mean=%.2fms p99=%.2fms throughput=%.0f RPS",
            batch_size, result.mean_latency_ms, result.p99_latency_ms, result.throughput_rps,
        )
        return result

    def run_scaling_analysis(
        self,
        X: np.ndarray,
        batch_sizes: Optional[List[int]] = None,
    ) -> List[InferenceBenchmarkResult]:
        """Benchmark across multiple batch sizes to show scaling behaviour."""
        batch_sizes = batch_sizes or [1, 4, 8, 16, 32, 64, 128, 256]
        results = []
        for bs in batch_sizes:
            n_req = max(100, 1000 // bs)
            result = self.run(X, batch_size=bs, n_requests=n_req)
            results.append(result)
        return results

    @staticmethod
    def print_summary(result: InferenceBenchmarkResult) -> None:
        print(f"\n{'─' * 50}")
        print(f"  Inference Benchmark — {result.model_name}")
        print(f"{'─' * 50}")
        print(f"  Batch size      : {result.batch_size}")
        print(f"  Requests        : {result.n_requests}")
        print(f"  Total time      : {result.total_time_s:.2f}s")
        print(f"  Throughput      : {result.throughput_rps:.0f} records/s")
        print(f"  Mean latency    : {result.mean_latency_ms:.2f}ms")
        print(f"  p50 latency     : {result.p50_latency_ms:.2f}ms")
        print(f"  p95 latency     : {result.p95_latency_ms:.2f}ms")
        print(f"  p99 latency     : {result.p99_latency_ms:.2f}ms")
        print(f"{'─' * 50}\n")


class ModelComparisonBenchmark:
    """
    Compares the ANN against classical ML baselines.

    Baselines: Logistic Regression, Random Forest, GradientBoosting, XGBoost (optional).
    """

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ann_model=None,
    ) -> List[ModelComparisonResult]:
        """Train and evaluate all models on the same split."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, precision_recall_curve
        )

        models_to_evaluate = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]

        try:
            from xgboost import XGBClassifier
            models_to_evaluate.append(
                ("XGBoost", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0))
            )
        except ImportError:
            logger.info("XGBoost not installed; skipping.")

        results = []

        for name, clf in models_to_evaluate:
            logger.info("Training %s…", name)
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            # Inference latency (100 single-record predictions)
            t0 = time.perf_counter()
            for _ in range(100):
                clf.predict_proba(X_test[:1])
            latency_ms = ((time.perf_counter() - t0) / 100) * 1000

            # PR-AUC
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = float(np.trapz(prec[::-1], rec[::-1]))

            results.append(ModelComparisonResult(
                model_name=name,
                accuracy=round(accuracy_score(y_test, y_pred), 4),
                precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
                recall=round(recall_score(y_test, y_pred, zero_division=0), 4),
                f1=round(f1_score(y_test, y_pred, zero_division=0), 4),
                roc_auc=round(roc_auc_score(y_test, y_proba), 4),
                pr_auc=round(pr_auc, 4),
                train_time_s=round(train_time, 3),
                inference_latency_ms=round(latency_ms, 3),
            ))

        if ann_model is not None:
            y_proba_ann = ann_model.predict(X_test, verbose=0).ravel()
            y_pred_ann = (y_proba_ann >= 0.5).astype(int)

            t0 = time.perf_counter()
            for _ in range(100):
                ann_model.predict(X_test[:1], verbose=0)
            latency_ms = ((time.perf_counter() - t0) / 100) * 1000

            prec, rec, _ = precision_recall_curve(y_test, y_proba_ann)
            pr_auc = float(np.trapz(prec[::-1], rec[::-1]))

            results.append(ModelComparisonResult(
                model_name="ANN (Keras)",
                accuracy=round(accuracy_score(y_test, y_pred_ann), 4),
                precision=round(precision_score(y_test, y_pred_ann, zero_division=0), 4),
                recall=round(recall_score(y_test, y_pred_ann, zero_division=0), 4),
                f1=round(f1_score(y_test, y_pred_ann, zero_division=0), 4),
                roc_auc=round(roc_auc_score(y_test, y_proba_ann), 4),
                pr_auc=round(pr_auc, 4),
                train_time_s=0.0,      # already trained
                inference_latency_ms=round(latency_ms, 3),
                n_parameters=int(ann_model.count_params()),
            ))

        return results

    @staticmethod
    def print_comparison_table(results: List[ModelComparisonResult]) -> None:
        header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ROC-AUC':>8} {'PR-AUC':>7} {'Train(s)':>9} {'Lat(ms)':>8}"
        print(f"\n{'─' * len(header)}")
        print(header)
        print("─" * len(header))
        for r in results:
            print(
                f"{r.model_name:<22} {r.accuracy:>6.4f} {r.precision:>6.4f} "
                f"{r.recall:>6.4f} {r.f1:>6.4f} {r.roc_auc:>8.4f} {r.pr_auc:>7.4f} "
                f"{r.train_time_s:>9.2f} {r.inference_latency_ms:>8.3f}"
            )
        print("─" * len(header) + "\n")


# ─────────────────────────────────────────────
# Report Writer
# ─────────────────────────────────────────────


class BenchmarkReporter:
    """Saves benchmark results to disk."""

    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_inference_results(self, results: List[InferenceBenchmarkResult]) -> Path:
        path = self.output_dir / "inference_benchmark.json"
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Inference benchmark saved: %s", path)
        return path

    def save_comparison_results(self, results: List[ModelComparisonResult]) -> Path:
        path = self.output_dir / "model_comparison.json"
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Model comparison saved: %s", path)
        return path

    def plot_inference_scaling(self, results: List[InferenceBenchmarkResult]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            batch_sizes = [r.batch_size for r in results]
            throughputs = [r.throughput_rps for r in results]
            p99s = [r.p99_latency_ms for r in results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(batch_sizes, throughputs, "o-", color="steelblue", linewidth=2)
            ax1.set_xlabel("Batch Size")
            ax1.set_ylabel("Throughput (records/s)")
            ax1.set_title("Inference Throughput vs Batch Size")
            ax1.set_xscale("log", base=2)
            ax1.grid(True, alpha=0.3)

            ax2.plot(batch_sizes, p99s, "o-", color="tomato", linewidth=2)
            ax2.set_xlabel("Batch Size")
            ax2.set_ylabel("p99 Latency (ms)")
            ax2.set_title("p99 Latency vs Batch Size")
            ax2.set_xscale("log", base=2)
            ax2.grid(True, alpha=0.3)

            fig.suptitle("Inference Scaling Analysis — ChurnANN", fontsize=13)
            fig.tight_layout()
            path = self.output_dir / "inference_scaling.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info("Inference scaling plot saved: %s", path)
        except Exception as e:
            logger.warning("Plotting failed: %s", e)
