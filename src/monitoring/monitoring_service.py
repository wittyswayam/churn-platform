"""
monitoring/monitoring_service.py
=================================
Production monitoring and telemetry for the churn inference service.

Tracks:
  - Request latency (p50, p95, p99)
  - Prediction distribution (score histogram, churn rate)
  - Input feature drift detection (Population Stability Index)
  - Model performance degradation alerts
  - Prometheus metrics exposition
  - Structured JSON logging for ELK/Loki ingestion

Design pattern: lightweight pull-based metrics via Prometheus;
push-based alerting via configurable webhook notifiers.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Metric Primitives
# ─────────────────────────────────────────────


class RollingStats:
    """Thread-safe rolling window statistics tracker."""

    def __init__(self, window_size: int = 1000):
        self._window: Deque[float] = deque(maxlen=window_size)
        self._lock = Lock()
        self._count_total: int = 0

    def record(self, value: float) -> None:
        with self._lock:
            self._window.append(value)
            self._count_total += 1

    def percentile(self, pct: float) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            return float(np.percentile(list(self._window), pct))

    @property
    def mean(self) -> float:
        with self._lock:
            return statistics.mean(self._window) if self._window else 0.0

    @property
    def count(self) -> int:
        return self._count_total

    def snapshot(self) -> Dict[str, float]:
        return {
            "count": self._count_total,
            "mean": self.mean,
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
        }


# ─────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────


class PSICalculator:
    """
    Population Stability Index (PSI) for detecting input feature drift.

    PSI < 0.10  → no significant change
    PSI 0.10–0.25 → moderate change; investigate
    PSI > 0.25  → major shift; retrain likely needed
    """

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self._reference_histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def fit_reference(self, X: np.ndarray, feature_names: List[str]) -> None:
        """Compute reference histograms from training data."""
        for i, name in enumerate(feature_names):
            counts, bin_edges = np.histogram(X[:, i], bins=self.n_bins)
            self._reference_histograms[name] = (counts / counts.sum(), bin_edges)
        logger.info("PSI reference histograms fitted for %d features.", len(feature_names))

    def compute_psi(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Compute PSI for each feature between reference and current batch."""
        psi_scores: Dict[str, float] = {}

        for i, name in enumerate(feature_names):
            if name not in self._reference_histograms:
                continue

            ref_dist, bin_edges = self._reference_histograms[name]
            curr_counts, _ = np.histogram(X[:, i], bins=bin_edges)
            curr_dist = curr_counts / (curr_counts.sum() + self.epsilon)

            # PSI = Σ (P_curr - P_ref) * ln(P_curr / P_ref)
            ref_dist = np.clip(ref_dist, self.epsilon, None)
            curr_dist = np.clip(curr_dist, self.epsilon, None)
            psi = float(np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist)))
            psi_scores[name] = round(psi, 6)

        return psi_scores

    def flag_drifted_features(
        self, psi_scores: Dict[str, float], threshold: float = 0.25
    ) -> List[str]:
        return [f for f, psi in psi_scores.items() if psi > threshold]


# ─────────────────────────────────────────────
# Alert System
# ─────────────────────────────────────────────


@dataclass
class Alert:
    level: str        # 'warning' | 'critical'
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AlertManager:
    """Manages alert generation and delivery."""

    def __init__(self, alert_log_path: Optional[str] = None):
        self._alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._log_path = Path(alert_log_path) if alert_log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        self._handlers.append(handler)

    def emit(self, level: str, message: str, **metadata) -> Alert:
        alert = Alert(level=level, message=message, metadata=metadata)
        self._alerts.append(alert)

        log_fn = logger.warning if level == "warning" else logger.error
        log_fn("[%s] %s | %s", level.upper(), message, metadata)

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler failed: %s", e)

        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")

        return alert

    @property
    def recent_alerts(self) -> List[Alert]:
        return self._alerts[-50:]


# ─────────────────────────────────────────────
# Monitoring Service
# ─────────────────────────────────────────────


class MonitoringService:
    """
    Central monitoring service for the churn inference platform.

    Tracks latency, prediction distribution, and input drift.
    Exposes a /metrics endpoint compatible with Prometheus.

    Usage
    -----
    >>> monitor = MonitoringService()
    >>> monitor.record_prediction(probability=0.73, latency_ms=4.2, features=X)
    >>> report = monitor.get_report()
    """

    def __init__(
        self,
        churn_alert_threshold: float = 0.70,    # alert if rolling churn rate > 70%
        latency_p99_threshold_ms: float = 100.0,
        drift_check_interval: int = 500,          # check drift every N predictions
        output_dir: str = "artifacts/monitoring",
    ):
        self.churn_alert_threshold = churn_alert_threshold
        self.latency_p99_threshold = latency_p99_threshold_ms
        self.drift_check_interval = drift_check_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metric trackers
        self._latency = RollingStats(window_size=1000)
        self._churn_probs: Deque[float] = deque(maxlen=1000)
        self._churn_labels: Deque[int] = deque(maxlen=1000)
        self._request_count: int = 0
        self._error_count: int = 0
        self._lock = Lock()

        # Drift detection
        self._psi_calc = PSICalculator()
        self._psi_scores: Dict[str, float] = {}
        self._feature_buffer: List[np.ndarray] = []

        # Alerting
        self.alerts = AlertManager(str(self.output_dir / "alerts.jsonl"))
        self.alerts.register_handler(self._log_alert)

        self._start_time = time.time()

    # ── Public API ──────────────────────────────

    def record_prediction(
        self,
        probability: float,
        latency_ms: float,
        features: Optional[np.ndarray] = None,
        error: bool = False,
    ) -> None:
        """Record a single inference event."""
        with self._lock:
            self._request_count += 1
            if error:
                self._error_count += 1
                return

            self._latency.record(latency_ms)
            self._churn_probs.append(probability)
            self._churn_labels.append(int(probability >= 0.5))

            if features is not None:
                self._feature_buffer.append(features.ravel())

        # Periodic drift check
        if self._request_count % self.drift_check_interval == 0:
            self._run_drift_check()

        # Threshold checks
        self._check_thresholds(latency_ms)

    def set_reference_distribution(
        self, X: np.ndarray, feature_names: List[str]
    ) -> None:
        """Set the training-time distribution for drift detection."""
        self._psi_calc.fit_reference(X, feature_names)
        self._feature_names = feature_names

    def get_report(self) -> Dict[str, Any]:
        """Return a full monitoring snapshot."""
        churn_probs = list(self._churn_probs)
        churn_labels = list(self._churn_labels)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "requests": {
                "total": self._request_count,
                "errors": self._error_count,
                "error_rate": round(self._error_count / max(self._request_count, 1), 4),
            },
            "latency_ms": self._latency.snapshot(),
            "predictions": {
                "churn_rate": round(sum(churn_labels) / max(len(churn_labels), 1), 4),
                "mean_probability": round(float(np.mean(churn_probs)) if churn_probs else 0.0, 4),
                "p95_probability": round(float(np.percentile(churn_probs, 95)) if churn_probs else 0.0, 4),
            },
            "drift": {
                "psi_scores": self._psi_scores,
                "drifted_features": self._psi_calc.flag_drifted_features(self._psi_scores),
            },
            "alerts": {
                "total": len(self.alerts.recent_alerts),
                "recent": [a.to_dict() for a in self.alerts.recent_alerts[-5:]],
            },
        }

    def export_prometheus_metrics(self) -> str:
        """Generate Prometheus text format metrics."""
        report = self.get_report()
        lines = [
            "# HELP churn_requests_total Total inference requests",
            "# TYPE churn_requests_total counter",
            f"churn_requests_total {report['requests']['total']}",
            "",
            "# HELP churn_error_rate Fraction of errored requests",
            "# TYPE churn_error_rate gauge",
            f"churn_error_rate {report['requests']['error_rate']}",
            "",
            "# HELP churn_latency_ms_p99 99th percentile inference latency",
            "# TYPE churn_latency_ms_p99 gauge",
            f"churn_latency_ms_p99 {report['latency_ms']['p99']}",
            "",
            "# HELP churn_prediction_rate Rolling churn prediction rate",
            "# TYPE churn_prediction_rate gauge",
            f"churn_prediction_rate {report['predictions']['churn_rate']}",
        ]

        for feat, psi in report["drift"]["psi_scores"].items():
            lines += [
                f"# HELP churn_psi_{feat} Population Stability Index for {feat}",
                f"# TYPE churn_psi_{feat} gauge",
                f"churn_psi_{feat} {psi}",
            ]

        return "\n".join(lines)

    def save_report(self) -> Path:
        report = self.get_report()
        path = self.output_dir / "monitoring_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Monitoring report saved: %s", path)
        return path

    # ── Private ──────────────────────────────────

    def _run_drift_check(self) -> None:
        if not self._feature_buffer:
            return
        X_current = np.vstack(self._feature_buffer)
        self._feature_buffer.clear()

        try:
            feature_names = getattr(self, "_feature_names", [])
            if feature_names:
                self._psi_scores = self._psi_calc.compute_psi(X_current, feature_names)
                drifted = self._psi_calc.flag_drifted_features(self._psi_scores)
                if drifted:
                    self.alerts.emit(
                        "warning",
                        f"Feature drift detected in: {drifted}",
                        psi_scores={f: self._psi_scores[f] for f in drifted},
                    )
        except Exception as e:
            logger.warning("Drift check failed: %s", e)

    def _check_thresholds(self, latency_ms: float) -> None:
        if latency_ms > self.latency_p99_threshold:
            self.alerts.emit(
                "warning",
                f"High latency: {latency_ms:.1f}ms (threshold: {self.latency_p99_threshold}ms)",
                latency_ms=latency_ms,
            )

        if len(self._churn_labels) >= 100:
            rate = sum(list(self._churn_labels)[-100:]) / 100
            if rate > self.churn_alert_threshold:
                self.alerts.emit(
                    "critical",
                    f"Elevated churn rate: {rate:.1%} (threshold: {self.churn_alert_threshold:.1%})",
                    churn_rate=rate,
                )

    @staticmethod
    def _log_alert(alert: Alert) -> None:
        logger.warning("ALERT [%s]: %s", alert.level, alert.message)
