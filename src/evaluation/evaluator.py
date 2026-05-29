"""
evaluation/evaluator.py
=======================
Comprehensive evaluation suite for binary churn classification.

Computes and reports:
  - Accuracy, precision, recall, F1 (macro + weighted + per-class)
  - ROC-AUC and PR-AUC
  - Confusion matrix (raw + normalised)
  - Calibration metrics (Brier score, ECE)
  - Business metrics (expected retention value at threshold)
  - Threshold optimisation (maximise F1, maximise recall@precision)
  - Saving evaluation reports as JSON and plots as PNG
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Full evaluation output for one model/split combination."""

    split: str                         # 'test', 'val', 'train'
    threshold: float = 0.5

    # Scalar metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0

    # Per-class
    class_report: Dict = field(default_factory=dict)

    # Confusion matrix
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    # Threshold analysis
    optimal_threshold_f1: float = 0.5
    optimal_threshold_recall_90p: Optional[float] = None  # threshold for recall ≥ 0.9

    # Calibration
    ece: float = 0.0   # Expected Calibration Error

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def false_negative_rate(self) -> float:
        """Miss rate: fraction of churners the model fails to catch."""
        denom = self.tp + self.fn
        return self.fn / denom if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────


class ModelEvaluator:
    """
    Evaluates a trained churn prediction model.

    Usage
    -----
    >>> evaluator = ModelEvaluator(output_dir="artifacts/evaluation")
    >>> result = evaluator.evaluate(model, X_test, y_test, split="test")
    >>> evaluator.save_report(result)
    >>> evaluator.plot_all(result, y_test, y_proba)
    """

    def __init__(
        self,
        output_dir: str = "artifacts/evaluation",
        threshold: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    def evaluate(
        self,
        model,
        X: np.ndarray,
        y_true: np.ndarray,
        split: str = "test",
    ) -> EvaluationResult:
        """Run the full evaluation suite."""
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_proba = model.predict(X, verbose=0).ravel()
        y_pred = (y_proba >= self.threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        roc_auc = roc_auc_score(y_true, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(np.trapz(precision_curve[::-1], recall_curve[::-1]))

        report = classification_report(y_true, y_pred, output_dict=True)

        opt_thresh_f1, recall_90p_thresh = self._optimise_thresholds(y_true, y_proba)
        ece = self._expected_calibration_error(y_true, y_proba)

        result = EvaluationResult(
            split=split,
            threshold=self.threshold,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier_score=brier_score_loss(y_true, y_proba),
            log_loss=log_loss(y_true, y_proba),
            class_report=report,
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            optimal_threshold_f1=opt_thresh_f1,
            optimal_threshold_recall_90p=recall_90p_thresh,
            ece=ece,
        )

        self._log_result(result)
        return result

    # ── Threshold Analysis ───────────────────────

    @staticmethod
    def _optimise_thresholds(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Tuple[float, Optional[float]]:
        """Find threshold that maximises F1 and threshold for recall ≥ 0.90."""
        from sklearn.metrics import f1_score, precision_recall_curve

        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.0)

        f1_scores = np.where(
            (precision_curve + recall_curve) == 0,
            0,
            2 * precision_curve * recall_curve / (precision_curve + recall_curve),
        )
        best_f1_idx = int(np.argmax(f1_scores))
        optimal_f1_threshold = float(thresholds[best_f1_idx])

        # Threshold for recall ≥ 0.90 with maximum precision
        recall_90_mask = recall_curve >= 0.90
        if recall_90_mask.any():
            best_idx = np.argmax(precision_curve[recall_90_mask])
            recall_90p_threshold = float(thresholds[recall_90_mask][best_idx])
        else:
            recall_90p_threshold = None

        return optimal_f1_threshold, recall_90p_threshold

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        n = len(y_true)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            avg_confidence = float(y_proba[mask].mean())
            avg_accuracy = float(y_true[mask].mean())
            ece += (mask.sum() / n) * abs(avg_confidence - avg_accuracy)
        return round(ece, 6)

    # ── Report & Logging ─────────────────────────

    def save_report(self, result: EvaluationResult) -> Path:
        """Serialise EvaluationResult to JSON."""
        path = self.output_dir / f"evaluation_{result.split}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Evaluation report saved: %s", path)
        return path

    def print_report(self, result: EvaluationResult) -> None:
        """Print a formatted summary to stdout."""
        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  Evaluation Report — split={result.split.upper()}")
        print(sep)
        print(f"  Threshold     : {result.threshold:.2f}")
        print(f"  Accuracy      : {result.accuracy:.4f}")
        print(f"  Precision     : {result.precision:.4f}")
        print(f"  Recall        : {result.recall:.4f}")
        print(f"  F1 Score      : {result.f1:.4f}")
        print(f"  ROC-AUC       : {result.roc_auc:.4f}")
        print(f"  PR-AUC        : {result.pr_auc:.4f}")
        print(f"  Brier Score   : {result.brier_score:.4f}")
        print(f"  ECE           : {result.ece:.4f}")
        print(sep)
        print(f"  Confusion Matrix:")
        print(f"    TN={result.tn:5d}  FP={result.fp:5d}")
        print(f"    FN={result.fn:5d}  TP={result.tp:5d}")
        print(f"  FNR (Miss Rate): {result.false_negative_rate:.4f}")
        print(sep)
        print(f"  Optimal threshold (max F1):     {result.optimal_threshold_f1:.3f}")
        if result.optimal_threshold_recall_90p:
            print(f"  Threshold (recall ≥ 90%):       {result.optimal_threshold_recall_90p:.3f}")
        print(f"{sep}\n")

    # ── Plotting ──────────────────────────────────

    def plot_all(
        self,
        result: EvaluationResult,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> None:
        """Generate and save all evaluation plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import (
                RocCurveDisplay,
                PrecisionRecallDisplay,
                ConfusionMatrixDisplay,
            )

            # 1. ROC Curve
            fig, ax = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="ChurnANN")
            ax.set_title(f"ROC Curve (AUC={result.roc_auc:.4f}) — {result.split}")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"roc_curve_{result.split}.png", dpi=150)
            plt.close(fig)

            # 2. PR Curve
            fig, ax = plt.subplots(figsize=(7, 5))
            PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="ChurnANN")
            ax.set_title(f"Precision-Recall Curve (AUC={result.pr_auc:.4f}) — {result.split}")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"pr_curve_{result.split}.png", dpi=150)
            plt.close(fig)

            # 3. Confusion Matrix
            y_pred = (y_proba >= result.threshold).astype(int)
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, ax=ax,
                display_labels=["Retained", "Churned"],
                cmap="Blues",
            )
            ax.set_title(f"Confusion Matrix (threshold={result.threshold:.2f}) — {result.split}")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"confusion_matrix_{result.split}.png", dpi=150)
            plt.close(fig)

            # 4. Probability Distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(y_proba[y_true == 0], bins=40, alpha=0.6, label="Retained (0)", color="steelblue")
            ax.hist(y_proba[y_true == 1], bins=40, alpha=0.6, label="Churned (1)", color="tomato")
            ax.axvline(result.threshold, color="black", linestyle="--", label=f"Threshold={result.threshold}")
            ax.set_xlabel("Predicted Churn Probability")
            ax.set_ylabel("Count")
            ax.set_title("Predicted Probability Distribution by Class")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / f"probability_dist_{result.split}.png", dpi=150)
            plt.close(fig)

            logger.info("Evaluation plots saved to %s", self.output_dir)

        except Exception as e:
            logger.warning("Plotting failed: %s", e)

    @staticmethod
    def _log_result(result: EvaluationResult) -> None:
        logger.info(
            "[%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f pr_auc=%.4f",
            result.split.upper(),
            result.accuracy, result.precision, result.recall,
            result.f1, result.roc_auc, result.pr_auc,
        )
