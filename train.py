"""
train.py — CLI entrypoint for model training.

Usage
-----
python train.py --data data/raw/ChurnPrediction.csv --epochs 100 --batch-size 32
python train.py --preset regularised --mlflow
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train Churn ANN")
    p.add_argument("--data", default="data/raw/ChurnPrediction.csv")
    p.add_argument("--preset", choices=["baseline", "regularised", "wide"], default="baseline")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden-units", nargs="+", type=int, default=None)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    p.add_argument("--experiment", default="churn-ann-baseline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Data Pipeline ──────────────────────────
    from src.data.pipeline import ChurnDataPipeline, PipelineConfig
    pipeline = ChurnDataPipeline(PipelineConfig(
        random_state=args.seed,
        artifacts_dir=f"{args.output_dir}/pipeline",
    ))
    split = pipeline.run(args.data)

    # ── Model ──────────────────────────────────
    from src.models.ann import ChurnANN, ModelConfig, baseline_config, regularised_config, wide_config

    if args.hidden_units:
        config = ModelConfig(
            input_dim=split.X_train.shape[1],
            hidden_units=args.hidden_units,
            dropout_rate=args.dropout,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    elif args.preset == "regularised":
        config = regularised_config(split.X_train.shape[1])
    elif args.preset == "wide":
        config = wide_config(split.X_train.shape[1])
    else:
        config = baseline_config(split.X_train.shape[1])

    config.learning_rate = args.learning_rate
    model = ChurnANN(config).build()

    # ── Trainer ────────────────────────────────
    from src.training.trainer import ChurnModelTrainer, TrainingConfig
    trainer = ChurnModelTrainer(TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
        tensorboard_log_dir=f"{args.output_dir}/tensorboard",
        enable_mlflow=args.mlflow,
        mlflow_experiment=args.experiment,
    ))
    result = trainer.train(
        model,
        split.X_train, split.y_train,
        split.X_val, split.y_val,
        model_config=config.to_dict(),
    )

    # ── Evaluation ─────────────────────────────
    from src.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(output_dir=f"{args.output_dir}/evaluation")
    eval_result = evaluator.evaluate(model, split.X_test, split.y_test, split="test")
    evaluator.print_report(eval_result)
    evaluator.save_report(eval_result)
    evaluator.plot_all(eval_result, split.y_test,
                       model.predict(split.X_test, verbose=0).ravel())

    # ── Registry ───────────────────────────────
    from src.utils.model_registry import ModelRegistry
    registry = ModelRegistry(root=f"{args.output_dir}/model_registry")
    entry = registry.register(
        name="churn-ann",
        model_path=result.model_path,
        metrics={
            "val_auc": result.best_val_auc,
            "val_loss": result.best_val_loss,
            "test_accuracy": eval_result.accuracy,
            "test_f1": eval_result.f1,
            "test_roc_auc": eval_result.roc_auc,
        },
        tags={"preset": args.preset, "epochs": str(args.epochs)},
    )
    logger.info("Registered model: %s v%s (id=%s)", entry.name, entry.version, entry.model_id)
    logger.info("Training complete. Best val AUC: %.4f", result.best_val_auc)


if __name__ == "__main__":
    main()
