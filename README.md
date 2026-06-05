# 🔮 Churn Platform — Enterprise Customer Churn Prediction System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/) [![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/) [![MLflow](https://img.shields.io/badge/MLflow-2.8+-blue.svg)](https://mlflow.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Author:** [wittyswayam](https://github.com/wittyswayam) &mdash; Production-grade ML platform for real-time customer churn prediction using Artificial Neural Networks.

---

## 1. Project Overview & Value Proposition

Customer churn costs the global banking and subscription economy billions of dollars annually. Identifying **which customers are likely to leave before they do** is one of the highest-ROI problems a data team can solve. Yet most churn projects stop at a notebook: a model is trained, metrics are printed, and the work is never operationalised into something others can use reliably.

**Churn Platform** closes that gap completely. Built on a European retail banking dataset with 10 demographic and behavioural features per customer, it delivers a full, production-ready MLOps pipeline that takes raw CSV data all the way to a monitored, containerised REST API. Monitoring, feature-drift detection, model versioning, Optuna-driven hyperparameter optimisation, and Kubernetes-native deployment are first-class concerns, not afterthoughts.

**Target Audience:**
- **ML Engineers** who need a reference architecture for productionising a binary classification model.
- **Data Scientists** who want reproducible, experiment-tracked training without writing boilerplate infrastructure.
- **Platform / DevOps Engineers** looking for a Dockerised, Kubernetes-ready ML service with Prometheus metrics out of the box.
- **Organisations** in retail banking, fintech, telecoms, or SaaS that need a pluggable churn-scoring microservice.

**Why built this way:** Every design decision prioritises reproducibility and operational safety. The `StandardScaler` is fitted exclusively on the training fold to eliminate data leakage. SHA-256 fingerprinting tracks dataset identity across runs. A local model registry enforces a `candidate → staging → champion → retired` promotion workflow. The FastAPI inference service loads the model once at startup via a singleton `InferenceEngine`, not once per request, keeping p99 latency well under 10 ms.

---

## 2. Architecture & Tech Stack

### System Design Overview

```
 Raw CSV
   |
   v
 ChurnDataPipeline  (validate -> clean -> encode -> split -> scale)
   |                                      |
   v                                      v
 DataSplit                     artifacts/ (scaler.pkl, encoding_maps.json)
   |
   v
 ChurnANN.build() -> ChurnModelTrainer -> ModelEvaluator -> ModelRegistry
                           |
                           v
              FastAPI InferenceService
    /predict | /predict/batch | /health | /metrics
                           |
                           v
            MonitoringService  (PSI drift + Prometheus)
```

### Technology Decisions

| Layer | Technology | Rationale |
|---|---|---|
| **Deep Learning** | TensorFlow / Keras 2.13+ | Functional API for composable layers; SavedModel format for portable serialisation across environments |
| **API Framework** | FastAPI + Uvicorn | Async-first, automatic OpenAPI docs, Pydantic v2 validation built in; 3-5x faster than Flask for I/O-bound work |
| **Data Validation** | Pydantic v2 | Field-level validators with descriptive error messages; zero-boilerplate request and response schema enforcement |
| **Experiment Tracking** | MLflow 2.8+ | Per-run parameter and metric logging; optional model registry backend; integrates with any cloud blob store |
| **HPO** | Optuna 3.4+ | TPE sampler converges faster than grid or random search; Median pruner kills unpromising trials early to save compute |
| **Preprocessing** | scikit-learn | `StandardScaler` fit-only-on-train pattern prevents leakage; `compute_class_weight` corrects imbalanced labels |
| **Observability** | Prometheus text format | Pull-based; zero extra runtime dependencies; works with any Prometheus/Grafana stack without a push gateway |
| **Containerisation** | Docker multi-stage build | Builder stage compiles C extensions cleanly; lean `python:3.11-slim` runtime image with a hardened non-root user |
| **Orchestration** | Kubernetes + HPA | 3-replica baseline with CPU/memory autoscaling from 2 to 10 pods; zero-downtime rolling update strategy |
| **Code Quality** | Black + isort + flake8 + mypy | Enforced auto-formatting and static typing configured centrally in `setup.cfg` |

---

## 3. Key Features - Deep Dive

**Leak-Free Data Pipeline (`src/data/pipeline.py`)** - `ChurnDataPipeline` validates the raw CSV against a strict `RAW_SCHEMA` type dictionary, removes exact duplicate rows, and applies ordinal or one-hot encoding to categorical fields (`Geography`, `Gender`). The `StandardScaler` is fitted **exclusively on the training fold** and applied to val and test sets, eliminating the most common cause of inflated offline metrics. Every run computes a SHA-256 fingerprint of the processed DataFrame for full data lineage tracing. Serialised artifacts (`scaler.pkl`, `encoding_maps.json`, `feature_names.json`) ensure inference replays byte-identical preprocessing without any ambiguity.

**Configurable ANN Architecture (`src/models/ann.py`)** - `ChurnANN` accepts a `ModelConfig` dataclass controlling every architectural parameter: hidden layer count, per-layer neuron width, activation function, kernel initialiser strategy, dropout probability, and L2 weight-decay coefficient. Three ready-to-use presets ship out of the box: `baseline_config` reproduces the original tutorial architecture exactly (two 6-unit layers), `regularised_config` adds dropout=0.3 and L2=1e-4 to a deeper three-layer network, and `wide_config` provides a shallow wide network for ablation studies. All Python, NumPy, and TensorFlow RNG seeds are fixed via `set_global_seed` before construction for bit-reproducible results across identical runs.

**Enterprise Training Loop (`src/training/trainer.py`)** - `ChurnModelTrainer` wraps Keras `model.fit` with a production-grade callback stack: `EarlyStopping` (restores best weights, patience=15), `ReduceLROnPlateau` (halves the learning rate after 7 stagnant epochs, floored at 1e-6), `ModelCheckpoint` (saves only when val_loss improves), `CSVLogger` (per-epoch audit trail), and optional `TensorBoard`. Balanced class weights are computed via scikit-learn's `compute_class_weight` to handle the approximately 20% churn minority class. When MLflow is enabled, every hyperparameter, per-epoch metric, and summary stat is logged to the configured experiment for full run comparison in the MLflow UI.

**Automated Hyperparameter Optimisation (`src/training/hyperparameter_search.py`)** - `HyperparameterSearch` drives an Optuna study over a rich search space: network depth (1-4 layers), per-layer width from {16, 32, 64, 128, 256}, dropout [0, 0.5], L2 coefficient [1e-6, 1e-2] on a log-uniform scale, learning rate [1e-5, 1e-2] log-uniform, optimizer (Adam or RMSprop), and batch size from {16, 32, 64, 128}. The **TPE sampler** builds a probabilistic surrogate model of the search space to focus evaluations on high-value regions. The **Median pruner** terminates trials whose intermediate `val_loss` falls below the running median, saving substantial compute time. Results are persisted to `artifacts/hpo/hpo_results.json` and a per-trial `all_trials.csv` for offline analysis.

**Comprehensive Evaluation Suite (`src/evaluation/evaluator.py`)** - `ModelEvaluator` computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, log-loss, and **Expected Calibration Error (ECE)**, a measure of how well the model's raw sigmoid probabilities reflect true outcome frequencies. It also performs threshold optimisation: finding the decision boundary that maximises F1, and separately the tightest threshold that still achieves recall >= 90%, which is critical in churn contexts where missing a churner costs far more than a false alert. ROC curve, PR curve, confusion matrix, and score distribution histogram plots are saved as PNGs to `artifacts/evaluation/`.

**Production Inference Service (`src/inference/inference_service.py`)** - The FastAPI application is created via a `create_app()` factory function, making it independently testable and configurable per deployment environment. The `InferenceEngine` singleton loads the Keras model and pickled scaler exactly once at startup via an `on_event("startup")` hook. Single `/predict` calls complete in approximately 2-5 ms on CPU hardware. The `/predict/batch` endpoint accepts up to 1,000 customer records, runs fully vectorised `model.predict` on the stacked feature matrix, and returns all scores in one round-trip. Pydantic validators reject invalid geography strings, out-of-range credit scores, or missing fields before they ever reach the model computation layer.

**Observability and Drift Detection (`src/monitoring/monitoring_service.py`)** - `MonitoringService` maintains thread-safe rolling windows of 1,000 samples tracking inference latency with p50, p95, and p99 percentile exposure. **Population Stability Index (PSI)** is computed every 500 predictions against training-time reference histograms: PSI below 0.10 indicates stable distributions, 0.10-0.25 warrants investigation, and above 0.25 triggers a warning alert identifying exactly which input features have drifted. The `AlertManager` logs alerts as structured JSONL lines compatible with ELK and Loki ingestion pipelines. A Prometheus-format `/metrics` endpoint requires zero additional dependencies to scrape.

**Model Registry (`src/utils/model_registry.py`)** - `ModelRegistry` manages versioned model artifacts on the local filesystem, backed by a human-readable JSON manifest file. Models move through a formal lifecycle: `candidate -> staging -> champion -> retired`. Promoting a new champion automatically retires the previous one, preventing silent model swaps in production. A `compare()` method generates a side-by-side metric diff table between any two registered versions to support data-driven promotion decisions. The registry design mirrors the MLflow Model Registry API, making a future migration straightforward.

---

## 4. Prerequisites & Installation

**System requirements:** Python 3.11 (TF 2.13-2.15 does not support 3.12+), `pip >= 23`, Docker Desktop >= 24 for containerised deployment, kubectl plus a live cluster for Kubernetes, and a minimum of 4 GB RAM.

```bash
# 1. Clone the repository
git clone https://github.com/wittyswayam/churn-platform.git
cd churn-platform

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# 5. (Optional) Start a local MLflow tracking server
mlflow server --host 127.0.0.1 --port 5000
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `artifacts/checkpoints/best_model.keras` | Path to the trained Keras model loaded by the inference service at startup |
| `SCALER_PATH` | `artifacts/pipeline/scaler.pkl` | Path to the serialised `StandardScaler` produced by the data pipeline |
| `THRESHOLD` | `0.5` | Default decision boundary for binary churn classification |
| `LOG_LEVEL` | `info` | Uvicorn and application log verbosity: `debug`, `info`, `warning`, or `error` |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | URI of the MLflow tracking server for experiment logging |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Suppresses TensorFlow C++ logs (`0`=all, `2`=errors only) |

### Common Setup Errors

| Error | Cause | Fix |
|---|---|---|
| `ImportError: libGL.so.1` | Missing OpenGL shared library on headless Linux | `apt-get install -y libgl1` |
| `scaler not found` warning at startup | Inference service started before training | Run `python train.py` first to generate all `artifacts/` |
| HTTP 503 at `/predict` on Kubernetes | Liveness probe fires before model finishes loading | Increase `initialDelaySeconds` in the Kubernetes liveness probe spec |
| `tensorflow` install OOM or build failure | pip attempts to compile from source | Add `--no-build-isolation` flag or download a pre-built wheel from the TF release page |

---

## 5. Usage Guide & Code Examples

```bash
# Train with the baseline preset (reproduces original two-layer architecture)
python train.py --data data/raw/ChurnPrediction.csv --preset baseline --epochs 100

# Regularised network with MLflow experiment tracking
python train.py --preset regularised --epochs 150 --mlflow --experiment churn-v2

# Fully custom architecture specification
python train.py --hidden-units 64 32 16 --dropout 0.3 --learning-rate 0.0005 --epochs 200
```

```python
# Programmatic hyperparameter search
from src.data.pipeline import ChurnDataPipeline
from src.training.hyperparameter_search import HyperparameterSearch, SearchConfig

split = ChurnDataPipeline().run("data/raw/ChurnPrediction.csv")
search = HyperparameterSearch(SearchConfig(n_trials=50, metric="val_auc"))
best_params = search.run(split.X_train, split.y_train, split.X_val, split.y_val)
best_model, best_config = search.build_best_model(best_params, input_dim=10)
search.plot_optimisation()   # saves PNG charts to artifacts/hpo/
```

```bash
# Start the inference API locally with hot-reload
uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000 --reload

# Docker build and run (mount host artifacts into the container)
docker build -t churn-platform:latest -f deployment/docker/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-platform:latest

# Docker Compose: starts both the API and a Prometheus side-car
docker compose -f deployment/docker/docker-compose.yml up

# Single customer prediction via curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":620,"Geography":"France","Gender":"Female",
       "Age":42,"Tenure":3,"Balance":75000,"NumOfProducts":2,
       "HasCrCard":1,"IsActiveMember":0,"EstimatedSalary":95000}'
# Response: {"churn_probability":0.743812,"churn_prediction":1,"threshold":0.5,"latency_ms":3.14}
```

---

## 6. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns service status, model loaded flag, and uptime in seconds |
| `POST` | `/predict` | Single-record churn score. Optional query param: `threshold` (float, default `0.5`) |
| `POST` | `/predict/batch` | Vectorised batch inference for up to 1,000 customers in a single HTTP request |
| `GET` | `/model/info` | Returns model name, total parameter count, input/output shapes, and full layer list |
| `GET` | `/metrics` | Prometheus-format text exposition of latency, churn rate, and per-feature PSI scores |
| `GET` | `/docs` | Auto-generated interactive Swagger UI (FastAPI OpenAPI integration) |

**`POST /predict/batch` request body:**
```json
{
  "customers": [
    {"CreditScore": 700, "Geography": "Germany", "Gender": "Male",
     "Age": 35, "Tenure": 5, "Balance": 120000, "NumOfProducts": 1,
     "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000}
  ],
  "threshold": 0.5
}
```

**Response:**
```json
{
  "predictions": [{"churn_probability": 0.312, "churn_prediction": 0,
    "threshold": 0.5, "latency_ms": 1.2, "model_version": "best_model"}],
  "batch_size": 1,
  "total_latency_ms": 1.2
}
```

---

## 7. Development & Testing

```bash
# Run the full test suite with a line-level coverage report
pytest tests/ --cov=src --cov-report=term-missing -q

# Run only the unit tests with verbose output
pytest tests/unit/ -v

# Code quality: format, sort imports, lint, and type-check
black src/ tests/ train.py
isort src/ tests/
flake8 src/ tests/
mypy src/

# Start TensorBoard to inspect training curves
tensorboard --logdir artifacts/tensorboard

# Start the MLflow tracking UI
mlflow ui --port 5000
```

**Local development workflow:**
1. Run `python train.py` once to produce all `artifacts/` directories and files.
2. Start the API: `uvicorn src.inference.inference_service:app --reload`.
3. Edit source files; Uvicorn hot-reloads the application automatically on each save.
4. Add or update tests inside `tests/unit/` following the `conftest.py` fixture pattern.
5. Before committing, always run `black`, `flake8`, `mypy`, and `pytest` in sequence to ensure the codebase is clean.

---

## 8. Contributing Guidelines

### Branch Naming

| Prefix | Use case | Example |
|---|---|---|
| `feat/` | New feature or capability | `feat/xgboost-baseline` |
| `fix/` | Bug fix | `fix/scaler-serialisation` |
| `docs/` | Documentation-only change | `docs/api-reference` |
| `refactor/` | Non-breaking code restructuring | `refactor/pipeline-config` |
| `chore/` | Tooling, CI configuration, or dependency bumps | `chore/bump-tensorflow` |

### Commit Message Format (Conventional Commits)

```
<type>(<scope>): <short imperative description under 72 characters>

[Optional body: explain the *why*, not the *what*. Wrap at 72 chars.]

[Optional footer: Closes #123, Breaking-Change: <description>]
```

**Real examples from this project:**
- `feat(inference): add configurable decision threshold per request`
- `fix(pipeline): prevent leakage when geography encoding is one-hot`
- `docs(readme): add Kubernetes deployment walkthrough with HPA config`
- `refactor(training): extract callback builder into private method`

### PR Review Checklist
- [ ] All existing tests pass with zero failures (`pytest -q`)
- [ ] New behaviour is covered by at least one test case
- [ ] Code is auto-formatted with `black` and import-sorted with `isort`
- [ ] `flake8` and `mypy` produce no new warnings or errors
- [ ] All changed public API methods have updated docstrings
- [ ] PR description explains the motivation, approach taken, and any notable trade-offs

To report a bug, open a GitHub Issue including your Python version, operating system, full traceback, steps to reproduce, and the output of `pip show tensorflow`. For security vulnerabilities, please do **not** open a public issue; contact the maintainer directly and privately via the GitHub profile page.

---

## 9. Roadmap

| Status | Feature |
|---|---|
| Planned | **SHAP explainability endpoint** - per-prediction feature attribution scores surfaced via the REST API; a working notebook prototype already exists in `notebooks/explainability.ipynb` |
| Planned | **XGBoost baseline model** - the dependency stub is already present and commented in `requirements.txt`; adds a gradient-boosted tree comparison to the full benchmark suite |
| Planned | **Distributed training with Ray Train** - stub present in requirements; enables multi-GPU and multi-node training for datasets that exceed single-machine memory limits |
| Planned | **Automated retraining trigger** - when the PSI drift score exceeds 0.25 across three consecutive check windows, automatically submit and monitor a new training job |
| Planned | **A/B serving layer** - a weighted Kubernetes `Service` object routing a configurable percentage of live production traffic to a challenger model alongside the reigning champion |
| Known Issue | `search.plot_optimisation()` requires the `kaleido` package for PNG export; install it separately with `pip install kaleido` if chart generation fails |
| Known Issue | TensorFlow 2.15 may rename the AUC metric key from `val_auc` to `val_auc_1` depending on Keras version; the HPO objective handles this defensively by scanning key prefixes |
| Optimisation | Replace the pickle-based scaler serialisation artifact with an ONNX preprocessing graph for fully language-agnostic, runtime-portable preprocessing across polyglot serving stacks |
| Optimisation | Add a Redis-backed prediction cache keyed on the raw input hash to short-circuit redundant model inference calls under sustained high request load |

---

## License

MIT (c) [wittyswayam](https://github.com/wittyswayam)
