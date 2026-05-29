# Architecture Reference

## Component Map

```
src/
├── data/pipeline.py          ChurnDataPipeline   → DataSplit
├── models/ann.py             ChurnANN            → keras.Model
├── training/trainer.py       ChurnModelTrainer   → TrainingResult
├── training/hpo.py           HyperparameterSearch→ best_params dict
├── evaluation/evaluator.py   ModelEvaluator      → EvaluationResult
├── inference/service.py      InferenceEngine     → FastAPI app
├── monitoring/service.py     MonitoringService   → metrics / alerts
└── utils/model_registry.py   ModelRegistry       → ModelEntry
```

## Data Flow

```
Raw CSV (14 cols)
  → drop [CustomerId, Surname]
  → map Geography {France:0, Germany:1, Spain:2}
  → map Gender {Female:0, Male:1}
  → features X (10 cols) + target y
  → stratified 70/10/20 split
  → StandardScaler.fit_transform(X_train)
  → StandardScaler.transform(X_val, X_test)
  → DataSplit (numpy arrays + scaler artifact)
```

## Inference Request Flow

```
Client POST /predict
  → Pydantic CustomerFeatures validation
  → InferenceEngine._preprocess()
      encoding maps → scaler.transform()
  → model.predict() [~2ms]
  → sigmoid threshold → binary label
  → PredictionResponse JSON
  → MonitoringService.record_prediction()
```

## Deployment Topology

```
                     ┌─────────────┐
          HTTP ──────► Ingress      │
                     │  (nginx)    │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │Pod 1   │   │Pod 2   │   │Pod 3   │  ← HPA: 2–10
         │API     │   │API     │   │API     │
         └────────┘   └────────┘   └────────┘
              │             │             │
              └─────────────┼─────────────┘
                            │ /metrics
                     ┌──────▼──────┐
                     │  Prometheus │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │   Grafana   │
                     └─────────────┘
```
