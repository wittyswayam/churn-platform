# MLOps Reference

## Experiment Tracking (MLflow)

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlflow_artifacts \
              --host 0.0.0.0 --port 5000

# Train with MLflow enabled
python train.py --mlflow --experiment churn-ann-v2
```

Access UI at `http://localhost:5000`

## Retraining Trigger Criteria

| Signal | Threshold | Action |
|---|---|---|
| PSI (any feature) | > 0.25 | Queue retraining |
| Rolling churn rate | > 70% | Investigate + retrain |
| Val AUC degradation | > 3pp drop | Emergency retrain |
| Calendar | Weekly | Scheduled run |

## Model Promotion Criteria

A candidate model is promoted to `champion` if:
1. `test_roc_auc ≥ 0.80`
2. `test_recall ≥ 0.50` (minimum sensitivity for business use)
3. p99 inference latency < 20ms

## CI/CD Pipeline

```
push to develop →  lint → test → train-smoke → docker-build → deploy-staging
push to main    →  lint → test → train-smoke → docker-build → deploy-staging → deploy-production
```

## Monitoring Stack

| Tool | Role | Port |
|---|---|---|
| MonitoringService | In-process metrics | — |
| Prometheus | Metric scraping | 9090 |
| Grafana | Dashboards | 3000 |
| MLflow | Experiment tracking | 5000 |
