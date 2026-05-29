# Deployment Guide

## Prerequisites

- Docker 24+
- kubectl (for Kubernetes)
- Python 3.10+

## Local Development

```bash
python train.py --epochs 100
uvicorn src.inference.inference_service:app --reload --port 8000
```

## Docker Compose (Full Stack)

```bash
cd deployment/docker
docker compose up -d --build

# Verify
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,...}
```

## Kubernetes

```bash
# Build and push image
docker build -f deployment/docker/Dockerfile -t ghcr.io/your-org/churn-platform:latest .
docker push ghcr.io/your-org/churn-platform:latest

# Apply manifests
kubectl apply -f deployment/kubernetes/deployment.yaml

# Watch rollout
kubectl rollout status deployment/churn-api -n churn-platform

# Scale manually
kubectl scale deployment/churn-api --replicas=5 -n churn-platform
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `artifacts/checkpoints/best_model.keras` | Path to Keras model |
| `SCALER_PATH` | `artifacts/pipeline/scaler.pkl` | Path to fitted scaler |
| `THRESHOLD` | `0.5` | Decision threshold |
| `LOG_LEVEL` | `info` | Uvicorn log level |

## Health Check

```
GET /health → {"status":"ok","model_loaded":true,"uptime_seconds":...}
```

## Rolling Restart (zero downtime)

```bash
kubectl rollout restart deployment/churn-api -n churn-platform
kubectl rollout status deployment/churn-api -n churn-platform
```
