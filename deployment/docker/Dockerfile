# ──────────────────────────────────────────────────────────────
# Stage 1: Builder — install Python dependencies
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
    -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean production image
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="churn-platform" \
      description="Customer Churn Prediction Inference Service" \
      version="1.0.0"

# Non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY train.py .

# Copy model artifacts (mounted at runtime in prod; baked in for demo)
COPY artifacts/ ./artifacts/

# Ensure non-root ownership
RUN chown -R app:app /app

USER app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: start inference API
CMD ["uvicorn", "src.inference.inference_service:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
