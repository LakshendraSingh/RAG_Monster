# ── builder stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
        gunicorn \
        -r requirements.txt

# ── final stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "--bind", "0.0.0.0:8000"]
