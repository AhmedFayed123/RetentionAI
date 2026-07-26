# syntax=docker/dockerfile:1

# ============================================================
# Production Dockerfile for the RetentionAI FastAPI ML API
# ============================================================
#
# Best practices applied:
#   - Python 3.12 slim base image (minimal OS footprint)
#   - Pinned, production-only dependencies (no Jupyter/pytest/debugpy)
#   - Layer caching: requirements installed before project copy
#   - Non-root user for least-privilege runtime
#   - --no-cache-dir to avoid pip cache bloat
#   - HEALTHCHECK for container orchestration
#   - PYTHONDONTWRITEBYTECODE / PYTHONUNBUFFERED for clean logs
#   - .dockerignore to keep build context lean
# ============================================================

FROM python:3.12-slim

# --- Environment variables ---
# PYTHONDONTWRITEBYTECODE: don't write .pyc files (saves space)
# PYTHONUNBUFFERED: force stdout/stderr to be unbuffered (clean logs)
# PIP_NO_CACHE_DIR: disable pip cache (reduces image size)
# PIP_DISABLE_PIP_VERSION_CHECK: suppress pip update warnings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Working directory ---
WORKDIR /app

# --- System dependencies & non-root user ---
# libgomp1: OpenMP runtime required by scikit-learn for parallel operations
# useradd: create a dedicated non-root user for security
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# --- Install Python dependencies ---
# Copied BEFORE the project to leverage Docker layer caching:
# dependencies only need to be reinstalled when requirements change.
COPY --chown=app:app requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# --- Copy the application source code ---
# Includes api/, src/, and artifacts/ (model files needed at inference time).
COPY --chown=app:app . .

# --- Expose the application port ---
EXPOSE 8000

# --- Switch to non-root user ---
USER app

# --- Health check ---
# Probes the /health endpoint; fails if the app is not responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# --- Run the application ---
# uvicorn serves the FastAPI app on all interfaces, port 8000.
# For higher throughput in production, consider adding --workers N
# or using gunicorn with uvicorn workers behind a reverse proxy.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
