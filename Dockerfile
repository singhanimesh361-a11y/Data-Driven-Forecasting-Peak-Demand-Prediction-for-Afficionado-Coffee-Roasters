# ============================================================================
# Afficionado Demand Intelligence Platform (ADIP)
# Multi-stage Production Dockerfile
# ============================================================================

# ---------------------------------------------------------------------------
# STAGE 1: Builder — compile native extensions, download heavy model artefacts
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="ADIP Team <adip@afficionado.dev>"
LABEL stage="builder"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies required for building C extensions (prophet, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies in an isolated prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Pre-download the Prophet/Stan model so first cold-start is instant.
# prophet compiles a Stan model on first import; we trigger that here.
RUN python -c "\
import sys; sys.path.insert(0, '/install/lib/python3.11/site-packages'); \
from prophet import Prophet; \
m = Prophet(); \
print('Prophet Stan model pre-compiled successfully')"

# ---------------------------------------------------------------------------
# STAGE 2: Runtime — lean image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="ADIP Team <adip@afficionado.dev>"
LABEL org.opencontainers.image.title="ADIP"
LABEL org.opencontainers.image.description="Afficionado Demand Intelligence Platform"
LABEL org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit-specific env vars
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#6F4E37" \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    # Application
    ADIP_ENV=production \
    TZ=UTC

# Runtime system dependencies (libpq for psycopg2, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source code and assets
COPY src/              ./src/
COPY dashboard/        ./dashboard/
COPY configs/          ./configs/
COPY data/processed/       ./data/processed/
COPY data/forecast_store/  ./data/forecast_store/

# Create non-root user for security
RUN groupadd --gid 1000 adip \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home adip \
    && chown -R adip:adip /app

USER adip

EXPOSE 8501

# Healthcheck: verify Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
