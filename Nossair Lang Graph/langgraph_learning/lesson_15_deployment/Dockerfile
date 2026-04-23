# =============================================================
# Dockerfile — Lesson 15: LangGraph FastAPI Deployment
# =============================================================
# Build:  docker build -t langgraph-api .
# Run:    docker run -p 8000:8000 --env-file .env langgraph-api
# =============================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for SQLite checkpoint database
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn — 4 workers for production
CMD ["uvicorn", "lesson_15_deployment.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30"]
