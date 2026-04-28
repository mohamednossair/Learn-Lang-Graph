"""Task 15.4 — Docker Deployment (save as Dockerfile)."""
DOCKERFILE = """
# Dockerfile for LangGraph Agent API
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with gunicorn + uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "task_15_1_api:app"]
"""

DOCKER_COMPOSE = """
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_MODEL=llama3.2
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
"""

if __name__ == "__main__":
    with open("Dockerfile", "w") as f:
        f.write(DOCKERFILE)
    with open("docker-compose.yml", "w") as f:
        f.write(DOCKER_COMPOSE)
    print("Created Dockerfile and docker-compose.yml")
    print("Build: docker build -t langgraph-api .")
    print("Run:   docker-compose up -d")
