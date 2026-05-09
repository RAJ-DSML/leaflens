FROM python:3.11-slim

WORKDIR /app

# Install uv and curl
RUN pip install uv && apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Download model from GCS at build time
RUN mkdir -p models && \
    curl -L "https://storage.googleapis.com/leaflens-models/best_model.pt" \
    -o models/best_model.pt

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]