FROM python:3.8-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .

# Debug: Check if weights exist before training
RUN echo "========== DEBUG INFO ==========" && \
    echo "Current directory:" && pwd && \
    echo "Contents of artifacts/models/:" && \
    ls -lah artifacts/models/ && \
    echo "Checking if file exists:" && \
    test -f artifacts/models/best_recommender_model.weights.h5 && echo "✓ File EXISTS" || echo "✗ File MISSING" && \
    echo "================================"

RUN python pipeline/training_pipeline.py

EXPOSE 8000

CMD ["python", "app.py"]