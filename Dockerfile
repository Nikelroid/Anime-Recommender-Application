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

RUN echo "========== CHECKING COPIED FILES ==========" && \
    ls -lah artifacts/models/ 2>/dev/null || echo "artifacts/models/ directory NOT FOUND" && \
    if [ -f artifacts/models/best_recommender_model.weights.h5 ]; then \
        echo "✓ Weights file EXISTS (size: $(du -h artifacts/models/best_recommender_model.weights.h5 | cut -f1))"; \
    else \
        echo "✗ Weights file MISSING"; \
    fi && \
    echo "==========================================="

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .

RUN python pipeline/training_pipeline.py 2>&1 | tee training_log.txt && cat training_log.txt

EXPOSE 8000

CMD ["python", "app.py"]