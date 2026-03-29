FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy source and only inference-required artifacts
COPY src/ src/
COPY app.py .
COPY artifacts/lgbm_model.joblib        artifacts/lgbm_model.joblib
COPY artifacts/ensemble_weights.joblib  artifacts/ensemble_weights.joblib
COPY artifacts/metrics.joblib           artifacts/metrics.joblib
COPY artifacts/roberta_model/           artifacts/roberta_model/

# HF Spaces runs on port 7860
EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
