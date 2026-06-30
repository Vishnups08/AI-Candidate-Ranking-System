FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Warm up and cache the embedding models inside the container image
RUN python precompute/download_models.py

EXPOSE 7860
ENV PORT=7860

# Run Flask backend binding to HF Spaces default port 7860
CMD ["python", "demo_server.py", "--candidates", "precompute/sample_candidates.json", "--port", "7860", "--host", "0.0.0.0", "--preload-model"]
