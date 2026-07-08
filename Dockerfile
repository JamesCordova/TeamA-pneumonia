# Base: CUDA 12.6 + cuDNN 9 — compatible with tensorflow==2.21.0
# Requires NVIDIA Container Toolkit on the host (or Docker Desktop + WSL2 on Windows)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.11 + build deps for psycopg2 and compiled wheels.
# cuda-nvcc-12-6 provides libdevice.10.bc required by TensorFlow XLA JIT on GPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        curl \
        build-essential \
        libpq-dev \
        cuda-nvcc-12-6 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy and install deps first (cached layer — rebuild only when requirements.txt changes)
COPY requirements.txt .

# requirements.txt may be UTF-16 LE (Windows default pip freeze encoding).
# iconv converts it to UTF-8; falls back to a plain copy if already UTF-8.
RUN iconv -f utf-16 -t utf-8 requirements.txt -o /tmp/req.txt 2>/dev/null \
    || cp requirements.txt /tmp/req.txt \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/req.txt

# Copy the rest of the project (changes here don't invalidate the pip layer above)
COPY . .

# Pre-create output directories so volumes mount cleanly
RUN mkdir -p \
    data/raw data/processed data/interim data/external \
    reports/figures \
    models \
    logs

# Default entrypoint: run any project script with
#   docker compose run --rm pneumonia scripts/train_sarima.py --department LIMA
ENTRYPOINT ["python"]
