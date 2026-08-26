# ============================================================
# Prism AI — Production Distributed Training & Inference Dockerfile
# ============================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}"

# Install core system packages & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace/prism-ai

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies
COPY requirements.txt setup.py pyproject.toml ./
RUN pip install -e .

# Install FlashAttention-2 with CUDA compilation
RUN pip install flash-attn --no-build-isolation

# Copy project source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "prism.inference.serving:app", "--host", "0.0.0.0", "--port", "8000"]
