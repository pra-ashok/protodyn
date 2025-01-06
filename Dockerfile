# Use NVIDIA CUDA base image
FROM nvcr.io/nvidia/k8s/cuda-sample:nbody

RUN echo /etc/os-release
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopenmpi-dev \
    libfftw3-dev \
    wget \
    python3.12 \
    python3.12-venv \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Create a working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install essential Python dependencies
# RUN pip3 install --no-cache-dir numpy cython

# # Install MDAnalysis and other dependencies
# RUN pip3 install --no-cache-dir -r requirements.txt \
#     torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy the application code (optional)
# COPY . /app/

# Set default command (optional)
# CMD ["python3"]