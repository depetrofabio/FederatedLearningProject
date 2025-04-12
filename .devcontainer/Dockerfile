FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# System packages for ML and Jupyter
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python ML + Jupyter stack
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    notebook \
    ipykernel \
    tensorflow \
    torch \
    torchvision \
    torchaudio

# Default working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888
    