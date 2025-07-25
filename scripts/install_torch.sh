#!/bin/bash
set -e

# Detect CUDA version (fallback to CPU)
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d"." -f1,2 2>/dev/null || echo "cpu")

echo "Detected environment CUDA driver version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "12.1" ]]; then
  echo "Installing PyTorch and PyG for CUDA 12.1..."
  pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

elif [[ "$CUDA_VERSION" == "11.8" ]]; then
  echo "Installing PyTorch and PyG for CUDA 11.8..."
  pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

else
  echo "Unknown or unsupported CUDA version ($CUDA_VERSION) — falling back to CPU install..."
  pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
fi

# Always install torch-geometric last
pip install torch-geometric
