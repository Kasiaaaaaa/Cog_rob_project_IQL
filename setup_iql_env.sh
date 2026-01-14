#!/usr/bin/env bash
set -e

ENV_NAME="iql_peg_in_hole"
PYTHON_VERSION="3.10"

echo "==> Creating conda environment: ${ENV_NAME}"
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}

echo "==> Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "==> Updating pip"
pip install --upgrade pip setuptools wheel

echo "==> Installing NumPy (pinned to avoid Gym / NumPy 2.0 issues)"
pip install numpy==1.23.5

echo "==> Installing JAX (CPU version)"
pip install "jax[cpu]==0.4.23"

echo "==> Installing core ML stack"
pip install \
  scipy \
  flax \
  optax \
  ml-collections \
  absl-py \
  tqdm \
  tensorboardX \
  tensorflow-probability

echo "==> Installing RL + simulation dependencies"
pip install \
  gym==0.26.2 \
  pybullet \
  imageio \
  imageio-ffmpeg \
  gdown

echo "==> Sanity check"
python - <<'PY'
import numpy as np
import jax
import flax
import gym
import pybullet
from tensorflow_probability.substrates import jax as tfp

print("OK ✅ Environment ready")
print("numpy:", np.__version__)
print("jax:", jax.__version__)
print("flax:", flax.__version__)
print("gym:", gym.__version__)
print("tfp:", tfp.__version__)
print("pybullet:", pybullet.__version__)
PY

echo ""
echo "==> Done."
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  python train_offline.py --dataset_path FULL_DATASET.npz"
