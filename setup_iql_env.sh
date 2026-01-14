#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG YOU MAY EDIT ----------
PYTHON_BIN="${PYTHON_BIN:-python3.10}"     # change to python3.9 if needed
VENV_DIR="${VENV_DIR:-venv_iql}"
USE_GPU="${USE_GPU:-0}"                   # set to 1 if you have NVIDIA + CUDA working
JAX_VERSION="${JAX_VERSION:-0.4.23}"
NUMPY_VERSION="${NUMPY_VERSION:-1.23.5}"
# ----------------------------------------

echo "==> Using Python: ${PYTHON_BIN}"
${PYTHON_BIN} --version

echo "==> Creating venv: ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"

echo "==> Activating venv"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing NumPy (pinned to avoid Gym/NumPy2 issues)"
python -m pip install "numpy==${NUMPY_VERSION}"

echo "==> Installing JAX"
if [[ "${USE_GPU}" == "1" ]]; then
  echo "    GPU mode requested. This assumes NVIDIA drivers + CUDA 12 are set up and nvidia-smi works."
  python -m pip install "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
  echo "    CPU mode."
  python -m pip install "jax[cpu]==${JAX_VERSION}"
fi

echo "==> Installing core ML + training deps"
python -m pip install \
  scipy \
  flax \
  optax \
  ml-collections \
  absl-py \
  tqdm \
  tensorboardX \
  "tensorflow-probability"

echo "==> Installing RL + sim deps"
python -m pip install \
  "gym==0.26.2" \
  pybullet \
  imageio \
  imageio-ffmpeg \
  gdown

echo "==> Sanity check imports"
python - <<'PY'
import numpy as np
import jax
import flax
import optax
import gym
import pybullet
from tensorflow_probability.substrates import jax as tfp

print("OK ✅")
print("numpy:", np.__version__)
print("jax:", jax.__version__)
print("flax:", flax.__version__)
print("optax:", optax.__version__)
print("gym:", gym.__version__)
print("tfp(substrates.jax):", tfp.__version__)
print("pybullet:", pybullet.__version__)
PY

echo ""
echo "==> Done."
echo "Next:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python train_offline.py --dataset_path ./YOUR_DATASET.npz --max_steps 1000"
