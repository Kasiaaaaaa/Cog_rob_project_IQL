#!/bin/bash
set -e

# ---------------- CONFIG ----------------
MAX_STEPS=250000
BATCH_SIZE=512
EVAL_INTERVAL=10000
LOG_INTERVAL=1000
DATA_ROOT="./datasets"          # where NPZ files live
SAVE_ROOT="./results"       # where logs/checkpoints go
# ---------------------------------------

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

for SHAPE in circle square triangle hexagon
do
    DATASET="${DATA_ROOT}/sac_fb_${SHAPE}.npz"
    SAVE_DIR="${SAVE_ROOT}/${SHAPE}_${REWARD}"

    echo "=============================================="
    echo "IQL training | shape=${SHAPE} | reward=${REWARD}"
    echo "Dataset: ${DATASET}"
    echo "Save dir: ${SAVE_DIR}"
    echo "=============================================="

    mkdir -p "${SAVE_DIR}"

    python train_offline.py \
      --dataset_path "${DATASET}" \
      --shape "${SHAPE}" \
      --reward "old" \
      --max_steps ${MAX_STEPS} \
      --batch_size ${BATCH_SIZE} \
      --eval_interval ${EVAL_INTERVAL} \
      --log_interval ${LOG_INTERVAL} \
      --save_dir "${SAVE_DIR}"

done
