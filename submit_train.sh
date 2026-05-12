#!/usr/bin/env bash
# submit_train.sh — SLURM job submission script for TetraDiffusion training
#
# Usage (single GPU):
#   sbatch submit_train.sh --category Golgi
#
# Usage (multi GPU):
#   sbatch --gres=gpu:2 submit_train.sh --category Golgi --multi_gpu
#
# Run name is auto-generated as <category_lowercase>_run.
# WandB project is always "TetraDiffusion".
# All extra arguments after known flags are forwarded to main.py.

# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff
#SBATCH --output=logs/slurm_%j_%x.out      # stdout  (logs/ must exist)
#SBATCH --error=logs/slurm_%j_%x.err       # stderr
#SBATCH --gres=gpu:1                       # number of GPUs (adjust for multi-GPU)
#SBATCH --cpus-per-task=12                  # CPU workers (matches num_workers in config)
#SBATCH --mem=256G                           # RAM
#SBATCH --time=48:00:00                     # wall time  (increase for long runs)
#SBATCH --partition=frida                     # partition name — change to match your cluster
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Move to the repo directory (SLURM runs scripts from /var/spool/...) ──────
# SLURM_SUBMIT_DIR is always the directory where sbatch was called from.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

# ─── Parse wrapper-level flags ────────────────────────────────────────────────
CATEGORY=""
RUN_NAME=""
DATA_PATH=""
MULTI_GPU=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)   CATEGORY="$2";   shift 2 ;;
        --data_path)  DATA_PATH="$2";  shift 2 ;;
        --multi_gpu)  MULTI_GPU=true;  shift   ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ─── Defaults ─────────────────────────────────────────────────────────────────
CATEGORY="${CATEGORY:-Golgi}"
RUN_NAME="${CATEGORY,,}_$(date +%Y%m%d_%H%M)"     # e.g. golgi_20260512_1423
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data/preprocessed}"
WANDB_PROJECT="TetraDiffusion"

# ─── Environment setup ────────────────────────────────────────────────────────
# Activate conda env — adjust name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate TetraDiffusion

# Fix CUDA library path: point to the nvrtc libs bundled with the pip nvidia packages,
# which are in a non-standard location that PyTorch's JIT compiler may not find.
NVRTC_LIB="$(python3 -c 'import os, nvidia.cuda_nvrtc; print(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "lib"))' 2>/dev/null || echo "")"
if [ -n "$NVRTC_LIB" ]; then
    export LD_LIBRARY_PATH="${NVRTC_LIB}:${LD_LIBRARY_PATH:-}"
    echo "NVRTC lib path: $NVRTC_LIB"
fi

export WANDB_MODE=online
export WANDB_DIR="${REPO_DIR}/wandb"          # store all wandb run data under repo/wandb/
export TORCHDYNAMO_DISABLE=1

mkdir -p logs "${WANDB_DIR}"

# ─── Print job info ────────────────────────────────────────────────────────────
echo "================================================"
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : $(hostname)"
echo "GPUs         : $(echo ${CUDA_VISIBLE_DEVICES:-all})"
echo "Category     : $CATEGORY"
echo "Run name     : $RUN_NAME"
echo "Data path    : $DATA_PATH"
echo "Multi-GPU    : $MULTI_GPU"
echo "Repo dir     : $REPO_DIR"
echo "Date         : $(date)"
echo "================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ─── Launch ───────────────────────────────────────────────────────────────────
if [ "$MULTI_GPU" = true ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Launching multi-GPU training on $NUM_GPUS GPUs"
    accelerate launch \
        --multi_gpu \
        --num_processes "$NUM_GPUS" \
        --gpu_ids all \
        main.py \
        --data_path    "$DATA_PATH" \
        --shapenet_id  "$CATEGORY" \
        --grid_res     128 \
        --name         "$RUN_NAME" \
        --batch_size   2 \
        --wandb_project "$WANDB_PROJECT" \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching single-GPU training"
    python3 main.py \
        --data_path    "$DATA_PATH" \
        --shapenet_id  "$CATEGORY" \
        --grid_res     128 \
        --name         "$RUN_NAME" \
        --batch_size   2 \
        --wandb_project "$WANDB_PROJECT" \
        "${EXTRA_ARGS[@]}"
fi

echo "================================================"
echo "Training finished: $(date)"
echo "================================================"

