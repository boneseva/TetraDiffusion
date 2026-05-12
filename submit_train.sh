#!/usr/bin/env bash
# submit_train.sh — SLURM job submission script for TetraDiffusion training
#
# Usage (single GPU):
#   sbatch submit_train.sh --category Golgi --name golgi_run
#
# Usage (multi GPU):
#   sbatch --gres=gpu:4 submit_train.sh --category Golgi --name golgi_run --multi_gpu
#
# All extra arguments after the script name are forwarded to main.py.

# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff
#SBATCH --output=logs/slurm_%j_%x.out      # stdout  (logs/ must exist)
#SBATCH --error=logs/slurm_%j_%x.err       # stderr
#SBATCH --gres=gpu:A100_80GB:1              # request A100 80GB (use L4:1 for quick tests)
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
WANDB_PROJECT=""
MULTI_GPU=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)       CATEGORY="$2";       shift 2 ;;
        --name)           RUN_NAME="$2";        shift 2 ;;
        --data_path)      DATA_PATH="$2";       shift 2 ;;
        --wandb_project)  WANDB_PROJECT="$2";   shift 2 ;;
        --multi_gpu)      MULTI_GPU=true;       shift   ;;
        *)                EXTRA_ARGS+=("$1");   shift   ;;
    esac
done

# ─── Defaults ─────────────────────────────────────────────────────────────────
CATEGORY="${CATEGORY:-Golgi}"
RUN_NAME="${RUN_NAME:-${CATEGORY,,}_run}"          # e.g. golgi_run
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data/preprocessed}"

# ─── Environment setup ────────────────────────────────────────────────────────
# Activate conda env — adjust name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate TetraDiffusion

export WANDB_MODE=online
export TORCHDYNAMO_DISABLE=1

mkdir -p logs

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
        --data_path   "$DATA_PATH" \
        --shapenet_id "$CATEGORY" \
        --grid_res    128 \
        --name        "$RUN_NAME" \
        --batch_size  2 \
        ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching single-GPU training"
    python3 main.py \
        --data_path   "$DATA_PATH" \
        --shapenet_id "$CATEGORY" \
        --grid_res    128 \
        --name        "$RUN_NAME" \
        --batch_size  2 \
        ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
        "${EXTRA_ARGS[@]}"
fi

echo "================================================"
echo "Training finished: $(date)"
echo "================================================"

