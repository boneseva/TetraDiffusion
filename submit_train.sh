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
#SBATCH --gres=gpu:1                        # number of GPUs (override with --gres=gpu:N)
#SBATCH --cpus-per-task=12                  # CPU workers (matches num_workers in config)
#SBATCH --mem=64G                           # RAM
#SBATCH --time=48:00:00                     # wall time  (increase for long runs)
#SBATCH --partition=gpu                     # partition name — change to match your cluster
# #SBATCH --account=my_account             # uncomment + set if your cluster needs it
# #SBATCH --nodelist=node01                # uncomment to pin to a specific node
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Parse wrapper-level flags ────────────────────────────────────────────────
CATEGORY=""
RUN_NAME=""
DATA_PATH=""
MULTI_GPU=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)     CATEGORY="$2";   shift 2 ;;
        --name)         RUN_NAME="$2";   shift 2 ;;
        --data_path)    DATA_PATH="$2";  shift 2 ;;
        --multi_gpu)    MULTI_GPU=true;  shift   ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ─── Defaults ─────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATEGORY="${CATEGORY:-Golgi}"
RUN_NAME="${RUN_NAME:-${CATEGORY,,}_run}"          # e.g. golgi_run
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data/preprocessed}"

# ─── Environment setup ────────────────────────────────────────────────────────
cd "$REPO_DIR"
mkdir -p logs

# Activate conda env — adjust name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate TetraDiffusion

export WANDB_MODE=offline           # change to 'online' if you have internet on compute nodes
export TORCHDYNAMO_DISABLE=1        # avoids nvcc permission errors on cluster nodes

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
        --data_path  "$DATA_PATH" \
        --shapenet_id "$CATEGORY" \
        --grid_res   128 \
        --name       "$RUN_NAME" \
        --batch_size 2 \
        --wandb_project "TetraDiffusion-${CATEGORY}" \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching single-GPU training"
    python3 main.py \
        --data_path  "$DATA_PATH" \
        --shapenet_id "$CATEGORY" \
        --grid_res   128 \
        --name       "$RUN_NAME" \
        --batch_size 2 \
        --wandb_project "TetraDiffusion-${CATEGORY}" \
        "${EXTRA_ARGS[@]}"
fi

echo "================================================"
echo "Training finished: $(date)"
echo "================================================"

