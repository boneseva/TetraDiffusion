#!/usr/bin/env bash
# submit_train.sh — SLURM job submission script for TetraDiffusion training
#
# Usage (single GPU):
#   sbatch submit_train.sh --category Golgi
#
# Usage (multi GPU):
#   sbatch --gres=gpu:2 submit_train.sh --category Golgi --multi_gpu
#
# Container runtime is auto-detected: Pyxis > Enroot > Singularity > Conda.
# Run name is auto-generated as <category_lowercase>_<YYYYMMDD_HHMM>.
# WandB project is always "TetraDiffusion".

# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff
#SBATCH --output=/shared/home/eva.bones/TetraDiffusion/logs/slurm_%j_%x.out
#SBATCH --error=/shared/home/eva.bones/TetraDiffusion/logs/slurm_%j_%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=frida
# Pyxis container (path resolved at runtime via REPO_DIR — see below)
# NOTE: --container-image cannot use shell variables in #SBATCH lines,
#       so we pass it via srun/the launch command instead (see bottom).
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

# ─── Container / environment setup ───────────────────────────────────────────
CONTAINER="${CONTAINER:-${REPO_DIR}/pytorch2604_tetradiff.sqfs}"

# Pyxis flags: mount repo dir + home dir inside the container
PYXIS_FLAGS="--container-image=${CONTAINER} \
             --container-mounts=${REPO_DIR}:${REPO_DIR} \
             --container-mount-home \
             --container-workdir=${REPO_DIR}"

export WANDB_MODE=online
export WANDB_DIR="${REPO_DIR}"          # wandb will create ${REPO_DIR}/wandb/ here (no nesting)
export TORCHDYNAMO_DISABLE=1

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/wandb" "${REPO_DIR}/runs" || true

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
    srun $PYXIS_FLAGS accelerate launch \
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
    srun $PYXIS_FLAGS python3 main.py \
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

