#!/usr/bin/env bash
# submit_inference.sh — SLURM job submission script for TetraDiffusion inference
#
# Usage:
#   sbatch submit_inference.sh --run_name er_20260513_0727 --num_images 8
#
# Container runtime is auto-detected the same way as submit_train.sh.

# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff_infer
#SBATCH --output=/shared/home/eva.bones/TetraDiffusion/logs/slurm_infer_%j_%x.out
#SBATCH --error=/shared/home/eva.bones/TetraDiffusion/logs/slurm_infer_%j_%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=frida
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Move to repo directory (SLURM runs scripts from /var/spool/...)
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

# Parse wrapper-level flags
RUN_NAME=""
NUM_IMAGES=8
DEVICE="cuda"
CUDA_DEVICE=0
OUT_SUBDIR=""
MULTI_GPU=false
FORCE_LOAD_WEIGHTS=true
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)           RUN_NAME="$2"; shift 2 ;;
        --num_images)         NUM_IMAGES="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        --cuda_device)        CUDA_DEVICE="$2"; shift 2 ;;
        --out_subdir)         OUT_SUBDIR="$2"; shift 2 ;;
        --multi_gpu)          MULTI_GPU=true; shift   ;;
        --skip_load_weights)  FORCE_LOAD_WEIGHTS=false; shift ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --run_name is required (the name of the run folder under runs/)"
    exit 1
fi

# Defaults
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data/preprocessed}"
CONTAINER="${CONTAINER:-${REPO_DIR}/pytorch2604_tetradiff.sqfs}"
PYXIS_FLAGS="--container-image=${CONTAINER} \
             --container-mounts=${REPO_DIR}:${REPO_DIR} \
             --container-mount-home \
             --container-workdir=${REPO_DIR}"

# Inference runs should not try to connect to wandb by default on login nodes
export WANDB_MODE=offline
export WANDB_DISABLED=true
export WANDB_SILENT=true
export WANDB_API_KEY=''
export TORCHDYNAMO_DISABLE=1

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/wandb" "${REPO_DIR}/runs" || true

RUN_DIR="${REPO_DIR}/runs/${RUN_NAME}"

# Default out subdir if not provided
if [[ -z "$OUT_SUBDIR" ]]; then
    OUT_SUBDIR="inference_$(date +%Y%m%d_%H%M%S)"
fi

echo "================================================"
echo "Inference run     : $RUN_NAME"
echo "Run folder        : $RUN_DIR"
echo "Node              : $(hostname)"
echo "Device            : $DEVICE (cuda_device=$CUDA_DEVICE)"
echo "Multi-GPU         : $MULTI_GPU"
echo "Force load weights: $FORCE_LOAD_WEIGHTS"
echo "Repo dir          : $REPO_DIR"
echo "Date              : $(date)"
echo "================================================"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run folder does not exist: $RUN_DIR"
    exit 1
fi

# Show run files
ls -la "$RUN_DIR" || true

# Launch
if [ "$MULTI_GPU" = true ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Launching multi-GPU inference on $NUM_GPUS GPUs"
    INFERENCE_CMD="accelerate launch --multi_gpu --num_processes $NUM_GPUS --gpu_ids all inference.py --config_path $RUN_DIR --num_images $NUM_IMAGES --device cuda --cuda_device $CUDA_DEVICE"

    if [ "$FORCE_LOAD_WEIGHTS" = true ]; then
        INFERENCE_CMD="$INFERENCE_CMD --force_load_weights"
    fi

    srun $PYXIS_FLAGS $INFERENCE_CMD "${EXTRA_ARGS[@]}"
else
    echo "Launching single-node inference"
    INFERENCE_CMD="python3 inference.py --config_path $RUN_DIR --num_images $NUM_IMAGES --device $DEVICE --cuda_device $CUDA_DEVICE --out_subdir $OUT_SUBDIR --wandb_offline"

    if [ "$FORCE_LOAD_WEIGHTS" = true ]; then
        INFERENCE_CMD="$INFERENCE_CMD --force_load_weights"
    fi

    srun $PYXIS_FLAGS $INFERENCE_CMD "${EXTRA_ARGS[@]}"
fi

echo "================================================"
echo "Inference finished: $(date)"
echo "================================================"

