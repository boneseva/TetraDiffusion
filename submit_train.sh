#!/usr/bin/env bash
# submit_train.sh — SLURM job submission script for TetraDiffusion training
#
# Usage (single GPU):
#   sbatch submit_train.sh --category Golgi
#
# Usage (multi GPU):
#   sbatch --gres=gpu:2 submit_train.sh --category Golgi --multi_gpu
#
# Ablation flags (passed through to main.py via EXTRA_ARGS):
#   --no_bio_loss                        baseline: no biological constraints
#   --bio_loss_weight 0.01               override constraint weight
#   --bio_loss_type laplacian            only smoothness loss (no curvature)
#   --bio_loss_type curvature            only bending-energy loss
#   --bio_loss_type both                 both (default)
#   --lr_schedule warmup_constant        linear warmup → constant  (default)
#   --lr_schedule constant               flat LR, no warmup
#   --lr_schedule warmup_cosine          linear warmup → cosine decay
#   --lr_schedule cosine                 cosine decay from step 0
#
# Example ablation pairs:
#   sbatch submit_train.sh --category Mitochondria --name mito_with_bio
#   sbatch submit_train.sh --category Mitochondria --name mito_no_bio --no_bio_loss
#
#   sbatch submit_train.sh --category Golgi --name golgi_warmup    --lr_schedule warmup_constant
#   sbatch submit_train.sh --category Golgi --name golgi_cosine    --lr_schedule warmup_cosine
#
# Container runtime is auto-detected: Pyxis > Enroot > Singularity > Conda.
# Run name is auto-generated as <category_lowercase>_<YYYYMMDD_HHMM>.
# SLURM job name is dynamically renamed to tetradiff_<Category> at runtime.
# WandB project is always "TetraDiffusion".

# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff
#SBATCH --output=/shared/home/eva.bones/TetraDiffusion/logs/slurm_%j_%x.out
#SBATCH --error=/shared/home/eva.bones/TetraDiffusion/logs/slurm_%j_%x.err
#SBATCH --gres=gpu:B200:1
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
# NOTE: job name is renamed below once CATEGORY is known (scontrol cannot be
#       called before argument parsing, and #SBATCH lines don't expand vars).
CATEGORY=""
CATEGORIES=()        # multi-category list  (--categories Cat1 Cat2 ...)
RUN_NAME=""
NAME_EXPLICIT=false   # true when --name is passed explicitly
DATA_PATH=""
CSV_PATH=""
UROCELL=false
MULTI_GPU=false
RESUME=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)    CATEGORY="$2";  shift 2 ;;
        --categories)  shift
                       # consume all following non-flag words as category names
                       while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                           CATEGORIES+=("$1"); shift
                       done ;;
        --data_path)   DATA_PATH="$2"; shift 2 ;;
        --csv_path)    CSV_PATH="$2";  shift 2 ;;
        # Convenience shortcut: --urocell sets data_path + csv_path for UroCell
        # and prefixes the auto-generated run/wandb name with "urocell_"
        --urocell)     UROCELL=true
                       DATA_PATH="${REPO_DIR}/data_urocell"
                       CSV_PATH="${REPO_DIR}/lib/all_urocell.csv"
                       shift ;;
        --name)        RUN_NAME="$2"; NAME_EXPLICIT=true; shift 2 ;;
        --multi_gpu)   MULTI_GPU=true; shift   ;;
        --resume)      RESUME=true;    shift   ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# If --categories was given, it takes priority over --category.
# If neither was given, fall back to default single category.
if [ ${#CATEGORIES[@]} -gt 0 ]; then
    CATEGORY="${CATEGORIES[0]}"          # used for job-name / run-name generation
elif [ -n "$CATEGORY" ]; then
    CATEGORIES=("$CATEGORY")
else
    CATEGORY="Golgi"
    CATEGORIES=("Golgi")
fi

# ─── Defaults ─────────────────────────────────────────────────────────────────
# CATEGORY / CATEGORIES already set in the parsing block above.
# When resuming, --name MUST be provided so we target the existing run folder.
if [ "$RESUME" = true ] && [ -z "$RUN_NAME" ]; then
    echo "ERROR: --resume requires --name <run_name> so the existing checkpoint folder can be found." >&2
    exit 1
fi
RUN_NAME="${RUN_NAME:-${CATEGORY,,}_$(date +%Y%m%d_%H%M)}"   # e.g. golgi_20260512_1423
# When --urocell was used without an explicit --name, prefix run/wandb name with "urocell_"
[ "$UROCELL" = true ] && [ "$NAME_EXPLICIT" = false ] && RUN_NAME="urocell_${RUN_NAME}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data/preprocessed}"
CSV_PATH="${CSV_PATH:-${REPO_DIR}/lib/all.csv}"
WANDB_PROJECT="TetraDiffusion"

# ─── Rename SLURM job to include the category (not possible in #SBATCH lines) ─
# This makes "squeue" show e.g. "tetradiff_Mitochondria" instead of "tetradiff".
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" JobName="tetradiff_${CATEGORY}" || true
fi

# Build optional resume flag forwarded to main.py
RESUME_FLAG=()
[ "$RESUME" = true ] && RESUME_FLAG=(--resume)

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
echo "Category     : ${CATEGORIES[*]}"
echo "Run name     : $RUN_NAME"
echo "Data path    : $DATA_PATH"
echo "Multi-GPU    : $MULTI_GPU"
echo "Resume       : $RESUME"
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
        --csv_path     "$CSV_PATH" \
        --shapenet_id  "${CATEGORIES[@]}" \
        --grid_res     128 \
        --name         "$RUN_NAME" \
        --batch_size   4 \
        --ga           1 \
        --wandb_project "$WANDB_PROJECT" \
        "${RESUME_FLAG[@]}" \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching single-GPU training"
    srun $PYXIS_FLAGS python3 main.py \
        --data_path    "$DATA_PATH" \
        --csv_path     "$CSV_PATH" \
        --shapenet_id  "${CATEGORIES[@]}" \
        --grid_res     128 \
        --name         "$RUN_NAME" \
        --batch_size   4 \
        --ga           1 \
        --wandb_project "$WANDB_PROJECT" \
        "${RESUME_FLAG[@]}" \
        "${EXTRA_ARGS[@]}"
fi

echo "================================================"
echo "Training finished: $(date)"
echo "================================================"

