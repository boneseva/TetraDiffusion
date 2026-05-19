#!/usr/bin/env bash
# submit_inference.sh — SLURM job submission script for TetraDiffusion inference
#
# Usage:
#   sbatch submit_inference.sh
#   sbatch submit_inference.sh --run_name er_20260513_0727 --generation_mode --num_images 8  (default)
#   sbatch submit_inference.sh --run_name er_20260513_0727 --comparison_mode --num_images 8
#   sbatch submit_inference.sh --list_runs
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
LIST_RUNS=false
INFERENCE_MODE=""
EXTRA_ARGS=()

is_inference_ready_run() {
    local run_dir="$1"
    [[ -d "$run_dir" && -f "$run_dir/config.yaml" && -f "$run_dir/ds.pth" ]] || return 1
    if [[ "$FORCE_LOAD_WEIGHTS" == true ]]; then
        compgen -G "$run_dir/model-*.pt" > /dev/null
        return
    fi
    return 0
}

list_available_runs() {
    local runs_dir="$1"
    local run_dir run_name checkpoint_status
    shopt -s nullglob
    for run_dir in "$runs_dir"/*; do
        [[ -d "$run_dir" ]] || continue
        [[ -f "$run_dir/config.yaml" && -f "$run_dir/ds.pth" ]] || continue
        run_name="$(basename "$run_dir")"
        if compgen -G "$run_dir/model-*.pt" > /dev/null; then
            checkpoint_status="checkpoint"
        else
            checkpoint_status="no-checkpoint"
        fi
        printf '%s\t%s\n' "$(get_run_sort_key "$run_dir")" "${run_name} [${checkpoint_status}]"
    done | sort -rn | cut -f2-
    shopt -u nullglob
}

get_run_sort_key() {
    local run_dir="$1"
    local run_name="$(basename "$run_dir")"
    if [[ "$run_name" =~ ([0-9]{8})_([0-9]{4})([0-9]{2})?$ ]]; then
        printf '2%s%s%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]:-00}"
    else
        printf '1%020d\n' "$(stat -c %Y "$run_dir")"
    fi
}

resolve_latest_run() {
    local runs_dir="$1"
    local run_dir
    shopt -s nullglob
    for run_dir in "$runs_dir"/*; do
        [[ -d "$run_dir" ]] || continue
        if is_inference_ready_run "$run_dir"; then
            printf '%s\t%s\n' "$(get_run_sort_key "$run_dir")" "$(basename "$run_dir")"
        fi
    done | sort -rn | head -n 1 | cut -f2-
    shopt -u nullglob
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)           RUN_NAME="$2"; shift 2 ;;
        --num_images)         NUM_IMAGES="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        --cuda_device)        CUDA_DEVICE="$2"; shift 2 ;;
        --out_subdir)         OUT_SUBDIR="$2"; shift 2 ;;
        --comparison_mode)    [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "comparison" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
                              INFERENCE_MODE="comparison"; shift ;;
        --generation_mode)    [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "generation" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
                              INFERENCE_MODE="generation"; shift ;;
        --stochastic_sampling) [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "generation" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
                              INFERENCE_MODE="generation"; EXTRA_ARGS+=("$1"); shift ;;
        --multi_gpu)          MULTI_GPU=true; shift   ;;
        --skip_load_weights)  FORCE_LOAD_WEIGHTS=false; shift ;;
        --list_runs)          LIST_RUNS=true; shift ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$INFERENCE_MODE" == "" ]]; then
    INFERENCE_MODE="generation"
    EXTRA_ARGS=(--generation_mode "${EXTRA_ARGS[@]}")
elif [[ "$INFERENCE_MODE" == "comparison" ]]; then
    EXTRA_ARGS=(--comparison_mode "${EXTRA_ARGS[@]}")
elif [[ "$INFERENCE_MODE" == "generation" ]]; then
    EXTRA_ARGS=(--generation_mode "${EXTRA_ARGS[@]}")
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

if [[ "$LIST_RUNS" == true ]]; then
    echo "Available inference-ready runs in ${REPO_DIR}/runs:"
    AVAILABLE_RUNS="$(list_available_runs "${REPO_DIR}/runs")"
    if [[ -z "$AVAILABLE_RUNS" ]]; then
        echo "  (none found)"
    else
        while IFS= read -r line; do
            echo "  - $line"
        done <<< "$AVAILABLE_RUNS"
    fi
    exit 0
fi

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="$(resolve_latest_run "${REPO_DIR}/runs")"
    if [[ -z "$RUN_NAME" ]]; then
        echo "ERROR: could not find an inference-ready run in ${REPO_DIR}/runs"
        echo "       Expected at least config.yaml and ds.pth; model-*.pt is also required unless --skip_load_weights is used."
        exit 1
    fi
    echo "Auto-selected latest run: $RUN_NAME"
fi

RUN_DIR="${REPO_DIR}/runs/${RUN_NAME}"

echo "================================================"
echo "Inference run     : $RUN_NAME"
echo "Run folder        : $RUN_DIR"
echo "Node              : $(hostname)"
echo "Device            : $DEVICE (cuda_device=$CUDA_DEVICE)"
echo "Multi-GPU         : $MULTI_GPU"
echo "Inference mode    : $INFERENCE_MODE"
echo "Force load weights: $FORCE_LOAD_WEIGHTS"
echo "Repo dir          : $REPO_DIR"
echo "Date              : $(date)"
echo "================================================"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run folder does not exist: $RUN_DIR"
    exit 1
fi

if ! is_inference_ready_run "$RUN_DIR"; then
    echo "ERROR: run folder is not inference-ready: $RUN_DIR"
    echo "       Required: config.yaml and ds.pth; also model-*.pt unless --skip_load_weights is used."
    exit 1
fi

# Show run files
ls -la "$RUN_DIR" || true

# Launch
if [ "$MULTI_GPU" = true ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Launching multi-GPU inference on $NUM_GPUS GPUs"
    INFERENCE_CMD="accelerate launch --multi_gpu --num_processes $NUM_GPUS --gpu_ids all inference.py --config_path $RUN_DIR --num_images $NUM_IMAGES --device cuda --cuda_device $CUDA_DEVICE"

    if [[ -n "$OUT_SUBDIR" ]]; then
        INFERENCE_CMD="$INFERENCE_CMD --out_subdir $OUT_SUBDIR"
    fi

    INFERENCE_CMD="$INFERENCE_CMD --wandb_offline"

    if [ "$FORCE_LOAD_WEIGHTS" = true ]; then
        INFERENCE_CMD="$INFERENCE_CMD --force_load_weights"
    fi

    srun $PYXIS_FLAGS $INFERENCE_CMD "${EXTRA_ARGS[@]}"
else
    echo "Launching single-node inference"
    INFERENCE_CMD="python3 inference.py --config_path $RUN_DIR --num_images $NUM_IMAGES --device $DEVICE --cuda_device $CUDA_DEVICE --wandb_offline"

    if [[ -n "$OUT_SUBDIR" ]]; then
        INFERENCE_CMD="$INFERENCE_CMD --out_subdir $OUT_SUBDIR"
    fi

    if [ "$FORCE_LOAD_WEIGHTS" = true ]; then
        INFERENCE_CMD="$INFERENCE_CMD --force_load_weights"
    fi

    srun $PYXIS_FLAGS $INFERENCE_CMD "${EXTRA_ARGS[@]}"
fi

echo "================================================"
echo "Inference finished: $(date)"
echo "================================================"

