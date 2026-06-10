#!/usr/bin/env bash
# run_interactive.sh — open an interactive GPU shell in the TetraDiffusion container
#
# Usage:
#   bash run_interactive.sh                  # 1 GPU, dev partition, 2 h
#   bash run_interactive.sh --gpus 2         # request 2 H100s
#   bash run_interactive.sh --time 04:00:00  # longer session
#   bash run_interactive.sh --partition frida
#
# Override container path:
#   CONTAINER=/path/to/other.sqfs bash run_interactive.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Defaults ─────────────────────────────────────────────────────────────────
NUM_GPUS=1
PARTITION="dev"
WALL_TIME="02:00:00"
CPUS=8
MEM="64G"
CONTAINER="${CONTAINER:-${REPO_DIR}/pytorch2604_tetradiff.sqfs}"

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      NUM_GPUS="$2";   shift 2 ;;
        --partition) PARTITION="$2";  shift 2 ;;
        --time)      WALL_TIME="$2";  shift 2 ;;
        --cpus)      CPUS="$2";       shift 2 ;;
        --mem)       MEM="$2";        shift 2 ;;
        *)
            echo "Usage: bash run_interactive.sh [--gpus N] [--partition NAME] [--time HH:MM:SS] [--cpus N] [--mem Xg]" >&2
            exit 1 ;;
    esac
done

echo "Requesting: ${NUM_GPUS}x H100 | ${CPUS} CPUs | ${MEM} | ${WALL_TIME} | partition=${PARTITION}"
echo "Container : ${CONTAINER}"
echo "Repo      : ${REPO_DIR}"
echo "→ Waiting for allocation…"

exec srun \
    --pty \
    --partition="${PARTITION}" \
    --gres="gpu:H100:${NUM_GPUS}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${WALL_TIME}" \
    --job-name="tetradiff_shell" \
    --container-image="${CONTAINER}" \
    --container-mounts="${REPO_DIR}:${REPO_DIR}" \
    --container-mount-home \
    --container-workdir="${REPO_DIR}" \
    bash
