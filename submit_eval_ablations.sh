#!/usr/bin/env bash
# submit_eval_ablations.sh — SLURM script to run evaluation & metric comparisons on ablation runs.
#
# Target dataset: data_test/organelles/lyso (UroCell lysosome dataset from ablation_fast_sweep.sh)
#
# Usage:
#   sbatch submit_eval_ablations.sh
#   sbatch submit_eval_ablations.sh --filter "abl_"                        (default)
#   sbatch submit_eval_ablations.sh --filter "abl_bio_" --points 2048
#   bash submit_eval_ablations.sh                                          (run locally without sbatch)
#

# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff_eval
#SBATCH --output=/shared/home/eva.bones/TetraDiffusion/logs/slurm_eval_%j_%x.out
#SBATCH --error=/shared/home/eva.bones/TetraDiffusion/logs/slurm_eval_%j_%x.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=frida
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Repo root directory setup
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/evaluation/results" || true

# Defaults matching ablation_fast_sweep.sh (UroCell lysosome dataset)
RUNS_DIR="${REPO_DIR}/runs"
GT_DIR="${REPO_DIR}/data_test/organelles/lyso"
FILTER="abl_"
POINTS=2048
FSCORE_THRESH=0.02

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs_dir)       RUNS_DIR="$2"; shift 2 ;;
        --gt_dir)         GT_DIR="$2"; shift 2 ;;
        --filter|-f)      FILTER="$2"; shift 2 ;;
        --points)         POINTS="$2"; shift 2 ;;
        --fscore_thresh)  FSCORE_THRESH="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

echo "================================================"
echo "TetraDiffusion Ablation Evaluation"
echo "================================================"
echo "Runs directory  : ${RUNS_DIR}"
echo "GT directory    : ${GT_DIR}"
echo "Run Filter      : ${FILTER}"
echo "Point cloud size: ${POINTS}"
echo "F-Score threshold: ${FSCORE_THRESH}"
echo "Node            : $(hostname)"
echo "Date            : $(date)"
echo "================================================"

if [[ ! -d "$GT_DIR" ]]; then
    echo "ERROR: Ground truth directory not found at ${GT_DIR}" >&2
    exit 1
fi

CONTAINER="${CONTAINER:-${REPO_DIR}/pytorch2604_tetradiff.sqfs}"

if [[ -f "$CONTAINER" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Launching Pyxis container evaluation..."
    PYXIS_FLAGS="--container-image=${CONTAINER} \
                 --container-mounts=${REPO_DIR}:${REPO_DIR} \
                 --container-mount-home \
                 --container-workdir=${REPO_DIR}"

    srun $PYXIS_FLAGS python3 evaluation/compare.py \
        --runs_dir "$RUNS_DIR" \
        --gt_dir "$GT_DIR" \
        --filter "$FILTER" \
        --points "$POINTS" \
        --fscore_thresh "$FSCORE_THRESH"
else
    echo "Launching Python evaluation..."
    python3 evaluation/compare.py \
        --runs_dir "$RUNS_DIR" \
        --gt_dir "$GT_DIR" \
        --filter "$FILTER" \
        --points "$POINTS" \
        --fscore_thresh "$FSCORE_THRESH"
fi

echo ""
echo "================================================"
echo "Evaluation completed: $(date)"
echo "Results written to: ${REPO_DIR}/evaluation/results/"
echo "  - Markdown summary : ${REPO_DIR}/evaluation/results/evaluation_summary.md"
echo "  - CSV table        : ${REPO_DIR}/evaluation/results/evaluation_summary.csv"
echo "================================================"
