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

export PYTHONUNBUFFERED=1

# Repo root directory setup
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/evaluation/results" || true

# Try to activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate tetradiffusion 2>/dev/null || conda activate base 2>/dev/null || true
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
    source "$HOME/miniconda3/bin/activate" tetradiffusion 2>/dev/null || true
elif [[ -f "$HOME/anaconda3/bin/activate" ]]; then
    source "$HOME/anaconda3/bin/activate" tetradiffusion 2>/dev/null || true
fi

# Defaults matching ablation_fast_sweep.sh (UroCell lysosome dataset)
RUNS_DIR="${REPO_DIR}/runs"
GT_DIR="${REPO_DIR}/data_test/organelles/lyso"
FILTER="abl_"
POINTS=2048
FSCORE_THRESH=0.05

FORCE_FLAG=()
EXTRA_ARGS=()

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs_dir)       RUNS_DIR="$2"; shift 2 ;;
        --gt_dir)         GT_DIR="$2"; shift 2 ;;
        --filter|-f)      FILTER="$2"; shift 2 ;;
        --points)         POINTS="$2"; shift 2 ;;
        --fscore_thresh)  FSCORE_THRESH="$2"; shift 2 ;;
        --force)          FORCE_FLAG=(--force); shift ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
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
echo "Force Re-eval   : ${FORCE_FLAG[*]:-false}"
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
        --fscore_thresh "$FSCORE_THRESH" \
        "${FORCE_FLAG[@]}" \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching Python evaluation using $(which python3)..."
    python3 evaluation/compare.py \
        --runs_dir "$RUNS_DIR" \
        --gt_dir "$GT_DIR" \
        --filter "$FILTER" \
        --points "$POINTS" \
        --fscore_thresh "$FSCORE_THRESH" \
        "${FORCE_FLAG[@]}" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "================================================"
echo "Evaluation completed: $(date)"
echo "Results written to: ${REPO_DIR}/evaluation/results/"
echo "  - Markdown summary : ${REPO_DIR}/evaluation/results/evaluation_summary.md"
echo "  - CSV table        : ${REPO_DIR}/evaluation/results/evaluation_summary.csv"
echo "================================================"
