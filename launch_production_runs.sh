#!/usr/bin/env bash
# launch_production_runs.sh — Script to Launch or Resume Production Training Runs
#
# Target: 400,000 steps, batch size 16 on large-memory GPU nodes (excluding 40GB/80GB nodes).
#
# Usage:
#   bash launch_production_runs.sh                        # Submit all 4 organelle production runs
#   bash launch_production_runs.sh --resume               # Resume all 4 organelle runs from latest checkpoint
#   bash launch_production_runs.sh --dry_run              # Preview commands without submitting
#   bash launch_production_runs.sh --category Lysosome     # Submit only Lysosome run
#   bash launch_production_runs.sh --resume --category Lysosome
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_train.sh"

# SLURM resource options
GRES_REQ="gpu:1"
EXCLUDE="aga,apl,ixh,axa,ana"
TIME_LIMIT="24:00:00"

DRY_RUN=false
RESUME=false
TARGET_CATEGORY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)      DRY_RUN=true; shift ;;
        --resume)       RESUME=true; shift ;;
        --category)     TARGET_CATEGORY="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Build resume flag
RESUME_FLAG=()
[ "$RESUME" = true ] && RESUME_FLAG=(--resume)

submit_run() {
    local cat="$1"
    local name="$2"
    local bio_w="$3"
    local bio_t="$4"
    local bg_w="$5"

    local cmd=(
        sbatch
        --time="$TIME_LIMIT"
        --gres="$GRES_REQ"
        --exclude="$EXCLUDE"
        "$SUBMIT"
        --category          "$cat"
        --name              "$name"
        --num_steps         200000
        --batch_size        16
        --test_every        2000
        --offset_noise      0.1
        --bio_loss_weight   "$bio_w"
        --bio_loss_type     "$bio_t"
        --sdf_bg_loss_weight "$bg_w"
        "${RESUME_FLAG[@]}"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ${cmd[*]}"
    else
        local output
        output=$("${cmd[@]}" 2>&1) || true
        local jid
        jid=$(echo "$output" | grep -oP '(?<=Submitted batch job )\d+' || true)
        if [ -n "$jid" ]; then
            echo "  ✓ OK    Queued $cat ($name) -> Job ID: $jid"
        else
            echo "  ✗ FAIL  $cat ($name) -> sbatch error: $output"
        fi
        sleep 1
    fi
}

should_run() {
    local cat="$1"
    [[ -z "$TARGET_CATEGORY" || "${TARGET_CATEGORY,,}" == "${cat,,}" ]]
}

echo "============================================================"
echo " TetraDiffusion — Production Training Launcher / Resumer"
echo "============================================================"
echo " Mode        : $( [ "$RESUME" = true ] && echo "RESUME (continuing from checkpoint)" || echo "NEW RUN" )"
echo " Batch Size  : 16"
echo " Steps/Run   : 400,000"
echo " Exclude     : $EXCLUDE"
echo "============================================================"
echo ""

# 1. Lysosomes
if should_run "Lysosome"; then
    submit_run "Lysosome" "lyso_final_prod" "0.01" "both" "0.05"
fi

# 2. Mitochondria
if should_run "Mitochondria"; then
    submit_run "Mitochondria" "mito_final_prod" "0.005" "both" "0.05"
fi

# 3. Golgi
if should_run "Golgi"; then
    submit_run "Golgi" "golgi_final_prod" "0.005" "laplacian" "0.05"
fi

# 4. Endoplasmic Reticulum
if should_run "ER"; then
    submit_run "ER" "er_final_prod" "0.003" "laplacian" "0.02"
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] No jobs were submitted."
else
    echo "  Monitor jobs : squeue -u \$USER"
    echo "  To resume later when time limit expires, run:"
    echo "    bash launch_production_runs.sh --resume"
fi
echo "============================================================"
