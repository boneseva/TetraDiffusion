#!/usr/bin/env bash
# launch_production_runs_urocell.sh — Script to Launch or Resume Production Training Runs for UroCell Dataset
#
# Target: 400,000 steps, batch size 16 on large-memory GPU nodes (excluding 40GB/80GB nodes).
#
# Usage:
#   bash launch_production_runs_urocell.sh                        # Submit all 3 UroCell organelle production runs
#   bash launch_production_runs_urocell.sh --resume               # Resume all 3 UroCell organelle runs from latest checkpoint
#   bash launch_production_runs_urocell.sh --dry_run              # Preview commands without submitting
#   bash launch_production_runs_urocell.sh --category lyso        # Submit only Lysosome run
#   bash launch_production_runs_urocell.sh --resume --category mito
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
        --urocell
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
    local key="$1"
    local alt1="$2"
    local alt2="${3:-}"
    if [[ -z "$TARGET_CATEGORY" ]]; then
        return 0
    fi
    local target="${TARGET_CATEGORY,,}"
    [[ "$target" == "${key,,}" || "$target" == "${alt1,,}" || ( -n "$alt2" && "$target" == "${alt2,,}" ) ]]
}

echo "============================================================"
echo " TetraDiffusion — UroCell Production Training Launcher / Resumer"
echo "============================================================"
echo " Dataset     : UroCell (data_urocell/preprocessed)"
echo " Mode        : $( [ "$RESUME" = true ] && echo "RESUME (continuing from checkpoint)" || echo "NEW RUN" )"
echo " Batch Size  : 16"
echo " Steps/Run   : 400,000"
echo " Exclude     : $EXCLUDE"
echo "============================================================"
echo ""

# 1. Lysosomes (lyso)
if should_run "lyso" "Lysosome"; then
    submit_run "lyso" "urocell_lyso_final_prod" "0.01" "both" "0.05"
fi

# 2. Mitochondria (mito)
if should_run "mito" "Mitochondria"; then
    submit_run "mito" "urocell_mito_final_prod" "0.005" "both" "0.05"
fi

# 3. Fusiform Vesicles (fv)
if should_run "fv" "FV" "Vesicle"; then
    submit_run "fv" "urocell_fv_final_prod" "0.01" "both" "0.05"
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] No jobs were submitted."
else
    echo "  Monitor jobs : squeue -u \$USER"
    echo "  To resume later when time limit expires, run:"
    echo "    bash launch_production_runs_urocell.sh --resume"
fi
echo "============================================================"
