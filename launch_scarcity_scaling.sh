#!/usr/bin/env bash
# ==============================================================================
# launch_scarcity_scaling.sh — Experiment 3: Data Scarcity Scaling Launcher
# ==============================================================================
#
# Target Dataset : OpenOrganelle Mitochondria
# Target GPUs    : B200 / B300 (excluding 40GB/80GB nodes aga,apl,ixh,axa,ana)
# Training Steps : 10,000 steps (~20-30 mins per run on B200/B300)
#
# Usage:
#   bash launch_scarcity_scaling.sh                     # Submit all 8 scaling runs on B200
#   bash launch_scarcity_scaling.sh --gpu b300          # Target B300 GPUs
#   bash launch_scarcity_scaling.sh --dry_run           # Preview sbatch commands without submitting
#
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_train.sh"

# Default resource options
GPU_TYPE="1"
GRES_REQ="gpu:1"
EXCLUDE="aga,apl,ixh,axa,ana"
TIME_LIMIT="12:00:00"
NUM_STEPS="15000"
SEED="42"
CATEGORY="Mitochondria"

SINGLE=false
TARGET_FRACTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)      DRY_RUN=true; shift ;;
        --gpu)          GPU_TYPE="$2"; GRES_REQ="gpu:${2}:1"; shift 2 ;;
        --num_steps)    NUM_STEPS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --single)       SINGLE=true; shift ;;
        --fraction)     TARGET_FRACTION="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

echo "============================================================"
echo " Experiment 3 — Data Scarcity Scaling Launcher (Mitochondria)"
echo "============================================================"
echo " GPU Target  : $GRES_REQ"
echo " Steps/Run   : $NUM_STEPS"
echo " Seed        : $SEED"
echo " Exclude     : $EXCLUDE"
echo " Single Run  : $SINGLE"
echo " Mode        : $( [ "$DRY_RUN" = true ] && echo "DRY RUN (preview only)" || echo "SUBMIT TO SLURM" )"
echo "============================================================"
echo ""

submit_scaling_run() {
    local frac="$1"
    local bio_flag="$2"  # "bio_on" or "bio_off"
    local extra_bio_args=()

    if [ "$bio_flag" == "bio_off" ]; then
        extra_bio_args=(--no_bio_loss)
    fi

    # Convert fraction e.g. 0.25 -> 25 (pure bash)
    local frac_percent
    case "$frac" in
        "1.00") frac_percent="100" ;;
        "0.75") frac_percent="75" ;;
        "0.50") frac_percent="50" ;;
        "0.25") frac_percent="25" ;;
        *)      frac_percent="${frac//./}" ;;
    esac
    local name="mito_f${frac_percent}_${bio_flag}"

    local cmd=(
        sbatch
        --time="$TIME_LIMIT"
        --gres="$GRES_REQ"
        --exclude="$EXCLUDE"
        "$SUBMIT"
        --category          "$CATEGORY"
        --name              "$name"
        --num_steps         "$NUM_STEPS"
        --batch_size        16
        --dataset_fraction  "$frac"
        --seed              "$SEED"
        --train_split
        "${extra_bio_args[@]}"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ${cmd[*]}"
    else
        local output
        output=$("${cmd[@]}" 2>&1) || true
        local jid
        jid=$(echo "$output" | grep -oP '(?<=Submitted batch job )\d+' || true)
        if [ -n "$jid" ]; then
            echo "  ✓ OK    Queued $name (frac=$frac, $bio_flag) -> Job ID: $jid"
        else
            echo "  ✗ FAIL  $name -> sbatch error: $output"
        fi
        sleep 1
    fi
}

if [ "$SINGLE" = true ]; then
    FRACTIONS=("1.00")
elif [ -n "$TARGET_FRACTION" ]; then
    FRACTIONS=("$TARGET_FRACTION")
else
    FRACTIONS=("1.00" "0.75" "0.50" "0.25")
fi

for frac in "${FRACTIONS[@]}"; do
    submit_scaling_run "$frac" "bio_on"
    if [ "$SINGLE" = false ]; then
        submit_scaling_run "$frac" "bio_off"
    fi
done

echo ""
echo "Done! Submitted ${#FRACTIONS[@]} paired fractions (8 total runs)."
