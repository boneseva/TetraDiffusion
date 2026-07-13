#!/usr/bin/env bash
# ============================================================
#  ablation_fast_sweep.sh -- Fast Ablation Parameter Sweep
#  Target dataset : data_test  (small UroCell lysosome, 26 shapes)
#  Training cap   : 50 000 steps per run  (~1-2 h on H100)
#  Batch size      : 8  (fits comfortably with color=True, grid_pruning)
# ============================================================
#
# Runs 8 tiers, each sweeping ONE axis while holding all others
# at the current best-known baseline.  Runs within a tier are
# independent -- SLURM schedules them in parallel.
#
# Usage:
#   bash ablation_fast_sweep.sh              # submit all jobs
#   bash ablation_fast_sweep.sh --dry_run    # print commands only
#   bash ablation_fast_sweep.sh --tier 3     # submit only one tier
#   bash ablation_fast_sweep.sh --dry_run --tier 4
#
# -- Tier overview -------------------------------------------
#   T1  Batch size            3 runs   4 / 8 / 16
#   T2  Offset noise          4 runs   0.0 / 0.05 / 0.1 / 0.2
#   T3  Bio loss ON / OFF     2 runs   baseline vs no-bio
#   T4  Bio loss weight lam   4 runs   0.001 / 0.005 / 0.01 / 0.05
#   T5  Bio loss type         3 runs   laplacian / curvature / both
#   T6  SNR gate mode         3 runs   soft / hard_0.3 / none
#   T7  SDF background loss   3 runs   0 / 0.05 / 0.1
#   T8  Inference steps       (post-training, no new SLURM jobs)
#
#   TOTAL: 22 training runs
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_train.sh"

# -- Fixed paths for data_test (small UroCell lysosome) ------
# MeshLoader globs: data_test/preprocessed/lyso/*/mesh_data/sample.pth
# Model IDs won't be in the CSV -- all 26 shapes are loaded automatically.
DATA_PATH="${SCRIPT_DIR}/data_test/preprocessed"
CSV_PATH="${SCRIPT_DIR}/lib/all_urocell.csv"
CATEGORY="lyso"

# -- Default SLURM resources for fast sweeps -----------------
# 3 hours is plenty for 50k steps on data_test. Asking for a short time
# allows SLURM to backfill the jobs almost immediately.
TIME_LIMIT="03:00:00"
# Requesting generic 'gpu:1' allows the jobs to run on ANY available GPU
# (e.g. A100, RTX3090) instead of waiting solely for H100s.
GRES_REQ="gpu:1"

# -- Argument parsing ----------------------------------------
DRY_RUN=false
TARGET_TIER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run) DRY_RUN=true; shift ;;
        --tier)    TARGET_TIER="$2"; shift 2 ;;
        --time)    TIME_LIMIT="$2"; shift 2 ;;
        --gres)    GRES_REQ="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# -- Submission helper ----------------------------------------
SUBMITTED=()

submit() {
    local name="$1"; shift
    # Prepend SLURM options (time, gres) before the script name to override #SBATCH defaults.
    local cmd=(sbatch --time="$TIME_LIMIT" --gres="$GRES_REQ" "$SUBMIT" --name "$name" \
        --data_path    "$DATA_PATH" \
        --csv_path     "$CSV_PATH" \
        --category     "$CATEGORY" \
        --num_steps    50000 \
        --batch_size   4 \
        --wandb_project "TetraDiffusion_ablation" \
        "$@")

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ${cmd[*]}"
        SUBMITTED+=("$name")
    else
        local jid
        jid=$(${cmd[@]} 2>&1 | grep -oP '(?<=Submitted batch job )\d+' || true)
        if [ -n "$jid" ]; then
            echo "  OK  Queued  $name  (job $jid)"
            SUBMITTED+=("$name ($jid)")
        else
            echo "  FAIL  $name"
        fi
    fi
}

run_tier() { [[ -z "$TARGET_TIER" || "$TARGET_TIER" == "$1" ]]; }

# =====================================================================
#  TIER 1 -- Batch size
#  Sweep: 2 / 4(baseline) / 8
#  Note: If batch size 8 causes OOM, use gradient accumulation instead.
# =====================================================================
if run_tier 1; then
    echo ""
    echo "==== TIER 1 -- Batch size ===="
    submit "abl_bs2"  --batch_size 2
    submit "abl_bs4"  --batch_size 4
    submit "abl_bs8"  --batch_size 8
fi

# =====================================================================
#  TIER 2 -- Offset noise
#  Sweep: 0.0 / 0.05 / 0.1(default) / 0.2
#  Offset noise prevents mean-collapse on small datasets.
#  Too much shifts the data distribution away from zero.
# =====================================================================
if run_tier 2; then
    echo ""
    echo "==== TIER 2 -- Offset noise ===="
    submit "abl_onoise0"   --offset_noise 0.0
    submit "abl_onoise005" --offset_noise 0.05
    submit "abl_onoise01"  --offset_noise 0.1
    submit "abl_onoise02"  --offset_noise 0.2
fi

# =====================================================================
#  TIER 3 -- Bio loss ON vs OFF  (primary signal check)
#  Most important 2-run comparison; run this tier first to confirm
#  the bio prior produces a measurable difference on this dataset.
# =====================================================================
if run_tier 3; then
    echo ""
    echo "==== TIER 3 -- Bio loss ON vs OFF ===="
    submit "abl_bio_on"              # bio_loss_weight=0.005, type=both, snr_gate=soft
    submit "abl_bio_off" --no_bio_loss
fi

# =====================================================================
#  TIER 4 -- Bio loss weight lambda
#  Sweep: 0.001 / 0.005(default) / 0.01 / 0.05
#  On binary-mask organelle data the boundary term may need a
#  higher weight to dominate over the diffusion reconstruction loss.
# =====================================================================
if run_tier 4; then
    echo ""
    echo "==== TIER 4 -- Bio loss weight ===="
    submit "abl_bw1e3"  --bio_loss_weight 0.001
    submit "abl_bw5e3"  --bio_loss_weight 0.005
    submit "abl_bw1e2"  --bio_loss_weight 0.01
    submit "abl_bw5e2"  --bio_loss_weight 0.05
fi

# =====================================================================
#  TIER 5 -- Bio loss type decomposition
#  Does smoothness (Laplacian) or bending energy (curvature) matter more?
#  Lysosomes are near-spherical -> curvature term should be informative.
# =====================================================================
if run_tier 5; then
    echo ""
    echo "==== TIER 5 -- Bio loss type ===="
    submit "abl_bt_lap"  --bio_loss_type laplacian
    submit "abl_bt_curv" --bio_loss_type curvature
    submit "abl_bt_both" --bio_loss_type both
fi

# =====================================================================
#  TIER 6 -- SNR gate mode
#  Controls at which noise levels the bio loss is active.
#  soft     = continuous SNR/(SNR+1) weighting (default)
#  hard_0.3 = step-function: active at t<0.3 (low noise), off otherwise
#  none     = apply bio loss uniformly at ALL noise levels
# =====================================================================
if run_tier 6; then
    echo ""
    echo "==== TIER 6 -- SNR gate mode ===="
    submit "abl_snr_soft"  --snr_gate soft
    submit "abl_snr_h03"   --snr_gate hard_0.3
    submit "abl_snr_none"  --snr_gate none
fi

# =====================================================================
#  TIER 7 -- SDF background loss
#  Addresses multi-component generation artefacts.
#  See docs/sdf_background_loss.md for full rationale.
#  Expected winner: 0.05 -- enough to suppress background floaters
#  without distorting the primary organelle shape.
# =====================================================================
if run_tier 7; then
    echo ""
    echo "==== TIER 7 -- SDF background loss ===="
    submit "abl_bg0"    --sdf_bg_loss_weight 0.0
    submit "abl_bg005"  --sdf_bg_loss_weight 0.05
    submit "abl_bg01"   --sdf_bg_loss_weight 0.1
fi

# =====================================================================
#  TIER 8 -- Inference sampling steps (post-training, no new SLURM jobs)
#  Train one good model (abl_bio_on), then evaluate it at different
#  denoising step budgets.  --sampling_steps is now a CLI flag so
#  inference.py can override the list stored in the run config.yaml.
# =====================================================================
if run_tier 8; then
    echo ""
    echo "==== TIER 8 -- Inference steps (post-training) ===="
    echo "  After abl_bio_on finishes, run:"
    echo "    python inference.py --config_path runs/abl_bio_on --sampling_steps 32"
    echo "    python inference.py --config_path runs/abl_bio_on --sampling_steps 50"
    echo "    python inference.py --config_path runs/abl_bio_on --sampling_steps 100"
    echo "  (No new SLURM training jobs needed -- reuses the same checkpoint.)"
fi

# =====================================================================
#  Summary
# =====================================================================
echo ""
echo "============================================================"
echo "  FAST ABLATION SWEEP -- SUMMARY"
echo "============================================================"
echo "  Dataset    : data_test/preprocessed/lyso  (26 shapes)"
echo "  Steps/run  : 50 000"
echo "  Batch size : 4  (baseline)"
echo "  WandB      : TetraDiffusion_ablation"
echo ""
echo "  ${#SUBMITTED[@]} job(s) queued:"
for entry in "${SUBMITTED[@]}"; do
    echo "    * $entry"
done
echo ""
echo "  Monitor   : squeue -u $USER"
echo "  Live logs : tail -f logs/slurm_<jobid>_*.out"
echo ""
echo "  Quick single-tier test:"
echo "    bash ablation_fast_sweep.sh --dry_run --tier 3"
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "  *** DRY RUN -- no jobs were actually submitted ***"
fi
echo "============================================================"
