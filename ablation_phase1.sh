#!/usr/bin/env bash
# ============================================================
#  ablation_phase1.sh — TetraDiffusion Phase 1 Ablation Batch
# ============================================================
#
# Queues all Phase 1 ablation jobs with a single command:
#
#   bash ablation_phase1.sh            # submit all 12 jobs
#   bash ablation_phase1.sh --dry_run  # print sbatch commands without submitting
#
# ── What gets queued ────────────────────────────────────────
#
#  TIER 1 — Bio ON vs OFF × 4 datasets  (8 jobs)
#  The primary results table (paper §8 / TODO 2c).
#  Covers all four dataset/organelle combinations:
#    • Mitochondria  — regular DB
#    • Lysosome      — regular DB
#    • Mitochondria  — UroCell
#    • Lysosome      — UroCell
#
#  TOTAL: 8 runs.
#  Each run: 1× H100, up to 48 h (SLURM limit set in submit_train.sh).
#  Runs are independent — SLURM will schedule them in parallel up
#  to the number of free GPUs on the partition.
#
# ── Deferred (run separately) ────────────────────────────────
#  • Tier 2: loss decomposition (laplacian / curvature only)
#  • SNR gating sweep     — needs --snr_gate CLI flag (TODO 2b)
#  • Constraint weight λ  — 4 more mito runs        (TODO 2e)
#  • Sigma (σ) sweep      — 5 more mito runs        (TODO 2f)
#  • Sample efficiency    — needs --train_fraction   (TODO 2h)
#
# ── Run dirs & WandB ────────────────────────────────────────
#  All runs land in  runs/<name>/
#  WandB project:    TetraDiffusion
#  Naming convention:
#    <dataset>_<organelle>_bio_<on|off|laplacian|curvature>
#    e.g. reg_mito_bio_on, urocell_lyso_bio_off
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_train.sh"

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry_run" ]] && DRY_RUN=true
done

# ── Helper ──────────────────────────────────────────────────
SUBMITTED=()   # collects job-IDs (or names when dry-running)

submit() {
    # submit [name] [extra args...]
    local name="$1"; shift
    local cmd=(sbatch "$SUBMIT" --name "$name" "$@")

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ${cmd[*]}"
        SUBMITTED+=("$name")
    else
        local jid
        jid=$(${cmd[@]} 2>&1 | grep -oP '(?<=Submitted batch job )\d+' || true)
        if [ -n "$jid" ]; then
            echo "  ✓  Queued  $name  (job $jid)"
            SUBMITTED+=("$name ($jid)")
        else
            echo "  ✗  FAILED  $name"
        fi
    fi
}

# ════════════════════════════════════════════════════════════
#  TIER 1 — Bio ON vs OFF × 4 datasets
# ════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════════════"
echo "  TIER 1 — Bio ON vs OFF  (8 runs)"
echo "══════════════════════════════════════════════════════"

# ── Regular DB — Mitochondria ────────────────────────────────
echo ""
echo "  [1/8]  Regular DB / Mitochondria / Bio ON"
# submit "reg_mito_bio_on" \
#    --category Mitochondria

echo "  [2/8]  Regular DB / Mitochondria / Bio OFF"
# submit "reg_mito_bio_off" \
#    --category Mitochondria \
#    --no_bio_loss

# ── Regular DB — Lysosome ────────────────────────────────────
echo ""
echo "  [3/8]  Regular DB / Lysosome / Bio ON"
# submit "reg_lyso_bio_on" \
#    --category Lysosome

echo "  [4/8]  Regular DB / Lysosome / Bio OFF"
# submit "reg_lyso_bio_off" \
#    --category Lysosome \
#    --no_bio_loss

# ── UroCell — Mitochondria ───────────────────────────────────
echo ""
echo "  [5/8]  UroCell / Mitochondria / Bio ON"
submit "urocell_mito_bio_on" \
    --urocell \
    --category mito

echo "  [6/8]  UroCell / Mitochondria / Bio OFF"
submit "urocell_mito_bio_off" \
    --urocell \
    --category mito \
    --no_bio_loss

# ── UroCell — Lysosome ───────────────────────────────────────
echo ""
echo "  [7/8]  UroCell / Lysosome / Bio ON"
submit "urocell_lyso_bio_on" \
    --urocell \
    --category lyso

echo "  [8/8]  UroCell / Lysosome / Bio OFF"
submit "urocell_lyso_bio_off" \
    --urocell \
    --category lyso \
    --no_bio_loss


# ════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════════════"
echo "  TIER 1 ABLATION BATCH — SUMMARY"
echo "══════════════════════════════════════════════════════"
echo "  ${#SUBMITTED[@]} / 8 jobs queued successfully:"
echo ""
for entry in "${SUBMITTED[@]}"; do
    echo "    • $entry"
done
echo ""
echo "  Monitor with:  squeue -u \$USER"
echo "  Live logs:     tail -f logs/slurm_<jobid>_*.out"
echo "  WandB:         https://wandb.ai/your-team/TetraDiffusion"
echo ""
echo "  Inference (after each run completes):"
echo "    sbatch submit_inference.sh --config_path runs/<name>"
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "  *** DRY RUN — no jobs were actually submitted ***"
fi
echo "══════════════════════════════════════════════════════"
