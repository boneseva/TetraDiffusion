#!/usr/bin/env bash
# submit_batch_probe.sh — submit one probe job per node type on Frida.
#
# Pins each job to a specific node with --nodelist so the result is
# unambiguous. Results land in logs/bs_probe_<NODE>_<jobid>.out
#
# Usage:
#   bash scripts/submit_batch_probe.sh              # probe all nodes below
#   bash scripts/submit_batch_probe.sh --dry_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/pytorch2604_tetradiff.sqfs}"
DRY_RUN=false
[[ "${1:-}" == "--dry_run" ]] && DRY_RUN=true

mkdir -p "${SCRIPT_DIR}/logs"

# ── One representative node per GPU type ──────────────────────────────────────
# Format: "LABEL:NODE"  (one node per probe — result tells you GPU + VRAM)
NODES=(
    "ixh:ixh"           # H100 80GB HBM3
    "ana:ana"           # A100 80GB PCIe
    "aga:aga"           # A100 SXM4 40GB
    "axa:axa"           # A100 SXM4 40GB (same family as aga, quick sanity check)
    "ixb1:ixb1"         # unknown — probe to find out
    "apl:apl"           # unknown — probe to find out
)

submit_probe() {
    local label="$1"
    local node="$2"

    local cmd=(sbatch
        --job-name="bs_probe_${label}"
        --output="${SCRIPT_DIR}/logs/bs_probe_${label}_%j.out"
        --error="${SCRIPT_DIR}/logs/bs_probe_${label}_%j.err"
        --nodelist="${node}"
        --gres=gpu:1
        --cpus-per-task=4
        --mem=32G
        --time=00:30:00
        --partition=frida
        --wrap="
            cd ${SCRIPT_DIR}
            srun \
              --container-image=${CONTAINER} \
              --container-mounts=${SCRIPT_DIR}:${SCRIPT_DIR} \
              --container-mount-home \
              --container-workdir=${SCRIPT_DIR} \
              python3 scripts/probe_batch_size.py \
                --data_path ${SCRIPT_DIR}/data_test/preprocessed \
                --csv_path  ${SCRIPT_DIR}/lib/all_urocell.csv \
                --category  lyso \
                --sizes     4 8 16 32 48 64
        "
    )

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ${cmd[*]}"
        return
    fi

    local out
    out=$("${cmd[@]}" 2>&1)
    local jid
    jid=$(echo "$out" | grep -oP '(?<=Submitted batch job )\d+' || true)
    if [ -n "$jid" ]; then
        echo "  ✓  ${label} (${node})  job ${jid}  → logs/bs_probe_${label}_${jid}.out"
    else
        echo "  ✗  ${label} (${node})  — ${out}"
    fi
}

echo ""
echo "================================================"
echo "  Batch-size probe — one job per node"
echo "================================================"
for entry in "${NODES[@]}"; do
    label="${entry%%:*}"
    node="${entry#*:}"
    submit_probe "$label" "$node"
done
echo ""
echo "  Watch results:  tail -f logs/bs_probe_*.out"
echo "  Or per node:    tail -f logs/bs_probe_ixh_*.out"
echo "================================================"
