#!/usr/bin/env bash
# launch.sh — wrapper that submits submit_train.sh with the correct SLURM job name.
#
# Usage (new run):
#   bash launch.sh --category Golgi
#   bash launch.sh --category ER --multi_gpu
#
# Usage (resume after timeout):
#   bash launch.sh --category Golgi --name golgi_20260512_1423 --resume
#   bash launch.sh --category ER    --name er_20260512_1423    --resume --multi_gpu
#
# --name is REQUIRED when resuming so the existing run folder (runs/<name>/) is targeted.
# The SLURM job will be named  tetradiff_<category>  (e.g. tetradiff_Golgi).
# All arguments are forwarded to submit_train.sh unchanged.

set -euo pipefail

# ── Extract --category value so we can name the job ──────────────────────────
CATEGORY="unknown"
args=("$@")
for (( i=0; i<${#args[@]}; i++ )); do
    if [[ "${args[$i]}" == "--category" && $((i+1)) -lt ${#args[@]} ]]; then
        CATEGORY="${args[$((i+1))]}"
        break
    fi
done

JOB_NAME="tetradiff_${CATEGORY}"

echo "Submitting job: $JOB_NAME"
sbatch --job-name="$JOB_NAME" submit_train.sh "$@"

