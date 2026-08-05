#!/usr/bin/env bash
# launch_inference_filter.sh — Launch SLURM inference jobs on runs matching a filter (e.g., '_prod_')
#
# Usage:
#   bash launch_inference_filter.sh                             # Run inference on all ready runs with '_prod_' in name
#   bash launch_inference_filter.sh --filter _prod_            # Explicit filter
#   bash launch_inference_filter.sh --filter er                # Run on all ready runs matching 'er'
#   bash launch_inference_filter.sh --filter _prod_ --num_images 16  # Custom sample count per run
#   bash launch_inference_filter.sh --filter _prod_ --dry_run   # Print matching runs without submitting
#   bash launch_inference_filter.sh --list_runs                # List all ready runs matching filter
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"

FILTER="_prod_"
NUM_IMAGES=8
FORCE_LOAD_WEIGHTS=true
LIST_RUNS=false
DRY_RUN=false
INFERENCE_MODE=""
PASSTHROUGH_ARGS=()

is_inference_ready_run() {
    local run_dir="$1"
    [[ -d "$run_dir" && -f "$run_dir/config.yaml" && -f "$run_dir/ds.pth" ]] || return 1
    if [[ "$FORCE_LOAD_WEIGHTS" == true ]]; then
        compgen -G "$run_dir/model-*.pt" > /dev/null 2>&1
        return $?
    fi
    return 0
}

get_run_sort_key() {
    local run_dir="$1"
    local run_name="$(basename "$run_dir")"
    if [[ "$run_name" =~ ([0-9]{8})_([0-9]{4})([0-9]{2})?$ ]]; then
        printf '2%s%s%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]:-00}"
    else
        printf '1%020d' "$(stat -c %Y "$run_dir" 2>/dev/null || echo 0)"
    fi
}

collect_matching_runs() {
    local run_dir run_name
    shopt -s nullglob
    for run_dir in "$RUNS_DIR"/*; do
        [[ -d "$run_dir" ]] || continue
        run_name="$(basename "$run_dir")"
        if [[ -n "$FILTER" && "$run_name" != *"$FILTER"* ]]; then
            continue
        fi
        is_inference_ready_run "$run_dir" || continue
        printf '%s\t%s\n' "$(get_run_sort_key "$run_dir")" "$run_name"
    done | sort -rn | cut -f2-
    shopt -u nullglob
}

list_available_matching_runs() {
    local run_dir run_name checkpoint_status
    shopt -s nullglob
    for run_dir in "$RUNS_DIR"/*; do
        [[ -d "$run_dir" ]] || continue
        run_name="$(basename "$run_dir")"
        if [[ -n "$FILTER" && "$run_name" != *"$FILTER"* ]]; then
            continue
        fi
        [[ -f "$run_dir/config.yaml" && -f "$run_dir/ds.pth" ]] || continue
        if compgen -G "$run_dir/model-*.pt" > /dev/null 2>&1; then
            checkpoint_status="checkpoint ✓"
        else
            checkpoint_status="no-checkpoint"
        fi
        printf '%s\t%s\n' "$(get_run_sort_key "$run_dir")" "${run_name}  [${checkpoint_status}]"
    done | sort -rn | cut -f2-
    shopt -u nullglob
}

# ── argument parsing ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)             FILTER="$2";               shift 2 ;;
        --num_images)         NUM_IMAGES="$2";           PASSTHROUGH_ARGS+=("--num_images" "$2"); shift 2 ;;
        --dry_run)            DRY_RUN=true;              shift   ;;
        --list_runs)          LIST_RUNS=true;             shift   ;;
        --comparison_mode)    [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "comparison" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
                              INFERENCE_MODE="comparison"; shift   ;;
        --generation_mode)    [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "generation" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
                              INFERENCE_MODE="generation"; shift   ;;
        --skip_load_weights)  FORCE_LOAD_WEIGHTS=false; PASSTHROUGH_ARGS+=("$1"); shift   ;;
        *)                    PASSTHROUGH_ARGS+=("$1");   shift   ;;
    esac
done

if [[ "$INFERENCE_MODE" == "" ]]; then
    INFERENCE_MODE="generation"
    PASSTHROUGH_ARGS=(--generation_mode "${PASSTHROUGH_ARGS[@]}")
elif [[ "$INFERENCE_MODE" == "comparison" ]]; then
    PASSTHROUGH_ARGS=(--comparison_mode "${PASSTHROUGH_ARGS[@]}")
elif [[ "$INFERENCE_MODE" == "generation" ]]; then
    PASSTHROUGH_ARGS=(--generation_mode "${PASSTHROUGH_ARGS[@]}")
fi

# ── list mode ─────────────────────────────────────────────────────────────────

if [[ "$LIST_RUNS" == true ]]; then
    echo "Inference-ready runs matching filter '${FILTER}' in ${RUNS_DIR} (newest first):"
    AVAILABLE_RUNS="$(list_available_matching_runs)"
    if [[ -z "$AVAILABLE_RUNS" ]]; then
        echo "  (none found matching '${FILTER}')"
    else
        while IFS= read -r line; do echo "  $line"; done <<< "$AVAILABLE_RUNS"
    fi
    exit 0
fi

# ── collect matching runs ─────────────────────────────────────────────────────

SUBMIT_RUNS=()
while IFS= read -r name; do
    [[ -n "$name" ]] && SUBMIT_RUNS+=("$name")
done < <(collect_matching_runs)

if [[ ${#SUBMIT_RUNS[@]} -eq 0 ]]; then
    echo "ERROR: No inference-ready runs matching filter '${FILTER}' found in ${RUNS_DIR}"
    echo "       Required per run: config.yaml, ds.pth, and at least one model-*.pt checkpoint."
    exit 1
fi

echo "=========================================================================="
echo " TetraDiffusion Inference Launcher (Filtered)"
echo " Filter          : '${FILTER}'"
echo " Matching runs   : ${#SUBMIT_RUNS[@]}"
echo " Inference mode  : ${INFERENCE_MODE}"
echo " Num images/run  : ${NUM_IMAGES}"
echo " Dry run         : ${DRY_RUN}"
echo "=========================================================================="
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run mode active. The following ${#SUBMIT_RUNS[@]} run(s) would be submitted:"
    for run in "${SUBMIT_RUNS[@]}"; do
        echo "  [DRY RUN] sbatch --job-name=tetradiff_infer_${run} submit_inference.sh --run_name ${run} ${PASSTHROUGH_ARGS[*]}"
    done
    echo ""
    echo "To execute for real, remove the --dry_run flag."
    exit 0
fi

for run in "${SUBMIT_RUNS[@]}"; do
    JOB_NAME="tetradiff_infer_${run}"
    echo "  → Submitting inference for: ${run}"
    sbatch --job-name="$JOB_NAME" submit_inference.sh \
        --run_name "$run" \
        "${PASSTHROUGH_ARGS[@]}"
done

echo ""
echo "Successfully submitted ${#SUBMIT_RUNS[@]} job(s)."
echo "Use 'squeue -u \$USER' to monitor active SLURM inference jobs."
