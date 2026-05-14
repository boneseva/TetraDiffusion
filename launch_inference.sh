#!/usr/bin/env bash
# launch_inference.sh — wrapper that submits submit_inference.sh with the correct SLURM job name.
#
# Usage:
#   bash launch_inference.sh                                          # submit inference for ALL runs with checkpoints
#   bash launch_inference.sh --run_name er_20260513_0727             # target a specific run
#   bash launch_inference.sh --run_name er_20260513_0727 --num_images 8  # with extra options
#   bash launch_inference.sh --list_runs                             # print available runs, then exit
#
# Flags:
#   --run_name            Target a specific run folder (default: all inference-ready runs)
#   --num_images          Number of meshes to generate per run (default: 8)
#   --device              Device to use: cuda or cpu (default: cuda)
#   --cuda_device         GPU device index (default: 0)
#   --out_subdir          Custom output subdirectory inside each run folder
#   --comparison_mode     Deterministic comparison mode (default)
#   --generation_mode     Stochastic generation mode
#   --multi_gpu           Use multi-GPU inference (default: single GPU)
#   --skip_load_weights   Do NOT force load trained weights (not recommended)
#   --list_runs           Print available inference-ready runs and exit
#
# IMPORTANT: By default this script submits one SLURM job per inference-ready run.
#            Inference-ready = has config.yaml + ds.pth + at least one model-*.pt checkpoint.
#            Default inference mode is --comparison_mode so the run logs clearly show deterministic comparison behavior.
#            Use --run_name to restrict to a single run.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"

RUN_NAME=""
FORCE_LOAD_WEIGHTS=true
LIST_RUNS=false
INFERENCE_MODE=""
PASSTHROUGH_ARGS=()   # flags forwarded to every sbatch call (excludes --run_name, --list_runs)

# ── helpers ────────────────────────────────────────────────────────────────────

is_inference_ready_run() {
	local run_dir="$1"
	[[ -d "$run_dir" && -f "$run_dir/config.yaml" && -f "$run_dir/ds.pth" ]] || return 1
	if [[ "$FORCE_LOAD_WEIGHTS" == true ]]; then
		compgen -G "$run_dir/model-*.pt" > /dev/null
		return
	fi
	return 0
}

get_run_sort_key() {
	local run_name="$(basename "$1")"
	if [[ "$run_name" =~ ([0-9]{8})_([0-9]{4})([0-9]{2})?$ ]]; then
		printf '2%s%s%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]:-00}"
	else
		printf '1%020d' "$(stat -c %Y "$1")"
	fi
}

# Returns all inference-ready run names, newest-first (one per line)
collect_ready_runs() {
	local run_dir
	shopt -s nullglob
	for run_dir in "$RUNS_DIR"/*; do
		[[ -d "$run_dir" ]] || continue
		is_inference_ready_run "$run_dir" || continue
		printf '%s\t%s\n' "$(get_run_sort_key "$run_dir")" "$(basename "$run_dir")"
	done | sort -rn | cut -f2-
	shopt -u nullglob
}

list_available_runs() {
	local run_dir run_name checkpoint_status
	shopt -s nullglob
	for run_dir in "$RUNS_DIR"/*; do
		[[ -d "$run_dir" ]] || continue
		[[ -f "$run_dir/config.yaml" && -f "$run_dir/ds.pth" ]] || continue
		run_name="$(basename "$run_dir")"
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
		--run_name)           RUN_NAME="$2";               shift 2 ;;
		--list_runs)          LIST_RUNS=true;               shift   ;;
		--comparison_mode)    [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "comparison" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
		                      INFERENCE_MODE="comparison"; shift   ;;
		--generation_mode)    [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "generation" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
		                      INFERENCE_MODE="generation"; shift   ;;
		--stochastic_sampling) [[ -n "$INFERENCE_MODE" && "$INFERENCE_MODE" != "generation" ]] && { echo "ERROR: conflicting inference mode flags"; exit 1; }
		                      INFERENCE_MODE="generation"
		                      PASSTHROUGH_ARGS+=("$1");     shift   ;;
		--skip_load_weights)  FORCE_LOAD_WEIGHTS=false
		                      PASSTHROUGH_ARGS+=("$1");     shift   ;;
		*)                    PASSTHROUGH_ARGS+=("$1");     shift   ;;
	esac
done

if [[ "$INFERENCE_MODE" == "" ]]; then
	INFERENCE_MODE="generation"
	PASSTHROUGH_ARGS=(--comparison_mode "${PASSTHROUGH_ARGS[@]}")
elif [[ "$INFERENCE_MODE" == "comparison" ]]; then
	PASSTHROUGH_ARGS=(--comparison_mode "${PASSTHROUGH_ARGS[@]}")
elif [[ "$INFERENCE_MODE" == "generation" ]]; then
	PASSTHROUGH_ARGS=(--generation_mode "${PASSTHROUGH_ARGS[@]}")
fi

# ── list mode ─────────────────────────────────────────────────────────────────

if [[ "$LIST_RUNS" == true ]]; then
	echo "Inference-ready runs in ${RUNS_DIR}  (newest first):"
	AVAILABLE_RUNS="$(list_available_runs)"
	if [[ -z "$AVAILABLE_RUNS" ]]; then
		echo "  (none found — need config.yaml + ds.pth + model-*.pt)"
	else
		while IFS= read -r line; do echo "  $line"; done <<< "$AVAILABLE_RUNS"
	fi
	exit 0
fi

# ── build list of runs to submit ──────────────────────────────────────────────

SUBMIT_RUNS=()

if [[ -n "$RUN_NAME" ]]; then
	# explicit single run
	SUBMIT_RUNS=("$RUN_NAME")
else
	# all inference-ready runs
	while IFS= read -r name; do
		[[ -n "$name" ]] && SUBMIT_RUNS+=("$name")
	done < <(collect_ready_runs)

	if [[ ${#SUBMIT_RUNS[@]} -eq 0 ]]; then
		echo "ERROR: no inference-ready runs found in ${RUNS_DIR}"
		echo "       Each run needs config.yaml, ds.pth, and at least one model-*.pt"
		exit 1
	fi
fi

# ── submit one SLURM job per run ──────────────────────────────────────────────

echo "Submitting ${#SUBMIT_RUNS[@]} inference job(s)..."
echo "Inference mode: ${INFERENCE_MODE}"
echo ""

for run in "${SUBMIT_RUNS[@]}"; do
	JOB_NAME="tetradiff_infer_${run}"
	echo "  → $run"
	sbatch --job-name="$JOB_NAME" submit_inference.sh \
		--run_name "$run" \
		"${PASSTHROUGH_ARGS[@]}"
done

echo ""
echo "Done — use 'squeue -u \$USER' to track progress."

