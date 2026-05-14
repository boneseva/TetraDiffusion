#!/usr/bin/env bash
# launch_inference.sh — wrapper that submits submit_inference.sh with the correct SLURM job name.
#
# Usage:
#   bash launch_inference.sh --run_name er_20260513_0727 --num_images 8
#   bash launch_inference.sh --run_name er_20260513_0727 --num_images 8 --skip_load_weights
#
# Flags:
#   --run_name            Name of the inference run folder (required)
#   --num_images          Number of meshes to generate (default: 8)
#   --device              Device to use: cuda or cpu (default: cuda)
#   --cuda_device         GPU device index (default: 0)
#   --out_subdir          Custom output subdirectory (default: organelle-aware inference folder name)
#   --multi_gpu           Use multi-GPU inference (default: single GPU)
#   --skip_load_weights   Do NOT force load trained weights (default: load weights)
#
# IMPORTANT: By default, this script enables --force_load_weights to load trained model checkpoints.
#            If --out_subdir is omitted, inference.py creates an organelle-aware folder name.
#            Use --skip_load_weights if you want to run inference with random weights (not recommended).
#
set -euo pipefail

# Extract --run_name value so we can name the job
RUN_NAME="unknown"
args=("$@")
for (( i=0; i<${#args[@]}; i++ )); do
	if [[ "${args[$i]}" == "--run_name" && $((i+1)) -lt ${#args[@]} ]]; then
		RUN_NAME="${args[$((i+1))]}"
		break
	fi
done

JOB_NAME="tetradiff_infer_${RUN_NAME}"

echo "Submitting job: $JOB_NAME"
echo "  (with automatic weight loading enabled by default)"
sbatch --job-name="$JOB_NAME" submit_inference.sh "$@"

