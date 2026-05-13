
#!/usr/bin/env bash
# launch_inference.sh — wrapper that submits submit_inference.sh with the correct SLURM job name.
#
# Usage:
#   bash launch_inference.sh --run_name er_20260513_0727 --num_images 8
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
sbatch --job-name="$JOB_NAME" submit_inference.sh "$@"

