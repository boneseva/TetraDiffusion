#!/usr/bin/env bash
# submit_preprocess.sh — SLURM array job submission script for TetraDiffusion preprocessing
#
# Runs preprocessing/fit_many.py -- one SLURM array task per category so all
# categories are processed in parallel (or sequentially if not using arrays).
#
# ─── Quick-start examples ──────────────────────────────────────────────────────
#
# 1. Auto-discover categories from input_root and launch ONE JOB PER CATEGORY:
#      bash submit_preprocess.sh --input_root /shared/home/eva.bones/TetraDiffusion/data_urocell/organelles_raw \
#                                --output_root /shared/home/eva.bones/TetraDiffusion/data_urocell/preprocessed \
#                                --launch_array
#
#    This calls "sbatch --array=0-N submit_preprocess.sh ..." automatically.
#
# 2. Manual array (you control the indices):
#      sbatch --array=0-3 submit_preprocess.sh \
#             --input_root /path/to/organelles_raw \
#             --output_root /path/to/preprocessed \
#             --categories ER Golgi Lysosome Mitochondria
#
# 3. Single category (no array):
#      sbatch submit_preprocess.sh \
#             --input_root /path/to/organelles_raw \
#             --output_root /path/to/preprocessed \
#             --category ER
#
# 4. Dry-run to see what jobs would be dispatched:
#      bash submit_preprocess.sh --input_root /path/to/organelles_raw \
#                                --output_root /path/to/preprocessed \
#                                --dry_run --launch_array
#
# ─── fit_many.py pass-through flags ───────────────────────────────────────────
#   --dmtet_grid 128              grid resolution (64 / 128 / 192, default 128)
#   --iter 3000                   optimisation iterations per mesh (default 3000)
#   --batch 3                     renderer batch size per step (default 3)
#   --train_res 1024 1024         training image resolution
#   --texture_res 512 512         texture resolution
#   --overwrite                   re-fit even if sample.pth already exists
#   --sanitize                    sanitize OBJ files in-place before fitting
#   --obj_glob "*/*/**/*.obj"     custom glob pattern relative to input_root
#
# ─── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=tetradiff_preproc
#SBATCH --output=/shared/home/eva.bones/TetraDiffusion/logs/slurm_preproc_%A_%a_%x.out
#SBATCH --error=/shared/home/eva.bones/TetraDiffusion/logs/slurm_preproc_%A_%a_%x.err
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=frida
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Repo root ────────────────────────────────────────────────────────────────
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

# ─── Argument parsing ─────────────────────────────────────────────────────────
INPUT_ROOT=""
OUTPUT_ROOT=""
CATEGORY=""
CATEGORIES=()
LAUNCH_ARRAY=false
DRY_RUN_FLAG=""
EXTRA_FIT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_root)   INPUT_ROOT="$2";  shift 2 ;;
        --output_root)  OUTPUT_ROOT="$2"; shift 2 ;;
        --category)     CATEGORY="$2";   shift 2 ;;
        --categories)   shift
                        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                            CATEGORIES+=("$1"); shift
                        done ;;
        --launch_array) LAUNCH_ARRAY=true; shift ;;
        --dry_run)      DRY_RUN_FLAG="--dry_run"; shift ;;
        --all_csv|--update_all_csv)
                        EXTRA_FIT_ARGS+=("--update_all_csv" "$2"); shift 2 ;;
        # Pass everything else straight to fit_many.py
        *)              EXTRA_FIT_ARGS+=("$1"); shift ;;
    esac
done

# ─── Validate required arguments ──────────────────────────────────────────────
if [[ -z "$INPUT_ROOT" || -z "$OUTPUT_ROOT" ]]; then
    echo "ERROR: --input_root and --output_root are required." >&2
    echo "       Run: bash submit_preprocess.sh --help  (or read the header)" >&2
    exit 1
fi

INPUT_ROOT="$(realpath "$INPUT_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"   # -m: don't require it to exist yet

# ─── Resolve category list ────────────────────────────────────────────────────
# Priority: --categories > --category > auto-discover subdirs of input_root
if [[ ${#CATEGORIES[@]} -eq 0 && -n "$CATEGORY" ]]; then
    CATEGORIES=("$CATEGORY")
fi

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
    echo "No --categories given; auto-discovering from ${INPUT_ROOT} ..."
    mapfile -t CATEGORIES < <(
        find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
    )
    if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
        echo "ERROR: no subdirectories found in ${INPUT_ROOT}" >&2
        exit 1
    fi
    echo "Discovered categories: ${CATEGORIES[*]}"
fi

NUM_CATS=${#CATEGORIES[@]}

# ─── --launch_array mode: re-submit this script via sbatch with --array ───────
# When the user calls `bash submit_preprocess.sh --launch_array ...` from the
# login node we just re-sbatch ourselves with the correct --array range.
if [[ "$LAUNCH_ARRAY" == true && -z "${SLURM_JOB_ID:-}" ]]; then
    ARRAY_RANGE="0-$((NUM_CATS - 1))"

    # Rebuild the category list as explicit --categories args so the re-launched
    # job does not have to re-discover them and gets the same order.
    CATS_ARGS=("--categories" "${CATEGORIES[@]}")

    DRY_SUFFIX=()
    [[ -n "$DRY_RUN_FLAG" ]] && DRY_SUFFIX=("--dry_run")

    SBATCH_CMD=(
        sbatch --array="${ARRAY_RANGE}"
        "$0"
        --input_root  "$INPUT_ROOT"
        --output_root "$OUTPUT_ROOT"
        "${CATS_ARGS[@]}"
        "${DRY_SUFFIX[@]}"
        "${EXTRA_FIT_ARGS[@]}"
    )

    if [[ -n "$DRY_RUN_FLAG" ]]; then
        echo "[DRY-RUN] Would submit array job (${NUM_CATS} tasks, array=${ARRAY_RANGE})"
        echo "  Categories : ${CATEGORIES[*]}"
        echo "  Command    : ${SBATCH_CMD[*]}"
        exit 0
    fi

    echo "Submitting array job (${NUM_CATS} tasks, array=${ARRAY_RANGE}) ..."
    echo "  Categories: ${CATEGORIES[*]}"
    "${SBATCH_CMD[@]}"
    exit 0
fi

# ─── Select category for this task ────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK_IDX="${SLURM_ARRAY_TASK_ID}"
    if [[ "$TASK_IDX" -ge "$NUM_CATS" ]]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_IDX} but only ${NUM_CATS} categories defined." >&2
        exit 1
    fi
    ACTIVE_CATEGORY="${CATEGORIES[$TASK_IDX]}"
else
    # Not an array job — process all categories sequentially (or just one if --category was given)
    if [[ ${#CATEGORIES[@]} -gt 1 ]]; then
        echo "WARNING: Running all ${NUM_CATS} categories sequentially (no array)."
        echo "         Use --launch_array or submit with 'sbatch --array=0-$((NUM_CATS-1))' for parallel execution."
    fi
    ACTIVE_CATEGORY=""   # handled in the loop below
fi

# ─── Container setup (mirrors submit_train.sh) ────────────────────────────────
CONTAINER="${CONTAINER:-${REPO_DIR}/pytorch2604_tetradiff_updated.sqfs}"
PYXIS_FLAGS="--container-image=${CONTAINER} \
             --container-mounts=${REPO_DIR}:${REPO_DIR},${INPUT_ROOT}:${INPUT_ROOT},${OUTPUT_ROOT}:${OUTPUT_ROOT} \
             --container-mount-home \
             --container-workdir=${REPO_DIR}"

export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true

mkdir -p "${REPO_DIR}/logs" || true

# ─── Helper: run fit_many.py for one category ─────────────────────────────────
run_category() {
    local cat="$1"

    echo "================================================"
    echo "Job ID       : ${SLURM_JOB_ID:-local}  (array task: ${SLURM_ARRAY_TASK_ID:-none})"
    echo "Node         : $(hostname)"
    echo "GPU          : $(echo ${CUDA_VISIBLE_DEVICES:-auto})"
    echo "Category     : ${cat}"
    echo "Input root   : ${INPUT_ROOT}"
    echo "Output root  : ${OUTPUT_ROOT}"
    echo "Date         : $(date)"
    echo "================================================"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    # Update SLURM job name to include the category (best-effort).
    # Only do this for non-array jobs: in array jobs all tasks share the same
    # SLURM_JOB_ID, so renaming it from one task would overwrite the names set
    # by the other tasks.  Array tasks are already identified by their index in
    # squeue output (e.g. tetradiff_preproc[0], tetradiff_preproc[1], ...).
    if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        scontrol update "JobId=${SLURM_JOB_ID}" "JobName=preproc_${cat}" 2>/dev/null || true
    fi

    FIT_CMD=(
        python3 preprocessing/fit_many.py
        --input_root  "$INPUT_ROOT"
        --output_root "$OUTPUT_ROOT"
        --category    "$cat"
    )
    [[ -n "$DRY_RUN_FLAG" ]] && FIT_CMD+=("--dry_run")
    FIT_CMD+=("${EXTRA_FIT_ARGS[@]}")

    echo "Command: ${FIT_CMD[*]}"

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # ── Install preprocessing-only deps if not already present ────────────
        # nvdiffrast is not in the main container image.  We install into
        # ~/.local (--user) so the install persists across array tasks and only
        # actually runs on the first task that needs it.  Subsequent tasks skip
        # the heavy build because the import check passes immediately.
        # If compute nodes have no internet, run the install once interactively:
        #   srun --pty --gres=gpu:1 --partition=frida \
        #        srun <pyxis-flags> bash -c "pip install --user git+https://github.com/NVlabs/nvdiffrast.git xatlas"
        srun $PYXIS_FLAGS bash -c "
            python3 -c 'import nvdiffrast' 2>/dev/null || {
                echo '[preproc] Installing nvdiffrast...'
                pip install --user --quiet git+https://github.com/NVlabs/nvdiffrast.git
            }
            python3 -c 'import xatlas' 2>/dev/null || {
                echo '[preproc] Installing xatlas...'
                pip install --user --quiet xatlas
            }
        "
        srun $PYXIS_FLAGS "${FIT_CMD[@]}"
    else
        # Local / interactive run — no srun/container
        "${FIT_CMD[@]}"
    fi

    echo "================================================"
    echo "Preprocessing finished for '${cat}': $(date)"
    echo "================================================"
}

# ─── Execute ──────────────────────────────────────────────────────────────────
if [[ -n "$ACTIVE_CATEGORY" ]]; then
    run_category "$ACTIVE_CATEGORY"
else
    # Sequential fallback (no array)
    for cat in "${CATEGORIES[@]}"; do
        run_category "$cat"
    done
fi

