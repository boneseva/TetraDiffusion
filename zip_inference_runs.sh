#!/usr/bin/env bash
# zip_inference_runs.sh
#
# Zips inference_* folders inside the runs/ directory into a single archive.
# Output: runs/inference_runs_<timestamp>.zip (or custom path via --output)
#
# Usage:
#   bash zip_inference_runs.sh
#   bash zip_inference_runs.sh --output my_inference.zip     # custom output path
#   bash zip_inference_runs.sh --filter abl_                 # filter runs starting with or containing 'abl_'
#   bash zip_inference_runs.sh -f "*bio*" -o bio_runs.zip    # glob pattern filter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT=""
FILTER=""
OUTPUT_SPECIFIED=false

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o)
            OUTPUT="$2"
            OUTPUT_SPECIFIED=true
            shift 2
            ;;
        --filter|-f)
            FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${RUNS_DIR}" ]]; then
    echo "ERROR: runs/ directory not found at ${RUNS_DIR}" >&2
    exit 1
fi

# Set default output name if not explicitly specified
if [[ "$OUTPUT_SPECIFIED" == false ]]; then
    if [[ -n "$FILTER" ]]; then
        CLEAN_FILTER="$(echo "$FILTER" | tr -cd 'a-zA-Z0-9_-')"
        OUTPUT="${RUNS_DIR}/inference_runs_${CLEAN_FILTER}_${TIMESTAMP}.zip"
    else
        OUTPUT="${RUNS_DIR}/inference_runs_${TIMESTAMP}.zip"
    fi
fi

matches_filter() {
    local path="$1"
    local rel_path="${path#${RUNS_DIR}/}"
    local run_name="${rel_path%%/*}"

    if [[ -z "$FILTER" ]]; then
        return 0
    fi

    local run_name_lower="${run_name,,}"
    local filter_lower="${FILTER,,}"

    if [[ "$FILTER" == *'*'* || "$FILTER" == *'?'* ]]; then
        if [[ "$run_name_lower" == $filter_lower || "$rel_path" == $filter_lower ]]; then
            return 0
        fi
    else
        if [[ "$run_name_lower" == "$filter_lower"* || "$run_name_lower" == *"$filter_lower"* ]]; then
            return 0
        fi
    fi

    return 1
}

# Collect inference_* folders (any depth under runs/)
mapfile -t ALL_FOLDERS < <(find "${RUNS_DIR}" -type d -name "inference_*" | sort)

FOLDERS=()
for f in "${ALL_FOLDERS[@]}"; do
    if matches_filter "$f"; then
        FOLDERS+=("$f")
    fi
done

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    if [[ -n "$FILTER" ]]; then
        echo "No inference_* folders found matching filter '${FILTER}' in ${RUNS_DIR}" >&2
    else
        echo "No inference_* folders found in ${RUNS_DIR}" >&2
    fi
    exit 1
fi

echo "Found ${#FOLDERS[@]} inference folder(s)${FILTER:+ matching filter '${FILTER}'}:"
for f in "${FOLDERS[@]}"; do
    echo "  ${f}"
done
echo ""
echo "Zipping into: ${OUTPUT}"

# Build zip from repo root so paths inside the archive mirror the on-disk layout
cd "${SCRIPT_DIR}"
RELATIVE_FOLDERS=()
for f in "${FOLDERS[@]}"; do
    # Strip the repo root prefix to get a path like runs/.../inference_*
    RELATIVE_FOLDERS+=("${f#${SCRIPT_DIR}/}")
done

zip -r "${OUTPUT}" "${RELATIVE_FOLDERS[@]}"

echo ""
echo "Done. Archive size: $(du -sh "${OUTPUT}" | cut -f1)  →  ${OUTPUT}"


