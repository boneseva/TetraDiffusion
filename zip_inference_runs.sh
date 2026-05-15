#!/usr/bin/env bash
# zip_inference_runs.sh
#
# Zips all inference_* folders inside the runs/ directory into a single archive.
# Output: runs/inference_runs_<timestamp>.zip
#
# Usage:
#   bash zip_inference_runs.sh
#   bash zip_inference_runs.sh --output my_inference.zip   # custom output path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RUNS_DIR}/inference_runs_${TIMESTAMP}.zip"

# Allow overriding output path via --output flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o)
            OUTPUT="$2"
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

# Collect inference_* folders
mapfile -t FOLDERS < <(find "${RUNS_DIR}" -maxdepth 1 -type d -name "inference_*" | sort)

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    echo "No inference_* folders found in ${RUNS_DIR}" >&2
    exit 1
fi

echo "Found ${#FOLDERS[@]} inference folder(s):"
for f in "${FOLDERS[@]}"; do
    echo "  ${f}"
done
echo ""
echo "Zipping into: ${OUTPUT}"

# Build zip from repo root so paths inside the archive are runs/inference_*/...
cd "${SCRIPT_DIR}"
RELATIVE_FOLDERS=()
for f in "${FOLDERS[@]}"; do
    RELATIVE_FOLDERS+=("runs/$(basename "${f}")")
done

zip -r "${OUTPUT}" "${RELATIVE_FOLDERS[@]}"

echo ""
echo "Done. Archive size: $(du -sh "${OUTPUT}" | cut -f1)  →  ${OUTPUT}"

