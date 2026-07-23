#!/usr/bin/env bash
# pipeline/run.sh — Bash wrapper for TetraDiffusion Unified Pipeline CLI & Dashboard
#
# Usage:
#   bash pipeline/run.sh status
#   bash pipeline/run.sh register --dataset urocell --category fv
#   bash pipeline/run.sh train --dataset urocell --category fv
#   bash pipeline/run.sh infer --run_name urocell_fv_final_prod
#   bash pipeline/run.sh evaluate --dataset urocell
#   bash pipeline/run.sh dashboard --port 7860
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python3 "${SCRIPT_DIR}/cli.py" "$@"
