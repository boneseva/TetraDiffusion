#!/usr/bin/env bash
# pipeline/sync_from_cluster.sh — Helper script to sync TetraDiffusion from HPC Cluster to VM
#
# Usage (run this script on your target Virtual Machine):
#   bash pipeline/sync_from_cluster.sh <REMOTE_USER_HOST_AND_PATH> [LOCAL_TARGET_DIR]
#
# Example:
#   bash pipeline/sync_from_cluster.sh eva.bones@login-frida:/shared/home/eva.bones/TetraDiffusion ~/TetraDiffusion
#
# Optional flags:
#   --watch               Run rsync continuously every N seconds (default: 30s)
#   --dry_run             Preview files that would be synced without copying
#

set -euo pipefail

REMOTE_SRC=""
LOCAL_DST="$(pwd)"
WATCH=false
WATCH_INTERVAL=30
DRY_RUN=false
EXTRA_RSYNC_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)        WATCH=true; shift ;;
        --interval)     WATCH_INTERVAL="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=true; EXTRA_RSYNC_FLAGS+=("--dry-run"); shift ;;
        -*)             echo "Unknown flag: $1" >&2; exit 1 ;;
        *)              if [[ -z "$REMOTE_SRC" ]]; then
                            REMOTE_SRC="$1"
                        else
                            LOCAL_DST="$1"
                        fi
                        shift ;;
    esac
done

if [[ -z "$REMOTE_SRC" ]]; then
    echo "ERROR: Remote source required." >&2
    echo "Usage: bash pipeline/sync_from_cluster.sh <REMOTE_USER_HOST:PATH> [LOCAL_TARGET_DIR]" >&2
    echo "Example: bash pipeline/sync_from_cluster.sh eva.bones@login-frida:/shared/home/eva.bones/TetraDiffusion ~/TetraDiffusion" >&2
    exit 1
fi

# Standard excludes to keep sync fast and avoid unnecessary bandwidth
EXCLUDES=(
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=.git/"
    "--exclude=.idea/"
    "--exclude=.vscode/"
    "--exclude=*.sqfs"
    "--exclude=wandb/latest-run/"
    "--exclude=*.tmp"
    "--exclude=scratch/"
)

do_sync() {
    echo "=========================================================================="
    echo " 🔄 Syncing TetraDiffusion from HPC Cluster..."
    echo " Remote: ${REMOTE_SRC}"
    echo " Local : ${LOCAL_DST}"
    echo " Time  : $(date)"
    echo "=========================================================================="

    mkdir -p "${LOCAL_DST}"

    rsync -avzP \
        "${EXCLUDES[@]}" \
        "${EXTRA_RSYNC_FLAGS[@]}" \
        "${REMOTE_SRC}/" \
        "${LOCAL_DST}/"

    echo "✓ Sync finished cleanly!"
}

if [[ "$WATCH" == true ]]; then
    echo "Starting continuous watch mode (syncing every ${WATCH_INTERVAL} seconds)..."
    while true; do
        do_sync || true
        echo "Sleeping for ${WATCH_INTERVAL}s... (Press Ctrl+C to stop)"
        sleep "$WATCH_INTERVAL"
    done
else
    do_sync
fi
