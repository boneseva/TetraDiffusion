#!/usr/bin/env bash
# reorganize_objs.sh
#
# Converts flat layout:
#   <input_root>/<category>/<model_id>.obj
#
# Into the layout expected by fit_many.py:
#   <output_root>/<category>/<model_id>/<model_id>.obj
#
# Usage:
#   bash reorganize_objs.sh <input_root> <output_root>
#
# Example:
#   bash reorganize_objs.sh /home/evalocal/data_urocell/organelles /home/evalocal/data_urocell/organelles_raw

set -euo pipefail

INPUT_ROOT="${1:?Usage: $0 <input_root> <output_root>}"
OUTPUT_ROOT="${2:?Usage: $0 <input_root> <output_root>}"

if [ ! -d "$INPUT_ROOT" ]; then
  echo "ERROR: input_root does not exist: $INPUT_ROOT"
  exit 1
fi

echo "Input root:  $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo ""

count=0
skipped=0

# Iterate over every .obj file two levels deep: category/model_id.obj
for obj_path in "$INPUT_ROOT"/*/*.obj; do
  # Guard against empty glob
  [ -f "$obj_path" ] || continue

  category=$(basename "$(dirname "$obj_path")")
  filename=$(basename "$obj_path")
  model_id="${filename%.obj}"

  dest_dir="$OUTPUT_ROOT/$category/$model_id"
  dest_file="$dest_dir/$filename"

  if [ -f "$dest_file" ]; then
    echo "[SKIP] Already exists: $dest_file"
    skipped=$((skipped + 1))
    continue
  fi

  mkdir -p "$dest_dir"
  cp "$obj_path" "$dest_file"
  echo "[COPY] $obj_path  ->  $dest_file"
  count=$((count + 1))
done

echo ""
echo "Done. Copied: $count  Skipped (already exist): $skipped"
echo ""
echo "Next step — run preprocessing from inside the preprocessing/ folder:"
echo ""
echo "  cd $(dirname "$0")/preprocessing"
echo "  python fit_many.py \\"
echo "    --input_root  \"$OUTPUT_ROOT\" \\"
echo "    --output_root \"/your/data/output_root\" \\"
echo "    --dmtet_grid 128 \\"
echo "    --iter 3000 \\"
echo "    --sanitize \\"
echo "    --update_all_csv ../lib/all.csv"
