#!/usr/bin/env python3
"""
list_checkpoints.py — show step numbers for all model-*.pt files
without loading any tensor weights.

Usage:
    python scripts/list_checkpoints.py               # scan runs/
    python scripts/list_checkpoints.py runs/abl_bio_on
"""
import sys
import os
import json
import pickle
import glob
import datetime


# ── Stub unpickler: replaces all torch tensors / storages with None ──────────
class _SkipTensors(pickle.Unpickler):
    """Loads only plain Python scalars/dicts/lists — skips all tensor data."""
    def find_class(self, module, name):
        if 'torch' in module or name in ('Tensor', 'storage', 'Storage'):
            return lambda *a, **k: None
        return super().find_class(module, name)


def read_step(pt_path):
    """Return (step, saved_at, source) without loading tensors."""
    # Fast path: JSON sidecar written by the updated Trainer.save()
    json_path = pt_path.replace('.pt', '.json')
    if os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        return meta.get('step', '?'), meta.get('saved_at', ''), 'json'

    mtime = datetime.datetime.fromtimestamp(
        os.path.getmtime(pt_path)).isoformat(timespec='seconds')

    # Try new zip-based format first (torch >= 1.6 default)
    try:
        import zipfile
        with zipfile.ZipFile(pt_path) as zf:
            # find the data.pkl entry regardless of prefix
            pkl_name = next((n for n in zf.namelist() if n.endswith('data.pkl')), None)
            if pkl_name:
                with zf.open(pkl_name) as pkl:
                    data = _SkipTensors(pkl).load()
                step = data.get('step', '?') if isinstance(data, dict) else '?'
                return step, mtime, 'pt'
    except Exception:
        pass

    # Fall back to legacy raw-pickle format
    try:
        with open(pt_path, 'rb') as f:
            data = _SkipTensors(f).load()
        step = data.get('step', '?') if isinstance(data, dict) else '?'
        return step, mtime, 'pt'
    except Exception as e:
        return f'ERR({e})', mtime, 'pt'



def main():
    search_root = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    pattern = os.path.join(search_root, '**', 'model-*.pt')
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f'No model-*.pt files found under {search_root!r}')
        return

    rows = []
    for pt in files:
        step, ts, src = read_step(pt)
        rows.append((step, ts, src, pt))

    # Sort by step (errors last)
    rows.sort(key=lambda r: r[0] if isinstance(r[0], int) else 999_999_999)

    print(f"{'STEP':>8}  {'SAVED AT':>19}  {'SRC':>4}  PATH")
    print('-' * 80)
    for step, ts, src, path in rows:
        print(f"{str(step):>8}  {ts:>19}  {src:>4}  {path}")


if __name__ == '__main__':
    main()
