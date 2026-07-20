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

    def persistent_load(self, pid):
        # PyTorch stores tensor data as persistent IDs pointing to binary blobs.
        # Return None so the unpickler can continue reading the dict structure.
        return None


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
    import argparse
    parser = argparse.ArgumentParser(description='List and manage TetraDiffusion checkpoints.')
    parser.add_argument('root', nargs='?', default='runs',
                        help='Directory to scan (default: runs/)')
    parser.add_argument('--clean', action='store_true',
                        help='Delete the lower-step checkpoint in each run, keeping only the best.')
    parser.add_argument('--yes', action='store_true',
                        help='Actually delete (default is dry-run, only prints what would be removed).')
    args = parser.parse_args()

    pattern = os.path.join(args.root, '**', 'model-*.pt')
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f'No model-*.pt files found under {args.root!r}')
        return

    rows = []
    for pt in files:
        step, ts, src = read_step(pt)
        rows.append((step, ts, src, pt))

    # Sort by step (errors last)
    rows.sort(key=lambda r: r[0] if isinstance(r[0], int) else 999_999_999)

    if not args.clean:
        print(f"{'STEP':>8}  {'SAVED AT':>19}  {'SRC':>4}  PATH")
        print('-' * 80)
        for step, ts, src, path in rows:
            print(f"{str(step):>8}  {ts:>19}  {src:>4}  {path}")
        return

    # ── --clean mode ──────────────────────────────────────────────────────────
    # Group by run folder (parent directory of the .pt file)
    from collections import defaultdict
    by_run = defaultdict(list)
    for step, ts, src, path in rows:
        run_dir = os.path.dirname(path)
        by_run[run_dir].append((step, path))

    to_delete = []
    for run_dir, entries in sorted(by_run.items()):
        valid = [(s, p) for s, p in entries if isinstance(s, int)]
        errors = [(s, p) for s, p in entries if not isinstance(s, int)]
        if len(valid) <= 1:
            continue  # nothing to clean
        best_step = max(s for s, _ in valid)
        for step, path in valid:
            if step < best_step:
                to_delete.append((run_dir, step, best_step, path))
        for step, path in errors:
            print(f"  SKIP (unreadable, step={step}): {path}")

    if not to_delete:
        print("Nothing to clean — each run already has only one checkpoint or all steps are equal.")
        return

    print(f"{'RUN':<35}  {'DEL STEP':>8}  {'KEEP STEP':>9}  FILE")
    print('-' * 80)
    for run_dir, step, best, path in to_delete:
        tag = '' if args.yes else '  [dry-run]'
        print(f"{os.path.basename(run_dir):<35}  {step:>8}  {best:>9}  {os.path.basename(path)}{tag}")

    if not args.yes:
        print(f"\n{len(to_delete)} file(s) would be deleted. Re-run with --yes to confirm.")
        return

    for _, _, _, path in to_delete:
        os.remove(path)
        sidecar = path.replace('.pt', '.json')
        if os.path.isfile(sidecar):
            os.remove(sidecar)
    print(f"\nDeleted {len(to_delete)} checkpoint(s).")


if __name__ == '__main__':
    main()
