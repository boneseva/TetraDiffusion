#!/usr/bin/env python3
"""
Scan a dataset folder for OBJ parsing/rendering issues.
Run from repo root like:
  python preprocessing/scan_objs.py --input_root /path/to/organelles_raw --category mito
"""
import os
import sys
import argparse

# Ensure we can import local preprocessing render modules
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from render import obj


def check_obj(path):
    try:
        m = obj.load_obj(path, clear_ks=True, mtl_override=None, scale=0.95)
        # report shapes
        vshape = tuple(m.v_pos.shape) if getattr(m, 'v_pos', None) is not None else None
        fshape = tuple(m.t_pos_idx.shape) if getattr(m, 't_pos_idx', None) is not None else None
        tex = getattr(m, 'v_tex', None)
        ttex = getattr(m, 't_tex_idx', None)
        return True, f"OK: v_pos={vshape} t_pos_idx={fshape} v_tex={'yes' if tex is not None else 'no'} t_tex_idx={'yes' if ttex is not None else 'no'}"
    except Exception as e:
        return False, f"ERROR: {e}"


def scan(input_root, category=None, max_files=None):
    pattern_root = os.path.join(input_root, category) if category else input_root
    results = []
    count = 0
    for root, dirs, files in os.walk(pattern_root):
        for fn in files:
            if not fn.lower().endswith('.obj'):
                continue
            path = os.path.join(root, fn)
            count += 1
            ok, msg = check_obj(path)
            print(f"[{count}] {path} -> {msg}")
            results.append((path, ok, msg))
            if max_files and count >= max_files:
                return results
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--category', default=None)
    parser.add_argument('--max_files', type=int, default=None)
    args = parser.parse_args()

    scan(args.input_root, args.category, args.max_files)

if __name__ == '__main__':
    main()

