"""Simple OBJ sanitizer.

Converts face elements like 'f v/vt/vn' or 'f v//vn' to 'f v v v' (keep only vertex indices).
Creates a backup '<filename>.bak' when overwriting in place.

Usage:
  python sanitize_obj.py <path-to-obj> [--inplace]
  python sanitize_obj.py --glob "data/**/*.obj" --dry_run
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def sanitize_file(path: Path, inplace: bool = True) -> Path:
    text = path.read_text(encoding='utf-8')
    out_lines = []
    changed = False
    for line in text.splitlines(keepends=False):
        if line.strip().lower().startswith('f '):
            parts = line.split()
            verts = parts[1:]
            new_verts = []
            for v in verts:
                # take the part before the first slash (vertex index)
                new_verts.append(v.split('/')[0])
            new_line = 'f ' + ' '.join(new_verts)
            if new_line != line:
                changed = True
            out_lines.append(new_line)
        else:
            out_lines.append(line)

    if not changed and inplace:
        return path

    if inplace:
        bak = path.with_suffix(path.suffix + '.bak')
        path.replace(bak)
        path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
        return path
    else:
        new_path = path.with_name(path.stem + '_sanitized' + path.suffix)
        new_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
        return new_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='OBJ file(s) or glob')
    parser.add_argument('--inplace', action='store_true', help='Overwrite files (create .bak backup)')
    parser.add_argument('--dry_run', action='store_true', help='Print which files would be changed')
    args = parser.parse_args(argv)

    for p in args.paths:
        ppath = Path(p)
        if ppath.exists() and ppath.is_file():
            if args.dry_run:
                print(p)
                continue
            out = sanitize_file(ppath, inplace=args.inplace)
            print(f"Sanitized: {out}")
        else:
            # treat as glob
            for f in sorted(Path('.').glob(p)):
                if not f.is_file():
                    continue
                if args.dry_run:
                    print(f)
                    continue
                out = sanitize_file(f, inplace=args.inplace)
                print(f"Sanitized: {out}")


if __name__ == '__main__':
    raise SystemExit(main())

