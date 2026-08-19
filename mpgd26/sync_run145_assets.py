#!/usr/bin/env python3
"""
sync_run145_assets.py — copy the run_145 figures the deck shows out of the
analysis directory, and record where each one came from.

These three slide assets were hand-copied on 2026-08-13 and there was nothing
in the repo that said so, so when run_145 was re-reconstructed on the corrected
sharing kernel the deck kept showing the previous reconstruction's pictures
beside the new numbers. Identified by md5 against the parked originals; this
script is so it cannot happen silently again.

    ../.venv/bin/python mpgd26/sync_run145_assets.py [--check]
"""
import argparse
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = ('/media/dylan/data/x17/beam_july/analysis/wft/run_145/'
            'stat090_0000')
DEST = os.path.join(HERE, 'slides', 'assets', 'img')

ASSETS = {
    'run145_image.png':
        f'{ANALYSIS}/imaging/note_figs_fullcov/fig2_image.png',
    'run145_pointing.png':
        f'{ANALYSIS}/imaging/note_figs_fullcov/fig1_tan_vs_u.png',
    'run145_wall3d_all.png':
        f'{ANALYSIS}/imaging_fullcov/wall_3d/wall3d_run145_all_arms.png',
}


def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='report staleness, copy nothing')
    a = ap.parse_args()
    rec, stale = {}, []
    for name, src in ASSETS.items():
        dst = os.path.join(DEST, name)
        if not os.path.isfile(src):
            print(f'MISSING SOURCE  {name}  <- {src}')
            stale.append(name)
            continue
        same = os.path.isfile(dst) and md5(dst) == md5(src)
        if not same:
            stale.append(name)
            if not a.check:
                shutil.copy2(src, dst)
        print(f'{"ok   " if same else ("STALE" if a.check else "copied")}  '
              f'{name}  <- {src.replace(ANALYSIS, "<run_145/stat090_0000>")}')
        rec[name] = dict(source=src, md5=md5(src) if not a.check or same
                         else md5(src))
    if not a.check:
        with open(os.path.join(DEST, 'run145_assets.json'), 'w') as f:
            json.dump(rec, f, indent=1)
    return 1 if (a.check and stale) else 0


if __name__ == '__main__':
    raise SystemExit(main())
