#!/usr/bin/env python3
"""stamp_select.py BUNDLE_DIR [--select cost] [--out DIR | --in-place] -- set xy_pairing['select'] in a calib bundle.

For a bench reconstruction that reuses a pass-1 bundle (Analysis_pass1/<rel>/wft/calib_bundle, made before the
'select' key existed, so read as 'gate'): copy it with select = 'cost' (PAIRING_FIX_2026-11-02.md §5).  Only
bundle.json changes (the key, plus a provenance line); arrays.npz and w0kw.json are copied unchanged.  The pass-1
bundles themselves are left as they are by default: they describe how the pass-1 tables were made.  A new bench pass
needs none of this: rerun/w12/set_w0kw.py --table writes select='cost' (--xy-select) into the bundle it produces.
"""
import argparse, datetime, json, os, shutil, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from wft.calib import check_xy_pairing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle')
    ap.add_argument('--select', default='cost', choices=['gate', 'cost'])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--out')
    g.add_argument('--in-place', action='store_true')
    a = ap.parse_args()
    out = a.bundle if a.in_place else a.out
    if not a.in_place:
        shutil.copytree(a.bundle, out, dirs_exist_ok=True)
    p = os.path.join(out, 'bundle.json')
    b = json.load(open(p))
    if not (b.get('xy_pairing') or {}).get('features'):
        sys.exit(f'{a.bundle}: no xy_pairing calibration, select has no effect')
    old = b['xy_pairing'].get('select')
    b['xy_pairing']['select'] = a.select
    check_xy_pairing(b['xy_pairing'], where=p)
    b.setdefault('provenance', {}).setdefault('edits', []).append(dict(
        what=f"xy_pairing select {old!r} -> {a.select!r}", by='rerun/effbudget/single/stamp_select.py',
        why='rerun/effbudget/single/PAIRING_FIX_2026-11-02.md', when=datetime.date.today().isoformat(),
        source=os.path.abspath(a.bundle)))
    json.dump(b, open(p, 'w'), indent=1)
    print(f'{p}: select {old!r} -> {a.select!r}')


if __name__ == '__main__':
    main()
