#!/usr/bin/env python3
"""Structural side-by-side of one of our partials against an official one.

Prints, per file: trees, branches (name/type), entries, distinct bunches, and
entries per bunch.  This is what tells whether a size difference between the
two processings is more hits, more branches, or simply more beam in the file.

Usage:  python inspect_pair.py <ours.root> <official.root>
"""
import sys

import numpy as np
import uproot


def describe(path):
    f = uproot.open(path)
    keys = sorted({k.split(';')[0] for k in f.keys()})
    print(f'\n{"=" * 78}\n{path}')
    print(f'top-level keys ({len(keys)}): {keys}')
    for k in keys:
        try:
            obj = f[k]
        except Exception as e:
            print(f'  {k}: unreadable ({type(e).__name__})')
            continue
        if not hasattr(obj, 'num_entries'):
            try:
                s = obj.member('fString')
                print(f'  {k}: string object, {len(s)} chars')
            except Exception:
                print(f'  {k}: {type(obj).__name__}')
            continue
        n = obj.num_entries
        brs = [(b.name, b.typename) for b in obj.branches]
        nb = ''
        if 'BunchNumber' in [b[0] for b in brs] and n:
            bn = obj['BunchNumber'].array(library='np')
            u = np.unique(bn)
            nb = f'  bunches={u.size} [{u.min()}..{u.max()}]  hits/bunch={n / u.size:.1f}'
        print(f'  {k:<10} entries={n:>10}  branches={len(brs)}{nb}')
        print(f'     {", ".join(f"{a}:{b}" for a, b in brs)}')


def main():
    for p in sys.argv[1:]:
        describe(p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
