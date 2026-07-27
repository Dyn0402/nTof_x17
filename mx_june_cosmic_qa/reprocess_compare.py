#!/usr/bin/env python3
"""
Old-vs-new hit yield after the 2026-07-24 waveform-analyzer reprocessing.

Reads the pre-overwrite snapshot written by reprocess_cosmic_bench.py
(reprocess_logs/old_combined_counts_*.json) and compares it with the combined
hits now on disk, per subrun and per FEU.

    ../.venv/bin/python reprocess_compare.py [snapshot.json]

Expectation from the analyzer validation (det3 weekend FEU08): the new unified
trigger recovers hits the old derivative trigger dropped, so ratios > 1 are the
norm. Ratios < 1 are worth a look -- the honest 5-10 sigma losses are the
baseline-bias fix rejecting sub-threshold pulses, but a large deficit means
something else (wrong pedestal, ZS/CNS interaction).
"""

import glob
import json
import os
import sys
from collections import defaultdict

import uproot

BASE_DATA = '/media/dylan/data/x17/cosmic_bench'
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reprocess_logs')


def newest_snapshot():
    snaps = sorted(glob.glob(os.path.join(LOG_DIR, 'old_combined_counts_*.json')))
    if not snaps:
        sys.exit('no snapshot found in reprocess_logs/')
    return snaps[-1]


def counts_now(rel_paths):
    """Entry count + per-FEU breakdown of the combined files on disk today."""
    out = {}
    for rel in rel_paths:
        path = os.path.join(BASE_DATA, rel)
        if not os.path.exists(path):
            out[rel] = None  # quarantined into _old_analyzer/ or never rebuilt
            continue
        try:
            with uproot.open(path) as fh:
                key = fh.keys(recursive=False)[0]
                tree = fh[key]
                feu = tree['feu'].array(library='np')
                per_feu = {int(f): int((feu == f).sum()) for f in set(feu.tolist())}
                out[rel] = (int(tree.num_entries), per_feu)
        except Exception as e:
            out[rel] = f'ERROR: {e}'
    return out


def main():
    snap_path = sys.argv[1] if len(sys.argv) > 1 else newest_snapshot()
    with open(snap_path) as fh:
        old = json.load(fh)
    print(f'snapshot: {snap_path}  ({len(old)} old combined files)\n')

    new = counts_now(old.keys())

    by_sub_old, by_sub_new, by_sub_feu = (defaultdict(int), defaultdict(int),
                                          defaultdict(lambda: defaultdict(int)))
    missing, errors = [], []
    for rel, o in old.items():
        sub = os.path.dirname(os.path.dirname(rel))
        n = new.get(rel)
        if n is None:
            missing.append(rel)
            continue
        if isinstance(n, str) or isinstance(o, str):
            errors.append((rel, o, n))
            continue
        by_sub_old[sub] += o
        by_sub_new[sub] += n[0]
        for f, c in n[1].items():
            by_sub_feu[sub][f] += c

    print(f'{"subrun":<70} {"old hits":>12} {"new hits":>12} {"ratio":>7}  per-FEU (new)')
    for sub in sorted(by_sub_new):
        o, n = by_sub_old[sub], by_sub_new[sub]
        ratio = f'{n/o:.2f}x' if o else '   n/a'
        feus = ' '.join(f'{f}:{c/1e3:.0f}k' for f, c in sorted(by_sub_feu[sub].items()))
        flag = '  <-- CHECK' if o and n / o < 0.9 else ''
        print(f'{sub:<70} {o:>12,} {n:>12,} {ratio:>7}  {feus}{flag}')

    tot_o = sum(by_sub_old.values())
    tot_n = sum(by_sub_new.values())
    print(f'\n{"TOTAL":<70} {tot_o:>12,} {tot_n:>12,} '
          f'{tot_n/tot_o:.2f}x' if tot_o else '')
    if missing:
        print(f'\n{len(missing)} old combined files not present now '
              f'(quarantined to _old_analyzer/ -- old-generation, decoded input absent):')
        for rel in sorted(missing)[:10]:
            print(f'   {rel}')
        if len(missing) > 10:
            print(f'   ... and {len(missing)-10} more')
    if errors:
        print(f'\n{len(errors)} files could not be read:')
        for rel, o, n in errors:
            print(f'   {rel}: old={o} new={n}')


if __name__ == '__main__':
    main()
