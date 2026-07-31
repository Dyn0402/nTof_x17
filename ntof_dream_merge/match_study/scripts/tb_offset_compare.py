#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_offset_compare.py -- the wall top/bottom "cable offsets" are a PSA artifact.

`dream_trigger` carries a measured table of per-segment t_top - t_bottom offsets
of either ~0 or ~+-32..40 ns and reads them as "a cabling difference", because
that is what the OFFICIAL processing of run 224572 shows. On the v12
reprocessing, with the same estimator and the same bunches, they are gone: every
segment sits within a few ns of zero. Wall pulse-shape fitting (v4_walshapes
onwards) is the change that did it, so the tens-of-ns structure was the flash
finder / leading-edge timing of the old reconstruction, not cables.

This matters twice over. The trigger emulation pairs the two bar ends inside
+-25 ns of the measured offset; on a file where the offsets are real that
measurement is mandatory, and on v12 a plain window about zero is correct and the
stale table would be actively wrong.

USAGE
    python tb_offset_compare.py [--nb 150]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import DATA, NTOF_RUN, V12

ARMS = ('A', 'B', 'C', 'D')


def measure(bunches):
    from ntof_dream_merge import fast_singles as fs
    fs.REPAIR_TFLASH = False
    return {a: fs.measure_tb_offsets(NTOF_RUN, bunches, a) for a in ARMS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', type=int, default=150)
    args = ap.parse_args()

    # official first: use_variant() would repoint the reader for good
    import ntof_dream_merge.ntof_io as nio
    e = nio.bunch_edges(NTOF_RUN, 'WALA')
    good = np.flatnonzero(np.diff(e) > 0) + 1
    bl = good[len(good) // 3:len(good) // 3 + args.nb]
    print(f'run {NTOF_RUN}, bunches {bl[0]}-{bl[-1]}, '
          f'modal t_top - t_bottom per segment [ns]\n')
    off = measure(bl)
    print('  OFFICIAL processing')
    for a in ARMS:
        print(f'    wall {a}: ' + '  '.join(f'{off[a][g]:+6.1f}' for g in range(4)))

    from study_common import use_variant
    use_variant(V12)
    off12 = measure(bl)
    print('\n  v12_liqpileup reprocessing (same bunches, same estimator)')
    for a in ARMS:
        print(f'    wall {a}: ' + '  '.join(f'{off12[a][g]:+6.1f}' for g in range(4)))

    o1 = np.array([[off[a][g] for g in range(4)] for a in ARMS])
    o2 = np.array([[off12[a][g] for g in range(4)] for a in ARMS])
    print(f'\n  max |offset|: official {np.abs(o1).max():.1f} ns, '
          f'v12 {np.abs(o2).max():.1f} ns')
    with open(DATA / 'tb_offsets_official_vs_v12.json', 'w') as f:
        json.dump(dict(bunches=[int(bl[0]), int(bl[-1])],
                       official={a: off[a] for a in ARMS},
                       v12={a: off12[a] for a in ARMS}), f, indent=1,
                  default=float)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
