#!/usr/bin/env python3
"""Does the PSA `satuflag` fire on the pulses that actually clip?

Takes the clipped-block table written by `saturation_examples.py`
(`sat_blocks_224572.txt`: det, segment, bunch, trigger, time, runs, widest,
region) and looks in the reprocessed trees for a flagged hit at that place.

Matching: the reprocessed output is split into parts of 10 raw segments each
(part = seg//10 + 1), and the `segment`/`BunchNumber` branches carry the raw
numbering, so a clipped block is located exactly. Within it the only freedom is
time, and the raw sample index and the PSA `tof` agree to well under a
microsecond, so a +-1 us window is generous.

    python check_satuflag.py <sat_blocks.txt> <reproc_dir>
"""
import sys
from pathlib import Path

import numpy as np
import uproot

WIN = 1000.0            # ns around the raw clip time
BR = ['segment', 'BunchNumber', 'tof', 'amp', 'satuflag', 'chi2']


def load(reproc, part, det, cache):
    key = (part, det)
    if key not in cache:
        p = Path(reproc) / f'run224572_{part:04d}.root'
        if not p.exists():
            cache[key] = None
        else:
            cache[key] = uproot.open(p)[det].arrays(BR, library='np')
    return cache[key]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    rows = []
    for line in Path(sys.argv[1]).read_text().splitlines():
        f = line.split()
        if len(f) < 8 or not f[0].startswith('LIQ'):
            continue
        rows.append(dict(det=f[0], seg=int(f[1]), bunch=int(f[2]), trig=int(f[3]),
                         t_ms=float(f[4]), runs=int(f[5]), widest=int(f[6]),
                         region=f[7]))

    cache = {}
    print(f'{"det":5s} {"seg":>4s} {"bunch":>6s} {"t [ms]":>9s} {"region":>8s} '
          f'{"widest":>7s} | {"hits in window":>14s} {"flagged":>8s} '
          f'{"best amp":>10s} {"verdict":>9s}')
    hit, miss, nofile = 0, 0, 0
    for r in rows:
        part = r['seg'] // 10 + 1
        a = load(sys.argv[2], part, r['det'], cache)
        if a is None:
            nofile += 1
            continue
        t = r['t_ms'] * 1e6
        m = ((a['segment'] == r['seg']) & (a['BunchNumber'] == r['bunch']) &
             (np.abs(a['tof'] - t) < WIN))
        n = int(m.sum())
        nf = int((m & (a['satuflag'] != 0)).sum())
        amp = float(np.nanmax(a['amp'][m])) if n else float('nan')
        ok = nf > 0
        hit += ok
        miss += (not ok)
        print(f'{r["det"]:5s} {r["seg"]:4d} {r["bunch"]:6d} {r["t_ms"]:9.3f} '
              f'{r["region"]:>8s} {r["widest"]:7d} | {n:14d} {nf:8d} '
              f'{amp:10.0f} {"FLAGGED" if ok else "MISSED":>9s}')
    print(f'\n{hit} of {hit + miss} clipped blocks have a flagged hit '
          f'({nofile} skipped, no local part file)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
