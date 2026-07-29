#!/usr/bin/env python3
"""Per-pulse test: is the flagged hit THE clipped pulse, or just a hit nearby?

Reads the ns-precision clip lists from `dump_clips.py` and, for each clipped
run, finds in the reprocessed trees:

  * the nearest hit of that detector in that (segment, bunch), and
  * the nearest hit with `satuflag` set,

reporting both time differences against the raw clip time. A ns-level match is a
per-pulse identification; a match only at the microsecond level is not.

It also measures the accidental rate: how many flagged hits that detector has
per bunch, so the probability of a chance match inside the quoted window can be
read off rather than assumed.

    python verify_satuflag.py <reproc_dir> <clips_*.txt> [...]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import uproot

BR = ['segment', 'BunchNumber', 'tof', 'amp', 'satuflag', 'chi2']
TIGHT = 100.0            # ns; a per-pulse match
WINDOW = 20e6            # ns; acquisition window, for the accidental rate


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    reproc = Path(sys.argv[1])
    rows = []
    for f in sys.argv[2:]:
        for line in Path(f).read_text().splitlines():
            p = line.split()
            if len(p) != 7:
                continue
            rows.append(dict(det=p[0], seg=int(p[1]), bunch=int(p[2]),
                             trig=int(p[3]), t=float(p[4]), n=int(p[5]),
                             region=p[6]))
    rows.sort(key=lambda r: (r['seg'], r['bunch'], r['det'], r['t']))

    cache, flagged_per_bunch = {}, defaultdict(list)
    print(f'{"det":5s} {"seg":>4s} {"bunch":>6s} {"t_clip [ns]":>12s} {"n":>4s} '
          f'{"region":>8s} | {"dt nearest hit":>15s} {"dt nearest FLAG":>16s} '
          f'{"amp of that hit":>16s}')
    res = []
    for r in rows:
        part = r['seg'] // 10 + 1
        key = (part, r['det'])
        if key not in cache:
            p = reproc / f'run224572_{part:04d}.root'
            cache[key] = uproot.open(p)[r['det']].arrays(BR, library='np') \
                if p.exists() else None
        a = cache[key]
        if a is None:
            continue
        m = (a['segment'] == r['seg']) & (a['BunchNumber'] == r['bunch'])
        tof, amp, sat = a['tof'][m], a['amp'][m], a['satuflag'][m]
        if not len(tof):
            continue
        flagged_per_bunch[r['det']].append(int((sat != 0).sum()))
        d_all = tof - r['t']
        i_all = int(np.argmin(np.abs(d_all)))
        fs = np.flatnonzero(sat != 0)
        if len(fs):
            j = fs[int(np.argmin(np.abs(tof[fs] - r['t'])))]
            d_flag, a_flag = float(tof[j] - r['t']), float(amp[j])
        else:
            d_flag, a_flag = float('nan'), float('nan')
        res.append((r, float(d_all[i_all]), d_flag))
        print(f'{r["det"]:5s} {r["seg"]:4d} {r["bunch"]:6d} {r["t"]:12.0f} '
              f'{r["n"]:4d} {r["region"]:>8s} | {d_all[i_all]:15.1f} '
              f'{d_flag:16.1f} {a_flag:16.0f}')

    d_flag = np.array([x[2] for x in res])
    tight = np.abs(d_flag) < TIGHT
    phys = np.array([x[0]['region'] == 'physics' for x in res])
    print(f'\n{len(res)} clipped runs matched against the trees')
    print(f'  flagged hit within {TIGHT:.0f} ns : {int(tight.sum())} / {len(res)}'
          f'   (physics-time: {int((tight & phys).sum())} / {int(phys.sum())})')
    print(f'  |dt| median {np.nanmedian(np.abs(d_flag)):.1f} ns, '
          f'p90 {np.nanpercentile(np.abs(d_flag), 90):.1f} ns, '
          f'max {np.nanmax(np.abs(d_flag)):.1f} ns')
    print('\naccidental rate: flagged hits per (segment,bunch) seen here')
    for det, v in sorted(flagged_per_bunch.items()):
        v = np.array(v)
        rate = v.mean() / WINDOW * 2 * TIGHT
        print(f'  {det}: mean {v.mean():5.2f} flagged hits per bunch '
              f'-> P(chance match in +-{TIGHT:.0f} ns) = {rate:.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
