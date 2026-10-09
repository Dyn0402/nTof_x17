#!/usr/bin/env python3
"""headon_split.py -- does the run_71 late-charge loss depend on anything but time?

Same clean stack as headon_stack.py (RAW, masked CM, missing samples NaN,
per-strip pre-trigger baseline), X summed over +-2 (already converged), Y over
+-8 (where its RC spread is contained).  Events are split into classes by
quantities that attachment does not care about, and the late/early ratio
R = level(2.4-2.7 us) / level(1.08-1.26 us) is compared across classes:

  charge   total X+-2 charge over the window (time-symmetric: picks events with
           more ionisation, not events with early or late clusters).  Space
           charge / gain saturation would make bright events lose more.
  rate     triggers within +-0.25 s of the event (beam intensity).  Ion space
           charge in the drift volume would grow with rate.
  spill    position within the SPS spill (time since the spill's first trigger).
  posx/posy  beam position across the chamber (det4's own lead strip).  A field
           or geometry defect would be local.

    headon_split.py <cache.npz> [--nboot 100]
Output: results/headon_split.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headon_stack as H                                     # noqa: E402

VIEWS = (('x', 2), ('y', 8))
NCLS = 3


def ratio(C, h):
    M = np.nanmean(C, axis=0)
    M = M - np.nanmean(M[:, H.PRE], axis=1, keepdims=True)
    S = np.nansum(M[H.KEEP - h:H.KEEP + h + 1], axis=0)
    t = np.arange(H.NSMP) * H.SNS
    return S[(t >= 2400) & (t < 2700)].mean() / S[(t >= 1080) & (t <= 1260)].mean()


def spill_phase(tw):
    """seconds since the first trigger of the spill (gap > 2 s starts a new spill)."""
    o = np.argsort(tw); ts = tw[o]
    start = np.r_[True, np.diff(ts) > 2.0]
    sid = np.cumsum(start) - 1
    t0 = ts[start][sid]
    ph = np.empty_like(tw); ph[o] = ts - t0
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache')
    ap.add_argument('--nboot', type=int, default=100)
    a = ap.parse_args()
    ev, ch, s, amp, lead, plat, tw = H.load(a.cache)
    rng = np.random.default_rng(7)
    res = {}
    labs = [l for l, _ in H.PLATEAUS]
    pcode = np.full(len(plat), -1, np.int8)
    for i, l in enumerate(labs):
        pcode[plat == l] = i
    for i, (lab, E) in enumerate(H.PLATEAUS):
        events = np.unique(ev[pcode[ev] == i])
        cubes = {v: H.cube(ev, ch, s, amp, lead, events, v) for v, _ in VIEWS}
        twe = tw[events]
        ok_t = np.isfinite(twe)
        # rate: triggers of the SAME plateau within +-0.25 s (only extracted events are
        # visible here, a fixed fraction of all triggers, so it is a relative rate)
        ts = np.sort(twe[ok_t])
        rate = np.full(len(events), np.nan)
        rate[ok_t] = (np.searchsorted(ts, twe[ok_t] + 0.25) - np.searchsorted(ts, twe[ok_t] - 0.25))
        ph = np.full(len(events), np.nan); ph[ok_t] = spill_phase(twe[ok_t])
        qx = np.nansum(cubes['x'][:, H.KEEP - 2:H.KEEP + 3, 12:58], axis=(1, 2))
        keys = {'charge': qx, 'rate': rate, 'spill': ph,
                'posx': lead['x'][events], 'posy': lead['y'][events]}
        r = {'E_Vcm': E, 'n': int(len(events))}
        print(f'== {lab} ({E} V/cm), n = {len(events)};  R = level(2.4-2.7 us) / level(1.08-1.26 us)')
        for key, val in keys.items():
            good = np.isfinite(val)
            if key == 'charge':                       # drop the faint and the giant 5 %
                lo, hi = np.quantile(val[good], [0.05, 0.95]); good &= (val > lo) & (val < hi)
            edges = np.quantile(val[good], np.linspace(0, 1, NCLS + 1))
            rows = []
            for c in range(NCLS):
                m = good & (val >= edges[c]) & (val <= edges[c + 1])
                row = {'range': [float(edges[c]), float(edges[c + 1])], 'n': int(m.sum())}
                for v, h in VIEWS:
                    Cm = cubes[v][m]
                    rv = ratio(Cm, h)
                    bs = [ratio(Cm[rng.integers(0, len(Cm), len(Cm))], h) for _ in range(a.nboot)]
                    row[v] = [float(rv), float(np.std(bs))]
                rows.append(row)
            r[key] = rows
            print(f'  {key:6s} ' + ' | '.join(
                f'[{row["range"][0]:.4g},{row["range"][1]:.4g}] X {row["x"][0]:.3f}±{row["x"][1]:.3f} '
                f'Y {row["y"][0]:.3f}±{row["y"][1]:.3f}' for row in rows))
        res[lab] = r
        del cubes
    json.dump(res, open(os.path.join(HERE, 'results', 'headon_split.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
