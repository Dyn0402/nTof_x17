#!/usr/bin/env python3
"""A per-hit after-pulse flag for the plastics, and the scan that sets it.

The rule, in one sentence: **a hit is an after-pulse candidate if a much larger
hit landed on the same channel a short time before it.**

    hit i is flagged  <=>  exists j on the same (bunch, channel) with
                           0 < t_i - t_j <= T_HOLD  and  amp_i < RATIO * amp_j

That is the whole metric. It needs no DREAM information, no template and no
fitting; it is computable on the n_TOF hit stream alone, which means it can be
applied inside the slim's pass 2 where the full per-bunch stream is in hand.

Two properties of the effect make this the right shape (see report.html):

  * after-pulses are strictly FORWARD in time from a larger pulse -- 4.13 excess
    hits per leader forward against 0.90 backward;
  * their amplitude is ~120 ADC almost independently of the leader, i.e. a few
    per cent of it, so an amplitude RATIO separates them from genuine hits far
    better than a flat dead time does.

The flag must be computed with a full T_HOLD of LOOKBACK, on the unwindowed hit
stream. An after-pulse whose parent sits just outside the slim window is exactly
the case a slim-only recomputation gets wrong.

    python afterpulse_flag.py --parts 1            # the (T_HOLD, RATIO) scan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

REPROC = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
T_PHYS_LO, T_PHYS_HI = 20_000.0, 19_000_000.0

# Defaults chosen by the scan in this file; see report_veto.html.
T_HOLD_NS = 1000.0
RATIO = 0.20


def prev_max_amp(grp, tof, amp, t_holds, already_sorted=False):
    """Largest amplitude among preceding hits on the same channel, per lookback.

    Returns an array of shape (len(t_holds), n) in the input order. The flag for
    any ratio is then simply `amp < ratio * prev_max_amp[i]`, so a whole
    (T_HOLD, RATIO) scan costs one pass over the data instead of one per point.
    """
    grp = np.asarray(grp)
    tof = np.asarray(tof, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    t_holds = np.asarray(t_holds, dtype=np.float64)
    if already_sorted:
        order = np.arange(tof.size)
        g, t, a = grp, tof, amp
    else:
        order = np.lexsort((tof, grp))
        g, t, a = grp[order], tof[order], amp[order]

    out = np.zeros((t_holds.size, t.size))
    tmax = float(t_holds.max())
    active = np.arange(1, t.size)
    for k in range(1, t.size):
        j = active - k
        keep = j >= 0
        active, j = active[keep], j[keep]
        if active.size == 0:
            break
        dt = t[active] - t[j]
        inwin = (g[j] == g[active]) & (dt <= tmax) & (dt > 0)
        active, j, dt = active[inwin], j[inwin], dt[inwin]
        if active.size == 0:
            break
        for w, th in enumerate(t_holds):
            m = dt <= th
            if m.any():
                np.maximum.at(out[w], active[m], a[j[m]])
    res = np.empty_like(out)
    res[:, order] = out
    return res


def flag_afterpulses(grp, tof, amp, t_hold=T_HOLD_NS, ratio=RATIO,
                     already_sorted=False):
    """Boolean 'this hit is in the shadow of a bigger recent hit on this channel'.

    `grp` is any integer channel key that also separates bunches -- the slim's
    (BunchNumber, detn) packed together. `tof` and `amp` are the hit time in ns
    and the amplitude to compare (use the MEASURED `amp_0`, not the fitted
    `amp`, which runs away on the pile-up the flag exists to catch).

    Returns the flag in the ORDER THE ARRAYS CAME IN, so it can be assigned
    straight back onto a hit table.
    """
    grp = np.asarray(grp)
    tof = np.asarray(tof, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    if already_sorted:
        order = np.arange(tof.size)
        g, t, a = grp, tof, amp
    else:
        order = np.lexsort((tof, grp))
        g, t, a = grp[order], tof[order], amp[order]

    out = np.zeros(t.size, dtype=bool)
    # Walk back one neighbour at a time, pruning as we go: if a hit's k-th
    # predecessor already falls outside the window (or into another channel),
    # its (k+1)-th must too, because times ascend within a group. So the work is
    # proportional to the number of in-window PAIRS -- ~0.7 per hit at the
    # plastics' 720 kHz -- not to depth x length.
    active = np.arange(1, t.size)
    for k in range(1, t.size):
        j = active - k
        keep = j >= 0
        active, j = active[keep], j[keep]
        if active.size == 0:
            break
        inwin = (g[j] == g[active]) & (t[active] - t[j] <= t_hold) \
            & (t[active] - t[j] > 0)
        active, j = active[inwin], j[inwin]
        if active.size == 0:
            break
        out[active[a[active] < ratio * a[j]]] = True
    res = np.empty_like(out)
    res[order] = out
    return res


def scan(det, parts, t_holds, ratios):
    rows = []
    arrs = []
    for p in parts:
        f = REPROC / f'run224572_{p:04d}.root'
        if f.exists():
            with uproot.open(f) as fh:
                arrs.append(fh[det].arrays(
                    ['segment', 'BunchNumber', 'detn', 'tof', 'amp_0'],
                    library='np'))
    a = {k: np.concatenate([x[k] for x in arrs]) for k in arrs[0]}
    phys = (a['tof'] > T_PHYS_LO) & (a['tof'] < T_PHYS_HI)
    grp = ((a['segment'][phys].astype(np.int64) * 100000
            + a['BunchNumber'][phys]) * 100 + a['detn'][phys])
    tof = a['tof'][phys].astype(np.float64)
    amp = a['amp_0'][phys].astype(np.float64)
    order = np.lexsort((tof, grp))
    grp, tof, amp = grp[order], tof[order], amp[order]

    # the correlated excess this is meant to remove, and the uncorrelated
    # population it must not: measured the same way as in afterpulse_spectrum.py
    prev = np.full(tof.size, np.inf)
    prev[1:] = np.where(grp[1:] == grp[:-1], tof[1:] - tof[:-1], np.inf)
    lead = (amp > 3000.0) & (prev > 5000.0)

    for th in t_holds:
        for r in ratios:
            fl = flag_afterpulses(grp, tof, amp, th, r, already_sorted=True)
            rows.append(dict(t_hold=th, ratio=r,
                             flagged_frac=float(fl.mean()),
                             flagged_of_big=float(fl[amp > 1000].mean()),
                             leaders_lost=float(fl[lead].mean())))
            print(f'  T={th:6.0f} ns  R={r:.2f}   flagged {fl.mean() * 100:5.2f} % '
                  f'of all hits, {fl[amp > 1000].mean() * 100:5.2f} % of hits '
                  f'above 1000 ADC, {fl[lead].mean() * 100:5.3f} % of leaders')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', type=int, nargs='+', default=[1])
    ap.add_argument('--dets', nargs='+', default=['PSSB', 'WALA'])
    ap.add_argument('-o', '--out', default='flag_scan.json')
    args = ap.parse_args()
    res = {}
    for det in args.dets:
        print(f'{det}:')
        res[det] = scan(det, args.parts,
                        [200.0, 500.0, 1000.0, 2000.0],
                        [0.05, 0.10, 0.20, 0.35])
    Path(args.out).write_text(json.dumps(res))
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
