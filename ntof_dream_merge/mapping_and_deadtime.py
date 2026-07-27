#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mapping_and_deadtime.py -- two checks on why n_TOF records so few plastic hits.

Geometry (MX17_Full_Geant/src/DetectorConstruction.cc:698-702): behind each SiPM
wall sit TWO wrapped plastic bars side by side, placed at -uOff and +uOff --
"BackTapeL" and "BackTapeR". So PSS<arm> detn 1/2 are the Left/Right bar of that
arm, not two readouts of one bar.

CHECK 1 -- IS THE WALL<->PLASTIC MAPPING RIGHT? Coincidence excess for all 4x4
combinations of wall arm and plastic tree, late hits, +-60 ns of the peak:

              PSSA   PSSB   PSSC   PSSD
      WALA     508     16     44     37
      WALB      20    324     36     13
      WALC      17     11    603     10
      WALD      30     20     28    742

Strongly diagonal -- on-diagonal 324-742 against off-diagonal 10-44. The nominal
tree-to-tree mapping is correct and mismapping is excluded.

But the per-bar breakdown found a bug of OURS. Same-arm excess by detn:

      WALA  detn1   90   detn2  422        WALC  detn1  314   detn2  321
      WALB  detn1  230   detn2   94        WALD  detn1  615   detn2  133

Both bars carry real coincidences, and which one dominates just follows where the
flux crosses that arm. dream_trigger.D_PMTS excludes PSSD1 because the D-L input
to the N1081B is broken -- true, and correct when EMULATING the trigger, but wrong
when looking for a coincident plastic hit in the n_TOF data, where the digitiser
records both bars regardless. PSSD1 is in fact the STRONGER partner of WALD.
Using both bars everywhere raises the plastic match rate 41.1 % -> 52.0 %.

CHECK 2 -- IS THE PSA MERGING CLOSE PULSES? Gap between consecutive hits in the
same channel, per bunch:

      PSSA  n=424 k  min  5.9 ns  p0.1  8.2 ns  median 208 ns
      PSSC  n=771 k  min  4.9 ns  p0.1  7.8 ns  median 105 ns
      WALA  n=226 k  min  1.1 ns  p0.1 32.3 ns  median 13.5 us
      WALC  n=233 k  min  6.5 ns  p0.1 36.0 ns  median 11.6 us

No truncation: the plastic PSA resolves pulses 5-6 ns apart, and the gap
distribution rises monotonically into the smallest bin rather than being cut off.
Dead time / double-pulse resolution is therefore NOT why plastic hits are missing.

RESOLVED (2026-07-28). Neither geometry nor the trigger: the official PSS tflash
is wrong in 37-85 % of bunches, shifting t_since_flash of those (tree, bunch)
combinations by up to 11.6 us and moving the true partner out of the accept
bands. Repairing the time base (tflash_repair.py, applied by ntof_io by default)
lifts the plastic partner fraction to 99.7 %. See
FINDINGS_2026-07-28_pss_tflash.md. The checks in this module remain valid -- they
were run on nearest-dt peaks, which the sane-tflash bunches dominate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ntof_dream_merge.ntof_io import read_bunches   # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
LATE_NS = 20e6
PEAK_NS = 60.0


def _nearest_dt(wt, wb, pt, pb):
    ptab = {b: np.sort(pt[pb == b]) for b in np.unique(pb)}
    out = []
    for b in np.unique(wb):
        a = np.sort(wt[wb == b])
        c = ptab.get(b)
        if c is None or c.size == 0 or a.size == 0:
            continue
        j = np.searchsorted(c, a)
        j0, j1 = np.clip(j - 1, 0, c.size - 1), np.clip(j, 0, c.size - 1)
        d0, d1 = a - c[j0], a - c[j1]
        out.append(np.where(np.abs(d0) <= np.abs(d1), d0, d1))
    return np.concatenate(out) if out else np.array([])


def _excess(d):
    d = d[np.abs(d) < 500]
    if d.size == 0:
        return 0.0
    h, e = np.histogram(d, bins=250, range=(-500, 500))
    c = 0.5 * (e[1:] + e[:-1])
    ped = np.median(h[np.abs(c) > 300])
    core = np.abs(c) < PEAK_NS
    return float((h[core] - ped).sum())


def mapping_matrix(ntof_run, bunches):
    W, P = {}, {}
    for a in ARMS:
        h = read_bunches(ntof_run, f'WAL{a}', bunches, branches=('BunchNumber', 'detn'))
        m = h['t_since_flash_ns'] > LATE_NS
        W[a] = (h['BunchNumber'][m], h['t_since_flash_ns'][m])
        h = read_bunches(ntof_run, f'PSS{a}', bunches, branches=('BunchNumber', 'detn'))
        m = h['t_since_flash_ns'] > LATE_NS
        P[a] = (h['BunchNumber'][m], h['t_since_flash_ns'][m], h['detn'][m])
    mat = {(w, p): _excess(_nearest_dt(W[w][1], W[w][0], P[p][1], P[p][0]))
           for w in ARMS for p in ARMS}
    per_bar = {}
    for w in ARMS:
        for dn in (1, 2):
            s = P[w][2] == dn
            per_bar[(w, dn)] = (_excess(_nearest_dt(W[w][1], W[w][0],
                                                    P[w][1][s], P[w][0][s])), int(s.sum()))
    return mat, per_bar


def gap_distribution(ntof_run, bunches, tree):
    h = read_bunches(ntof_run, tree, bunches, branches=('BunchNumber', 'detn'))
    m = h['t_since_flash_ns'] > 1e6
    b, t, d = h['BunchNumber'][m], h['t_since_flash_ns'][m], h['detn'][m]
    gaps = []
    for dn in np.unique(d):
        for bb in np.unique(b):
            s = (d == dn) & (b == bb)
            if s.sum() < 3:
                continue
            gaps.append(np.diff(np.sort(t[s])))
    return np.concatenate(gaps) if gaps else np.array([])


if __name__ == '__main__':
    from ntof_dream_merge.bunch_join import dream_event_to_bunch

    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]

    mat, per_bar = mapping_matrix(nt, bunches)
    print('wall x plastic coincidence excess (diagonal = nominal mapping):')
    print('          ' + '  '.join(f'{"PSS"+p:>8}' for p in ARMS))
    for w in ARMS:
        print(f'   WAL{w}: ' + '  '.join(f'{mat[(w, p)]:8.0f}' for p in ARMS))
    print('\nper plastic bar (same arm):')
    for w in ARMS:
        print(f'   WAL{w}: ' + '  '.join(
            f'detn{d}={per_bar[(w, d)][0]:7.0f} (n={per_bar[(w, d)][1]:6d})'
            for d in (1, 2)))

    print('\ngap between consecutive hits in a channel:')
    for tree in ('PSSA', 'PSSC', 'WALA', 'WALC'):
        g = gap_distribution(nt, bunches, tree)
        print(f'   {tree}: n={g.size:8,}  min {g.min():7.1f}  p0.1 {np.percentile(g, 0.1):7.1f}'
              f'  p1 {np.percentile(g, 1):7.1f}  median {np.median(g):10.1f} ns')
