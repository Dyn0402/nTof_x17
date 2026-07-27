#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intra_burst_align.py -- per-event DREAM <-> n_TOF alignment inside a burst.

bunch_join.py settles WHICH beam pulse a DREAM burst belongs to. This settles
WHICH scintillator pulse inside that ~73 ms burst fired each DREAM trigger, which
is what PLAN.md Phase 4 calls "the real work".

The handle is the trigger itself. DREAM runs on PS + SINGLES, and a SINGLES is the
per-sector coincidence wall .AND. plastic (M3 of the N1081B chain -- see
mx_july_beam_qa/30_trigger_emulation.py). Every DREAM event was therefore caused by
a scintillator coincidence that the n_TOF DAQ *also* digitised, so rebuilding those
coincidences on the n_TOF side gives a list of trigger candidates to match against.
Matching raw hits instead drowns in accidentals: PSSC alone has ~46 k hits/bunch.

WHAT MAKES IT WORK -- the two clocks run at different rates.

Matching candidates to events directly gives a one-sided excess smeared over
0 to +10 us: real, causal (nothing at negative dt), but useless for 1:1 assignment.
That smear is not jitter, it is a constant RATE mismatch between the DREAM and
n_TOF clocks. Both sides measure time from the same gamma flash, so a fractional
rate error k shows up as a lag growing linearly across the burst:

    dt(t) = t_nTOF - t_DREAM = t0 + k * t_since_flash,   k ~ 1.09e-4  (109 ppm)

109 ppm is an ordinary free-running-crystal difference, and over a 73 ms burst it
integrates to ~8 us -- which is exactly the observed smear. Removing it collapses
the excess into a sharp line:

    main peak   sigma 33 ns,  3312 excess pairs vs 120 accidentals  (96 % pure)
    satellite   +330 ns,      1133 excess pairs (~1/3 of the main peak)

measured on 100 bunches of run_79/stat090_0000 <-> 224572, all four arms summed.
The +330 ns satellite is unexplained -- see OPEN QUESTIONS at the bottom.

So per-event alignment IS possible, and to ~30 ns rather than the microseconds
PLAN.md feared. The tt_dream_match order-based fallback is not needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ntof_dream_merge.ntof_io import read_bunches            # noqa: E402

ARMS = ('A', 'B', 'C', 'D')

COINC_NS = 20.0        # wall/plastic coincidence window; the hardware gate is a
                       # 20 ns input gate&delay on both legs
SEARCH_NS = 40_000.0   # dt search half-window before the clock correction is known
CORE_NS = 100.0        # half-width of the accept window around the main peak
SAT_NS = (240.0, 420.0)   # the +330 ns satellite


def trigger_candidates(ntof_run: int, bunches, arm: str,
                       coinc_ns: float = COINC_NS) -> tuple[np.ndarray, np.ndarray]:
    """
    n_TOF-side SINGLES candidates for one arm: wall hit with a plastic hit within
    `coinc_ns`. Returns (BunchNumber, t_since_flash_ns) with the WALL hit's time.

    This is deliberately looser than the hardware: no discriminator thresholds and
    no per-segment analog sum, so it over-produces candidates (~3-4 k/bunch vs the
    ~100/bunch the thresholded chain would give). That costs accidentals but cannot
    lose a real trigger, which is the right trade while the alignment is being
    established. Tightening it is the obvious lever for improving purity later.
    """
    w = read_bunches(ntof_run, f'WAL{arm}', bunches, branches=('BunchNumber',))
    p = read_bunches(ntof_run, f'PSS{arm}', bunches, branches=('BunchNumber',))
    wb, wt = w['BunchNumber'], w['t_since_flash_ns']
    pb, pt = p['BunchNumber'], p['t_since_flash_ns']
    ow = np.lexsort((wt, wb)); wb, wt = wb[ow], wt[ow]
    op = np.lexsort((pt, pb)); pb, pt = pb[op], pt[op]

    cb, ct = [], []
    for b in np.unique(wb):
        ws = wt[wb == b]
        ps = pt[pb == b]
        if ws.size == 0 or ps.size == 0:
            continue
        j = np.searchsorted(ps, ws)
        j0 = np.clip(j - 1, 0, ps.size - 1)
        j1 = np.clip(j, 0, ps.size - 1)
        d = np.minimum(np.abs(ps[j0] - ws), np.abs(ps[j1] - ws))
        m = d <= coinc_ns
        cb.append(np.full(int(m.sum()), b))
        ct.append(ws[m])
    if not cb:
        return np.array([]), np.array([])
    return np.concatenate(cb), np.concatenate(ct)


def dt_pairs(events: pd.DataFrame, cand_bunch, cand_t,
             window_ns: float = SEARCH_NS):
    """
    All (event t_since_flash, dt) pairs within +-window_ns.

    Returns flat arrays so the clock fit sees every pair, not just the nearest --
    the nearest-hit choice biases the pedestal and hides the satellite.
    """
    ET, DT, EV = [], [], []
    order = np.lexsort((cand_t, cand_bunch))
    cb, ct = cand_bunch[order], cand_t[order]
    for b, g in events.groupby('BunchNumber'):
        s, e = np.searchsorted(cb, [b, b + 1])
        cc = ct[s:e]
        if cc.size == 0:
            continue
        et = g['t_since_flash_ns'].to_numpy().astype(float)
        eid = g['eventId'].to_numpy()
        lo = np.searchsorted(cc, et - window_ns)
        hi = np.searchsorted(cc, et + window_ns)
        for k in range(et.size):
            if hi[k] > lo[k]:
                DT.append(cc[lo[k]:hi[k]] - et[k])
                ET.append(np.full(hi[k] - lo[k], et[k]))
                EV.append(np.full(hi[k] - lo[k], eid[k]))
    if not DT:
        return (np.array([]),) * 3
    return np.concatenate(ET), np.concatenate(DT), np.concatenate(EV)


def fit_clock(ET, DT, k_range=(0.5e-4, 1.6e-4), k_step=1e-7,
              core_ns: float = CORE_NS) -> tuple[float, float]:
    """
    Fit (k, t0) in dt = t0 + k * t_since_flash by maximising the number of pairs
    inside a +-core_ns window -- a 1-D scan over k with t0 read off as the mode, so
    there is no seed to get wrong and no local-minimum risk.
    """
    best = (0.0, 0.0, -1)
    nb = int(4000 / 5)
    for k in np.arange(*k_range, k_step):
        r = DT - k * ET
        h, e = np.histogram(r, bins=nb, range=(-2000, 2000))
        c = 0.5 * (e[1:] + e[:-1])
        sm = np.convolve(h, np.ones(int(2 * core_ns / 5)), mode='same')
        i = int(sm.argmax())
        if sm[i] > best[2]:
            best = (float(k), float(c[i]), float(sm[i]))
    return best[0], best[1]


def align(ntof_run: int, events: pd.DataFrame, bunches=None,
          k: float | None = None, t0: float | None = None) -> dict:
    """
    Full Phase-4 pass over a set of bunches: build candidates for all four arms,
    pair them with the DREAM events, fit the clock and report the peak.
    """
    if bunches is None:
        bunches = np.sort(events.loc[events['BunchNumber'] > 0, 'BunchNumber'].unique())
    ev = events[(events['BunchNumber'].isin(bunches)) & (~events['is_flash'])]

    ET, DT, EV, ARM = [], [], [], []
    per_arm = {}
    for arm in ARMS:
        cb, ct = trigger_candidates(ntof_run, bunches, arm)
        per_arm[arm] = len(ct) / max(len(bunches), 1)
        et, dt, eid = dt_pairs(ev, cb, ct)
        ET.append(et); DT.append(dt); EV.append(eid)
        ARM.append(np.full(dt.size, arm))
    ET = np.concatenate(ET); DT = np.concatenate(DT)
    EV = np.concatenate(EV); ARM = np.concatenate(ARM)

    if k is None or t0 is None:
        k, t0 = fit_clock(ET, DT)
    resid = DT - k * ET - t0

    h, e = np.histogram(resid, bins=200, range=(-1000, 3000))
    c = 0.5 * (e[1:] + e[:-1])
    ped_per_ns = np.median(h[c > 2000]) / 20.0

    def band(lo, hi):
        n = int(((resid >= lo) & (resid < hi)).sum())
        acc = ped_per_ns * (hi - lo)
        return n, acc, n - acc

    n_main, acc_main, exc_main = band(-CORE_NS, CORE_NS)
    n_sat, acc_sat, exc_sat = band(*SAT_NS)
    core = np.abs(resid) < CORE_NS

    # Two different denominators, easy to confuse: `exc_main` counts PAIRS (one
    # DREAM event can pair with candidates in several arms, so pairs > events),
    # while `n_events_matched` counts DISTINCT DREAM events with at least one
    # candidate in the core window. The efficiency quoted anywhere must be the
    # latter over n_events.
    n_events_matched = int(np.unique(EV[core]).size) if core.any() else 0

    return dict(k=k, t0=t0, n_events=len(ev), n_bunches=len(bunches),
                cand_per_bunch=per_arm, ET=ET, DT=DT, resid=resid,
                eventId=EV, arm=ARM,
                ped_per_ns=ped_per_ns,
                main=(n_main, acc_main, exc_main),
                satellite=(n_sat, acc_sat, exc_sat),
                n_events_matched=n_events_matched,
                sigma_ns=float(resid[core].std()) if core.any() else np.nan)


if __name__ == '__main__':
    from ntof_dream_merge.bunch_join import dream_event_to_bunch

    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    r = align(nt, ev, bunches)

    print(f'{run}/{sub} <-> {nt}   {r["n_bunches"]} bunches, {r["n_events"]:,} events')
    print('  candidates/bunch: ' +
          '  '.join(f'{a}={v:.0f}' for a, v in r['cand_per_bunch'].items()))
    print(f'  clock fit: k = {r["k"]*1e6:.2f} ppm   t0 = {r["t0"]:+.1f} ns')
    n, a, x = r['main']
    print(f'  main peak |dt| < {CORE_NS:.0f} ns: {n} pairs, {a:.0f} accidental, '
          f'{x:.0f} real ({x/n:.1%} pure), sigma {r["sigma_ns"]:.0f} ns')
    print(f'    -> {r["n_events_matched"]:,} distinct DREAM events matched '
          f'({r["n_events_matched"]/r["n_events"]:.1%} of {r["n_events"]:,})')
    n, a, x = r['satellite']
    print(f'  satellite {SAT_NS[0]:.0f}-{SAT_NS[1]:.0f} ns: {x:.0f} real '
          f'({x/max(r["main"][2],1):.2f} x the main peak)')
