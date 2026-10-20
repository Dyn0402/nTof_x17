#!/usr/bin/env python3
"""
trigger_cand.py -- does the hardware trigger see the wall undershoot?  Emulated
trigger candidates in the flash_tail waveforms, checked against the DREAM
triggers actually recorded in the same bunch.

    python -m ntof_scint_recovery.trigger_cand <run> --products <flash_*.root> [...] \
        --slims <ntof_hits_*_<run>.root> [...] --out trigcand_<run>.npz

Candidate (per arm, the hardware condition of ntof_scint_stack.extract): the two
ends of a wall bar within +-30 ns, their amplitude SUM >= 0.6 x WALL_THR[arm],
and the larger plastic bar of the arm >= 0.6 x PLAS_THR[arm] within -60..+100 ns
of the wall pair.  Amplitudes are the template-fit amplitudes over the LOCAL
baseline, i.e. what the pulse would be with no undershoot; if the analog trigger
path shares the undershoot, the effective sum it sees is lower, and the
efficiency curve against wall_sum / threshold shifts right while the baseline is
down (10 us - 1 ms), and not later.

Per candidate: t_us (since the flash at the wall), arm, wall_sum, plas, the
nearest DREAM trigger's dt (t_pred - t, ns), the time since the previous DREAM
trigger of that bunch (DREAM is dead ~205 us after each), and the bunch's
earliest trigger time.  Matching and the live-time cut are left to the analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

from ntof_scint_recovery.flash_summary import C_FALLBACK
from ntof_scint_recovery.pulses import pulses_file

WALL_THR = {0: 26.0, 1: 35.0, 2: 38.0, 3: 37.0}     # mV, ntof_scint_stack.extract (A B C D)
PLAS_THR = {0: 112.0, 1: 136.0, 2: 151.0, 3: 130.0}
LOOSE = 0.6
T_MIN_US = 100.0


def candidates(P: dict) -> dict:
    amp = P['amp_fit'].copy()
    bad = ~np.isfinite(amp)
    amp[bad] = P['amp_mV'][bad]
    out = {k: [] for k in ('bunch', 'arm', 't_us', 'wall_sum', 'plas', 'ppp')}
    late = P['t_us'] > T_MIN_US
    for b in np.unique(P['bunch']):
        mb = late & (P['bunch'] == b)
        for arm in range(4):
            pl = mb & (P['det'] == 4 + arm)
            tp, ap = P['t_us'][pl] * 1e3, amp[pl]
            o = np.argsort(tp)
            tp, ap = tp[o], ap[o]
            for g in range(4):
                e1 = mb & (P['det'] == arm) & (P['detn'] == 2 * g + 1)
                e2 = mb & (P['det'] == arm) & (P['detn'] == 2 * g + 2)
                if not e1.any() or not e2.any():
                    continue
                t1, a1 = P['t_us'][e1] * 1e3, amp[e1]
                o2 = np.argsort(P['t_us'][e2])
                t2, a2 = P['t_us'][e2][o2] * 1e3, amp[e2][o2]
                j = np.clip(np.searchsorted(t2, t1), 1, len(t2) - 1)
                jj = np.where(np.abs(t2[j] - t1) < np.abs(t2[j - 1] - t1), j, j - 1)
                ok = np.abs(t2[jj] - t1) <= 30
                s = a1 + a2[jj]
                ok &= s >= LOOSE * WALL_THR[arm]
                if not ok.any() or not len(tp):
                    continue
                tw = 0.5 * (t1 + t2[jj])[ok]
                sw = s[ok]
                lo = np.searchsorted(tp, tw - 60)
                hi = np.searchsorted(tp, tw + 100)
                pmax = np.array([ap[x:y].max() if y > x else 0.0 for x, y in zip(lo, hi)])
                k = pmax >= LOOSE * PLAS_THR[arm]
                n = int(k.sum())
                out['bunch'] += [int(b)] * n
                out['arm'] += [arm] * n
                out['t_us'] += list(tw[k] / 1e3)
                out['wall_sum'] += list(sw[k])
                out['plas'] += list(pmax[k])
                out['ppp'] += [float(P['ppp'][P['bunch'] == b][0])] * n
    return {k: np.array(v) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run', type=int)
    ap.add_argument('--products', nargs='+', required=True)
    ap.add_argument('--slims', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    ev = {k: [] for k in ('bunch', 't_pred_ns', 'arm', 'is_flash')}
    for s in a.slims:
        e = uproot.open(s)['events'].arrays(list(ev), library='np')
        for k in ev:
            ev[k].append(e[k])
    ev = {k: np.concatenate(v) for k, v in ev.items()}
    trig = {}
    for b in np.unique(ev['bunch']):
        m = (ev['bunch'] == b) & (ev['is_flash'] == 0) & np.isfinite(ev['t_pred_ns'])
        trig[int(b)] = (np.sort(ev['t_pred_ns'][m]), ev['arm'][m][np.argsort(ev['t_pred_ns'][m])])
    parts = []
    for p in a.products:
        P = pulses_file(p, C_FALLBACK)
        parts.append(candidates(P))
    Cd = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    n = len(Cd['t_us'])
    dt = np.full(n, np.nan)
    tarm = np.full(n, -9)
    prev = np.full(n, np.nan)
    first = np.full(n, np.nan)
    has = np.zeros(n, bool)
    for i in range(n):
        b = int(Cd['bunch'][i])
        if b not in trig:
            continue
        t, arms = trig[b]
        has[i] = True
        if not len(t):
            continue
        tc = Cd['t_us'][i] * 1e3
        j = np.searchsorted(t, tc)
        cand = [x for x in (j - 1, j) if 0 <= x < len(t)]
        k = min(cand, key=lambda x: abs(t[x] - tc))
        dt[i] = t[k] - tc
        tarm[i] = arms[k]
        # previous trigger strictly before the candidate (minus a 1 us match margin)
        jp = np.searchsorted(t, tc - 1000) - 1
        prev[i] = (tc - t[jp]) / 1e3 if jp >= 0 else np.inf
        first[i] = t[0] / 1e3
    np.savez_compressed(a.out, run=a.run, has_slim=has, dt_ns=dt, trig_arm=tarm, prev_us=prev, first_us=first, **Cd)
    print(f'run {a.run}: {n} candidates, {has.sum()} in bunches with a slim, '
          f'{len(trig)} slim bunches -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
