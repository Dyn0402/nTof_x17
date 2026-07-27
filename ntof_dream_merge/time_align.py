#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
time_align.py -- verify (and if needed correct) the n_TOF internal timing before
using n_TOF hit times as a reference for anything else.

WHY THIS EXISTS. mx_july_beam_qa/calib/time_offsets_run*.json records a real
wall-vs-plastic delay for the mid-July runs: t_wall - t_pss is -25 to -40 ns over
runs 224404-224489, structured per channel pair. That is the same size as the
DREAM<->n_TOF match resolution, so it would be a leading systematic if it were
still there -- and applying a stale one would be just as bad as ignoring a live
one, since it would inject a 30 ns shift that is not in the data.

WHAT THE DATA SAYS FOR run224572. Measured in situ from wall/plastic coincidences
(the same estimator the calib files use: nearest-plastic dt per wall hit, sampled
at tof-tflash > 100 us to stay out of the flash):

    station A -0.5 ns   B +0.5 ns   C -0.5 ns   D +0.5 ns     (sigma ~13.5 ns)
    per-channel spread over all 32 wall channels: RMS 1.2 ns, range -3.0..+4.3

i.e. by run224572 the two subsystems are aligned to well under a nanosecond and
the per-channel structure is gone. The stored -32 ns offsets from run224489 are
NOT applicable here -- the mean shifted by +33 ns between the two runs, while the
channel-to-channel shape of the old files is stable to ~1 ns. So: measure per run,
do not carry the calibration across the recalibration boundary.

CONSEQUENCE FOR THE MERGE. The 37 ns width of the DREAM<->n_TOF peak is therefore
not detector misalignment. Budget: a wall-plastic pair resolves to sigma 13.5 ns,
so one n_TOF detector is ~9.5 ns, leaving sqrt(37^2 - 9.5^2) ~ 36 ns on the DREAM
side -- its trigger timestamp is 10 ns granular and its trigger latency jitters.
Sharpening the match further is a DREAM-side problem, not an n_TOF one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ntof_dream_merge.ntof_io import read_bunches   # noqa: E402

LATE_NS = 100_000.0     # sample well past the flash, where singles dominate
MAX_DT_NS = 60.0        # coincidence window for the offset estimator
WARN_NS = 5.0           # above this, the offsets are worth correcting for


def _nearest_dt(wt, wb, pt, pb, sel=None):
    ptab = {b: np.sort(pt[pb == b]) for b in np.unique(pb)}
    out = []
    idx = np.ones(wt.size, bool) if sel is None else sel
    for b in np.unique(wb[idx]):
        a = np.sort(wt[idx & (wb == b)])
        c = ptab.get(b)
        if c is None or c.size == 0 or a.size == 0:
            continue
        j = np.searchsorted(c, a)
        j0 = np.clip(j - 1, 0, c.size - 1)
        j1 = np.clip(j, 0, c.size - 1)
        d0, d1 = a - c[j0], a - c[j1]
        out.append(np.where(np.abs(d0) <= np.abs(d1), d0, d1))
    return np.concatenate(out) if out else np.array([])


def measure(ntof_run: int, bunches, arms='ABCD') -> dict:
    """
    Per-station and per-wall-channel wall-vs-plastic offsets, measured in situ.

    Returns {'station': {A: ns, ...}, 'channel': {(A,1): ns, ...}, 'sigma': {...}}.
    Offsets are t_wall - t_pss, so subtracting them from the wall time aligns the
    two subsystems.
    """
    station, channel, sigma = {}, {}, {}
    for st in arms:
        w = read_bunches(ntof_run, f'WAL{st}', bunches, branches=('BunchNumber', 'detn'))
        p = read_bunches(ntof_run, f'PSS{st}', bunches, branches=('BunchNumber', 'detn'))
        mw = w['t_since_flash_ns'] > LATE_NS
        mp = p['t_since_flash_ns'] > LATE_NS
        wb, wt, wd = w['BunchNumber'][mw], w['t_since_flash_ns'][mw], w['detn'][mw]
        pb, pt = p['BunchNumber'][mp], p['t_since_flash_ns'][mp]

        d = _nearest_dt(wt, wb, pt, pb)
        d = d[np.abs(d) < 500]
        if d.size:
            h, e = np.histogram(d, bins=1000, range=(-500, 500))
            c = 0.5 * (e[1:] + e[:-1])
            pk = float(c[h.argmax()])
            station[st] = pk
            sigma[st] = float(d[np.abs(d - pk) < 30].std())
        for wc in range(1, 9):
            dc = _nearest_dt(wt, wb, pt, pb, sel=(wd == wc))
            dc = dc[np.abs(dc) < MAX_DT_NS]
            if dc.size > 200:
                channel[(st, wc)] = float(np.median(dc))
    return dict(station=station, channel=channel, sigma=sigma)


def report(m: dict) -> bool:
    """Print the measurement; return True if any correction exceeds WARN_NS."""
    print('n_TOF internal timing (t_wall - t_pss, in situ):')
    for st, v in m['station'].items():
        print(f'  station {st}: {v:+5.1f} ns   (sigma {m["sigma"][st]:.1f} ns)')
    ch = np.array(list(m['channel'].values()))
    if ch.size:
        print(f'  per-wall-channel: RMS {ch.std():.1f} ns, '
              f'range {ch.min():+.1f} .. {ch.max():+.1f} ns')
    big = max(abs(v) for v in m['station'].values()) if m['station'] else 0.0
    big = max(big, float(np.abs(ch).max()) if ch.size else 0.0)
    if big > WARN_NS:
        print(f'  -> {big:.1f} ns exceeds {WARN_NS:g} ns: CORRECT before matching.')
    else:
        print(f'  -> all within {WARN_NS:g} ns: already aligned, no correction needed.')
    return big > WARN_NS


if __name__ == '__main__':
    from ntof_dream_merge.bunch_join import dream_event_to_bunch

    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    print(f'run{nt}, {len(bunches)} bunches from {run}/{sub}\n')
    report(measure(nt, bunches))
