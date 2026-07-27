#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding-window (boxcar) tracking efficiency vs TIME SINCE FLASH for run_67.

WHY THIS EXISTS (2026-07-25). `analyze_tracks.py` / `detA_2d.py` bin dt into a
handful of HAND-PICKED windows (scan_lib.WINDOW_SETS). That is fine for a 2-D HV
map but it throws away the shape of the post-flash recovery: the interesting
structure sits in the first ~10 ms and a 1-10 ms box smears it into one number.
This module replaces the hand binning along dt ONLY, with a boxcar:

    for each centre c on a regular grid, use every event with |dt - c| <= W/2

Operator's choice (2026-07-25): **LINEAR fixed width, W = 6 ms, step 1 ms**
over the 1-81 ms gate. One number describes the smoothing everywhere, which
makes curves directly comparable between cells; the cost is that the 1-10 ms
knee is resolved no finer than 6 ms. `--width` / `--step` override it.

*** THE POINTS ARE CORRELATED. *** Neighbouring centres are 1 ms apart but each
box is 6 ms wide, so consecutive points share ~5/6 of their events. The error
bars are per-point binomial errors and are NOT independent: a smooth wiggle
spanning fewer than ~6 ms of dt is the smoothing kernel, not a feature. Roughly
width/step = 6 points must pass before a genuinely new measurement exists.

AXES. The cell is (mip, drift, resist, det) and every cell gets its own boxcar
curve vs dt. Drift is kept as a full 4th axis (operator's choice, 2026-07-25) --
NOT pooled -- so nothing is hidden, at the price of ~1/3 the events per point
relative to pooling.

METRIC. Same as the rest of the package, so numbers are comparable:
efficiency = P(3D x/y pair) per recorded trigger, denominator = events the
detector was READ OUT for (never its produced-hits flag -- post-flash blindness
is the inefficiency being measured). `p_trk` (any single-plane track segment) is
carried alongside but is noise-inflated on the bad-M1 dets B/C/D; **Det A is the
reference**. `blind_frac` is reported as an observable, not a cut.

Outputs -> <OUT_BASE>/slide/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import scan_lib as L  # noqa: E402
import stats as ST  # noqa: E402

# operator-chosen boxcar (2026-07-25)
WIDTH_MS = 6.0
STEP_MS = 1.0

# A boxcar point with fewer events than this is dropped rather than drawn with a
# huge error bar: near the gate edges the box hangs off the end of the
# acceptance and the last centres are built from a handful of events.
MIN_N_PER_POINT = 60


def gate_edges(ev, bin_ms=1.0, frac=0.5, k=5, verbose=True):
    """MEASURE the dt acceptance edges instead of trusting the nominal gate.

    scan_lib's nominal gate is 1-81 ms, but run_67 actually stops accepting at
    ~76 ms. Trusting the nominal 81 ms puts the last few boxes half outside the
    acceptance, where the denominator quietly loses events -- which does not
    look like an error, it looks like a noisy efficiency feature at the end of
    the curve.

    The edge is a CLIFF (2536 events in the 75-76 ms bin, 457 in 76-77, 0
    after), sitting on top of a steadily DECLINING rate (the in-gate event rate
    roughly halves between 5 and 75 ms). So the edge must be found against the
    LOCAL level, not a global one: an earlier version compared each bin to
    `frac x median(all bins)` and, because of that decline, amputated the curve
    at 63 ms while calling it an acceptance edge. Here each bin is compared to
    the median of its `k` inward neighbours, which tracks the decline and fires
    only on the discontinuity. Verified stable for frac in 0.35-0.5.
    """
    dt = ev.loc[~ev['is_leader'], 'dt_ms'].to_numpy()
    edges = np.arange(0.0, np.ceil(dt.max()) + bin_ms, bin_ms)
    h, _ = np.histogram(dt, bins=edges)
    lo = hi = None
    for i in range(len(h) - 1, k, -1):          # walk in from the right
        loc = np.median(h[i - k:i])
        if loc > 0 and h[i] >= frac * loc:
            hi = float(edges[i + 1])
            break
    for i in range(0, len(h) - k):              # walk in from the left
        loc = np.median(h[i + 1:i + 1 + k])
        if loc > 0 and h[i] >= frac * loc:
            lo = float(edges[i])
            break
    if lo is None or hi is None or hi <= lo:    # pathological -> nominal
        lo, hi = L.READOUT_START_MS, L.GATE_CLOSE_MS
    if verbose:
        print(f'  measured dt acceptance: {lo:g}-{hi:g} ms '
              f'(nominal {L.READOUT_START_MS:g}-{L.GATE_CLOSE_MS:g})')
    return lo, hi


def centers(lo, hi, width=WIDTH_MS, step=STEP_MS):
    """Boxcar centres, restricted so every box lies FULLY inside [lo, hi].

    A box that hangs off the acceptance edge mixes 'no tracks here' with 'no
    acceptance here'. Whole-boxes-inside means every point shares the same dt
    acceptance and the points stay comparable.
    """
    first = lo + width / 2.0
    last = hi - width / 2.0
    if last < first:
        return np.array([])
    n = int(np.floor((last - first) / step)) + 1
    return first + step * np.arange(n)


def boxcar_counts(dt, hit, cen, width):
    """(k, n) per centre: events in [c-W/2, c+W/2] and how many 'hit'.

    Sorted-array + cumsum, so cost is O(N log N) once, not O(N) per centre.
    """
    dt = np.asarray(dt, float)
    hit = np.asarray(hit).astype(np.int64)
    order = np.argsort(dt, kind='stable')
    dts = dt[order]
    csum = np.concatenate([[0], np.cumsum(hit[order])])
    lo = np.searchsorted(dts, cen - width / 2.0, side='left')
    hi = np.searchsorted(dts, cen + width / 2.0, side='right')
    return (csum[hi] - csum[lo]).astype(float), (hi - lo).astype(float)


def curve(ev_det, Ld, cen, width, min_n=MIN_N_PER_POINT):
    """Boxcar curve for ONE cell and detector.

    `ev_det` must already be restricted to events this detector was read out
    for. Returns a tidy frame, one row per surviving centre.
    """
    dt = ev_det['dt_ms'].to_numpy()
    k_pair, n = boxcar_counts(dt, ev_det[f'n_pair_{Ld}'] > 0, cen, width)
    k_trk, _ = boxcar_counts(dt, ev_det[f'n_trkseg_{Ld}'] > 0, cen, width)
    k_blind, _ = boxcar_counts(dt, ~ev_det[f'live_{Ld}'].to_numpy(bool), cen, width)
    p_pair, e_pair = ST.binom_err(k_pair, n)
    p_trk, e_trk = ST.binom_err(k_trk, n)
    out = pd.DataFrame({
        'dt_ms': cen, 'n': n, 'k_pair': k_pair, 'k_trk': k_trk,
        'p_pair': p_pair, 'e_pair': e_pair,
        'p_trk': p_trk, 'e_trk': e_trk,
        'blind_frac': np.where(n > 0, k_blind / np.maximum(n, 1), np.nan),
    })
    return out[out.n >= min_n].reset_index(drop=True)


def build(ev, width=WIDTH_MS, step=STEP_MS, min_n=MIN_N_PER_POINT,
          verbose=True, group_extra=()):
    """Boxcar curves for every (mip, drift, resist, det [, *group_extra]) cell.

    `group_extra` adds further cell keys — e.g. ('iband',) to split by beam
    pulse intensity (see intensity.py). Rows whose extra key is blank/NaN are
    dropped, so an unmatched population can never leak into a slice.

    Returns a tidy long frame; this is the single table every figure and the
    CSV are drawn from.
    """
    group_extra = list(group_extra)
    if group_extra:
        before = len(ev)
        for k in group_extra:
            ev = ev[ev[k].notna() & (ev[k] != '')]
        if verbose and len(ev) != before:
            print(f'  {before - len(ev)} event(s) dropped for missing '
                  f'{"/".join(group_extra)}')
    # The flash leader is the trigger itself, not a physics probe: it sits at
    # dt=0 and was never reco'd (its n_pair_* are NaN, which would silently
    # count as a FAILED event in the denominator). Drop it explicitly rather
    # than relying on the first box happening to start above 0.
    ev = ev[~ev['is_leader']].copy()
    lo, hi = gate_edges(ev, verbose=verbose)
    cen = centers(lo, hi, width=width, step=step)
    if not len(cen):
        sys.exit(f'boxcar width {width} ms does not fit the {lo:g}-{hi:g} ms gate')
    rows = []
    keys = ['mip', 'drift', 'resist'] + group_extra
    for kv, grp in ev.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        mip, dr, r = kv[0], kv[1], kv[2]
        for Ld in 'ABCD':
            sub = grp[grp[f'readout_{Ld}']]
            if sub.empty:
                continue
            c = curve(sub, Ld, cen, width, min_n)
            if c.empty:
                continue
            c['mip'], c['drift'], c['resist'], c['det'] = mip, dr, r, Ld
            for k, v in zip(group_extra, kv[3:]):
                c[k] = v
            c['resist_eff'] = L.resist_for_det(r, Ld)
            rows.append(c)
    if not rows:
        sys.exit('no cells survived — is the cache built?')
    out = pd.concat(rows, ignore_index=True)
    out['width_ms'] = width
    out['step_ms'] = step
    if verbose:
        # run_67's grid is ragged: drifts 500/600/700 carry the full 7-point
        # resist ladder, drift 400 was truncated after ~2 sub-runs at the
        # highest gain. Nothing here marginalises over resist, so a thin block
        # is still a legitimate per-cell measurement (cf. detA_2d) and is NOT
        # dropped -- but it must be SAID, or a 1-resist drift-400 panel reads
        # like a measured surface next to the full ones.
        cells = (out.drop_duplicates(['mip', 'drift', 'resist'])
                 .groupby(['mip', 'drift']).size())
        thin = cells[cells < cells.max()]
        if len(thin):
            print('  ragged grid — these (mip, drift) blocks are incomplete '
                  f'(<{cells.max()} resist points); their panels are thin by '
                  'construction, not by physics:')
            for (mip, dr), k in thin.items():
                print(f'    mip {mip}, drift {dr} V: {k} resist point(s)')
        print(f'  boxcar: W={width} ms, step={step} ms, '
              f'{len(cen)} centres ({cen[0]:g}-{cen[-1]:g} ms), '
              f'{out.groupby(["mip","drift","resist","det"]).ngroups} cells, '
              f'{len(out)} points')
        print(f'  median events per point: {out.n.median():.0f} '
              f'(min kept {min_n})')
    return out
