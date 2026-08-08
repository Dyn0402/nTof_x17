#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clockfit.py -- the DREAM -> n_TOF time map, fitted from scratch for one segment.

    t_nTOF = t_DREAM (1 + K + dk_b) + T0 + a_arm + da_b

Nothing here reads a stored constant. `K`, `T0` and the per-arm offsets are per
(DREAM run, n_TOF processing) pair and do NOT transfer -- using another pair's
values is a 1.35 % rate error, 1 us at 80 ms, which no window catches. The
per-bunch terms are per bunch by construction. See
`../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`.

This is the same arithmetic as `match_study/scripts/fit_timebase.py` and
`fit_perbunch.py`, lifted out of the study's module-level globals and cached
`.npz` files so it can run on an arbitrary segment. `validate.py` checks it
reproduces the published constants on the reference pair.
"""
from __future__ import annotations

import numpy as np

ARMS = ('A', 'B', 'C', 'D')
KEY_SCALE = 1e9          # bunch/time packing; |t| stays under 8e7 ns

# Starting point for the iteration. Any value in the right ballpark converges --
# the fit is re-centred three times -- but starting near the last pair's
# solution keeps the first core selection populated.
K_SEED, T0_SEED = 1.1e-4, -250.0

CORE_NS = 250.0          # global fit: residuals inside this define the core
ITERS = 3
PB_MIN_EVENTS = 20       # per-bunch fit: below this, do not fit the bunch
PB_CORE_NS = 200.0
PB_TRIM_NS = 100.0
SEARCH_NS = 2000.0       # how far out to collect candidates when fitting


def pack(bunch, t):
    b = np.asarray(bunch, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if t.size and np.abs(t).max() >= 0.4 * KEY_SCALE:
        raise ValueError(f'time base reaches {np.abs(t).max():.3g} ns, too large '
                         f'for KEY_SCALE={KEY_SCALE:.3g}')
    return b * KEY_SCALE + t


def predict(t_dream, K, T0, arm_off=0.0, dk=0.0, da=0.0):
    """Where in the n_TOF time base a DREAM trigger at `t_dream` should be."""
    return np.asarray(t_dream, float) * (1.0 + K + dk) + T0 + arm_off + da


def residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0, arm_off=None, cd_arm=None,
              extra=None, shift=0.0, search=SEARCH_NS):
    """(event index, residual, candidate index) for candidates near an event.

    Every candidate within +-`search` of the event's predicted position and in
    the same bunch. `arm_off` (a length-4 array) is applied to the CANDIDATES,
    which is what puts all four arms on one time base so a single (K, T0)
    describes the whole sample. `extra` is a per-event correction added to the
    prediction -- the per-bunch clock term.
    """
    ct = np.asarray(cd_t, float)
    if arm_off is not None and cd_arm is not None:
        ct = ct - np.asarray(arm_off, float)[np.asarray(cd_arm)]
    order = np.lexsort((ct, np.asarray(cd_bunch)))
    ct, cb = ct[order], np.asarray(cd_bunch)[order]

    pred = predict(ev_t, K, T0) + shift
    if extra is not None:
        pred = pred + np.nan_to_num(np.asarray(extra, float))
    key_c = pack(cb, ct)
    lo = np.searchsorted(key_c, pack(ev_bunch, pred - search), side='left')
    hi = np.searchsorted(key_c, pack(ev_bunch, pred + search), side='right')
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        e = np.array([], np.int64)
        return e, np.array([]), e
    ev_idx = np.repeat(np.arange(pred.size), n)
    ci = np.repeat(lo, n) + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
    return ev_idx, ct[ci] - np.repeat(pred, n), order[ci]


def _line(ev_t, ei, r, core=CORE_NS, nbin=24):
    """Robust r = a + b t: per-time-bin MEDIAN, then a weighted straight line.

    The residual band is asymmetric -- a fast rise on the early side and a tail
    on the late one, because it is a trigger-latency distribution and not a
    resolution -- so a least-squares mean is pulled by the tail and drifts with
    the local rate. The median of the core is stable against both.
    """
    m = np.abs(r) < core
    if m.sum() < 200:
        return np.nan, 0.0, int(m.sum()), np.nan
    t, rr = np.asarray(ev_t, float)[ei[m]], r[m]
    edges = np.geomspace(max(t.min(), 1e5), t.max(), nbin + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbin - 1)
    tc, rc, wc = [], [], []
    for i in range(nbin):
        s = idx == i
        if s.sum() < 50:
            continue
        tc.append(np.median(t[s])); rc.append(np.median(rr[s])); wc.append(s.sum())
    if len(tc) < 3:
        return float(np.median(rr)), 0.0, int(m.sum()), float(rr.std())
    tc, rc, wc = np.array(tc), np.array(rc), np.array(wc, float)
    b, a = np.polyfit(tc, rc, 1, w=np.sqrt(wc))
    return float(a), float(b), int(m.sum()), float(np.std(rc - (a + b * tc)))


def fit_global(ev_bunch, ev_t, cd_bunch, cd_t, cd_arm, iters=ITERS,
               K=K_SEED, T0=T0_SEED, log=print):
    """(K, T0, arm_off[4], info) for one segment, from its own candidates.

    Iterated so the core selection is centred on its own solution, then the
    per-arm offsets are measured at the converged point. The four arms' trigger
    paths differ by ~25 ns, invisible at +-150 ns and dominant below +-50.
    """
    info = {'iters': []}
    for it in range(iters):
        ei, r, _ = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0)
        a, b, n, s = _line(ev_t, ei, r)
        if not np.isfinite(a):
            raise RuntimeError(
                f'time-base fit found only {n} candidates inside +-{CORE_NS:g} ns '
                'of the seed map; the segment is too small, or the seed is wrong '
                'for this pair')
        T0, K = T0 + a, K + b
        info['iters'].append(dict(shift_ns=a, slope=b, K=K, T0_ns=T0, n=n,
                                  bin_spread_ns=s))
        log(f'    iter {it}: shift {a:+7.2f} ns  slope {b:+.3e}  '
            f'-> K = {K:.6e}  T0 = {T0:+.2f} ns   (n = {n:,})')

    ei, r, ci = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0)
    arm = np.asarray(cd_arm)[ci]
    off = np.zeros(4)
    info['per_arm'] = {}
    for ai, name in enumerate(ARMS):
        s = arm == ai
        aa, _, nn, ss = _line(ev_t, ei[s], r[s])
        off[ai] = 0.0 if not np.isfinite(aa) else aa
        info['per_arm'][name] = dict(a_ns=off[ai], n=int(nn), spread_ns=ss)
        log(f'    arm {name}: {off[ai]:+7.2f} ns  (n = {nn:,})')
    info['K'], info['T0_ns'] = K, T0
    return K, T0, off, info


def _fit_bunch(t, r):
    if t.size < PB_MIN_EVENTS:
        return np.nan, np.nan, 0
    b, a = np.polyfit(t, r, 1)
    keep = np.abs(r - (a + b * t)) < PB_TRIM_NS
    if PB_MIN_EVENTS <= keep.sum() < t.size:
        b, a = np.polyfit(t[keep], r[keep], 1)
    return float(a), float(b), int(keep.sum())


def fit_perbunch(ev_bunch, ev_t, cd_bunch, cd_t, cd_arm, K, T0, arm_off,
                 log=print):
    """Per-event clock correction, in-sample and cross-validated.

    The residual left after the global fit grows with time since flash -- 9 ns
    at 1 ms, 44 ns beyond 40 ms -- because the DREAM timestamp clock wanders
    ~1 ppm between bursts. That is a rate error, not a resolution, and fitting
    (da_b, dk_b) per bunch flattens it to 6 ns over the whole 80 ms.

    Returns (corr_in, corr_cv, params). `corr_in` is the bunch's own fit applied
    to its own events and is what production uses. `corr_cv` is fitted on half a
    bunch's MATCHED events and applied to the other half whether matched or not,
    so an efficiency quoted with it is not the fit reading back its own input.
    """
    ev_t = np.asarray(ev_t, float)
    ev_bunch = np.asarray(ev_bunch)
    ei, r, _ = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0,
                         arm_off=arm_off, cd_arm=cd_arm, search=400.0)
    best = np.full(ev_t.size, np.nan)
    if ei.size:
        o = np.argsort(np.abs(r))[::-1]      # last write wins -> smallest |r|
        best[ei[o]] = r[o]
    good = np.isfinite(best) & (np.abs(best) < PB_CORE_NS)

    corr_in = np.full(ev_t.size, np.nan)
    corr_cv = np.full(ev_t.size, np.nan)
    params = {}
    for b in np.unique(ev_bunch):
        idx = np.flatnonzero(ev_bunch == b)
        fit_idx = idx[good[idx]]
        if fit_idx.size < PB_MIN_EVENTS:
            continue
        a, k, n = _fit_bunch(ev_t[fit_idx], best[fit_idx])
        if not np.isfinite(a):
            continue
        params[int(b)] = (a, k, n)
        corr_in[idx] = a + k * ev_t[idx]
        half = np.arange(idx.size) % 2
        for h in (0, 1):
            f = idx[(half == h) & good[idx]]
            e = idx[half == 1 - h]
            if f.size < PB_MIN_EVENTS:
                continue
            aa, kk, _ = _fit_bunch(ev_t[f], best[f])
            if np.isfinite(aa):
                corr_cv[e] = aa + kk * ev_t[e]
    nb = len(np.unique(ev_bunch))
    log(f'    per-bunch: fitted {len(params)} of {nb} bunches, '
        f'{good.sum():,} core events; '
        f'{np.isfinite(corr_in).sum()/max(ev_t.size,1):.1%} of events corrected')
    if params:
        das = np.array([v[0] for v in params.values()])
        dks = np.array([v[1] for v in params.values()])
        log(f'    da RMS {das.std():.2f} ns, dk RMS {dks.std()*1e6:.2f} ppm')
    return corr_in, corr_cv, params


def efficiency(ev_bunch, ev_t, cd_bunch, cd_t, cd_arm, K, T0, arm_off,
               corr, window_ns=25.0, shift_ns=100_000.0):
    """(efficiency, accidental rate, matched mask, best residual) at +-window.

    The accidental rate is the identical match with the DREAM time shifted by
    `shift_ns`; the local rate structure varies too much across the 80 ms for a
    neighbouring-window sideband to be a fair control.
    """
    out = {}
    for name, sh in (('sig', 0.0), ('ctl', shift_ns)):
        ei, r, ci = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0,
                              arm_off=arm_off, cd_arm=cd_arm, extra=corr,
                              shift=sh, search=max(400.0, 4 * window_ns))
        m = np.zeros(np.size(ev_t), bool)
        best = np.full(np.size(ev_t), np.nan)
        arm = np.full(np.size(ev_t), -1, np.int8)
        if ei.size:
            hit = np.abs(r) <= window_ns
            m[np.unique(ei[hit])] = True
            o = np.argsort(np.abs(r))[::-1]      # last write wins -> smallest |r|
            best[ei[o]] = r[o]
            arm[ei[o]] = np.asarray(cd_arm)[ci[o]]
        # `best` is the nearest candidate at any distance, which is a useful
        # diagnostic for an event that did NOT match. `arm` is not: outside the
        # accept window it is whatever accidental happened to be closest, so
        # null it rather than hand out an arm nobody should trust.
        arm[~m] = -1
        out[name] = (m, best, arm)
    ok = np.isfinite(corr) if corr is not None else np.ones(np.size(ev_t), bool)
    n = max(int(ok.sum()), 1)
    eff = float((out['sig'][0] & ok).sum()) / n
    acc = float((out['ctl'][0] & ok).sum()) / n
    return dict(efficiency=eff, accidental=acc,
                purity=(1 - acc / eff) if eff else float('nan'),
                n_events=int(np.size(ev_t)), n_fittable=int(ok.sum()),
                matched=out['sig'][0], residual_ns=out['sig'][1],
                arm=out['sig'][2])
