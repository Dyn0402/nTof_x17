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

# Starting point for the COARSE SEARCH only -- see `bootstrap`. It is not a
# seed the fit relies on: T0 is per (DREAM sub-run, n_TOF run) pair and moves by
# hundreds of ns between pairs, which is more than the +-250 ns core the
# iteration selects on.
#
# This used to be a real seed, and it was silently fragile. run_79 x 224572
# converged from it, so the pipeline validated; run_77 x 224571 sits at
# T0 = +109 ns, 360 ns away, and 7 of 9 segments died with "found only 69
# candidates". The two that lived started from 312 candidates against a
# hard floor of 200 -- the difference between success and failure was how much
# of the latency TAIL happened to fall inside the core, not anything physical.
K_SEED, T0_SEED = 1.1e-4, -250.0

# The coarse search. Wide enough to cover any offset seen between pairs with
# room to spare, and binned finely enough that the peak is unambiguous.
BOOT_SEARCH_NS = 50_000.0
BOOT_BIN_NS = 20.0
BOOT_MIN_PEAK = 150      # counts in the tallest bin
# The smallest peak a caller may lower the floor to, for a segment that is
# KNOWN to be small -- a sub-run tail of 6-17 real bursts whose lock was
# established burst by burst (burst_bruteforce.py, 2026-08-16). Below ~40
# counts the Poisson tail of the floor's tallest bin over ~5000 bins starts to
# matter; above it, with the sigma test still applied, a wrong lock cannot
# pass. The default stays 150; a lowered floor is recorded in the product.
BOOT_MIN_PEAK_FLOOR = 40
BOOT_MIN_SNR = 6.0       # tallest bin over the floor -- REPORTED, not a cut
# The acceptance test is significance, not the peak/floor RATIO. A ratio is
# only meaningful for a peak narrower than the bin: a coincidence spread over
# microseconds gives ratio 1.4 and 35 sigma at once, and the ratio test threw
# it away (measured on run_132 x 224662, 2026-08-09). With ~5000 bins the
# tallest bin of a flat histogram sits ~4 sigma high by chance, so 8 is safe.
BOOT_MIN_SIGMA = 8.0
BOOT_FLOOR_GAP_NS = 2000.0   # "beside it" = further than this from the peak

# The wide fallback, used only when the +-50 us search finds nothing. Covers
# the whole 80 ms burst by FFT cross-correlation -- see `xcorr_lag`.
XC_BIN_NS = 1000.0
XC_BURST_MS = 80.0
XC_BUNCHES = 300         # enough for a clear peak; the cost is linear
XC_MIN_Z = 8.0           # robust z of the tallest lag over the flat background

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


def xcorr_lag(ev_bunch, ev_t, cd_bunch, cd_t, bin_ns=XC_BIN_NS,
              burst_ms=XC_BURST_MS, max_bunches=XC_BUNCHES, log=print):
    """The coarse lag between DREAM triggers and n_TOF candidates, at ANY offset.

    `bootstrap` looks in +-50 us, which covers every offset seen on the pairs
    this pipeline was built on. It is not enough: measured 2026-08-09,
    run_132/stat090_0005 x 224662 sits at -0.982 ms, and 54 DREAM sub-runs of
    the July campaign failed for exactly this reason while their joins, bunch
    ranges and candidate rates all looked perfectly normal.

    Brute force cannot search a 80 ms range at ns resolution, so bin both time
    series per bunch and cross-correlate by FFT: every lag at once, O(N log N).
    The peak comes out smeared by ~K*80ms ~ 9 us because the rate ratio is not
    applied here -- that is fine, this only has to get close enough for the
    fine search to take over.
    """
    nb = int(burst_ms * 1e6 / bin_ns)
    acc = np.zeros(2 * nb)
    used = 0
    for b in np.unique(ev_bunch)[:max_bunches]:
        te, tc = ev_t[ev_bunch == b], cd_t[cd_bunch == b]
        if te.size < 5 or tc.size < 5:
            continue
        a = np.bincount(np.clip((te / bin_ns).astype(int), 0, nb - 1),
                        minlength=nb).astype(float)
        c = np.bincount(np.clip((tc / bin_ns).astype(int), 0, nb - 1),
                        minlength=nb).astype(float)
        acc += np.fft.irfft(np.conj(np.fft.rfft(a, 2 * nb))
                            * np.fft.rfft(c, 2 * nb), 2 * nb)
        used += 1
    if not used:
        return None
    lags = np.arange(2 * nb) * bin_ns
    lags[nb:] -= 2 * nb * bin_ns
    o = np.argsort(lags)
    lags, acc = lags[o], acc[o]
    med = float(np.median(acc))
    mad = float(np.median(np.abs(acc - med))) * 1.4826
    i = int(np.argmax(acc))
    z = (acc[i] - med) / max(mad, 1e-9)
    log(f'    wide scan: lag {lags[i]/1e6:+.4f} ms, robust z {z:.0f} '
        f'({used} bunches, {bin_ns/1000:g} us bins over +-{burst_ms:g} ms)')
    return (float(lags[i]), float(z)) if z >= XC_MIN_Z else None


def bootstrap(ev_bunch, ev_t, cd_bunch, cd_t, K=K_SEED, T0=T0_SEED,
              search=BOOT_SEARCH_NS, bin_ns=BOOT_BIN_NS, log=print,
              _retry=True, min_peak=BOOT_MIN_PEAK):
    """Find T0 for this pair by coarse search, with no reliance on the seed.

    The iterated fit can only see candidates already inside +-CORE_NS of where
    it is looking, so it cannot walk to a solution hundreds of ns away. This
    looks first, over a window wide enough that being wrong is obvious: one
    histogram of every candidate within +-`search`, whose tallest bin is the
    coincidence peak sitting on a flat accidental floor.

    Returns (T0, info). Raises if there is no peak -- which is the honest
    outcome for a mis-paired segment, and much better than fitting the floor.
    """
    ei, r, _ = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0, search=search)
    if r.size == 0:
        raise RuntimeError(
            f'no candidates at all within +-{search:g} ns of the coarse map; '
            f'this DREAM sub-run and n_TOF run are probably not the same time')
    edges = np.arange(-search, search + bin_ns, bin_ns)
    h, _ = np.histogram(r, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    i = int(h.argmax())
    peak, tall = float(centres[i]), int(h[i])

    far = np.abs(centres - peak) > BOOT_FLOOR_GAP_NS
    floor = float(np.median(h[far])) if far.any() else 0.0
    snr = tall / max(floor, 1.0)
    log(f'    bootstrap: peak {tall:,} counts at {peak:+.0f} ns, '
        f'floor {floor:.0f}/bin, S/N {snr:.0f}  '
        f'({r.size:,} candidates in +-{search/1000:g} us)')

    # Nothing here? The offset may simply be bigger than this window. Look
    # across the whole burst before giving up -- a ms-scale offset is a
    # flash-reference problem, and the data behind it is perfectly good.
    excess = tall - floor
    sigma = excess / np.sqrt(max(floor, 1.0))
    log(f'    -> excess {excess:,.0f} over the floor = {sigma:.1f} sigma')
    min_peak = max(int(min_peak), BOOT_MIN_PEAK_FLOOR)
    weak = tall < min_peak or sigma < BOOT_MIN_SIGMA
    if min_peak != BOOT_MIN_PEAK:
        log(f'    (peak floor lowered to {min_peak} counts for this segment)')

    if weak and _retry:
        log('    no peak in the fine window; trying the whole burst')
        wide = xcorr_lag(ev_bunch, ev_t, cd_bunch, cd_t, log=log)
        if wide is not None:
            lag, z = wide
            # Re-centring alone is not enough. The wide scan gives the lag but
            # not the rate ratio, and at these offsets the correlation arrives
            # spread over microseconds -- a 20 ns histogram dilutes it below
            # the floor (measured: peak 164 over a floor of 87, S/N 1.9, on a
            # correlation the wide scan saw at z 21). So walk the resolution
            # down, re-centring at each step, and let the width come with it.
            T0w, finest, widest = T0 + lag, None, None
            for bw, sw in ((2000.0, 40_000.0), (500.0, 10_000.0),
                           (100.0, 2_000.0), (bin_ns, search)):
                try:
                    T0w, info = bootstrap(ev_bunch, ev_t, cd_bunch, cd_t, K,
                                          T0w, sw, bw, log=log, _retry=False,
                                          min_peak=min_peak)
                    finest, widest = bw, widest or info
                except RuntimeError:
                    break
            else:
                info.update(wide_lag_ns=lag, wide_z=z,
                            recovered_by_wide_scan=True)
                log(f'    recovered at a {lag/1e6:+.4f} ms offset')
                return T0w, info
            # Report the WIDTH, because that is the physics. A correlation
            # significant at 2 us bins but not at 500 ns is ~microseconds
            # wide; a real coincidence here is ~6 ns. Knowing it is broad
            # rather than absent is the difference between "these hours have
            # no data" and "these hours were triggered on something else".
            if finest and widest:
                spread = widest['excess'] / max(info.get('excess', 1), 1)
                raise RuntimeError(
                    f'a correlation IS present at {lag/1e6:+.4f} ms '
                    f'(wide-scan z {z:.0f}, {widest["sigma"]:.0f} sigma at '
                    f'{finest:.0f} ns bins) but it is ~{spread*finest/1000:.0f} '
                    f'us WIDE and does not sharpen -- a real coincidence here '
                    f'is ~6 ns. This sub-run is not missing n_TOF data; '
                    f'whatever DREAM was triggering on in these hours is only '
                    f'loosely associated with the n_TOF hits. Not slimmable '
                    f'until that is understood.')
            log('    the wide scan found a lag but nothing sharpened')

    if weak:
        raise RuntimeError(
            f'no time-base peak: tallest bin has {tall} counts at {peak:+.0f} '
            f'ns over a floor of {floor:.0f} ({sigma:.1f} sigma, need '
            f'{BOOT_MIN_SIGMA:g}), and the whole-burst scan found no usable lag '
            f'either. Either the segment is too small to fit, or this DREAM '
            f'sub-run really does not overlap this n_TOF run.')
    # Keep the coarse histogram itself, rebinned to ~500 points. A summary
    # cannot show a SECOND peak, an asymmetric shoulder or a floor that slopes
    # -- the shapes that say the map is nearly-degenerate rather than clean --
    # and this is the only place the wide view exists.
    keep = max(1, len(h) // 500)
    trim = (len(h) // keep) * keep
    coarse = h[:trim].reshape(-1, keep).sum(axis=1)
    return T0 + peak, dict(
        peak_ns=peak, counts=tall, floor=floor, snr=snr,
        excess=float(excess), sigma=float(sigma), bin_ns=float(bin_ns),
        min_peak=int(min_peak),
        n_candidates=int(r.size), search_ns=search,
        hist=dict(lo_ns=float(edges[0]), bin_ns=float(bin_ns * keep),
                  counts=[int(c) for c in coarse]))


def fit_global(ev_bunch, ev_t, cd_bunch, cd_t, cd_arm, iters=ITERS,
               K=K_SEED, T0=T0_SEED, boot=True, log=print,
               min_peak=BOOT_MIN_PEAK):
    """(K, T0, arm_off[4], info) for one segment, from its own candidates.

    A coarse search fixes T0 first (`bootstrap`), because the iteration below
    only sees candidates within +-CORE_NS of its current guess and cannot walk
    to a peak further away than that. Then it is iterated so the core selection
    is centred on its own solution, and the per-arm offsets are measured at the
    converged point. The four arms' trigger paths differ by ~25 ns, invisible
    at +-150 ns and dominant below +-50.
    """
    info = {'iters': []}
    if boot:
        T0, info['bootstrap'] = bootstrap(ev_bunch, ev_t, cd_bunch, cd_t,
                                          K, T0, log=log, min_peak=min_peak)
    for it in range(iters):
        ei, r, _ = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0)
        a, b, n, s = _line(ev_t, ei, r)
        if not np.isfinite(a):
            raise RuntimeError(
                f'time-base fit found only {n} candidates inside +-{CORE_NS:g} '
                f'ns of T0 = {T0:+.0f} ns on iteration {it}. The coarse search '
                f'located a peak, so this is a segment with too few events to '
                f'fit rather than a mis-paired one.')
        T0, K = T0 + a, K + b
        info['iters'].append(dict(shift_ns=a, slope=b, K=K, T0_ns=T0, n=n,
                                  bin_spread_ns=s))
        log(f'    iter {it}: shift {a:+7.2f} ns  slope {b:+.3e}  '
            f'-> K = {K:.6e}  T0 = {T0:+.2f} ns   (n = {n:,})')

    # Per-arm offsets are CONSTANTS in the model -- the slope belongs to K and
    # is already fitted above, shared by all four arms. This used to reuse
    # `_line` per arm and keep its intercept, i.e. an extrapolation to
    # t_dream = 0 from data that starts at 0.1 ms, with a free slope fitted on
    # a quarter of the statistics. On the reference segment (25k candidates
    # per arm) the two agree to < 0.3 ns; on a 3-minute segment (~700 per arm)
    # the slope noise displaced the stored offsets by up to 12 ns while the
    # true offsets sat at their fleet values -- run_78/stat090_lat051_c0_0005,
    # arm C stored -10.3 ns against a fleet median of +1.5, with the matched
    # residuals unimodal at +12 ns to say so. A refined median has no slope to
    # get wrong: sigma ~ 1.25 * 6.5 ns / sqrt(n) ~ 0.3 ns even on that segment.
    ei, r, ci = residuals(ev_bunch, ev_t, cd_bunch, cd_t, K, T0)
    arm = np.asarray(cd_arm)[ci]
    off = np.zeros(4)
    info['per_arm'] = {}
    for ai, name in enumerate(ARMS):
        s = arm == ai
        rr = r[s][np.abs(r[s]) < CORE_NS]
        nn = int(rr.size)
        if nn >= 50:
            med = float(np.median(rr))
            fine = rr[np.abs(rr - med) < 50.0]       # re-centre off the floor
            aa = float(np.median(fine)) if fine.size >= 50 else med
            ss = float(1.4826 * np.median(np.abs(fine - aa))) if fine.size \
                else float('nan')
        else:
            aa, ss = 0.0, float('nan')
        off[ai] = aa
        info['per_arm'][name] = dict(a_ns=off[ai], n=nn, spread_ns=ss)
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
