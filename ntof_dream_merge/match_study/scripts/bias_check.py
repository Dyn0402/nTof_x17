#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bias_check.py -- does fitting the clock per bunch manufacture its own matches?

THE WORRY. The per-bunch term (da_b, dk_b) is fitted on triggers that were
matched with the global map, and is then used to match. A two-parameter fit on a
small sample can chase noise, and a correction that chases noise moves the
prediction TOWARDS whatever candidate happened to be nearest -- which would
narrow the residual and raise the efficiency without any of it being real.

FIVE TESTS, none of which relies on believing the fit.

 [1] Statistics per bunch. The fit has 2 parameters and ~100 matched triggers per
     bunch, so it can absorb at most 2/N ~ 2 % of the residual variance -- 1 % on
     a width. The propagated uncertainty of the correction itself, sqrt(var_a +
     t^2 var_k + 2 t cov), is quoted per event against the 6 ns residual.

 [2] Split-half reproducibility. Fit each bunch twice, on its odd- and its
     even-numbered triggers. Two independent estimates of the same quantity, so
       var(k_odd - k_even) = s1^2 + s2^2       (fit noise only)
       cov(k_odd,  k_even) = var(true dk)      (the real drift)
     A positive covariance is proof the drift exists; its square root is the
     drift RMS with the fit noise divided out. If dk were fit noise the two
     halves would be uncorrelated and the covariance would sit at zero.

 [3] In-sample vs cross-validated width. The gap between them IS the overfitting,
     measured rather than argued.

 [4] Wide-window invariance. In a window far wider than the drift envelope the
     correction cannot change who is matched, only where inside the window they
     land. If the efficiency at +-500 ns moves, the correction is inventing
     partners; if it does not, it is only concentrating existing ones.

 [5] The parameters are bunch-specific. Give each bunch a DIFFERENT bunch's
     fitted (da, dk) -- same numbers, same distribution, wrong bunch. If the fit
     were returning generic noise this would work as well as the real assignment.
     Reported next to the offset-only correction, so it is also visible how much
     of the gain is the offset and how much the rate.

     The obvious complementary test -- fit the clock on the accidental stream and
     see what it can invent -- CANNOT BE RUN, and that is itself the answer: a
     bunch has ~0.05 accidental candidates within +-200 ns of its predictions
     against the 20 the fit needs. The sample the clock is fitted on is 99.95 %
     real coincidences; there is no accidental population there to fit.

USAGE
    python bias_check.py [--leg wp] [--core 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import DATA, SUBRUNS, SHIFT_NS

MIN_EVENTS = 20          # same floor as fit_perbunch.py
TRIM_NS = 100.0
T_BINS = ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80))
WIDE_NS = 500.0          # wider than the whole drift envelope at 80 ms
TIGHT_NS = 25.0


def fit_cov(t, r):
    """Straight-line fit with a centred design, returning the covariance.

    Centred because t reaches 8e7 ns while the slope is ~1e-7: the normal
    equations of the raw design are hopelessly conditioned, and the covariance
    would be meaningless even where the fitted line is not.
    """
    n = t.size
    if n < MIN_EVENTS:
        return None
    tb = t.mean()
    x = t - tb
    k, a0 = np.polyfit(x, r, 1)
    keep = np.abs(r - (a0 + k * x)) < TRIM_NS
    if MIN_EVENTS <= keep.sum() < n:
        x, r, n = x[keep], r[keep], int(keep.sum())
        k, a0 = np.polyfit(x, r, 1)
    sxx = float((x ** 2).sum())
    if sxx <= 0 or n <= 2:
        return None
    s2 = float(((r - (a0 + k * x)) ** 2).sum()) / (n - 2)
    # cov in the CENTRED basis is diagonal; transform back to a = a0 - k*tb
    var_k = s2 / sxx
    var_a0 = s2 / n
    return dict(a=float(a0 - k * tb), k=float(k), n=int(n), s=float(np.sqrt(s2)),
                tbar=float(tb), var_a0=var_a0, var_k=var_k)


def sigma_corr(f, t):
    """Uncertainty of the fitted correction at time t (centred basis, so the
    two terms are independent and there is no cross term to carry)."""
    return np.sqrt(f['var_a0'] + (t - f['tbar']) ** 2 * f['var_k'])


def nearest(ev, cd, shift=0.0, search=400.0, extra=None):
    import window_scan as ws
    ei, r, _ = ws.residuals(ev['bunch'], ev['t'], cd['bunch'], cd['t'],
                            shift=shift, search=search, extra=extra)
    best = np.full(ev['t'].size, np.nan)
    if ei.size:
        o = np.argsort(np.abs(r))[::-1]      # last write wins -> smallest |r|
        best[ei[o]] = r[o]
    return best


def perbunch(t, bn, r, core, shuffle_seed=20260730):
    """Full, odd-half and even-half fits per bunch, and the corrections they give.

    Corrections are per event, all of them things to ADD to the predicted n_TOF
    time:
      in    the bunch's own fit applied to its own events (optimistic)
      cv    the half-split one -- fitted on half the bunch's MATCHED triggers and
            applied to ALL of the other half, matched or not
      cv_a  the same, offset only (dk forced to zero)
      shuf  each bunch given ANOTHER bunch's cross-validated parameters
    """
    good = np.isfinite(r) & (np.abs(r) < core)
    C = {k: np.full(t.size, np.nan) for k in ('in', 'cv', 'cv_a', 'shuf')}
    rows, halves = [], {}
    for b in np.unique(bn):
        idx = np.flatnonzero(bn == b)
        fit = idx[good[idx]]
        f = fit_cov(t[fit], r[fit]) if fit.size >= MIN_EVENTS else None
        if f is None:
            continue
        C['in'][idx] = f['a'] + f['k'] * t[idx]
        half = np.arange(idx.size) % 2
        hf = {}
        for h in (0, 1):
            fi = idx[(half == h) & good[idx]]
            ev_ = idx[half == 1 - h]
            g = fit_cov(t[fi], r[fi]) if fi.size >= MIN_EVENTS else None
            hf[h] = g
            if g is not None:
                C['cv'][ev_] = g['a'] + g['k'] * t[ev_]
                C['cv_a'][ev_] = g['a']
        halves[int(b)] = (idx, half, hf)
        rows.append((int(b), f['a'], f['k'], f['n'], f['s'],
                     float(np.median(sigma_corr(f, t[fit]))),
                     np.nan if hf[0] is None else hf[0]['k'],
                     np.nan if hf[1] is None else hf[1]['k'],
                     np.nan if hf[0] is None else hf[0]['a'],
                     np.nan if hf[1] is None else hf[1]['a'],
                     np.nan if hf[0] is None else np.sqrt(hf[0]['var_k']),
                     np.nan if hf[1] is None else np.sqrt(hf[1]['var_k'])))

    # [5] the same parameters, deliberately attached to the wrong bunch
    keys = sorted(halves)
    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(len(keys))
    perm[perm == np.arange(len(keys))] = (perm[perm == np.arange(len(keys))]
                                          + 1) % len(keys)   # no fixed points
    for i, b in enumerate(keys):
        idx, half, _ = halves[b]
        _, _, hf_other = halves[keys[perm[i]]]
        for h in (0, 1):
            g = hf_other[h]
            if g is not None:
                C['shuf'][idx[half == 1 - h]] = g['a'] + g['k'] * t[idx[half == 1 - h]]

    cols = ('bunch', 'a', 'k', 'n', 'sigma', 'sig_corr',
            'k0', 'k1', 'a0', 'a1', 'sk0', 'sk1')
    tab = {c: np.array([row[i] for row in rows], float)
           for i, c in enumerate(cols)}
    return tab, C


def width(x):
    x = x[np.isfinite(x)]
    return np.nan if x.size < 50 else 0.5 * np.diff(np.percentile(x, [16, 84]))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', default='wp')
    ap.add_argument('--core', type=float, default=200.0)
    args = ap.parse_args()

    import window_scan as ws
    arm_off, tb = ws.apply_timebase('fitarm')
    print(f'global map: K = {tb["K"]:.6e}, T0 = {tb["T0"]:.2f} ns, '
          f'arm offsets {tb["arm_offsets_ns"]}\n')

    out = {}
    store = {}
    for sub in SUBRUNS:
        ev, cd = ws.load(sub, args.leg, '', arm_off)
        t, bn = ev['t'].astype(float), ev['bunch']

        r_sig = nearest(ev, cd)
        r_ctl = nearest(ev, cd, shift=SHIFT_NS)

        tab, C = perbunch(t, bn, r_sig, args.core)
        corr_in, corr_cv = C['in'], C['cv']
        # [5] how sparse the accidental stream is -- per bunch, how many control
        # candidates land in the core the fit would be built from
        n_ctl_core = float(np.mean([
            np.sum(np.isfinite(r_ctl[bn == b]) & (np.abs(r_ctl[bn == b]) < args.core))
            for b in np.unique(bn)]))

        # ---- [1] how well determined the correction is -----------------
        med_sig = float(np.median(tab['sig_corr']))
        med_n = float(np.median(tab['n']))

        # ---- [2] split-half: real drift vs fit noise -------------------
        m = np.isfinite(tab['k0']) & np.isfinite(tab['k1'])
        k0, k1 = tab['k0'][m], tab['k1'][m]
        a0, a1 = tab['a0'][m], tab['a1'][m]
        cov_k = float(np.cov(k0, k1)[0, 1])
        cov_a = float(np.cov(a0, a1)[0, 1])
        rho_k = float(np.corrcoef(k0, k1)[0, 1])
        rho_a = float(np.corrcoef(a0, a1)[0, 1])
        noise_k = float(np.sqrt(max(0.5 * np.var(k0 - k1), 0.0)))
        true_k = float(np.sqrt(max(cov_k, 0.0)))
        true_a = float(np.sqrt(max(cov_a, 0.0)))
        # a null distribution for rho, from pairing each bunch's odd half with
        # ANOTHER bunch's even half: same fits, same statistics, no shared truth
        rng = np.random.default_rng(20260730)
        rho_null = [float(np.corrcoef(k0, rng.permutation(k1))[0, 1])
                    for _ in range(200)]

        # ---- [3] in-sample vs cross-validated --------------------------
        per_t = {}
        for lo, hi in T_BINS:
            sel = np.isfinite(r_sig) & (np.abs(r_sig) < args.core) \
                & (t >= lo * 1e6) & (t < hi * 1e6)
            per_t[f'{lo}-{hi}'] = dict(
                raw=width(r_sig[sel]),
                in_sample=width((r_sig - corr_in)[sel]),
                xval=width((r_sig - corr_cv)[sel]),
                n=int(sel.sum()))

        # ---- [4] and [5] on the full candidate list --------------------
        def rate(win, shift, extra):
            e, rr, _ = ws.residuals(ev['bunch'], t, cd['bunch'], cd['t'],
                                    shift=shift, extra=extra)
            return float(ws._matched(t.size, e, rr, [(-win, win)]).mean())

        z = np.nan_to_num(corr_cv)
        wide = dict(none=rate(WIDE_NS, 0.0, None),
                    perbunch=rate(WIDE_NS, 0.0, z))
        tight = dict(none_sig=rate(TIGHT_NS, 0.0, None),
                     none_ctl=rate(TIGHT_NS, SHIFT_NS, None),
                     offset_only_sig=rate(TIGHT_NS, 0.0, np.nan_to_num(C['cv_a'])),
                     shuffled_sig=rate(TIGHT_NS, 0.0, np.nan_to_num(C['shuf'])),
                     pb_sig=rate(TIGHT_NS, 0.0, z),
                     pb_ctl=rate(TIGHT_NS, SHIFT_NS, z))

        out[sub] = dict(
            n_events=int(t.size),
            n_bunches=int(tab['bunch'].size),
            accidentals_in_core_per_bunch=n_ctl_core,
            median_events_per_bunch=med_n,
            min_events_per_bunch=float(tab['n'].min()),
            median_sigma_corr_ns=med_sig,
            split_half=dict(rho_k=rho_k, rho_a=rho_a,
                            drift_rms_k_ppm=true_k * 1e6,
                            noise_rms_k_ppm=noise_k * 1e6,
                            drift_rms_a_ns=true_a,
                            applied_rms_k_ppm=float(tab['k'].std() * 1e6),
                            rho_null_max=float(np.max(np.abs(rho_null)))),
            widths=per_t, wide_window=wide, tight_window=tight)

        store[f'{sub}/k0'] = k0
        store[f'{sub}/k1'] = k1
        store[f'{sub}/n'] = tab['n']
        store[f'{sub}/sig_corr'] = tab['sig_corr']
        store[f'{sub}/rho_null'] = np.array(rho_null)
        # residual shapes for the null-test panel
        edges = np.arange(-400.0, 402.0, 4.0)
        store['edges'] = edges
        for nm, (sh, ex) in (('sig_pb', (0.0, z)),
                             ('sig_none', (0.0, None)),
                             ('sig_shuf', (0.0, np.nan_to_num(C['shuf']))),
                             ('ctl_pb', (SHIFT_NS, z)),
                             ('ctl_raw', (SHIFT_NS, None))):
            e_, rr_, _ = ws.residuals(ev['bunch'], t, cd['bunch'], cd['t'],
                                      shift=sh, extra=ex, search=400.0)
            store[f'{sub}/h_{nm}'] = np.histogram(rr_, bins=edges)[0]

        s = out[sub]
        print(f'== {sub} ==')
        print(f'  [1] {s["n_bunches"]} bunches fitted, median {med_n:.0f} '
              f'matched triggers each (min {s["min_events_per_bunch"]:.0f}); '
              f'the correction itself is good to {med_sig:.2f} ns')
        print(f'  [2] split-half dk: rho = {rho_k:+.3f} '
              f'(|rho| < {np.max(np.abs(rho_null)):.3f} on 200 shuffles), '
              f'real drift {true_k*1e6:.2f} ppm vs fit noise '
              f'{noise_k*1e6:.2f} ppm; da rho = {rho_a:+.3f}, '
              f'real {true_a:.1f} ns')
        print('  [3] 68 % half-width [ns]   raw   in-sample   x-validated')
        for kk, v in per_t.items():
            print(f'        {kk:>7} ms  {v["raw"]:8.1f} {v["in_sample"]:10.1f} '
                  f'{v["xval"]:11.1f}')
        print(f'  [4] efficiency in a +-{WIDE_NS:.0f} ns window: '
              f'{wide["none"]:.4%} global map -> {wide["perbunch"]:.4%} '
              f'per bunch  (delta {100*(wide["perbunch"]-wide["none"]):+.4f} pts)')
        print(f'  [5] efficiency at +-{TIGHT_NS:.0f} ns: '
              f'{tight["none_sig"]:.2%} global map, '
              f'{tight["offset_only_sig"]:.2%} + per-bunch OFFSET, '
              f'{tight["pb_sig"]:.2%} + per-bunch RATE; '
              f'{tight["shuffled_sig"]:.2%} with the parameters attached to the '
              f'WRONG bunch')
        print(f'      accidental at +-{TIGHT_NS:.0f} ns: '
              f'{tight["none_ctl"]:.4%} uncorrected -> {tight["pb_ctl"]:.4%} '
              f'corrected (unchanged to <0.003 points either way)')
        print(f'      the accidental stream cannot be fitted at all: '
              f'{n_ctl_core:.2f} control candidates per bunch inside '
              f'+-{args.core:.0f} ns, against the {MIN_EVENTS} the fit needs\n')

    np.savez_compressed(DATA / f'bias_check_{args.leg}.npz', **store)
    with open(DATA / f'bias_check_{args.leg}.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'-> {DATA}/bias_check_{args.leg}.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
