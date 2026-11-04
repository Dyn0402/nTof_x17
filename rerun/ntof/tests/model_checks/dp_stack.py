#!/usr/bin/env python3
"""dp_stack.py -- model check 3: the drift diffusion Dp on the n_TOF unsaturated head-on X stack (and on the bench,
with the same code, as the closure of the method against the pass-1 input).

    dp_stack.py --cache C.pkl [--cache C2.pkl ...] --bundle B --out J.json --label L [--source ntof|bench]
                [--tan 0.03] [--nboot 20] [--gap 28.08]

The estimator is the bench one that set every pass-1 Dp (physics_inputs.json "Dp": cloud_basics unsat_width.py ->
x_width.measure): the unbiased head-on stack of rc_diffusion.stack (9 strips around the brightest, per event not
normalised, aligned on the first crossing of 60 ADC by the 9-strip sum, plain mean), fitted with x_width.fit_x

    W_o(t) = A sum_u F_o(sigma(u)) h(t - dt - u),   sigma^2 = sigma0^2 + 2 Dd u,   u uniform over [0, T]

with sigma0, Dd, A, dt free and T free ('free') or pinned at gap / v ('Tpin', v = the bundle's in-situ timing v);
Dp = sqrt(2 Dd) [mm/sqrt(ns)] and D_T = sqrt(2 Dd / v) [um/sqrt(cm)].  h = the bundle's X template.

Selection (the bench rules, with the reco standing in for M3 at n_TOF):
  * |tan_x| < --tan: the M3 reference on the bench (cache 'tan_x'), the fitted tan at n_TOF;
  * window max < 3500 ADC (unsat_width: stacks must drop saturated events, FINDINGS §30);
  * the bundle's dead X channels are NaN in the stack ('bad', as strip_gain_audit's flags on the bench);
  * n_TOF only: a full-gap column (fitted q_uend within [0.75, 1.35] x gap / v), the brightest strip within one strip
    of the fitted track, and no oscillating centre strip (min > -5 sigma) -- beam deposits that stop in the gas and
    the near-oscillator channels are not drift columns.
Alignment: 'threshold' is the bench rule (first 60 ADC crossing, from sample --k-min = 1 here since the n_TOF column
starts in the first samples); 'trigger' stacks on the window clock (cosmic trigger jitter ~13 ns, NTOF plan §6).
Splits by |tan_x| (< 0.015 / 0.015-0.03) bound the track-walk part of the width.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
CB = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
sys.path[:0] = [REPO, CB, os.path.join(REPO, 'mx_june_cosmic_qa')]
GAPS = {'mx17_3': 28.08, 'mx17_2': 31.12, 'mx17_6': 28.0, 'mx17_7': 29.44, 'mx17_4': 30.48}


def template(bundle):
    from rc_diffusion import STEP
    z = np.load(os.path.join(bundle, 'arrays.npz'))
    g, h = z['grid'], z['tmpl_x']
    tg = np.arange(0, g[-1] - g[0], STEP)
    hh = np.interp(tg + g[0], g, h)
    return hh / hh.max()


def select(ev, cal, source, tan, gap):
    dead = set(int(c) for c in (cal.dead or {}).get('x', []))
    T = gap / (cal.v_drift * 1e-3)
    out, why = {}, dict(total=0, tan=0, sat=0, column=0, other=0, osc=0, kept=0)
    for k, e in ev.items():
        if 'x' not in e:
            continue
        why['total'] += 1
        if abs(e['tan_x']) >= tan:
            continue
        why['tan'] += 1
        W = np.asarray(e['x']['W'], float)
        if W.max() >= 3500:
            continue
        why['sat'] += 1
        if source == 'ntof':
            f = e['fit_x']
            if not (0.75 * T <= f['q_uend'] <= 1.35 * T):
                continue
            why['column'] += 1
            pos = np.asarray(e['x']['pos'], float)
            c = int(np.argmin(np.abs(pos - f['p0'])))
            if abs(int(np.argmax(W.sum(1))) - c) > 1:
                continue
            why['other'] += 1
            if W[c].min() < -5.0 * float(e['x']['noise'][c]):
                continue
            why['osc'] += 1
        ch = np.asarray(e['x']['ch'], int)
        bad = np.flatnonzero(np.isin(ch, list(dead)))
        e = dict(e, x=dict(e['x'], W=W, bad=bad) if len(bad) else dict(e['x'], W=W))
        out[k] = e
        why['kept'] += 1
    return out, why


def stack(ev, plane, tan_max=0.03, k_min=1, align='threshold'):
    """rc_diffusion.stack (unbiased: not normalised, aligned on the first 60 ADC crossing of the 9-strip sum, plain
    mean, centre = the brightest strip), with the crossing allowed from sample k_min = 1 instead of 3: at n_TOF the
    column starts in the first samples of the 20-sample window (latency 27), so the bench rule would drop it.
    Samples before the window are NaN, as on the bench; a grid column is kept where > 60 % of the events have it."""
    from rc_diffusion import GRID, OFFS, THR_ADC, SAMPLE_NS
    rows = []
    for e in ev.values():
        if plane not in e or abs(e[f'tan_{plane}']) > tan_max:
            continue
        W = np.asarray(e[plane]['W'], float)
        if 'bad' in e[plane]:
            W = W.copy(); W[e[plane]['bad']] = np.nan
        ic = int(np.nanargmax(np.nansum(W, 1)))
        if ic < 4 or ic > len(W) - 5 or not np.isfinite(W[ic]).all():
            continue
        s = np.nansum(W[ic - 4:ic + 5], 0)
        if align == 'trigger':           # trigger-placed: the window's own clock (cosmics: ~13 ns common jitter)
            t = np.arange(len(s)) * SAMPLE_NS
            rows.append([np.interp(GRID, t, W[ic + o], left=np.nan, right=np.nan) for o in OFFS])
            continue
        k = next((k for k in range(1, len(s)) if s[k - 1] < THR_ADC <= s[k]), None)
        if k is None or k < k_min:
            continue
        t = (np.arange(len(s)) - (k - 1 + (THR_ADC - s[k - 1]) / (s[k] - s[k - 1]))) * SAMPLE_NS
        rows.append([np.interp(GRID, t, W[ic + o], left=np.nan, right=np.nan) for o in OFFS])
    if not rows:
        return 0, np.full((len(OFFS), len(GRID)), np.nan)
    R = np.array(rows)
    M = np.full((len(OFFS), len(GRID)), np.nan)
    for i in range(len(OFFS)):
        for j in range(len(GRID)):
            col = R[:, i, j]; col = col[np.isfinite(col)]
            if len(col) > 0.6 * len(R):
                M[i, j] = col.mean()
    M /= np.nanmax(np.nansum(M, axis=0))
    return len(R), M


K_MIN = 1
#: 's0pin' arm: sigma0 AND T pinned (x_width's split arm) -- for window-limited columns (C, as det6 on the bench),
#: where sigma0 and Dd trade; 0.36 mm is the bench det3 / n_TOF A free stack sigma0 (0.33-0.37)
SIG0_PIN = 0.36
ALIGN = 'threshold'


def fit_set(h, ev, T0, tan, nboot, seed=7):
    import x_width as xw
    rng = np.random.default_rng(seed)
    res = {}
    for arm, free, s0 in (('free', True, None), ('Tpin', False, None), ('s0pin', False, SIG0_PIN)):
        reps, n0 = [], 0
        for b in range(nboot + 1):
            if b == 0:
                e = ev
            else:
                ks = list(ev)
                e = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
            n, M = stack(e, 'x', tan_max=tan, k_min=K_MIN, align=ALIGN)
            if b == 0:
                n0 = n
            if n < 20 or not np.isfinite(M).any():
                reps.append(None)
                continue
            reps.append(xw.fit_x(h, M, T0, free_T=free, sig0_fix=s0))
        c = reps[0]
        if c is None:
            continue
        bs = [r for r in reps[1:] if r is not None]
        c = dict(c, n=int(n0))
        c['Dp'] = float(np.sqrt(2 * c['Dd']))
        for k in ('sig0', 'Dd', 'T', 'sdrift'):
            c[k + '_err'] = float(np.std([r[k] for r in bs])) if bs else np.nan
        c['Dp_err'] = float(np.std([np.sqrt(2 * r['Dd']) for r in bs])) if bs else np.nan
        res[arm] = c
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', action='append', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--source', choices=('ntof', 'bench'), default='ntof')
    ap.add_argument('--tan', type=float, default=0.03)
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--gap', type=float, default=None)
    ap.add_argument('--align', choices=('threshold', 'trigger'), default='threshold',
                    help="'trigger': no per-event shift (the window clock); the bench rule is 'threshold'")
    ap.add_argument('--k-min', type=int, default=1, help='earliest sample of the 60 ADC crossing (bench rule 3)')
    a = ap.parse_args()
    global K_MIN, ALIGN
    K_MIN, ALIGN = a.k_min, a.align
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(a.bundle)
    gap = a.gap or GAPS.get(cal.detector, 28.0)
    ev = {}
    for i, c in enumerate(a.cache):
        for k, e in pickle.load(open(c, 'rb')).items():
            ev[(i, k)] = e
    sel, why = select(ev, cal, a.source, a.tan, gap)
    h = template(a.bundle)
    v = cal.v_drift
    T0 = gap / (v * 1e-3)
    out = dict(label=a.label, align=a.align, k_min=a.k_min, caches=a.cache, bundle=a.bundle, source=a.source, tan=a.tan, gap=gap, v=v, T0=T0,
               Dp_bundle=cal.hyper['Dp'], selection=why)
    out['all'] = fit_set(h, sel, T0, a.tan, a.nboot)
    for lo, hi in ((0.0, 0.015), (0.015, a.tan)):
        sub = {k: e for k, e in sel.items() if lo <= abs(e['tan_x']) < hi}
        out[f'tan_{lo}_{hi}'] = fit_set(h, sub, T0, a.tan, a.nboot) if len(sub) >= 30 else {}
    for arm in ('free', 'Tpin', 's0pin'):
        r = out['all'].get(arm)
        if r:
            r['DT_um_rtcm'] = float(np.sqrt(2 * r['Dd'] / (v * 1e-3)) * np.sqrt(10) * 1e3)
            r['DT_err'] = float(r['DT_um_rtcm'] * 0.5 * r['Dd_err'] / max(r['Dd'], 1e-12))
            print(f'{a.label} {arm}: n {r["n"]} sig0 {r["sig0"]:.3f}+-{r["sig0_err"]:.3f}  Dp {r["Dp"]:.4f}+-'
                  f'{r["Dp_err"]:.4f} (bundle {cal.hyper["Dp"]})  D_T {r["DT_um_rtcm"]:.0f}+-{r["DT_err"]:.0f} um/rtcm  '
                  f'T {r["T"]:.0f}+-{r["T_err"]:.0f} (T0 {T0:.0f})  rms {r["rms"]:.4f}', flush=True)
    print('selection', why)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
