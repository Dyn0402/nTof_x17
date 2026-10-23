#!/usr/bin/env python3
"""xasym_toy.py -- what does the unmodelled X +-1 asymmetry cost the symmetric forward model? (FINDINGS §43,
the one check of XASYM_CONTINGENCY.md)

Toy on det3's production model (calib_bundle_r06, symmetric X).  For every real det3 calibration event the X
view is fitted ref-pinned to get a realistic charge profile q and t0; the truth is the M3 reference (p0, tan).
Waveforms are then generated from the model at the truth, Gaussian noise at each strip's own noise is added and
samples are clipped at the ADC ceiling, and the event is refitted with the production fit (blind start from
init_guess, t0 prior sigma 5 ns at the true t0).

The injection is §42's measured term: a one-sided copy of each strip's prompt pulse onto ONE neighbour,
amplitude ``a`` x the strip's own pulse, delayed ``d`` ns (det3 head-on: the -1 minus +1 stacked pulse is
0.13 x the centre pulse, ~114 ns late; aQ = +0.165).  ``side`` -1 puts it on the lower-x neighbour, +1 on the
upper.  The injected -1 minus +1 pulse comes out 0.124 x the centre pulse (data 0.13).  Head-on, the aQ it adds
is +0.118 on r06's footprint and +0.160 on sigma_p0_x 0.25 (data +0.165): read it from the arms' 'aQ head-on'
(daQ in the paired lines averages over all angles).

Positions are compared at the event's charge-centroid time (x at t0 + <u>_q on the fitted and on the true line),
not at the mesh: p0 at the mesh slides by w x 60 ns whenever the fit moves its t0 by one depth bin, which at
|tan| > 0.15 happens to half the fits even in the closure arm and says nothing about where the track is.

Arms (generator -> fitter):
  sym       r06            -> r06               closure: the noise/clipping floor
  inj-      r06 + inj(-1)  -> r06               the cost, one sign
  inj+      r06 + inj(+1)  -> r06               the other sign (bias must flip)
  s25       r06, sigma_p0_x 0.25 -> same        closure with the per-event footprint
  s25inj-   s25 + inj(-1)  -> s25               the cost with the per-event footprint (the plan's nominal)
  s25inj-42 s25 + inj(-1)  -> r06 (0.42)        footprint left at r06 under the same truth

    xasym_toy.py [--seeds 3] [--a 0.13] [--delay 114]   -> results/xasym_toy_det3.json
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO]

W_DIR = ('/media/dylan/data/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
         'long_run_resist_490V_drift_1000V/mx17_3/wft')
BUNDLE = f'{W_DIR}/calib_bundle_r06'
CACHE = f'{W_DIR}/calib_work/calib_cache.pkl'
T0_SIGMA = 5.0
TAN_BINS = (-0.7, -0.3, -0.1, -0.03, 0.03, 0.1, 0.3, 0.7)
ARMS = {'sym': (None, 0, None), 'inj-': (None, -1, None), 'inj+': (None, +1, None),
        's25': (0.25, 0, 0.25), 's25inj-': (0.25, -1, 0.25), 's25inj-42': (0.25, -1, None)}
_EV = None


def _init():
    global _EV
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(CACHE, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(BUNDLE))


def s68(x):
    x = np.asarray(x, float)
    return float(np.percentile(np.abs(x - np.median(x)), 68.27)) if len(x) else float('nan')


def inj_matrix(pos, p0, w, t0, hyper, a, delay, side):
    """(n*NSAMP, K): strip j's prompt fraction, a x, delayed, onto strip j + side."""
    from wft import model as wm
    s0 = float(hyper.get('sigma_p0_x', hyper['sigma_p0']))
    F = wm.strip_fractions(pos, p0, w, s0, hyper['Dp'])
    Hd = np.interp(wm.TS[:, None] - (t0 + wm.UK[None, :] + delay), wm.TGRID, wm.TMPL['x'], left=0, right=0)
    Fs = np.zeros_like(F)
    if side < 0:
        Fs[:-1] = F[1:]            # strip j-1 gets a copy of strip j (pos sorted ascending)
    else:
        Fs[1:] = F[:-1]
    return (a * Fs[:, None, :] * Hd[None, :, :]).reshape(len(pos) * wm.NSAMP, wm.K)


def one(payload):
    eid, a, delay, seeds = payload
    from wft import model as wm
    ev = _EV[eid]
    P = ev['x']
    W = np.asarray(P['W'], float)
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    order = np.argsort(P['pos'])
    P = {k: np.asarray(v)[order] for k, v in P.items()}
    v = float(wm.CAL.v_drift)
    h0 = dict(wm.HYPER)
    p0r, tr = float(ev['ref_mesh_x']), float(ev['tan_x'])
    wr = tr * v * 1e-3
    t0g = wm.init_guess(P, 'x', 0.0, None, v)[2]
    try:
        r = wm.fit_plane_raw(P, 'x', p0r, wr, t0g, hyper=h0, fix_p0w=(p0r, wr))
    except Exception:
        return eid, None
    if r['q'] is None or not np.isfinite(r['chi2']):
        return eid, None
    q, t0 = np.asarray(r['q'], float), float(r['t0'])
    tc = t0 + float(np.sum(q * wm.UK) / np.sum(q))          # charge-centroid time: where the position is quoted
    pos = np.asarray(P['pos'], float)
    g = wm.GAIN['x'][np.asarray(P['ch'], int)]
    noise = np.maximum(np.asarray(P['noise'], float), 3.0)
    out = dict(tr=tr, p0r=p0r, qsum=float(q.sum()), arms={})
    # aQ in gain-corrected units, positive lobes, about the brightest strip of the uninjected event (§42's definition
    # on the -1/+1 neighbours of the centre); same strip for every arm, so the arms' difference is the injection's
    j_c = int(np.argmax((wm.build_matrix('x', pos, p0r, wr, t0, h0) @ q).reshape(len(pos), wm.NSAMP).max(1)))
    for name, (s_gen, side, s_fit) in ARMS.items():
        hg = dict(h0)
        if s_gen is not None:
            hg['sigma_p0_x'] = s_gen
        hf = dict(h0)
        if s_fit is not None:
            hf['sigma_p0_x'] = s_fit
        M = wm.build_matrix('x', pos, p0r, wr, t0, hg)
        if side:
            M = M + inj_matrix(pos, p0r, wr, t0, hg, a, delay, side)
        clean = (M @ q).reshape(len(pos), wm.NSAMP) * g[:, None]
        Q = np.clip(clean / g[:, None], 0, None).sum(1)
        j = j_c
        aq = float((Q[j - 1] - Q[j + 1]) / (Q[j - 1] + Q[j + 1])) if 0 < j < len(Q) - 1 else float('nan')
        res = []
        for s in range(seeds):
            rng = np.random.default_rng((eid * 1000 + s) * 7 + len(name))
            Wn = np.minimum(clean + rng.normal(0, noise[:, None], clean.shape), wm.SAT)
            Pn = dict(P)
            Pn['W'] = Wn
            p0i, wi, _ = wm.init_guess(Pn, 'x', 0.0, None, v)
            try:
                f = wm.fit_plane_raw(Pn, 'x', p0i, wi, t0, hyper=hf, t0_prior=(t0, T0_SIGMA))
            except Exception:
                continue
            if not np.isfinite(f['chi2']):
                continue
            res.append(dict(tan=float(f['w'] * 1e3 / v), p0=float(f['p0']), chi2=float(f['chi2'] / max(f['dof'], 1)),
                            dt0=float(f['t0'] - t0),
                            dxc=float(f['p0'] + f['w'] * (tc - f['t0']) - (p0r + wr * (tc - t0)))))
        out['arms'][name] = dict(aq=aq, fits=res)
    return eid, out


def summarise(res):
    ok = [r for r in res.values() if r]
    S = {}
    for name in ARMS:
        tr, dth, dp, chi, aq, dxc, dt0 = [], [], [], [], [], [], []
        for r in ok:
            A = r['arms'][name]
            if abs(r['tr']) < 0.03 and np.isfinite(A['aq']):
                aq.append(A['aq'])
            for f in A['fits']:
                tr.append(r['tr'])
                dth.append(np.degrees(np.arctan(f['tan']) - np.arctan(r['tr'])))
                dp.append(f['p0'] - r['p0r'])
                chi.append(f['chi2'])
                dxc.append(f['dxc'])
                dt0.append(f['dt0'])
        tr, dth, dp, dxc, dt0 = map(np.array, (tr, dth, dp, dxc, dt0))
        good = np.abs(dth) < 10                       # wrong-minimum fits are counted, not averaged
        bins = []
        for lo, hi in zip(TAN_BINS[:-1], TAN_BINS[1:]):
            m = good & (tr >= lo) & (tr < hi)
            bins.append(dict(lo=lo, hi=hi, n=int(m.sum()), bias=float(np.median(dth[m])) if m.any() else None,
                             s68=s68(dth[m]) if m.any() else None,
                             dxc_um=float(np.median(dxc[m]) * 1e3) if m.any() else None,
                             dxc_s68_um=s68(dxc[m]) * 1e3 if m.any() else None))
        S[name] = dict(n=int(len(dth)), n_bad=int((~good).sum()), bias_deg=float(np.median(dth[good])),
                       s68_deg=s68(dth[good]), dxc_um=float(np.median(dxc[good]) * 1e3), dxc_s68_um=s68(dxc[good]) * 1e3,
                       dt0_s68_ns=s68(dt0[good]), frac_slip=float(np.mean(np.abs(dt0[good]) > 30)), dp0_um=float(np.median(dp[good]) * 1e3), dp0_s68_um=s68(dp[good]) * 1e3,
                       chi2_median=float(np.median(chi)), aq_headon=float(np.median(aq)) if aq else None,
                       n_headon=len(aq), bins=bins, dth=dth.tolist(), tr=tr.tolist(), dp=dp.tolist(), dxc=dxc.tolist())
    return S


def paired(res, a, b):
    """Per-fit difference of arm b against arm a (same event, same seed index): the cost without the floor."""
    d_th, d_p, d_aq, d_th0 = [], [], [], []
    for r in res.values():
        if not r:
            continue
        if np.isfinite(r['arms'][a]['aq']) and np.isfinite(r['arms'][b]['aq']):
            d_aq.append(r['arms'][b]['aq'] - r['arms'][a]['aq'])
        fa, fb = r['arms'][a]['fits'], r['arms'][b]['fits']
        for x, y in zip(fa, fb):
            ta, tb = np.degrees(np.arctan(x['tan'])), np.degrees(np.arctan(y['tan']))
            if abs(ta - np.degrees(np.arctan(r['tr']))) < 10 and abs(tb - np.degrees(np.arctan(r['tr']))) < 10:
                d_th.append(tb - ta)
                d_p.append(y['dxc'] - x['dxc'])
                if abs(r['tr']) < 0.05:
                    d_th0.append(tb - ta)
    d_th, d_p, d_th0 = np.array(d_th), np.array(d_p), np.array(d_th0)
    n = len(d_th)

    def med_err(x):                                   # median and its bootstrap error
        rng = np.random.default_rng(1)
        bs = [np.median(rng.choice(x, len(x))) for _ in range(300)]
        return float(np.median(x)), float(np.std(bs))
    m, e = med_err(d_th)
    m0, e0 = med_err(d_th0)
    mp, ep = med_err(d_p * 1e3)
    return dict(n=n, dtheta_median=m, dtheta_err=e, dtheta_s68=s68(d_th), n_headon=len(d_th0), dtheta_headon=m0,
                dtheta_headon_err=e0, dxc_median_um=mp, dxc_err_um=ep, dxc_s68_um=s68(d_p) * 1e3,
                frac_dxc_gt_200um=float(np.mean(np.abs(d_p) > 0.2)), d_aq_mean=float(np.mean(d_aq)),
                d_aq_err=float(np.std(d_aq) / np.sqrt(len(d_aq))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--a', type=float, default=0.13)
    ap.add_argument('--delay', type=float, default=114.0)
    ap.add_argument('--max-ev', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=7)
    ap.add_argument('--tag', default='')
    p = ap.parse_args()
    ev = pickle.load(open(CACHE, 'rb'))
    ids = [e for e in sorted(ev) if 'x' in ev[e]][:p.max_ev]
    del ev
    with ProcessPoolExecutor(p.jobs, initializer=_init) as pool:
        res = dict(pool.map(one, [(e, p.a, p.delay, p.seeds) for e in ids], chunksize=4))
    S = summarise(res)
    pairs = {f'{b} vs {a}': paired(res, a, b) for a, b in
             (('sym', 'inj-'), ('sym', 'inj+'), ('s25', 's25inj-'), ('s25', 's25inj-42'))}
    print(f'det3 X toy: {sum(1 for r in res.values() if r)} events x {p.seeds} seeds, a {p.a}, delay {p.delay} ns')
    for k, s in S.items():
        print(f'  {k:10s} n {s["n"]:5d} bad {s["n_bad"]:3d}  bias {s["bias_deg"]:+.4f}  s68 {s["s68_deg"]:.4f} deg  '
              f'x(tc) {s["dxc_um"]:+6.1f} um (s68 {s["dxc_s68_um"]:.0f}; slip {s["frac_slip"]:.2f})  chi2 {s["chi2_median"]:.2f}  '
              f'aQ head-on {s["aq_headon"] if s["aq_headon"] is None else round(s["aq_headon"], 3)} (n {s["n_headon"]})')
        print('             by tan: ' + '  '.join(f'[{b["lo"]:+.2f},{b["hi"]:+.2f}) {b["bias"]:+.3f}/{b["dxc_um"]:+.0f}um' for b in s['bins'] if b['n']))
    for k, d in pairs.items():
        print(f'  paired {k:20s} n {d["n"]}  dtheta median {d["dtheta_median"]:+.4f} +- {d["dtheta_err"]:.4f} deg (s68 {d["dtheta_s68"]:.4f}; '
              f'|tan|<0.05 {d["dtheta_headon"]:+.4f} +- {d["dtheta_headon_err"]:.4f}, n {d["n_headon"]})  '
              f'x(tc) {d["dxc_median_um"]:+.1f} +- {d["dxc_err_um"]:.1f} um (s68 {d["dxc_s68_um"]:.0f}, >200um {d["frac_dxc_gt_200um"]:.3f})  '
              f'daQ {d["d_aq_mean"]:+.3f} +- {d["d_aq_err"]:.3f}')
    os.makedirs(f'{HERE}/results', exist_ok=True)
    json.dump(dict(det='det3', bundle=BUNDLE, a=p.a, delay=p.delay, seeds=p.seeds, arms=S, paired=pairs),
              open(f'{HERE}/results/xasym_toy_det3{p.tag}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
