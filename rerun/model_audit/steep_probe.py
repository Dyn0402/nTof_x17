#!/usr/bin/env python3
"""steep_probe.py --det det3 [--tan-min 0.25] [--variants base noprof dp_lo dp_hi asymp] [--jobs 8]

Why do steep tracks read shallow (F4/F16)?  On held-out events with |tan_ref| >= tan_min: the production fit
(wft.reco.fit_plane, pivot start) and, at the M3 angle (w_ref = tan_ref·kw·v + w0, p0 scanned ±1.5 mm, t0 and the
profile fitted), the best chi2 the model can reach.  dchi2 = chi2(M3 angle) − chi2(fit): > 0 means the MODEL prefers the
fitted (compressed) angle — a model mismatch; < 0 means the optimiser missed the better basin.  Per variant.
Writes <out>/steep_probe_<det>.json.
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
_EV = None; _CAL = None


def _init(cache, bundle, hyper_over, hyper_scale):
    global _EV, _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    for k, f in (hyper_scale or {}).items():
        cal.hyper[k] *= f
    cal.hyper.update(hyper_over or {})
    wm.use_calibration(cal); wm.set_depth_bins(cal.n_depth_bins); wm.COL_GUARD = 0.01; wr.PIVOT_START = True
    _CAL = cal


def one(eid):
    from wft import model as wm, reco as wr
    ev = _EV[eid]; out = {'eid': int(eid)}
    for plane in 'xy':
        P = ev.get(plane)
        if P is None:
            continue
        tr = float(ev[f'tan_{plane}']); ref = float(ev[f'ref_mesh_{plane}'])
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        try:
            f = wr.fit_plane(P, plane, _CAL)
        except Exception:
            f = None
        if f is None:
            continue
        hyper = dict(_CAL.hyper)
        if np.isfinite(getattr(f, '_sigma_p0', np.nan)):
            hyper[f'sigma_p0_{plane}'] = float(f._sigma_p0)      # the footprint the production fit chose
        w_ref = (tr * _CAL.kw.get(plane, 1.0) * _CAL.v_drift + _CAL.w0.get(plane, 0.0)) * 1e-3
        best = (np.inf, np.nan)
        for dp in np.arange(-1.5, 1.51, 0.15):
            r = wm.fit_plane_raw(P, plane, ref + dp, w_ref, f.t0, hyper=hyper, fix_p0w=(ref + dp, w_ref))
            if r['chi2'] < best[0]:
                best = (r['chi2'], ref + dp)
        # and the same scan at the FITTED w (so the p0-scan granularity cancels)
        bestf = (np.inf, np.nan)
        for dp in np.arange(-1.5, 1.51, 0.15):
            r = wm.fit_plane_raw(P, plane, f.p0 + dp, f.w, f.t0, hyper=hyper, fix_p0w=(f.p0 + dp, f.w))
            if r['chi2'] < bestf[0]:
                bestf = (r['chi2'], f.p0 + dp)
        out[plane] = dict(tan_ref=tr, w_ref=w_ref * 1e3, w_fit=f.w * 1e3, chi2_fit=f.chi2, chi2_fit_scan=bestf[0], chi2_ref=best[0],
                          p0_fit=f.p0, p0_ref=ref, p0_ref_best=best[1], dof=f.dof, sigma_used=float(getattr(f, '_sigma_p0', np.nan)),
                          nsat=int((W >= wm.SAT).sum()), q_uend=f.q_uend)
    return out


VARIANTS = {'base': ({}, {}), 'noprof': ({'sigma_p0_prof_x': [], 'sigma_p0_prof_y': []}, {}), 'dp_lo': ({}, {'Dp': 0.7}),
            'dp_hi': ({}, {'Dp': 1.3}), 'asymp': ({'rc_asym_x': 0.13}, {}), 'asymn': ({'rc_asym_x': -0.13}, {}),
            'rcd_hi': ({}, {'rc_D_y': 1.5}), 'rcd_lo': ({}, {'rc_D_y': 0.6}), 's0wide': ({'sigma_p0_x': 0.42, 'sigma_p0_y': 0.50}, {})}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--det', required=True); ap.add_argument('--tan-min', type=float, default=0.25)
    ap.add_argument('--variants', nargs='*', default=['base', 'noprof', 'dp_lo', 'dp_hi']); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--max', type=int, default=250); ap.add_argument('--out', default=os.path.expanduser('~/x17/cosmic_bench/model_audit'))
    a = ap.parse_args()
    cache = os.path.expanduser(f'~/x17/cosmic_bench/vladder/{a.det}/cache.pkl'); bundle = os.path.join(a.out, a.det, 'calib_bundle')
    C = pickle.load(open(cache, 'rb')); ids = sorted(C)[180:]
    steep = [e for e in ids if max(abs(C[e]['tan_x']), abs(C[e]['tan_y'])) >= a.tan_min][:a.max]
    print(f'{a.det}: {len(steep)} held-out events with a view at |tan| >= {a.tan_min}', flush=True)
    del C
    res = {}
    for var in a.variants:
        hy, hs = VARIANTS[var]
        with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(cache, bundle, hy, hs)) as pool:
            rows = list(pool.map(one, steep, chunksize=2))
        res[var] = rows
        for plane in 'xy':
            R = [r[plane] for r in rows if plane in r and abs(r[plane]['tan_ref']) >= a.tan_min]
            if not R:
                continue
            ratio = np.array([r['w_fit'] / r['w_ref'] for r in R]); d = np.array([r['chi2_ref'] - r['chi2_fit_scan'] for r in R])
            dof = np.array([r['dof'] for r in R])
            print(f'  {var:7s} {plane}: n {len(R)}  median w_fit/w_ref {np.median(ratio):.3f}  dchi2(ref-fit): median {np.median(d):+.0f} '
                  f'({np.median(d / dof):+.3f}/dof), frac model prefers fit (d>0) {np.mean(d > 0):.2f}, frac |ratio-1|>0.15 {np.mean(np.abs(ratio - 1) > 0.15):.2f}', flush=True)
    json.dump(res, open(os.path.join(a.out, f'steep_probe_{a.det}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
