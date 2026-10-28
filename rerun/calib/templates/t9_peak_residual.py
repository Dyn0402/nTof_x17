"""T9: is the bright-peak residual (cloud_basics §37: data above model by +6-7 % of the peak on bright pulses) an
amplitude effect of the template, or something the model assembles?  T1 found the template SHAPE independent of
amplitude (500-3500 ADC), so a template-driven residual should not grow with amplitude.

Held-out unsaturated planes (cache events 180..), fitted from the reference start with t0 pinned (wft.model.fit_plane_raw,
main's fitter), bundle given.  Residual (data - model) / model peak (median over planes; the mean is also stored) on the brightest strip and its +-1 / +-2
neighbours, at sample offsets -4..+8 from the brightest strip's peak, stacked in amplitude bins.
Usage: python -I t9_peak_residual.py BUNDLE OUT.json [det]"""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, REPO)
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/{}.pkl')
K = np.arange(-4, 9)
AMP = [(300, 800), (800, 1500), (1500, 2500), (2500, 3549)]
_EV = None


def _init(cache, bundle):
    global _EV
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _EV = pickle.load(open(cache, 'rb'))
    wm.use_calibration(CalibrationBundle.load(bundle)); wm.set_depth_bins(24)
    if wm.COL_GUARD <= 0:
        wm.COL_GUARD = 0.01


def one(eid):
    from wft import model as wm
    ev = _EV[eid]; cal = wm.CAL; out = []
    for plane in 'xy':
        if plane not in ev:
            continue
        P = ev[plane]; W0 = np.asarray(P['W'])
        if W0.max() >= wm.SAT:
            continue
        if W0.shape[1] != wm.NSAMP:
            wm.set_nsamp(W0.shape[1])
        ft = ev.get(f'ftst_{plane}'); t0a = (cal.t0_abs or {}).get(plane, {})
        if ft is None or int(ft) not in t0a:
            continue
        t0p = float(t0a[int(ft)]); tr = float(ev[f'tan_{plane}']); p0r = float(ev[f'ref_mesh_{plane}'])
        try:
            r = wm.fit_plane_raw(P, plane, p0r, tr * cal.v_drift * 1e-3, t0p, t0_prior=(t0p, 0.05))
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        W, noise, pos, sat = wm.prep_plane(P, plane)
        M = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], r['q'], wm.HYPER)
        c, j0 = np.unravel_index(np.argmax(W), W.shape)
        a = float(W[c, j0]); mpk = max(float(M.max()), 1.0)
        res = np.full((5, len(K)), np.nan)
        for io, o in enumerate((-2, -1, 0, 1, 2)):
            if 0 <= c + o < len(W):
                for ik, k in enumerate(K):
                    if 0 <= j0 + k < W.shape[1]:
                        res[io, ik] = (W[c + o, j0 + k] - M[c + o, j0 + k]) / mpk
        out.append((plane, a, abs(tr), res.tolist(), float(r['chi2']) / max(int((~sat).sum()), 1)))
    return out


def main(bundle, outp, det='det3'):
    ids = sorted(pickle.load(open(CACHE.format(det), 'rb')))[180:]
    with ProcessPoolExecutor(7, initializer=_init, initargs=(CACHE.format(det), bundle)) as pool:
        rows = [r for rs in pool.map(one, ids, chunksize=4) for r in rs]
    summ = {}
    for plane in 'xy':
        R = [r for r in rows if r[0] == plane]
        for lo, hi in AMP:
            S = np.array([r[3] for r in R if lo <= r[1] < hi])
            if len(S) < 15:
                continue
            mean = np.nanmean(S, axis=0); m = np.nanmedian(S, axis=0)
            summ[f'{plane}_{lo}'] = dict(n=len(S), lo=lo, hi=hi, mean=mean.tolist(), median=m.tolist(),
                                         chi2dof=float(np.median([r[4] for r in R if lo <= r[1] < hi])))
            c = m[2]
            print(f'{det} {plane} amp {lo:4d}-{hi:4d} n={len(S):4d} chi2/dof {summ[f"{plane}_{lo}"]["chi2dof"]:6.1f} | '
                  f'centre k=-2..+3: ' + ' '.join(f'{x:+.3f}' for x in c[2:8]) +
                  f' | ±1 at k=0: {m[1][4]:+.3f} {m[3][4]:+.3f} | ±2 at k=0: {m[0][4]:+.3f} {m[4][4]:+.3f}')
    json.dump(dict(bundle=bundle, K=K.tolist(), summary=summ), open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
