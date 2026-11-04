#!/usr/bin/env python3
"""template_resid.py -- model check 1: does the carried bench X template describe n_TOF head-on columns?

    template_resid.py --cache C.pkl --bundle B --out J.json --label L [--source ntof|bench] [--tan 0.05] [--pin]
                      [--planes x] [--workers 3] [--max-planes N]

Every unsaturated (window max < 3500 ADC, > 60 ADC) single-track plane with |tan| < --tan (default 0.6; summaries
also per |tan| bin, head-on = < 0.05) of the cache is
fitted with the pass model (wft.model.fit_plane_raw, the bundle installed: physical RC kernel, sigma0 0.30, the bundle
Dp, the bundle's X template), and the residual (data - model) / model peak is read

  * on the brightest strip and its +-1/+-2 neighbours at sample offsets k = -4..+10 from the brightest strip's peak
    (rerun/calib/templates/t9_peak_residual.py, extended to the late side where an ion tail would show), and
  * on the 5-strip sum against time after the END of the drift column, t - (t0 + T), T = gap / v of the bundle.

Two fits per plane:
  'prod'  the production charge basis (K = bundle n_depth_bins, so the profile may extend to the window end), and
  'box'   the profile limited to the column, K = ceil(T / 60 ns) + 2: with no depth bins after the column a template
          whose tail is too short or too long cannot hide in late charge, and shows as a residual.
The centre strip is the one under the fitted track; planes whose brightest strip is more than one strip from it carry a
second deposit and are skipped, and so are planes whose centre strip swings below -max(5 sigma, 0.25 x its own peak)
(oscillating channels: D's 448-451 ring at +-350 ADC on beam) and fits whose model exceeds 10x the data (runaway).  Plus, from 'prod', the fraction of the fitted charge profile beyond T + 120 ns (q_late) -- what the production fit does
with a tail mismatch -- and chi2/dof.

Starts: --source ntof takes the reco's own (p0, w, t0) (ntof_planes.py 'fit_<p>'), t0 free; --source bench the M3
reference (ref_mesh, tan * v) and t0 = the bundle's pinned latency (t0_abs by ftst), free unless --pin (sigma 0.05 ns,
the bench pass).  Medians over planes with bootstrap errors (30 resamples).
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)

K_OFF = np.arange(-4, 11)
T_BINS = np.arange(-900, 1501, 60)          # t - (t0 + T) [ns], bin edges
UNSAT, AMIN = 3500.0, 60.0
AMP_BINS = ((60, 300), (300, 1500), (1500, 3500))
TAN_BINS = ((0, 0.05), (0.05, 0.25), (0.25, 0.6))
GAPS = {'mx17_3': 28.08, 'mx17_2': 31.12, 'mx17_6': 28.0, 'mx17_7': 29.44, 'mx17_4': 30.48}
_EV = _CAL = None
_OPT = {}


def _init(cache, bundle, opt):
    global _EV, _CAL, _OPT
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _EV = pickle.load(open(cache, 'rb'))
    _CAL = CalibrationBundle.load(bundle)
    wm.use_calibration(_CAL)
    _OPT = opt


def _resid(W, M, c, j0, mpk):
    res = np.full((5, len(K_OFF)), np.nan)
    for io, o in enumerate((-2, -1, 0, 1, 2)):
        if 0 <= c + o < len(W):
            for ik, k in enumerate(K_OFF):
                if 0 <= j0 + k < W.shape[1]:
                    res[io, ik] = (W[c + o, j0 + k] - M[c + o, j0 + k]) / mpk
    return res


def one(eid):
    from wft import model as wm
    ev = _EV[eid]
    cal = _CAL
    out = []
    gap = GAPS.get(cal.detector, 28.0)
    T = gap / (cal.v_drift * 1e-3)
    for plane in _OPT['planes']:
        if plane not in ev or abs(ev.get(f'tan_{plane}', 9)) >= _OPT['tan']:
            continue
        P = ev[plane]
        W0 = np.asarray(P['W'], float)
        if W0.max() >= UNSAT or W0.max() < AMIN:
            continue
        ns = W0.shape[1]
        if ns != wm.NSAMP:
            wm.set_nsamp(ns)
        prior = None
        if _OPT['source'] == 'ntof':
            f = ev[f'fit_{plane}']
            p0, w, t0 = f['p0'], f['w'], f['t0']
        else:
            t0a = (cal.t0_abs or {}).get(plane, {})
            ft = ev.get(f'ftst_{plane}')
            if ft is None or int(ft) not in t0a:
                continue
            t0 = float(t0a[int(ft)])
            p0, w = float(ev[f'ref_mesh_{plane}']), float(ev[f'tan_{plane}']) * cal.v_drift * 1e-3
            if _OPT['pin']:
                prior = (t0, 0.05)
        rec = dict(eid=int(eid), plane=plane, tan=float(ev[f'tan_{plane}']), amp=float(W0.max()))
        for arm, kbins in (('prod', int(cal.n_depth_bins)), ('box', int(np.ceil(T / wm.DT)) + 2)):
            wm.set_depth_bins(kbins)
            try:
                r = wm.fit_plane_raw(P, plane, p0, w, t0, t0_prior=prior)
            except Exception:
                r = None
            wm.set_depth_bins(int(cal.n_depth_bins))
            if r is None or not np.isfinite(r.get('chi2', np.inf)) or r.get('q') is None:
                continue
            W, noise, pos, sat = wm.prep_plane(P, plane)
            wm.set_depth_bins(kbins)
            M = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], r['q'], wm.HYPER)
            wm.set_depth_bins(int(cal.n_depth_bins))
            # centre = the strip under the fitted track (mid-column); a plane whose brightest strip is elsewhere carries
            # a second deposit the single-track fit does not model -> not a template test, skipped
            c = int(np.argmin(np.abs(pos - (r['p0'] + r['w'] * 0.5 * T))))
            if abs(int(np.argmax(W.sum(1))) - c) > 1:
                rec['other_deposit'] = True
                break
            # oscillating channels (n_TOF near-oscillators / coupled pairs, P2 not built yet): a real deposit never
            # swings the centre strip below -5 sigma of its noise; such planes are skipped as not-a-signal
            if W[c].min() < -max(5.0 * noise[c], 0.25 * W[c].max()):
                rec['oscillation'] = True
                break
            if not np.isfinite(M).all() or M.max() > 10.0 * max(W.max(), 1.0):
                rec['runaway'] = True                     # NNLS charge runaway (q_sum >> data): not a model test
                break
            j0 = int(np.argmax(W[c]))
            mpk = max(float(M.max()), 1.0)
            lo, hi = max(c - 2, 0), min(c + 3, len(W))
            s_d, s_m = W[lo:hi].sum(0), M[lo:hi].sum(0)
            trel = np.arange(ns) * wm.SNS - (r['t0'] + T)
            ib = np.digitize(trel, T_BINS) - 1
            tres = np.full(len(T_BINS) - 1, np.nan)
            ok = (ib >= 0) & (ib < len(tres))
            tres[ib[ok]] = (s_d[ok] - s_m[ok]) / max(float(s_m.max()), 1.0)
            q = np.asarray(r['q'], float)
            uk = (np.arange(len(q)) + 0.5) * wm.DT
            dof = max(int((~np.asarray(sat, bool)).sum()) - 3 - len(q), 1)
            rec[arm] = dict(p0=float(r['p0']), w=float(r['w']), t0=float(r['t0']), chi2dof=float(r['chi2']) / dof,
                            res=_resid(W, M, c, j0, mpk).tolist(), tres=tres.tolist(),
                            q_late=float(q[uk > T + 120.0].sum() / max(q.sum(), 1e-9)),
                            j0=j0, c_edge=bool(c < 2 or c > len(W) - 3))
        if 'prod' in rec and not rec.get('other_deposit') and not rec.get('oscillation') and not rec.get('runaway'):
            out.append(rec)
    return out


def summarise(rows, nboot=30, seed=1):
    rng = np.random.default_rng(seed)
    s = {}
    for arm in ('prod', 'box'):
        R = [r[arm] for r in rows if arm in r and not r[arm]['c_edge']]
        if len(R) < 15:
            continue
        res = np.array([r['res'] for r in R])                 # (n, 5, nk)
        tres = np.array([r['tres'] for r in R])
        ql = np.array([r['q_late'] for r in R])
        cd = np.array([r['chi2dof'] for r in R])

        def med(A):
            with np.errstate(all='ignore'):
                m = np.nanmedian(A, axis=0)
                b = [np.nanmedian(A[rng.integers(0, len(A), len(A))], axis=0) for _ in range(nboot)]
                cnt = np.isfinite(A).sum(0)
            return m, np.nanstd(b, axis=0), cnt
        m, e, n_ = med(res)
        tm, te, tn = med(tres)
        s[arm] = dict(n=len(R), res_med=m.tolist(), res_err=e.tolist(), res_n=n_.tolist(),
                      tres_med=tm.tolist(), tres_err=te.tolist(), tres_n=tn.tolist(),
                      q_late_med=float(np.median(ql)), q_late_mean=float(ql.mean()),
                      q_late_err=float(np.std([rng.choice(ql, len(ql)).mean() for _ in range(nboot)])),
                      chi2dof_med=float(np.median(cd)))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--source', choices=('ntof', 'bench'), default='ntof')
    ap.add_argument('--tan', type=float, default=0.6)
    ap.add_argument('--pin', action='store_true')
    ap.add_argument('--planes', default='x')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--max-planes', type=int, default=None)
    a = ap.parse_args()
    ids = sorted(pickle.load(open(a.cache, 'rb')))
    if a.max_planes:
        ids = ids[:a.max_planes]
    opt = dict(source=a.source, tan=a.tan, pin=a.pin, planes=a.planes)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache, a.bundle, opt)) as pool:
        rows = [r for rs in pool.map(one, ids, chunksize=4) for r in rs]
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(a.bundle)
    out = dict(label=a.label, cache=a.cache, bundle=a.bundle, source=a.source, tan=a.tan, pin=a.pin,
               detector=cal.detector, v=cal.v_drift, gap=GAPS.get(cal.detector), K=int(cal.n_depth_bins),
               k_off=K_OFF.tolist(), t_bins=T_BINS.tolist(), planes={})
    for plane in a.planes:
        R = [r for r in rows if r['plane'] == plane]
        out['planes'][plane] = dict(n_planes=len(R), amp_med=float(np.median([r['amp'] for r in R])) if R else None,
                                    **summarise(R))
        out['planes'][plane]['by_amp'] = {f'{lo}-{hi}': summarise([r for r in R if lo <= r['amp'] < hi])
                                          for lo, hi in AMP_BINS}
        out['planes'][plane]['by_tan'] = {f'{lo}-{hi}': summarise([r for r in R if lo <= abs(r['tan']) < hi])
                                          for lo, hi in TAN_BINS}
        sm = out['planes'][plane]
        for arm in ('prod', 'box'):
            if arm in sm:
                c = np.array(sm[arm]['res_med'])[2]
                print(f'{a.label} {plane} {arm}: n {sm[arm]["n"]} amp {sm["amp_med"]:.0f} chi2/dof '
                      f'{sm[arm]["chi2dof_med"]:.1f} q_late {sm[arm]["q_late_mean"]:.3f}  centre k=-1..+6 '
                      + ' '.join(f'{x:+.3f}' for x in c[3:11]))
    out['rows'] = rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
