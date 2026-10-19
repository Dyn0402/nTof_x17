#!/usr/bin/env python3
"""sat_closure.py -- how much does clipping cost the per-event forward fit, and is censoring the best treatment?
(FINDINGS §41)

Unsaturated bench events (X window maximum 1500-3400 ADC) are clipped artificially at a fraction f of their own
maximum, so the truth (the unclipped waveform) is known, and refitted four ways (production bundle, ref start,
held-out events as in chi2_split.py):

  ref    the unclipped event (the truth for the comparison)
  cens   clipped, samples at the clip censored: left out of the charge solve, one-sided penalty in chi2 (production)
  cens2  as cens, with the bound model >= clip enforced inside the charge solve (wft.model.SAT_SOLVE)
  naive  clipped, the clipped samples taken at face value
  drop   clipped, every strip with a clipped sample removed

Per arm: the fitted angle and position against the ref arm (the cost of clipping) and against the M3 reference
(the resolution), and the total fitted charge against the ref arm.  The clip level is set so the clipped runs
are as long as the real ones (det3: median 4 samples).

    sat_closure.py <det> [--fracs 0.4,0.6] [--max-held 600]  -> results/sat_closure_<det>.json
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
sys.path[:0] = [HERE, REPO, os.path.join(HERE, '..', 'plane_ratio')]
import plane_bench as PB                                     # noqa: E402
from chi2_split import SETS, A                               # noqa: E402

ARMS = ('ref', 'cens', 'cens2', 'naive', 'drop')
LO, HI = 1500.0, 3400.0


def one(payload):
    eid, hyper, v, t0abs, fracs = payload
    from wft import model as wm
    ev = PB._EV[eid]
    P = ev.get('x')
    if P is None:
        return eid, None
    W = np.asarray(P['W'], float)
    if not (LO <= W.max() < HI):
        return eid, None
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    ft = ev['ftst_x']
    if ft not in t0abs['x']:
        return eid, None
    t0p = t0abs['x'][ft]
    p0r, tr = ev['ref_mesh_x'], ev['tan_x']
    sat0 = wm.SAT

    def fit(Pp, sat_level, solve=False):
        wm.SAT = sat_level
        wm.SAT_SOLVE = solve
        try:
            r = wm.fit_plane_raw(Pp, 'x', p0r, tr * v * 1e-3, t0p, hyper=hyper, t0_prior=(t0p, PB.T0_SIGMA))
        except Exception:
            return None
        finally:
            wm.SAT = sat0
            wm.SAT_SOLVE = False
        if not np.isfinite(r['chi2']):
            return None
        return dict(tan=float(r['w'] * 1e3 / v), p0=float(r['p0']), q=float(np.sum(r['q'])), chi2=float(r['chi2'] / max(r['dof'], 1)))

    out = dict(tr=float(tr), p0r=float(p0r), wmax=float(W.max()), ref=fit(P, sat0))
    if out['ref'] is None:
        return eid, None
    for f in fracs:
        C = f * W.max()
        Wc = np.minimum(W, C)
        clipped = W >= C
        runs = [int(m.sum()) for m in clipped if m.any()]
        Pc = dict(P); Pc['W'] = Wc
        keep = ~clipped.any(1)
        Pd = {k: (np.asarray(val)[keep] if hasattr(val, '__len__') and len(val) == len(W) else val) for k, val in P.items()}
        out[f'{f:g}'] = dict(nclip_strips=len(runs), run_len=float(np.median(runs)) if runs else 0.0,
                             cens=fit(Pc, C - 1e-6), cens2=fit(Pc, C - 1e-6, True), naive=fit(Pc, sat0),
                             drop=fit(Pd, sat0) if keep.sum() >= 3 else None)
    return eid, out


def summarise(res, fracs):
    rows = {}
    ok = [r for r in res.values() if r]
    for f in fracs:
        k = f'{f:g}'
        R = {}
        for arm in ('cens', 'cens2', 'naive', 'drop'):
            sel = [r for r in ok if r[k][arm] is not None]
            dt = np.array([r[k][arm]['tan'] - r['ref']['tan'] for r in sel])
            dp = np.array([r[k][arm]['p0'] - r['ref']['p0'] for r in sel])
            dq = np.array([r[k][arm]['q'] / r['ref']['q'] - 1 for r in sel])
            res_ref = np.array([r[k][arm]['tan'] - r['tr'] for r in sel])
            R[arm] = dict(n=len(sel), dtan_mean=float(dt.mean()), dtan_s68=float(PB.s68(dt)), dp0_mean=float(dp.mean()),
                          dp0_s68=float(PB.s68(dp)), dq_median=float(np.median(dq)), dq_s68=float(PB.s68(dq)),
                          res_s68=float(PB.s68(res_ref)), res_mean=float(res_ref.mean()))
        r0 = np.array([r['ref']['tan'] - r['tr'] for r in ok])
        rows[k] = dict(n=len(ok), run_len_median=float(np.median([r[k]['run_len'] for r in ok])),
                       nclip_median=float(np.median([r[k]['nclip_strips'] for r in ok])),
                       ref_res_s68=float(PB.s68(r0)), ref_res_mean=float(r0.mean()), arms=R)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('det')
    ap.add_argument('--fracs', default='0.4,0.6')
    ap.add_argument('--max-held', type=int, default=1500)
    ap.add_argument('--jobs', type=int, default=14)
    a = ap.parse_args()
    fracs = [float(x) for x in a.fracs.split(',')]
    d, bun, train = SETS[a.det]
    W = f'{A}/{d}/wft'
    cache = f'{W}/plane_ratio/big_cache_3000.pkl'
    tcache = f'{W}/calib_work/calib_cache.pkl' if train == 'calib' else cache
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle
    os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')
    cal = CalibrationBundle.load(f'{W}/{bun}')
    v = float(cal.v_drift)
    h = {k: float(q) for k, q in cal.hyper.items() if k != 'kTauY'}
    tev = pickle.load(open(tcache, 'rb'))
    train_ids = sorted(tev)[:180]
    t0abs, _ = wc.measure_t0_abs({e: tev[e] for e in train_ids}, f'{W}/{bun}', h, v)
    big = pickle.load(open(cache, 'rb'))
    held = [e for e in sorted(big) if e not in set(train_ids)][:a.max_held]
    del big
    with ProcessPoolExecutor(a.jobs, initializer=PB._init, initargs=(cache, f'{W}/{bun}', 0.0)) as pool:
        res = {int(e): o for e, o in pool.map(one, [(e, h, v, t0abs, fracs) for e in held], chunksize=8)}
    rows = summarise(res, fracs)
    for k, r in rows.items():
        print(f'{a.det} clip at {k} x max: n {r["n"]}, clipped strips {r["nclip_median"]:.0f}, run {r["run_len_median"]:.0f} samples; '
              f'unclipped res s68 {r["ref_res_s68"]:.4f}')
        for arm, q in r['arms'].items():
            print(f'   {arm:5s} n {q["n"]:4d}  dtan {q["dtan_mean"]:+.4f} s68 {q["dtan_s68"]:.4f}  dp0 {q["dp0_mean"]:+.3f} s68 {q["dp0_s68"]:.3f} mm  '
                  f'dq {q["dq_median"]:+.3f} s68 {q["dq_s68"]:.3f}  | vs M3: s68 {q["res_s68"]:.4f}', flush=True)
    json.dump(dict(det=a.det, fracs=fracs, window=[LO, HI], rows=rows), open(f'{HERE}/results/sat_closure_{a.det}.json', 'w'),
              indent=1, default=float)


if __name__ == '__main__':
    main()
