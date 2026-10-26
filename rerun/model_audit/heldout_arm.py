#!/usr/bin/env python3
"""heldout_arm.py --cache c.pkl --bundle b --out o.npz [--start 180 --n 1320] [--k 24] [--jobs 8]
                  [--hyper '{"sigma_p0_x": 0.2}'] [--v V] [--nsamp N] [--two-sided] [--t0-fix] [--tag T]

Held-out events of a calibration cache through the PASS's own fitter (wft.reco.fit_plane: pivot start, sigma_p0
profile, column guard), one arm = one bundle (+ hyper overrides, + WFT_* environment read by wft.model at import).
Writes an npz whose ``per`` rows are compare.py-compatible: (eid, view, tan_ref, w[um/ns], chi2, dof, wmax, nsat)
followed by (t0, p0, ref_mesh, sigma_p0_used, tan_fit_bundle, tan_err, q_uend, q_sum), and a JSON summary beside it.
Environment arms: WFT_RC_T0_INTERP, WFT_MODEL_FRAC, WFT_SAT_SOLVE, WFT_ZS_CENSOR (set in the calling environment).
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
_EV = None; _OPT = None


def _init(cache, bundle, k, hyper_over, v, nsamp, two_sided, t0fix, no_dead=False):
    global _EV, _OPT
    sys.path.insert(0, REPO)
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    if v:
        cal.v_drift = float(v)
    cal.hyper.update(hyper_over or {})
    if no_dead:
        cal.dead, cal.hot = {}, {}       # bench channel numbering does not apply off the bench (SPS FEU 03)
    wm.use_calibration(cal); wm.set_depth_bins(k)
    if wm.COL_GUARD <= 0:
        wm.COL_GUARD = 0.01
    if nsamp:
        wm.set_nsamp(nsamp)
    wr.PIVOT_START = True
    wr.TWO_SIDED = bool(two_sided)
    _OPT = dict(t0fix=t0fix)


def one(eid):
    from wft import model as wm, reco as wr
    cal = wm.CAL; ev = _EV[eid]; rows = []
    for vi, plane in enumerate('xy'):
        if plane not in ev:
            continue
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        tr = float(ev[f'tan_{plane}']); ref = float(ev.get(f'ref_mesh_{plane}', np.nan))
        prior = None
        if _OPT['t0fix']:
            ft = ev.get(f'ftst_{plane}')
            t0a = (cal.t0_abs or {}).get(plane, {})
            pred = t0a.get(int(ft)) if ft is not None else None
            if pred is not None:
                prior = (float(pred), 0.05)        # effectively pins t0 at the trigger prediction
        try:
            f = wr.fit_plane(P, plane, cal, t0_prior=prior)
        except Exception:
            f = None
        if f is None:
            continue
        nsat = int((W >= wm.SAT).sum())
        rows.append((int(eid), vi, tr, f.w * 1e3, f.chi2, f.dof, float(W.max()), nsat,
                     f.t0, f.p0, ref, float(getattr(f, '_sigma_p0', np.nan)), f.tan_theta, f.tan_err, f.q_uend, f.q_sum))
    return rows


def summarize(per, v_lad=None):
    out = {}
    for vi, view in enumerate('xy'):
        R = per[per[:, 1] == vi]
        if len(R) < 20:
            continue
        tr, w = R[:, 2], R[:, 3]
        m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
        c = np.median(w[np.abs(tr) < 0.05]) if (np.abs(tr) < 0.05).sum() > 20 else 0.0
        s = float(np.median((w[m] - c) / tr[m])) if m.sum() > 30 else (v_lad or np.nan)
        d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5
        S = dict(n=int(len(R)), ladder=s, offset=float(c), far_pct=float(100 * far.mean()), chi2dof_p50=float(np.median(R[:, 4] / np.maximum(R[:, 5], 1))))
        for cl, mm in (('all', np.ones_like(far)), ('headon', np.abs(tr) < 0.1), ('inclined', np.abs(tr) >= 0.1), ('steep', np.abs(tr) >= 0.28),
                       ('steep_pos', tr >= 0.28), ('steep_neg', tr <= -0.28)):
            k = mm & ~far
            S[cl] = dict(n=int(mm.sum()), s68=float(np.percentile(np.abs(d[k] - np.median(d[k])), 68)) if k.sum() > 10 else np.nan,
                         bias=float(np.median(d[k])) if k.sum() > 5 else np.nan, far_pct=float(100 * far[mm].mean()) if mm.sum() else np.nan)
        out[view] = S
    # x-y t0 coincidence on events with both views
    X = {int(r[0]): r for r in per[per[:, 1] == 0]}; Y = {int(r[0]): r for r in per[per[:, 1] == 1]}
    both = sorted(set(X) & set(Y))
    if len(both) > 20:
        dt = np.array([X[e][8] - Y[e][8] for e in both])
        out['xy'] = dict(n=len(both), dt0_median=float(np.median(dt)), dt0_rsig=float(1.4826 * np.median(np.abs(dt - np.median(dt)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1320); ap.add_argument('--start', type=int, default=180)
    ap.add_argument('--k', type=int, default=24); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--hyper', default=None, help='JSON dict of hyper overrides')
    ap.add_argument('--v', type=float, default=None); ap.add_argument('--nsamp', type=int, default=None)
    ap.add_argument('--two-sided', action='store_true'); ap.add_argument('--t0-fix', action='store_true')
    ap.add_argument('--no-dead', action='store_true', help='drop the bundle dead/hot lists (channel numbering of another cabling)')
    ap.add_argument('--ids', default=None, help='JSON list of event ids (overrides --start/--n)')
    a = ap.parse_args()
    hy = json.loads(a.hyper) if a.hyper else {}
    C = pickle.load(open(a.cache, 'rb'))
    ids = json.load(open(a.ids)) if a.ids else sorted(C)[a.start:a.start + a.n]
    del C
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.k, hy, a.v, a.nsamp, a.two_sided, a.t0_fix, a.no_dead)) as pool:
        rows = [r for rs in pool.map(one, ids, chunksize=2) for r in rs]
    per = np.array(rows, float)
    env = {k: v for k, v in os.environ.items() if k.startswith('WFT_')}
    summ = summarize(per)
    np.savez(a.out, per=per, bundle=a.bundle, hyper=json.dumps(hy), env=json.dumps(env), args=json.dumps(vars(a)))
    json.dump(dict(args=vars(a), env=env, summary=summ), open(a.out.replace('.npz', '.json'), 'w'), indent=1)
    print(a.out, json.dumps({v: {k: summ[v][k] for k in ('n', 'ladder', 'far_pct', 'chi2dof_p50')} for v in summ if v in ('x', 'y')}),
          json.dumps({v: {c: round(summ[v][c]['s68'], 3) for c in ('all', 'headon', 'inclined', 'steep')} for v in summ if v in ('x', 'y')}),
          summ.get('xy'))


if __name__ == '__main__':
    main()
