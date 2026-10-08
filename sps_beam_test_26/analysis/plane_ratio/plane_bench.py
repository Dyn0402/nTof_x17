#!/usr/bin/env python3
"""plane_bench.py -- score per-view kernel arms against production, held-out.

The 18_ladder_bench protocol on a larger cache:

  * each arm gets its OWN absolute-t0 table, measured ref-pinned on the
    calibration training events under its own kernel (a table from another
    kernel puts the pulse elsewhere and the 5 ns prior then drags the fit),
  * free (p0, w, t0) fits on HELD-OUT events only (the big cache minus the
    calibration training ids), against the M3 reference.  Two starts:
      ref   -- seeded at the reference track, as 18_ladder_bench does
               (comparable with history; flatters absolute numbers),
      blind -- seeded by init_guess from the waveforms alone (tan 0),
               closer to what production sees,
  * per plane: robust sigma_theta, the |theta| < 5 deg band, angle slope,
    bias, p0 residual, implied-v flatness (v * w_fit / w_ref in |tan| bins,
    spread = max - min of the bin medians), fit chi2/dof,
  * PAIRED bootstrap against the first arm (production) on the events every
    arm reconstructed.

    plane_bench.py --cache big_cache.pkl --train-cache calib_cache.pkl \
        --bundle calib_bundle_r06 --arms arms/*.json --out bench.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)

T0_SIGMA = 5.0
TAN_BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7)
_EV = None


def _init(cache, bundle):
    global _EV
    os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')   # diag arms only
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle))


def _geo(payload):
    eid, hyper, v, t0abs, start = payload
    from wft import model as wm
    ev = _EV[eid]
    out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        ft = ev[f'ftst_{plane}']
        if ft not in t0abs[plane]:
            continue
        t0p = t0abs[plane][ft]
        p0r, tr = ev[f'ref_mesh_{plane}'], ev[f'tan_{plane}']
        if start == 'ref':
            p0i, wi = p0r, tr * v * 1e-3
        else:
            p0i, wi, _t = wm.init_guess(P, plane, 0.0, None, v)
        try:
            r = wm.fit_plane_raw(P, plane, p0i, wi, t0p, hyper=hyper,
                                 t0_prior=(t0p, T0_SIGMA))
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        out[plane] = (float(tr), float(r['w'] * 1e3 / v), float(p0r),
                      float(r['p0']), float(r['chi2'] / max(r['dof'], 1)))
    return eid, out


def rsig(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return float(0.7413 * (np.percentile(x, 75) - np.percentile(x, 25)))


def s68(x):
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 68.27)) if len(x) else np.nan


def metrics(a):
    """a: (n, 5) of (tan_ref, tan_fit, p0_ref, p0_fit, chi2dof)."""
    tr, tf, p0r, p0f, cd = a.T
    d = tf - tr
    k = np.abs(d) < 0.15                          # 18_ladder_bench's core
    dth = np.degrees(np.arctan(tf)) - np.degrees(np.arctan(tr))
    head = np.abs(np.degrees(np.arctan(tr))) < 5.0
    out = dict(n=int(len(a)), n_core=int(k.sum()),
               sig_theta=float(np.degrees(np.arctan(rsig(d[k])))),
               s68_deg=s68(dth[k]),
               s68_head_deg=s68(dth[k & head]),
               bias_deg=float(np.median(dth[k])),
               slope=float(np.polyfit(tr[k], tf[k], 1)[0]),
               sig_p0=rsig((p0f - p0r)[k]),
               chi2dof=float(np.median(cd)),
               out=float(1 - k.mean()))
    # implied-v flatness: v_fit/v_ref = tan_fit/tan_ref, in |tan_ref| bins
    at = np.abs(tr)
    meds = []
    for lo, hi in zip(TAN_BINS[1:-1], TAN_BINS[2:]):
        m = k & (at >= lo) & (at < hi)
        if m.sum() >= 15:
            meds.append(float(np.median(tf[m] / tr[m])))
    out['vratio_bins'] = meds
    out['vratio_spread'] = float(max(meds) - min(meds)) if len(meds) > 1 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--train-cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--arms', nargs='+', required=True,
                    help='recal.py json outputs; production is scored first')
    ap.add_argument('--n-train', type=int, default=180)
    ap.add_argument('--max-held', type=int, default=2000)
    ap.add_argument('--starts', default='ref,blind')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    from wft import calibrate as wc
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(a.bundle)
    v = float(cal.v_drift)
    with open(a.train_cache, 'rb') as f:
        tev = pickle.load(f)
    train_ids = sorted(tev)[:a.n_train]
    train = {e: tev[e] for e in train_ids}
    with open(a.cache, 'rb') as f:
        big = pickle.load(f)
    held = [e for e in sorted(big) if e not in set(train_ids)][:a.max_held]
    print(f'[bench] v={v:.2f}  train {len(train)}  held-out {len(held)}', flush=True)

    arms = {'production': {k: float(q) for k, q in cal.hyper.items()
                           if k != 'kTauY'}}
    for p in a.arms:
        r = json.load(open(p))
        arms[r['arm']] = {k: float(q) for k, q in r['hyper'].items()}

    starts = a.starts.split(',')
    rows, resid = {}, {}
    for name, h in arms.items():
        t0 = time.time()
        os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')
        t0abs, _ = wc.measure_t0_abs(train, a.bundle, h, v)
        rows[name] = dict(hyper=h)
        with ProcessPoolExecutor(a.jobs, initializer=_init,
                                 initargs=(a.cache, a.bundle)) as pool:
            for st in starts:
                got = {'x': {}, 'y': {}}
                for e, o in pool.map(_geo, [(e, h, v, t0abs, st) for e in held],
                                     chunksize=8):
                    for p, tup in o.items():
                        got[p][e] = tup
                resid[(name, st)] = got
                rows[name][st] = {p: metrics(np.array(list(got[p].values())))
                                  for p in ('x', 'y')}
                for p in ('x', 'y'):
                    q = rows[name][st][p]
                    print(f'{name:13} {st:5} {p}: s68 {q["s68_deg"]:.3f} '
                          f'(head {q["s68_head_deg"]:.3f})  rsig {q["sig_theta"]:.3f}  '
                          f'slope {q["slope"]:.4f}  bias {q["bias_deg"]:+.3f}  '
                          f'sig_p0 {q["sig_p0"]:.3f}  vspread {q["vratio_spread"]:.3f}  '
                          f'chi2/dof {q["chi2dof"]:.0f}  out {100 * q["out"]:.1f} %',
                          flush=True)
        print(f'   ({time.time() - t0:.0f} s)', flush=True)

    # ---- paired bootstrap vs production --------------------------------
    rng = np.random.default_rng(20261009)
    names = list(arms)
    for st in starts:
        for p in ('x', 'y'):
            common = set(resid[(names[0], st)][p])
            for n in names[1:]:
                common &= set(resid[(n, st)][p])
            common = sorted(common)
            A = {n: np.array([resid[(n, st)][p][e] for e in common]) for n in names}
            keep = np.ones(len(common), bool)
            for n in names:
                keep &= np.abs(A[n][:, 1] - A[n][:, 0]) < 0.15
            nk = int(keep.sum())
            bi = rng.integers(0, nk, size=(a.nboot, nk))

            def stats(arr):
                tr, tf = arr[:, 0], arr[:, 1]
                dth = np.degrees(np.arctan(tf)) - np.degrees(np.arctan(tr))
                head = np.abs(np.degrees(np.arctan(tr))) < 5.0
                s = np.array([s68(dth[i]) for i in bi])
                sh = np.array([s68(dth[i][head[i]]) for i in bi])
                return s, sh
            base = stats(A[names[0]][keep])
            for n in names[1:]:
                s, sh = stats(A[n][keep])
                d, dh = s - base[0], sh - base[1]
                rows[n].setdefault('paired', {}).setdefault(st, {})[p] = dict(
                    n=nk, d_s68=float(d.mean()), d_s68_err=float(d.std()),
                    d_s68_head=float(dh.mean()), d_s68_head_err=float(dh.std()))
    print('\nPaired s68 difference vs production (negative = better):')
    for n in names[1:]:
        for st in starts:
            for p in ('x', 'y'):
                q = rows[n]['paired'][st][p]
                print(f'  {n:13} {st:5} {p}: all {q["d_s68"]:+.3f} ± {q["d_s68_err"]:.3f}'
                      f'   head-on {q["d_s68_head"]:+.3f} ± {q["d_s68_head_err"]:.3f}'
                      f'   (n={q["n"]})')
    with open(a.out, 'w') as f:
        json.dump(rows, f, indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
