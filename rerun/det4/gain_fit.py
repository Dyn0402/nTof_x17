#!/usr/bin/env python3
"""gain_fit.py --cache c.pkl --bundle b --out-bundle b_g [--start 580] [--n 920] [--iters 3] [--jobs 8]

Per-channel gain map measured inside the waveform fit, for the bundle's gain_x / gain_y.

wft.model.prep_plane divides each strip's waveform and noise by GAIN[ch]; the chi2 is then sum ((W - g M) / n)^2,
i.e. the model multiplied strip by strip.  Across the resistive strips (X) nothing spreads after the charge lands, so
that multiplier is the amplification gain at pad resolution: det4's X bands (gain x20 inside 12 mm, edges ~5 mm,
memory det4-amplification-stripes) are exactly this.  On Y it is the readout/coupling gain per strip.

Each iteration fits events [start, start + n) of the cache with the current gains (pivot fitter, rerun/seedfit),
and updates g_c *= sum(Wc M / nc^2) / sum(M^2 / nc^2) over the channel's unsaturated samples, Wc and nc the
gain-corrected data and noise, M the model.  Channels with too little model signal keep their gain.  The scale is
free (q absorbs it), so the result is normalised to median 1 over the updated channels.  Use events disjoint from
the evaluation set (resid_map.py defaults to [180, 580)).
"""
import argparse, json, os, pickle, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
MIN_INFO = 50.0                 # sum M^2 / n^2 a channel needs before its gain is updated
_EV = None


def _init(cache, bundle, gains, k):
    global _EV
    sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'seedfit')]
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _EV = pickle.load(open(cache, 'rb'))
    wm.use_calibration(CalibrationBundle.load(bundle)); wm.set_depth_bins(k); wm.COL_GUARD = 0.01
    wm.GAIN = {p: np.asarray(gains[p], float) for p in 'xy'}


def one(eid):
    from wft import model as wm
    import fitters as F
    cal = wm.CAL; ev = _EV[eid]; out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        ft = ev.get(f'ftst_{plane}')
        t0p = (cal.t0_abs or {}).get(plane, {}).get(str(ft), (cal.t0_abs or {}).get(plane, {}).get(ft))
        try:
            r = F.fit_pivot(P, plane, t0p)
        except Exception:
            continue
        W, noise, pos, sat = wm.prep_plane(P, plane)
        chi, q = wm.chi2_plane(plane, W, noise, pos, sat, r['p0'], r['w'], r['t0'], wm.HYPER, snap_t0=False)
        if q is None:
            continue
        M = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], q, wm.HYPER)
        ok = ~sat
        iw = 1.0 / noise[:, None] ** 2
        a = (np.where(ok, W * M, 0) * iw).sum(1)
        b = (np.where(ok, M * M, 0) * iw).sum(1)
        out[plane] = (np.asarray(P['ch'], int).tolist(), a.tolist(), b.tolist(), float(chi), int(ok.sum()))
    return eid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--out-bundle', required=True)
    ap.add_argument('--start', type=int, default=580); ap.add_argument('--n', type=int, default=920)
    ap.add_argument('--iters', type=int, default=3); ap.add_argument('--k', type=int, default=24)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    ids = sorted(pickle.load(open(a.cache, 'rb')))[a.start:a.start + a.n]
    z = np.load(os.path.join(a.bundle, 'arrays.npz'))
    g = {p: np.array(z[f'gain_{p}'], float) for p in 'xy'}
    log = []
    for it in range(a.iters):
        with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, g, a.k)) as pool:
            res = dict(pool.map(one, ids, chunksize=4))
        rec = {}
        for p in 'xy':
            A, B = np.zeros(len(g[p])), np.zeros(len(g[p]))
            chi = dof = 0
            for d in res.values():
                if p in d:
                    ch, aa, bb, c, n = d[p]
                    np.add.at(A, ch, aa); np.add.at(B, ch, bb); chi += c; dof += n
            upd = B > MIN_INFO
            r = np.where(upd, np.clip(A / np.where(upd, B, 1), 0.1, 10.0), 1.0)
            g[p] = g[p] * r
            g[p] /= np.median(g[p][upd])
            rec[p] = dict(n_upd=int(upd.sum()), chi2_dof=chi / max(dof, 1), step_rms=float(np.std(np.log(r[upd]))),
                          lng_rms=float(np.std(np.log(np.clip(g[p][upd], 1e-3, None)))))
        log.append(rec); print('iter', it, json.dumps(rec), flush=True)
    shutil.rmtree(a.out_bundle, ignore_errors=True); shutil.copytree(a.bundle, a.out_bundle)
    arr = dict(np.load(os.path.join(a.bundle, 'arrays.npz')))
    arr['gain_x'], arr['gain_y'] = g['x'], g['y']
    np.savez(os.path.join(a.out_bundle, 'arrays.npz'), **arr)
    bj = os.path.join(a.out_bundle, 'bundle.json'); b = json.load(open(bj))
    b.setdefault('provenance', {})['gain_map'] = dict(
        tool='rerun/det4/gain_fit.py', source_bundle=a.bundle, cache=a.cache, events=[a.start, a.start + a.n],
        iters=log, note='per-channel model multiplier fitted inside the waveform fit; disjoint from resid_map [180,580)')
    json.dump(b, open(bj, 'w'), indent=1)
    print('wrote', a.out_bundle)


if __name__ == '__main__':
    main()
