#!/usr/bin/env python3
"""zs_treatment.py --plateau raw700 [--nev 400] [--zs 4] [--jobs 4] [--chunk I N] [--out DIR]

How should zero-suppressed samples enter the waveform fit?  Truth from RAW: run_71 head-on events (sps_cache.py), each
fitted by the production fitter (wft.reco.fit_plane: pivot start, sigma0 profile; det4 promoted bench bundle, dead list
dropped, K 60, 64 samples) four ways on the SAME event:
  raw    the RAW waveforms, the FEU's dropped sample groups (exact 0) censored            -- the reference
  zero   DREAM ZS emulated at k sigma (samples < k*noise -> 0), zeros fitted as measured 0 -- WFT_ZS_CENSOR=0
  cens   the same ZS waveforms, zeros censored (no information either way)                 -- WFT_ZS_CENSOR=1 (F16/F20)
  bound  the same ZS waveforms, zeros censored AND bounded above by k*noise                -- WFT_ZS_BOUND=k (new)
Per arm: w (head-on: should be the chamber's tilt, X ~ -0.015*v, Y ~ 0), p0, t0, chi2/dof; paired against raw.
Output <out>/zs_treatment_<plateau>_zs<k>[_cII].pkl; 'collect' merges and prints.
"""
import argparse, glob, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
DATA = os.environ.get('WS_SPS_DATA', '/media/dylan/data/x17/cosmic_bench/outliers/sps')
BUNDLE = os.environ.get('WS_SPS_BUNDLE', os.path.expanduser('~/x17/cosmic_bench/model_audit/det4/calib_bundle'))
ARMS = ('raw', 'zero', 'cens', 'bound')
_EV = None; _CAL = None; _K = 4.0


def _init(plateau, k):
    global _EV, _CAL, _K
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = pickle.load(open(os.path.join(DATA, f'sps_run71_{plateau}.pkl'), 'rb'))
    cal = CalibrationBundle.load(BUNDLE); cal.dead, cal.hot = {}, {}
    wm.use_calibration(cal); wm.set_depth_bins(60); wm.set_nsamp(64); wm.COL_GUARD = 0.01; wr.PIVOT_START = True
    _CAL = cal; _K = k


def one(eid):
    from wft import model as wm, reco as wr
    e = _EV[eid]; out = {'eid': int(eid)}
    for v in 'xy':
        P0 = e[v]; Wr = np.asarray(P0['W'], float); noise = np.asarray(P0['noise'], float)
        drop = Wr == 0.0                                  # the RAW packet loss
        Wz = np.where(Wr < _K * noise[:, None], 0.0, Wr)  # sample-level ZS
        res = {}
        for arm in ARMS:
            W = Wr if arm == 'raw' else Wz
            wm.ZS_CENSOR = arm in ('cens', 'bound')
            wm.ZS_BOUND = _K if arm == 'bound' else 0.0
            # the RAW drops are missing in every arm (real ZS data has none; censored, never bounded)
            P = dict(P0); P['W'] = W; P['missing'] = drop
            try:
                f = wr.fit_plane(P, v, _CAL)
            except Exception:
                f = None
            if f is None:
                continue
            res[arm] = dict(w=f.w * 1e3, p0=f.p0, t0=f.t0, chi2=f.chi2, dof=f.dof, s0=float(f._sigma_p0), uend=f.q_uend)
        out[v] = res
        out[v]['amax'] = float(Wr.max()); out[v]['ref'] = float(e[f'ref_mesh_{v}'])
        out[v]['n_kept'] = int((Wz > 0).sum()); out[v]['n_drop'] = int(drop.sum())
    wm.ZS_CENSOR = False; wm.ZS_BOUND = 0.0
    return out


def rsig(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def report(R, tag):
    print(f'{tag}: {len(R)} events')
    for v in 'xy':
        rows = [r[v] for r in R if v in r and all(a in r[v] for a in ARMS)]
        ref = np.array([r['raw']['w'] for r in rows])
        print(f'  {v}: n {len(rows)}  RAW w median {np.median(ref):+.3f} rsig {rsig(ref):.3f} um/ns')
        for a in ARMS[1:]:
            w = np.array([r[a]['w'] for r in rows]); dw = w - ref
            dp = np.array([r[a]['p0'] - r['raw']['p0'] for r in rows]); dt = np.array([r[a]['t0'] - r['raw']['t0'] for r in rows])
            c = np.array([r[a]['chi2'] / max(r[a]['dof'], 1) for r in rows])
            far = np.mean(np.abs(dw) > 2.0)
            print(f'    {a:5s}: w median {np.median(w):+.3f} rsig {rsig(w):.3f} | vs RAW: dw med {np.median(dw):+.3f} rsig {rsig(dw):.3f} '
                  f'|dw|>2 {far:.3f}  dp0 rsig {rsig(dp) * 1e3:.0f} um  dt0 med {np.median(dt):+.1f} rsig {rsig(dt):.1f} ns  chi2/dof {np.median(c):.1f}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['fit', 'collect'])
    ap.add_argument('--plateau', default='raw700'); ap.add_argument('--zs', type=float, default=4.0)
    ap.add_argument('--nev', type=int, default=600); ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--chunk', type=int, nargs=2, default=None); ap.add_argument('--out', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tag = f'zs_treatment_{a.plateau}_zs{a.zs:g}'
    if a.cmd == 'collect':
        R = []
        for f in sorted(glob.glob(os.path.join(a.out, f'{tag}_c*.pkl'))):
            R += pickle.load(open(f, 'rb'))
        pickle.dump(R, open(os.path.join(a.out, f'{tag}.pkl'), 'wb')); report(R, tag); return
    E = pickle.load(open(os.path.join(DATA, f'sps_run71_{a.plateau}.pkl'), 'rb'))
    ids = [k for k in sorted(E) if all(np.asarray(E[k][v]['W']).shape[0] == 9 and 250 <= np.asarray(E[k][v]['W']).max() <= 3200
                                       for v in 'xy')][:a.nev]
    del E
    if a.chunk:
        ids = ids[a.chunk[0]::a.chunk[1]]; tag += f'_c{a.chunk[0]:02d}'
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.plateau, a.zs)) as pool:
        R = list(pool.map(one, ids, chunksize=2))
    pickle.dump(R, open(os.path.join(a.out, f'{tag}.pkl'), 'wb'))
    if not a.chunk:
        report(R, tag)


if __name__ == '__main__':
    main()
