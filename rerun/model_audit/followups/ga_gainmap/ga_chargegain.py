#!/usr/bin/env python3
"""ga_chargegain.py -- GA task 3b: the footprint-scale gain GRADIENT, which the in-fit map cannot see, put into the
bundle's per-channel gain slot from the track-charge profile, measured on finalize events DISJOINT from the
calibration cache the arm is evaluated on.

Why a second map: the in-fit ratio data/model is blind to a gain that is linear across a cluster -- the fit absorbs
it as a shifted, equally wide cloud (centroid + a*sigma^2(t)), so chi2 and the ratio both stay flat -- yet that shift
grows with drift time (a * Dp^2 * t) and tilts the fitted slope by a * Dp^2 / v.  Only an external measurement of
ln g(x) (here: median ln(q cos theta) of head-on tracks against the M3 position) can supply it.

  map    : ws_v_w0/<det>.parquet events NOT in the cache, |tan_ref| < 0.15, both views ok, no spark; 1 mm median of
           ln(q cos theta), Gaussian-smoothed 1.5 mm (ga_chargemap's S profile); per channel = value at the strip
           position (cache 'pos'); outside the measured range 1.  Normalised to median 1.  Both views.
  bundle : the promoted bundle with gain_x / gain_y replaced -> ~/x17/cosmic_bench/model_audit/ga/<det>/calib_bundle_qgain
  eval   : heldout_arm.py on the cache events [0, 1500) (the fitter and settings of ga_gainfit.py's base arm)

    ga_chargegain.py --det det4 [--eval] [--jobs 8]   -> results/qgain_<det>.npz (+ heldout rows results/qgain_eval_<det>.npz)
"""
import argparse, json, os, pickle, shutil, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, HERE)
import ga_chargemap as gcm  # noqa: E402

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')


def build(det):
    cache = os.path.join(ROOT, 'ws_t0', 'caches', f'{det}.pkl')
    E = pickle.load(open(cache, 'rb'))
    d = pd.read_parquet(os.path.join(ROOT, 'ws_v_w0', f'{det}.parquet'))
    d = d[~d.index.isin(list(E))]
    d = d[d.x_ok.fillna(False).astype(bool) & d.y_ok.fillna(False).astype(bool) & ~d.spark.astype(bool)]
    out = dict(n_events=len(d))
    gains = {}
    for p, other in (('x', 'y'), ('y', 'x')):
        pos_ch = np.full(512, np.nan)
        for e in E.values():
            if p in e:
                pos_ch[np.asarray(e[p]['ch'], int)] = np.asarray(e[p]['pos'], float)
        t, to = d[f'tan_ref_{p}'].to_numpy(float), d[f'tan_ref_{other}'].to_numpy(float)
        pos = (d[f'ref_mesh_{p}'] + gcm.MID * d[f'tan_ref_{p}']).to_numpy(float)
        q = d[f'{p}_q_sum'].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(pos) & np.isfinite(q) & (q > 0) & (np.abs(t) < gcm.TMAX) & np.isfinite(to)
        l = np.log(q[ok] / np.sqrt(1 + t[ok] ** 2 + to[ok] ** 2)); l -= np.median(l)
        x = pos[ok]
        edges = np.arange(np.floor(np.percentile(x, 0.5)), np.ceil(np.percentile(x, 99.5)) + gcm.BIN, gcm.BIN)
        c = 0.5 * (edges[1:] + edges[:-1])
        G, N = gcm.profile(x, l, edges)
        S = gcm.gsmooth(c, G, gcm.SMALL_MM)
        okS = np.isfinite(S)
        lg = np.where(np.isfinite(pos_ch) & (pos_ch >= c[okS].min()) & (pos_ch <= c[okS].max()),
                      np.interp(np.nan_to_num(pos_ch, nan=0.0), c[okS], S[okS]), 0.0)
        lg -= np.median(lg[np.isfinite(pos_ch)])
        gains[p] = np.exp(np.clip(lg, -3, 3))
        out[p] = dict(n_tracks=int(ok.sum()), rms_lng=float(np.std(lg[np.isfinite(pos_ch)])))
        out[f'pos_{p}'] = pos_ch
    return gains, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--det', required=True); ap.add_argument('--eval', action='store_true')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    gains, info = build(a.det)
    src = os.path.join(ROOT, a.det, 'calib_bundle')
    dst = os.path.join(ROOT, 'ga', a.det, 'calib_bundle_qgain')
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copytree(src, dst)
    arr = dict(np.load(os.path.join(src, 'arrays.npz')))
    assert np.allclose(arr['gain_x'], 1) and np.allclose(arr['gain_y'], 1), 'source bundle already carries a gain map'
    arr['gain_x'], arr['gain_y'] = gains['x'], gains['y']
    np.savez(os.path.join(dst, 'arrays.npz'), **arr)
    bj = os.path.join(dst, 'bundle.json'); b = json.load(open(bj))
    b.setdefault('provenance', {})['gain_map'] = dict(
        tool='rerun/model_audit/followups/ga_gainmap/ga_chargegain.py', source_bundle=src,
        note='STUDY ARM: per-channel gain from the head-on track-charge profile (1.5 mm), finalize events disjoint '
             'from the ws_t0 cache', n_events=info['n_events'])
    json.dump(b, open(bj, 'w'), indent=1)
    np.savez(os.path.join(HERE, 'results', f'qgain_{a.det}.npz'), gain_x=gains['x'], gain_y=gains['y'],
             pos_x=info['pos_x'], pos_y=info['pos_y'])
    print(a.det, {k: v for k, v in info.items() if not k.startswith('pos')}, '->', dst, flush=True)
    if a.eval:
        out = os.path.join(HERE, 'results', f'qgain_eval_{a.det}.npz')
        cmd = [sys.executable, os.path.join(REPO, 'rerun', 'model_audit', 'heldout_arm.py'),
               '--cache', os.path.join(ROOT, 'ws_t0', 'caches', f'{a.det}.pkl'), '--bundle', dst, '--out', out,
               '--start', '0', '--n', '1500', '--jobs', str(a.jobs)]
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
