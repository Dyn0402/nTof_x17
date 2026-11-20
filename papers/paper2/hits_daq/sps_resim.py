#!/usr/bin/env python3
"""sps_resim.py <sps window pickle(s)> --bundle B --out OUT.parquet [--arms SPEC] [--nmax N] [--jobs N]
-- Arm S (resim.py) on the SPS known-angle rows: why the beam rows read steeper than the bench law.

Chamber E on the 25.4 deg stage in Ar/CO2/iso reads a hits-ladder / waveform-fit slope ratio of 1.07-1.13 on the inclined
(Y) view, against 1.03-1.05 from the bench law 1 + C/N^2 with the bench C_Y (DAG_SCORES_2026-11-13.md).  The gas is slow
there (v = 7.5-10 um/ns against 33-36 on the bench), so the charge column lasts 2.8-3.7 us instead of 0.8 us, and the RC
spreading along the resistive strips (sigma^2 += 2 D_RC s, s the time since landing) has that much longer to act.  This
script tests that reading on the planes themselves: every inclined plane is regenerated through the pass model with a
modified kernel, in two modes,

  res    W' = M(p0, w, t0, q; hyper') + (W - M(hyper))    the plane's own residual added back; 'base' is then the data
  noise  W' = M(p0, w, t0, q; hyper') + white noise       the pure model: does the kernel alone produce the ratio?
  zs     the 'noise' window with the data's suppressed samples (W == 0 on these ZS rows) set to 0

and the DREAM leading-edge ladder (emulate.dream, tiers.ladder_u, as the V0 tier) is fitted on W'.  The score is the
median u / w over the planes (w the generating slope, so no reference and no drift velocity), per row, arm and mode.
The windows are padded by PAD strips as in resim.py.  Arms: resim.parse_variants SPEC (default: the kernel taken apart).
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from resim import PAD, padded, parse_variants  # noqa: E402
from tiers import ladder_u  # noqa: E402

# 'gas' = the Magboltz diffusion of the beam's wet Ar/CO2/iso, in place of the bench bundle's 0.020 the pass rows were
# fitted with (sps_dp_scan.py).  'Dp=gas' is resolved per row by gas_dp(): Ar/CO2/iso 93/5/2 (the ATLAS NSW gas,
# docs/GAS_MIXTURES.md) at the row's gas-fit composition, 0.0049-0.0054.  Until 2026-11-20 it was a fixed 0.006, from the
# 95/3/2 grid (a wrong mixture label).
ARMS = ('base:,rc0:rc_D_y=0,rc2:rc_D_y=x2,dp0:Dp=1e-6,gas:Dp=gas,gasrc0:Dp=gas;rc_D_y=0,'
        'rc0dp0:rc_D_y=0;Dp=1e-6,point:rc_D_y=0;Dp=1e-6;sigma_p0=0.07')
GAP_REGISTRY_CM = 2.88                     # the registry's field labels: HV / 28.8 mm
GASFIT_GROUP = {'run57': 'co2_run57', 'run59': 'co2_run59'}
G = {}


def row_field(key):
    """drift field of an SPS CO2 known-angle row [V/cm], the registry convention (HV / 28.8 mm); run59 is at 700 V."""
    hv = float(key.split('__d')[1]) if '__d' in key else 700.0
    return hv / GAP_REGISTRY_CM


def gas_dp(meta, planes):
    """Dp [mm/sqrt(ns)] of the row's gas: sigma^2 = D_T^2 z = D_T^2 v t, so Dp = D_T sqrt(v), with D_T from Magboltz
    (cloud_basics GasGrid 'co2' on air_merged, Ar/CO2/iso 93/5/2) at the row's water and air from the SPS gas fit
    (rerun/sps_gasfit/results/verdict.json) and v the row's own drift velocity (median |w| / tan(stage)), since the
    diffusion acts over the measured drift time."""
    import json
    repo = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
    sys.path.insert(0, os.path.join(repo, 'sps_beam_test_26', 'analysis', 'cloud_basics'))
    import gasmodel
    V = json.load(open(os.path.join(repo, 'rerun', 'sps_gasfit', 'results', 'verdict.json')))
    grp = GASFIT_GROUP[meta['key'].split('_')[0]]
    c = next(t for t in V['table'] if t['group'] == grp)
    E = row_field(meta['key'])
    DT = gasmodel.GasGrid('co2', 'air_merged')(c['water'], c['air'], E)['DT']          # um / sqrt(cm)
    w = np.median([abs(p['w']) for p in planes if p['view'] == 'y'])                     # mm / ns
    v = w / np.tan(np.radians(float(meta['stage_deg'])))                                 # mm / ns
    dp = DT * 1e-3 / np.sqrt(10.0) * np.sqrt(v)                                          # mm/sqrt(mm) * sqrt(mm/ns)
    return dict(Dp=float(dp), DT_um_rtcm=float(DT), v_um_ns=float(v * 1e3), E_Vcm=float(E), group=grp,
                water=float(c['water']), air=float(c['air']), o2_ppm=float(c['o2_ppm']))


def init(bundle, meta, K, spec, nsig):
    from wft import model as wm
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(bundle)
    cal.v_drift = meta['v_drift']
    cal.bow, cal.v_map, cal.w0 = {}, {}, {}          # as sps_reco._init: bench mounting terms never apply at H4
    wm.use_calibration(cal)
    wm.set_nsamp(meta['nsamp'])
    wm.set_depth_bins(K)
    E.SNS = float(meta.get('sns', 60.0))
    G.update(wm=wm, V=parse_variants(spec, dict(meta.get('hyper') or cal.hyper)), nsig=nsig, key=meta['key'])


def one(p):
    wm, v = G['wm'], p['view']
    rng = np.random.default_rng(p['event_id'] * 2 + (v == 'y'))
    g = p['gain'].astype(float)
    R = p['W'].astype(float) - p['M'].astype(float)          # sps_extract windows are in ADC (model times the gain)
    pos2, nz2, g2 = padded(p, rng)
    ns = R.shape[1]
    R2 = np.vstack([rng.normal(0, 1, (PAD, ns)) * nz2[0], R, rng.normal(0, 1, (PAD, ns)) * nz2[-1]])
    N2 = rng.normal(0, 1, (len(pos2), ns)) * nz2[:, None]
    q = p['q'].astype(float)
    Z2 = np.vstack([np.ones((PAD, ns), bool), p['W'] == 0, np.ones((PAD, ns), bool)])
    rows = []
    for name, hy in G['V'].items():
        M2 = wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'], q, hy) * g2[:, None]
        for mode, W2 in (('res', M2 + R2), ('noise', M2 + N2), ('zs', np.where(Z2, 0.0, M2 + N2))):
            h = E.dream(W2, nz2, nsig=G['nsig'])
            u, n = ladder_u(pos2, h['t'], h['a'])
            ur, nr = ladder_u(pos2, h['t'], h['a'], robust=True)
            rows.append(dict(row=G['key'], event_id=p['event_id'], view=v, arm=name, mode=mode, w_true=p['w'], u=u, ur=ur,
                             n_hits=n, n_rob=nr, t_span=float(np.nanmax(h['t']) - np.nanmin(h['t'])) if n > 1 else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default=ARMS)
    ap.add_argument('--views', default='y')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--nsig', type=float, default=5.0)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--summary', help='also write the per-row, per-arm summary JSON (results/sps_resim.json)')
    a = ap.parse_args()
    out, gas = [], {}
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = D['meta']
        sel = [p for p in D['planes'] if p['view'] in a.views and p.get('q') is not None]
        if a.nmax:
            sel = sel[:a.nmax]
        K = len(sel[0]['q'])
        spec = a.arms
        if 'Dp=gas' in spec:
            gas[meta['key']] = gas_dp(meta, D['planes'])
            spec = spec.replace('Dp=gas', f"Dp={gas[meta['key']]['Dp']:.6f}")
            print(f"   gas Dp {meta['key']}: {gas[meta['key']]}", flush=True)
        with mp.Pool(a.jobs, initializer=init, initargs=(a.bundle, meta, K, spec, a.nsig)) as pool:
            for rows in pool.imap_unordered(one, sel, chunksize=20):
                out += rows
        T = pd.DataFrame([r for r in out if r['row'] == meta['key']])
        print(f"== {meta['key']}  {len(sel)} planes, K {K}, hyper {meta.get('hyper')}")
        for (mode, arm), g in T.groupby(['mode', 'arm'], sort=False):
            k = np.isfinite(g.u) & (np.abs(g.w_true) > 0)
            print(f'   {mode:5s} {arm:8s} ratio {np.median(g.u[k] / g.w_true[k]):.3f}  robust {np.nanmedian(g.ur[k] / g.w_true[k]):.3f}'
                  f'  n_hits {np.mean(g.n_hits[k]):.1f}  dropped by robust {np.mean(g.n_hits[k] - g.n_rob[k]):.2f}  cov {k.mean():.2f}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(out).to_parquet(a.out)
    print(f'[sps_resim] wrote {a.out}')
    if a.summary:
        summary(pd.DataFrame(out), gas, a.summary)


def summary(T, gas, path):
    """median ratio u / w per row, arm and mode, with 1.2533 x 1.4826 MAD / sqrt(n) as the error of the median."""
    import json
    rows = {}
    for r, g in T.groupby('row', sort=False):
        y = g[(g.arm == 'base') & (g['mode'] == 'res')]
        R = rows[r] = dict(v_stage_um_ns=float(np.median(np.abs(y.w_true)) * 1e3 / np.tan(np.radians(25.4))), arms={})
        if r in gas:
            R['gas_dp'] = gas[r]
        for (arm, mode), h in g.groupby(['arm', 'mode'], sort=False):
            k = np.isfinite(h.u) & (np.abs(h.w_true) > 0)
            x = (h.u[k] / h.w_true[k]).values
            R['arms'].setdefault(arm, {})[mode] = dict(
                ratio=float(np.median(x)), ratio_robust=float(np.nanmedian(h.ur[k] / h.w_true[k])),
                n_hits=float(np.mean(h.n_hits[k])), dropped=float(np.mean(h.n_hits[k] - h.n_rob[k])),
                coverage=float(k.mean()), ratio_err=float(1.2533 * 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(len(x))))
    doc = ('sps_resim.py on the five SPS CO2 rows at the 25.4 deg stage, inclined (Y) view: median hits-ladder slope / '
           'generating slope per arm and mode (res = the plane residual added back, base/res is the data; noise = pure model '
           '+ white noise; zs = noise with the data zero suppression). gas = the Magboltz Dp of the row (Ar/CO2/iso 93/5/2 at '
           'the gas-fit composition, gas_dp: 0.0049-0.0054; 0.006 from the wrong 95/3/2 grid before 2026-11-20), '
           'base = the pass Dp 0.020.')
    json.dump(dict(_doc=doc, rows=rows), open(path, 'w'), indent=1)
    print(f'[sps_resim] wrote {path}')


if __name__ == '__main__':
    main()
