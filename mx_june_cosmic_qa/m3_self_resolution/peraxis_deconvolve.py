#!/usr/bin/env python3
"""Per-axis det3 residual widths (wft and hits) through the 02_efficiency accounting,
plus the M3 pointing deconvolution done PER AXIS at the run's own z_mean.

Why this exists (2026-08-10). `analyze.py` / `M3_SELF_RESOLUTION.md` deconvolve a
*radial* residual width (0.47 mm = rstd of |r| = hypot(dx,dy), the
`m3_cut_tradeoff.py` / `02_efficiency.py` "core sigma") against a *per-axis*
pointing P (0.22 mm). Those are different quantities, so that sigma_DUT ~ 0.40 mm
is not a per-axis detector resolution and cannot be compared with the per-axis
0.61/0.73 mm quoted from `residuals.png`. This script does the same subtraction
consistently: per-axis residual widths (three estimators) minus per-axis P(z),
with P recomputed from `results.json`'s measured per-plane sigma_k at the run's
own fitted z_mean (714 mm for det3, not the nominal 702).

    ../../.venv/bin/python peraxis_deconvolve.py          # sat_det3, wft + hits
Cross-check built in: the radial rstd it reprints must equal the published
`state/det3/efficiency__efficiency_breakdown*.json` core_sigma_mm (0.4597 wft /
0.4477 hits) -- that is what validates the accounting.
"""
import os, sys, json, pickle
import numpy as np

REPO = '/home/dylan/PycharmProjects/nTof_x17'
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis'),
                os.path.join(REPO, 'mx_june_wft')]
from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
setup_paths()
import matplotlib; matplotlib.use('Agg')
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_positions, get_xy_angles
from wft import compat
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS

KEY = sys.argv[1] if len(sys.argv) > 1 else 'sat_det3'
SOURCES = ('wft', 'hits') if KEY == 'sat_det3' else ('hits',)
R = 5.0


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v))


def s68(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    q = np.percentile(np.abs(v - np.median(v)), 68.27)
    return float(q)


def gauss_core(v, n_iter=6, nsig=2.5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    mu, sig = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
    for _ in range(n_iter):
        m = np.abs(v - mu) < nsig * sig
        if m.sum() < 20:
            break
        mu, sig = v[m].mean(), v[m].std()
    return float(mu), float(sig), int(m.sum())


def hat_coeffs(z_used, z_eval):
    A = np.vstack([np.ones_like(z_used), z_used]).T
    M = np.linalg.inv(A.T @ A)
    return np.array([1.0, z_eval]) @ M @ A.T


cfg = get_config(KEY)
rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
print(f'recipe: chi2<{M3_CHI2_CUT} NClus>={M3_MIN_NCLUS}')

# detection bookkeeping (spark veto) -- same as 02_efficiency
fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
            if f.endswith('.root') and '_datrun_' in f)
raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                         expressions=['eventId', 'feu', 'channel', 'significance'],
                         library='pd')
det_raw = raw[raw['feu'].isin(cfg.MX17_FEUS)]
det_lo, det_hi = int(det_raw['eventId'].min()), int(det_raw['eventId'].max())
mult_raw = det_raw.groupby('eventId').size()
mult = (cm.apply_significance_floor(det_raw, rel=SIG_REL_FLOOR)
        .groupby('eventId').size().reindex(mult_raw.index).fillna(0).astype(int))
mult_by_ev = mult.to_dict()

out = {}
for source in SOURCES:
    if source == 'wft':
        align = os.path.join(cfg.OUT_BASE, 'wft', 'alignment', 'alignment.json')
        params = cm.load_alignment(align)
        df = compat.load_table(os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet'),
                               max_dropped=None)
        results = compat.as_event_results(df)
    else:
        align = os.path.join(cfg.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json')
        params = cm.load_alignment(align)
        cache = os.path.join(cfg.OUT_BASE, 'cache', 'event_results.pkl')
        results = pickle.load(open(cache, 'rb'))
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in results
            if r.has_both and np.isfinite(r.det_x_aligned_mm)
            and np.isfinite(r.det_y_aligned_mm)}
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr)
    evn = [int(v) for v in evn]
    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    dx, dy = [], []
    for e, x, y in zip(evn, px, py):
        if e < det_lo or e > det_hi:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        if mult_by_ev.get(e, 0) > SPARK_VETO_HITS:
            continue
        if e in reco:
            dx.append(x - reco[e][0]); dy.append(y - reco[e][1])
    dx = np.array(dx); dy = np.array(dy)
    r = np.hypot(dx, dy)
    keep = r < 15
    res = dict(source=source, z_mean=float(params.z_mean), n=int(len(dx)),
               n_keep=int(keep.sum()),
               radial_rstd=rstd(r[keep]), median_r=float(np.median(r)),
               within5=float(100.0 * (r <= R).mean()))
    for nm, v in (('x', dx), ('y', dy)):
        vk = v[keep]
        mu, sg, nc = gauss_core(vk)
        res[nm] = dict(median=float(np.median(vk)), rstd=rstd(vk), s68=s68(vk),
                       core_mu=mu, core_sigma=sg, n_core=nc)
    out[source] = res
    print(json.dumps(res, indent=1))

# ---- M3 pointing per axis at this run's z_mean ----
rj = json.load(open(os.path.join(REPO, 'mx_june_cosmic_qa', 'm3_self_resolution',
                                 'results.json')))
Z = np.array(rj['z_layers'])
P = {}
for c in ('X', 'Y'):
    sk = np.array([L['sigma_geomean_um'] / 1000.0 for L in rj['coords'][c]['layers']])
    for z in (232.0, 702.0, 714.0):
        g = hat_coeffs(Z, z)
        P.setdefault(c, {})[z] = float(np.sqrt(np.sum(g ** 2 * sk ** 2)))
print('\nM3 pointing P(z) per axis [mm]:', json.dumps(P, indent=1))

print('\n=== per-axis deconvolution ===')
for source in SOURCES:
    z = out[source]['z_mean']
    zk = 714.0 if abs(z - 714) < 3 else 702.0
    for ax, C in (('x', 'X'), ('y', 'Y')):
        for est in ('core_sigma', 's68', 'rstd'):
            sm = out[source][ax][est]
            p = P[C][zk]
            d2 = sm ** 2 - p ** 2
            print(f'{source:5s} {ax} {est:10s} sigma_meas={sm:.3f}  P={p:.3f}  '
                  f'refvar={100*(p/sm)**2:4.1f}%  sigma_DUT={np.sqrt(max(d2,0)):.3f}')

# radial-vs-per-axis consistency: what radial rstd do the per-axis sigmas predict?
rng = np.random.default_rng(1)
for source in SOURCES:
    sx, sy = out[source]['x']['core_sigma'], out[source]['y']['core_sigma']
    sim = np.hypot(rng.normal(0, sx, 400000), rng.normal(0, sy, 400000))
    print(f'{source}: per-axis core ({sx:.3f},{sy:.3f}) -> predicted radial rstd '
          f'{rstd(sim[sim<15]):.3f}  (measured {out[source]["radial_rstd"]:.3f})')

json.dump({'residuals': out, 'pointing': {c: {str(k): v for k, v in d.items()}
                                          for c, d in P.items()}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'peraxis_{KEY}.json'), 'w'), indent=1)
print(f'\nwrote peraxis_{KEY}.json')
