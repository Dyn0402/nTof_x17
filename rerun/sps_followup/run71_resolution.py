#!/usr/bin/env python3
"""run71_resolution.py --tel DIR --pass-meta DIR [--out results/run71_resolution.json]

det4 position resolution of RAW run_71 (head-on, Ar/CF4/iso 88/10/2, the pass-1 run71 rows' reconstruction) against
the uRWELL telescope.  Never against ref_x/ref_y: those are det4's own charge centroid; here they are only a second det4
ESTIMATOR, residualled against the same telescope.

Inputs: tel_<group>.parquet from det4_reco_events.py (the events with a uRWELL track, reconstructed as the pass did) and
the pass rows' events.meta.json (their chi2/dof medians: quality_ok = chi2/dof <= 5 x the PASS ROW's median, so the
quality cut is the pass's, not this sample's).

Method = sps_beam_test_26/analysis/spatial_resolution/resolution.py (run 53):
  track at det4 = front + (back - front) z / 1370   (interpolation: a straight track is exact, divergence cancels)
  det4 -> track by an affine in the det4 view that follows the telescope coordinate (robust lstsq, 3-sigma MAD trim;
  run 53 used both views, but here det4's Y view is dead over the telescope), residual core by a fixed-window
  Gaussian fit, and
      sigma_det4^2 = sigma_res^2 - (z/1370)^2 sigma_back^2 - (1 - z/1370)^2 sigma_front^2
  with the reference terms MEASURED on run 53: back uRWELL = 0.301 x its local pitch (0.5 / 1.5 / 1.0 mm zones), front
  binary (1.0 mm / sqrt 12).  z of det4 from a z-scan of the residual width.
Selection per telescope coordinate: front and back lit with a single peak in that coordinate (urwell_scan.py), det4 x
and y ok + quality_ok.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'spatial_resolution'))
from resolution import gauss_sigma, robust_lstsq, ZONES, BINARY   # noqa: E402

Z_BACK = 1370.0
F_BACK = 0.301                     # back uRWELL sigma / pitch, measured on run 53 (spatial_resolution/results.json "fit")
LIT = 100.0
PLATEAUS = ('raw700', 'raw450', 'raw275')
ZGRID = (1000, 1050, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1250)


def pointing(z, pitch):
    return float(np.sqrt((z / Z_BACK) ** 2 * (F_BACK * pitch) ** 2 + (1 - z / Z_BACK) ** 2 * (BINARY * 1.0) ** 2))


def zone_of(coord, b):
    out = np.full(len(b), np.nan)
    for lo, hi, p in ZONES[coord]:
        out[(b >= lo) & (b <= hi)] = p
    return out


def resid(d, coord, e, z):
    """Track coordinate at z against ONE det4 view e (an affine a e + b): at H4 the telescope's x runs along det4's
    y and its y along det4's x, and over the telescope's footprint det4's Y view is dead (README §i), so a 2-D affine
    would only add the dead view's artefact positions as noise."""
    t = d[f'f{coord}_p'] + (d[f'b{coord}_p'] - d[f'f{coord}_p']) * z / Z_BACK
    M = np.column_stack([d[e], np.ones(len(d))])
    c, r = robust_lstsq(M, t.values)
    return r, c


def one_view(d, coord, e, z, win):
    r, c = resid(d, coord, e, z)
    zp = zone_of(coord, d[f'b{coord}_p'].values)
    s, e, n = gauss_sigma(r, win=win)
    zones = {}
    for p in (0.5, 1.0, 1.5):
        sz, ez, nz = gauss_sigma(r[zp == p], win=win)
        if np.isfinite(sz):
            pt = pointing(z, p)
            zones[str(p)] = dict(n=int(nz), sigma_res=float(sz), err=float(ez), pointing=pt,
                                 sigma_det4=float(np.sqrt(max(sz ** 2 - pt ** 2, 0.0))))
    # all zones together: pointing averaged in quadrature over the zone mix of the sample
    m = np.isfinite(zp)
    pt_mix = float(np.sqrt(np.mean([pointing(z, p) ** 2 for p in zp[m]]))) if m.any() else np.nan
    s68 = float(np.percentile(np.abs(r - np.median(r)), 68.27))
    return dict(n=int(len(d)), n_core=int(n), sigma_res=float(s), err=float(e), pointing_mix=pt_mix,
                sigma_det4=float(np.sqrt(max(s ** 2 - pt_mix ** 2, 0.0))) if np.isfinite(s) else np.nan,
                sigma_det4_err=float(s * e / np.sqrt(max(s ** 2 - pt_mix ** 2, 1e-12))) if np.isfinite(s) else np.nan,
                s68=s68, coef=[float(x) for x in c], zones=zones,
                zone_counts={str(p): int((zp == p).sum()) for p in (0.5, 1.0, 1.5)}), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tel', required=True); ap.add_argument('--pass-meta', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'run71_resolution.json')); ap.add_argument('--win', type=float, default=1.2)
    a = ap.parse_args()
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.tel, 'tel_*.parquet')))], ignore_index=True)
    out = dict(n_reconstructed=int(len(T)))
    resid_dump = {}
    for pl in PLATEAUS:
        d = T[T.plateau == pl].copy()
        if not len(d):
            continue
        med = json.load(open(os.path.join(a.pass_meta, f'run71_raw__{pl}', 'events.meta.json')))['quality_rule']['chi2dof_median']
        for v in 'xy':
            r = d[f'{v}_chi2'] / d[f'{v}_dof'].clip(lower=1)
            d[f'{v}_qok'] = d[f'{v}_ok'].astype(bool) & np.isfinite(r) & (r <= 5.0 * med[v])
        det = (d.x_qok & d.y_qok).values
        R = dict(n_events=int(len(d)), f_det4_ok=float(det.mean()), views={}, own_centroid={}, zscan={})
        for coord in 'xy':
            sel = det & (d[f'f{coord}_s3'] > LIT).values & (d[f'b{coord}_s3'] > LIT).values & \
                (d[f'f{coord}_npk'] == 1).values & (d[f'b{coord}_npk'] == 1).values
            s = d[sel].reset_index(drop=True)
            t = s[f'f{coord}_p'] + (s[f'b{coord}_p'] - s[f'f{coord}_p']) * 1140.0 / Z_BACK
            cor = {v: float(np.corrcoef(t, s[f'{v}_p0'])[0, 1]) if len(s) > 5 else np.nan for v in 'xy'}
            dv = max(cor, key=lambda v: abs(cor[v]) if np.isfinite(cor[v]) else -1)
            R.setdefault('corr', {})[coord] = dict(cor, det4_view=dv, n=int(len(s)))
            if not abs(cor[dv]) > 0.3:          # no det4 view follows this telescope coordinate
                print(f'{pl} uRW-{coord}: n {len(s)}, corr with det4 x {cor["x"]:+.2f}, y {cor["y"]:+.2f}: not measurable')
                continue
            zs = []
            for z in ZGRID:
                rz, _ = resid(s, coord, f'{dv}_p0', z)
                zs.append(dict(z=z, sigma=float(gauss_sigma(rz, win=a.win)[0])))
            R['zscan'][coord] = zs
            R['views'][coord], rr = one_view(s, coord, f'{dv}_p0', 1140.0, a.win)
            R['views'][coord]['det4_view'] = dv
            R['own_centroid'][coord], _ = one_view(s[np.isfinite(s[f'ref_{dv}'])].reset_index(drop=True), coord,
                                                   f'ref_{dv}', 1140.0, a.win)
            resid_dump[f'{pl}_{coord}'] = rr
            resid_dump[f'{pl}_{coord}_b'] = s[f'b{coord}_p'].values
            q = R['views'][coord]; oc = R['own_centroid'][coord]
            zb = min(zs, key=lambda x: x['sigma'] if np.isfinite(x['sigma']) else 1e9)['z']
            print(f"{pl} uRW-{coord} vs det4 {dv}: n {q['n']} (core {q['n_core']})  res {q['sigma_res']*1e3:4.0f}+-{q['err']*1e3:.0f} um, "
                  f"pointing {q['pointing_mix']*1e3:.0f} -> det4 wft {q['sigma_det4']*1e3:4.0f}+-{q['sigma_det4_err']*1e3:.0f} um | "
                  f"own centroid {oc['sigma_det4']*1e3:4.0f} um | s68 {q['s68']*1e3:.0f} | z-scan best {zb}", flush=True)
            for p, zq in q['zones'].items():
                print(f"     back pitch {p}: n {zq['n']}  res {zq['sigma_res']*1e3:4.0f}+-{zq['err']*1e3:.0f}  -> det4 {zq['sigma_det4']*1e3:4.0f}")
        out[pl] = R
    # the pass rows' own det4 positions: the Y view's pile-up at the dead-band edge (README §i)
    edges = np.arange(100.0, 305.0, 2.5)
    out['pass_p0_hist'] = dict(edges=edges.tolist())
    for pl in PLATEAUS:
        p = pd.read_parquet(os.path.join(a.pass_meta, f'run71_raw__{pl}', 'events.parquet'), columns=['x_p0', 'y_p0', 'x_ok', 'y_ok'])
        out['pass_p0_hist'][pl] = {v: np.histogram(p[f'{v}_p0'][p[f'{v}_ok'].astype(bool)], bins=edges)[0].tolist() for v in 'xy'}
        out['pass_p0_hist'][pl]['y_frac_155_160'] = float(((p.y_p0 >= 155) & (p.y_p0 < 160)).mean())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    np.savez(os.path.join(os.path.dirname(a.out), 'run71_residuals.npz'), **resid_dump)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
