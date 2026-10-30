#!/usr/bin/env python3
"""analyze.py --tel DIR [--out results/audit.json]

det4 position resolution against the uRWELL telescope, per dataset and estimator, by the run-53 method
(sps_beam_test_26/analysis/spatial_resolution/resolution.py, imported, not re-implemented):
  track at det4 = front + (back - front) z / 1370; det4 (X, Y) -> track (x, y) by a robust 2-D affine (3-sigma MAD
  trim); residual core by the fixed-window (+-1.2 mm) Gaussian; per back-plane pitch zone
      sigma_det4^2 = sigma_res^2 - (z/1370)^2 (f pitch)^2 - (1 - z/1370)^2 (1.0/sqrt12)^2,   f = 1/sqrt12 (binary)
  and, where a coordinate has all three zones, the free-slope fit sigma_res^2 = sigma_det4^2 + W_back (f pitch)^2 that
  MEASURES f (run 53: f = 0.301, det4 = 176 um).
At H4 the telescope x runs along det4 Y and its y along det4 X (roll ~90 deg): 'uRW-x' measures det4 Y, 'uRW-y' det4 X.

Estimators, on one common event set per dataset (every estimator valid on the event):
  pass      the pass-1 SPS chain WITH the seed fix (tel_job.py 'new'): wft p0, ok and chi2/dof <= 5 x the sample median
  pass_old  the pass-1 chain as run (pre-fix seed; differs only where the seed moved)
  centroid  det4's own charge centroid over the pass window (ref_*)
  hits      the run-53 estimator: combined_hits feu 3, amp > 60, leading cluster in position, one cluster per view (ZS)
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'spatial_resolution'),
                os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
from resolution import gauss_sigma, robust_lstsq, ZONES, BINARY, det4_clusters   # noqa: E402
from det4_sps_map import POSITION_MM, VIEW                                          # noqa: E402

Z_BACK = 1370.0
Z_DET4 = 1120.0
ZGRID = np.arange(1000.0, 1261.0, 20.0)
#: datasets: tag -> (file prefixes, registry keys or None, det4 hit-time gate for the hits estimator [ns])
DATASETS = {
    'run53 (ZS32, Ar/CO2/iso)': (['r53_'], None, (600, 1850)),
    'run53 qmin100': (['r53q100_'], None, (600, 1850)),
    'run55 d700 (ZS64, Ar/CO2/iso)': (['r55_'], ['run55_flatdrift__d700'], None),
    'run55 d600 (ZS64, Ar/CO2/iso)': (['r55_'], ['run55_flatdrift__d600'], None),
    'run63 flat700 (ZS64, CF4)': (['r63op02_', 'r63op03_'], ['run63_flat__flat700'], None),
    'run66 r755+r780 (ZS32, CF4)': (['r66_'], ['run66_flat_resist__r755', 'run66_flat_resist__r780'], None),
    'run66 r580-r730 (ZS32, CF4)': (['r66_'], [f'run66_flat_resist__r{r}' for r in (580, 605, 630, 655, 680, 705, 730)], None),
    'run71 raw700, followup telescope': (['r71_'], ['run71_raw__raw700'], None),
    'run71 raw700 (RAW64, CF4)': (['r71v2_'], ['run71_raw__raw700'], None),
    'run71 raw450 (RAW64, CF4)': (['r71v2_'], ['run71_raw__raw450'], None),
    'run71 raw275 (RAW64, CF4)': (['r71v2_'], ['run71_raw__raw275'], None),
}


def pointing(z, pitch, f=BINARY):
    return float(np.sqrt((z / Z_BACK) ** 2 * (f * pitch) ** 2 + (1 - z / Z_BACK) ** 2 * (BINARY * 1.0) ** 2))


def zone_of(coord, b):
    out = np.full(len(b), np.nan)
    for lo, hi, p in ZONES[coord]:
        out[(b >= lo) & (b <= hi)] = p
    return out


def load(tel, prefixes, keys):
    fs = sorted(f for p in prefixes for f in glob.glob(os.path.join(tel, p + '[0-9][0-9][0-9].parquet')))
    if not fs:
        return None, None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if keys:
        d = d[d.key.isin(keys)]
    hits = {}
    for f in fs:
        h = f.replace('.parquet', '_hits.npz')
        if os.path.exists(h):
            hits[os.path.basename(f)[:-8].split('_')[-1]] = dict(np.load(h))
    return d.reset_index(drop=True), hits


def hits_estimator(d, hits, gate):
    """det4 lead cluster per view from the combined_hits feu-3 hits (resolution.py det4_clusters), one cluster per view."""
    out = {v: np.full(len(d), np.nan) for v in 'xy'}
    ncl = {v: np.zeros(len(d), int) for v in 'xy'}
    for g, h in hits.items():
        rows = np.flatnonzero((d.group == g).values)
        if not len(rows):
            continue
        idx = {int(e): i for i, e in enumerate(d.eid.values[rows])}
        m = np.isin(h['ev'], list(idx))
        if gate:
            m &= (h['time'] > gate[0]) & (h['time'] < gate[1])
        ev = np.array([idx[int(e)] for e in h['ev'][m]], int)
        ch = h['ch'][m].astype(int); amp = h['amp'][m].astype(float)
        isx = VIEW[ch] == 'x'
        for v, k in (('x', isx), ('y', ~isx)):
            if not k.any():
                continue
            lead, nc, _ = det4_clusters(ev[k], POSITION_MM[ch[k]], amp[k], len(rows))
            out[v][rows] = lead; ncl[v][rows] = nc
    for v in 'xy':
        out[v][ncl[v] != 1] = np.nan
    return out


def telescope_ok(d):
    """Single-cluster track in all four plane-views (ZS: n == 1; RAW: one in-time peak above 100 ADC)."""
    if 'fx_size' in d:                      # ZS (combined_hits clusters)
        return np.all([(d[f'{p}_n'] == 1).values for p in ('fx', 'fy', 'bx', 'by')], axis=0)
    return np.all([((d[f'{p}_n'] == 1) & (d[f'{p}_q'] > 100.0)).values for p in ('fx', 'fy', 'bx', 'by')], axis=0)


def quality(d, tag):
    ok = {}
    for v in 'xy':
        r = d[f'{tag}_{v}_chi2'] / d[f'{tag}_{v}_dof'].clip(lower=1)
        good = d[f'{tag}_{v}_ok'].astype(float).fillna(0).astype(bool) & np.isfinite(r)
        med = np.nanmedian(r[good]) if good.any() else np.nan
        ok[v] = (good & (r <= 5.0 * med)).values
    return ok


def resid2d(D, ex, ey, z):
    tx = D['fx_p'] + (D['bx_p'] - D['fx_p']) * z / Z_BACK
    ty = D['fy_p'] + (D['by_p'] - D['fy_p']) * z / Z_BACK
    M = np.column_stack([ex, ey, np.ones(len(ex))])
    cx, rx = robust_lstsq(M, tx.values)
    cy, ry = robust_lstsq(M, ty.values)
    return rx, ry, np.array([[cx[0], cx[1]], [cy[0], cy[1]]])


def seeded_fit(M, t, d_main):
    """Affine t ~ M (columns det4 X, det4 Y, 1), robust to a 40-60 % outlier fraction: det4 self-seeds, so a track
    through a dead amplification stripe leaves det4 reading a pile-up particle elsewhere.  Seed: slope +-1 on the det4
    view that follows t (at H4 roll ~90 deg: telescope x <-> det4 Y, telescope y <-> det4 X) and the mode of t - s d;
    then lstsq inside 3 mm, 2 mm, and the 3-sigma MAD trim of resolution.robust_lstsq."""
    best = None
    for s in (1.0, -1.0):
        off = t - s * d_main
        lo, hi = np.nanpercentile(off, [1, 99])
        h, e = np.histogram(off, bins=np.arange(lo - 1, hi + 1, 0.5))
        if best is None or h.max() > best[0]:
            best = (h.max(), s, e[np.argmax(h)] + 0.25)
    keep = np.abs(t - best[1] * d_main - best[2]) < 3.0
    for win in (3.0, 2.0):
        coef, *_ = np.linalg.lstsq(M[keep], t[keep], rcond=None)
        keep = np.abs(t - M @ coef) < win
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(M[keep], t[keep], rcond=None)
        r = t - M @ coef
        sg = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < 3 * sg
    return coef, t - M @ coef


def resid2d_seeded(D, ex, ey, z):
    tx = (D['fx_p'] + (D['bx_p'] - D['fx_p']) * z / Z_BACK).values
    ty = (D['fy_p'] + (D['by_p'] - D['fy_p']) * z / Z_BACK).values
    M = np.column_stack([ex, ey, np.ones(len(ex))])
    cx, rx = seeded_fit(M, tx, ey)
    cy, ry = seeded_fit(M, ty, ex)
    return rx, ry, np.array([[cx[0], cx[1]], [cy[0], cy[1]]])


#: cross-view fiducial [mm]: a view's resolution is measured on events whose OTHER view agrees with the telescope, so
#: det4 saw this track (not a pile-up particle while the track crossed a dead stripe).  Independent of the view measured.
FID_MM = 1.5


def decompose(r, b, coord, z):
    zp = zone_of(coord, b)
    rows = []
    for p in (0.5, 1.0, 1.5):
        s, e, n = gauss_sigma(r[zp == p])
        if np.isfinite(s) and n >= 150:
            pt = pointing(z, p)
            rows.append(dict(pitch=p, n=int(n), sigma_res=float(s), err=float(e), pointing_binary=pt,
                             sigma_det4=float(np.sqrt(max(s ** 2 - pt ** 2, 0.0)))))
    fit = None
    if len(rows) == 3:
        # as resolution.py: the intercept carries the (3 %) front term too, so this is directly its 176 um
        wb = (z / Z_BACK) ** 2
        P = np.array([q['pitch'] for q in rows]); y = np.array([q['sigma_res'] ** 2 for q in rows])
        ye = np.array([2 * q['sigma_res'] * q['err'] for q in rows])
        X = np.column_stack([np.ones(3), P ** 2]); Ci = np.diag(1 / ye ** 2)
        cov = np.linalg.inv(X.T @ Ci @ X); bb = cov @ (X.T @ Ci @ y)
        fit = dict(f_back=float(np.sqrt(max(bb[1], 0) / wb)), sigma_det4=float(np.sqrt(max(bb[0], 0))),
                   sigma_det4_err=float(0.5 * np.sqrt(cov[0, 0]) / np.sqrt(max(bb[0], 1e-9))),
                   chi2=float((y - X @ bb) @ Ci @ (y - X @ bb)))
    s, e, n = gauss_sigma(r)
    return dict(n_core=int(n), sigma_res=float(s), err=float(e), zones=rows, fit=fit,
                core_frac=float(np.mean(np.abs(r - np.median(r)) < 1.2)),
                s68=float(np.percentile(np.abs(r - np.median(r)), 68.27)))


def one_dataset(d, hits, gate):
    tel = telescope_ok(d)
    est = {}
    qn = quality(d, 'new')
    est['pass'] = (d.new_x_p0.values, d.new_y_p0.values, qn['x'] & qn['y'])
    # pre-fix: the old fit where the seed moved, the (identical) new fit elsewhere
    oldx = np.where(d.old_seed_x.values >= 0, d.old_x_p0.values, d.new_x_p0.values)
    oldy = np.where(d.old_seed_y.values >= 0, d.old_y_p0.values, d.new_y_p0.values)
    qo = quality(d.assign(**{f'old_{v}_{c}': np.where(d[f'old_seed_{v}'].values >= 0, d[f'old_{v}_{c}'].values, d[f'new_{v}_{c}'].values)
                             for v in 'xy' for c in ('ok', 'chi2', 'dof')}), 'old')
    est['pass_old'] = (oldx, oldy, qo['x'] & qo['y'])
    est['centroid'] = (d.new_ref_x.values, d.new_ref_y.values, np.isfinite(d.new_ref_x.values) & np.isfinite(d.new_ref_y.values))
    if hits:
        h = hits_estimator(d, hits, gate)
        if np.mean(np.isfinite(h['x']) & np.isfinite(h['y'])) > 0.2:      # combined_hits of runs 63/66 carry no feu 3
            est['hits'] = (h['x'], h['y'], np.isfinite(h['x']) & np.isfinite(h['y']))
    common = tel & np.all([e[2] & np.isfinite(e[0]) & np.isfinite(e[1]) for e in est.values()], axis=0)
    out = dict(n_rows=int(len(d)), n_tel=int(tel.sum()), n_common=int(common.sum()),
               n_seed_moved_y=int((d.old_seed_y.values >= 0).sum()), estimators={})
    D = d[common].reset_index(drop=True)
    resid = {}
    for name, (ex, ey, _) in est.items():
        ex, ey = ex[common], ey[common]
        if len(D) < 50:
            continue
        zs = []
        for z in ZGRID:
            rx, ry, _ = resid2d_seeded(D, ex, ey, z)
            fx_, fy_ = np.abs(ry) < FID_MM, np.abs(rx) < FID_MM
            zs.append((float(z), float(gauss_sigma(rx[fx_])[0]), float(gauss_sigma(ry[fy_])[0])))
        rx, ry, A = resid2d_seeded(D, ex, ey, Z_DET4)
        fid_Y, fid_X = np.abs(ry) < FID_MM, np.abs(rx) < FID_MM
        R = dict(det_A=float(np.linalg.det(A)), A=A.tolist(),
                 roll_deg=float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))),
                 zscan=zs, both_core=float(np.mean((np.abs(rx) < 1.2) & (np.abs(ry) < 1.2))),
                 # telescope x measures det4 Y; telescope y measures det4 X
                 det4_Y=decompose(rx[fid_Y], D['bx_p'].values[fid_Y], 'x', Z_DET4),
                 det4_X=decompose(ry[fid_X], D['by_p'].values[fid_X], 'y', Z_DET4))
        R['det4_Y']['n_fid'] = int(fid_Y.sum()); R['det4_X']['n_fid'] = int(fid_X.sum())
        out['estimators'][name] = R
        resid[name] = (np.where(fid_Y, rx, np.nan), np.where(fid_X, ry, np.nan))
    return out, resid, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tel', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'audit.json'))
    a = ap.parse_args()
    res = {}; dump = {}
    for name, (prefixes, keys, gate) in DATASETS.items():
        d, hits = load(a.tel, prefixes, keys)
        if d is None or not len(d):
            print(f'{name}: no data'); continue
        R, resid, D = one_dataset(d, hits, gate)
        res[name] = R
        print(f'\n== {name}: rows {R["n_rows"]}, telescope {R["n_tel"]}, common {R["n_common"]}, Y seed moved {R["n_seed_moved_y"]}')
        for est, E in R['estimators'].items():
            line = f'  {est:9s} det(A) {E["det_A"]:+.3f} roll {E["roll_deg"]:+6.1f} both-core {E["both_core"]:.2f}'
            for view in ('det4_X', 'det4_Y'):
                V = E[view]
                zs = ' '.join(f'{q["pitch"]}:{q["sigma_res"] * 1e3:.0f}->{q["sigma_det4"] * 1e3:.0f}' for q in V['zones'])
                fit = f' FIT f={V["fit"]["f_back"]:.3f} det4={V["fit"]["sigma_det4"] * 1e3:.0f}+-{V["fit"]["sigma_det4_err"] * 1e3:.0f}' if V['fit'] else ''
                line += f' | {view[-1]}: n {V["n_fid"]} res {V["sigma_res"] * 1e3:.0f} [{zs}]{fit}'
            print(line)
            zmin_x = min(E['zscan'], key=lambda t: t[2] if np.isfinite(t[2]) else 9)[0]
            zmin_y = min(E['zscan'], key=lambda t: t[1] if np.isfinite(t[1]) else 9)[0]
            print(f'            z-scan minimum: det4 X {zmin_x:.0f} mm, det4 Y {zmin_y:.0f} mm')
            dump[f'{name}|{est}|x'] = resid[est][1]; dump[f'{name}|{est}|y'] = resid[est][0]
        dump[f'{name}|bx'] = D['bx_p'].values; dump[f'{name}|by'] = D['by_p'].values
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    np.savez_compressed(a.out.replace('.json', '_resid.npz'), **dump)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
