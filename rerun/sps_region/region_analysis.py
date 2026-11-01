#!/usr/bin/env python3
"""region_analysis.py [--out results/region.json]  -- det4 at SPS H4 measured on ONE small, fully working region.

Why: the whole-beam-spot numbers (rerun/sps_audit) mix det4's dead amplification stripes, bands of different gain and
the telescope's three back-plane pitch zones.  On det4 X the zones fall on different live bands (zone 1.0 = the X band
148-162 mm, zone 1.5 = 200-214 mm, zone 0.5 = 232 mm), so the free 3-zone fit there mixes chamber and telescope and
fails (chi2/1 7-88).  Here everything is measured in one region R where det4 works in both views:

  R = det4 X 150-160 mm  x  Y 222-246 mm   (run 63 flat700, Ar/CF4/iso 88/10/2, resist 700 V, 243 V/cm, ZS 64)

chosen on the maps (``maps``): X inside the widest live band of the highest gain (148-162 mm, >= 2 mm from both
edges); Y in the telescope's 1.0 mm back-plane zone in both coordinates (Y >= 219 mm), clear of the oscillating
Y channels (ch 372-376 at 154.6-158.7 mm and ch 510 at 250.4 mm), in the densest part of the spot.

Telescope pointing in R: one zone (1.0 mm in both coordinates), so f cannot be fitted in R.  f is measured on run 63's
Y view in the SAME X band (all three Y zones are populated there, and det4's gain is flat along Y in the band); the
whole-spot fits (0.25-0.34) give the systematic.

Sections: maps, resolution (run 63; run 66 and run 53 cross-checks), footprint (run 63 ZS in R; run 71 RAW in R),
timing (run 71 RAW and run 63 ZS strip sums in R; paper1_timing estimators), angle and charge in R.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (DATA, load_tel, align, telescope_at, seeded_fit, gauss_sigma, zone_of, pointing, Z_DET4,  # noqa: E402
                    Z_BACK, FID_MM, HOT_Y_MM, WIN, REPO, BINARY)
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_timing'))

R = dict(x=(150.0, 160.0), y=(222.0, 246.0))
BAND_X = (150.0, 160.0)               # the X band used for the f calibration (Y view, all Y)
BAND_X2 = (201.0, 214.0)              # second live band, cross-check of f
MAP_BIN = 1.0
F_SYS = (0.25, 0.34)                  # whole-spot f range (sps_audit section 7)


def in_r(x, y, r=R):
    return (x >= r['x'][0]) & (x < r['x'][1]) & (y >= r['y'][0]) & (y < r['y'][1])


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan
    q = np.percentile(a, [16, 84]); return float((q[1] - q[0]) / 2)


def boot(fn, a, n=200, seed=1):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed)
    v = [fn(a[rng.integers(0, len(a), len(a))]) for _ in range(n)]
    return float(np.nanstd(v))


def pointing_f(p, f, z=Z_DET4):
    return pointing(z, p, f)


# ------------------------------------------------------------------ maps
def maps(d, a, eff=None):
    xe = np.arange(140.0, 245.0 + 1e-9, MAP_BIN); ye = np.arange(150.0, 262.0 + 1e-9, MAP_BIN)
    px, py = a['px'], a['py']
    agX, agY = np.abs(a['ry']) < FID_MM, np.abs(a['rx']) < FID_MM
    H = np.histogram2d(px, py, [xe, ye])[0]
    HX = np.histogram2d(px[agX], py[agX], [xe, ye])[0]; HY = np.histogram2d(px[agY], py[agY], [xe, ye])[0]
    out = dict(xe=xe.tolist(), ye=ye.tolist(), n=H.tolist(), agree_x=HX.tolist(), agree_y=HY.tolist())
    # gain: median lead-strip charge of agreeing tracks, 2 mm cells
    xe2, ye2 = xe[::2], ye[::2]
    for v, ag in (('x', agX), ('y', agY)):
        q = d[f'd4_{v}_c_q_lead'].values
        G = np.full((len(xe2) - 1, len(ye2) - 1), np.nan)
        ix = np.digitize(px, xe2) - 1; iy = np.digitize(py, ye2) - 1
        ok = ag & (ix >= 0) & (ix < len(xe2) - 1) & (iy >= 0) & (iy < len(ye2) - 1) & np.isfinite(q)
        df = pd.DataFrame(dict(i=ix[ok], j=iy[ok], q=q[ok])).groupby(['i', 'j']).q.agg(['median', 'size'])
        for (i, j), r in df.iterrows():
            if r['size'] >= 20:
                G[i, j] = r['median']
        out[f'q_lead_{v}'] = np.where(np.isfinite(G), G, -1).tolist()
    out['xe2'] = xe2.tolist(); out['ye2'] = ye2.tolist()
    # 1-D profiles across X (the stripes) and along Y in the band
    prof = []
    for x in xe[:-1]:
        m = (px >= x) & (px < x + MAP_BIN)
        if m.sum() >= 30:
            prof.append(dict(x=float(x + MAP_BIN / 2), n=int(m.sum()), agree_x=float(agX[m].mean()), agree_y=float(agY[m].mean()),
                             q_x=float(np.nanmedian(d.d4_x_c_q_lead.values[m & agX])) if (m & agX).sum() > 10 else None,
                             zone_by=float(np.nanmedian(zone_of('y', d.by_p.values[m])))))
    out['profile_x'] = prof
    prof = []
    for y in ye[:-1:2]:
        m = (py >= y) & (py < y + 2) & (px >= BAND_X[0]) & (px < BAND_X[1])
        if m.sum() >= 30:
            prof.append(dict(y=float(y + 1), n=int(m.sum()), agree_x=float(agX[m].mean()), agree_y=float(agY[m].mean()),
                             q_y=float(np.nanmedian(d.d4_y_c_q_lead.values[m & agY])) if (m & agY).sum() > 10 else None,
                             zone_bx=float(np.nanmedian(zone_of('x', d.bx_p.values[m])))))
    out['profile_y_band'] = prof
    # telescope zone boundaries in det4 coordinates: linear map back-plane coordinate -> predicted det4 coordinate
    zb = {}
    for coord, det4, b in (('y', px, d.by_p.values), ('x', py, d.bx_p.values)):
        sl, ic = np.polyfit(b, det4, 1)
        from common import ZONES
        zb[coord] = [dict(pitch=p, det4_lo=float(min(sl * lo + ic, sl * hi + ic)), det4_hi=float(max(sl * lo + ic, sl * hi + ic)))
                     for lo, hi, p in ZONES[coord]]
    out['zones_det4'] = dict(X=zb['y'], Y=zb['x'])
    if eff is not None:
        out.update(eff)
    return out


# ------------------------------------------------------------------ resolution
def zone_fit(r, b, coord, nmin=500):
    """zone fit sigma_res^2 = s_det4^2 + W_back (f p)^2 + front term, on the zones with >= nmin core tracks (the sps_audit
    decompose, weighted, with errors on f).  Inside one det4 X band only the 1.0 and 1.5 mm Y zones are populated
    (the 0.5 mm zone, Y 164-173, is the beam's and the band's edge: 50-270 tracks), so the fit is two zones, two unknowns."""
    zp = zone_of(coord, b); rows = []
    for p in (0.5, 1.0, 1.5):
        s, e, n = gauss_sigma(r[zp == p])
        if np.isfinite(s):
            rows.append((p, float(s), float(e), int(n)))
    use = [q for q in rows if q[3] >= nmin]
    out = dict(zones=[dict(pitch=p, sigma_res=s, err=e, n=n, used=n >= nmin) for p, s, e, n in rows], fit=None)
    if len(use) < 2:
        return out
    wb = (Z_DET4 / Z_BACK) ** 2; wf = (1 - Z_DET4 / Z_BACK) ** 2 * BINARY ** 2
    P = np.array([q[0] for q in use]); y = np.array([q[1] ** 2 - wf for q in use]); ye = np.array([2 * q[1] * q[2] for q in use])
    X = np.column_stack([np.ones(len(use)), wb * P ** 2]); Ci = np.diag(1 / ye ** 2)
    cov = np.linalg.inv(X.T @ Ci @ X); bb = cov @ (X.T @ Ci @ y)
    f = float(np.sqrt(max(bb[1], 0))); ef = float(0.5 * np.sqrt(cov[1, 1]) / max(f, 1e-6))
    sd = float(np.sqrt(max(bb[0], 0))); esd = float(0.5 * np.sqrt(cov[0, 0]) / max(sd, 1e-6))
    out['fit'] = dict(f=f, f_err=ef, sigma_det4=sd, sigma_det4_err=esd, n_zones=len(use),
                      chi2=float((y - X @ bb) @ Ci @ (y - X @ bb)))
    return out


def f_calibration(d, a, est):
    """f from det4 Y's residual in all three Y zones, inside one det4 X band (gain flat along Y there)."""
    out = {}
    for name, band in (('band 150-160', BAND_X), ('band 201-214', BAND_X2)):
        m = (a['px'] >= band[0]) & (a['px'] < band[1])
        D = d[m].reset_index(drop=True)
        ex, ey = D[f'd4_x_{est}'].values, D[f'd4_y_{est}'].values
        ok = np.isfinite(ex) & np.isfinite(ey); D = D[ok].reset_index(drop=True); ex, ey = ex[ok], ey[ok]
        tx, ty = telescope_at(D)
        M = np.column_stack([ex, ey, np.ones(len(ex))])
        _, rx = seeded_fit(M, tx, ey); _, ry = seeded_fit(M, ty, ex)
        fY = np.abs(ry) < FID_MM                     # the other view (X) agrees
        out[name] = zone_fit(rx[fY], D.bx_p.values[fY], 'x')
    return out


def resolution(d, a, est, f, f_rng, r=R):
    """residual core in region r, local affine (seeded_fit) refitted on r's tracks, cross-view fiducial.
    Three telescope samples: all single-cluster tracks; 'b2' = the back-plane cluster of the coordinate that measures
    this view has 2 strips (interpolated, much better than the binary 1-strip); 'b2f2' = front too.  Their residual is
    det4 (+) telescope, so it is a model-free UPPER bound on det4.  det4 itself: sqrt(res_all^2 - P(f)^2), P the zone's
    pointing with the common f; f_rng gives the systematic."""
    m = in_r(a['px'], a['py'], r)
    D = d[m].reset_index(drop=True)
    ex, ey = D[f'd4_x_{est}'].values.astype(float), D[f'd4_y_{est}'].values.astype(float)
    ok = np.isfinite(ex) & np.isfinite(ey); D = D[ok].reset_index(drop=True); ex, ey = ex[ok], ey[ok]
    if len(D) < 200:
        return dict(n=int(len(D)))
    tx, ty = telescope_at(D)
    M = np.column_stack([ex, ey, np.ones(len(ex))])
    _, rx = seeded_fit(M, tx, ey); _, ry = seeded_fit(M, ty, ex)
    fX, fY = np.abs(rx) < FID_MM, np.abs(ry) < FID_MM
    zX = zone_of('y', D.by_p.values); zY = zone_of('x', D.bx_p.values)
    out = dict(n_region=int(m.sum()), n=int(len(D)), fid_frac=float(np.mean(fX & fY)))
    for v, res, fid, zz, tb, tf in (('X', ry, fX, zX, 'by', 'fy'), ('Y', rx, fY, zY, 'bx', 'fx')):
        s, e, n = gauss_sigma(res[fid])
        p_main = float(np.nanmedian(zz[fid]))
        det = lambda pp: float(np.sqrt(max(s ** 2 - pp ** 2, 0.0)))
        P = pointing_f(p_main, f); sd = det(P)
        o = dict(n_core=int(n), sigma_res=float(s), err=float(e), zone=p_main, zone_frac=float(np.mean(zz[fid] == p_main)),
                 pointing=P, sigma_det4=sd, stat=float(s * e / max(sd, 1e-9)),
                 sys_range=sorted([det(pointing_f(p_main, f_rng[0])), det(pointing_f(p_main, f_rng[1]))]),
                 hist=np.histogram(res[fid], bins=np.arange(-1.5, 1.5001, 0.05))[0].tolist())
        for name, sel in (('b2', D[f'{tb}_size'].values == 2), ('b2f2', (D[f'{tb}_size'].values == 2) & (D[f'{tf}_size'].values == 2)),
                          ('b1', D[f'{tb}_size'].values == 1)):
            s2, e2, n2 = gauss_sigma(res[fid & sel])
            o[f'res_{name}'] = [float(s2), float(e2), int(n2)]
            if name == 'b2':
                o['hist_b2'] = np.histogram(res[fid & sel], bins=np.arange(-1.5, 1.5001, 0.05))[0].tolist()
        # the b2 pointing the f model implies (res_all^2 - res_b2^2 = P_all^2 - P_b2^2): a consistency check of the model
        o['P_b2_implied'] = float(np.sqrt(max(P ** 2 - (s ** 2 - o['res_b2'][0] ** 2), 0.0))) if np.isfinite(o['res_b2'][0]) else None
        out[v] = o
    # uniformity: the X and Y residual cores in four Y quarters
    qs = []
    pyr = a['py'][m][ok]
    for k in range(4):
        y0 = r['y'][0] + k * (r['y'][1] - r['y'][0]) / 4; y1 = y0 + (r['y'][1] - r['y'][0]) / 4
        mm = (pyr >= y0) & (pyr < y1)
        qs.append(dict(y=[y0, y1], n=int(mm.sum()), X=[float(x) for x in gauss_sigma(ry[fX & mm])[:2]],
                       Y=[float(x) for x in gauss_sigma(rx[fY & mm])[:2]]))
    out['uniformity'] = qs
    # angle and charge on the same fiducial events (head-on; mount 0 deg)
    for v in 'xy':
        t = D[f'd4_{v}_tan_theta'].values[fX & fY]
        st = s68(t)
        out[f'tan_{v}'] = dict(median=float(np.nanmedian(t)), s68=st, n=int(np.isfinite(t).sum()),
                                theta_s68_deg=float(np.degrees(np.arctan(st))) if np.isfinite(st) else None)
        q = D[f'd4_{v}_c_q_sum'].values[fX & fY]
        out[f'q_{v}'] = dict(q_sum_median=float(np.nanmedian(q)), q_lead_median=float(np.nanmedian(D[f'd4_{v}_c_q_lead'].values[fX & fY])),
                             hist=np.histogram(q, bins=np.arange(0, 3001, 100))[0].tolist())
    return out


# ------------------------------------------------------------------ footprint
def footprint_tel(d, a):
    """ZS cluster shape (sps_cluster quantities stored per event) in R and over the whole spot, both views agreeing."""
    ag = (np.abs(a['rx']) < FID_MM) & (np.abs(a['ry']) < FID_MM)
    out = {}
    for name, m in (('region', ag & in_r(a['px'], a['py'])), ('whole spot', ag)):
        row = dict(n=int(m.sum()))
        for v in 'xy':
            row[v] = dict(strips_mean=float(np.nanmean(d[f'd4_{v}_c_n_strips'].values[m])),
                          strips_median=float(np.nanmedian(d[f'd4_{v}_c_n_strips'].values[m])),
                          lead_frac=float(np.nanmedian(d[f'd4_{v}_c_lead_frac'].values[m])),
                          nb=float(np.nanmedian(np.r_[d[f'd4_{v}_c_nb_lo'].values[m], d[f'd4_{v}_c_nb_hi'].values[m]])),
                          q_sum=float(np.nanmedian(d[f'd4_{v}_c_q_sum'].values[m])),
                          q_lead=float(np.nanmedian(d[f'd4_{v}_c_q_lead'].values[m])),
                          strips_hist=np.bincount(np.clip(np.nan_to_num(d[f'd4_{v}_c_n_strips'].values[m]).astype(int), 0, 8), minlength=9).tolist())
        out[name] = row
    return out


def footprint_raw(key):
    """threshold-free footprint (RAW run71): det4 positions select R directly (no telescope in these tables)."""
    t = pd.read_parquet(os.path.join(DATA, 'footprint', f'{key}.parquet'))
    out = {}
    for name, m in (('region', in_r(t.x_c_pos.values, t.y_c_pos.values)), ('whole spot', np.isfinite(t.x_c_pos.values) & np.isfinite(t.y_c_pos.values))):
        row = dict(n=int(m.sum()))
        for v in 'xy':
            row[v] = dict(strips_mean=float(np.nanmean(t[f'{v}_c_n_strips'].values[m])),
                          lead_frac=float(np.nanmedian(t[f'{v}_c_lead_frac'].values[m])),
                          nb=float(np.nanmedian(np.r_[t[f'{v}_c_nb_lo'].values[m], t[f'{v}_c_nb_hi'].values[m]])),
                          q_sum=float(np.nanmedian(t[f'{v}_c_q_sum'].values[m])),
                          strips_hist=np.bincount(np.clip(np.nan_to_num(t[f'{v}_c_n_strips'].values[m]).astype(int), 0, 8), minlength=9).tolist())
        out[name] = row
    return out


# ------------------------------------------------------------------ timing
def timing(keys):
    import estimators as E
    b = np.load(os.path.join(DATA, 'bundle', 'arrays.npz'))
    kern = E.kernel(b['grid'], b['tmpl_x'])
    out = {}
    for key in keys:
        z = np.load(os.path.join(DATA, 'timing', f'{key}.npz'))
        base = z['x_ok'] & z['y_ok'] & (np.nan_to_num(z['x_q']) >= 150) & (np.nan_to_num(z['y_q']) >= 150)
        ns = z['x_S'].shape[1]
        E.EDGE_FRAC = 0.25; ed = E.Edge(kern, ns)
        T = {}
        for p in 'xy':
            S = z[f'{p}_S'].astype(float)
            T[('edge25', p)] = np.array([ed.fit(s) if base[i] else np.nan for i, s in enumerate(S)])
            T[('cf20', p)] = np.array([E.cf(s, 0.2) if base[i] else np.nan for i, s in enumerate(S)])
        reg = in_r(z['x_pos'], z['y_pos'])
        ftst = z['ftst'].astype(float)
        qmin = np.fmin(z['x_q'], z['y_q'])
        res = dict(n_all=int(base.sum()), n_region=int((base & reg).sum()))
        for est in ('edge25', 'cf20'):
            D = T[(est, 'x')] - T[(est, 'y')]; Mn = (T[(est, 'x')] + T[(est, 'y')]) / 2
            r = {}
            for name, m in (('region', base & reg), ('whole spot', base)):
                r[name] = dict(sigma_xy=s68(D[m]) / np.sqrt(2), sigma_xy_err=boot(s68, D[m]) / np.sqrt(2),
                               abs_raw=s68(Mn[m]), abs_ftst=s68(Mn[m] + 10 * ftst[m]), abs_ftst_minus=s68(Mn[m] - 10 * ftst[m]),
                               q_median=float(np.nanmedian(qmin[m])))
            # amplitude dependence in R and in the spot: X-Y/sqrt2 in bins of the smaller view charge
            edges = np.array([150, 300, 500, 800, 1300, 2500, 1e9])
            curve = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                for name, m0 in (('region', base & reg), ('whole spot', base)):
                    m = m0 & (qmin >= lo) & (qmin < hi)
                    if m.sum() >= 40:
                        curve.append(dict(sample=name, q_lo=float(lo), q_hi=float(hi), n=int(m.sum()),
                                          q_med=float(np.median(qmin[m])), sigma_xy=s68(D[m]) / np.sqrt(2)))
            r['vs_charge'] = curve
            r['hist_region'] = np.histogram(D[base & reg], bins=np.arange(-600, 601, 30))[0].tolist()
            res[est] = r
        out[key] = res
        e = res['edge25']
        print(f'  timing {key:24s} region n {res["n_region"]:4d}: edge25 X-Y/sqrt2 {e["region"]["sigma_xy"]:.0f} +- {e["region"]["sigma_xy_err"]:.0f} ns '
              f'(spot {e["whole spot"]["sigma_xy"]:.0f}), vs trigger {e["region"]["abs_raw"]:.0f}; cf20 {res["cf20"]["region"]["sigma_xy"]:.0f}', flush=True)
    return out


# ------------------------------------------------------------------ efficiency (eff_job.py tables, all telescope tracks)
def efficiency(d63, a63):
    fs = sorted(glob.glob(os.path.join(DATA, 'eff', 'E_*.parquet')))
    if not fs:
        return None
    e = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    # telescope -> predicted det4 with the run-63 alignment (same events' telescope coordinates)
    tx, ty = telescope_at(e)
    T = np.column_stack([tx, ty, np.ones(len(e))])
    tx3, ty3 = telescope_at(d63); T3 = np.column_stack([tx3, ty3, np.ones(len(d63))])
    good = (np.abs(a63['rx']) < FID_MM) & (np.abs(a63['ry']) < FID_MM)
    bX, *_ = np.linalg.lstsq(T3[good], d63.d4_x_c_pos.values[good], rcond=None)
    bY, *_ = np.linalg.lstsq(T3[good], d63.d4_y_c_pos.values[good], rcond=None)
    px, py = T @ bX, T @ bY
    hit = {}
    for v, p in (('x', px), ('y', py)):
        c = np.column_stack([e[f'{v}_c{i}'].values for i in range(4)]) if f'{v}_c0' in e else np.full((len(e), 4), np.nan)
        hit[v] = np.any(np.abs(c - p[:, None]) < 2.0, axis=1)
    both = hit['x'] & hit['y']
    xe = np.arange(140.0, 245.0 + 1e-9, MAP_BIN); ye = np.arange(150.0, 262.0 + 1e-9, MAP_BIN)
    H = np.histogram2d(px, py, [xe, ye])[0]; HB = np.histogram2d(px[both], py[both], [xe, ye])[0]
    m = in_r(px, py)
    out = dict(eff_n=H.tolist(), eff_both=HB.tolist(), eff_tracks=int(len(e)))
    k = int(m.sum())
    for name, h in (('both', both), ('x', hit['x']), ('y', hit['y'])):
        p = float(h[m].mean()); out[f'eff_region_{name}'] = [p, float(np.sqrt(p * (1 - p) / max(k, 1))), k]
    out['eff_region_any_signal'] = float(np.mean(np.isfinite(e['x_a0'].values[m]) & np.isfinite(e['y_a0'].values[m])))
    # sensitivity: the window +-2 mm -> +-1, +-3 mm
    for w in (1.0, 3.0):
        hh = [np.any(np.abs(np.column_stack([e[f'{v}_c{i}'].values for i in range(4)]) - p[:, None]) < w, axis=1)
              for v, p in (('x', px), ('y', py))]
        out[f'eff_region_both_win{w:.0f}'] = float((hh[0] & hh[1])[m].mean())
    # efficiency profile across X (the stripes), Y in R's range
    prof = []
    for x in xe[:-1]:
        mm = (px >= x) & (px < x + MAP_BIN) & (py >= R['y'][0]) & (py < R['y'][1])
        if mm.sum() >= 30:
            prof.append(dict(x=float(x + 0.5), n=int(mm.sum()), eff=float(both[mm].mean())))
    out['eff_profile_x'] = prof
    # uniformity inside R: 2 x 4 cells
    cells = []
    for i in range(2):
        for j in range(4):
            x0 = R['x'][0] + i * 5; y0 = R['y'][0] + j * 6
            mm = (px >= x0) & (px < x0 + 5) & (py >= y0) & (py < y0 + 6)
            cells.append(dict(x=[x0, x0 + 5], y=[y0, y0 + 6], n=int(mm.sum()), eff=float(both[mm].mean()) if mm.sum() else None))
    out['eff_cells'] = cells
    print(f'  efficiency in R: both views {out["eff_region_both"][0]:.3f} +- {out["eff_region_both"][1]:.3f} (n {k}); '
          f'X {out["eff_region_x"][0]:.3f}, Y {out["eff_region_y"][0]:.3f}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'region.json'))
    a_ = ap.parse_args()
    res = dict(region=R, band_x=BAND_X, data=DATA)
    d63 = load_tel('run63'); a63 = align(d63)
    eff = efficiency(d63, a63)
    res['maps'] = maps(d63, a63, eff)
    if eff:
        res['efficiency'] = {k: v for k, v in eff.items() if not k.startswith('eff_n') and k != 'eff_both'}
    # f from det4 Y in the band (all three Y zones), per estimator
    res['f_cal'] = {est: f_calibration(d63, a63, est) for est in ('p0', 'p_mid', 'c_pos')}
    for est in ('p0', 'p_mid', 'c_pos'):
        fc = res['f_cal'][est]['band 150-160']['fit']
        print(f'  f from Y zones in X band 150-160 ({est}): f {fc["f"]:.3f} +- {fc["f_err"]:.3f}, det4 Y {fc["sigma_det4"]*1e3:.0f} '
              f'+- {fc["sigma_det4_err"]*1e3:.0f} um, chi2 {fc["chi2"]:.2f}; band 201-214: f {res["f_cal"][est]["band 201-214"]["fit"]["f"]:.3f}', flush=True)
    fits = [res['f_cal'][e][b]['fit'] for e in ('p0', 'p_mid', 'c_pos') for b in ('band 150-160', 'band 201-214')]
    fs_ = np.array([q['f'] for q in fits])
    f_c, f_rng = float(fs_.mean()), (float(fs_.min()), float(fs_.max()))
    res['f_common'] = dict(f=f_c, range=f_rng, n_fits=len(fs_),
                           note='mean of the 2-zone (1.0, 1.5 mm) fits of det4 Y inside the X bands 150-160 and 201-214, three estimators')
    print(f'  common f {f_c:.3f} (range {f_rng[0]:.3f}-{f_rng[1]:.3f})', flush=True)
    res['resolution'] = {}
    REGIONS = {'R': R, 'R2 (cross-check: X band 201-214)': dict(x=(202.0, 213.0), y=R['y'])}
    res['regions'] = REGIONS
    for tag in ('run63', 'run66', 'run53'):
        d = d63 if tag == 'run63' else load_tel(tag)
        if tag == 'run66':          # the two highest-gain resist plateaus only (sps_audit's 'run66 r755+r780')
            d = d[d.key.isin(['run66_flat_resist__r755', 'run66_flat_resist__r780'])].reset_index(drop=True)
        a = a63 if tag == 'run63' else align(d)
        res['resolution'][tag] = {}
        for rname, rr in REGIONS.items():
            if tag != 'run63' and rname != 'R':
                continue
            for est in ('p0', 'p_mid', 'c_pos'):
                r = resolution(d, a, est, f_c, f_rng, rr)
                res['resolution'][tag][f'{rname}|{est}'] = r
                if 'X' in r:
                    X, Y = r['X'], r['Y']
                    print(f'  {tag} {rname[:2]} {est:6s} n {r["n"]:5d}: X res {X["sigma_res"]*1e3:.0f} b2 {X["res_b2"][0]*1e3:.0f} b2f2 {X["res_b2f2"][0]*1e3:.0f}'
                          f' -> det4 {X["sigma_det4"]*1e3:.0f} ({X["sys_range"][0]*1e3:.0f}-{X["sys_range"][1]*1e3:.0f}) | Y res {Y["sigma_res"]*1e3:.0f}'
                          f' b2 {Y["res_b2"][0]*1e3:.0f} -> det4 {Y["sigma_det4"]*1e3:.0f} ({Y["sys_range"][0]*1e3:.0f}-{Y["sys_range"][1]*1e3:.0f});'
                          f' zones {X["zone"]}/{Y["zone"]}; P_b2 implied {X["P_b2_implied"]*1e3 if X["P_b2_implied"] is not None else -1:.0f}/'
                          f'{Y["P_b2_implied"]*1e3 if Y["P_b2_implied"] is not None else -1:.0f}', flush=True)
    res['footprint'] = dict(run63_zs=footprint_tel(d63, a63), run71_raw700=footprint_raw('run71_raw__raw700'),
                            run71_raw450=footprint_raw('run71_raw__raw450'), run71_raw275=footprint_raw('run71_raw__raw275'))
    res['timing'] = timing(['run71_raw__raw700', 'run71_raw__raw450', 'run63_flat__flat700', 'run66_flat_resist__r780'])
    os.makedirs(os.path.dirname(a_.out), exist_ok=True)
    json.dump(res, open(a_.out, 'w'), indent=1, default=float)
    print('wrote', a_.out)


if __name__ == '__main__':
    main()
