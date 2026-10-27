#!/usr/bin/env python3
"""vgas.py -- FA item 1: v(t) and v(x, y) as a standard calibration, separated from the bow, and gated on held-out tracks.

Inputs per golden row: colsum.npz (every trigger: model-free column duration T from the strip sum, charge centroid,
DREAM timestamp; colsum.py) and the golden per-event table (ws_v_w0: fitted w, M3 tan_ref, reference positions).
det3 drift-scan colsums give v(E) with no M3 and no angle.

1. Clock.  DREAM timestamps, ordered by acquisition then timestamp; TICK_NS calibrated on the 15-min scan rows.
2. v(t).  Clean triggers (both views: >= 5 pulse strips, one cluster, no saturated strip), T_evt = mean of the views.
   Block medians of T for block sizes N = 25 ... 16000 events; the Allan deviation of consecutive block medians falls
   as 1/sqrt(N) while statistics dominate and rises once the gas drifts: its minimum is the block size, chosen from the
   data per row.  v(t) = v_run * Tbar / T_block(t) (linear between block centres).
   Cross-check: the ladder (robust line w = v tan + a, M3) in 8 equal-count time blocks against 1/T in the same windows.
3. v(x, y), separated from the bow.  The bow changes the local gap g(x, y) and the local tilt; the gas sets v.  Two
   measurements per cell, both time-normalised with v(t):
     ladder   w = v_c tan + a_c + g_c (P - P_c)   per view (the slope is v; the intercept and its gradient are the local
              tilt, i.e. the bow, fitted together so the acceptance's position-angle correlation cannot leak into v);
     column   T_c = g_c / v_c  (no angle, no M3).
   The bow also moves v through the field: E = V / g, v = v_gas(x, y) (E/E0)^beta with beta = dln v / dln E from the
   drift scan.  Solving the two cell measurements for the two unknowns:
        dln g = dln T + dln v ;   dln v_gas = dln v + beta dln g
   gives the gap map (FB's bow, from timing) and the residual gas map.  If dln v_gas ~ 0, v(x, y) is the bow's field
   effect and can be PREDICTED from the column-duration map alone (no M3: transferable to n_TOF):
        dln v_pred = -beta / (1 + beta) dln T.
4. Gate (held-out by event-id parity: maps from even events, metrics on odd events, paired bootstrap):
   conversions  prod (bundle w0, kw, v) | +v(t) | +v(t)+v(x,y) ladder | +v(t)+v(x,y) predicted from T |
                +v(t)+w0(x,y) (bow intercept map only) | +v(t)+v(x,y)+w0(x,y) | ws_widthtilt angle_map (reference)
   metrics      s68 of the angle residual (all, |tan_ref| > 0.1); the spread of the angle scale over 8 time blocks and
                over 3 x 3 cells (Huber slope tan_fit vs tan_ref per block): what a constant v gets wrong locally.
Outputs results/vgas_<det>.json, results/vgas_summary.json; figures by make_figures.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_common import ROOT, FA, RES, DETS, GOLD_ROW, CENTRE, huber, golden_table, load_colsum, tmean  # noqa: E402

TICK_NS = float(os.environ.get('FA_TICK_NS', '10.0'))
GAP_MM = 30.0
RNG = np.random.default_rng(17)
CELL = 100.0          # ladder cells [mm]
TCELL = 50.0          # column-duration cells [mm]
SCAN = {500: 'drift_scan_resist_490V_drift_500V', 700: 'drift_scan_resist_490V_drift_700V',
        900: 'drift_scan_resist_490V_drift_900V', 1100: 'drift_scan_resist_490V_drift_1100V'}


def clean(df, sat_ok=False):
    m = (df.x_has & df.y_has & (df.x_nsig >= 5) & (df.y_nsig >= 5) & (df.x_ncl == 1) & (df.y_ncl == 1)
         & np.isfinite(df.x_T) & np.isfinite(df.y_T)).to_numpy()
    if not sat_ok:
        m &= ~df.x_sat.to_numpy() & ~df.y_sat.to_numpy()
    return m


def clock(df, tags):
    """hours since the first trigger: acquisitions in stem order, DREAM timestamp within (a file index continues it)."""
    stem = np.array([t.rsplit('_', 1)[0] for t in tags])
    order = {s: i for i, s in enumerate(sorted(set(stem)))}
    a = np.array([order[stem[k]] for k in df.acq.to_numpy()])
    ts = np.where(df.x_ts.to_numpy() >= 0, df.x_ts.to_numpy(), df.y_ts.to_numpy()).astype(float)
    h = ts * TICK_NS * 1e-9 / 3600.0
    # acquisitions follow each other: offset each by the previous one's end (the gap between them is not recorded)
    off = 0.0; out = np.zeros(len(h))
    for k in range(a.max() + 1):
        m = a == k
        out[m] = h[m] + off
        off = out[m].max() if m.any() else off
    return out


def allan(T, sizes):
    res = []
    med = np.median(T)
    for N in sizes:
        nb = len(T) // N
        if nb < 4:
            break
        m = np.array([tmean(T[k * N:(k + 1) * N]) for k in range(nb)])
        res.append((int(N), float(100 * np.sqrt(0.5 * np.mean(np.diff(m) ** 2)) / med), int(nb)))
    return res


def blocks(h, T, N):
    nb = max(1, len(T) // N)
    idx = np.array_split(np.arange(len(T)), nb)
    tc = np.array([np.median(h[i]) for i in idx]); Tb = np.array([tmean(T[i]) for i in idx])
    eb = np.array([1.2533 * 1.4826 * np.median(np.abs(T[i] - np.median(T[i]))) / np.sqrt(len(i)) for i in idx])
    return tc, Tb, eb


def ladder(w, tr, P=None):
    m = np.isfinite(w) & np.isfinite(tr) & (np.abs(tr) < 0.45)
    cols = [tr[m], np.ones(m.sum())] + ([P[m] - np.median(P[m])] if P is not None else [])
    b, cov, n = huber(np.column_stack(cols), w[m], it=25)
    return b, np.sqrt(np.diag(cov)), n


def cellmap(x, y, val, size, lo=0.0, hi=400.0, nmin=150, stat=tmean):
    e = np.arange(lo, hi + 1e-9, size); nc = len(e) - 1
    M = np.full((nc, nc), np.nan); E = np.full((nc, nc), np.nan); Nn = np.zeros((nc, nc), int)
    ix = np.digitize(x, e) - 1; iy = np.digitize(y, e) - 1
    for i in range(nc):
        for j in range(nc):
            m = (ix == i) & (iy == j) & np.isfinite(val)
            Nn[i, j] = m.sum()
            if m.sum() >= nmin:
                v = val[m]; M[i, j] = stat(v)
                E[i, j] = 1.2533 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(m.sum())
    return e, M, E, Nn


def interp2(e, M, x, y):
    """bilinear interpolation of a cell map at (x, y); cells without a value fall back to the map's median."""
    c = 0.5 * (e[1:] + e[:-1]); Mf = np.where(np.isfinite(M), M, np.nanmedian(M))
    xi = np.clip(np.interp(x, c, np.arange(len(c))), 0, len(c) - 1); yi = np.clip(np.interp(y, c, np.arange(len(c))), 0, len(c) - 1)
    i0 = np.floor(xi).astype(int); j0 = np.floor(yi).astype(int); i1 = np.minimum(i0 + 1, len(c) - 1); j1 = np.minimum(j0 + 1, len(c) - 1)
    fx, fy = xi - i0, yi - j0
    return ((1 - fx) * (1 - fy) * Mf[i0, j0] + fx * (1 - fy) * Mf[i1, j0] + (1 - fx) * fy * Mf[i0, j1] + fx * fy * Mf[i1, j1])


def ladder_cells(t, wcorr, size=CELL, nmin=120):
    """per view, per cell: w = v_c tan + a_c + g_c (P - P_c) -> v_c, a_c (+ errors)."""
    e = np.arange(0.0, 400.0 + 1e-9, size); nc = len(e) - 1
    out = {}
    X, Y = t.ref_mesh_x.to_numpy(), t.ref_mesh_y.to_numpy()
    ix = np.digitize(X, e) - 1; iy = np.digitize(Y, e) - 1
    for p in 'xy':
        V = np.full((nc, nc), np.nan); VE = V.copy(); A = V.copy(); AE = V.copy()
        ok = t[f'{p}_ok'].astype(bool).to_numpy()
        P = t[f'ref_mesh_{p}'].to_numpy()
        for i in range(nc):
            for j in range(nc):
                m = ok & (ix == i) & (iy == j)
                if m.sum() < nmin:
                    continue
                b, eb, n = ladder(wcorr[p][m], t[f'tan_ref_{p}'].to_numpy()[m], P[m])
                V[i, j], VE[i, j], A[i, j], AE[i, j] = b[0], eb[0], b[1], eb[1]
        out[p] = dict(e=e, v=V, v_err=VE, a=A, a_err=AE)
    return out


def beta_from_scan(det3_T1000):
    """dln v / dln E from the det3 drift-scan column durations (v ~ 1/T at fixed gap)."""
    pts = []
    for V, sub in SCAN.items():
        f = os.path.join(FA, 'mx17_det3_saturday_scan_6-27-26', sub, 'mx17_3', 'fa', 'colsum.npz')
        if not os.path.exists(f):
            continue
        df, _, _ = load_colsum(f)
        m = clean(df)
        T = 0.5 * (df.x_T.to_numpy() + df.y_T.to_numpy())[m]
        pts.append((V, tmean(T), float(1.2533 * 1.4826 * np.median(np.abs(T - np.median(T))) / np.sqrt(len(T))), int(m.sum())))
    if det3_T1000:
        pts.append((1000, det3_T1000[0], det3_T1000[1], det3_T1000[2]))
    pts.sort()
    use = [p for p in pts if p[0] >= 700]
    if len(use) < 2:
        return None, pts
    lv = np.log([GAP_MM / (p[1] * 1e-3) for p in use]); lE = np.log([p[0] for p in use])
    b = np.polyfit(lE, lv, 1)[0]
    return float(b), pts


def scale_spread(tf, tr, labels, nmin=150):
    vals = []
    for k in np.unique(labels):
        m = (labels == k) & np.isfinite(tf) & np.isfinite(tr) & (np.abs(tr) < 0.45)
        if m.sum() < nmin:
            continue
        b, _, _ = huber(np.column_stack([tr[m], np.ones(m.sum())]), tf[m], it=15)
        vals.append(b[0])
    vals = np.array(vals)
    return float(100 * np.std(vals) / np.mean(vals)) if len(vals) > 1 else np.nan


def metrics(tf, tr, tl, cl):
    d = np.degrees(np.arctan(tf) - np.arctan(tr))
    k = np.isfinite(d) & (np.abs(d) < 5)
    s68 = lambda m: float(np.percentile(np.abs(d[m & k] - np.median(d[m & k])), 68))
    return np.array([s68(np.ones_like(k)), s68(np.abs(tr) > 0.1), scale_spread(tf, tr, tl), scale_spread(tf, tr, cl)])


def gate(det, t, rt, h_evt, cal, Tmap, beta, amap):
    """held-out gate: maps from even event ids, metrics on odd ones."""
    ev = (t.event_id.to_numpy() % 2 == 0)
    od = ~ev
    X, Y = t.ref_mesh_x.to_numpy(), t.ref_mesh_y.to_numpy()
    tl = np.minimum((pd.Series(h_evt).rank(method='first').to_numpy() / len(t) * 8).astype(int), 7)
    cl = np.minimum((X // 133.4).astype(int), 2) * 3 + np.minimum((Y // 133.4).astype(int), 2)
    w = {p: t[f'{p}_w'].to_numpy() * 1e3 for p in 'xy'}
    wt = {p: w[p] / rt for p in 'xy'}                        # time-normalised lateral speed
    # maps from the even half
    lc = ladder_cells(t[ev].reset_index(drop=True), {p: wt[p][ev] for p in 'xy'})
    out = {}
    for p in 'xy':
        tr = t[f'tan_ref_{p}'].to_numpy(); ok = t[f'{p}_ok'].astype(bool).to_numpy() & np.isfinite(w[p]) & np.isfinite(tr)
        v, kw, w0 = cal['v_drift'], cal['kw'].get(p, 1.0), cal['w0'].get(p, 0.0)
        L = lc[p]; vglob = np.nanmedian(L['v'])
        fxy = interp2(L['e'], L['v'] / vglob, X, Y)
        a_xy = interp2(L['e'], L['a'], X, Y)
        lnT = np.log(interp2(Tmap[0], Tmap[1], X, Y))
        fT = np.exp(-beta / (1 + beta) * lnT) if beta is not None else np.ones(len(X))
        # k from the even half: the ladder v map against the column map in the same cells (k = -beta/(1+beta): all bow;
        # k = -1: all gas); then v(x, y) = T(x, y)^k needs M3 only once per chamber, the T map comes from any run
        cc = 0.5 * (L['e'][1:] + L['e'][:-1]); CX, CY = np.meshgrid(cc, cc, indexing='ij')
        lt = np.log(interp2(Tmap[0], Tmap[1], CX.ravel(), CY.ravel())); lv = np.log((L['v'] / vglob).ravel())
        le = (L['v_err'] / L['v']).ravel(); mk = np.isfinite(lv) & np.isfinite(lt) & np.isfinite(le) & (le > 0)
        kfit = float(np.sum(lv[mk] * lt[mk] / le[mk] ** 2) / np.sum(lt[mk] ** 2 / le[mk] ** 2)) if mk.sum() > 3 else -1.0
        fTk = np.exp(kfit * lnT)
        conv = {
            'prod': (w[p] - w0) / (kw * v),
            '+vt': (wt[p] - w0) / (kw * v),
            '+vt+vxy': (wt[p] - w0) / (kw * v * fxy),
            '+vt+vxyT': (wt[p] - w0) / (kw * v * fT),
            '+vt+vxyTk': (wt[p] - w0) / (kw * v * fTk),
            '+vt+vxyTk+w0xy': (wt[p] - a_xy) / (kw * v * fTk),
            '+vt+w0xy': (wt[p] - a_xy) / (kw * v),
            '+vt+vxy+w0xy': (wt[p] - a_xy) / (kw * v * fxy),
        }
        m = amap.get(p)
        if m:
            pm = t[f'{p}_p0'].to_numpy() + 15.0 * conv['prod']
            conv['angle_map'] = (w[p] - np.interp(pm, m['p'], m['w0'])) / (np.interp(pm, m['p'], m['s']) * kw * v)
        res, boots, x0s = {}, {}, {}
        # one common held-out sample and one set of bootstrap resamples for every conversion (paired deltas)
        sel = od & ok & np.all([np.isfinite(tf) for tf in conv.values()], axis=0)
        idx = np.flatnonzero(sel)
        resamples = [RNG.choice(idx, len(idx)) for _ in range(40)]
        for name, tf in conv.items():
            # each conversion gets its own global scale and offset from the even half (kw, w0 re-measured)
            mm = ev & ok & np.isfinite(tf) & (np.abs(tr) < 0.45)
            b, _, _ = huber(np.column_stack([tr[mm], np.ones(mm.sum())]), tf[mm], it=20)
            tfc = (tf - b[1]) / b[0]
            x0s[name] = metrics(tfc[idx], tr[idx], tl[idx], cl[idx])
            boots[name] = np.array([metrics(tfc[i], tr[i], tl[i], cl[i]) for i in resamples])
        for name in conv:
            x0 = x0s[name]
            res[name] = dict(n=int(len(idx)), s68=x0[0], s68_incl=x0[1], scale_spread_time_pct=x0[2], scale_spread_cells_pct=x0[3],
                             err=np.nanstd(boots[name], 0).tolist(), delta_vs_prod=(x0 - x0s['prod']).tolist(),
                             delta_err=np.nanstd(boots[name] - boots['prod'], 0).tolist())
        res['k_even'] = kfit
        out[p] = res
    return out, lc


def analyse(det, beta):
    cs = os.path.join(FA, GOLD_ROW[det], 'fa', 'colsum.npz')
    df, S, raw = load_colsum(cs)
    tags = [str(s) for s in raw['acq_tags']]
    df['h'] = clock(df, tags)
    # every one-cluster trigger, saturated or not: a saturated strip sum reads a different absolute T (det2 +1 %,
    # det7 +7 %, det4 +16 %) but follows the same trend in time, so each class is normalised to its own median and
    # put on the unsaturated class's scale (det6/det7 saturate on 80-90 % of tracks: without them v(t) starves)
    m = clean(df, sat_ok=True)
    c = df[m].sort_values('h').reset_index(drop=True)
    traw = 0.5 * (c.x_T + c.y_T).to_numpy()
    sat = (c.x_sat | c.y_sat).to_numpy()
    ref = tmean(traw[~sat]) if (~sat).sum() > 200 else tmean(traw)
    c['T'] = np.where(sat, traw / tmean(traw[sat]) * ref, traw / tmean(traw[~sat]) * ref) if (~sat).sum() > 200 \
        else traw
    c['sat'] = sat
    T = c['T'].to_numpy(); h = c['h'].to_numpy()
    rate = len(c) / max(h.max() - h.min(), 1e-6)          # clean triggers per hour
    sizes = np.unique(np.geomspace(25, 16000, 26).astype(int))
    al = allan(T, sizes)
    best = min(al, key=lambda r: r[1]) if al else (500, np.nan, 0)
    N = best[0]
    tc, Tb, eb = blocks(h, T, N)
    Tbar = tmean(T)
    rt_of = lambda hh: Tbar / np.interp(hh, tc, Tb)          # v(t) / v_run
    # golden table, event times through the trigger id
    t = golden_table(det)
    dfu = df.drop_duplicates('eid', keep=False).set_index('eid')
    t = t[t.event_id.isin(dfu.index)].reset_index(drop=True)
    h_evt = dfu.loc[t.event_id.to_numpy(), 'h'].to_numpy()
    rt = rt_of(h_evt)
    # ladder per time block vs 1/T in the same windows
    lb = np.minimum((pd.Series(h_evt).rank(method='first').to_numpy() / len(t) * 8).astype(int), 7)
    lad = []
    for k in range(8):
        mk = lb == k
        lo, hi = h_evt[mk].min(), h_evt[mk].max()
        Tm = T[(h >= lo) & (h <= hi)]
        row = dict(h_lo=float(lo), h_hi=float(hi), T=tmean(Tm),
                   T_err=float(1.2533 * 1.4826 * np.median(np.abs(Tm - np.median(Tm))) / np.sqrt(len(Tm))), n_T=int(len(Tm)))
        for p in 'xy':
            ok = mk & t[f'{p}_ok'].astype(bool).to_numpy()
            b, eb_, n = ladder(t[f'{p}_w'].to_numpy()[ok] * 1e3, t[f'tan_ref_{p}'].to_numpy()[ok])
            row[f'v_{p}'] = float(b[0]); row[f'v_{p}_err'] = float(eb_[0])
        lad.append(row)
    vl = np.array([0.5 * (r['v_x'] + r['v_y']) for r in lad]); invT = np.array([1 / r['T'] for r in lad])
    lad_slope = float(np.polyfit(np.log(invT / invT.mean()), np.log(vl / vl.mean()), 1)[0])
    # T(x, y) map, time-normalised, from every clean trigger (centroids from the waveforms)
    Tn = T * Tbar / np.interp(h, tc, Tb) / Tbar
    e, TM, TE, TN = cellmap(c.x_cen.to_numpy(), c.y_cen.to_numpy(), Tn, TCELL)
    TM = TM / np.nanmedian(TM); TE = TE / np.nanmedian(TM)
    # ladder v(x, y) on all matched events (time-normalised), and the two-map solution
    wt = {p: t[f'{p}_w'].to_numpy() * 1e3 / rt for p in 'xy'}
    lc = ladder_cells(t, wt)
    vX = lc['x']['v'] / np.nanmedian(lc['x']['v']); vY = lc['y']['v'] / np.nanmedian(lc['y']['v'])
    eX = lc['x']['v_err'] / np.nanmedian(lc['x']['v']); eY = lc['y']['v_err'] / np.nanmedian(lc['y']['v'])
    wX, wY = 1 / eX ** 2, 1 / eY ** 2
    vXY = (vX * wX + vY * wY) / (wX + wY); eXY = 1 / np.sqrt(wX + wY)
    eL = lc['x']['e']
    _, TMc, TEc, _ = cellmap(c.x_cen.to_numpy(), c.y_cen.to_numpy(), Tn, CELL)
    TMc = TMc / np.nanmedian(TMc)
    two = {}
    if beta is not None:
        dlnv = np.log(vXY); dlnT = np.log(TMc)
        dlng = dlnT + dlnv
        dlnvgas = dlnv + beta * dlng
        pred = -beta / (1 + beta) * dlnT
        ok = np.isfinite(dlnv) & np.isfinite(dlnT)
        r = float(np.corrcoef(dlnv[ok], pred[ok])[0, 1]) if ok.sum() > 3 else np.nan
        # k = dln v / dln T across the cells: -beta/(1+beta) if the bow (gap -> field) is all of it, -1 if gas is
        e2 = (eXY ** 2)[ok]
        k = float(np.sum(dlnv[ok] * dlnT[ok] / e2) / np.sum(dlnT[ok] ** 2 / e2)); ke = float(1 / np.sqrt(np.sum(dlnT[ok] ** 2 / e2)))
        chik = float(np.sum((dlnv[ok] - k * dlnT[ok]) ** 2 / e2))
        # share of the cell-to-cell T variance carried by the gap: dln T = (1+beta) dln g - dln v_gas
        two = dict(k=k, k_err=ke, k_bow=float(-beta / (1 + beta)), chi2_k=chik,
                   gap_share_of_T=float(np.nanvar((1 + beta) * dlng[ok]) / max(np.nanvar(dlnT[ok]), 1e-12)),dlng_pct=(100 * dlng).tolist(), dlnvgas_pct=(100 * dlnvgas).tolist(), pred_dlnv_pct=(100 * pred).tolist(),
                   meas_dlnv_pct=(100 * dlnv).tolist(), meas_err_pct=(100 * eXY).tolist(), corr_meas_pred=r,
                   rms_meas_pct=float(100 * np.nanstd(dlnv[ok])), rms_pred_pct=float(100 * np.nanstd(pred[ok])),
                   rms_gas_pct=float(100 * np.nanstd(dlnvgas[ok])), rms_gap_pct=float(100 * np.nanstd(dlng[ok])),
                   chi2_meas_vs_pred=float(np.nansum(((dlnv - pred) / eXY)[ok] ** 2)), ndf=int(ok.sum()),
                   chi2_meas_vs_flat=float(np.nansum((dlnv / eXY)[ok] ** 2)))
    # X vs Y v maps: do the two views see the same v(x, y) (gas) once their global scales are divided out?
    okc = np.isfinite(vX) & np.isfinite(vY)
    xy_maps = dict(corr=float(np.corrcoef(vX[okc], vY[okc])[0, 1]) if okc.sum() > 3 else np.nan,
                   chi2=float(np.nansum(((vX - vY) ** 2 / (eX ** 2 + eY ** 2))[okc])), ndf=int(okc.sum()))
    cal = json.load(open(os.path.join(ROOT, det, 'calib_bundle', 'bundle.json')))
    amap = json.load(open(os.path.join(ROOT, 'ws_widthtilt', 'angle_maps.json')))[det]
    G, _ = gate(det, t, rt, h_evt, cal, (e, TM), beta, amap)
    R = dict(det=det, n_triggers=int(len(df)), n_clean=int(len(c)), hours=float(h.max() - h.min()), rate_per_h=float(rate),
             tick_ns=TICK_NS, Tbar=Tbar, T_evt_rsig=float(1.4826 * np.median(np.abs(T - Tbar))),
             allan=al, block_N=int(N), block_minutes=float(60 * N / rate), block_allan_pct=float(best[1]),
             vt=dict(h=tc.tolist(), T=Tb.tolist(), T_err=eb.tolist(), v_rel=(Tbar / Tb).tolist(),
                     range_pct=float(100 * (Tbar / Tb).max() - 100 * (Tbar / Tb).min())),
             ladder_blocks=lad, ladder_vs_T_slope=lad_slope,
             Tmap=dict(e=e.tolist(), T_rel=TM.tolist(), err=TE.tolist(), n=TN.tolist(),
                       rms_pct=float(100 * np.nanstd(TM)), p5_95_pct=[float(100 * (np.nanpercentile(TM, q) - 1)) for q in (5, 95)]),
             vmap=dict(e=eL.tolist(), vX=vX.tolist(), vY=vY.tolist(), vXY=vXY.tolist(), err=eXY.tolist(),
                       aX=lc['x']['a'].tolist(), aY=lc['y']['a'].tolist(), rms_pct=float(100 * np.nanstd(vXY))),
             xy_maps=xy_maps, two_map=two, gate=G)
    print(f"{det}: {len(c)} clean of {len(df)} triggers over {R['hours']:.2f} h ({rate:.0f}/h); T {Tbar:.0f} ns "
          f"(per event rsig {R['T_evt_rsig']:.0f}); block N {N} = {R['block_minutes']:.1f} min (Allan {best[1]:.2f} %); "
          f"v(t) range {R['vt']['range_pct']:.1f} %; ladder-vs-1/T slope {lad_slope:.2f}; T map rms {R['Tmap']['rms_pct']:.1f} %; "
          f"v map rms {R['vmap']['rms_pct']:.1f} %; X~Y maps corr {xy_maps['corr']:.2f} chi2 {xy_maps['chi2']:.0f}/{xy_maps['ndf']}")
    if two:
        print(f"    two-map: meas rms {two['rms_meas_pct']:.1f} %, predicted from T {two['rms_pred_pct']:.1f} %, corr {two['corr_meas_pred']:.2f}, "
              f"chi2 meas-pred {two['chi2_meas_vs_pred']:.0f} vs flat {two['chi2_meas_vs_flat']:.0f} /{two['ndf']}; gap rms {two['rms_gap_pct']:.1f} %, gas rms {two['rms_gas_pct']:.1f} %; "
              f"k = dlnv/dlnT {two['k']:+.2f}±{two['k_err']:.2f} (bow-only {two['k_bow']:+.2f}, gas-only -1; chi2 {two['chi2_k']:.0f})")
    for p in 'xy':
        for name, g in [(k_, g_) for k_, g_ in G[p].items() if isinstance(g_, dict)]:
            print(f"    gate {p} {name:14s} s68 {g['s68']:.3f} incl {g['s68_incl']:.3f} scale spread time {g['scale_spread_time_pct']:.2f} "
                  f"cells {g['scale_spread_cells_pct']:.2f} | d " + ' '.join(f'{a:+.3f}±{b:.3f}' for a, b in zip(g['delta_vs_prod'], g['delta_err'])))
    return R


def main():
    dets = sys.argv[1:] or DETS
    # det3 golden T for the drift-scan v(E)
    f3 = os.path.join(FA, GOLD_ROW['det3'], 'fa', 'colsum.npz')
    d3 = None
    if os.path.exists(f3):
        df, _, _ = load_colsum(f3); m = clean(df); T = 0.5 * (df.x_T.to_numpy() + df.y_T.to_numpy())[m]
        d3 = (tmean(T), float(1.2533 * 1.4826 * np.median(np.abs(T - np.median(T))) / np.sqrt(len(T))), int(m.sum()))
    beta, pts = beta_from_scan(d3)
    print('drift scan T(V):', pts, ' beta =', beta)
    summ = dict(beta=beta, scan=pts, tick_ns=TICK_NS)
    for det in dets:
        if not os.path.exists(os.path.join(FA, GOLD_ROW[det], 'fa', 'colsum.npz')):
            print('no colsum for', det); continue
        R = analyse(det, beta)
        json.dump(R, open(os.path.join(RES, f'vgas_{det}.json'), 'w'), indent=1, default=float)
        summ[det] = {k: R[k] for k in ('n_clean', 'hours', 'block_N', 'block_minutes', 'block_allan_pct', 'ladder_vs_T_slope', 'xy_maps')}
        summ[det].update(vt_range_pct=R['vt']['range_pct'], Tmap_rms_pct=R['Tmap']['rms_pct'], vmap_rms_pct=R['vmap']['rms_pct'],
                         two_map={k: v for k, v in R['two_map'].items() if not isinstance(v, list)})
    json.dump(summ, open(os.path.join(RES, 'vgas_summary.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
