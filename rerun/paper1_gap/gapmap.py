#!/usr/bin/env python3
"""gapmap.py <work> -- the drift gap of chamber A (det3) across its surface, three ways, on pass 1 (Paper I Sec. 5.1).

Two runs at the operating point (490 V mesh, 1000 V drift), the same mounting (top slot):
  run1  sat_det3, 6-27 saturday long run     (rerun/paper1_gap/build_incl.py caches; pass-1 tables)
  run2  g_det3_wknd, 6-28 p2 overnight        (same)

Methods, per position cell (the track at mid-gap, ref_mesh + 14 mm x tan_ref, inside the >= 30 mm interior):

  G  geometric, no time: gap_extent's half-height edges of the X charge of inclined tracks against
     z = (x - x_mesh)/tan_ref (|tan| 0.2-0.8), the S4 recipe (rerun/calib/s4/s4r_gap.py: mask = pass dead list + audit
     flags, mean of the unsaturated and satmask selections).  Per cell the + and - tan halves are averaged (a local drift
     tilt delta reads g (1 + delta/tan) on one half and g (1 - delta/tan) on the other).
  T  column duration: colsum.py's per-trigger box fit T (strip sum of the track's cluster, any angle), matched to the
     pass-1 M3 table; unsaturated single clusters; 20 % trimmed mean per cell.  T = g / v(V/g).
  V  angle-scale velocity: per cell and view, w*1e3 = a + v tan_ref (robust line), the pass-1 fitted rate against the
     M3 slope.  No gap enters.
  VT the paper's product, per cell: g = v_ladder x T.  The field dependence cancels: T ~ g^(1+beta), v ~ g^-beta.

A real gap change dg/g predicts dlnT = (1 + beta) dg/g and dlnv = -beta dg/g (beta = dln v / dln E = 0.68, Magboltz at
0.97 % water and 355 V/cm, rerun/paper1_gas/gasinv.py); a gas change at constant gap predicts dlnT = -dlnv and a flat
product.  The board bow (pass-1 bow block, mesh height h with z up, cathode above) predicts dg = -(h - <h>) if the
cathode stays flat and dg = 0 if it bows with the board.

Writes <work>/gapmap.json and figures under rerun/paper1_gap/figures/.
"""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CB = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
sys.path[:0] = [CB, REPO]
import gap_extent as GE  # noqa: E402

GAP0 = 28.19            # chamber-wide geometric gap (S4 re-check, interior), mm
BETA = 0.68             # dln v / dln E at 355 V/cm, 0.97 % water (Magboltz bench grid)
MID = 14.0              # mid-gap depth used for the cell position, mm
EDGE = 30.0             # interior margin, mm
XA, YA = (0.0, 398.6), (18.0, 379.9)      # chamber A active area (common/mx17_active_area.TRUE_ACTIVE)
IN_X, IN_Y = (XA[0] + EDGE, XA[1] - EDGE), (YA[0] + EDGE, YA[1] - EDGE)
MASK_X = [39, 262, 267, 307, 332, 398, 400, 501]   # rerun/calib/s4/results/s4r/mask_det3.json 'union'
MASK_Y = None                                       # the pass bundle's Y dead list (read in main)
TAN_LADDER = 0.6
NBOOT = 40

RUNS = {
    'run1': dict(key='sat_det3', rel='mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V',
                 colsum='sat_colsum.npz'),
    'run2': dict(key='g_det3_wknd', rel='mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check',
                 colsum='p2_colsum.npz'),
}


def edges(n, lo, hi):
    return np.linspace(lo, hi, n + 1)


GRIDS = {'3x3': (edges(3, *IN_X), edges(3, *IN_Y)),
         '4x4': (edges(4, *IN_X), edges(4, *IN_Y)),
         'x5': (edges(5, *IN_X), np.array(IN_Y)),
         'y5': (np.array(IN_X), edges(5, *IN_Y))}


def cell_of(x, y, ex, ey):
    i = np.searchsorted(ex, x) - 1
    j = np.searchsorted(ey, y) - 1
    ok = (i >= 0) & (i < len(ex) - 1) & (j >= 0) & (j < len(ey) - 1)
    return np.where(ok, i, -1), np.where(ok, j, -1)


# ---------------------------------------------------------------- G: geometric
def g_estimate(ev, bad, plane='x'):
    """S4's gap of a set of events with the +/- tan halves averaged; also the plain (pooled) value.  X: window
    integral (the S4 estimator).  Y: peak sample (the Y integral drains through the RC spreading; the peak version reads
    high by a blur-dependent 0.6-2.3 mm in the gap_extent toy, so Y is used relative to its own chamber value)."""
    q = 'int' if plane == 'x' else 'peak'
    out = {}
    for lab, sm in (('default', False), ('satmask', True)):
        GE.SATMASK = sm
        n, P, E = GE.profile(ev, plane, bad, qmode=q)
        pooled = GE.half(P, E)['gap'] if n >= 40 else np.nan
        hs = []
        for s in (1, -1):
            ns, Ps, Es = GE.profile(ev, plane, bad, qmode=q, tsign=s)
            hs.append(GE.half(Ps, Es)['gap'] if ns >= 25 else np.nan)
        out[lab] = (pooled, *hs, n)
    GE.SATMASK = False
    sym = [0.5 * (out[k][1] + out[k][2]) for k in ('default', 'satmask')]
    pool = [out[k][0] for k in ('default', 'satmask')]
    return dict(sym=float(np.nanmean(sym)) if np.isfinite(sym).any() else np.nan,
                pooled=float(np.nanmean(pool)) if np.isfinite(pool).any() else np.nan,
                plus=float(np.nanmean([out[k][1] for k in out])), minus=float(np.nanmean([out[k][2] for k in out])),
                n=int(out['satmask'][3]), n_unsat=int(out['default'][3]))


def _g_cell(args):
    ev, bad, nboot, seed, plane = args
    r = g_estimate(ev, bad, plane)
    rng = np.random.default_rng(seed)
    bs, bp = [], []
    for _ in range(nboot):
        sub = [ev[i] for i in rng.integers(0, len(ev), len(ev))]
        b = g_estimate(sub, bad, plane)
        bs.append(b['sym']); bp.append(b['pooled'])
    bs, bp = np.array(bs), np.array(bp)
    r['sym_err'] = float(np.nanstd(bs)) if np.isfinite(bs).sum() > 5 else np.nan
    r['pooled_err'] = float(np.nanstd(bp)) if np.isfinite(bp).sum() > 5 else np.nan
    r['boot_fail'] = int((~np.isfinite(bs)).sum())
    return r


def g_maps(incl, pool, plane='x'):
    bad = set(MASK_X) if plane == 'x' else set(MASK_Y)
    ev = list(incl.values())
    xm = np.array([e['ref_mesh_x'] + MID * e['tan_x'] for e in ev])
    ym = np.array([e['ref_mesh_y'] + MID * e['tan_y'] for e in ev])
    out = {}
    inside = (xm > IN_X[0]) & (xm < IN_X[1]) & (ym > IN_Y[0]) & (ym < IN_Y[1])
    allin = [e for e, k in zip(ev, inside) if k]
    out['all'] = pool.submit(_g_cell, (allin, bad, NBOOT, 1, plane))
    for gname, (ex, ey) in GRIDS.items():
        i, j = cell_of(xm, ym, ex, ey)
        for a in range(len(ex) - 1):
            for b in range(len(ey) - 1):
                sel = [e for e, ii, jj in zip(ev, i, j) if ii == a and jj == b]
                out[f'{gname}:{a},{b}'] = pool.submit(_g_cell, (sel, bad, NBOOT, 100 * a + b + 7, plane))
    return out


# ---------------------------------------------------------------- tables
def load_table(work, rel):
    d = pd.read_parquet(f'{work}/pass1/{rel}/bow_gate_table.parquet')
    d = d[(d.n_tracks == 1) & d.x_ok & d.y_ok & d.x_quality_ok & d.y_quality_ok & ~d.spark
          & (d.Chi2X < 1) & (d.Chi2Y < 1) & (d.NClusX >= 4) & (d.NClusY >= 4)].copy()
    d['xm'] = d.ref_mesh_x + MID * d.tan_ref_x
    d['ym'] = d.ref_mesh_y + MID * d.tan_ref_y
    d = d[(d.xm > IN_X[0]) & (d.xm < IN_X[1]) & (d.ym > IN_Y[0]) & (d.ym < IN_Y[1])]
    return d


def tmean(a, cut=0.2):
    a = np.sort(np.asarray(a, float)); a = a[np.isfinite(a)]
    k = int(cut * len(a))
    return float(a[k:len(a) - k].mean()) if len(a) - 2 * k > 5 else np.nan


def boot(f, a, nb=200, seed=3):
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    v = [f(a[rng.integers(0, len(a), len(a))]) for _ in range(nb)]
    return float(np.nanstd(v))


def colsum_T(work, run, tab):
    z = np.load(f'{work}/colsum/{RUNS[run]["colsum"]}')
    tags = z['acq_tags']
    cs = pd.DataFrame(dict(acq_tag=tags[z['acq']], event_id=z['eid'], xT=z['x_T'], yT=z['y_T'], xq=z['x_q'], yq=z['y_q'],
                           ok=z['x_has'] & z['y_has'] & (z['x_ncl'] == 1) & (z['y_ncl'] == 1) & ~z['x_sat'] & ~z['y_sat']
                           & (z['x_nsig'] >= 5) & (z['y_nsig'] >= 5),
                           sat=z['x_sat'] | z['y_sat']))
    m = tab.merge(cs, on=['acq_tag', 'event_id'], how='inner')
    n_all = len(m)
    m = m[m.ok & m.xT.between(400, 1400) & m.yT.between(400, 1400)].copy()
    m['T'] = 0.5 * (m.xT + m.yT)
    # amplitude walk of T, removed globally (a gain map must not read as a gap map)
    lq = np.log(0.5 * (m.xq + m.yq))
    A = np.vstack([np.ones(len(m)), lq - lq.median()]).T
    c = np.linalg.lstsq(A, m['T'].values, rcond=None)[0]
    m['T_amp'] = m['T'] - c[1] * (lq - lq.median())
    return m, dict(n_matched=n_all, n_clean=len(m), T_slope_per_lnq=float(c[1]))


def ladder(d, plane):
    """robust w*1e3 = a + v tan_ref: two passes, 3.5 robust sigma."""
    t = d[f'tan_ref_{plane}'].values; w = d[f'{plane}_w'].values * 1e3
    s = np.abs(t) <= TAN_LADDER
    t, w = t[s], w[s]
    if len(t) < 40:
        return np.nan, np.nan, 0
    keep = np.ones(len(t), bool)
    for _ in range(3):
        A = np.vstack([np.ones(keep.sum()), t[keep]]).T
        c = np.linalg.lstsq(A, w[keep], rcond=None)[0]
        r = w - (c[0] + c[1] * t)
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < 3.5 * rs
    A = np.vstack([np.ones(keep.sum()), t[keep]]).T
    c, res = np.linalg.lstsq(A, w[keep], rcond=None)[:2]
    sig2 = float(np.sum((w[keep] - A @ c) ** 2) / max(keep.sum() - 2, 1))
    cov = sig2 * np.linalg.inv(A.T @ A)
    return float(c[1]), float(np.sqrt(cov[1, 1])), int(keep.sum())


def tv_maps(tab, m):
    out = {}
    vX0, _, _ = ladder(tab, 'x'); vY0, _, _ = ladder(tab, 'y')
    T0 = tmean(m['T_amp'])
    out['all'] = dict(vX=vX0, vY=vY0, T=T0, T_raw=tmean(m['T']), n_T=len(m), n_v=len(tab))
    hb = tab[['x_bow_h', 'y_bow_h']].where(lambda q: q.abs() < 10)
    tab = tab.assign(h=hb.mean(axis=1))
    h0 = float(tab.h.mean())
    for gname, (ex, ey) in GRIDS.items():
        it, jt = cell_of(tab.xm.values, tab.ym.values, ex, ey)
        im, jm = cell_of(m.xm.values, m.ym.values, ex, ey)
        for a in range(len(ex) - 1):
            for b in range(len(ey) - 1):
                dc = tab[(it == a) & (jt == b)]
                mc = m[(im == a) & (jm == b)]
                vx, ex_, nx = ladder(dc, 'x'); vy, ey_, ny = ladder(dc, 'y')
                rx, ry = vx / vX0, vy / vY0
                wx, wy = 1 / (ex_ / vX0) ** 2, 1 / (ey_ / vY0) ** 2
                v_rel = (rx * wx + ry * wy) / (wx + wy)
                v_err = 1 / np.sqrt(wx + wy)
                T = tmean(mc['T_amp']); Te = boot(tmean, mc['T_amp'].values) if len(mc) > 20 else np.nan
                Traw = tmean(mc['T'])
                out[f'{gname}:{a},{b}'] = dict(
                    v_rel=float(v_rel), v_err=float(v_err), vX_rel=float(rx), vY_rel=float(ry),
                    vX_err=float(ex_ / vX0), vY_err=float(ey_ / vY0), n_v=int(len(dc)),
                    T_rel=float(T / T0), T_err=float(Te / T0), T_raw_rel=float(Traw / out['all']['T_raw']), n_T=int(len(mc)),
                    h=float(dc.h.mean() - h0) if len(dc) else np.nan,
                    xc=float(0.5 * (ex[a] + ex[a + 1])), yc=float(0.5 * (ey[b] + ey[b + 1])))
    return out


def derive(cell):
    """gap deviations (mm) from the timing quantities of one cell."""
    lt, lv = np.log(cell['T_rel']), np.log(cell['v_rel'])
    et, evv = cell['T_err'] / cell['T_rel'], cell['v_err'] / cell['v_rel']
    return dict(dg_VT=GAP0 * (cell['T_rel'] * cell['v_rel'] - 1), dg_VT_err=GAP0 * np.hypot(et, evv),
                dg_T=GAP0 * lt / (1 + BETA), dg_T_err=GAP0 * et / (1 + BETA),
                dg_V=-GAP0 * lv / BETA, dg_V_err=GAP0 * evv / BETA,
                dg_bow=-cell['h'])


def main():
    work = sys.argv[1]
    res = {}
    with ProcessPoolExecutor(14) as pool:
        futs = {}
        for run, R in RUNS.items():
            incl = pickle.load(open(f'{work}/build_{R["key"]}/out/{R["key"]}/incl.pkl', 'rb'))
            global MASK_Y
            MASK_Y = json.load(open(f'{work}/mirror/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/'
                                    f'{R["rel"]}/mx17_3/wft/calib_bundle/bundle.json'))['dead']['y']
            futs[run] = (g_maps(incl, pool, 'x'), g_maps(incl, pool, 'y'))
        for run, R in RUNS.items():
            tab = load_table(work, R['rel'])
            m, tinfo = colsum_T(work, run, tab)
            tv = tv_maps(tab, m)
            G = {k: f.result() for k, f in futs[run][0].items()}
            GY = {k: f.result() for k, f in futs[run][1].items()}
            cells = {}
            for k in tv:
                c = dict(tv[k]); c.update({f'G_{kk}': vv for kk, vv in G[k].items()})
                c.update({f'GY_{kk}': vv for kk, vv in GY[k].items()})
                if k != 'all':
                    c.update(derive(c))
                cells[k] = c
            res[run] = dict(T_info=tinfo, cells=cells)
            print(run, json.dumps(cells['all'], default=float), json.dumps(tinfo), flush=True)
    json.dump(dict(gap0=GAP0, beta=BETA, interior=dict(x=IN_X, y=IN_Y), grids={k: [list(map(float, a)), list(map(float, b))]
                   for k, (a, b) in GRIDS.items()}, runs=res), open(f'{work}/gapmap.json', 'w'), indent=1, default=float)
    print('wrote', f'{work}/gapmap.json')


if __name__ == '__main__':
    main()
