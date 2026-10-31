#!/usr/bin/env python3
"""vtc_measure.py <key> --colsum C.npz --headon H.pkl --out OUT.json [--survey S.json] [--nblk 4] [--nboot 20]
-- the v(t) cross-check of one bench pass-1 row (CROSSCHECK_2026-10-31): the survey's M3 ladder against the two
timing estimators that do not use the M3 angle scale, in the same time blocks.

    ladder   vt_bench's estimator (set_w0kw's kw per block / the row's kw, after the bow): an angle scale, so it moves
             with anything that rescales the fitted w (v, but also a brightness-dependent fit scale; FA: the ladder
             moved 1.45x (det4) / 1.68x (det6) faster than 1/T on the golden rows).
    column   FA's model-free column duration (colsum.py, every trigger; no M3, no track model): v_rel = T_row / T_block,
             clean triggers (both views >= 5 pulse strips, one cluster, unsaturated), the mean of the two views,
             20 % trimmed mean.  Variants: per view, the saturated class, the interior (both centroids >= 30 mm inside
             the active area), and 'std' = standardised against the row's composition (each trigger's T over the
             row-wide median of its cell, 4 x 4 cells of 100 mm x amplitude tercile): a drift of WHERE the triggers sit
             (det4's stripes) or of their brightness cannot move it.  'head' = the X column of the stack's own events
             (the head-on cache, |tan_x| < 0.03; X view alone, one cluster, unsaturated / 'head_sat' saturated): the
             same events as the stack through the other estimator.
    stack    the S4 head-on stack (x_width.measure, |tan_x| < 0.03 from M3 -- the selection only, not the scale) with the
             re-check recipe: X mask = the row bundle's dead list + strip_gain_audit's flags, unsaturated (< 3500 ADC),
             pass template; whole area (relative v needs no interior cut) and the >= 30 mm interior.
Blocks: nblk equal-count blocks in the ladder's kw-window events (the ladder needs the counts; the column has ~10x
more triggers).  The column is also given at the survey's own block edges (--survey).  Per estimator a weighted line
gives dv over the row (error inflated by sqrt(chi2/dof) above 1).  Per block also: the gain (median summed charge of
the clean triggers / the row's) and the share of clean triggers in the interior, for the 'stripes / gain fake' check.
Read-only on Analysis_pass1.
"""
import argparse
import csv
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CB = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
sys.path[:0] = [os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'rerun', 'vt_all'), CB, REPO]
import s4_stacks  # noqa: E402
import s4f_fit  # noqa: E402
import x_width as xw  # noqa: E402
import vt_bench as vb  # noqa: E402
from vt_common import ladder, kw_window, blocks_equal_count, stem_time  # noqa: E402
from strip_gain_audit import bad_set  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402

GOLD = {'mx17_2': 'o22_long_det2', 'mx17_3': 'sat_det3', 'mx17_4': 'g_det4', 'mx17_6': 'g_det6_long',
        'mx17_7': 'g_det7_long'}   # strip_gain_audit entries (a chamber property)
CH_MIN = {'mx17_7': 256}            # det7 X: quote ch 256-511 only (faulty connectors 0/1/3)
SAT = 3500.0
EDGE = 30.0
NSIG_MIN = 5
TRIM = 0.2
NB_BOOT = 200


def tmean(a):
    a = np.sort(np.asarray(a, float)[np.isfinite(np.asarray(a, float))])
    k = int(TRIM * len(a))
    return float(np.mean(a[k:len(a) - k])) if len(a) > 2 * k + 2 else np.nan


def tmean_err(a, rng):
    a = np.asarray(a, float)[np.isfinite(np.asarray(a, float))]
    if len(a) < 20:
        return np.nan
    return float(np.std([tmean(a[rng.integers(0, len(a), len(a))]) for _ in range(NB_BOOT)]))


def interior(det, x, y, d=EDGE):
    b = TRUE_ACTIVE_BY_DET[det]
    return np.minimum.reduce([x - b['x'][0], b['x'][1] - x, y - b['y'][0], b['y'][1] - y]) >= d


def colsum_table(path, det):
    """per trigger: t_h on the bench clock, T (x, y, mean), class flags, centroids, charge."""
    z = np.load(path, allow_pickle=True)
    tags = z['acq_tags']
    ts = np.where(z['x_ts'] >= 0, z['x_ts'], z['y_ts']).astype(float)
    t0 = np.array([(stem_time(t) - vb.T_REF).total_seconds() / 3600 for t in tags])
    t_h = t0[z['acq'].astype(int)] + ts * vb.TICK_S / 3600
    t_h[ts < 0] = np.nan
    one = lambda p: (z[f'{p}_has'] & (z[f'{p}_nsig'] >= NSIG_MIN) & (z[f'{p}_ncl'] == 1) & np.isfinite(z[f'{p}_T']))
    both = one('x') & one('y')
    T = 0.5 * (z['x_T'] + z['y_T'])
    q = 0.5 * (z['x_q'] + z['y_q'])
    unsat = both & ~z['x_sat'] & ~z['y_sat']
    sat = both & z['x_sat'] & z['y_sat']
    inn = interior(det, z['x_cen'], z['y_cen'])
    # X alone (the head-on events have too few pulse strips for 'both'): a fitted X column, one cluster
    xok = z['x_has'] & (z['x_ncl'] == 1) & np.isfinite(z['x_T'])
    return dict(eid=z['eid'].astype(int), acq=z['acq'].astype(int), t_h=t_h, T=T, Tx=z['x_T'], Ty=z['y_T'], q=q,
                unsat=unsat, sat=sat, inn=inn & both, xc=z['x_cen'], yc=z['y_cen'],
                x_ok_unsat=xok & ~z['x_sat'], x_ok_sat=xok & z['x_sat'])


def std_ratio(C, sel):
    """each trigger's T over the row-wide median T of its (4 x 4 x 100 mm) x (amplitude tercile) cell."""
    q3 = np.nanquantile(C['q'][sel], [1 / 3, 2 / 3])
    cell = (np.clip((C['xc'] // 100).astype(int), 0, 3) * 4 + np.clip((C['yc'] // 100).astype(int), 0, 3)) * 3 \
        + np.searchsorted(q3, C['q'])
    rat = np.full(len(C['T']), np.nan)
    for c in np.unique(cell[sel]):
        k = sel & (cell == c)
        if k.sum() >= 30:
            rat[k] = C['T'][k] / np.median(C['T'][k])
    return rat


def line(t, r, e):
    k = np.isfinite(t) & np.isfinite(r) & np.isfinite(e) & (e > 0)
    if k.sum() < 3:
        return None
    t, r, e = t[k], r[k], e[k]
    w = 1 / e ** 2
    tm = np.sum(w * t) / np.sum(w)
    A = np.column_stack([np.ones_like(t), t - tm]) / e[:, None]
    c, *_ = np.linalg.lstsq(A, r / e, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    chi2 = float(np.sum((r / e - A @ c) ** 2)); dof = len(t) - 2
    infl = float(np.sqrt(max(1.0, chi2 / dof))) if dof > 0 else 1.0
    return dict(slope=float(c[1] / c[0]), slope_err=float(np.sqrt(cov[1, 1]) / c[0] * infl), chi2=chi2, dof=dof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--colsum', required=True); ap.add_argument('--headon', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--survey'); ap.add_argument('--root', default=vb.ROOT)
    ap.add_argument('--nblk', type=int, default=4); ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    r = reg[a.key]
    det = r['det']
    wdir = os.path.join(a.root, r['run'], r['subrun'], det, 'wft')
    m, cal, df, V = vb.load_row(wdir, a.root)
    t_lo, t_hi = float(np.nanmin(df.t_h)), float(np.nanmax(df.t_h))
    dur = t_hi - t_lo
    out = dict(key=a.key, det=det, t_start=t_lo, t_end=t_hi, duration_h=dur, v_drift=cal.v_drift, kw=cal.kw)

    # ---- blocks (equal count in the ladder's kw-window events, the view with more of them)
    pb = max(('x', 'y'), key=lambda p: kw_window(V[p]['tp']).sum())
    nkw = int(kw_window(V[pb]['tp']).sum())
    edges = blocks_equal_count(V[pb]['t_h'], kw_window(V[pb]['tp']), max(150, nkw // a.nblk))
    sets = {'blk': list(zip(edges[:-1], edges[1:]))}
    if a.survey and os.path.exists(a.survey):
        sv = json.load(open(a.survey))
        sets['survey'] = [(b['lo'], b['hi']) for b in sv['blocks']]
        out['survey'] = dict(dv=sv.get('dv_total'), dv_err=sv.get('dv_err'), verdict=sv.get('verdict'))

    # ---- column (colsum, every trigger)
    C = colsum_table(a.colsum, det)
    inrow = np.isfinite(C['t_h']) & (C['t_h'] >= t_lo - 0.05) & (C['t_h'] <= t_hi + 0.05)
    out['colsum'] = dict(n_trig=int(len(C['T'])), n_in_row=int(inrow.sum()), n_unsat=int((C['unsat'] & inrow).sum()),
                         eid_unique=bool(len(np.unique(C['eid'])) == len(C['eid'])))
    C['std'] = std_ratio(C, C['unsat'] & inrow)
    # the column on the head-on cache's own events (M3 |tan| < 0.06 in a view): same events as the stack, other estimator
    H0 = pickle.load(open(a.headon, 'rb'))
    hx = {k for k, e in H0.items() if abs(e['tan_x']) < xw.TAN}
    ish = np.isin(C['eid'], np.fromiter(hx, int)) if out['colsum']['eid_unique'] else np.zeros(len(C['eid']), bool)
    cols = {'col': ('T', C['unsat']), 'col_x': ('Tx', C['unsat']), 'col_y': ('Ty', C['unsat']),
            'col_sat': ('T', C['sat']), 'col_in30': ('T', C['unsat'] & C['inn']), 'col_std': ('std', C['unsat']),
            'col_head': ('Tx', ish & C['x_ok_unsat']), 'col_head_sat': ('Tx', ish & C['x_ok_sat'])}
    del H0
    rowT = {n: tmean(C[f][s & inrow]) for n, (f, s) in cols.items()}
    q_row = float(np.nanmedian(C['q'][C['unsat'] & inrow]))
    n_in_row = int((C['unsat'] & inrow).sum())
    inn_row = float((C['unsat'] & inrow & C['inn']).sum() / max(1, n_in_row))

    # ---- ladder row values
    kr = {p: ladder(cal, V[p]['w'], V[p]['tp'])[1] for p in ('x', 'y')}

    # ---- stack: head-on cache, masked, eid -> t_h through colsum
    H = pickle.load(open(a.headon, 'rb'))
    t_of = dict(zip(C['eid'].tolist(), C['t_h'].tolist())) if out['colsum']['eid_unique'] else {}
    b = json.load(open(os.path.join(wdir, 'calib_bundle', 'bundle.json')))
    audit = json.load(open(os.path.join(CB, 'results', 'strip_gain_audit.json')))
    mk = sorted(set(b['dead']['x']) | bad_set(audit[GOLD[det]], 'x'))
    H = s4f_fit.mask(H, set(mk))
    if det in CH_MIN:
        H = {k: e for k, e in H.items() if e['x']['ch'][np.argmax(e['x']['W'].max(axis=1))] >= CH_MIN[det]}
    H = {k: e for k, e in H.items() if e['x']['W'].max() < SAT and np.isfinite(t_of.get(k, np.nan))}
    H_in = {k: e for k, e in H.items() if interior(det, np.array([e['ref_mesh_x']]), np.array([e['ref_mesh_y']]))[0]}
    h = s4_stacks.pass_template(os.path.join(wdir, 'calib_bundle'))
    T0 = 30.0 / (cal.v_drift * 1e-3)
    out['stack'] = dict(mask_x=mk, n_mask=len(mk), n_unsat=int(sum(abs(e['tan_x']) < xw.TAN for e in H.values())),
                        n_unsat_in30=int(sum(abs(e['tan_x']) < xw.TAN for e in H_in.values())), template='row bundle')
    pool = ProcessPoolExecutor(a.workers)
    stack_row = {}
    for nm, ev in (('stack', H), ('stack_in30', H_in)):
        n = sum(abs(e['tan_x']) < xw.TAN for e in ev.values())
        stack_row[nm] = xw.measure(pool, h, ev, T0, a.nboot) if n >= xw.MIN_EV else None
    out['stack']['row'] = stack_row

    for sname, blks in sets.items():
        res = []
        for lo, hi in blks:
            B = dict(lo=float(lo), hi=float(hi))
            kk = inrow & (C['t_h'] >= lo) & (C['t_h'] < hi)
            for n, (f, s) in cols.items():
                x = C[f][s & kk]
                Tb, Te = tmean(x), tmean_err(x, rng)
                B[n] = dict(n=int(np.isfinite(x).sum()), T=Tb, T_err=Te, r=rowT[n] / Tb if Tb else np.nan,
                            r_err=rowT[n] * Te / Tb ** 2 if Tb else np.nan)
            ku = C['unsat'] & kk
            B['t'] = float(np.nanmedian(C['t_h'][ku])) if ku.any() else np.nan
            B['gain'] = float(np.nanmedian(C['q'][ku]) / q_row) if ku.any() else np.nan
            B['in30_share'] = float((ku & C['inn']).sum() / max(1, ku.sum())) / inn_row if inn_row else np.nan
            if sname == 'blk':
                for p in ('x', 'y'):
                    k = (V[p]['t_h'] >= lo) & (V[p]['t_h'] < hi)
                    _, kw, nn, e = ladder(cal, V[p]['w'][k], V[p]['tp'][k])
                    B[f'lad_{p}'] = dict(n=nn, r=kw / kr[p], r_err=e / kr[p])
                rr = [B[f'lad_{p}'] for p in ('x', 'y') if np.isfinite(B[f'lad_{p}']['r'])]
                B['lad'] = dict(r=float(np.mean([x['r'] for x in rr])),
                                r_err=float(np.sqrt(np.sum([x['r_err'] ** 2 for x in rr])) / len(rr)))
                for nm, ev in (('stack', H), ('stack_in30', H_in)):
                    sub = {k: e for k, e in ev.items() if lo <= t_of[k] < hi}
                    n = sum(abs(e['tan_x']) < xw.TAN for e in sub.values())
                    if n >= xw.MIN_EV and stack_row[nm]:
                        mm = xw.measure(pool, h, sub, T0, a.nboot)
                        Te = mm['T'] * mm['v_err'] / mm['v']
                        Tr = stack_row[nm]['T']
                        B[nm] = dict(n=int(n), T=mm['T'], T_err=Te, r=Tr / mm['T'], r_err=Tr * Te / mm['T'] ** 2,
                                     Dp=float(np.sqrt(2 * mm['Dd'])), sig0=mm['sig0'])
                    else:
                        B[nm] = dict(n=int(n), r=np.nan, r_err=np.nan)
                print(f"[vtc] {a.key} blk {lo:.2f}-{hi:.2f}: lad {B['lad']['r']:.4f}  col {B['col']['r']:.4f}  "
                      f"std {B['col_std']['r']:.4f}  stack {B['stack']['r']:.4f}  gain {B['gain']:.3f}", flush=True)
            res.append(B)
        L = {}
        names = [n for n in res[0] if isinstance(res[0][n], dict) and 'r' in res[0][n]]
        tt = np.array([B['t'] for B in res])
        for n in names:
            f = line(tt, np.array([B[n]['r'] for B in res], float), np.array([B[n]['r_err'] for B in res], float))
            if f:
                L[n] = dict(dv=f['slope'] * dur, dv_err=f['slope_err'] * dur, chi2=f['chi2'], dof=f['dof'])
        out[sname] = dict(blocks=res, lines=L)
    pool.shutdown()
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print('[vtc] lines', json.dumps({s: {n: round(v['dv'] * 100, 2) for n, v in out[s]['lines'].items()}
                                     for s in sets}), flush=True)


if __name__ == '__main__':
    main()
