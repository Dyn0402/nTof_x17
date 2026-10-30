#!/usr/bin/env python3
"""vt_bench.py <registry key> [...] --root R --out-dir D -- within-row v(t) of one bench pass-1 row (VT_PLAN_2026-10-30).

The estimator is rerun/vt_within_rows' (the M3 ladder of rerun/w12/set_w0kw.py, per time block, after the tan-form bow):
per view, kw on blocks of ~N_PER events in the kw window, divided by the row's own kw, is v_rel(t) = v(t) / v_row.
A joint line (one slope, one intercept per view) gives the drift over the row; the gate (GATE_* below) decides whether
the row needs a v(t) correction.  Read-only on Analysis_pass1.  Per row: <out-dir>/<key>.json, and for a row the gate
flags, <out-dir>/<key>.vt_map.json (wft v_map['t'] block, the line clamped to the row).
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'vt_within_rows')]
from wft import reco as wr  # noqa: E402
from vt_common import ladder, bow_aS, stem_time, kw_window, blocks_equal_count, TICK_S  # noqa: E402

ROOT = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
T_REF = dt.datetime(2026, 6, 1, 0, 0)      # bench clock: hours since 2026-06-01 00:00 CERN local
N_PER = 300                                # kw-window events per block and view (vt_within_rows: ~1 % on kw)
N_MIN = 150                                # fewest per block; a row with < 2 x N_MIN gets no time measurement
GATE_DV = 0.02                             # |v change over the row| that needs a correction (fraction)
GATE_Z = 3.0                               # ... and the slope's significance (after the excess-scatter inflation)
GATE_CHI2 = 3.0                            # line chi2/dof above this (dof >= 5): v jumps block to block, no line


def clock_h(acq_tag, timestamp):
    t0 = {t: (stem_time(t) - T_REF).total_seconds() / 3600 for t in pd.unique(acq_tag)}
    return np.array([t0[t] for t in acq_tag]) + np.asarray(timestamp, float) * TICK_S / 3600


def golden_bow(root, det, src):
    for f in glob.glob(os.path.join(root, '*', '*', det, 'wft', 'calib_bundle', 'bundle.json')):
        b = json.load(open(f))
        if b.get('run_key') == src:
            return b.get('bow') or {}
    raise FileNotFoundError(f'no golden bundle {src} for {det}')


def load_row(wdir, root):
    """vt_common.load_row without the 6-26 specifics: the row's calibration, clock and per-view ladder arrays."""
    m = json.load(open(os.path.join(wdir, 'events.meta.json')))
    ac = m['angle_constants']
    det = m['run']['detector']
    cal = SimpleNamespace(v_drift=float(m['bundle']['v_drift']), w0=dict(ac['w0']), kw=dict(ac['kw']), bow={})
    if (m.get('bow') or {}).get('applied'):
        cal.bow = golden_bow(root, det, m['bow']['source'])
    df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    if 'n_tracks' in df:
        df = df.sort_values('n_tracks')
    df = df.drop_duplicates('event_id', keep='first').set_index('event_id')
    df['t_h'] = clock_h(df.acq_tag.to_numpy(), df.timestamp.to_numpy())
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0).fillna(0.0).to_numpy()
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0).fillna(0.0).to_numpy()
    z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    views = {}
    for p in ('x', 'y'):
        a, S = bow_aS(cal, p, U, W)
        e = z[f'{p}_eid'].astype(int)
        k = np.isin(e, df.index.to_numpy())
        e = e[k]
        pos = df.index.get_indexer(e)
        views[p] = dict(w=z[f'{p}_w'][k] * 1e3, tp=(z[f'{p}_tan_ref'][k] + a[pos]) * S[pos],
                        t_h=df.t_h.to_numpy()[pos])
    return m, cal, df, views


def line_fit(blocks):
    """joint weighted line v_rel_p(t) = c_p + s (t - tm), one slope; returns the fit and the constant-model chi2."""
    rows = [(b['t'], p, b[p]['r'], b[p]['r_err']) for b in blocks for p in ('x', 'y')
            if np.isfinite(b[p]['r']) and np.isfinite(b[p]['r_err']) and b[p]['r_err'] > 0]
    if len(rows) < 3:
        return None
    t = np.array([r[0] for r in rows]); y = np.array([r[2] for r in rows]); e = np.array([r[3] for r in rows])
    views = sorted({r[1] for r in rows})
    tm = float(np.average(t, weights=1 / e ** 2))
    A = np.column_stack([np.array([r[1] == p for r in rows], float) for p in views] + [t - tm])
    Aw, yw = A / e[:, None], y / e
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    chi2 = float(np.sum((yw - Aw @ coef) ** 2)); dof = len(y) - A.shape[1]
    A0 = A[:, :-1] / e[:, None]
    c0, *_ = np.linalg.lstsq(A0, yw, rcond=None)
    chi2_0 = float(np.sum((yw - A0 @ c0) ** 2))
    s, s_err = float(coef[-1]), float(np.sqrt(cov[-1, -1]))
    infl = float(np.sqrt(max(1.0, chi2 / dof))) if dof > 0 else 1.0
    return dict(slope_per_h=s, slope_err=s_err, inflate=infl, z=abs(s) / (s_err * infl), tm=tm, chi2=chi2, dof=dof,
                chi2_const=chi2_0, dof_const=len(y) - len(views), n_points=len(y),
                intercept={p: float(c) for p, c in zip(views, coef[:-1])})


def measure(key, reg, root):
    r = reg[key]
    wdir = os.path.join(root, r['run'], r['subrun'], r['det'], 'wft')
    m, cal, df, V = load_row(wdir, root)
    out = dict(key=key, det=r['det'], run=r['run'], subrun=r['subrun'], cls=r['cls'], refit=r['refit'],
               v_drift=cal.v_drift, kw_applied=cal.kw, n_events=int(len(df)), t_start=float(np.nanmin(df.t_h)),
               t_end=float(np.nanmax(df.t_h)), clock=f'hours since {T_REF:%Y-%m-%d %H:%M} CERN local: acq_tag stem '
               '+ DREAM timestamp x 10 ns', row={}, blocks=[])
    out['duration_h'] = out['t_end'] - out['t_start']
    for p in ('x', 'y'):
        w0, kw, n, e = ladder(cal, V[p]['w'], V[p]['tp'])
        out['row'][p] = dict(w0=w0, kw=kw, n_kw=n, kw_err=e)
    pb = max(('x', 'y'), key=lambda p: kw_window(V[p]['tp']).sum())
    n_kw = int(kw_window(V[pb]['tp']).sum())
    out['n_kw'] = n_kw
    if n_kw < 2 * N_MIN:
        out['verdict'] = 'too few tracks'
        return out
    edges = blocks_equal_count(V[pb]['t_h'], kw_window(V[pb]['tp']), max(N_MIN, min(N_PER, n_kw // 2)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = dict(lo=float(lo), hi=float(hi)); tt = []
        for p in ('x', 'y'):
            k = (V[p]['t_h'] >= lo) & (V[p]['t_h'] < hi)
            w0, kw, n, e = ladder(cal, V[p]['w'][k], V[p]['tp'][k])
            kr = out['row'][p]['kw']
            b[p] = dict(kw=kw, n=n, kw_err=e, r=kw / kr if kr else np.nan, r_err=e / kr if kr else np.nan)
            tt.append(V[p]['t_h'][k & kw_window(V[p]['tp'])])
        tt = np.concatenate(tt)
        b['t'] = float(np.median(tt)) if len(tt) else np.nan
        out['blocks'].append(b)
    return gate(out)


def gate(out):
    """The v(t) gate of VT_PLAN_2026-10-30 §3, on a row's blocks (also re-applied by make_report.py to stored JSONs).

    correct          >= 3 blocks, |dv| >= GATE_DV over the row at >= GATE_Z sigma (error inflated by the line's
                     chi2/dof), and the two views' own lines agree (|dv_x - dv_y| < 3 sigma): v is common to both views.
    candidate        the same, on 2 blocks: one line through two points has no check of its own; take the
                     chamber's night curve or the neighbouring rows before correcting.
    views disagree   the joint line passes but the views' lines do not agree: a ladder problem, not v.
    unstable         the line's chi2/dof > GATE_CHI2 (dof >= 5): the blocks jump rather than drift (det3 6-22: +-15 %
                     between neighbouring 20-min blocks, both views together, while det2 on the same gas is smooth to
                     1 %).  Not a v(t) correction; a per-block v or an exclusion, decided per row.
    significant < 2 %, flat, too few tracks / too few blocks.
    """
    out['n_blocks'] = len(out['blocks'])
    f = line_fit(out['blocks']) if out['blocks'] else None
    out['fit'] = f
    out['needs_correction'] = False
    if f is None:
        out.setdefault('verdict', 'too few blocks')
        return out
    out['dv_total'] = f['slope_per_h'] * out['duration_h']
    out['dv_err'] = f['slope_err'] * f['inflate'] * out['duration_h']
    out['views'] = {}
    nan = dict(r=np.nan, r_err=np.nan)
    for p in ('x', 'y'):
        one = [{'t': b['t'], p: b[p], ('y' if p == 'x' else 'x'): nan} for b in out['blocks']]   # this view only
        g = line_fit(one) or _two_point(out['blocks'], p)
        if g:
            out['views'][p] = dict(dv=g['slope_per_h'] * out['duration_h'],
                                   dv_err=g['slope_err'] * g['inflate'] * out['duration_h'])
    vx, vy = out['views'].get('x'), out['views'].get('y')
    agree = not (vx and vy) or abs(vx['dv'] - vy['dv']) < 3 * np.hypot(vx['dv_err'], vy['dv_err'])
    passes = abs(out['dv_total']) >= GATE_DV and f['z'] >= GATE_Z
    if f['dof'] >= 5 and f['chi2'] / f['dof'] > GATE_CHI2:
        out['verdict'] = 'unstable'
    elif passes and not agree:
        out['verdict'] = 'views disagree'
    elif passes and out['n_blocks'] >= 3:
        out['needs_correction'] = True
        out['verdict'] = 'correct'
    elif passes:
        out['verdict'] = 'candidate (2 blocks)'
    else:
        out['verdict'] = 'significant, < 2 %' if f['z'] >= GATE_Z else 'flat'
    return out


def _two_point(blocks, p):
    """a view's line through exactly two blocks (no chi2 of its own)."""
    b = [x for x in blocks if np.isfinite(x[p]['r']) and np.isfinite(x[p]['r_err'])]
    if len(b) != 2 or b[1]['t'] == b[0]['t']:
        return None
    dt_ = b[1]['t'] - b[0]['t']
    return dict(slope_per_h=(b[1][p]['r'] - b[0][p]['r']) / dt_, slope_err=np.hypot(b[0][p]['r_err'], b[1][p]['r_err']) / dt_,
                inflate=1.0)


def vt_map(res):
    """the row's line as a wft v_map['t'] block, relative to the row's own v (mean 1 over the row's events)."""
    f = res['fit']
    t = np.array([res['t_start'], res['t_end']])
    c = np.mean(list(f['intercept'].values()))
    r = (c + f['slope_per_h'] * (t - f['tm'])) / c
    return dict(model='line', clock=res['clock'], t_h=t.tolist(), v_rel=r.tolist(), v_drift=res['v_drift'],
                source='rerun/vt_all/vt_bench.py', pass1_key=res['key'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+'); ap.add_argument('--root', default=ROOT); ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    os.makedirs(a.out_dir, exist_ok=True)
    for key in a.keys:
        res = measure(key, reg, a.root)
        json.dump(res, open(os.path.join(a.out_dir, f'{key}.json'), 'w'), indent=1, default=float)
        if res.get('needs_correction'):
            json.dump(vt_map(res), open(os.path.join(a.out_dir, f'{key}.vt_map.json'), 'w'), indent=1)
        print(key, res['verdict'], {k: res.get(k) for k in ('duration_h', 'n_kw', 'n_blocks', 'dv_total', 'dv_err')},
              flush=True)


if __name__ == '__main__':
    main()
