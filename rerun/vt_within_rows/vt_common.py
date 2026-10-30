"""vt_common.py -- shared helpers of the within-row v(t) study (HANDOFF 2026-10-30 step 3, REPROCESSING_TODO C: GC).

A row's tables are read-only pass-1 products (Analysis_pass1/<run>/<subrun>/<det>/wft/): events.parquet (fitted w, p0,
tangents, DREAM timestamp + acq_tag), events.meta.json (the v, w0, kw the row was reconstructed with, and the bow block's
source row), angles/per_event.npz (03_angles: fitted w and the M3 reference tan per view), and the bow block from the
chamber's golden row (calib_bundle/bundle.json).

The angle mapping of a pass-1 row with a tan-form bow block (wft.reco.tan_bow) is

    tan = (w*1e3 - w0) / (kw v S(U, W)) - a(U, W)

so a drift velocity that moves within the row, v -> v r(t), enters only through the denominator.  r(t) is measured with
the estimator that gave the row its kw (rerun/w12/set_w0kw.py), per time block:  t' = (tan_ref + a) S,
w0 = median(w - v t') over |t'| < 0.30,  kw = median((w - w0) / (v t')) over 0.10 < |t'| < 0.40.
"""
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
sys.path.insert(0, REPO)
from wft import reco as wr  # noqa: E402

RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
for _d in (RES, FIG):
    os.makedirs(_d, exist_ok=True)

TICK_S = 10e-9                                  # DREAM timestamp tick (gc_common.TICK_S)
T_REF = dt.datetime(2026, 6, 26, 0, 0)          # clock origin: hours since 6-26 00:00 local CERN time
RUNS = ('mx17_det6_det7_hv_scan_6-26-26', 'mx17_det6_det7_overnight_6-26-26')
DETS = ('mx17_6', 'mx17_7')
NAME = {'mx17_6': 'det6', 'mx17_7': 'det7'}
COL = {'mx17_6': '#eda100', 'mx17_7': '#6250d6'}  # gc_common colours
# the estimator's windows (set_w0kw.py)
W0_TAN, KW_LO, KW_HI = 0.30, 0.10, 0.40


def data_root():
    """Local pull of the pass-1 products of the two 6-26 runs (fetch.sh); argument, $VT_DATA, else $WORK_DATA/vt."""
    r = os.environ.get('VT_DATA') or os.path.join(os.environ.get('WORK_DATA') or os.path.expanduser('~/x17'), 'vt')
    return r


def stem_time(tag):
    """'260626_00H58_000' -> datetime of the acquisition start (minute resolution; gc_common.stem_time)."""
    d, hm = tag.split('_')[:2]
    return dt.datetime(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]), int(hm[:2]), int(hm[3:5]))


def clock_h(acq_tag, timestamp):
    """hours since T_REF: the acquisition's stem time + DREAM timestamp x 10 ns."""
    t0 = {t: (stem_time(t) - T_REF).total_seconds() / 3600 for t in pd.unique(acq_tag)}
    return np.array([t0[t] for t in acq_tag]) + np.asarray(timestamp, float) * TICK_S / 3600


def rows(root):
    """[(run, subrun, det, wft dir)] for every row with a table, both runs, both chambers."""
    out = []
    for run in RUNS:
        for d in sorted(glob.glob(os.path.join(root, run, '*', '*', 'wft', 'events.parquet'))):
            p = d.split(os.sep)
            if p[-3] in DETS:
                out.append((run, p[-4], p[-3], os.path.dirname(d)))
    return out


def load_row(wdir, root):
    """One row: the event table with clock, the calibration it was reconstructed with, and per view the M3-matched
    arrays of the ladder (w, tan_ref, bow a and S at the event's U, W)."""
    m = json.load(open(os.path.join(wdir, 'events.meta.json')))
    ac = m['angle_constants']
    run, det = m['run']['run'], m['run']['detector']
    cal = SimpleNamespace(v_drift=float(m['bundle']['v_drift']), w0=dict(ac['w0']), kw=dict(ac['kw']), v_map={},
                          bow={}, angle_map={})
    if (m.get('bow') or {}).get('applied'):
        src = m['bow']['source']                 # the golden row of the chamber: its bundle carries the block
        g = glob.glob(os.path.join(root, '*', '*', det, 'wft', 'calib_bundle', 'bundle.json'))
        g = [f for f in g if json.load(open(f)).get('run_key') == src]
        cal.bow = json.load(open(g[0]))['bow']
    df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    if 'n_tracks' in df:
        df = df.sort_values('n_tracks')
    df = df.drop_duplicates('event_id', keep='first').set_index('event_id')   # 03_angles' join key is event_id
    df['t_h'] = clock_h(df.acq_tag.to_numpy(), df.timestamp.to_numpy())
    # (U, W) as the reconstruction had them (wft.reco.bow_UW: constant-mapping tangents, off-plate view -> 0)
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0)
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0)
    df['U'], df['W'] = U.fillna(0.0), W.fillna(0.0)
    for p in ('x', 'y'):
        a, S = bow_aS(cal, p, df.U.to_numpy(), df.W.to_numpy())
        df[f'{p}_a'], df[f'{p}_S'] = a, S
    z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    views = {}
    for p in ('x', 'y'):
        e = z[f'{p}_eid'].astype(int)
        k = np.isin(e, df.index.to_numpy())
        e = e[k]
        r = df.loc[e]
        views[p] = dict(eid=e, w=z[f'{p}_w'][k] * 1e3, tan_ref=z[f'{p}_tan_ref'][k], t_h=r.t_h.to_numpy(),
                        a=r[f'{p}_a'].to_numpy(), S=r[f'{p}_S'].to_numpy(), tan_fit=z[f'{p}_tan_fit'][k])
    return dict(run=run, subrun=m['run']['sub_run'], det=det, key=m['run']['key'], cal=cal, df=df, views=views,
                resist_V=_resist(m['run']['sub_run'], det), wdir=wdir)


def _resist(subrun, det):
    import re
    s = re.search(rf'{NAME[det]}_(\d+)V', subrun)
    return int(s.group(1)) if s else (495 if det == 'mx17_6' else None)


def bow_aS(cal, p, U, W):
    """per event: the tan-form bow offset a(U, W) and the scale S(U, W) of wft.reco.tan_bow (a = 0, S = 1 without)."""
    b = cal.bow or {}
    s = b.get(f's_{p}') or {}
    S = 1.0 + s.get('su', 0.0) * U + s.get('sw', 0.0) * W
    at = b.get(f'a_{p}')
    if at:
        a = (at.get('cu', 0.0) * U + at.get('cw', 0.0) * W + at.get('auu', 0.0) * U * U / 2
             + at.get('aww', 0.0) * W * W / 2 + at.get('auw', 0.0) * U * W)
    else:
        a = np.zeros_like(U)
        if b.get(f'A_{p}'):
            raise NotImplementedError('w-form bow block: not used by pass 1')
    return np.asarray(a, float), np.asarray(S, float)


def tan_map(cal, p, w_um, a, S, r=1.0):
    """wft.reco.tan_bow with v -> v r (r = v_rel(t), scalar or per event)."""
    return (w_um - cal.w0[p]) / (cal.kw[p] * cal.v_drift * r * S) - a


def ladder(cal, w, tp):
    """set_w0kw.py's w0 and kw on (w [um/ns], t' = (tan_ref + a) S); (w0, kw, n_kw, kw_err)."""
    v = cal.v_drift
    ok = np.isfinite(w) & np.isfinite(tp)
    s = ok & (np.abs(tp) < W0_TAN)
    if s.sum() < 30:
        return np.nan, np.nan, 0, np.nan
    w0 = float(np.median(w[s] - v * tp[s]))
    s1 = ok & (np.abs(tp) > KW_LO) & (np.abs(tp) < KW_HI)
    if s1.sum() < 30:
        return w0, np.nan, int(s1.sum()), np.nan
    q = (w[s1] - w0) / (v * tp[s1])
    kw = float(np.median(q))
    err = float(1.2533 * 1.4826 * np.median(np.abs(q - kw)) / np.sqrt(s1.sum()))
    return w0, kw, int(s1.sum()), err


def kw_window(tp):
    return np.isfinite(tp) & (np.abs(tp) > KW_LO) & (np.abs(tp) < KW_HI)


def blocks_equal_count(t, sel, n_per):
    """block edges in t holding ~n_per selected events each (at least one block)."""
    ts = np.sort(t[sel & np.isfinite(t)])
    nb = max(1, int(len(ts) // n_per))
    q = np.quantile(ts, np.linspace(0, 1, nb + 1)) if len(ts) else np.array([np.nan, np.nan])
    q[0], q[-1] = -np.inf, np.inf
    return q


def s68(x):
    x = np.abs(np.asarray(x, float)); x = x[np.isfinite(x)]
    return float(np.percentile(x, 68.27)) if len(x) else np.nan
