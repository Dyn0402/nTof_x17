#!/usr/bin/env python3
"""timeseries.py [--data DIR] -- Paper I: chamber A (det3) Runs 1+2 against wall-clock time, every bench measurement.

Dylan 2026-11-03: "Chamber A Runs 1+2 is the most important run of our paper. Can we do an analysis vs time of all of
our measurements?"  Run 1 = sat_det3 (6-27 22:49 -> 6-28 01:10), Run 2 = p2_det3 (6-28 01:34 -> 08:47), both 490 V
mesh / 1000 V drift.  Every quantity uses the definition, the selection and the estimator of the paper's integrated
number, so the bins sum (or pool) back to it; only the events are split by time.

Clock: the acquisition's stem time (acq_tag, minute) + DREAM timestamp x 10 ns (vt_within_rows/vt_common.clock_h);
rays with no chamber event (no fit at all) take their time from the neighbouring trigger numbers (event ids are one
counter per acquisition, ~5 Hz; spark_timeline.py does the same).  Hours since 6-27 00:00 local CERN time.

Quantities (definition -- source it reproduces):
  v_lad     angle-scale drift velocity: the M3 ladder (vt_common.ladder = rerun/w12/set_w0kw.py) per time block and view,
            v_lad = v_row x kw(block), X and Y inverse-variance averaged.  Anchored to the timing velocity as gas_drying
            does: v = k v_lad with k = 35.8 / v_lad(Run 1) (the row's timing v, physics_inputs.json, over its own ladder
            v).  -- rerun/vt_all/vt_bench.py, rerun/paper1_gas/pg_gas.py attachment_626.
  water     Magboltz inversion of v at E = 1000 V / 28.19 mm (gasinv.water_err, the pg_gas gap) -- gas_drying (b).
  v_col     M3-free cross-check: the column duration T = end30 - edge25 of the strip sum (paper1_timing per-trigger
            npz, M3-matched), v_col = 35.8 x median T(Run 1) / median T(block) -- vt_all crosscheck 'column'.
  eff       single-track efficiency, interior (30 mm x / 8 mm y), R = 5 mm, every ray counts
            (single_track_area_events_*.parquet: cls == 'single', 'credit' = a candidate within 5 mm of the M3 ray in
            both views, as single_track_area.py counts it) -- the paper's 95.7 %; full area too (94.8 %).
  pos core  Gaussian fit to the per-view mesh residual, |res| < 1.5 mm, 0.25 mm bins, Poisson weights, on the
            production fits of radius.py's sample -- residuals figure (0.55 / 0.53 mm).
  s68       68th percentile of |d - median d|, d = theta_fit - theta_ref, effbudget 03_angles interior_good
            (paper1_eff/angles per_event.npz) -- angle_resolution (0.987 / 0.900 deg).
  timing    s68(t_X - t_Y)/sqrt 2 (uncorrelated part) and s68((t_X + t_Y)/2) (event time against the trigger), edge25,
            M3-matched single tracks (x_ok, y_ok, n_tracks == 1), s68 = half the 16-84 % range -- matched_rows.py
            (22.4 / 22.2 ns).  Also the median event time (latency stability).
  q50       median fitted q_X + q_Y of interior M3-matched tracks (r <= 5 mm, both views ok) and the fraction with a strip
            at the ADC rail (n_sat_strips > 0 in either view) -- hv_metrics.py per_track.
Binning: equal-duration bins inside each run (never across the 23 min between the runs), width per quantity (BIN_MIN)
chosen so that the per-bin statistical error is well below the effect looked for (stated in the report).
Errors: binomial (eff, saturation), the ladder's median error (v), fit covariance (pos core), bootstrap (s68, timing, q50).
Trend per quantity: weighted constant (chi2/dof) and a weighted line in time (slope per hour, its significance, and the
line's value at the first and the last bin).  Writes results/timeseries.json and results/bins.csv.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'vt_within_rows'), os.path.join(REPO, 'rerun', 'paper1_gas')]
from vt_common import ladder, bow_aS, stem_time, TICK_S  # noqa: E402
from wft import reco as wr  # noqa: E402

RES = os.path.join(HERE, 'results')
EFFRES = os.path.join(REPO, 'rerun', 'effbudget', 'single', 'results')
RUNS = {'run1': dict(key='sat_det3', tkey='sat_det3',
                     rel='mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
        'run2': dict(key='p2_det3', tkey='c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3',
                     rel='mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3')}
T_REF = dt.datetime(2026, 6, 27, 0, 0)        # hours since 6-27 00:00 local CERN time
V_TIMING_RUN1 = 35.8                          # det3 timing v of Run 1 (rerun/calib/physics_inputs.json v_timing_um_ns)
GAP_MM = 28.19                                # pg_gas.GAP['det3'] (S4 re-check), for E only
E_VCM = 1000.0 / (GAP_MM / 10.0)
#: bin width per quantity [minutes]
BIN_MIN = dict(v=30, eff=30, pos=30, ang=30, tim=20, q=10)
NBOOT = 200
MAX_DROPPED = 2


def clock_h(acq_tag, ts):
    t0 = {t: (stem_time(t) - T_REF).total_seconds() / 3600 for t in pd.unique(acq_tag)}
    return np.array([t0[t] for t in acq_tag]) + np.asarray(ts, float) * TICK_S / 3600


def s68_abs(d):
    """angle_resolution.py: 68th percentile of |d - median|."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return float(np.percentile(np.abs(d - np.median(d)), 68.0)) if len(d) >= 30 else np.nan


def s68_half(a):
    """timing (matched_rows.py): half the 16-84 % range."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 30:
        return np.nan
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def boot(fn, arrays, rng, n=NBOOT):
    m = len(arrays[0])
    vals = [fn(*[a[ii] for a in arrays]) for ii in (rng.integers(0, m, m) for _ in range(n))]
    return float(np.nanstd(vals))


def gauss_core(res):
    """make_figures.residuals: Gaussian fit to |res| < 1.5 mm of the 0.25 mm histogram, Poisson weights."""
    from scipy.optimize import curve_fit
    be = np.linspace(-40, 40, 321); cv = 0.5 * (be[1:] + be[:-1])
    h = np.histogram(res, be)[0].astype(float)
    k = np.abs(cv) < 1.5
    g = lambda x, A, m, s: A * np.exp(-0.5 * ((x - m) / s) ** 2)
    try:
        p, cov = curve_fit(g, cv[k], h[k], p0=[max(h.max(), 1), 0.0, 0.5], sigma=np.sqrt(np.maximum(h[k], 1)))
        return float(abs(p[2])), float(np.sqrt(cov[2, 2]))
    except Exception:
        return np.nan, np.nan


class Run:
    def __init__(self, name, cfg, data):
        self.name, self.key, self.rel = name, cfg['key'], cfg['rel']
        w = os.path.join(data, 'ana', self.rel, 'wft')
        self.meta = json.load(open(os.path.join(w, 'events.meta.json')))
        E = pd.read_parquet(os.path.join(w, 'events.parquet'))
        E['t_h'] = clock_h(E.acq_tag.to_numpy(), E.timestamp.to_numpy())
        self.E = E
        u = E.drop_duplicates('event_id').sort_values('event_id')
        self.eid_sorted, self.t_sorted = u.event_id.to_numpy(float), u.t_h.to_numpy()
        self.t_lo, self.t_hi = float(E.t_h.min()), float(E.t_h.max())
        self.wdir, self.data, self.cfg = w, data, cfg

    def t_of(self, eid):
        """time of any trigger number of the row (exact for fitted events, interpolated for the rest)."""
        return np.interp(np.asarray(eid, float), self.eid_sorted, self.t_sorted)

    def edges(self, minutes):
        n = max(1, int(round((self.t_hi - self.t_lo) * 60 / minutes)))
        return np.linspace(self.t_lo, self.t_hi + 1e-9, n + 1)


def per_bin(runs, q, fn):
    """fn(run, selector over [lo, hi)) -> dict; one row per bin of every run."""
    rows = []
    for R in runs:
        e = R.edges(BIN_MIN[q])
        for lo, hi in zip(e[:-1], e[1:]):
            d = fn(R, lo, hi)
            if d is None:
                continue
            rows.append(dict(q=q, run=R.name, t_lo=lo, t_hi=hi, t=0.5 * (lo + hi), **d))
    return rows


# ---------------------------------------------------------------- drift velocity
def ladder_inputs(R):
    """vt_bench.load_row: the row's calibration and per-view (w, t' = (tan_ref + a) S, t) of 03_angles."""
    from types import SimpleNamespace
    m = R.meta; ac = m['angle_constants']
    bow = json.load(open(os.path.join(R.data, 'ana', RUNS['run1']['rel'], 'wft', 'calib_bundle', 'bundle.json')))['bow']
    assert bow.get('source_row') == m['bow']['source'] == 'sat_det3'
    cal = SimpleNamespace(v_drift=float(m['bundle']['v_drift']), w0=dict(ac['w0']), kw=dict(ac['kw']), bow=bow)
    df = R.E.sort_values('n_tracks').drop_duplicates('event_id', keep='first').set_index('event_id')
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0).fillna(0.0).to_numpy()
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0).fillna(0.0).to_numpy()
    z = np.load(os.path.join(R.wdir, 'angles', 'per_event.npz'))
    views = {}
    for p in 'xy':
        a, S = bow_aS(cal, p, U, W)
        e = z[f'{p}_eid'].astype(int); k = np.isin(e, df.index.to_numpy()); e = e[k]
        pos = df.index.get_indexer(e)
        views[p] = dict(w=z[f'{p}_w'][k] * 1e3, tp=(z[f'{p}_tan_ref'][k] + a[pos]) * S[pos], t=df.t_h.to_numpy()[pos])
    return cal, views


def v_ladder(R, lo, hi, L):
    cal, views = L[R.name]
    o, ws = {}, []
    for p in 'xy':
        V = views[p]; s = (V['t'] >= lo) & (V['t'] < hi)
        w0, kw, n, err = ladder(cal, V['w'][s], V['tp'][s])
        if not np.isfinite(kw):
            return None
        o[f'v_lad_{p}'], o[f'v_lad_{p}_err'], o[f'n_kw_{p}'] = cal.v_drift * kw, cal.v_drift * err, n
        ws.append((cal.v_drift * kw, cal.v_drift * err))
    w = np.array([1 / e ** 2 for _, e in ws])
    o['v_lad'] = float(np.sum(w * np.array([v for v, _ in ws])) / w.sum()); o['v_lad_err'] = float(w.sum() ** -0.5)
    return o


# ---------------------------------------------------------------- timing (and the column-duration v)
def timing_table(R, data):
    z = np.load(os.path.join(data, 'timing_rows', R.cfg['tkey'] + '.npz'))
    tags = list(z['acq_tags'])
    T = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], event_id=z['eid'], ts=z['ts'], tx=z['tx'], ty=z['ty'],
                          ex=z['ex'], ey=z['ey'], amp=z['amp']))
    E = R.E[['acq_tag', 'event_id', 'x_ok', 'y_ok', 'n_tracks']]
    m = T.merge(E, on=['acq_tag', 'event_id'])
    m = m[m.x_ok & m.y_ok & (m.n_tracks == 1)].copy()          # matched_rows.py
    m['t_h'] = clock_h(m.acq_tag.to_numpy(), m.ts.to_numpy())
    m['Tcol'] = 0.5 * ((m.ex - m.tx) + (m.ey - m.ty))
    return m


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'),
                                                   'claude-jobs', '1ccd2342', 'timeseries'))
    ap.add_argument('--eff-events', default=EFFRES, help='dir with single_track_area_events_{sat_det3,p2_det3}.parquet '
                    '(default: the committed ones)')
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(7)
    runs = [Run(n, c, a.data) for n, c in RUNS.items()]
    out = dict(clock='hours since 2026-06-27 00:00 local CERN time', bin_minutes=BIN_MIN,
               runs={R.name: dict(key=R.key, t_lo=R.t_lo, t_hi=R.t_hi, n_events=int(R.E.event_id.nunique()),
                                  v_drift_row=float(R.meta['bundle']['v_drift']), kw=R.meta['angle_constants']['kw'])
                     for R in runs})
    rows = []

    # --- v: ladder (anchored) + column duration
    L = {R.name: ladder_inputs(R) for R in runs}
    whole = {R.name: v_ladder(R, -1e9, 1e9, L) for R in runs}
    k = V_TIMING_RUN1 / whole['run1']['v_lad']
    out['v'] = dict(k_anchor=k, v_lad_whole={n: w['v_lad'] for n, w in whole.items()},
                    v_lad_whole_err={n: w['v_lad_err'] for n, w in whole.items()},
                    anchored_whole={n: k * w['v_lad'] for n, w in whole.items()}, E_Vcm=E_VCM,
                    note='k = 35.8 / v_lad(Run 1); v = k v_lad (pg_gas attachment_626 convention)')
    import gasinv as GI
    vr = per_bin(runs, 'v', lambda R, lo, hi: v_ladder(R, lo, hi, L))
    for r in vr:
        r['v'], r['v_err'] = k * r['v_lad'], k * r['v_lad_err']
        w = GI.water_err(r['v'], r['v_err'], E_VCM)
        if w:
            r['water'], r['water_lo'], r['water_hi'] = w['water'], w['lo'], w['hi']
            r['water_branches'] = len(w['branches'])
    rows += vr
    TT = {R.name: timing_table(R, a.data) for R in runs}
    T1 = float(np.nanmedian(TT['run1'].Tcol))

    def vcol(R, lo, hi):
        m = TT[R.name]; s = (m.t_h >= lo) & (m.t_h < hi)
        T = m.Tcol[s].to_numpy(float); T = T[np.isfinite(T)]
        if len(T) < 100:
            return None
        med = float(np.median(T)); e = boot(np.median, [T], rng, 100)
        return dict(T_col=med, T_col_err=e, v_col=V_TIMING_RUN1 * T1 / med, v_col_err=V_TIMING_RUN1 * T1 * e / med ** 2,
                    n=int(len(T)))
    BIN_MIN['vcol'] = BIN_MIN['v']
    rows += [dict(r, q='vcol') for r in per_bin(runs, 'v', vcol)]

    # --- efficiency (single-track; interior and full area)
    def eff(R, lo, hi):
        S = R.eff; s = (S.t >= lo) & (S.t < hi)
        o = {}
        for lab, sel in (('int', S.region == 'interior'), ('all', np.ones(len(S), bool))):
            x = S[s & sel & (S.cls == 'single')]
            n, kk = len(x), int(x.credit.sum())
            if n < 50:
                return None
            p = kk / n
            o.update({f'eff_{lab}': 100 * p, f'eff_{lab}_err': 100 * np.sqrt(p * (1 - p) / n), f'n_{lab}': n, f'k_{lab}': kk})
        return o
    for R in runs:
        S = pd.read_parquet(os.path.join(a.eff_events, f'single_track_area_events_{R.key}.parquet'))
        S['t'] = R.t_of(S.index.to_numpy())
        R.eff = S
    rows += per_bin(runs, 'eff', eff)

    # --- position core and gain proxy (rays of the budget)
    for R in runs:
        B = pd.read_parquet(os.path.join(a.data, 'budget', R.rel, 'rays_det3.parquet')).drop_duplicates('event_id')
        B = B[(~B.not_measured) & (B.region == 'interior') & B.good_x].reset_index(drop=True)
        B['t'] = R.t_of(B.event_id.to_numpy())
        B['ok'] = (B.x_ok & B.y_ok & (B.x_n_dropped.fillna(99) <= MAX_DROPPED) & (B.y_n_dropped.fillna(99) <= MAX_DROPPED))
        R.B = B
        M = R.E.merge(B.loc[B.r <= 5, ['event_id', 't']], on='event_id')       # hv_metrics.per_track
        M = M[(M.x_ok == True) & (M.y_ok == True)].copy()  # noqa: E712
        M['q'] = M.x_q_sum + M.y_q_sum
        M['sat'] = (M.x_n_sat_strips > 0) | (M.y_n_sat_strips > 0)
        R.M = M

    def pos(R, lo, hi):
        B = R.B; s = (B.t >= lo) & (B.t < hi) & B.ok
        if s.sum() < 200:
            return None
        o = dict(n=int(s.sum()))
        for p in 'xy':
            r = B[f'res_{p}'][s].to_numpy()
            o[f'pos_{p}'] = gauss_core(r)[0]
            o[f'pos_{p}_err'] = boot(lambda x: gauss_core(x)[0], [r], rng, 100)
        return o
    rows += per_bin(runs, 'pos', pos)

    def gain(R, lo, hi):
        M = R.M; s = (M.t >= lo) & (M.t < hi)
        if s.sum() < 50:
            return None
        q = M.q[s].to_numpy(float); f = float(M.sat[s].mean())
        return dict(q50=float(np.median(q)), q50_err=boot(np.median, [q], rng, 100), f_sat=100 * f,
                    f_sat_err=100 * np.sqrt(f * (1 - f) / s.sum()), n=int(s.sum()))
    rows += per_bin(runs, 'q', gain)

    # --- angle s68
    for R in runs:
        z = np.load(os.path.join(a.data, 'angles', R.rel, 'per_event.npz'))
        R.ang = {p: pd.DataFrame(dict(t=R.t_of(z[f'{p}_eid']), d=np.degrees(np.arctan(z[f'{p}_tan_fit']))
                                      - np.degrees(np.arctan(z[f'{p}_tan_ref'])))) for p in 'xy'}

    def ang(R, lo, hi):
        o = {}
        for p in 'xy':
            A = R.ang[p]; d = A.d[(A.t >= lo) & (A.t < hi)].to_numpy()
            if len(d) < 100:
                return None
            o[f's68_{p}'], o[f's68_{p}_err'], o[f'n_{p}'] = s68_abs(d), boot(s68_abs, [d], rng), len(d)
            o[f'med_{p}'] = float(np.median(d))
        return o
    rows += per_bin(runs, 'ang', ang)

    # --- timing
    def tim(R, lo, hi):
        m = TT[R.name]; s = (m.t_h >= lo) & (m.t_h < hi)
        if s.sum() < 100:
            return None
        tx, ty = m.tx[s].to_numpy(float), m.ty[s].to_numpy(float)
        fxy = lambda x, y: s68_half(x - y) / np.sqrt(2)
        fev = lambda x, y: s68_half((x + y) / 2)
        return dict(sigma_xy=fxy(tx, ty), sigma_xy_err=boot(fxy, [tx, ty], rng), sigma_evt=fev(tx, ty),
                    sigma_evt_err=boot(fev, [tx, ty], rng), t_evt_med=float(np.nanmedian((tx + ty) / 2)),
                    t_evt_med_err=boot(lambda x, y: np.nanmedian((x + y) / 2), [tx, ty], rng, 100), n=int(s.sum()))
    rows += per_bin(runs, 'tim', tim)

    # --- integrated values with these same codes (must reproduce the paper's numbers)
    allE = pd.concat([R.eff for R in runs]); allB = pd.concat([R.B for R in runs]); allM = pd.concat([R.M for R in runs])
    allT = pd.concat(TT.values())
    integ = {}
    for lab, sel in (('int', allE.region == 'interior'), ('all', np.ones(len(allE), bool))):
        x = allE[sel & (allE.cls == 'single')]
        integ[f'eff_{lab}'] = 100 * float(x.credit.mean()); integ[f'n_{lab}'] = int(len(x))
    for p in 'xy':
        integ[f'pos_{p}'] = gauss_core(allB[f'res_{p}'][allB.ok].to_numpy())[0]
        integ[f's68_{p}'] = s68_abs(np.concatenate([R.ang[p].d.to_numpy() for R in runs]))
    integ['sigma_xy'] = s68_half(allT.tx - allT.ty) / np.sqrt(2)
    integ['sigma_evt'] = s68_half(np.concatenate([((m.tx + m.ty) / 2 - np.nanmedian((m.tx + m.ty) / 2)).to_numpy() for m in TT.values()]))
    integ['sigma_xy_by_run'] = {n: s68_half(m.tx - m.ty) / np.sqrt(2) for n, m in TT.items()}
    integ['q50'] = float(allM.q.median()); integ['f_sat'] = 100 * float(allM.sat.mean())
    integ['q50_by_run'] = {R.name: float(R.M.q.median()) for R in runs}
    integ['f_sat_by_run'] = {R.name: 100 * float(R.M.sat.mean()) for R in runs}
    integ['T_col_by_run'] = {n: float(np.nanmedian(m.Tcol)) for n, m in TT.items()}
    out['integrated'] = integ

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'bins.csv'), index=False)
    out['trends'] = trends(df)
    json.dump(out, open(os.path.join(RES, 'timeseries.json'), 'w'), indent=1, default=float)
    for kq, v in out['trends'].items():
        print(f"{kq:12s} first {v['first']:.4g} last {v['last']:.4g}  line {v['line_start']:.4g} -> {v['line_end']:.4g}"
              f"  slope {v['slope_per_h']:+.3g} +- {v['slope_err']:.2g} /h ({v['z']:+.1f} sig)  const chi2 {v['chi2_const']:.1f}/{v['dof_const']}"
              f"  run2-run1 {v['run2_minus_run1']:+.3g} +- {v['run2_minus_run1_err']:.2g}")
    print(json.dumps(integ, indent=1, default=float))


SERIES = {'v': ('v', 'v_err'), 'vcol': ('v_col', 'v_col_err'), 'water': ('water', None),
          'eff_int': ('eff_int', 'eff_int_err'), 'eff_all': ('eff_all', 'eff_all_err'),
          'pos_x': ('pos_x', 'pos_x_err'), 'pos_y': ('pos_y', 'pos_y_err'),
          's68_x': ('s68_x', 's68_x_err'), 's68_y': ('s68_y', 's68_y_err'),
          'sigma_xy': ('sigma_xy', 'sigma_xy_err'), 'sigma_evt': ('sigma_evt', 'sigma_evt_err'),
          't_evt_med': ('t_evt_med', 't_evt_med_err'), 'q50': ('q50', 'q50_err'), 'f_sat': ('f_sat', 'f_sat_err')}
QOF = {'v': 'v', 'vcol': 'vcol', 'water': 'v', 'eff_int': 'eff', 'eff_all': 'eff', 'pos_x': 'pos', 'pos_y': 'pos',
       's68_x': 'ang', 's68_y': 'ang', 'sigma_xy': 'tim', 'sigma_evt': 'tim', 't_evt_med': 'tim', 'q50': 'q', 'f_sat': 'q'}


def trends(df):
    out = {}
    for name, (c, ce) in SERIES.items():
        d = df[(df.q == QOF[name])]
        if c not in d or d[c].isna().all():
            continue
        d = d[d[c].notna()]
        t, y = d.t.to_numpy(float), d[c].to_numpy(float)
        if ce is None:
            e = 0.5 * (d.water_hi - d.water_lo).abs().to_numpy(float)
        else:
            e = d[ce].to_numpy(float)
        e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e))
        w = 1 / e ** 2
        c0 = float(np.sum(w * y) / w.sum()); chi0 = float(np.sum(w * (y - c0) ** 2))
        A = np.vstack([t - t.mean(), np.ones_like(t)]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None])); p = cov @ (A.T @ (w * y))
        chi1 = float(np.sum(w * (y - A @ p) ** 2)); dof1 = len(y) - 2
        infl = max(1.0, np.sqrt(chi1 / dof1)) if dof1 > 0 else 1.0     # excess scatter inflates the slope error
        se = float(np.sqrt(cov[0, 0]) * infl)
        r1, r2 = d.run == 'run1', d.run == 'run2'
        m1 = float(np.sum(w[r1] * y[r1]) / w[r1].sum()); m2 = float(np.sum(w[r2] * y[r2]) / w[r2].sum())
        e12 = float(np.sqrt(1 / w[r1].sum() + 1 / w[r2].sum()))
        per_run = {}
        for rn, rk in (('run1', r1), ('run2', r2)):
            tt, yy, ww = t[rk], y[rk], w[rk]
            if len(tt) >= 3:
                B_ = np.vstack([tt - tt.mean(), np.ones_like(tt)]).T
                cv = np.linalg.inv(B_.T @ (B_ * ww[:, None])); pp = cv @ (B_.T @ (ww * yy))
                ch = float(np.sum(ww * (yy - B_ @ pp) ** 2)); inf_ = max(1.0, np.sqrt(ch / max(len(tt) - 2, 1)))
                per_run[rn] = dict(slope_per_h=float(pp[0]), slope_err=float(np.sqrt(cv[0, 0]) * inf_),
                                   z=float(pp[0] / (np.sqrt(cv[0, 0]) * inf_)), duration_h=float(tt[-1] - tt[0]),
                                   change=float(pp[0] * (tt[-1] - tt[0])), chi2_line=ch, dof=len(tt) - 2,
                                   line_start=float(pp[1] + pp[0] * (tt[0] - tt.mean())), line_end=float(pp[1] + pp[0] * (tt[-1] - tt.mean())),
                                   mean=float(np.sum(ww * yy) / ww.sum()))
        out[name] = dict(n_bins=int(len(y)), first=float(y[0]), first_err=float(e[0]), last=float(y[-1]),
                         last_err=float(e[-1]), mean=c0, chi2_const=chi0, dof_const=int(len(y) - 1),
                         slope_per_h=float(p[0]), slope_err=se, z=float(p[0] / se), chi2_line=chi1, dof_line=dof1,
                         line_start=float(p[1] + p[0] * (t[0] - t.mean())), line_end=float(p[1] + p[0] * (t[-1] - t.mean())),
                         t_first=float(t[0]), t_last=float(t[-1]), run1_mean=m1, run2_mean=m2,
                         run2_minus_run1=m2 - m1, run2_minus_run1_err=e12, per_run=per_run, rms=float(np.std(y)),
                         median_err=float(np.median(e)))
    return out


if __name__ == '__main__':
    main()
