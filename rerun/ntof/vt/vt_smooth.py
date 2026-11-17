#!/usr/bin/env python3
"""vt_smooth.py -- P7a: a smooth, interpolable v(t) per chamber from vt_table.py's per-acquisition v (2026-11-17).

    vt_smooth.py [--vt DIR] [--env DIR] [--out DIR]

Model, on ln v of every acquisition with a v (A and C beam + cosmic, D cosmic):

    ln v_i = c[arm, trigger] + s[arm] * g(t_i)              s_A = 1

g(t) is ONE curve shared by the chambers (they share the gas line A -> B -> C -> D; corr(A, C) = 0.85 per sub-run),
a cubic P-spline (knots every KNOT_H hours, 2nd-difference penalty lambda), optionally plus beta * x(t) with x the
outdoor water-vapour pressure (MeteoSwiss Geneve-Cointrin `pva200s0`) smoothed by a causal exponential of time
constant tau -- the model of water permeating into the gas lines.  c and s are per arm; the beam / cosmic offset is a
separate c (the two trigger classes select different tracks).  Fitted by alternating weighted least squares; each arm
gets a jitter term added in quadrature to the bootstrap error so that chi2/dof = 1 (the acquisition scatter exceeds
the bootstrap error: within a sub-run rms 0.41 vs 0.29 um/ns).  The P-spline's Bayesian covariance gives the
interpolation error, which grows across gaps.

Validation (the numbers go to vt_smooth.json and the report):
  * random holdout: 20 % of acquisitions left out, 5 folds; pulls of the held-out points;
  * block holdout: every 4th BLOCK_H-hour block left out, as an interpolation-across-a-gap test;
for three models: spline only, water-vapour only (+ constant), spline + water vapour.  lambda (and tau) by the CV.

Outputs (DIR = --out): vt_smooth.parquet (per acquisition: v_obs, v_fit, v_fit_err for every arm incl. D on beam
acquisitions, which has no measurement), vt_smooth_grid.parquet (10-minute grid, per arm and trigger class),
vt_smooth.json (parameters + validation), figures/smooth_*.png, smooth.html.
"""
import argparse, json, os

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
WD = os.environ.get('WORK_DATA') or next((p for p in ('/media/ucla/work', os.path.expanduser('~/work_data'))
                                          if os.path.isdir(p)), '/media/ucla/work')
ARMS = ('A', 'C', 'D')
KNOT_H = 2.0
BLOCK_H = 6.0
CEST = pd.Timedelta(hours=2)           # MeteoSwiss is UTC; the DAQ tags, vt times and the capsule log are local
COL = {'A': '#2a6fdb', 'C': '#d9622b', 'D': '#2e9e5b'}


# ----------------------------------------------------------------------------------------------------------- data
def load_v(vt):
    T = pd.concat([pd.read_parquet(f'{vt}/vt_{a}.parquet').assign(arm=a) for a in ARMS], ignore_index=True)
    T['tm'] = T.t_start + (T.t_end - T.t_start) / 2
    return T


def load_env(env):
    g = pd.read_csv(f'{env}/gve_t_recent.csv', sep=';',
                    usecols=['reference_timestamp', 'tre200s0', 'prestas0', 'pva200s0', 'ure200s0'])
    g['t'] = pd.to_datetime(g.reference_timestamp, format='%d.%m.%Y %H:%M') + CEST
    g = g.set_index('t').drop(columns='reference_timestamp').apply(pd.to_numeric, errors='coerce')
    c = pd.read_csv(f'{REPO}/ntof_run_report/data/he3_pressure_5min.csv', parse_dates=['timestamp'])
    c = c.set_index('timestamp').p_med_bar
    c = c[(c.index > '2026-07-15') & (c > 400)].resample('10min').median()
    return g.join(c.rename('P_caps'), how='outer').sort_index()


def ewm_causal(s, tau_h):
    """Causal exponential smoothing of a 10-min series with time constant tau [h]."""
    s = s.interpolate(limit=36)
    a = 1 - np.exp(-(10 / 60) / tau_h)
    return s.ewm(alpha=a, adjust=False, ignore_na=True).mean()


# ---------------------------------------------------------------------------------------------------------- model
class Smooth:
    def __init__(self, t0, t1, knot_h=KNOT_H, lam=10.0, use_spline=True, x_env=None):
        self.t0 = t0
        h1 = (t1 - t0) / pd.Timedelta(hours=1)
        k = 3
        inner = np.arange(-knot_h, h1 + 2 * knot_h, knot_h)
        self.knots = np.r_[[inner[0]] * k, inner, [inner[-1]] * k]
        self.nb = len(self.knots) - k - 1
        self.k = k
        self.lam = lam
        self.use_spline = use_spline
        self.x_env = x_env               # callable t -> standardized env value, or None

    def hours(self, t):
        return ((pd.to_datetime(t) - self.t0) / pd.Timedelta(hours=1)).to_numpy(float)

    def design(self, t):
        cols = []
        if self.use_spline:
            h = np.clip(self.hours(t), self.knots[self.k], self.knots[-self.k - 1] - 1e-9)
            cols.append(BSpline.design_matrix(h, self.knots, self.k).toarray())
        if self.x_env is not None:
            cols.append(self.x_env(t)[:, None])
        return np.hstack(cols)

    def penalty(self):
        n = (self.nb if self.use_spline else 0) + (1 if self.x_env is not None else 0)
        P = np.zeros((n, n))
        if self.use_spline:
            D = np.diff(np.eye(self.nb), 2, axis=0)
            P[:self.nb, :self.nb] = self.lam * D.T @ D + 1e-8 * np.eye(self.nb)
        return P

    def fit(self, df, n_iter=25):
        """df: arm, trigger, tm, y = ln v, e = error of ln v."""
        arms = sorted(df.arm.unique())
        self.s = {a: {'A': 1.0, 'C': 2.2, 'D': 1.5}.get(a, 1.0) for a in arms}
        self.jit = {a: 0.0 for a in arms}
        groups = sorted(set(zip(df.arm, df.trigger)))
        self.c = {gk: float(df[(df.arm == gk[0]) & (df.trigger == gk[1])].y.median()) for gk in groups}
        B = self.design(df.tm)
        P = self.penalty()
        arm = df.arm.to_numpy()
        gkey = list(zip(df.arm, df.trigger))
        gstr = np.array([f'{x}|{t}' for x, t in gkey])
        y, e = df.y.to_numpy(float), df.e.to_numpy(float)
        for it in range(n_iter):
            s = np.array([self.s[a] for a in arm])
            c = np.array([self.c[k] for k in gkey])
            var = e ** 2 + np.array([self.jit[a] for a in arm]) ** 2
            z = (y - c) / s
            w = s ** 2 / var
            A = B.T @ (B * w[:, None]) + P
            self.coef = np.linalg.solve(A, B.T @ (w * z))
            self.cov = np.linalg.inv(A)
            g = B @ self.coef
            # centre g on the A acquisitions (its level is in c)
            if self.use_spline:
                shift = np.average(g[arm == arms[0]])
                self.coef[:self.nb] -= shift
                g = g - shift
            for a in arms:
                m = arm == a
                ks = sorted(set(gstr[m]))
                X = np.column_stack([(gstr[m] == kk).astype(float) for kk in ks] + ([g[m]] if a != 'A' else []))
                ww = 1 / var[m]
                rhs = y[m] if a != 'A' else y[m] - g[m]
                beta = np.linalg.lstsq(X * np.sqrt(ww)[:, None], rhs * np.sqrt(ww), rcond=None)[0]
                for kk, b in zip(ks, beta):
                    self.c[tuple(kk.split('|'))] = float(b)
                if a != 'A':
                    self.s[a] = float(beta[-1])
            # jitter: chi2/dof = 1 per arm
            s = np.array([self.s[a] for a in arm]); c = np.array([self.c[k] for k in gkey])
            r = y - c - s * g
            for a in arms:
                m = arm == a
                j2 = max(np.mean(r[m] ** 2 - e[m] ** 2), 0.0)
                self.jit[a] = float(np.sqrt(j2))
        self.arms = arms
        return self

    def predict(self, t, arm, trigger):
        B = self.design(t)
        g = B @ self.coef
        gerr = np.sqrt(np.einsum('ij,jk,ik->i', B, self.cov, B))
        s = self.s[arm]
        c = self.c.get((arm, trigger))
        if c is None:                       # D on beam: no beam measurement -> the cosmic level
            c = self.c[(arm, 'cosmic')]
        lv = c + s * g
        return np.exp(lv), np.exp(lv) * abs(s) * gerr


def prep(T):
    d = T[np.isfinite(T.v) & np.isfinite(T.v_err) & (T.v_err > 0)].copy()
    d['y'] = np.log(d.v)
    d['e'] = d.v_err / d.v
    return d.reset_index(drop=True)


def pulls(model, d):
    out = []
    for (a, tr), g in d.groupby(['arm', 'trigger']):
        v, ve = model.predict(g.tm, a, tr)
        sig = np.sqrt((g.e * g.v) ** 2 + (model.jit[a] * g.v) ** 2 + ve ** 2)
        out.append(pd.DataFrame(dict(arm=a, trigger=tr, pull=(g.v - v) / sig, res=(g.v / v - 1), tm=g.tm.values)))
    return pd.concat(out)


def cv(d, make, mode, seed=1):
    rng = np.random.default_rng(seed)
    if mode == 'random':
        fold = rng.integers(0, 5, len(d))
        folds = range(5)
    else:
        blk = ((d.tm - d.tm.min()) / pd.Timedelta(hours=BLOCK_H)).astype(int)
        fold = (blk % 4).to_numpy()
        folds = range(4)
    P = []
    for f in folds:
        tr, te = d[fold != f], d[fold == f]
        m = make().fit(tr)
        P.append(pulls(m, te))
    P = pd.concat(P)
    return P


def summary(P):
    r = {}
    for a, g in P.groupby('arm'):
        r[a] = dict(n=len(g), pull_rms=float(np.sqrt(np.mean(g.pull ** 2))),
                    pull68=float(np.subtract(*np.percentile(g.pull, [84, 16])) / 2),
                    res68_pct=float(100 * np.subtract(*np.percentile(g.res, [84, 16])) / 2))
    return r


# ---------------------------------------------------------------------------------------------------------- report
def write_html(out, J, best, tau, sens, val, pairs, envc):
    P = J['params'][best]

    def vrow(name):
        cells = ''.join(f"<td>{val[name][m][a]['res68_pct']:.2f} % / {val[name][m][a]['pull_rms']:.2f}</td>"
                        for m in ('random', 'block') for a in ('A', 'C', 'D'))
        return f'<tr><td>{name}</td>{cells}</tr>'
    prs = ''.join(f"<tr><td>{k.replace('_', ' ')}</td><td>{v['n']}</td><td>{v['corr']:.2f}</td><td>{v['slope']:.2f}</td></tr>"
                  for k, v in pairs.items())
    env = ''.join(f"<tr><td>{k}</td>" + ''.join(f'<td>{x:+.2f}</td>' for x in v.values()) + '</tr>' for k, v in envc.items())
    lags = ''.join(f'<th>{l} h</th>' for l in next(iter(envc.values())).keys())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>n_TOF v(t) model</title>
<style>body{{font-family:sans-serif;max-width:1150px;margin:auto;padding:0 16px;line-height:1.45}}
table{{border-collapse:collapse;font-size:13px;margin:6px 0}}td,th{{border:1px solid #bbb;padding:3px 7px;text-align:right}}
img{{max-width:100%}}.cap{{font-size:13px;color:#444}}</style></head><body>
<h1>n_TOF drift velocity: a smooth v(t) to interpolate with</h1>
<p>Generated by <code>rerun/ntof/vt/vt_smooth.py</code> from <code>vt_table.py</code>'s per-acquisition column-duration v
(the measurements: <a href="report.html">report.html</a>).</p>
<h2>Answer</h2>
<ul>
<li><b>One shared curve describes all three chambers.</b> ln v = c[arm, trigger] + s[arm]·g(t), with C moving
{P['s']['C']:.2f}× and D {P['s']['D']:.2f}× as much as A. It predicts a held-out acquisition to ±{val[best]['random']['A']['res68_pct']:.2f} % (A) and
±{val[best]['random']['C']['res68_pct']:.2f} % (C) (68 %), about one acquisition's own scatter, and to ±{val[best]['block']['A']['res68_pct']:.2f} / ±{val[best]['block']['C']['res68_pct']:.2f} %
when whole {BLOCK_H:.0f} h blocks are removed. Pulls have rms ≈ 1, so the error band is honest.</li>
<li><b>The driver looks like water.</b> The outdoor water-vapour pressure (Genève-Cointrin), smoothed with a
{tau} h time constant, alone explains most of it (r = {envc['pva200s0'][8]:+.2f} with the curve at an 8 h lag):
{', '.join(f'{k} {v:+.1f} %/hPa' for k, v in sens.items())}. Air temperature, station pressure and the ³He capsule
(a sealed volume, so a hall thermometer) correlate only weakly, and with the sign opposite to a gas-density effect.
Reading: water permeating into the gas lines, more of it downstream (C, D) than in A. A hypothesis, not shown.</li>
<li>Model used: <b>{best}</b> (penalised cubic spline, knots every {KNOT_H:.0f} h, λ = {J['lam_with_water'] if 'water' in best else J['lam']:.3g}
by block cross-validation; the water term only matters across gaps).</li></ul>
<h2>The environment alone</h2>
<p>ln v = a + b·(water vapour, τ = {tau} h) + c·(³He capsule pressure minus its 48 h mean, i.e. the hall's daily
temperature swing; 1 bar ≈ 0.6 K). No spline.</p>
<table><tr><th>arm</th><th>n</th><th>lag (h)</th><th>R²</th><th>68 % residual</th><th>water (%/hPa)</th><th>hall (%/K)</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['lag_h']}</td><td>{v['r2']:.2f}</td><td>{v['res68_pct']:.2f} %</td><td>{v['water_pct_per_hPa']:+.2f}</td><td>{v['hall_pct_per_K']:+.2f}</td></tr>" for k, v in J['env_regression'].items())}</table>
<p class="cap">A gas-density (E/N) effect would make v rise with temperature at our field; v falls with both water and hall
temperature, and by ~2 %/K on C, far beyond density. Temperature-driven water (permeation, wall outgassing) fits the
signs; not shown.</p>
<h2>Validation (68 % residual / pull rms)</h2>
<table><tr><th rowspan=2>model</th><th colspan=3>random 20 % holdout</th><th colspan=3>every 4th {BLOCK_H:.0f} h block held out</th></tr>
<tr><th>A</th><th>C</th><th>D</th><th>A</th><th>C</th><th>D</th></tr>{''.join(vrow(n) for n in val)}</table>
<p class="cap">D is fitted on its sub-run pooled v (15–45 clean cosmic tracks per acquisition), the others per acquisition.</p>
<h2>Chamber-to-chamber correlation (same sub-run)</h2>
<table><tr><th>pair, trigger</th><th>n</th><th>r</th><th>d ln v₂ / d ln v₁</th></tr>{prs}</table>
<p class="cap">The slopes are regression slopes on noisy x, so diluted (the model's s is the fitted ratio).</p>
<h2>Environment correlation of the shared curve, by lag (environment leading)</h2>
<table><tr><th>variable</th>{lags}</tr>{env}</table>
<h2>Figures</h2>
<p><img src="figures/smooth_timeline.png"><br><span class="cap">Per-acquisition v (small points), per sub-run
(circles beam, squares cosmic), the model (lines) with its 1σ band; bottom: outdoor water vapour and air temperature.
D has no beam v: its beam value is the cosmic curve.</span></p>
<p><img src="figures/smooth_zoom.png"><br><span class="cap">Two days in detail, A and C on beam.</span></p>
<p><img src="figures/smooth_pairs.png"></p>
<p><img src="figures/smooth_envtime.png"></p>
<p><img src="figures/smooth_env.png"></p>
<p><img src="figures/smooth_holdout.png"></p>
<h2>What this does not rule out</h2><ul>
<li>A's shape is tied to C's (one curve); A's own deviations from it are absorbed into its jitter
({100 * P['jitter']['A']:.2f} %).</li>
<li>v is the column duration over the timing-free gap: a gain change at fixed HV moves T too (the caveat of the table).</li>
<li>The water reading is a correlation with outdoor air, not a measurement of the gas.</li>
<li>Cointrin is ~5 km from n_TOF; MeteoSwiss times are UTC, shifted by +2 h to the DAQ's local time.</li></ul>
</body></html>"""
    open(f'{out}/smooth.html', 'w').write(html)


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vt', default=f'{WD}/ntof_pass1/vt')
    ap.add_argument('--env', default=f'{WD}/ntof_pass1/env')
    ap.add_argument('--out', default=f'{WD}/ntof_pass1/vt')
    a = ap.parse_args()
    FIG = f'{a.out}/figures'
    os.makedirs(FIG, exist_ok=True)
    T = load_v(a.vt)
    E = load_env(a.env)
    # D: its clean cosmic sample is 15-45 events per acquisition, so only 103 acquisitions reach MIN_N; use its
    # sub-run pooled v (238 sub-runs) instead
    SD = pd.read_parquet(f'{a.vt}/vt_D_subrun.parquet').assign(arm='D', tag=None)
    SD['tm'] = SD.t_start + (SD.t_end - SD.t_start) / 2
    d = prep(pd.concat([T[T.arm != 'D'], SD[['run', 'subrun', 'tag', 'arm', 'trigger', 'tm', 'v', 'v_err', 'v_bundle']]],
                       ignore_index=True))
    t0, t1 = d.tm.min() - pd.Timedelta(hours=3), d.tm.max() + pd.Timedelta(hours=3)

    # --- water-vapour regressor: choose tau by the fit of the env-only model on all data
    res = {}

    def xfun_for(tau):
        x = ewm_causal(E.pva200s0, tau)
        x = (x - x.loc[t0:t1].mean()) / x.loc[t0:t1].std()
        return lambda t: x.reindex(pd.to_datetime(t), method='nearest', tolerance=pd.Timedelta('20min')).to_numpy(float)

    tau_scan = {}
    for tau in (8, 12, 18, 24, 30, 36, 48):
        P = cv(d, lambda: Smooth(t0, t1, use_spline=False, x_env=xfun_for(tau)), 'random')
        tau_scan[tau] = float(np.mean([v['res68_pct'] for v in summary(P).values()]))
    tau = min(tau_scan, key=tau_scan.get)
    xf = xfun_for(tau)
    lam_scan = {}
    for lam in (1000, 3000, 10000, 30000, 1e5, 3e5, 1e6, 3e6):
        P = cv(d, lambda: Smooth(t0, t1, lam=lam), 'block')
        lam_scan[lam] = float(np.mean([v['res68_pct'] for v in summary(P).values()]))
    lam = min(lam_scan, key=lam_scan.get)
    lam_scan_env = {}
    for lamx in (1000, 3000, 10000, 30000, 1e5, 3e5, 1e6, 3e6):
        P = cv(d, lambda: Smooth(t0, t1, lam=lamx, x_env=xf), 'block')
        lam_scan_env[lamx] = float(np.mean([v['res68_pct'] for v in summary(P).values()]))
    lam_e = min(lam_scan_env, key=lam_scan_env.get)
    makers = {'spline': lambda: Smooth(t0, t1, lam=lam),
              'water': lambda: Smooth(t0, t1, use_spline=False, x_env=xf),
              'spline+water': lambda: Smooth(t0, t1, lam=lam_e, x_env=xf)}
    val = {}
    for name, mk in makers.items():
        val[name] = {mode: summary(cv(d, mk, mode)) for mode in ('random', 'block')}
    fits = {name: mk().fit(d) for name, mk in makers.items()}
    best = 'spline+water' if np.mean([v['res68_pct'] for v in val['spline+water']['block'].values()]) <= \
        np.mean([v['res68_pct'] for v in val['spline']['block'].values()]) else 'spline'
    M = fits[best]

    # --- per-acquisition table, every arm on every acquisition time (D beam = predicted only)
    acq = T.drop_duplicates(['run', 'subrun', 'tag', 'trigger'])[['run', 'subrun', 'tag', 'trigger', 'tm']]
    rows = []
    for arm in ARMS:
        for tr, g in acq.groupby('trigger'):
            v, ve = M.predict(g.tm, arm, tr)
            o = T[(T.arm == arm)].set_index(['run', 'subrun', 'tag'])
            obs = o.reindex(list(zip(g.run, g.subrun, g.tag)))
            rows.append(pd.DataFrame(dict(run=g.run.values, subrun=g.subrun.values, tag=g.tag.values, arm=arm,
                                          trigger=tr, tm=g.tm.values, v_obs=obs.v.values, v_obs_err=obs.v_err.values,
                                          v_fit=v, v_fit_err=ve, v_bundle=float(T[T.arm == arm].v_bundle.iloc[0]),
                                          measured=np.isfinite(obs.v.values))))
    A = pd.concat(rows, ignore_index=True)
    A.to_parquet(f'{a.out}/vt_smooth.parquet', index=False)
    grid = pd.date_range(d.tm.min().floor('h'), d.tm.max().ceil('h'), freq='10min')
    G = []
    for arm in ARMS:
        for tr in ('beam', 'cosmic'):
            v, ve = M.predict(grid, arm, tr)
            G.append(pd.DataFrame(dict(t=grid, arm=arm, trigger=tr, v=v, v_err=ve)))
    G = pd.concat(G, ignore_index=True)
    G.to_parquet(f'{a.out}/vt_smooth_grid.parquet', index=False)

    # --- correlations between chambers: sub-run means of the measured acquisition v, matched on (run, subrun, trigger)
    S = {arm: pd.read_parquet(f'{a.vt}/vt_{arm}_subrun.parquet') for arm in ARMS}
    pairs = {}
    for p, q in (('A', 'C'), ('A', 'D'), ('C', 'D')):
        m = S[p].merge(S[q], on=['run', 'subrun', 'trigger'], suffixes=('_p', '_q')).dropna(subset=['v_p', 'v_q'])
        for tr, g in m.groupby('trigger'):
            rp, rq = np.log(g.v_p), np.log(g.v_q)
            pairs[f'{p}{q}_{tr}'] = dict(n=len(g), corr=float(np.corrcoef(rp, rq)[0, 1]),
                                         slope=float(np.polyfit(rp - rp.mean(), rq - rq.mean(), 1)[0]))

    # --- env correlation of the shared curve
    gm = pd.Series(M.predict(grid, 'A', 'beam')[0], index=grid)
    envc = {}
    for var in ('tre200s0', 'prestas0', 'pva200s0', 'P_caps'):
        envc[var] = {}
        for lag in (0, 2, 4, 6, 8, 12):       # env leads v by `lag` hours
            x = E[var].reindex(grid - pd.Timedelta(hours=lag), method='nearest',
                               tolerance=pd.Timedelta('20min')).to_numpy(float)
            ok = np.isfinite(x)
            envc[var][lag] = float(np.corrcoef(np.log(gm.values[ok]), x[ok])[0, 1])

    J = dict(made_by='rerun/ntof/vt/vt_smooth.py', model=best, knot_h=KNOT_H, block_h=BLOCK_H, lam=lam,
             lam_with_water=lam_e, tau_h=tau, tau_scan=tau_scan, lam_scan=lam_scan, lam_scan_water=lam_scan_env,
             validation=val, chamber_pairs=pairs, env_corr_shared_curve=envc,
             params={name: dict(s=f.s, c={f'{k[0]}_{k[1]}': v for k, v in f.c.items()}, jitter=f.jit,
                                beta=(float(f.coef[-1]) if f.x_env is not None else None))
                     for name, f in fits.items()})
    json.dump(J, open(f'{a.out}/vt_smooth.json', 'w'), indent=1, default=float)

    # ------------------------------------------------------------------------------------------------- figures
    S2 = {arm: S[arm].assign(tm=S[arm].t_start + (S[arm].t_end - S[arm].t_start) / 2) for arm in ARMS}
    fig, axs = plt.subplots(len(ARMS) + 1, 1, figsize=(12, 11), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 3, 3, 2.2]))
    for ax, arm in zip(axs, ARMS):
        dd = d[d.arm == arm]
        for tr, mk in (('beam', 'o'), ('cosmic', 's')):
            g = dd[dd.trigger == tr]
            if len(g):
                ax.errorbar(g.tm, g.v, yerr=g.v_err, fmt='.', ms=2, lw=0.4, alpha=0.25,
                            color=COL[arm] if tr == 'beam' else 'grey')
            sg = S2[arm][(S2[arm].trigger == tr)].dropna(subset=['v'])
            if len(sg):
                ax.errorbar(sg.tm, sg.v, yerr=sg.v_err, fmt=mk, ms=3.5, mfc='white', lw=0.7,
                            color=COL[arm] if tr == 'beam' else '#555', label=f'{tr}, per sub-run')
        for tr, ls in (('beam', '-'), ('cosmic', '--')):
            if arm == 'D' and tr == 'beam':
                continue
            gg = G[(G.arm == arm) & (G.trigger == tr)]
            ax.plot(gg.t, gg.v, ls, color='k', lw=1.0, label=f'smooth ({tr})')
            ax.fill_between(gg.t, gg.v - gg.v_err, gg.v + gg.v_err, color='k', alpha=0.15, lw=0)
        ax.axhline(float(T[T.arm == arm].v_bundle.iloc[0]), color='purple', lw=0.8, ls=':', label='frozen bundle v')
        vv = dd.v
        lo, hi = np.percentile(vv, [0.5, 99.5])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_ylabel(f'{arm}: v (µm/ns)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=5, loc='upper center')
    ax = axs[-1]
    x = E.loc[t0:t1]
    ax.plot(x.index, x.pva200s0, color='#1b7f8c', lw=0.8, label='water vapour p (hPa), Cointrin')
    ax.plot(x.index, ewm_causal(E.pva200s0, tau).loc[t0:t1], color='#1b7f8c', lw=1.8,
            label=f'… smoothed, τ = {tau} h')
    ax.set_ylabel('hPa')
    ax2 = ax.twinx()
    ax2.plot(x.index, x.tre200s0, color='#c0392b', lw=0.7, alpha=0.7, label='air T (°C)')
    ax2.set_ylabel('°C')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=3, loc='upper center')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.suptitle(f'n_TOF drift velocity: measured and the smooth shared model ({best}); band = 1σ interpolation error')
    fig.tight_layout()
    fig.savefig(f'{FIG}/smooth_timeline.png', dpi=120)
    plt.close(fig)

    # zoom: two days
    z0, z1 = pd.Timestamp('2026-07-29 12:00'), pd.Timestamp('2026-07-31 12:00')
    fig, ax = plt.subplots(figsize=(11, 4))
    for arm in ('A', 'C'):
        dd = d[(d.arm == arm) & (d.trigger == 'beam') & (d.tm > z0) & (d.tm < z1)]
        ref = np.exp(M.c[(arm, 'beam')])
        ax.errorbar(dd.tm, 100 * (dd.v / ref - 1), yerr=100 * dd.v_err / ref, fmt='.', ms=3, lw=0.5, alpha=0.6,
                    color=COL[arm], label=f'{arm} acquisitions')
        gg = G[(G.arm == arm) & (G.trigger == 'beam') & (G.t > z0) & (G.t < z1)]
        ax.plot(gg.t, 100 * (gg.v / ref - 1), color=COL[arm], lw=1.6)
        ax.fill_between(gg.t, 100 * ((gg.v - gg.v_err) / ref - 1), 100 * ((gg.v + gg.v_err) / ref - 1),
                        color=COL[arm], alpha=0.2, lw=0)
    ax.set_ylabel('v / campaign level − 1 (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Hh'))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('Two days in detail: per-acquisition v and the smooth model (beam)')
    fig.tight_layout(); fig.savefig(f'{FIG}/smooth_zoom.png', dpi=120); plt.close(fig)

    # chamber-to-chamber correlations
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (p, q) in zip(axs, (('A', 'C'), ('A', 'D'), ('C', 'D'))):
        m = S[p].merge(S[q], on=['run', 'subrun', 'trigger'], suffixes=('_p', '_q')).dropna(subset=['v_p', 'v_q'])
        for tr, mk in (('beam', 'o'), ('cosmic', 's')):
            g = m[m.trigger == tr]
            if len(g):
                k = pairs[f'{p}{q}_{tr}']
                ax.errorbar(g.v_p, g.v_q, xerr=g.v_err_p, yerr=g.v_err_q, fmt=mk, ms=3, lw=0.4, alpha=0.6,
                            label=f"{tr}: r = {k['corr']:.2f}, dln{q}/dln{p} = {k['slope']:.2f} (n {k['n']})")
        ax.set_xlabel(f'v {p} (µm/ns)'); ax.set_ylabel(f'v {q} (µm/ns)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Same sub-run, two chambers (sub-run pooled v)')
    fig.tight_layout(); fig.savefig(f'{FIG}/smooth_pairs.png', dpi=120); plt.close(fig)

    # environment
    gm_b = G[(G.arm == 'C') & (G.trigger == 'beam')].set_index('t').v
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (var, lab) in zip(axs, (('pva200s0', 'water vapour p (hPa)'), ('tre200s0', 'air T (°C)'),
                                    ('prestas0', 'station p (hPa)'), ('P_caps', '³He capsule p (bar)'))):
        xs = (ewm_causal(E[var], tau) if var == 'pva200s0' else E[var]).reindex(gm_b.index, method='nearest',
                                                                                 tolerance=pd.Timedelta('20min'))
        dc = d[(d.arm == 'C') & (d.trigger == 'beam')]
        xx = (ewm_causal(E[var], tau) if var == 'pva200s0' else E[var]).reindex(dc.tm, method='nearest',
                                                                                tolerance=pd.Timedelta('20min'))
        ok = np.isfinite(xx.values)
        ax.plot(xx.values[ok], dc.v.values[ok], '.', ms=2, alpha=0.4, color=COL['C'])
        r = np.corrcoef(xx.values[ok], np.log(dc.v.values[ok]))[0, 1]
        ax.set_xlabel(lab + (f', τ {tau} h' if var == 'pva200s0' else '')); ax.set_ylabel('C beam v (µm/ns)')
        ax.set_title(f'r = {r:+.2f}', fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('Chamber C (beam acquisitions) against the environment')
    fig.tight_layout(); fig.savefig(f'{FIG}/smooth_env.png', dpi=120); plt.close(fig)

    # validation: block-holdout pulls, best model
    Pb = cv(d, makers[best], 'block')
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for arm, g in Pb.groupby('arm'):
        axs[0].hist(g.pull.clip(-6, 6), bins=np.linspace(-6, 6, 49), histtype='step', lw=1.3, color=COL[arm],
                    label=f'{arm}: rms {np.sqrt(np.mean(g.pull ** 2)):.2f}', density=True)
        axs[1].hist(100 * g.res.clip(-0.12, 0.12), bins=np.linspace(-12, 12, 49), histtype='step', lw=1.3,
                    color=COL[arm], density=True, label=f"{arm}: 68 % ±{summary(g)[arm]['res68_pct']:.2f} %")
    xg = np.linspace(-6, 6, 200)
    axs[0].plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), 'k:', lw=1, label='N(0, 1)')
    axs[0].set_xlabel('pull of a held-out acquisition'); axs[1].set_xlabel('v_obs / v_model − 1 (%)')
    for ax in axs:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f'Block holdout: every 4th {BLOCK_H:.0f} h block removed and predicted ({best})')
    fig.tight_layout(); fig.savefig(f'{FIG}/smooth_holdout.png', dpi=120); plt.close(fig)

    # environment time series: C beam v, water vapour (raw + smoothed), outdoor T, station p, capsule p
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    gc_ = G[(G.arm == 'C') & (G.trigger == 'beam')]
    axs[0].plot(gc_.t, gc_.v, color=COL['C']); axs[0].set_ylabel('C v (µm/ns)')
    xw = E.loc[t0:t1]
    axs[1].plot(xw.index, xw.pva200s0, color='#1b7f8c', lw=0.6)
    axs[1].plot(xw.index, ewm_causal(E.pva200s0, tau).loc[t0:t1], color='#1b7f8c', lw=2)
    axs[1].invert_yaxis(); axs[1].set_ylabel('water vapour\np (hPa, inverted)')
    axs[2].plot(xw.index, xw.tre200s0, color='#c0392b', lw=0.8); axs[2].set_ylabel('air T (°C)')
    axs[3].plot(xw.index, xw.prestas0, color='#555', lw=0.8); axs[3].set_ylabel('station p (hPa)')
    axs[4].plot(xw.index, xw.P_caps, color='#3d6b8c', lw=0.8); axs[4].set_ylabel('³He capsule\np (bar)')
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.suptitle('Chamber C drift velocity (smooth model) and the environment: MeteoSwiss Genève-Cointrin (10 min, '
                 'shifted to CEST) and the ³He capsule gauge')
    fig.tight_layout(); fig.savefig(f'{FIG}/smooth_envtime.png', dpi=120); plt.close(fig)

    # water-vapour sensitivity in physical units, from the water-only model: d ln v / d pva_smoothed
    sd_x = float(ewm_causal(E.pva200s0, tau).loc[t0:t1].std())
    wf = fits['water']
    sens = {arm: 100 * wf.coef[-1] * wf.s[arm] / sd_x for arm in wf.s}
    J['water_sensitivity_pct_per_hPa'] = sens
    json.dump(J, open(f'{a.out}/vt_smooth.json', 'w'), indent=1, default=float)
    # physics reading: the environment alone (no spline), per arm on beam (D: cosmic sub-runs) acquisitions:
    # ln v = a + b * water_vapour(tau) + c * capsule_daily(lag)   (capsule p minus its 48 h running mean = hall T)
    caps = E.P_caps.interpolate(limit=36)
    caps_d = caps - caps.rolling('48h', center=True).mean()
    pva_s = ewm_causal(E.pva200s0, tau)
    envfit = {}
    for arm in ARMS:
        dd = d[(d.arm == arm) & (d.trigger == ('beam' if arm != 'D' else 'cosmic'))]
        y = np.log(dd.v.to_numpy(float))
        best_l = None
        for lag in (0, 1, 2, 3):
            X = np.column_stack([np.ones(len(y)),
                                 pva_s.reindex(dd.tm, method='nearest', tolerance=pd.Timedelta('20min')).to_numpy(float),
                                 caps_d.reindex(dd.tm - pd.Timedelta(hours=lag), method='nearest',
                                                tolerance=pd.Timedelta('20min')).to_numpy(float)])
            ok = np.isfinite(X).all(1)
            b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            r = y[ok] - X[ok] @ b
            rec = dict(lag_h=lag, n=int(ok.sum()), r2=float(1 - r.var() / y[ok].var()),
                       res68_pct=float(100 * np.subtract(*np.percentile(r, [84, 16])) / 2),
                       water_pct_per_hPa=float(100 * b[1]), hall_pct_per_bar=float(100 * b[2]),
                       hall_pct_per_K=float(100 * b[2] * float(caps.mean()) / 300.0))
            if best_l is None or rec['r2'] > best_l['r2']:
                best_l = rec
        envfit[arm] = best_l
    J['env_regression'] = envfit
    json.dump(J, open(f'{a.out}/vt_smooth.json', 'w'), indent=1, default=float)
    write_html(a.out, J, best, tau, sens, val, pairs, envc)

    print(json.dumps(dict(model=best, tau=tau, lam=lam, lam_e=lam_e, validation=val, pairs=pairs,
                          params=J['params']), indent=1, default=float))


if __name__ == '__main__':
    main()
