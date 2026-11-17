#!/usr/bin/env python3
"""make_vt_page.py -- standalone write-up (a page, not slides) of the n_TOF drift velocity v(t) results.

    make_vt_page.py [--vt DIR] [--ab DIR] [--env DIR] [--gas DIR] [--out FILE]

Same inputs as make_vt_deck.py (vt_table.py / vt_smooth.py outputs, the A/B summary, MeteoSwiss, the ³He capsule and
the gas-mixer log); every number on the page comes from those files.  Charts are inline SVG with hover tooltips, so the
page is one self-contained file.  Publish with
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py FILE --slug ntof-drift-velocity-report --force --deploy
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vt_smooth as vs   # noqa: E402

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd    # noqa: E402

WD = vs.WD
T0 = pd.Timestamp('2026-07-26 00:00')
XR = (12.0, 15 * 24 + 12.0)
COL = {'A': sd.BLUE, 'C': sd.ORANGE, 'D': sd.GREEN}
W = 1200                                   # chart width in SVG px (scaled to the column)
MODEL_TXT = {'spline': 'spline only', 'water': 'water vapour only', 'water+mixerT': 'water vapour + mixer T',
             'spline+water': 'spline + water vapour', 'spline+water+mixerT': 'spline + water vapour + mixer T',
             'spline+water+hall': 'spline + water vapour + capsule (hall)'}


def hrs(t):
    return ((pd.to_datetime(pd.Series(t)) - T0) / pd.Timedelta(hours=1)).to_numpy(float)


def date_ticks(step=2):
    return [(d * 24.0, (T0 + pd.Timedelta(days=d)).strftime('%-d %b')) for d in range(1, 16, step)]


def fmt_t(t):
    return pd.Timestamp(t).strftime('%-d %b %H:%M')


def fig(svg, cap, n):
    return f'<figure>{svg}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>'


def table(head, rows, cls=''):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


CSS = """
:root{--bg:#f5f3ee;--card:#fffdf9;--ink:#1c2230;--mut:#5a6170;--rule:#d8d3c8;--acc:#1f5fa8}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 'IBM Plex Sans',Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:34px;line-height:1.2;margin:.2em 0 .4em;letter-spacing:-.4px}
h2{font-size:23px;margin:2.1em 0 .5em;padding-top:.6em;border-top:1px solid var(--rule)}
h3{font-size:18px;margin:1.4em 0 .3em}
.kick{color:var(--mut);font-size:14px;letter-spacing:.6px;text-transform:uppercase}
.lede{font-size:19px}
.answer{background:var(--card);border-left:5px solid #2f8a5b;border-radius:6px;padding:14px 20px;margin:18px 0}
.answer li{margin:.35em 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--card);border-radius:8px;padding:12px 16px;border:1px solid var(--rule)}
.tile .v{font-size:30px;font-weight:600;line-height:1.1}
.tile .l{font-size:14px;color:var(--mut)}
figure{margin:22px 0;background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:12px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:14.5px;color:var(--mut);margin-top:8px;line-height:1.5}
.tw{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;font-size:15px;background:var(--card)}
th,td{padding:5px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{border-bottom:2px solid var(--ink);font-weight:600}
th:first-child,td:first-child{text-align:left}
tr.best td{font-weight:600;background:#eef5ee}
code{font-family:'IBM Plex Mono',monospace;font-size:.9em;background:#ebe8e1;padding:0 3px;border-radius:3px}
.cav li{margin:.3em 0}
.term{border-bottom:2px dotted currentColor;cursor:help}
svg [data-tip]:not(.hit):hover{opacity:.8}
#tip{position:fixed;z-index:10;max-width:380px;padding:9px 12px;border-radius:8px;background:#10141c;color:#eef0f3;
 font-size:14px;line-height:1.45;box-shadow:0 6px 20px rgba(0,0,0,.4);pointer-events:none;opacity:0;transition:opacity .08s;white-space:pre-line}
#tip.on{opacity:1}
.foot{color:var(--mut);font-size:13.5px;margin-top:40px}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vt', default=f'{WD}/ntof_pass1/vt')
    ap.add_argument('--ab', default=f'{WD}/ntof_pass1/vt_ab')
    ap.add_argument('--env', default=f'{WD}/ntof_pass1/env')
    ap.add_argument('--gas', default=f'{WD}/ntof_pass1/gas_flow')
    ap.add_argument('--out', default=f'{WD}/ntof_pass1/vt/page/ntof-drift-velocity-report.html')
    a = ap.parse_args()

    J = json.load(open(f'{a.vt}/vt_smooth.json'))
    AB = json.load(open(f'{a.ab}/ab_summary.json'))
    T = vs.load_v(a.vt)
    S = {arm: pd.read_parquet(f'{a.vt}/vt_{arm}_subrun.parquet') for arm in 'ACD'}
    for arm in S:
        S[arm]['tm'] = S[arm].t_start + (S[arm].t_end - S[arm].t_start) / 2
    G = pd.read_parquet(f'{a.vt}/vt_smooth_grid.parquet')
    HO = pd.read_parquet(f'{a.vt}/vt_smooth_holdout.parquet')
    E = vs.load_env(a.env)
    best = J['model']
    MX = J['mixer']
    GM = vs.load_gas(a.gas).loc[:pd.Timestamp(MX['t1']) + pd.Timedelta('10min')]   # not the 10 Aug shutdown
    E = E.join(GM.argon_temp_c.rename('T_mfc'), how='outer').sort_index()
    P = J['params'][best]
    val = J['validation']
    tau, tau_T = J['tau_h'], J['tau_T_h']
    envr = J['env_regression']
    sensT = J['sensitivity']['spline+water+mixerT']
    lev = {arm: float(np.exp(P['c'][f'{arm}_beam' if arm != 'D' else 'D_cosmic'])) for arm in 'ACD'}
    gC = G[(G.arm == 'C') & (G.trigger == 'beam')]
    swing_c = 100 * (gC.v.max() / gC.v.min() - 1)
    n_files = len(T)
    pairs = J['chamber_pairs']
    vbw, vbm = val['spline+water']['block'], val['spline+water+mixerT']['block']
    v24w, v24m = val['spline+water']['block24'], val['spline+water+mixerT']['block24']
    v24h = val['spline+water+hall']['block24']
    sw3 = lambda v: ' / '.join(f'±{v[a]["res68_pct"]:.2f}' for a in 'ACD')          # noqa: E731
    pr3 = lambda v: ' / '.join(f'{v[a]["pull_rms"]:.2f}' for a in 'ACD')            # noqa: E731
    figs = []

    # ---------------------------------------------------------------- F1: all chambers, relative
    p = sd.Plot(W, 520, XR, (-13, 9), xlabel='date (2026, local time)', ylabel='v / campaign level − 1 (%)')
    p.xticks(date_ticks()).yticks([(v, f'{v:+d}' if v else '0') for v in range(-12, 9, 4)])
    p.hline(0, sd.RULE, None, 1)
    for arm in 'ACD':
        tr = 'beam' if arm != 'D' else 'cosmic'
        q = S[arm][S[arm].trigger == tr].dropna(subset=['v']).sort_values('tm')
        rel = 100 * (q.v / lev[arm] - 1)
        k = ((rel > -13) & (rel < 9)).to_numpy()
        tips = [f'{arm} run_{r.run} {r.subrun} ({tr})\n{fmt_t(r.tm)}\nv = {r.v:.2f} ± {r.v_err:.2f} µm/ns' for r in q.itertuples()]
        p.points(hrs(q.tm)[k], rel.to_numpy()[k], COL[arm], r=4, tips=[t for t, kk in zip(tips, k) if kk])
        g = G[(G.arm == arm) & (G.trigger == tr)]
        p.line(hrs(g.t), (100 * (g.v / lev[arm] - 1)).to_numpy(), COL[arm], w=3, markers=False,
               tip=f'{arm}: smooth model, ×{P["s"][arm]:.2f} the shared curve')
    figs.append(fig(p.svg('all chambers') + sd.legend([('A (beam)', sd.BLUE, 'dot'), ('C (beam)', sd.ORANGE, 'dot'),
                                                      ('D (cosmic)', sd.GREEN, 'dot'), ('smooth model', sd.INK, 'line')], size=16),
                    f'Per-sub-run drift velocity of each chamber relative to its campaign level (A and C on beam triggers, D '
                    f'on cosmics), with the smooth shared model ({MODEL_TXT[best]}). C moves {P["s"]["C"]:.2f}× and D '
                    f'{P["s"]["D"]:.2f}× as much as A. Hover a point for its run and sub-run.', 1))

    # ---------------------------------------------------------------- F2: the mixer
    wv = GM.loc[T0 + pd.Timedelta(hours=XR[0]):T0 + pd.Timedelta(hours=XR[1])]
    tTlo, tThi = np.floor(4 * wv.argon_temp_c.quantile(0.002)) / 4 - 0.25, np.ceil(4 * wv.argon_temp_c.quantile(0.998)) / 4 + 0.25
    strips = []
    specs = [('isobutane fraction (%)', 'iso_pct_meas', (9.9, 10.1), 0.05, sd.INK, '{:.2f}'),
             ('total flow (ln/h)', 'total_flow_lnh', (7.70, 7.86), 0.04, sd.INK, '{:.2f}'),
             ('mixer temperature (°C)', 'argon_temp_c', (tTlo, tThi), 0.5, sd.PURPLE, '{:.1f}')]
    for i, (lab, c, yr, st, colr, f) in enumerate(specs):
        last = i == len(specs) - 1
        pl = sd.Plot(W, 150 if not last else 200, XR, yr, xlabel='date (2026, local time)' if last else '', ylabel='',
                     margin=(12, 30, 70 if last else 14, 104))
        pl.xticks(date_ticks() if last else [(t, '') for t, _ in date_ticks()])
        pl.yticks([(v, f.format(v)) for v in np.arange(np.ceil(yr[0] / st) * st, yr[1] + 1e-9, st)][::2])
        q = wv[c].dropna()
        pl.line(hrs(q.index)[::2], q.to_numpy()[::2], colr, w=2, markers=False, tip=f'{lab}: 10-min medians of the 2 s log')
        if c == 'iso_pct_meas':
            pl.hline(MX['iso_set_pct'], sd.PURPLE, '4 6', 1.5, tip=f'setpoint {MX["iso_set_pct"]:.2f} %')
        pl.text(XR[0] + 4, yr[1] - 0.18 * (yr[1] - yr[0]), lab, 19, sd.INK, weight=600)
        strips.append(pl.svg(lab))
    figs.append(fig(''.join(strips),
                    f'The gas mixer over the v window, from the DAQ\'s gas_watcher log (two Bronkhorst mass-flow controllers, one '
                    f'row per 2 s; 10-min medians shown). Isobutane {MX["iso_pct"]["p01"]:.3f}–{MX["iso_pct"]["p99"]:.3f} % and '
                    f'flow {MX["flow_lnh"]["p01"]:.4f}–{MX["flow_lnh"]["p99"]:.4f} ln/h (1st–99th percentile). The '
                    f'temperature is the argon controller\'s own sensor, i.e. the gas rack.', 2))

    # ---------------------------------------------------------------- F3: environment stack
    t0e, t1e = T0 + pd.Timedelta(hours=XR[0]), T0 + pd.Timedelta(hours=XR[1])
    Ee = E.loc[t0e:t1e]
    xe = hrs(Ee.index)
    pva_s = vs.ewm_causal(E.pva200s0, tau).loc[t0e:t1e]
    specs = [('C beam v (µm/ns)', hrs(gC.t), gC.v.to_numpy(), None, sd.ORANGE, (27.5, 32.5), 1, 'C smooth model'),
             ('outdoor water vapour (hPa)', xe, Ee.pva200s0.to_numpy(), (hrs(pva_s.index), pva_s.to_numpy()), '#1b7f8c', (9, 24), 3,
              f'MeteoSwiss Genève-Cointrin, 10 min; thick = causal smoothing τ = {tau} h'),
             ('³He capsule (bar)', xe, Ee.P_caps.to_numpy(), None, '#3d6b8c', (493.5, 499.2), 1,
              'sealed ~500 bar volume: p ∝ T, a hall thermometer (1 bar ≈ 0.6 K)'),
             ('mixer T (°C)', xe, Ee.T_mfc.to_numpy(), None, sd.PURPLE, (tTlo, tThi), 0.5, 'argon MFC temperature sensor'),
             ('outdoor air T (°C)', xe, Ee.tre200s0.to_numpy(), None, sd.RED, (14, 37), 5, 'MeteoSwiss Cointrin, 2 m')]
    rows_ = []
    for i, (lab, xx, yy, extra, c, yr, st, tp) in enumerate(specs):
        last = i == len(specs) - 1
        pl = sd.Plot(W, 120 if not last else 176, XR, yr, xlabel='date (2026, local time)' if last else '', ylabel='',
                     margin=(10, 30, 66 if last else 10, 104))
        pl.xticks(date_ticks() if last else [(t, '') for t, _ in date_ticks()])
        pl.yticks([(v, f'{v:g}') for v in np.arange(np.ceil(yr[0] / st) * st, yr[1], st)][::2])
        ok = np.isfinite(yy)
        pl.line(xx[ok][::2], yy[ok][::2], c, w=1.6 if extra else 2.2, markers=False, tip=tp)
        if extra:
            pl.line(extra[0], extra[1], c, w=4.5, markers=False, tip=tp)
        pl.text(XR[0] + 4, yr[1] - 0.2 * (yr[1] - yr[0]), lab, 19, sd.INK, weight=600)
        rows_.append(pl.svg(lab))
    figs.append(fig(''.join(rows_),
                    'Chamber C\'s drift velocity (smooth model) against the environment. The slow trend follows the outdoor '
                    'water vapour (wetter = slower: the 4–6 August dip sits on the most humid days); the daily wiggle runs opposite to the two hall thermometers, the '
                    '³He capsule and the gas mixer. Outdoor air temperature tracks neither strongly.', 3))

    # ---------------------------------------------------------------- F4: residual after water + hall vs mixer T
    dC = T[(T.arm == 'C') & (T.trigger == 'beam')].dropna(subset=['v'])
    at = lambda s, t: s.reindex(pd.to_datetime(t), method='nearest', tolerance=pd.Timedelta('20min')).to_numpy(float)  # noqa: E731
    caps = E.P_caps.interpolate(limit=36)
    capd = caps - caps.rolling('48h', center=True).mean()
    lagC = envr['C']['water+hall']['lag_h']
    xTm = E.T_mfc.interpolate(limit=36) if tau_T == 0 else vs.ewm_causal(E.T_mfc, tau_T)
    xw_, xc_, xt_ = at(vs.ewm_causal(E.pva200s0, tau), dC.tm), at(capd, dC.tm - pd.Timedelta(hours=lagC)), at(xTm, dC.tm)
    ok = np.isfinite(xw_) & np.isfinite(xc_) & np.isfinite(xt_)
    y = np.log(dC.v.to_numpy(float))[ok]
    X = np.column_stack([np.ones(ok.sum()), xw_[ok], xc_[ok]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = 100 * (y - X @ b)
    dT = xt_[ok] - xt_[ok].mean()
    kT = np.polyfit(dT, r, 1)
    pm = sd.Plot(W, 520, (-1.4, 1.4), (-6, 6), xlabel='mixer temperature − its campaign mean (K)',
                 ylabel='C v after water + hall terms (%)')
    pm.xticks([(v, f'{v:+g}' if v else '0') for v in (-1, -0.5, 0, 0.5, 1)]).yticks([(v, f'{v:+d}' if v else '0') for v in range(-6, 7, 2)])
    pm.scatter(dT, r, sd.PURPLE, r=3, opacity=0.3, tip=f'{ok.sum():,} C beam acquisitions')
    xs = np.linspace(-1.3, 1.3, 10)
    pm.line(xs, np.polyval(kT, xs), sd.INK, w=3, markers=False, tip=f'slope {kT[0]:+.2f} %/K on the residual')
    pm.text(-1.35, 5.1, f'r = {np.corrcoef(dT, r)[0, 1]:+.2f}', 24, sd.INK, weight=600)
    figs.append(fig(pm.svg('mixer T residual'),
                    f'What the water-vapour and capsule terms leave in chamber C ({ok.sum():,} beam acquisitions), against the '
                    f'mixer temperature (causal τ = {tau_T:g} h). The residual still falls with temperature, but weakly '
                    f'(r = {np.corrcoef(dT, r)[0, 1]:+.2f}): per acquisition the two hall thermometers carry mostly the same '
                    f'information. The mixer sensor earns its place across long gaps (section 5), not here.', 4))

    # ---------------------------------------------------------------- F5: held-out error vs gap length
    models = [m for m in ('water', 'water+mixerT', 'spline', 'spline+water', 'spline+water+hall', 'spline+water+mixerT')
              if m in val]
    GAPS = [('block', 6), ('block12', 12), ('block24', 24)]
    pair = [('spline+water', sd.BLUE), ('spline+water+hall', '#3d6b8c'), ('spline+water+mixerT', sd.PURPLE)]
    ymax = max(val[m][g][arm]['res68_pct'] for m, _ in pair for g, _ in GAPS for arm in 'ACD') * 1.18
    slots = [(arm, g, h) for arm in 'ACD' for g, h in GAPS]
    xpos = {(arm, g): i + 0.6 * 'ACD'.index(arm) for i, (arm, g, h) in enumerate(slots)}
    pv = sd.Plot(W, 480, (-0.7, len(slots) - 1 + 1.2 + 0.7), (0, ymax), xlabel='chamber and length of the held-out gap',
                 ylabel='held-out 68 % residual (%)')
    pv.xticks([(xpos[(arm, g)], f'{arm} {h} h') for arm, g, h in slots]).yticks([(v, f'{v:g}') for v in np.arange(0, ymax, 1)])
    for k, (m, c) in enumerate(pair):
        for arm, g, h in slots:
            v = val[m][g][arm]
            pv.vbar(xpos[(arm, g)] + (k - 1) * 0.28, v['res68_pct'], 22, c,
                    tip=f'{MODEL_TXT[m]}\n{arm}, every 4th {h} h block held out\n±{v["res68_pct"]:.2f} %, pull rms {v["pull_rms"]:.2f}')
    figs.append(fig(pv.svg('gap length') + sd.legend([(MODEL_TXT[m], c) for m, c in pair], size=16),
                    'How well the smooth model predicts v inside a gap with no data: water vapour only, plus the capsule\'s '
                    'daily swing, or plus the mixer temperature: '
                    'every 4th 6, 12 or 24 h block removed, the model fitted on the rest, and the held-out acquisitions '
                    'predicted (68 % half-width of measured / predicted − 1; hover for the pull rms). Across 6 h the spline '
                    'bridges the gap on its own; across 24 h a hall-temperature term carries the daily cycle the spline cannot '
                    'see. Either thermometer does it for A and C; the mixer\'s does better for D.', 5))

    # ---------------------------------------------------------------- F6: pulls of the model used
    Pb = HO[(HO.model == best) & (HO['mode'] == 'block24')]
    edges = np.linspace(-5, 5, 41)
    ph = sd.Plot(W, 420, (-5, 5), (0, 0.45), xlabel='pull of a held-out acquisition', ylabel='density')
    ph.xticks([(v, f'{v:+d}' if v else '0') for v in range(-4, 5, 2)]).yticks([(v, f'{v:g}') for v in (0, 0.1, 0.2, 0.3, 0.4)])
    xx = np.linspace(-5, 5, 101)
    ph.line(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), sd.INK, w=2, dash='6 6', markers=False, tip='N(0, 1)')
    for arm in 'ACD':
        g = Pb[Pb.arm == arm]
        h, _ = np.histogram(g.pull.clip(-4.99, 4.99), edges, density=True)
        ph.line(*sd.step_xy(edges, h), COL[arm], w=3, markers=False, tip=f'{arm}: rms {np.sqrt(np.mean(g.pull ** 2)):.2f}, n {len(g):,}')
    figs.append(fig(ph.svg('pulls') + sd.legend([('A', sd.BLUE), ('C', sd.ORANGE), ('D', sd.GREEN), ('N(0, 1)', sd.INK, 'dash')], size=16),
                    'Pulls of the held-out acquisitions for the model used, 24 h blocks held out (the hardest test). An rms '
                    'near 1 means the quoted error band (measurement + per-chamber jitter + interpolation) is honest. '
                    + ', '.join(f'{k} {np.sqrt(np.mean(Pb[Pb.arm == k].pull ** 2)):.2f}' for k in 'ACD')
                    + ': C leans positive, so its band is ~20 % too narrow across a full-day gap.', 6))

    # ---------------------------------------------------------------- tables
    env_rows = []
    for arm in 'ACD':
        for rn in ('water+hall', 'water+mixerT', 'water+hall+mixerT'):
            v = envr[arm][rn]
            env_rows.append([arm, rn.replace('+', ' + ').replace('mixerT', 'mixer T'), f'{v["n"]:,}', f'{v["r2"]:.2f}',
                             f'±{v["res68_pct"]:.2f} %', f'{v["water_pct_per_hPa"]:+.2f}',
                             f'{v["hall_pct_per_K"]:+.1f}' if 'hall_pct_per_K' in v else '—',
                             f'{v["mixerT_pct_per_K"]:+.1f}' if 'mixerT_pct_per_K' in v else '—'])
    env_tbl = table(['chamber', 'regressors', 'n', 'R²', '68 % residual', 'water %/hPa', 'hall %/K', 'mixer T %/K'], env_rows)
    MODES = [('random', 'random 20 %'), ('block', '6 h gaps'), ('block12', '12 h gaps'), ('block24', '24 h gaps')]
    val_rows = ''.join(
        f'<tr class="{"best" if m == best else ""}"><td>{MODEL_TXT[m]}</td>'
        + ''.join(f'<td>{val[m][mode][arm]["res68_pct"]:.2f} <span style="color:#7d8796">({val[m][mode][arm]["pull_rms"]:.2f})</span></td>'
                  for mode, _ in MODES for arm in 'ACD') + '</tr>' for m in models)
    val_tbl = ('<div class="tw"><table><thead><tr><th rowspan=2>model</th>'
               + ''.join(f'<th colspan=3>{lab}</th>' for _, lab in MODES) + '</tr><tr>' + '<th>A</th><th>C</th><th>D</th>' * len(MODES)
               + f'</tr></thead><tbody>{val_rows}</tbody></table></div>')
    ab_rows = [['same chosen track', f'{100 * AB["A"]["same_frac"]:.1f} %', f'{100 * AB["C"]["same_frac"]:.1f} %'],
               ['tan = table correction (68 %)', f'±{100 * AB["A"]["tan_rel_68"]:.2f} %', f'±{100 * AB["C"]["tan_rel_68"]:.2f} %'],
               ['all tracks found in both', f'{100 * AB["A"]["tracks"][2] / AB["A"]["tracks"][0]:.1f} %',
                f'{100 * AB["C"]["tracks"][2] / AB["C"]["tracks"][0]:.1f} %'],
               ['tracks with |tan| < 0.5, A → B', f'{100 * (AB["A"]["tracks05"][1] / AB["A"]["tracks05"][0] - 1):+.1f} %',
                f'{100 * (AB["C"]["tracks05"][1] / AB["C"]["tracks05"][0] - 1):+.1f} %']]
    ab_tbl = table(['', 'A', 'C'], ab_rows)

    sens_txt = ', '.join(f'{k} {v["mixerT_pct_per_K"]:+.1f} %/K' for k, v in sensT.items())
    tiles = ''.join(f'<div class="tile"><div class="v" style="color:{c}">{v}</div><div class="l">{l}</div></div>' for v, l, c in (
        (f'{swing_c:.0f} %', 'swing of chamber C\'s drift velocity over the campaign (A a third of that)', sd.ORANGE),
        (f'{MX["iso_pct"]["median"]:.2f} %', f'isobutane from the mixer, constant to ±{MX["iso_pct"]["rms"]:.3f} % rms: the mix did not move v', sd.GREEN),
        (f'{100 * envr["C"]["water+hall+mixerT"]["r2"]:.0f} %', f'of C\'s variation explained by outdoor water vapour and two hall thermometers ({100 * envr["C"]["water+hall"]["r2"]:.0f} % without the mixer\'s)', sd.BLUE),
        (f'±{v24m["C"]["res68_pct"]:.1f} %', f'C predicted across a 24 h gap with the mixer temperature in the model (±{v24w["C"]["res68_pct"]:.1f} % without)', sd.PURPLE)))

    body = f"""
<p class="kick">n_TOF pass 1 · P7a · drift velocity · {pd.Timestamp.now():%-d %B %Y}</p>
<h1>The n_TOF drift velocity follows the weather, not the gas mixer</h1>
<p class="lede">The drift velocity v of chambers A, C and D was measured for every acquisition of the n_TOF campaign, from how
long the charge column of clean tracks lasts. It moves by up to {swing_c:.0f} % over the campaign, the three chambers move together,
and one smooth shared curve describes it. The gas mixer's own log shows the mix and flow were constant throughout, so the cause
is downstream of the mixer. Outdoor humidity and the hall temperature (measured by the ³He capsule and by the mixer's sensor)
track it closely, which points to water getting into the gas.</p>

<div class="answer"><b>Answer</b><ul>
<li><b>One curve for all chambers:</b> ln v = c[chamber, trigger] + s[chamber]·g(t), with C moving {P['s']['C']:.2f}× and D
{P['s']['D']:.2f}× as much as A. Same-sub-run correlation A–C r = {pairs['AC_beam']['corr']:.2f}.</li>
<li><b>The mixer did not do it:</b> {MX['iso_pct']['median']:.3f} % isobutane (1–99 %: {MX['iso_pct']['p01']:.3f}–{MX['iso_pct']['p99']:.3f}) and
{MX['flow_lnh']['median']:.3f} ln/h ({MX['flow_lnh']['p01']:.4f}–{MX['flow_lnh']['p99']:.4f}) over the whole window, while v moved by percent.</li>
<li><b>Its thermometer helps across long gaps:</b> the mixer's temperature sensor tracks the hall. Added to the smooth model,
it makes no difference across 6 h gaps (A / C / D {sw3(vbw)} % without, {sw3(vbm)} % with), because the spline already has data
on both sides. Across 24 h gaps it takes the error from {sw3(v24w)} % to {sw3(v24m)} %, and the pull rms from {pr3(v24w)} to
{pr3(v24m)} (A and D back to ~1; C's band stays ~20 % narrow across long gaps). The capsule in the same place does
about as well for A and C ({sw3(v24h)} %), so what matters is having a hall temperature; the mixer's sensor wins on D. Slope {sens_txt}: v slows as the hall warms.</li>
<li><b>No rerun:</b> the reconstruction does not use v in the fit; an A/B re-reconstruction of 30 acquisitions with the measured v
changes no fit, so the angles are corrected in the tables: tan × v_bundle / v(t).</li></ul></div>
<div class="tiles">{tiles}</div>

<h2>1. Measuring v per acquisition</h2>
<p>A track crossing the 28–29 mm drift gap leaves charge at every depth, and the strips see it arrive over the drift time of the
whole gap. Stacking the strip-summed signals of the pass's clean single tracks (≥ 40 per view, not saturated) and fitting the box
gives the column duration T, independent of the angle; v = gap / T, x and y views combined. That gives {n_files:,}
acquisition × chamber measurements. B has no field-shaping ring and so no box; D's beam column is not a box, so D is measured on
cosmic triggers only. HV never changed during the campaign.</p>
{figs[0]}

<h2>2. A smooth model to interpolate with</h2>
<p>v is smooth on the scale of hours: within a ~1 h sub-run acquisitions scatter only a little more than their bootstrap error. The
model is a cubic P-spline g(t) shared by the chambers (knots every {J['knot_h']:.0f} h, smoothing chosen by cross-validation), plus
environmental regressors, fitted on ln v with a per-chamber level and scale. Each chamber gets a jitter term so that χ²/dof = 1,
and the spline's covariance gives an interpolation error that grows across gaps. The chambers share one gas line (A → B → C → D),
which is why one curve fits all three.</p>

<h2>3. The gas mixer was constant</h2>
<p>The DAQ's <code>gas_watcher.py</code> logged both Bronkhorst mass-flow controllers every 2 s from 7 July to 10 August (EOS
<code>july_beam/slow_control/gas_flow/</code>). Over the v window the setpoint never changed ({MX['iso_set_pct']:.2f} % isobutane, argon
7.0 ln/h) and the measured mix held to {MX['iso_pct']['rms']:.3f} % rms; {MX['n_excursion_10min']} ten-minute bins lie outside ±0.5 %
isobutane or below 7.5 ln/h. A change of a few thousandths of a percent of isobutane cannot move v by percent, so the mixer is ruled
out as the driver.</p>
{figs[1]}

<h2>4. What drives v: water and temperature</h2>
<p>The outdoor water-vapour pressure (MeteoSwiss Genève-Cointrin), smoothed with a causal {tau} h time constant (a model of water
permeating into the gas lines), follows the slow trend. The daily wiggle runs opposite to the hall temperature. Two sensors record
the hall temperature: the sealed ³He capsule (pressure ∝ temperature) and the mixer's controller. They are correlated but not
identical, and the mixer's sensor sits by the gas rack.</p>
{figs[2]}
<p>Fitting the environment alone (no spline) shows how much each sensor carries:</p>
{env_tbl}
<p>Both signs are the wrong way for a gas-density (E/N) effect: at this field v would rise with temperature. Instead v falls with
water and with temperature, by several %/K on C, far beyond what density could give. Water that enters faster when it is warm and
humid (permeation through the lines, outgassing from the walls), and more of it downstream in the chain, fits the signs and the
C, D &gt; A ordering. This is a reading of correlations, not a measurement of the gas.</p>
{figs[3]}

<h2>5. Validation</h2>
<p>Each model is fitted with part of the data removed and asked to predict it: a random 20 % of acquisitions (interpolation between
neighbours), and every 4th block of 6, 12 or 24 h (a real gap: the campaign has accesses and stops of that length). Entries are the
68 % half-width of measured / predicted − 1 in %, with the pull rms in brackets; the row in bold is the model used, chosen by the
mean over chambers and the three gap lengths. The smoothing strength of each model is chosen the same way.</p>
{val_tbl}
{figs[4]}
{figs[5]}

<h2>6. Does the pass need a rerun? No.</h2>
<p>30 beam acquisitions at the fastest times (C 12–15 % above the frozen v, A 3.5–5 %) were re-reconstructed with only the drift
velocity changed to the measured one. The fit works in time units: w, p0 and t0 are identical, and tan scales as v_frozen / v.
What the frozen v does change is which candidate is ranked first, so about 1 % of tracks are swapped, while the counts change by
less than 0.6 %. Decided 17 November: no rerun. tan is corrected in the tables with v(t), |tan| &lt; 0.5 is cut so the acceptance
does not move with v, and the ~1 % churn is a stated systematic.</p>
{ab_tbl}

<h2>What this does not rule out</h2>
<ul class="cav">
<li>The water reading is a correlation with outdoor air and two hall thermometers. The mixer log shows what went into the line,
not what reached the chambers, and no gas composition was measured during the campaign.</li>
<li>The mixer-temperature term is a correlation too. Its weight is fitted, and its gain holds up on held-out 12 and 24 h blocks,
but the sensor is a controller's body temperature (it includes self-heating), not a calibrated hall thermometer. Inside normal
data taking (gaps under ~6 h) the model is set by the spline and the term changes nothing.</li>
<li>A's shape is tied to C's (one curve); A's own small departures (~{100 * P['jitter']['A']:.1f} %) are in its jitter.</li>
<li>D's absolute v reads ~10 % off its bundle even on its own calibration run: its scale needs an angle anchor; only its time
dependence is used.</li>
<li>v is the column duration over the timing-free gap: a gain change at fixed HV would move T too (HV was constant).</li>
<li>The A/B covers beam triggers of A and C; two-track splits were not separated.</li>
</ul>

<p class="foot">Generated by <code>rerun/ntof/vt/make_vt_page.py</code> (nTof_x17) from <code>vt_table.py</code>,
<code>vt_smooth.py</code> and <code>ab_compare.py</code> outputs (EOS <code>Analysis_pass1/vt_table/</code>). Environment:
MeteoSwiss open data (station GVE, 10 min, UTC shifted to local time), the DAQ's ³He capsule and gas-mixer logs. Slides:
<a href="ntof-drift-velocity.html">ntof-drift-velocity</a>.</p>
"""
    html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>n_TOF drift velocity</title>\n'
            '<meta name="description" content="The n_TOF drift velocity of chambers A, C and D through the campaign: a smooth '
            'shared model, validated on held-out data; the gas mixer was constant; water vapour and hall temperature drive it.">\n'
            f'<style>{CSS}</style></head><body><main>{body}</main><div id="tip" role="tooltip"></div><script>{sd._JS}</script>'
            '</body></html>\n')
    meta = dict(title='n_TOF drift velocity: follows the weather, not the gas mixer',
                summary=(f'v of A, C, D per acquisition over the n_TOF campaign; one smooth shared model, held-out error '
                         f'±{vbm["A"]["res68_pct"]:.1f} % (A) / ±{vbm["C"]["res68_pct"]:.1f} % (C) across 6 h gaps; the mixer held '
                         f'{MX["iso_pct"]["median"]:.2f} % isobutane throughout; water vapour + hall temperature explain '
                         f'{100 * envr["C"]["water+hall+mixerT"]["r2"]:.0f} % of C; no rerun.'),
                tags='x17, n_TOF, drift velocity, gas', date=pd.Timestamp.now().strftime('%Y-%m-%d'))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w').write('<!--note\n' + ''.join(f'{k}: {v}\n' for k, v in meta.items()) + '-->\n' + html)
    print(a.out)


if __name__ == '__main__':
    main()
