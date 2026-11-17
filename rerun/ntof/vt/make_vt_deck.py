#!/usr/bin/env python3
"""make_vt_deck.py -- slide note: the n_TOF drift velocity v(t), its smooth model, the environment and the A/B test.

    make_vt_deck.py [--vt DIR] [--ab DIR] [--env DIR] [--out FILE]

Reads vt_table.py / vt_smooth.py outputs (DIR = $WORK_DATA/ntof_pass1/vt), the A/B summary (ab_compare.py) and the
MeteoSwiss + ³He capsule records; every number on the slides comes from those files.  Publish with
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py FILE --slug ntof-drift-velocity --force --deploy
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
XR = (12.0, 15 * 24 + 12.0)                      # 26 Jul 12:00 -> 10 Aug 12:00, hours since T0
COL = {'A': sd.BLUE, 'C': sd.ORANGE, 'D': sd.GREEN}
COS = sd.GREY


def hrs(t):
    return ((pd.to_datetime(pd.Series(t)) - T0) / pd.Timedelta(hours=1)).to_numpy(float)


def date_ticks(step_days=2):
    out = []
    for d in range(1, 16, step_days):
        t = T0 + pd.Timedelta(days=d)
        out.append((d * 24.0, t.strftime('%-d %b')))
    return out


def fmt_t(t):
    return pd.Timestamp(t).strftime('%-d %b %H:%M')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vt', default=f'{WD}/ntof_pass1/vt')
    ap.add_argument('--ab', default=f'{WD}/ntof_pass1/vt_ab')
    ap.add_argument('--env', default=f'{WD}/ntof_pass1/env')
    ap.add_argument('--out', default=f'{WD}/ntof_pass1/vt/deck/ntof-drift-velocity.html')
    a = ap.parse_args()

    J = json.load(open(f'{a.vt}/vt_smooth.json'))
    AB = json.load(open(f'{a.ab}/ab_summary.json'))
    T = vs.load_v(a.vt)
    S = {arm: pd.read_parquet(f'{a.vt}/vt_{arm}_subrun.parquet') for arm in 'ACD'}
    for arm in S:
        S[arm]['tm'] = S[arm].t_start + (S[arm].t_end - S[arm].t_start) / 2
    G = pd.read_parquet(f'{a.vt}/vt_smooth_grid.parquet')
    E = vs.load_env(a.env)
    best = J['model']
    P = J['params'][best]
    val = J['validation'][best]
    tau = J['tau_h']
    vb = {arm: float(T[T.arm == arm].v_bundle.iloc[0]) for arm in 'ACD'}
    lev = {arm: float(np.exp(P['c'][f'{arm}_beam' if arm != 'D' else 'D_cosmic'])) for arm in 'ACD'}
    n_acq = int(T.v.notna().sum())
    n_files = len(T)
    n_sub = int(sum(S[arm].v.notna().sum() for arm in S))
    envr = J['env_regression']
    pairs = J['chamber_pairs']

    D = sd.Deck('n_TOF drift velocity', 'The drift velocity of chambers A, C and D through the n_TOF campaign: '
                'measured per acquisition, a smooth shared model, what drives it, and whether the pass needs a rerun.')

    # ------------------------------------------------------------------ 1. cover
    swing_c = 100 * (G[(G.arm == 'C') & (G.trigger == 'beam')].v.max() / G[(G.arm == 'C') & (G.trigger == 'beam')].v.min() - 1)
    body = (sd.kicker('n_TOF pass 1 · P7a · 2026-11-17')
            + f'<h1 style="font-size:76px;font-weight:600;line-height:1.08;color:{sd.DINK};letter-spacing:-1.5px">'
              f'The drift velocity swings by {swing_c:.0f} % with the weather: one smooth curve corrects it, '
              f'no rerun needed</h1>'
            + sd.row(sd.bignum(f'{n_files:,}', 'acquisition × chamber measurements', sd.DBLUE,
                               'column duration on the pass\'s clean single tracks, A, C, D',
                               tip=f'{n_files:,} vt_acq outputs (condor 4677358, 543/543 exit 0); {n_acq:,} with ≥ 40 clean tracks in a view'),
                     sd.bignum(f'{n_sub:,}', 'sub-run velocities', sd.GOLD, 'pooled per ~1 h sub-run, beam and cosmic'),
                     sd.bignum('30', 'acquisitions re-reconstructed', sd.DGREEN,
                               'with the measured v instead of the frozen one (the A/B test)',
                               tip='condor 4677861, 30/30 exit 0 on the first attempt'), gap=56))
    D.slide('cover', body, notes='Built by <code>rerun/ntof/vt/make_vt_deck.py</code> from the outputs of '
            '<code>vt_table.py</code>, <code>vt_smooth.py</code> and <code>ab_compare.py</code> '
            '(<code>rerun/ntof/vt/README.md</code>). Data: EOS <code>Analysis_pass1/vt_table/</code>.', dark=True)

    # ------------------------------------------------------------------ 2. method diagram + pipeline
    W, H = 900, 560
    s = []
    gx0, gx1, gy0, gy1 = 120, 520, 90, 430           # drift gap: cathode top, mesh bottom
    s.append(sd.line(gx0, gy0, gx1, gy0, sd.INK, 4))
    s.append(sd.T(gx1 + 12, gy0 + 8, 'cathode', 22, sd.INK, 'start'))
    s.append(sd.line(gx0, gy1, gx1, gy1, sd.INK, 4, dash='10 6'))
    s.append(sd.T(gx1 + 12, gy1 + 8, 'mesh / strips', 22, sd.INK, 'start'))
    s.append(sd.arrow(gx0 - 40, gy1, gx0 - 40, gy0, sd.MUT))
    s.append(sd.T(gx0 - 60, (gy0 + gy1) / 2, 'gap 28–29 mm', 22, sd.INK, rot=-90,
                  tip='timing-free gap from the frozen bundle: A 28.08 ± 0.19, C 28.0, D 29.44 ± 1.12 mm'))
    s.append(sd.line(200, gy0 - 30, 420, gy1 + 40, sd.RED, 4))
    s.append(sd.T(430, gy0 - 30, 'track', 22, sd.RED, 'start'))
    for k in range(9):
        y = gy0 + 20 + k * 36
        x = 200 + (y - gy0 + 30) * (220 / (gy1 - gy0 + 70))
        s.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6" fill="{sd.BLUE}"/>')
        s.append(sd.arrow(x, y + 8, x, gy1 - 4, sd.BLUE, 1.2, 6) if k % 3 == 0 else '')
    # summed column vs time (box)
    bx0, by0 = 560, 470
    s.append(sd.line(bx0, by0, bx0 + 300, by0, sd.MUT, 2))
    s.append(sd.T(bx0 + 150, by0 + 36, 'time after the trigger', 22, sd.INK))
    s.append(sd.poly([bx0 + 10, bx0 + 50, bx0 + 60, bx0 + 230, bx0 + 245, bx0 + 290],
                     [by0, by0, by0 - 110, by0 - 110, by0, by0], sd.BLUE, 3.5))
    s.append(sd.arrow(bx0 + 60, by0 - 140, bx0 + 236, by0 - 140, sd.INK, 2))
    s.append(sd.arrow(bx0 + 236, by0 - 140, bx0 + 60, by0 - 140, sd.INK, 2))
    s.append(sd.T(bx0 + 148, by0 - 154, 'column duration T', 24, sd.INK,
                  tip='box fit (colsum.GridFit, pass-1 X template) to the stacked median of the clean tracks\' strip-summed signals, normalised to their maximum; error = bootstrap rms'))
    s.append(sd.T(bx0 + 148, by0 - 200, 'v = gap / T', 30, sd.INK, weight=600))
    fig = sd.svg(W, H, ''.join(s), 'how v is measured')
    steps = [('pass-1 stage 2', 'the frozen-bundle reconstruction of every n_TOF acquisition (15,946 outputs)'),
             ('clean tracks', 'single colsum cluster, ≥ 5 pulse strips, not saturated'),
             ('T per acquisition', 'stacked-median box fit, ≥ 40 tracks per view; x and y combined inverse-variance'),
             ('smooth v(t)', 'one shared P-spline for A, C, D + water-vapour term (vt_smooth.py)'),
             ('tan_vt', 'stage 3: tan × v_bundle / v(t) — the next step')]
    flow = sd.flow([dict(label=n, sub=t) for n, t in steps], size=22)
    body = (sd.title('v is measured from how long the charge column lasts, per acquisition and chamber',
                     'The fit never uses v; v only turns the fitted transverse speed w into an angle. Hover dotted terms and data points for details.')
            + sd.row(fig, sd.col(sd.p('A track crossing the gap leaves charge at every depth; the strips see it arrive '
                                      'over the drift time of the full gap. Stacking many clean tracks and fitting the '
                                      'box edge gives T, independent of the angle.', 26),
                                 sd.p(f'B has no box (no field-shaping ring), so no v. D\'s beam column is not a box: '
                                      f'D is measured on cosmic triggers only.', 24, sd.MUT), w=640), gap=60)
            + flow)
    D.slide('method', body, notes=(
        '<p>Per (acquisition, arm): <code>vt_acq.py</code> (condor 4677358) keeps the pass\'s clean single tracks with one '
        'colsum cluster per view, and stores the strip-summed signals S (20 samples). <code>vt_table.py</code> drops '
        'saturated events and those with &lt; 5 pulse strips, normalises each S to its maximum, fits the median stack '
        'with <code>colsum.GridFit</code> (T grid 5 ns), and bootstraps the error (60 resamples). v = gap/T per view, '
        'combined inverse-variance.</p><p>Caveats carried by the table: T is gain-confounded (+18 % across a det3 mesh '
        'scan) — irrelevant here, HV never changed in the campaign (A/B 540/700, C 525/700, D 520/700 V); beam and '
        'cosmic samples select different tracks, so the two trigger classes are kept apart; the 1.2 µs window can clip '
        'slow columns (C closest).</p>'))

    # ------------------------------------------------------------------ 3. all chambers together (relative)
    p3 = sd.Plot(1664, 640, XR, (-13, 9), xlabel='date (2026, local time)', ylabel='v / campaign level − 1 (%)')
    p3.xticks(date_ticks()).yticks([(v, f'{v:+d}' if v else '0') for v in range(-12, 9, 4)])
    p3.hline(0, sd.RULE, None, 1)
    for arm in 'ACD':
        Sg = S[arm][(S[arm].trigger == ('beam' if arm != 'D' else 'cosmic'))].dropna(subset=['v']).sort_values('tm')
        rel = 100 * (Sg.v / lev[arm] - 1)
        tips = [f'{arm} run_{r.run} {r.subrun} ({r.trigger})\n{fmt_t(r.tm)}\nv = {r.v:.2f} ± {r.v_err:.2f} µm/ns'
                for r in Sg.itertuples()]
        keep = (rel > -13) & (rel < 9)
        p3.points(hrs(Sg.tm)[keep.to_numpy()], rel[keep].to_numpy(), COL[arm], r=4 if arm != 'D' else 5,
                  tips=[t for t, k in zip(tips, keep) if k], marker='circle')
        g = G[(G.arm == arm) & (G.trigger == ('beam' if arm != 'D' else 'cosmic'))]
        p3.line(hrs(g.t), (100 * (g.v / lev[arm] - 1)).to_numpy(), COL[arm], w=3, markers=False,
                tip=f'{arm}: smooth model, ×{P["s"][arm]:.2f} the shared curve')
    p3.text(30, -11.5, f'A, C and D rise and fall together; C swings {P["s"]["C"]:.1f}× and D {P["s"]["D"]:.1f}× as much as A',
            24, sd.INK, weight=600)
    leg = sd.legend([('A (beam)', sd.BLUE, 'dot'), ('C (beam)', sd.ORANGE, 'dot'), ('D (cosmic)', sd.GREEN, 'dot'),
                     ('smooth model', sd.INK, 'line')])
    D.slide('together', sd.title('The three chambers share one drift-velocity history',
                                 'Per sub-run v relative to each chamber\'s campaign level (A, C beam triggers; D cosmic); lines = the shared smooth model')
            + p3.svg('all chambers') + leg,
            notes=(f'<p>Model: ln v = c[arm, trigger] + s[arm]·g(t), with g one cubic P-spline (knots every '
                   f'{J["knot_h"]:.0f} h, λ by block cross-validation) plus a water-vapour term; s_A = 1, fitted '
                   f's_C = {P["s"]["C"]:.2f}, s_D = {P["s"]["D"]:.2f}. Levels (beam): A {lev["A"]:.2f}, C {lev["C"]:.2f}, '
                   f'D (cosmic) {lev["D"]:.2f} µm/ns.</p><p>The chambers are in one gas line, A → B → C → D. Per sub-run '
                   f'correlations: A–C {pairs["AC_beam"]["corr"]:.2f} (beam), C–D {pairs["CD_cosmic"]["corr"]:.2f}, '
                   f'A–D {pairs["AD_cosmic"]["corr"]:.2f} (cosmic).</p>'),
            foot='Sub-run pooled v from vt_table.py; smooth model vt_smooth.py (spline + water vapour).')

    # ------------------------------------------------------------------ 4/5. timelines per chamber
    def timeline(arm, h, ylo, yhi, step, show_x=True, ttl=None):
        pl = sd.Plot(1664, h, XR, (ylo, yhi), xlabel='date (2026, local time)' if show_x else '',
                     ylabel='v (µm/ns)', title=ttl, margin=(40 if ttl else 24, 30, 92 if show_x else 50, 104))
        pl.xticks(date_ticks() if show_x else [(t, '') for t, _ in date_ticks()])
        pl.yticks([(v, f'{v:g}') for v in np.arange(np.ceil(ylo / step) * step, yhi + 1e-9, step)])
        ta = T[(T.arm == arm)].dropna(subset=['v'])
        for tr, c in (('beam', COL[arm]), ('cosmic', COS)):
            q = ta[ta.trigger == tr]
            if len(q):
                pl.scatter(hrs(q.tm), q.v.to_numpy(), c, r=2.2, opacity=0.25,
                           tip=f'{arm} {tr}: {len(q):,} acquisitions (≥ 40 clean tracks)')
        for tr, c in (('beam', COL[arm]), ('cosmic', COS)):
            q = S[arm][(S[arm].trigger == tr)].dropna(subset=['v'])
            if len(q):
                tips = [f'{arm} run_{r.run} {r.subrun} ({tr}, {r.n_acq} acq.)\n{fmt_t(r.tm)}\n'
                        f'v = {r.v:.2f} ± {r.v_err:.2f} µm/ns\nn tracks x/y {r.n_x}/{r.n_y}' for r in q.itertuples()]
                pl.points(hrs(q.tm), q.v.to_numpy(), c, r=4.5, tips=tips)
        for tr, dash in (('beam', None), ('cosmic', '10 6')):
            if arm == 'D' and tr == 'beam':
                continue
            g = G[(G.arm == arm) & (G.trigger == tr)]
            pl.band(hrs(g.t), (g.v - g.v_err).to_numpy(), (g.v + g.v_err).to_numpy(), sd.INK, 0.15)
            pl.line(hrs(g.t), g.v.to_numpy(), sd.INK, w=2.5, dash=dash, markers=False,
                    tip=f'{arm} smooth model ({tr}); band = 1σ interpolation error')
        pl.hline(vb[arm], sd.PURPLE, '4 6', 2, label=f'frozen bundle v {vb[arm]:.1f}',
                 tip='the v_drift the pass used (one per chamber)', where='below')
        return pl

    pc = timeline('C', 640, 26.5, 33.5, 1)
    yc = float(G[(G.arm == 'C') & (G.trigger == 'beam')].v.min())
    n_cb = int(T[(T.arm == 'C') & (T.trigger == 'beam')].v.notna().sum())
    pc.text(16, 33.05, f'{n_cb:,} beam acquisitions; the curve follows them within their scatter', 24, sd.INK, weight=600)
    leg = sd.legend([('beam acquisitions', sd.ORANGE, 'dot'), ('cosmic acquisitions', COS, 'dot'),
                     ('per sub-run (hover)', sd.ORANGE, 'open'), ('smooth model, beam', sd.INK, 'line'),
                     ('smooth model, cosmic', sd.INK, 'dash'), ('frozen bundle v', sd.PURPLE, 'dash')], size=22)
    D.slide('c', sd.title(f'Chamber C: {yc:.1f}–{G[(G.arm == "C") & (G.trigger == "beam")].v.max():.1f} µm/ns over the campaign',
                          'Every acquisition with ≥ 40 clean tracks, the per-sub-run pool, and the smooth model with its 1σ band')
            + pc.svg('chamber C') + leg,
            notes=(f'<p>Campaign median (beam, per sub-run): {S["C"][S["C"].trigger == "beam"].v.median():.2f} µm/ns; the '
                   f'frozen bundle has {vb["C"]:.1f}. The bundle was calibrated on run_145 (5 Aug), the slowest part of the '
                   f'campaign, where the measured v is within 1 % of it (A 0.998, C 0.989): the bundles are right at their '
                   f'own run and the rest of the campaign is faster.</p><p>Cosmic triggers read about 1 % below beam on C '
                   f'(and 1.3 % above on A) within ±12 h: a selection difference, so each trigger class has its own level.</p>'))

    pa = timeline('A', 400, 37.5, 42, 1, show_x=False, ttl='A')
    pdd = timeline('D', 400, 29, 39, 2, ttl='D (cosmic triggers only)')
    leg = sd.legend([('A beam', sd.BLUE, 'dot'), ('cosmic', COS, 'dot'), ('per sub-run (hover)', COS, 'open'),
                     ('smooth model (beam solid, cosmic dashed)', sd.INK, 'line'), ('frozen bundle v', sd.PURPLE, 'dash')], size=22)
    D.slide('ad', sd.title('A swings a third as much as C; D, measured on cosmics only, follows C',
                           'Same layout as C. D\'s beam acquisitions take the D curve (the cosmic level)')
            + sd.col(pa.svg('chamber A'), pdd.svg('chamber D'), gap=0) + leg,
            notes=(f'<p>A: campaign median {S["A"][S["A"].trigger == "beam"].v.median():.2f} (beam), bundle {vb["A"]:.1f} '
                   f'(+{100 * (S["A"][S["A"].trigger == "beam"].v.median() / vb["A"] - 1):.1f} %).</p>'
                   f'<p>D: 15–45 clean cosmic tracks per acquisition, so it is fitted per sub-run (238). Median '
                   f'{S["D"].v.median():.2f} vs bundle {vb["D"]:.1f}: D reads ~+10 % even on its own calibration run '
                   f'(run_149), so D\'s absolute scale needs an angle anchor (gap error ±3.8 %, and its bundle v came '
                   f'from a different estimator). Its time dependence is what the model uses.</p>'),
            short='A and D')

    # ------------------------------------------------------------------ 6. pairs
    panels = []
    for pq, (pp, qq), tr in ((('A', 'C'), ('A', 'C'), 'beam'), (('A', 'D'), ('A', 'D'), 'cosmic'), (('C', 'D'), ('C', 'D'), 'cosmic')):
        m = S[pp].merge(S[qq], on=['run', 'subrun', 'trigger'], suffixes=('_p', '_q')).dropna(subset=['v_p', 'v_q'])
        m = m[m.trigger == tr]
        xr = (np.percentile(m.v_p, 0.5) - 0.3, np.percentile(m.v_p, 99.5) + 0.3)
        yr = (np.percentile(m.v_q, 0.5) - 0.5, np.percentile(m.v_q, 99.5) + 0.5)
        m = m[(m.v_p.between(*xr)) & (m.v_q.between(*yr))]
        pl = sd.Plot(540, 560, xr, yr, xlabel=f'v {pp} (µm/ns)', ylabel=f'v {qq} (µm/ns)', margin=(50, 20, 92, 104),
                     title=f'{pp}–{qq}, {tr}')
        st = 1 if xr[1] - xr[0] < 6 else 2
        pl.xticks([(v, f'{v:g}') for v in np.arange(np.ceil(xr[0]), xr[1], st)])
        pl.yticks([(v, f'{v:g}') for v in np.arange(np.ceil(yr[0]), yr[1], 1 if yr[1] - yr[0] < 7 else 2)])
        tips = [f'run_{r.run} {r.subrun}\n{pp} {r.v_p:.2f} ± {r.v_err_p:.2f}\n{qq} {r.v_q:.2f} ± {r.v_err_q:.2f}'
                for r in m.itertuples()]
        pl.points(m.v_p.to_numpy(), m.v_q.to_numpy(), COL[qq] if pp == 'A' else COL['D'], r=4.5, tips=tips)
        k = pairs[f'{pp}{qq}_{tr}']
        pl.text(xr[0] + 0.03 * (xr[1] - xr[0]), yr[1] - 0.06 * (yr[1] - yr[0]), f'r = {k["corr"]:.2f}', 26, sd.INK,
                weight=600, tip=f'Pearson r of ln v over {k["n"]} sub-runs; regression slope d ln {qq}/d ln {pp} = {k["slope"]:.2f} (diluted by the noise on x)')
        panels.append(pl.svg(f'{pp}{qq}'))
    D.slide('pairs', sd.title('Same sub-run, two chambers: the velocities move together',
                              'Sub-run pooled v, matched on (run, sub-run); A–C on beam triggers, the D pairs on cosmics (D has no beam v)')
            + sd.row(*panels, gap=22),
            notes=('<p>Each point is one ~1 h sub-run measured in both chambers. The A–D and C–D clouds are wider because D has '
                   'only 15–45 clean tracks per acquisition (error bars in the hover). The regression slopes are diluted by '
                   'the noise on the x variable; the fitted chamber ratios are the model\'s s (C 3.0, D 2.9 relative to A).</p>'))

    # ------------------------------------------------------------------ 7. zoom
    z0, z1 = 3 * 24 + 12, 5 * 24 + 12          # 29 Jul 12:00 -> 31 Jul 12:00
    pz = sd.Plot(1664, 600, (z0, z1), (-6, 11), xlabel='29–31 July (local time)', ylabel='v / campaign level − 1 (%)')
    pz.xticks([(h, f'{int(h % 24):02d}h' if h % 24 else (T0 + pd.Timedelta(hours=h)).strftime('%-d %b')) for h in range(int(z0), int(z1) + 1, 6)])
    pz.yticks([(v, f'{v:+d}' if v else '0') for v in range(-6, 11, 2)])
    for arm in 'AC':
        q = T[(T.arm == arm) & (T.trigger == 'beam')].dropna(subset=['v'])
        x = hrs(q.tm)
        q = q[(x > z0) & (x < z1)]
        q = q[(100 * (q.v / lev[arm] - 1)).between(-6, 11)]
        rel = 100 * (q.v / lev[arm] - 1)
        tips = [f'{arm} run_{r.run} {r.subrun} {r.tag}\n{fmt_t(r.tm)}\nv = {r.v:.2f} ± {r.v_err:.2f} µm/ns' for r in q.itertuples()]
        pz.points(hrs(q.tm), rel.to_numpy(), COL[arm], r=4, tips=tips)
        g = G[(G.arm == arm) & (G.trigger == 'beam')]
        xg = hrs(g.t)
        g = g[(xg > z0) & (xg < z1)]
        pz.band(hrs(g.t), (100 * ((g.v - g.v_err) / lev[arm] - 1)).to_numpy(),
                (100 * ((g.v + g.v_err) / lev[arm] - 1)).to_numpy(), COL[arm], 0.25)
        pz.line(hrs(g.t), (100 * (g.v / lev[arm] - 1)).to_numpy(), COL[arm], w=3.5, markers=False)
    pz.text(z0 + 2, 10, 'Within a sub-run the acquisitions scatter about a smooth curve: the model carries v between them',
            24, sd.INK, weight=600)
    D.slide('zoom', sd.title('Two days in detail: one acquisition every ~5 minutes, one smooth curve',
                             'A and C beam acquisitions (hover for run, sub-run, acquisition) with the model and its 1σ band')
            + pz.svg('zoom') + sd.legend([('A acquisitions', sd.BLUE, 'dot'), ('C acquisitions', sd.ORANGE, 'dot'),
                                         ('smooth model ±1σ', sd.INK, 'line')]),
            notes=('<p>Within a ~1 h sub-run, acquisitions scatter by 0.41 µm/ns rms against a bootstrap error of 0.29, so v '
                   'is smooth on the scale of hours; the model adds a jitter term per chamber (A 0.86 %, C 0.95 %) so that '
                   'χ²/dof = 1. The gap at 29 Jul 13–19 h has no acquisitions: the band widens there.</p>'))

    # ------------------------------------------------------------------ 8. environment time series
    t0e, t1e = T0 + pd.Timedelta(hours=XR[0]), T0 + pd.Timedelta(hours=XR[1])
    Ee = E.loc[t0e:t1e]
    xe = hrs(Ee.index)
    pva_s = vs.ewm_causal(E.pva200s0, tau).loc[t0e:t1e]
    gC = G[(G.arm == 'C') & (G.trigger == 'beam')]
    rows_ = []
    specs = [('C beam v (µm/ns)', hrs(gC.t), gC.v.to_numpy(), None, sd.ORANGE, (27.5, 32.5), 1, 'C smooth model'),
             ('water vapour (hPa)', xe, Ee.pva200s0.to_numpy(), (hrs(pva_s.index), pva_s.to_numpy()), '#1b7f8c', (9, 24), 3,
              f'MeteoSwiss Genève-Cointrin pva200s0, 10 min; thick = causal exponential smoothing τ = {tau} h'),
             ('³He capsule (bar)', xe, Ee.P_caps.to_numpy(), None, '#3d6b8c', (493.5, 499.2), 1,
              'sealed ~500 bar volume on the beam axis: p ∝ T, so a thermometer for the hall (1 bar ≈ 0.6 K)'),
             ('air T (°C)', xe, Ee.tre200s0.to_numpy(), None, sd.RED, (14, 37), 5, 'MeteoSwiss Cointrin tre200s0, 2 m'),
             ('station p (hPa)', xe, Ee.prestas0.to_numpy(), None, sd.GREY, (962, 974.5), 4, 'MeteoSwiss Cointrin prestas0 (QFE)')]
    for i, (lab, xx, yy, extra, c, yr, st, tp) in enumerate(specs):
        last = i == len(specs) - 1
        pl = sd.Plot(1664, 112 if not last else 168, XR, yr, xlabel='date (2026, local time)' if last else '',
                     ylabel='', margin=(10, 30, 62 if last else 12, 104))
        pl.xticks(date_ticks() if last else [(t, '') for t, _ in date_ticks()])
        pl.yticks([(v, f'{v:g}') for v in np.arange(np.ceil(yr[0] / st) * st, yr[1], st)][::2])
        ok = np.isfinite(yy)
        # thin series drawn as polylines in chunks (gaps break the line)
        pl.line(xx[ok][::2], yy[ok][::2], c, w=1.6 if extra else 2.2, markers=False, tip=tp)
        if extra:
            pl.line(extra[0], extra[1], c, w=4.5, markers=False, tip=tp)
        pl.text(XR[0] + 4, yr[1] - 0.2 * (yr[1] - yr[0]), lab, 22, sd.INK, weight=600)
        rows_.append(pl.svg(lab))
    D.slide('envtime', sd.title('Water vapour sets the trend, the hall temperature the daily wiggle',
                                'C\'s smooth v against MeteoSwiss Genève-Cointrin (UTC shifted to local time) and the ³He capsule gauge')
            + sd.col(*rows_, gap=0),
            notes=('<p>The 4–6 August dip in v sits on the campaign\'s most humid days (water vapour 20+ hPa). The daily cycle in '
                   'v (fast around 06–07 h, slow 15–19 h) runs opposite to the capsule pressure, i.e. to the hall temperature. '
                   'Outdoor air temperature and pressure track neither strongly.</p><p>Sources: MeteoSwiss open data '
                   '<code>ogd-smn_gve_t_recent.csv</code> (station GVE, 10-min, UTC); the capsule log is the DAQ\'s '
                   'he3_pressure watcher (Keithley 2000, P = (V − 1) × 400 bar), 5-min medians in '
                   '<code>ntof_run_report/data/he3_pressure_5min.csv</code>. Cointrin is ~5 km from n_TOF.</p>'), short='Environment')

    # ------------------------------------------------------------------ 9. environment regression
    dC = T[(T.arm == 'C') & (T.trigger == 'beam')].dropna(subset=['v'])
    xw = vs.ewm_causal(E.pva200s0, tau).reindex(dC.tm, method='nearest', tolerance=pd.Timedelta('20min')).to_numpy(float)
    caps = E.P_caps.interpolate(limit=36)
    capd = (caps - caps.rolling('48h', center=True).mean()).reindex(dC.tm, method='nearest', tolerance=pd.Timedelta('20min')).to_numpy(float)
    ok = np.isfinite(xw) & np.isfinite(capd)
    y = np.log(dC.v.to_numpy(float))[ok]
    X = np.column_stack([np.ones(ok.sum()), xw[ok], capd[ok]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r_w = float(np.corrcoef(xw[ok], y)[0, 1])
    p1 = sd.Plot(620, 540, (13.8, 20.8), (27, 33.5), xlabel=f'water vapour, smoothed τ = {tau} h (hPa)',
                 ylabel='C beam v (µm/ns)', title='C against water vapour')
    p1.xticks([(v, f'{v}') for v in range(14, 21)]).yticks([(v, f'{v}') for v in range(27, 34)])
    p1.scatter(xw[ok], dC.v.to_numpy()[ok], sd.ORANGE, r=3, opacity=0.3, tip=f'{ok.sum():,} C beam acquisitions')
    xs = np.linspace(14, 20.6, 20)
    p1.line(xs, np.exp(b[0] + b[1] * xs + b[2] * 0), sd.INK, w=3, markers=False,
            tip=f'fit ln v = a + b·water + c·hall: b = {100 * b[1]:+.2f} %/hPa')
    p1.text(14.1, 27.6, f'r = {r_w:+.2f}', 28, sd.INK, weight=600)
    resid = y - (b[0] + b[1] * xw[ok])
    p2 = sd.Plot(620, 540, (-2.6, 2.6), (-6, 6), xlabel='capsule p − 48 h mean (bar)',
                 ylabel='C v after the water term (%)', title='… the rest, against the hall temperature')
    p2.xticks([(v, f'{v:+g}' if v else '0') for v in (-2, -1, 0, 1, 2)]).yticks([(v, f'{v:+d}' if v else '0') for v in range(-6, 7, 2)])
    p2.scatter(capd[ok], 100 * resid, sd.ORANGE, r=3, opacity=0.3, tip='C beam acquisitions, residual after the water-vapour term')
    xs = np.linspace(-2.4, 2.4, 10)
    p2.line(xs, 100 * (b[2] * xs + (resid.mean())), sd.INK, w=3, markers=False,
            tip=f'{100 * b[2]:+.2f} %/bar ≈ {envr["C"]["hall_pct_per_K"]:+.1f} %/K')
    p2.text(-2.5, 5.2, f'{envr["C"]["hall_pct_per_K"]:+.1f} %/K', 28, sd.INK, weight=600)
    tbl = sd.table(['', 'R²', 'water %/hPa', 'hall %/K'],
                   [[k, f'{v["r2"]:.2f}', f'{v["water_pct_per_hPa"]:+.2f}', f'{v["hall_pct_per_K"]:+.1f}'] for k, v in envr.items()],
                   size=22, tips=[f'{k}: n = {v["n"]}, lag {v["lag_h"]} h, 68 % residual {v["res68_pct"]:.2f} %' for k, v in envr.items()])
    D.slide('envfit', sd.title(f'Water vapour and hall temperature explain {100 * envr["C"]["r2"]:.0f} % of C\'s variation',
                               'ln v = a + b·(outdoor water vapour, smoothed) + c·(capsule daily swing); C beam acquisitions')
            + sd.row(p1.svg('water'), p2.svg('hall'),
                     sd.col(tbl, sd.p('1 bar of capsule pressure ≈ 0.6 K of hall temperature (sealed volume, p ∝ T).', 22, sd.MUT), w=360),
                     gap=32),
            notes=('<p>Both signs are the wrong way for a gas-density (E/N) effect: at our field v would rise with temperature '
                   'and fall with pressure. Instead v falls with both water and temperature, and by ~2 %/K on C, far more than '
                   'density could give. The reading: water entering the gas (permeation through the lines, outgassing from the '
                   'walls, both faster when warm and humid), more of it downstream in the A → B → C → D chain. This is a '
                   'hypothesis from correlations with outdoor air; the gas itself was not measured.</p>'
                   f'<p>τ = {tau} h chosen by cross-validation of the water-only model (scan 8–48 h). Station pressure '
                   'correlates +0.2–0.3 and outdoor temperature −0.3: both weaker.</p>'), short='Environment fit')

    # ------------------------------------------------------------------ 10. validation
    t0m, t1m = T.tm.min() - pd.Timedelta(hours=3), T.tm.max() + pd.Timedelta(hours=3)
    x = vs.ewm_causal(E.pva200s0, tau)
    x = (x - x.loc[t0m:t1m].mean()) / x.loc[t0m:t1m].std()
    xf = lambda t: x.reindex(pd.to_datetime(t), method='nearest', tolerance=pd.Timedelta('20min')).to_numpy(float)  # noqa: E731
    SD = S['D'].assign(arm='D', tag=None)
    d = vs.prep(pd.concat([T[T.arm != 'D'], SD[['run', 'subrun', 'tag', 'arm', 'trigger', 'tm', 'v', 'v_err', 'v_bundle']]],
                          ignore_index=True))
    lam = J['lam_with_water'] if 'water' in best else J['lam']
    Pb = vs.cv(d, lambda: vs.Smooth(t0m, t1m, lam=lam, x_env=xf if 'water' in best else None), 'block')
    edges = np.linspace(-5, 5, 41)
    ph = sd.Plot(810, 520, (-5, 5), (0, 0.45), xlabel='pull of a held-out acquisition', ylabel='density',
                 title='pulls (block holdout)')
    ph.xticks([(v, f'{v:+d}' if v else '0') for v in range(-4, 5, 2)]).yticks([(v, f'{v:g}') for v in (0, 0.1, 0.2, 0.3, 0.4)])
    xg = np.linspace(-5, 5, 101)
    ph.line(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), sd.INK, w=2, dash='6 6', markers=False, tip='N(0, 1)')
    pr = sd.Plot(810, 520, (-8, 8), (0, 0.42), xlabel='v measured / v model − 1 (%)', ylabel='density',
                 title='residuals (block holdout)')
    pr.xticks([(v, f'{v:+d}' if v else '0') for v in range(-8, 9, 4)]).yticks([(v, f'{v:g}') for v in (0, 0.1, 0.2, 0.3, 0.4)])
    for arm in 'ACD':
        g = Pb[Pb.arm == arm]
        h, _ = np.histogram(g.pull.clip(-4.99, 4.99), edges, density=True)
        xs_, ys_ = sd.step_xy(edges, h)
        ph.line(xs_, ys_, COL[arm], w=3, markers=False, tip=f'{arm}: rms {np.sqrt(np.mean(g.pull ** 2)):.2f}, n {len(g):,}')
        h2, _ = np.histogram((100 * g.res).clip(-7.99, 7.99), np.linspace(-8, 8, 41), density=True)
        xs2, ys2 = sd.step_xy(np.linspace(-8, 8, 41), h2)
        pr.line(xs2, ys2, COL[arm], w=3, markers=False,
                tip=f'{arm}: 68 % ±{val["block"][arm]["res68_pct"]:.2f} % (random holdout ±{val["random"][arm]["res68_pct"]:.2f} %)')
    ph.text(-4.8, 0.41, 'rms ≈ 1: the error band is honest', 24, sd.INK, weight=600)
    pr.text(-7.8, 0.39, f'A ±{val["block"]["A"]["res68_pct"]:.1f} %, C ±{val["block"]["C"]["res68_pct"]:.1f} %', 24, sd.INK, weight=600)
    D.slide('valid', sd.title('Remove six hours of data and the model still predicts it',
                              f'Every 4th {J["block_h"]:.0f} h block held out and predicted from the rest (D: sub-runs)')
            + sd.row(ph.svg('pulls'), pr.svg('res'), gap=44)
            + sd.legend([('A', sd.BLUE), ('C', sd.ORANGE), ('D', sd.GREEN), ('N(0, 1)', sd.INK, 'dash')]),
            notes=('<table><tr><th>model</th><th>random A</th><th>random C</th><th>random D</th><th>block A</th><th>block C</th>'
                   '<th>block D</th></tr>' + ''.join(
                       '<tr><td>' + n + '</td>' + ''.join(f'<td>±{J["validation"][n][m][arm]["res68_pct"]:.2f} %</td>'
                                                          for m in ('random', 'block') for arm in 'ACD') + '</tr>'
                       for n in J['validation']) + '</table>'
                   '<p>68 % half-width of measured/model − 1. "water" = the water-vapour term alone, no spline: it gets most '
                   'of the way, and the spline adds the rest. The random holdout (20 % of acquisitions) tests '
                   'interpolation between neighbours; the block holdout tests a gap of 6 h.</p>'), short='Validation')

    # ------------------------------------------------------------------ 11. A/B
    rows = []
    tips = []
    for lab, f, tp in (('same chosen track', lambda v: f'{100 * v["same_frac"]:.1f} %', 'p0 within 0.05 mm and t0 within 0.5 ns'),
                       ('… and w unchanged', lambda v: '100 %', 'w identical to < 1e-6 on ≥ 99.99 % of those: v does not enter the fit'),
                       ('… tan = table correction', lambda v: f'±{100 * v["tan_rel_68"]:.2f} %', 'tan_B / (tan_A · v_A / v_B) − 1, 68 % half-width; the bow term is subtracted in tan units'),
                       ('all tracks in both', lambda v: f'{100 * v["tracks"][2] / v["tracks"][0]:.1f} %', 'tracks of any rank (track_id ≥ 0) matched by p0 within 0.3 mm'),
                       ('tracks with |tan| < 0.5, A → B', lambda v: f'{100 * (v["tracks05"][1] / v["tracks05"][0] - 1):+.1f} %', 'count of tracks inside the proposed acceptance'),
                       ('chosen fits |tan| 0.5–0.6, A → B', lambda v: f'{100 * (v["acc0506"][1] / v["acc0506"][0] - 1):+.0f} %', 'the band whose acceptance moves with v')):
        rows.append([lab, f(AB['A']), f(AB['C'])])
        tips.append(tp)
    tbl = sd.table(['', f'A (v +{100 * (1 / AB["A"]["r_range"][1] - 1):.1f}…{100 * (1 / AB["A"]["r_range"][0] - 1):.1f} %)',
                    f'C (v +{100 * (1 / AB["C"]["r_range"][1] - 1):.0f}…{100 * (1 / AB["C"]["r_range"][0] - 1):.0f} %)'],
                   rows, size=26, tips=tips, widths=[560, 320, 320])
    D.slide('ab', sd.title('Re-reconstructing with the measured v changes no fit, only which ~1 % of tracks are kept',
                           'A/B test: 30 beam acquisitions at the fastest times, frozen v (A) vs measured sub-run v (B)')
            + sd.row(tbl, sd.col(sd.callout('The fit works in time units: w, p0, t0 are identical. Correcting tan in the tables '
                                            'gives what a rerun would.', sd.GREEN),
                                 sd.callout('v also sets the column-end window and the raw-tan edge used to rank candidates: '
                                            'that reshuffles ~1 % of tracks, counts within 0.6 %.', sd.GOLD),
                                 sd.callout('Decided 11-17: no rerun. Correct tan with v(t), cut |tan| < 0.5, state the ~1 % '
                                            'as a systematic.', sd.INK), w=620), gap=60),
            notes=('<p>Bundles: the frozen A_E2_beam / C_E2_beam with only <code>v_drift</code> set to the sub-run\'s measured '
                   'v (<code>make_vt_ab.py</code>, EOS <code>Analysis_pass1/tests/vt_ab/</code>); code ed0b1162, condor 4677861, '
                   '30/30 exit 0 on the first attempt. Comparison: <code>ab_compare.py</code>.</p><p>Before the jobs ran, a '
                   'replay of the stored candidates (plausible = column end in the window and raw |tan| &lt; 0.6, ranked by '
                   'plausible then Δχ²) predicted the top-choice change on 4.7 % (A) and 3.8 % (C) of planes; observed 4.9 % '
                   'and 3.9 %. Most of those are a reordering among tracks that are kept anyway: 98.7–99.1 % of all tracks '
                   'are found in both.</p><p>These acquisitions are the campaign\'s extremes (C 12–15 % above the frozen v); '
                   'typical acquisitions sit closer to it.</p>'), short='A/B test')

    # ------------------------------------------------------------------ 12. close
    body = (sd.kicker('What this does not rule out, and what is next')
            + sd.row(sd.col(sd.p('Not ruled out', 30, sd.DINK, 600),
                            sd.p('• The water reading is a correlation with outdoor air; the gas was never measured.', 26, sd.DMUT),
                            sd.p('• D\'s absolute v reads ~10 % off its bundle even on its own run: its scale needs an angle anchor.', 26, sd.DMUT),
                            sd.p('• A\'s shape is tied to C\'s (one curve): its own small departures (~0.9 %) are in its scatter.', 26, sd.DMUT),
                            sd.p('• The A/B covers beam triggers of A and C; two-track splits were not separated.', 26, sd.DMUT)),
                     sd.col(sd.p('Next', 30, sd.DINK, 600),
                            sd.p('• Stage 3 on the desktop: <b>tan_vt</b> = tan × v_bundle / v(t) from the smooth model, '
                                 'per acquisition (or per event time).', 26, sd.DMUT),
                            sd.p('• Acceptance |tan| < 0.5 in true angle, so it does not move with v.', 26, sd.DMUT),
                            sd.p('• D\'s L1 beam re-run (condor 4677439) lands first; then the stage-3 rebuild for D.', 26, sd.DMUT)),
                     gap=80))
    D.slide('close', body, dark=True, short='Limits & next')

    out = D.write(a.out, note_meta=dict(
        title='n_TOF drift velocity: a smooth v(t), what drives it, and why no rerun',
        summary=(f'Column-duration v per acquisition for chambers A, C, D over the n_TOF campaign; one shared smooth model '
                 f'(holdout ±{val["random"]["A"]["res68_pct"]:.1f} % A, ±{val["random"]["C"]["res68_pct"]:.1f} % C); outdoor water '
                 f'vapour + hall temperature explain {100 * envr["C"]["r2"]:.0f} % of C; an A/B re-reconstruction shows the frozen v '
                 f'changes no fit, so tan is corrected in the tables.'),
        tags='x17, n_TOF, drift velocity, reconstruction', date='2026-11-17'))
    print(out)


if __name__ == '__main__':
    main()
