#!/usr/bin/env python3
"""
make_qsum_runaway_deck.py -- the large-q_sum study as a figure-first slide note.

The same results as report.html (make_qsum_runaway_report.py), told in slides
with hover tooltips and a Details drop-down per slide, for
dylan-neff.web.cern.ch/notes. Built with dylan-cern-site/scripts/slidedoc.py.
Reads the outputs of `qsum_runaway census` and `qsum_runaway refit`, so
rerunning after either changes numbers, charts and claims together.

    python -m sept26_prelim_analysis.make_qsum_runaway_deck [--out PATH]
    python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug qsum-runaway --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402,F401
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK)
from sept26_prelim_analysis import qsum_runaway as qr                  # noqa: E402

N_SAMP, SAMPLE_NS, K_BINS, BIN_NS = 20, 60.0, 18, 60.0
T_LAST = (N_SAMP - 1) * SAMPLE_NS
#: class colours, kept on every slide
COL = {'normal, t0 ≤ 300': GREY, 'normal, t0 > 300': GOLD,
       'runaway, space': ORANGE, 'runaway, time': RED}

G = dict(
    qsum='q_sum: the total fitted charge of one plane, the sum of the 18 depth-bin charges the '
         'NNLS returns. Stage-3 q_total = x_q_sum + y_q_sum.',
    nnls=('NNLS: non-negative least squares. At each trial (p0, w, t0) the fit solves for the '
          'charge in each depth bin with no bound other than q ≥ 0, no prior, no regularisation '
          '(wft.model.chi2_plane).'),
    column=('Column: the waveform, on every strip and sample of the window, that unit charge in '
            'one depth bin would produce. A bin whose pulse falls outside the window has a column '
            'of ~0.'),
    guarded=(f'Guarded fit: the production fit with one change. Inside every NNLS solve, columns whose '
             f'noise-weighted norm is below {qr.REL:g} of the largest are dropped. A probe, '
             f'not a proposed fix.'),
    late=f'Late: fitted t0 > {qr.LATE_T0:.0f} ns, the class boundary the census shows.',
    coinc='Coincident: coinc_this_arm = 1, the track’s own arm had a scintillator hit in the '
          'accept window.',
    flat=f'Flat: |tan θ| < {qr.FLAT_TAN}.',
    t0='t0: when charge from the mesh (zero drift depth) reaches the strips, in ns from the '
       'first sample.',
)


def load():
    od = qr.out_dir()
    need = ['census.meta.json', 'census_per_arm.csv', 'census_t0_hist.csv', 'refit.parquet',
            'refit.meta.json', 'examples.pkl', 'figures.meta.json']
    miss = [n for n in need if not (od / n).exists()]
    if miss:
        raise SystemExit(f'missing {miss} in {od}: run qsum_runaway census/refit and the figures')
    R = pd.read_parquet(od / 'refit.parquet')
    late = R.t0_prod > qr.LATE_T0
    R['grp'] = np.select([(R.cls == 'normal') & ~late, (R.cls == 'normal') & late,
                          R.cls == 'space', R.cls == 'time'], list(COL), 'other')
    R['dtan'] = (R.tan_theta_grd - R.tan_theta_prod).abs()
    R['dp0'] = (R.p0_grd - R.p0_prod).abs()
    R['dt0'] = (R.t0_grd - R.t0_prod).abs()
    with open(od / 'examples.pkl', 'rb') as f:
        E = pickle.load(f)
    pick = json.loads((od / 'figures.meta.json').read_text())
    ex = {}
    for k, v in pick.items():
        for e in E:
            m = e['meta']
            if (m['arm'], m['event_id'], m['plane'], m['win']) == (
                    v['arm'], v['event_id'], v['plane'], v['win']):
                ex[k] = e
    return dict(cm=json.loads((od / 'census.meta.json').read_text()),
                rm=json.loads((od / 'refit.meta.json').read_text()),
                pa=pd.read_csv(od / 'census_per_arm.csv').set_index('arm'),
                h=pd.read_csv(od / 'census_t0_hist.csv'), R=R, ex=ex)


def summ(R):
    out = {}
    for g, s in R.groupby('grp'):
        out[g] = dict(n=len(s), dtan50=s.dtan.median(), dtan05=(s.dtan > 0.05).mean(),
                      dp050=s.dp0.median(), dt0=(s.dt0 > 5).mean(),
                      flat_p=(s.tan_theta_prod.abs() < qr.FLAT_TAN).mean(),
                      flat_g=(s.tan_theta_grd.abs() < qr.FLAT_TAN).mean(),
                      dchi=((s.chi2_grd - s.chi2_prod) / s.dof_prod).median(),
                      qobs=(s.q_sum_grd / s.q_obs.where(s.q_obs > 0)).median())
    return out


# --------------------------------------------------------------------------- #
def s_cover(D, d, S):
    cm = d['cm']
    t = S['runaway, time']
    run_ = d['R'][d['R'].cls != 'normal']
    ftime = (run_.cls == 'time').mean()
    nums = ''.join([
        sd.bignum(sd.pct(cm['big_any']), 'gated tracks with a runaway plane', DGREY,
                  f'q_sum &gt; 10⁶ ADC in x or y, of {cm["n_gated"]:,} gated tracks, whole campaign.',
                  tip='Stage-3 full pass, all 36 runs. The rate is the same before and after the '
                      '27 July access.'),
        sd.bignum(sd.pct(cm['late_big']), 'late class: geometry wrong', DRED,
                  f'{sd.pct(cm["coinc_late_big"], 1)} of the scintillator-coincident tracks.',
                  tip=G['late'] + '\n' + G['coinc']),
        sd.bignum(f'{100 * t["flat_p"]:.0f} → {100 * t["flat_g"]:.0f}%', 'flat late tracks are an artefact',
                  '#f0a36b', 'Time-censored runaways fitted with |tan| &lt; 0.01, production → '
                  'without the invisible bins.', tip=G['flat'])])
    body = (sd.kicker(f'n_TOF 2026 · stage-3 full pass · refit on {d["rm"]["run"]} · '
                      f'{dt.date.today():%-d %b %Y}')
            + '<h1 style="font-size:80px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1640px">'
              'The impossible charges are an unbounded fit, and on late tracks they take the angle with them</h1>'
            + '<div style="flex:1"></div>'
            + f'<p style="font-size:28px;color:{DMUT}">{100 * ftime:.0f}&thinsp;% of runaways are depth bins '
              'that arrive after the readout window closes. The rest are bins the slope pushes off the strip window.</p>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, f'''
<p>On 2026-09-10 the tracking QA found that one gated track in four carries a fitted charge that cannot
be real: q_total above 10⁶ ADC, reaching 10³⁴, on a 12-bit ADC. Their χ² was unremarkable, so nothing had
caught them. This note finds the mechanism, splits the tail into the classes it actually contains, and
measures what each class does to the <b>geometry</b>. Nobody uses q_sum for an opening angle, but
everybody uses p0, tan θ and t0.</p>
<p>Two samples: the campaign census (every gated track of the stage-3 full pass, no refitting) and a refit
of {d["rm"]["n_fits"]:,} plane fits from {d["rm"]["run"]}/{d["rm"]["sub_run"]} tag {d["rm"]["tag"]},
{d["rm"]["n_per_arm"]} events on each of the four arms. Everything is on the post-23-July (noisy)
configuration except the pre-access runs of the census.</p>''',
            dark=True, short='Answer')


def s_setup(D, d):
    W, H = 1040, 640
    x0, x1 = 120, 1000
    tmax = 1700.0
    X = lambda t: x0 + (t + 100) / (tmax + 100) * (x1 - x0)   # noqa: E731
    t_peak = float(d['ex']['time']['t_peak']) if 'time' in d['ex'] else 140.0
    o = []
    o.append(f'<rect x="{X(0):.1f}" y="40" width="{X(T_LAST) - X(0):.1f}" height="470" fill="#e6eef8" '
             f'stroke="{BLUE}" stroke-width="2"{sd.tipattr(f"Readout window: {N_SAMP} samples × {SAMPLE_NS:.0f} ns, last sample at {T_LAST:.0f} ns (run_config: n_samples_per_waveform = {N_SAMP})")}/>')
    o.append(sd.T(X(T_LAST / 2), 72, f'readout window: {N_SAMP} samples', 24, BLUE, weight=600))
    for i in range(N_SAMP):
        o.append(sd.line(X(i * SAMPLE_NS), 500, X(i * SAMPLE_NS), 510, BLUE, 2))
    o.append(sd.line(X(T_LAST), 30, X(T_LAST), 520, RED, 3, '8 6'))
    o.append(sd.T(X(T_LAST) + 8, 30, f'last sample {T_LAST:.0f} ns', 21, RED, 'start'))
    rows = [(0.0, 'track at t0 = 0 (in time)', 150), (450.0, 'track at t0 = 450 ns (late)', 330)]
    for t0, lab, y in rows:
        o.append(sd.T(X(-100), y - 34, lab, 24, INK, 'start', 600))
        n_bad = 0
        for k in range(K_BINS):
            a = t0 + (k + 0.5) * BIN_NS
            bad = a + t_peak > T_LAST
            n_bad += bad
            c = RED if bad else GREEN
            tp = (f'depth bin {k}: charge arrives at {a:.0f} ns, its pulse peaks at {a + t_peak:.0f} ns.\n'
                  + ('After the last sample: only the leading edge is in the window, column ~10⁻⁶ or less.'
                     if bad else 'Peak inside the window: observable.'))
            o.append(f'<rect x="{X(a - BIN_NS / 2) + 2:.1f}" y="{y - 14}" width="{X(BIN_NS) - X(0) - 4:.1f}" '
                     f'height="40" rx="5" fill="{c}" fill-opacity="0.8"{sd.tipattr(tp)}/>')
        o.append(sd.T(X(t0 + K_BINS * BIN_NS), y - 34, f'{n_bad} of {K_BINS} invisible', 22,
                      RED if n_bad else GREEN, 'end', 600))
        o.append(sd.arrow(X(t0), y + 46, X(t0 + K_BINS * BIN_NS), y + 46, MUT, 2, 10))
        o.append(sd.T(X(t0 + K_BINS * BIN_NS / 2), y + 74, f'depth grid: {K_BINS} × {BIN_NS:.0f} ns = '
                      f'{K_BINS * BIN_NS:.0f} ns after t0', 20, MUT))
    for t in (0, 500, 1000, 1500):
        o.append(sd.T(X(t), 560, f'{t}', 21))
    o.append(sd.T((x0 + x1) / 2, 600, 'time from the first sample [ns]', 22, INK))
    side = sd.col(
        sd.p(f'The fit models every waveform as Σ<sub>k</sub> q<sub>k</sub> × '
             f'{sd.term("column", G["column"])}<sub>k</sub>, and solves for q by '
             f'{sd.term("NNLS", G["nnls"])} with no bound.', 27),
        sd.p('A bin whose column is 10⁻¹² inside the window explains a 1-ADC wiggle in the last sample '
             'with q = 10¹². The model waveform barely changes, so χ² stays normal, but '
             f'{sd.term("q_sum", G["qsum"])} explodes.', 26),
        sd.callout('The depth grid is fixed at 1080 ns after t0; the window ends at 1140 ns. '
                   'Any late track has bins the data cannot see.', RED, 26),
        sd.p('Hover the dotted terms, the bins and every data point for definitions and numbers.', 22, MUT),
        gap=26, w=560)
    body = sd.title('The fit offers charge in bins the window cannot see',
                    'The production depth grid against the readout window, for an in-time and a late track.')
    body += sd.row(sd.svg(W, H, ''.join(o), 'depth grid against readout window'), side, gap=52)
    D.slide('setup', body, f'''
<p>The window is <code>n_samples_per_waveform = {N_SAMP}</code> at {SAMPLE_NS:.0f} ns (run_145 run_config),
so the last sample is at {T_LAST:.0f} ns. The bundle's depth grid is <code>n_depth_bins = {K_BINS}</code>
at {BIN_NS:.0f} ns. The impulse response peaks {t_peak:.0f} ns after a bin's charge arrives, so a bin is
effectively invisible once its arrival time t0 + u<sub>k</sub> exceeds {T_LAST - t_peak:.0f} ns.</p>
<p>A second route gives the same unobservable column: the fitted slope w carries a deep bin's centre
p0 + w·u<sub>k</sub> a few mm past the edge of the strip window, where only the erf tail of its transverse
spread reaches a strip. That is the space-censored class, two slides on.</p>
<p>Code: <code>wft/model.py</code> <code>build_matrix</code> and <code>chi2_plane</code>. The same
edge is why <code>q_uend</code> rails at 1080 ns on half the gated tracks (STATUS, q_per_len withdrawn).</p>''',
            short='Mechanism')


def s_t0(D, d):
    h = d['h']
    P = sd.Plot(1100, 400, x=(-600, 1200), y=(0, 1), ylabel='fraction q_sum > 10⁶',
                margin=(24, 30, 40, 120))
    P.yticks([(v, f'{v:.0%}'.replace('%', ' %')) for v in (0, 0.25, 0.5, 0.75, 1.0)])
    P.vline(qr.LATE_T0, MUT, '8 6', 2, label=f'{qr.LATE_T0:.0f} ns', tip=G['late'])
    Q = sd.Plot(1100, 300, x=(-600, 1200), y=(0, 0.065), xlabel='fitted t0 [ns]',
                ylabel='share of tracks', margin=(16, 30, 92, 120))
    Q.xticks([(v, str(v)) for v in (-500, -250, 0, 250, 500, 750, 1000)])
    Q.yticks([(v, f'{100 * v:.0f} %') for v in (0, 0.02, 0.04, 0.06)])
    Q.vline(qr.LATE_T0, MUT, '8 6', 2)
    early, late = {}, {}
    for p, c in (('x', PURPLE), ('y', ORANGE)):
        s = h[h.plane == p].pivot(index='t0_lo', columns='big', values='n').fillna(0)
        s.columns = [str(c_) for c_ in s.columns]
        n = s.sum(1)
        ok = n > 2000
        f = (s['True'] / n)[ok]
        x = (s.index[ok] + 10).tolist()
        tips = [f'{p} plane, t0 {int(a - 10)}–{int(a + 10)} ns\n{int(k):,} of {int(m):,} tracks have q_sum > 10⁶ '
                f'({100 * v:.1f} %)' for a, v, k, m in zip(x, f, s['True'][ok], n[ok])]
        P.line(x, f.tolist(), c, 3, markers=False, tips=tips, tip=f'{p} plane')
        Q.line((s.index + 10).tolist(), (n / n.sum()).tolist(), c, 2.5, markers=False)
        early[p] = float(s['True'][s.index < qr.LATE_T0].sum() / n[s.index < qr.LATE_T0].sum())
        late[p] = float(s['True'][s.index >= qr.LATE_T0].sum() / n[s.index >= qr.LATE_T0].sum())
    leg = sd.legend([('x plane', PURPLE), ('y plane', ORANGE)])
    side = sd.col(
        sd.p(f'Below {qr.LATE_T0:.0f} ns: <b>{100 * min(early.values()):.0f}–{100 * max(early.values()):.0f} %</b>. '
             f'Above it: <b>{100 * min(late.values()):.0f}–{100 * max(late.values()):.0f} %</b>.', 28),
        sd.p('The step sits where the bulk of a late track’s depth grid has slid past the last sample. '
             'The floor below it is the space route, which does not care about t0.', 25),
        sd.callout('In-time tracks peak at t0 ≈ 0; the late population is a fifth of all tracks, '
                   'with spikes on a 60 ns comb.', GREY, 25),
        gap=26, w=480)
    body = sd.title('The runaway rate switches on with t0',
                    f'Every gated track of the stage-3 full pass ({d["cm"]["n_gated"]:,}), per plane, '
                    'in 20 ns bins of fitted t0.')
    body += sd.row(sd.col(leg, P.svg('runaway fraction vs t0'), Q.svg('t0 distribution'), gap=0, w=1110),
                   side, gap=60)
    D.slide('t0', body, '''
<p>Census, no refitting: <code>qsum_runaway census</code> reads <code>stage3_fullpass/tracks_campaign.parquet</code>,
gated tracks only. Bins with fewer than 2 000 tracks are not drawn.</p>
<p>The spikes in the t0 distribution at ~450 and ~510 ns are a 60 ns comb, one depth bin apart. That is the
signature of the near-degenerate t0 minima of the fit, which the runaway bins make worse, not of physics.
The flat-track pile-up two slides on lives in the same spikes.</p>''',
            foot='Source: qsum_runaway census → census_t0_hist.csv. Hover the curves for counts per bin.',
            short='Rate vs t0')


def s_anatomy(D, d):
    if 'time' not in d['ex']:
        return
    e = d['ex']['time']
    m = e['meta']
    arr = np.asarray(e['arr'])
    qp, qg, pk = np.asarray(e['q_prod']), np.asarray(e['q_grd']), np.asarray(e['peak'])
    floor = 1e-2
    P = sd.Plot(1000, 600, x=(arr.min() - 40, arr.max() + 40), y=(floor, 1e10, 'log'),
                xlabel='depth-bin arrival time t0 + u [ns]', ylabel='fitted charge per bin [ADC]')
    P.xticks([(v, str(v)) for v in range(500, 1600, 250) if arr.min() - 40 <= v <= arr.max() + 40])
    P.yticks([(10.0 ** k, f'10{sd.sup(k)}') for k in range(-2, 11, 2)])
    edge = e['t_last'] - e['t_peak']
    P.raw(f'<rect x="{P.X(edge):.1f}" y="{P.y0}" width="{P.X(arr.max() + 40) - P.X(edge):.1f}" '
          f'height="{P.ph}" fill="{RED}" fill-opacity="0.07"/>', back=True)
    P.vline(edge, RED, '8 6', 2, label='pulse peaks after the last sample',
            tip=f'arrival > {edge:.0f} ns: the pulse peak ({e["t_peak"]:.0f} ns after arrival) is past '
                f'the last sample at {e["t_last"]:.0f} ns')
    tp = [f'bin {k}: arrives {a:.0f} ns\nproduction q = {sd.sci(q) if q > 0 else "0"}\n'
          f'column peak = {sd.sci(c) if c > 0 else "0"} ADC per unit q' for k, (a, q, c) in
          enumerate(zip(arr, qp, pk))]
    tg = [f'bin {k}: guarded q = {sd.sci(q) if q > 0 else "0 (drawn at the floor)"}'
          for k, q in enumerate(qg)]
    P.line(arr.tolist(), np.maximum(qp, floor).tolist(), RED, 3.5, tips=tp, tip='production')
    P.line(arr.tolist(), np.maximum(qg, floor).tolist(), GREEN, 3, '10 7', marker='open', tips=tg,
           tip='guarded')
    P.line(arr.tolist(), np.maximum(pk * 1e3, floor).tolist(), MUT, 2.5, '3 6', markers=False,
           tip='column peak × 10³: the peak ADC one unit of charge in this bin puts in the window')
    leg = sd.legend([('production q', RED), ('guarded q', GREEN, 'dash'), ('column peak × 10³', MUT, 'dash')])
    ts = np.arange(len(e['W'][0])) * SAMPLE_NS
    i = int(np.argmax(np.asarray(e['W']).max(1)))
    w_ = np.asarray(e['W'])[i]
    Q = sd.Plot(560, 330, x=(0, ts[-1]), y=(min(w_.min(), 0) * 1.1, w_.max() * 1.15),
                xlabel='sample time [ns]', ylabel='ADC', title='brightest strip', margin=(20, 20, 84, 96))
    Q.xticks([(v, str(v)) for v in (0, 500, 1000)])
    Q.yticks([(v, f'{v:.0f}') for v in np.linspace(0, w_.max(), 3).round(-2)])
    Q.line(ts.tolist(), w_.tolist(), INK, 3, markers=False, tip='data')
    Q.line(ts.tolist(), np.asarray(e['model_prod'])[i].tolist(), RED, 2.5, markers=False, tip='production model')
    Q.line(ts.tolist(), np.asarray(e['model_grd'])[i].tolist(), GREEN, 2.5, '10 7', markers=False,
           tip='guarded model')
    cap = (f'Chamber {m["arm"]} {m["plane"]}, event {m["event_id"]}: q_sum {sd.sci(m["q_sum_prod"])} → '
           f'{m["q_sum_grd"]:,.0f} ADC. Both models describe the strip; the runaway charge never shows.')
    side = sd.col(Q.svg('brightest strip'), sd.p(cap, 24), gap=14, w=580)
    kq = int(np.argmax(qp))
    where = (f'where the column is 10{sd.sup(int(np.floor(np.log10(pk[kq]))))}' if pk[kq] > 0
             else 'where the column is zero')
    body = sd.title(f'One fit: {sd.sci(m["q_sum_prod"], 0)} ADC hidden {where}',
                    'Fitted charge per depth bin, against when that bin’s charge reaches the strips.')
    body += sd.row(sd.col(leg, P.svg('q per depth bin'), gap=6, w=1010), side, gap=60)
    D.slide('anatomy', body, f'''
<p>The real charge (the track's pulse near 1000 ns) sits in the bins arriving at ~870–930 ns, in both fits.
The production NNLS then puts {sd.sci(qp.max(), 1)} ADC in the bins arriving after ~1400 ns. Their column
peak is 10⁻¹⁰–10⁻¹⁶ ADC per unit charge, so the waveform they add is a fraction of an ADC count in the last
sample. Without those bins the guarded fit reproduces the same strip.</p>
<p>This example is the clearest member of its class (picked by the figure script as the largest q_sum ×
peak-amplitude among flat time-censored runaways), so it is a picture of the mechanism, not of a typical
member.</p>''',
            foot=f'Source: qsum_runaway refit → examples.pkl ({d["rm"]["run"]} tag {d["rm"]["tag"]}).',
            short='One fit')


def s_routes(D, d, S):
    R = d['R']
    run_ = R[R.cls != 'normal']
    rows = []
    for arm in sorted(R.arm.unique()):
        r = run_[run_.arm == arm]
        n = (R.arm == arm).sum()
        rows.append((arm, (r.cls == 'time').sum(), (r.cls == 'space').sum(), n))
    vmax = max(t + s for _a, t, s, _n in rows)
    bars = []
    for arm, t, s, n in rows:
        wt, ws = 760 * t / vmax, 760 * s / vmax
        bars.append(
            f'<div style="display:flex;align-items:center;gap:20px">'
            f'<p style="width:60px;font-size:30px;font-weight:600;text-align:right">{arm}</p>'
            f'<div{sd.tipattr(f"chamber {arm}: {t} time-censored runaways of {n} plane fits")} '
            f'style="width:{wt:.0f}px;height:60px;background:{RED};border-radius:6px 0 0 6px"></div>'
            f'<div{sd.tipattr(f"chamber {arm}: {s} space-censored runaways of {n} plane fits")} '
            f'style="width:{ws:.0f}px;height:60px;background:{ORANGE};border-radius:0 6px 6px 0;margin-left:-20px"></div>'
            f'<p style="font-size:24px;color:{MUT};white-space:nowrap">{t} + {s} of {n} fits</p></div>')
    ft = (run_.cls == 'time').mean()
    side = sd.col(
        sd.card(sd.p(f'<b style="color:{RED}">Time-censored · {100 * ft:.0f} %</b>', 28)
                + sd.p('The dominant bin’s pulse peaks after the last sample. Mostly late tracks '
                       f'({100 * (run_[run_.cls == "time"].t0_prod > qr.LATE_T0).mean():.0f} % have t0 &gt; '
                       f'{qr.LATE_T0:.0f} ns).', 24)),
        sd.card(sd.p(f'<b style="color:{ORANGE}">Space-censored · {100 * (1 - ft):.0f} %</b>', 28)
                + sd.p('The dominant bin arrives in time, but the slope carries its centre past the edge of '
                       'the strip window. Any t0; includes coherent-ringing windows that are not tracks.', 24)),
        gap=24, w=640)
    body = sd.title('Two routes to an invisible bin, the same on every chamber',
                    f'Runaway plane fits (q_sum &gt; 10⁶) by the route of their largest bin, '
                    f'{d["rm"]["run"]} refit sample.')
    body += sd.row(sd.col(*bars, sd.legend([('time-censored', RED, 'box'), ('space-censored', ORANGE, 'box')]),
                          gap=48, w=1000), side, gap=40, align='center')
    D.slide('routes', body, f'''
<p>For each runaway fit, the depth bin holding the most charge is classified. <b>Time</b> if its arrival
plus the {d["ex"].get("time", {}).get("t_peak", 140):.0f} ns impulse-response peak is after the last sample;
otherwise <b>space</b> if its centre p0 + w·u lies outside the strip window. No runaway fell in neither
class.</p>
<p>In the median runaway fit all but ~10⁻⁷ of q_sum sits in bins whose noise-weighted column is below
1 % of the largest. The visible charge Σ q<sub>k</sub> × column-peak<sub>k</sub> is ordinary.</p>
<p>The space class is a mix. Its most extreme member in the sample is a window of coherent bipolar ringing
(±200 ADC, ~350 ns period, on every strip at once) that the model hardly fits. How much of the class is
noise rather than steep tracks is not measured: χ² explained-fraction does not separate the classes.</p>''',
            foot='Source: qsum_runaway refit → refit.parquet, column cls. Hover the bars for counts.',
            short='Two routes')


def _cdf(v, n=70):
    v = np.sort(np.clip(np.asarray(v, float), 1e-4, 10.0))
    qs = np.linspace(0, 1, n)
    return np.quantile(v, qs), qs


def s_geometry(D, d, S):
    R = d['R']
    P = sd.Plot(960, 620, x=(1e-4, 10, 'log'), y=(0, 1), xlabel='|tan θ guarded − tan θ production|',
                ylabel='cumulative fraction of plane fits')
    P.xticks(sd.log_ticks(-4, 1)).yticks([(v, f'{100 * v:.0f} %') for v in (0, 0.25, 0.5, 0.75, 1)])
    P.vline(0.05, MUT, '8 6', 2, label='0.05', tip='|Δtan| = 0.05, about 3° near normal incidence')
    for g, c in COL.items():
        s = R[R.grp == g]
        if not len(s):
            continue
        x, y = _cdf(s.dtan)
        tips = [f'{g}: {100 * b:.0f} % of {len(s)} fits move by ≤ {a:.3g}' for a, b in zip(x, y)]
        P.line(x.tolist(), y.tolist(), c, 3.5, markers=False, tips=None, tip=f'{g} ({len(s)} fits)')
        P.points(x[::7].tolist(), y[::7].tolist(), c, 5, tips=tips[::7])
    rows, tips = [], []
    for g in COL:
        if g not in S:
            continue
        s = S[g]
        rows.append([f'<span style="color:{COL[g]};font-weight:600">{g}</span>', f'{s["n"]}',
                     f'{s["dtan50"]:.3f}', sd.pct(s['dtan05']), f'{s["dp050"]:.2f}', sd.pct(s['dt0'])])
        tips.append(f'{g}: median |Δp0| {s["dp050"]:.3f} mm; median Δχ²/dof {s["dchi"]:+.3f}')
    tbl = sd.table(['class', 'n', 'Δtan', '&gt;0.05', 'Δp0', 't0 moved'], rows, 22,
                   tips=tips, widths=[190, None, None, None, None, 110])
    side = sd.col(tbl, sd.p('Δtan, Δp0: median absolute shift (Δp0 in mm). &gt;0.05: share beyond 0.05 in tan. '
                            'Hover a row for Δχ²/dof.', 20, MUT),
                  sd.callout('Early normal fits do not move: the guard is a clean probe. '
                                  '<b>Half to two thirds of late fits move</b>, runaway or not.', RED, 25), gap=24, w=660)
    body = sd.title('Take the invisible bins away and the late tracks move',
                    f'Production against the {sd.term("guarded", G["guarded"])} fit, same windows, '
                    f'{len(R):,} plane fits on four chambers.')
    body += sd.row(P.svg('delta tan CDF'), side, gap=40)
    D.slide('geometry', body, f'''
<p>Each window is fitted twice from scratch: production, and the guarded fit, which is identical except that
every NNLS solve inside the minimisation drops columns below {qr.REL:g} of the largest. "t0 moved" is
|Δt0| &gt; 5 ns. Values below 10⁻⁴ are piled at the left edge, values above 10 at the right.</p>
<p>The late normal tracks move too ({sd.pct(S["normal, t0 > 300"]["dtan05"])} beyond 0.05). Their deep bins are
just as invisible; NNLS simply did not fill them past 10⁶. <b>The q_sum threshold is a symptom marker; the
class is t0.</b></p>
<p>The guard is not the right answer: on the time-censored class it is worse on χ² (median
{S["runaway, time"]["dchi"]:+.3f}/dof), because the truncated tail carries some real signal. What it
establishes is that the production geometry of late tracks <i>depends</i> on bins the data cannot
constrain.</p>''',
            short='Geometry')


def s_flat(D, d, S):
    R = d['R']
    L = R[R.t0_prod > qr.LATE_T0]
    bins = np.linspace(0, 1.5, 31)
    P = sd.Plot(1080, 620, x=(0, 1.5), y=(0, 0.5), xlabel='|tan θ| (raw; last bin = overflow)',
                ylabel='fraction of fits')
    P.xticks([(v, f'{v:g}') for v in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)])
    P.yticks([(v, f'{100 * v:.0f} %') for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)])
    series = [(L.cls == 'time', 'time-censored runaways', RED), (L.cls == 'normal', 'normal late tracks', GOLD)]
    for sel, lab, c in series:
        for fit, dash, nm in (('prod', None, 'production'), ('grd', '10 7', 'guarded')):
            v = L[sel][f'tan_theta_{fit}'].abs().clip(upper=1.499)
            hh, _ = np.histogram(v, bins)
            f = hh / max(hh.sum(), 1)
            xs, ys = sd.step_xy(bins.tolist(), f.tolist())
            P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], c, 3.5 if fit == 'prod' else 3, dash,
                          tip=f'{lab}, {nm} (n = {int(sel.sum())})'))
            ctr = (bins[:-1] + bins[1:]) / 2
            P.points(ctr[:1].tolist(), f[:1].tolist(), c, 7, tips=[
                f'{lab}, {nm}: {100 * f[0]:.1f} % in the first bin (|tan| < {bins[1]:.2f}); '
                f'{100 * (v < qr.FLAT_TAN).mean():.1f} % below {qr.FLAT_TAN}'],
                marker='circle' if fit == 'prod' else 'open')
    leg = sd.legend([('time-censored, production', RED), ('time-censored, guarded', RED, 'dash'),
                     ('normal late, production', GOLD), ('normal late, guarded', GOLD, 'dash')], 22)
    t = S['runaway, time']
    side = sd.col(
        sd.p(f'<b>{100 * t["flat_p"]:.0f} %</b> of time-censored runaways are fitted '
             f'{sd.term("flat", G["flat"])}. Without the invisible bins: <b>{100 * t["flat_g"]:.0f} %</b>.', 28),
        sd.p('The guarded runaways then look like their normal neighbours at the same t0. A flat track at '
             'late t0 is the fit parking the slope while the runaway bins absorb the end of the window.', 25),
        sd.callout(f'Campaign: {sd.pct(d["cm"]["late_big"], 1)} of gated tracks are in this class.', RED, 25),
        gap=26, w=520)
    body = sd.title('The flat tracks at late t0 are made by the runaway',
                    f'|tan θ| of fits with t0 &gt; {qr.LATE_T0:.0f} ns, production and guarded, '
                    f'{d["rm"]["run"]} refit sample.')
    body += sd.row(sd.col(leg, P.svg('tan distribution, late fits'), gap=6, w=1090), side, gap=50)
    C = pd.read_csv(qr.out_dir() / 'census.csv')
    C = C[C.plane == 'x']
    late_b = C.band == 't0 > 300'

    def wflat(m):
        c = C[m]
        return float((c.flat * c.n).sum() / c.n.sum())
    fl = (wflat(late_b & C.big), wflat(late_b & ~C.big), wflat(~late_b & C.big))
    D.slide('flat', body, f'''
<p>The same holds campaign-wide. In the census, {100 * fl[0]:.0f} % of late runaway x-plane tracks have
|tan_raw_x| &lt; {qr.FLAT_TAN}, against {100 * fl[1]:.0f} % of late tracks below the threshold and
{100 * fl[2]:.1f} % of early runaways. The early runaways are, if anything, steep: steep tracks are the ones
whose deep bins walk off the strip window.</p>''',
            foot='Hover the first-bin markers for the exact flat fractions.', short='Flat tracks')


def s_scale(D, d):
    pa = d['pa']
    rows, tips = [], []
    for arm, r in pa.iterrows():
        rows.append([f'<b>{arm}</b>', f'{int(r.n):,}', sd.pct(r.big_any, 1),
                     f'<b style="color:{RED}">{sd.pct(r.late_big, 1)}</b>', sd.pct(r.early_big, 1),
                     f'<b style="color:{RED}">{sd.pct(r.coinc_late_big, 1)}</b>', sd.pct(r.coinc_early_big, 1)])
        tips.append(f'chamber {arm}: x {sd.pct(r.big_x, 1)}, y {sd.pct(r.big_y, 1)} runaway; '
                    f'{int(r.coinc_n):,} coincident tracks')
    cm = d['cm']
    rows.append(['<b>all</b>', f'{cm["n_gated"]:,}', sd.pct(cm['big_any'], 1),
                 f'<b style="color:{RED}">{sd.pct(cm["late_big"], 1)}</b>', sd.pct(cm['early_big'], 1),
                 f'<b style="color:{RED}">{sd.pct(cm["coinc_late_big"], 1)}</b>', sd.pct(cm['coinc_early_big'], 1)])
    tips.append('campaign total')
    tbl = sd.table(['chamber', 'gated tracks', 'runaway', 'late (geometry)', 'early (charge)',
                    'late, coincident', 'early, coincident'], rows, 30, tips=tips)
    cond = ', '.join(f'{k.replace("_27jul", "")} {sd.pct(v, 1)}' for k, v in cm['by_condition'].items())
    body = sd.title('How much of the sample this touches',
                    'Gated tracks of the stage-3 full pass, by class of their runaway plane.')
    body += tbl
    body += sd.row(
        sd.callout(f'<b>Late class</b>: a runaway plane with t0 &gt; {qr.LATE_T0:.0f} ns. Its angle and t0 are '
                   'not trustworthy. Late tracks below threshold are affected too and are not counted here.', RED, 25),
        sd.callout(f'<b>Early class</b>: charge unusable, geometry fine in the median with a third moving by '
                   f'&gt; 0.05 in tan. Same rate across the 27 July access ({cond}).', ORANGE, 25), gap=48)
    D.slide('scale', body, f'''
<p>"Coincident" is <code>coinc_this_arm = 1</code>. The early column counts every runaway plane that is not
late, whatever its route.</p>
<p>Source: <code>qsum_runaway census</code> → <code>census_per_arm.csv</code>, <code>census.meta.json</code>.</p>''',
            short='Scale')


def s_close(D, d, S):
    items = [
        ('The guard is a probe', f'Worse on χ² for late tracks ({S["runaway, time"]["dchi"]:+.2f}/dof). It shows the '
                                 'production geometry depends on invisible bins, not which geometry is right.'),
        ('No external reference', 'Neither fit was checked against scintillator pointing (det_a_scint).'),
        ('One tag of one run', f'{d["rm"]["run"]} tag {d["rm"]["tag"]}. The census t0 dependence is campaign-wide; '
                               'the per-class shifts are not.'),
        ('What the late tracks are', 'Less often coincident than in-time tracks: probably out-of-time particles. '
                                     'Not established.'),
        ('Noise in the space class', 'Coherent-ringing windows are in it, in an unmeasured fraction.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:14px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:27px;font-weight:600;width:400px">{a}</p>'
                    f'<p style="font-size:23px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    dec = [('Now, no re-pass', 'q_sum and q_total unusable. Flag t0 &gt; 300 ns tracks before any angle or '
                               'pointing study; a late cut costs ~a fifth of the tracks.'),
           ('October re-pass (O10)', 'Stop offering invisible bins: truncate the depth grid per fit, or a '
                                     'depth-continuity prior on the NNLS. Store q_obs.'),
           ('Validate first', 'Any fix against scintillator pointing on arm A before it rides the re-pass.')]
    drows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:14px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:27px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:23px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in dec)
    body = (f'<div style="display:flex;gap:80px">'
            f'<div style="flex:1.25;display:flex;flex-direction:column;gap:8px">'
            f'<h2 style="font-size:52px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:8px">'
            f'<h2 style="font-size:52px;font-weight:600">What to do</h2>{drows}</div></div>')
    D.slide('close', body, '''
<p>Code: <code>sept26_prelim_analysis/qsum_runaway.py</code> (census, refit),
<code>make_qsum_runaway_figures.py</code>, <code>make_qsum_runaway_report.py</code>, and this deck,
<code>make_qsum_runaway_deck.py</code>. Long-form report with every table:
<code>~/x17/sept26_prelim/qsum_runaway/report.html</code>. Record: STATUS.md 2026-10-02; October list item O10.</p>''',
            dark=True, short='Caveats & next')


def build(out: Path) -> Path:
    d = load()
    S = summ(d['R'])
    D = sd.Deck('Large q_sum Tracks',
                'The impossible fitted charges on a third of the n_TOF tracks: an unbounded NNLS over depth '
                'bins the readout window cannot see, and what it does to the late tracks’ angles.')
    s_cover(D, d, S)
    s_setup(D, d)
    s_t0(D, d)
    s_anatomy(D, d)
    s_routes(D, d, S)
    s_geometry(D, d, S)
    s_flat(D, d, S)
    s_scale(D, d)
    s_close(D, d, S)
    meta = dict(title='The large-q_sum tracks: an unbounded fit, and the late tracks’ angles',
                summary=(f'{sd.pct(d["cm"]["big_any"])} of gated tracks carry q_sum > 10⁶ from NNLS charge in depth '
                         f'bins the window cannot see; the late {sd.pct(d["cm"]["late_big"])} also have the wrong '
                         f'angle.'),
                tags='X17,reconstruction,tracking', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by '
                                     'nTof_x17/sept26_prelim_analysis/make_qsum_runaway_deck.py with slidedoc.py.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (qr.out_dir() / 'deck' / 'qsum-runaway.html')
    print('wrote', build(out))


if __name__ == '__main__':
    main()
