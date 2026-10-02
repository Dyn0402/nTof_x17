#!/usr/bin/env python3
"""
make_two_track_deck.py -- the two-track limit study as a figure-first slide note.

The same results as report.html (make_two_track_limit_report.py), told in
slides with hover tooltips and a Details drop-down per slide, for
dylan-neff.web.cern.ch/notes. Built with dylan-cern-site/scripts/slidedoc.py.
Reads the report's figure CSVs and its loaders, so rerunning after the F
rescan merges fills the operating-point slide by itself.

    python -m sept26_prelim_analysis.make_two_track_deck [--out PATH]
    python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK)
from sept26_prelim_analysis import make_two_track_limit_report as R   # noqa: E402

CONTRACT = R.CONTRACT_CLEAN_SPLIT
LADDER_CLUSTER = 4348153
ARM_COL = dict(A=BLUE, C=ORANGE)
SEP_TICKS = [(0.25, '0.25'), (0.5, '0.5'), (1, '1'), (2, '2'), (4, '4'), (8, '8'), (12, '12')]
PCT_TICKS = [(v / 100, f'{v}%') for v in (0, 25, 50, 75, 100)]


def figdir() -> Path:
    d = R.src_dir() / 'report' / 'figures'
    if not d.exists():
        raise SystemExit(f'no report figures at {d}: run make_two_track_limit_report first')
    return d


# --------------------------------------------------------------------------- #
# glossary: tooltips used across slides
# --------------------------------------------------------------------------- #
G = dict(
    F=('TWO_TRACK_F: the split threshold. A two-track fit replaces the one-track fit '
       'when fstat ≥ F (≥ 0.4 F when the other view corroborates). Production: 300.'),
    fstat=('fstat = Δχ² / scale: how much better two lines describe the window than one, '
           'in units of the fit’s own noise scale.'),
    trigger=('The probe that decides whether a split is even attempted (residual z-score). '
             'It fires on ~0 % of inclined pairs below 12 mm.'),
    guard='TWO_MIN_SEP_MM = 1.2: two children closer than 1.2 mm are refused, whatever fstat says.',
    twin=('Perfect-model twin: each real donor refitted alone, rebuilt by the forward model from '
          'that fit, plus white noise at each strip’s pedestal σ. Twin vs real isolates model mismatch.'),
    splitab=('split-ab: real triggers re-reconstructed by both chains on the same triggers, '
             'matched to the frozen full pass. Measures the cost of a threshold; it has no truth.'),
    profc=('Profile pairing: the x/y pairing cost gains a term comparing the two views’ '
           'depth-charge profiles on an absolute-time grid (xy_pairing_*_profc.json, '
           'calibrated on stat090_0001).'),
    contract='The contract: clean single muons split ≤ 0.66 % (current production’s rate on C), '
             'and no event loses a track.',
    overlay=('Overlay bench: two clean single-track donors from the same chamber, file tag and '
             'trigger phase, summed (--overlay replace). Truth is the donors’ own fits.'),
    lam=('λ: the expected Δχ² between the best one-track and the true two-track description of a '
         'noise-free window, at the run’s 13.3 ADC noise. What a perfect analysis has to work with.'),
)


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, B, C):
    def b(arm, var, band):
        r = B[(B.arm == arm) & (B.variant == var)]
        return float(r[band].iloc[0])
    lt = '&lt; 12 mm'
    a0, a1 = b('A', 'before', lt), b('A', 'fixed + profile pairing', lt)
    c0, c1 = b('C', 'before', lt), b('C', 'fixed + profile pairing', lt)
    cc = C.set_index(['arm', 'chain'])
    ta = (f'Overlay bench, all seven file tags, condor cluster 4334051.\n'
          f'A 12–24 mm: {100 * b("A", "before", "12–24 mm"):.0f} → '
          f'{100 * b("A", "fixed + profile pairing", "12–24 mm"):.0f} %')
    tc = (f'C 12–24 mm: {100 * b("C", "before", "12–24 mm"):.0f} → '
          f'{100 * b("C", "fixed + profile pairing", "12–24 mm"):.0f} %')
    nc = (f'Clean singles split: A {100 * cc.loc[("A", "fixed"), "frac_clean_split"]:.2f} %, '
          f'C {100 * cc.loc[("C", "fixed"), "frac_clean_split"]:.2f} % (≤ 0.66 %). No event loses a track.')
    nums = ''.join([
        sd.bignum(f'{100 * a0:.0f} → {100 * a1:.0f}%', 'Chamber A, &lt; 12 mm', DBLUE,
                  'Production → fixed chain + profile pairing, at a threshold matched to production’s '
                  'false-split rate.', tip=ta),
        sd.bignum(f'{100 * c0:.0f} → {100 * c1:.0f}%', 'Chamber C, &lt; 12 mm', '#f0a36b',
                  'Same comparison. C is held back by a stricter matched threshold (F = 2400).', tip=tc),
        sd.bignum('passes', 'The real-trigger contract', DGREEN, nc, tip=G['contract'])])
    body = (sd.kicker('run_145 stat090_0000 · chambers A and C · 2 Oct 2026')
            + '<h1 style="font-size:84px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1600px">'
              'Two tracks in one plane: the limit is a strip pitch, and an opt-in chain gets most of the way there</h1>'
            + '<div style="flex:1"></div>'
            + f'<p style="font-size:28px;color:{DMUT}">Coincident pairs closer than 12 mm, found <i>and</i> '
              'correctly x/y-paired, on the real-overlay bench.</p>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, f'''
<p>Three limits are compared throughout. The <b>physical limit</b> is what a perfect analysis could resolve if the forward model were exact: about one strip pitch. The <b>real-track limit</b> is the same ideal fit on real tracks, where the forward model is imperfect: about 2–3 mm. <b>Production</b> is the current wft.reco chain, which was far from both.</p>
<p>The <b>fixed chain</b> is four opt-in switches: <code>WFT_TWO_TRACK_SCALE=two</code>, <code>WFT_TWO_TRACK_SEARCH=grid</code>, no trigger, and every candidate tried. <b>Profile pairing</b> is the x/y pairing calibration <code>xy_pairing_*_profc.json</code>. With all switches off, output is production’s, bit for bit.</p>
<p>Not yet decided: the operating threshold F per chamber (a real-trigger rescan, condor cluster {LADDER_CLUSTER}, is running), and whether to ship to the production re-pass.</p>''',
            dark=True, short='Answer')


def s_problem(D):
    # schematic: drift gap, strips along the mesh, two tracks d apart, and the per-strip charge
    W, H = 900, 700
    x0, x1, ztop, zbot = 80, 860, 60, 420
    o = []
    o.append(f'<rect x="{x0}" y="{ztop}" width="{x1 - x0}" height="{zbot - ztop}" fill="#eef1f6" stroke="{RULE}"/>')
    o.append(sd.T(x0 + 10, ztop - 16, 'drift cathode', 20, MUT, 'start'))
    for i in range(27):
        sx = x0 + 5 + i * 28.8
        o.append(f'<rect x="{sx:.1f}" y="{zbot + 4}" width="20" height="10" fill="#9aa3b2"/>')
    o.append(sd.T(x0 + 10, zbot + 42, 'mesh · strips (pitch ~0.8 mm, 61-strip window)', 20, MUT, 'start'))
    # two inclined tracks
    tx = [(380, 300), (430, 350)]
    for (xa, xb), c, lab in zip(tx, (BLUE, ORANGE), ('track 1', 'track 2')):
        o.append(sd.line(xa, ztop, xb, zbot, c, 5))
        o.append(sd.T(xa - 6 if c == BLUE else xa + 10, ztop + 30, lab, 20, c, 'end' if c == BLUE else 'start', 600))
    o.append(sd.arrow(340, 240, 390, 240, INK, 2, 10) + sd.arrow(390, 240, 340, 240, INK, 2, 10))
    o.append(sd.T(365, 228, 'd', 26, INK, 'middle', 600,
                  tip='Separation d: r.m.s. distance between the two lines over the drift column. '
                      'Real pairs are not parallel, so a 0.3 mm mesh gap can still carry Δχ² ≈ 1 500.'))
    o.append(sd.T(x1 - 6, (ztop + zbot) / 2, 'drift depth z  (time ×  v)', 20, MUT, 'end'))
    # charge humps under the strips
    import numpy as np
    xs = np.linspace(x0, x1, 300)
    def g(m, s, a):
        return a * np.exp(-0.5 * ((xs - m) / s) ** 2)
    base = H - 20
    q1, q2 = g(300, 34, 110), g(350, 34, 110)
    o.append(sd.poly(list(xs) + [x1, x0], list(base - q1 - q2) + [base, base], None,
                     fill='#d6dbe4', tip='Charge each strip records: both tracks, summed, then spread to ±2 '
                                         'neighbours by the resistive layer (the sharing kernel).'))
    o.append(sd.poly(xs, base - q1, BLUE, 2.5, dash='6 5'))
    o.append(sd.poly(xs, base - q2, ORANGE, 2.5, dash='6 5'))
    o.append(sd.T(x1 - 10, base - 150, 'charge per strip, both tracks summed', 20, MUT, 'end'))
    o.append(sd.line(x0, base, x1, base, MUT, 1.5))
    schem = sd.svg(W, H, ''.join(o), 'two tracks in one drift plane')

    steps = [
        dict(label='Fit one track', sub='forward model: line + t0 + depth-charge profile', color=GREY,
             tip='The one-track fit every plane gets. Its χ² is the reference.'),
        dict(label='Trigger?', sub='attempt a split only if residuals look like a second track', color=RED,
             tip=G['trigger']),
        dict(label='Fit two tracks', sub='seeded from the parent fit', color=GREY,
             tip='Production seeds the two children from the one-track parent; that parent is itself a slanted '
                 'compromise line, so the search can settle in an "X" of wrong children.'),
        dict(label='fstat ≥ F?', sub='and the 1.2 mm guard passes', color=PURPLE,
             tip=G['fstat'] + '\n\n' + G['F'] + '\n\n' + G['guard']),
    ]
    right = sd.col(
        sd.p('<b>The question:</b> at what separation can two tracks in one readout plane be told apart, '
             'and how far is the reconstruction from that?', 28),
        sd.flow(steps[:2], size=24), sd.flow(steps[2:], size=24),
        sd.callout(f'Three limits throughout: <b>physical</b> (perfect model, ideal fit) · '
                   f'<b>real-track</b> (ideal fit on real windows) · <b>production</b> (what we run). '
                   f'Hover the {sd.term("dotted terms", "Like this one. Charts work the same way: hover any point.")} '
                   'and chart points for definitions and counts.', BLUE, 24),
        gap=26)
    body = sd.title('Two tracks in one plane share strips; the reconstruction must decide to split',
                    'A one-track fit is always made. A two-track fit replaces it only past a threshold.')
    body += sd.row(schem, right, gap=56)
    D.slide('problem', body, '''
<p>Each readout plane (x or y view) of a micro-TPC chamber sees a track as a line in (strip, drift time). Two tracks in the same plane share strips wherever they are closer than a few pitches, and the resistive layer spreads each strip's charge to its neighbours, so their signals overlap.</p>
<p>Reconstruction (<code>wft.reco</code>) fits one track to every window. A <b>trigger</b> (a residual z-score probe) decides whether to try a split; the two-track fit is then accepted when its improvement <code>fstat</code> clears the threshold <code>TWO_TRACK_F</code> (300 in production, 120 when the other view already sees two tracks), and a distinguishability guard refuses children closer than 1.2 mm.</p>
<p>Two populations matter: real pairs (which we want to split) and single tracks (which we must not split). Every threshold trades one against the other, so every comparison in this note is made at a matched rate of false splits on real single tracks.</p>''',
            short='The problem')


def s_r1(D, F):
    r1 = pd.read_csv(F / 'r1_asimov.csv')
    r4 = pd.read_csv(F / 'r4_synth_ladder.csv')
    thr = r4[r4.stage == 'ideal'].groupby('arm').thr.median()
    P = sd.Plot(1060, 640, x=(0.25, 12, 'log'), y=(1e-3, 1e5, 'log'),
                xlabel='separation d [mm]', ylabel='expected Δχ²  λ')
    P.xticks(SEP_TICKS).yticks(sd.log_ticks(-3, 5))
    for arm in ('A', 'C'):
        for tan, dash in ((0.0, None), (0.3, '10 7')):
            g = r1[(r1.arm == arm) & (r1.tan == tan)].sort_values('d')
            tips = [f'chamber {arm}, tan θ = {tan}\nd = {d} mm\nλ = {l:,.3g}\npeak S/N {s:.0f}'
                    for d, l, s in zip(g.d, g.lam, g.peak_snr)]
            P.line(list(g.d), [max(v, 1e-3) for v in g.lam], ARM_COL[arm], 4, dash, tips=tips,
                   marker='circle' if tan == 0 else 'open', tip=f'{arm}, tan θ = {tan}')
    t = float(thr.mean())
    P.hline(t, RED, '8 6', 2, label=f'Δχ² ≈ {t:.0f}: 1 % false-split threshold, perfect model', anchor='end',
            tip='99th percentile of Δχ² on 400 synthetic single tracks per cell: the threshold a perfect '
                'analysis needs to split no more than 1 % of singles.', where='below')
    leg = sd.legend([('A, tan 0', BLUE), ('A, tan 0.3', BLUE, 'dash'),
                     ('C, tan 0', ORANGE), ('C, tan 0.3', ORANGE, 'dash')])
    side = sd.col(
        sd.p(f'{sd.term("λ", G["lam"])} collapses like ~d⁴ near zero and saturates once the tracks stop '
             'overlapping.', 28),
        sd.p('Vertical tracks cross the threshold at <b>~0.75 mm</b>, about one strip pitch. '
             'An inclined track (tan 0.3) spreads each depth slice over fewer strips per unit charge, '
             'so the same d carries ~50× less information: <b>~1.5 mm</b>.', 26),
        sd.callout('This is the physical limit. Nothing below assumes it can be beaten.', GREY, 26),
        gap=28, w=520)
    body = sd.title('The information limit: about one strip pitch',
                    'Noise-free forward-model pairs, 1 450 ADC·bin each, scored at the run’s noise (13.3 ADC).')
    body += sd.row(sd.col(P.svg('lambda vs separation'), leg, gap=8, w=1080), side, gap=60)
    D.slide('r1', body, '''
<p>Two tracks with the same slope and t0 are generated noise-free by the forward model under the run_145 bundle, with a flat depth profile to 840 ns. The best single track is fitted to that window; the χ² it leaves is λ, the Δχ² a perfect analysis expects between one and two tracks (its Asimov value).</p>
<p>Chamber C’s tan 0 curve flattens between 0.5 and 1 mm because C’s bundle puts a vertical track on one strip (σ<sub>p0</sub> = 0.039 mm, not revalidated): its tan 0 numbers are the model’s, not the chamber’s. No clean real donor has |tan θ| &lt; 0.05 anyway.</p>
<p>Source: <code>two_track_limit.py asimov</code> → <code>r1_asimov.csv</code>.</p>''',
            foot='R1 · r1_asimov.csv. Hover any point for λ and the peak signal-to-noise.', short='Physical limit')


def s_synth(D, F):
    S = pd.read_csv(F / 'r4_synth_ladder.csv')
    stages = [('ideal', 'ideal fit', INK, None),
              ('fixed: scale + grid, no trigger', 'fixed chain', GREEN, None),
              ('production, no trigger', 'production, trigger off', GREY, '10 7'),
              ('production, with trigger', 'production', RED, None)]
    panels = []
    for arm in ('A', 'C'):
        for tan in (0.0, 0.3):
            P = sd.Plot(800, 300 if arm == 'A' else 330, x=(0.25, 12, 'log'), y=(0, 1),
                        title=f'chamber {arm}, tan θ = {tan}', margin=(10, 24, 46 if arm == 'A' else 76, 92),
                        xlabel='d [mm]' if arm == 'C' else '')
            P.xticks(SEP_TICKS).yticks([(0, '0'), (0.5, '50%'), (1, '100%')])
            for st, lab, c, dash in stages:
                g = S[(S.arm == arm) & (S.tan == tan) & (S.stage == st)].sort_values('d')
                tips = [f'{lab}\nd = {d} mm: {100 * e:.0f} % of {n} pairs' for d, e, n in zip(g.d, g.eff, g.n)]
                P.line(list(g.d), list(g.eff), c, 3.5 if st != 'ideal' else 4.5, dash, r=4.5, tips=tips, tip=lab)
            panels.append(P.svg(f'{arm} tan {tan}'))
    grid = (f'<div style="display:grid;grid-template-columns:800px 800px;gap:10px 64px">{"".join(panels)}</div>')
    leg = sd.legend([(lab, c, 'dash' if dash else 'line') for _s, lab, c, dash in stages])
    body = sd.title('Production loses pairs the ideal fit resolves',
                    'Same synthetic planes, same seeds. Pairs resolved vs separation: both lines within 1 mm r.m.s. '
                    'of distinct true lines. 100 pairs per point; hover for counts.')
    body += leg + grid
    D.slide('synth', body, '''
<p>The ideal fit (pure χ², one- and two-track fits from the truth and from a broad start set, threshold at the 99th percentile of Δχ² on 400 synthetic singles) is run against production’s own fit, probe, trigger and joint fit on <b>the same planes</b> (same seeds), cut to a production window.</p>
<p>Production’s losses are algorithmic, each one measured:</p>
<ol>
<li><b>The trigger</b> fires on ~0 % of tan 0.3 pairs below 12 mm (and on none of A’s vertical pairs at 2–4 mm).</li>
<li><b>The fstat scale.</b> <code>scale = χ²_one/dof</code> contains the second track’s unexplained charge, so fstat ≈ λ/(1+λ/dof) &lt; dof ≈ 200–300 &lt; F = 300 for a close vertical pair. Fix: <code>WFT_TWO_TRACK_SCALE=two</code> takes the scale from the two-track fit.</li>
<li><b>The search.</b> Children seeded from the one-track parent settle into an “X”. Every unfound case had a lower-χ² true solution (27/27). Fix: <code>WFT_TWO_TRACK_SEARCH=grid</code>, a global grid of parallel line pairs then Nelder–Mead: 96–100 %, median χ² gap to truth 0.</li>
<li><b>The 1.2 mm guard</b> is a hard floor where the ideal is already 77–100 %. It was kept: on real singles it blocks 42 % of false splits.</li>
</ol>
<p>False splits on synthetic singles are 0–2 % for every chain, identical with and without the fixes.</p>
<p>Source: <code>two_track_limit.py oracle</code> / <code>synthprod</code> → <code>r4_synth_ladder.csv</code>.</p>''',
            short='Synthetic planes')


def s_null(D, F):
    N = pd.read_csv(F / 'r3_null.csv')
    P = sd.Plot(900, 600, x=(0, 4), y=(1, 2000, 'log'), ylabel='Δχ² of the best split on a single track')
    P.yticks([(1, '1'), (10, '10'), (100, '100'), (1000, '1 000')])
    xs = {('A', 'twin'): 0.6, ('A', 'real'): 1.4, ('C', 'twin'): 2.6, ('C', 'real'): 3.4}
    for r in N.itertuples():
        x = xs[(r.arm, r.population)]
        c = ARM_COL[r.arm]
        op = 0.35 if r.population == 'twin' else 1.0
        P.vbar(x, r.q99, 120, c, opacity=op, base=1,
               tip=f'chamber {r.arm}, {r.population}: {r.n} well-modelled single tracks\n'
                   f'median Δχ² {r.median:.1f}\n99th percentile {r.q99:.0f} (the 1 % threshold)',
               label=f'{r.q99:.0f}')
        P.raw(sd.line(P.X(x) - 60, P.Y(r.median), P.X(x) + 60, P.Y(r.median), INK, 3))
        P.raw(sd.T(P.X(x), P.y0 + P.ph + 32, r.population, 22, INK))
    for arm, x in (('A', 1), ('C', 3)):
        P.raw(sd.T(P.X(x), P.y0 + P.ph + 66, f'chamber {arm}', 24, ARM_COL[arm], weight=600))
    ra = N.set_index(['arm', 'population']).q99
    fa, fc = ra['A', 'real'] / ra['A', 'twin'], ra['C', 'real'] / ra['C', 'twin']
    side = sd.col(
        sd.p(f'Each real single track is refitted with the ideal two-track fit, and so is its '
             f'{sd.term("perfect-model twin", G["twin"])}. The twin’s split gains what noise gives; '
             'the real track also gains whatever the forward model fails to describe.', 26),
        sd.p(f'So the threshold that keeps false splits at 1 % rises <b>×{fa:.0f}</b> in A and '
             f'<b>×{fc:.0f}</b> in C, and the mismatch grows with the track’s charge.', 28),
        sd.callout('That is why real tracks resolve from ~2–3 mm, not one pitch: the model, not the '
                   'information, sets the real limit.', ORANGE, 26),
        sd.p('Bar: 99th percentile (the threshold). Black tick: median.', 22, MUT),
        gap=26, w=640)
    body = sd.title(f'Real tracks are not their perfect-model twins: the threshold rises ×{fa:.0f}–{fc:.0f}',
                    'Δχ² gained by splitting well-modelled single tracks (χ²/dof &lt; 2), ideal fit.')
    body += sd.row(P.svg('null delta chi2'), side, gap=72)
    D.slide('null', body, '''
<p>The windows are 61 strips wide, so some hold charge the donor does not explain (another cluster, saturation). A window counts as <b>well modelled</b> when its donor, refitted alone on it, reaches χ²/dof &lt; 2: A 87 % of singles, C 55 %. On all windows the thresholds are set by that foreign charge instead, and the real ideal fit collapses (see the next slide’s Details).</p>
<p>Two things were tried to close the gap and dropped: a fractional model error ε in the χ² (no gain at a fixed false-split rate), and a looser guard (it blocks 42 % of false splits on real singles).</p>''',
            foot='R3 · r3_null.csv. 600 single donors per (chamber, view) before the χ²/dof cut.', short='Model mismatch')


def s_real(D, F):
    T = pd.read_csv(F / 'r3_real_ladder.csv')
    ser = [('twin', 'twin, ideal fit', INK, '10 7'), ('real', 'real, ideal fit', PURPLE, None),
           ('fixed', 'fixed chain', GREEN, None), ('prod', 'production', RED, None)]
    panels = []
    for arm in ('A', 'C'):
        P = sd.Plot(810, 560, x=(0.25, 24, 'log'), y=(0, 1), title=f'chamber {arm}',
                    xlabel='r.m.s. separation [mm]', ylabel='pairs resolved' if arm == 'A' else '')
        P.xticks([(0.25, '0.25'), (0.5, '0.5'), (1, '1'), (2, '2'), (4, '4'), (8, '8'), (16, '16')]).yticks(PCT_TICKS)
        g = T[T.arm == arm].sort_values('sep_mid')
        for k, lab, c, dash in ser:
            tips = [f'{lab}\n{lo:g}–{hi:g} mm: {100 * v:.0f} % of {n} pairs' for lo, hi, v, n in zip(g.lo, g.hi, g[k], g.n)]
            P.line(list(g.sep_mid), list(g[k]), c, 4, dash, r=5.5, tips=tips, tip=lab)
        panels.append(P.svg(f'real overlays {arm}'))
    leg = sd.legend([(lab, c, 'dash' if dash else 'line') for _k, lab, c, dash in ser])
    body = sd.title('On real overlays the ideal fit resolves from ~2–3 mm; the fixed chain closes most of the gap',
                    f'Two clean donors, coincident (|Δt0| &lt; 30 ns), overlaid; well-modelled windows. Fixed chain at a '
                    f'threshold matched to production’s false-split rate on real singles.')
    body += leg + sd.row(*panels, gap=44)
    D.slide('real', body, '''
<p>Each pair is two clean single-track donors of one chamber, file tag and trigger phase, coincident in this view, overlaid with <code>--overlay replace</code>. 40 pairs per bin of mesh separation; separation is quoted as the r.m.s. over the drift column because real pairs are not parallel.</p>
<p>Unweighted means over 3–16 mm: twin A 100 %, C 98 %; ideal on the real windows A 99 %, C 99 %; production A 39 %, C 54 %; fixed A 88 %, C 80 %.</p>
<p>C’s well-modelled sample is small below 2 mm (3–9 pairs per bin; hover for n). On all windows, unrestricted, the real ideal fit collapses to 0–15 % in C because the thresholds are set by foreign charge in the window; production and the fixed chain, which work on their own windows, do not.</p>''',
            foot='R3 & R4 · r3_real_ladder.csv. Matched thresholds: A F = 1000, C F = 2400.', short='Real overlays')


def s_roc(D, F):
    S = pd.read_csv(F / 'r3_roc.csv')
    pick = {'A': 1000, 'C': 2400}
    panels = []
    for arm in ('A', 'C'):
        P = sd.Plot(810, 560, x=(0, 0.06), y=(0, 0.8), title=f'chamber {arm}',
                    xlabel='real single tracks split', ylabel='real pairs resolved' if arm == 'A' else '')
        P.xticks([(v / 100, f'{v}%') for v in range(0, 7)]).yticks([(v / 100, f'{v}%') for v in (0, 20, 40, 60, 80)])
        for var, lab, c in (('current_f0', 'production', RED), ('fixed_f0', 'fixed chain', GREEN)):
            g = S[(S.arm == arm) & (S.variant == var) & (S.fsr <= 0.06)].sort_values('F', ascending=False)
            tips = [f'{lab}, F = {f:g}\nsingles split {100 * a:.1f} %\npairs resolved {100 * e:.0f} %'
                    for f, a, e in zip(g.F, g.fsr, g.eff)]
            P.line(list(g.fsr), list(g.eff), c, 3.5, r=5, tips=tips, tip=lab)
            op = g[g.F == (300 if var == 'current_f0' else pick[arm])]
            if len(op):
                P.points(list(op.fsr), list(op.eff), c, r=13, marker='open',
                         tips=[f'operating point: {lab}, F = {int(op.F.iloc[0])}\n'
                               f'{100 * op.fsr.iloc[0]:.1f} % singles split, {100 * op.eff.iloc[0]:.0f} % pairs'])
        panels.append(P.svg(f'roc {arm}'))
    body = sd.title('At an equal false-split rate the fixed chain resolves far more pairs',
                    f'Each curve scans the threshold {sd.term("F", G["F"])}. Circles: the operating points '
                    'compared everywhere else (production at 300; fixed at the lowest F splitting no more singles).')
    body += sd.legend([('production', RED), ('fixed chain', GREEN), ('operating point', INK, 'dot')]) + sd.row(*panels, gap=44)
    D.slide('roc', body, '''
<p>One production run per chain at threshold 0 records every split attempt with its fstat; any threshold is then applied offline (the corroborated threshold kept at 0.4 F, as in production). False splits: fraction of 600 real single donors per view that get an accepted split in either view. Pairs: all windows, all separations.</p>
<table><tr><th>chamber</th><th>chain</th><th>F</th><th>singles split</th><th>pairs resolved</th></tr>
<tr><td>A</td><td>production</td><td>300</td><td>1.0 %</td><td>37 %</td></tr>
<tr><td>A</td><td>fixed</td><td>1000</td><td>1.0 %</td><td>68 %</td></tr>
<tr><td>C</td><td>production</td><td>300</td><td>1.7 %</td><td>46 %</td></tr>
<tr><td>C</td><td>fixed</td><td>2400</td><td>1.2 %</td><td>58 %</td></tr></table>
<p>The fixed chain at F = 0 splits 40–50 % of singles: it is the threshold, not the chain, that controls false splits. These are overlay donors; on real triggers the false-split rate is lower (slide “Contract”), which is why the rescan may allow a lower F.</p>''',
            foot='R3 · r3_roc.csv (r3_scan.parquet). Hover a point for its F.', short='Equal false-split rate')


def s_chain(D):
    prod = [dict(label='Trigger', sub='residual z-score probe', color=RED, tip=G['trigger']),
            dict(label='Seed from parent', sub='children start from the one-track fit', color=RED,
                 tip='Local search: settles in an “X” of slanted children at close vertical separation.'),
            dict(label='fstat scale = χ²₁/dof', sub='holds the 2nd track’s charge', color=RED,
                 tip='fstat ≈ λ/(1+λ/dof) < dof, so close vertical pairs can never reach F = 300.'),
            dict(label='F = 300', sub='120 if corroborated', color=GREY, tip=G['F']),
            dict(label='x/y pairing', sub='charge & time only', color=GREY,
                 tip='xy_pairing_*.json: the cost from charge balance and time.')]
    fix = [dict(label='No trigger', sub='every candidate tried', color=GREEN,
                tip='TWO_TRACK_RESID_Z = −inf, TWO_TRACK_MAX_TRY = 99, TWO_TRACK_SELECTED_ONLY = false.'),
           dict(label='Grid search', sub='parallel line pairs, then Nelder–Mead', color=GREEN,
                tip='WFT_TWO_TRACK_SEARCH=grid: strip step, common tan −0.4…0.4, tied t0 at parent ± 60 ns, '
                    'NNLS per point, refine the best 3. Median χ² gap to truth 0. 2–10 k evaluations per window.'),
           dict(label='Scale from 2-track fit', sub='2nd track no longer dilutes fstat', color=GREEN,
                tip='The fstat scale is taken from the two-track fit, so the second track’s charge no longer '
                    'dilutes the statistic.'),
           dict(label='F matched', sub='A 1200 · C 2400 (rescan running)', color=PURPLE,
                tip='The lowest F whose false-split rate on real singles does not exceed production’s at 300. '
                    'Bench runs use A 1200; the R3 scan picks A 1000.'),
           dict(label='+ profile pairing', sub='depth-profile term in the x/y cost', color=GREEN, tip=G['profc'])]
    lab = lambda t, c: f'<p style="font-size:28px;font-weight:600;color:{c};width:200px;align-self:center">{t}</p>'
    body = sd.title('The fixed chain: four opt-in switches and a pairing calibration',
                    'Same forward model, same guard. With every switch off the output is production’s, bit for bit.')
    body += sd.row(lab('production', RED), sd.flow(prod, size=26), gap=24, align='stretch')
    body += sd.row(lab('fixed', GREEN), sd.flow(fix, size=26), gap=24, align='stretch')
    body += sd.row(
        sd.card(sd.p('<b>Kept:</b> the 1.2 mm distinguishability guard. On real singles it blocks 42 % of false '
                     'splits, so loosening it buys pairs only by splitting muons.', 24), tip=G['guard']),
        sd.card(sd.p('<b>Kept:</b> the rescue floor (<code>WFT_SIG_FLOOR_LOCAL_MM=16</code>, mode rescue), already part '
                     'of the bench’s “before” configuration.', 24)),
        sd.card(sd.p('<b>Cost:</b> 5–30 s per attempted window for the grid search. Compute is not a constraint '
                     '(condor), so it is not traded against efficiency.', 24)),
        gap=28)
    D.slide('chain', body, '''
<p>Worker options for a condor job (as in <code>make_two_track_package.py</code>): <code>TWO_TRACK_SCALE=two</code>, <code>TWO_TRACK_SEARCH=grid</code>, <code>TWO_TRACK_RESID_Z=-inf</code>, <code>TWO_TRACK_MAX_TRY=99</code>, <code>TWO_TRACK_SELECTED_ONLY=false</code>, <code>TWO_TRACK_F</code> and <code>TWO_TRACK_F_CORROB</code> = 0.4 F.</p>
<p>Profile pairing needs bundles carrying <code>xy_pairing_{A,C}_profc.json</code> (constrained depth-profile term, calibrated on stat090_0001 so it is out of sample for the bench, stat090_0000). The first, unconstrained calibration was in-sample and is superseded.</p>''',
            short='The fixed chain')


def bench_counts():
    """bench_table() with the number of coincident pairs per band, for tooltips."""
    from sept26_prelim_analysis import intra_bench as ib
    import numpy as np
    base, out = ib.out_dir(), []
    for arm in R.ARMS:
        for lab, tmpl in R.BENCH_VARIANTS:
            d = base / tmpl.format(arm=arm)
            if not (d / 'overlays.parquet').exists():
                continue
            M = pd.read_parquet(d / 'overlays.parquet')
            C = pd.read_parquet(d / 'candidates.parquet')
            Dn = pd.read_parquet(d / 'donors.parquet' if (d / 'donors.parquet').exists() else base / 'donors.parquet')
            S = ib.score(M, C, Dn)
            o = S[(S['mode'] == 'overlay') & (S.cls == 'coincident') & (S.arm == arm)]
            ev = o.groupby('oid').agg(sx=('sep_x', 'first'), sy=('sep_y', 'first'), both=('track_found', 'all'))
            sep = np.minimum(ev.sx, ev.sy)
            for lo, hi, b in R.BENCH_BANDS:
                m = (sep >= lo) & (sep < hi)
                out.append(dict(arm=arm, variant=lab, band=b, eff=float(ev.both[m].mean()), n=int(m.sum()),
                                k=int(ev.both[m].sum())))
    return pd.DataFrame(out)


def s_bench(D, BC):
    cols = {'before': GREY, 'fixed': '#7fbf9c', 'fixed + profile pairing': GREEN}
    panels = []
    bands = [b for _l, _h, b in R.BENCH_BANDS]
    for arm in ('A', 'C'):
        P = sd.Plot(810, 560, x=(-0.5, 2.5), y=(0, 1), title=f'chamber {arm}',
                    ylabel='pairs found and x/y-paired' if arm == 'A' else '', margin=(24, 30, 80, 104))
        P.yticks(PCT_TICKS)
        for i, b in enumerate(bands):
            P.raw(sd.T(P.X(i), P.y0 + P.ph + 36, b.replace('&lt;', '<').replace('&ge;', '≥'), 23, INK))
            for j, (v, c) in enumerate(cols.items()):
                r = BC[(BC.arm == arm) & (BC.variant == v) & (BC.band == b)]
                if not len(r):
                    continue
                r = r.iloc[0]
                P.vbar((P.X(i) + (j - 1) * 64,), r.eff, 56, c, label=f'{100 * r.eff:.0f}',
                       tip=f'{v}, {b.replace("&lt;", "<").replace("&ge;", "≥")}\n{r.k}/{r.n} coincident pairs '
                           f'({100 * r.eff:.1f} %)')
        panels.append(P.svg(f'bench {arm}'))
    body = sd.title('At event level, close pairs in A triple; C gains most at 12–24 mm',
                    f'Full {sd.term("overlay bench", G["overlay"])}, coincident pairs, both donors found <i>and</i> '
                    'correctly x/y-paired. All seven file tags (condor cluster 4334051).')
    body += sd.legend([('production (before)', GREY, 'box'), ('fixed chain', '#7fbf9c', 'box'),
                       ('fixed + profile pairing', GREEN, 'box')]) + sd.row(*panels, gap=44)
    D.slide('bench', body, '''
<p>“Before” is the bench’s best production configuration (<code>pairing_rescue16_two_final_replace</code>: x/y pairing with the rescue floor). The fixed chain runs at its matched thresholds, A F = 1200, C F = 2400, corroborated at 0.4 F.</p>
<p>At ≥ 24 mm the tracks do not overlap; what remains lost there is x/y pairing, which profile pairing helps in A (85 → 89 %) and leaves flat in C. Below 12 mm C is held back by its strict threshold: the overlay scan says C resolves 58 % at F = 2400, 69 % at 1200 and 70 % at 1000, which is what the real-trigger rescan is testing.</p>
<p>Parallel, co-located tracks are degenerate at any Δt and stay lost; they must be carried as inefficiency.</p>''',
            foot='intra_bench variants pairing_rescue16_two_final_replace, fixed_{A,C}_replace, fixed_{A,C}_replace_profc. '
                 'Hover a bar for k/n.', short='Event-level bench')


def s_contract(D, C):
    rows = C.set_index(['arm', 'chain'])
    P = sd.Plot(780, 560, x=(-0.5, 3.5), y=(0, 0.02), ylabel='clean single muons split',
                margin=(24, 30, 110, 110))
    P.yticks([(v / 1000, f'{v / 10:.1f}%') for v in (0, 5, 10, 15, 20)])
    P.hline(CONTRACT, RED, '10 7', 2.5, label='contract 0.66 %', tip=G['contract'], anchor='start')
    i = 0
    for arm in ('A', 'C'):
        for ch in ('current', 'fixed'):
            r = rows.loc[(arm, ch)]
            k, n = int(r.clean_singles_split), int(r.clean_singles)
            ul = R._cp_upper(k, n)
            x = P.X(i)
            c = ARM_COL[arm] if ch == 'fixed' else GREY
            P.raw(sd.line(x, P.Y(r.frac_clean_split), x, P.Y(ul), c, 3))
            P.raw(sd.line(x - 12, P.Y(ul), x + 12, P.Y(ul), c, 3))
            P.points([i], [r.frac_clean_split], c, r=11,
                     tips=[f'chamber {arm}, {ch}\n{k}/{n} clean singles split ({100 * r.frac_clean_split:.2f} %)\n'
                           f'90 % CL upper limit {100 * ul:.2f} %'])
            P.raw(sd.T(x, P.y0 + P.ph + 34, ch if ch == 'fixed' else 'production', 22, INK))
            i += 1
        P.raw(sd.T(P.X(i - 1.5), P.y0 + P.ph + 72, f'chamber {arm}', 24, ARM_COL[arm], weight=600))

    def pair_bars(field, label, tipfmt):
        out = []
        for arm in ('A', 'C'):
            for ch in ('current', 'fixed'):
                r = rows.loc[(arm, ch)]
                v = int(r[field])
                out.append((f'{arm} {"production" if ch == "current" else "fixed"}', v,
                            GREY if ch == 'current' else ARM_COL[arm], tipfmt(arm, ch, r)))
        vmax = max(o[1] for o in out)
        return sd.col(sd.p(label, 26, INK, 600), sd.hbars(out, vmax, width=330, h=28, label_w=220, size=23), gap=12)

    right = sd.col(
        pair_bars('not_recovered', 'Production tracks not recovered',
                  lambda a, c, r: f'{int(r.not_recovered)} of {int(r.prod_tracks):,} frozen-pass tracks have no '
                                  f'match after re-reconstruction ({a}, {c}).'),
        pair_bars('events_split', 'Events with an accepted split',
                  lambda a, c, r: f'{int(r.events_split)} of {int(r.triggers):,} triggers; '
                                  f'{int(r.events_more_tracks)} gain a track.'),
        sd.p(f'Events losing a track: <b>0</b> in every row. Clean tracks lost: A 1 → 0, C 3 → 3.', 24),
        gap=30, w=820)
    body = sd.title('On real triggers both chambers pass the contract',
                    f'{sd.term("split-ab", G["splitab"])}: 22 434 (A) and 22 788 (C) triggers, re-reconstructed by both chains. '
                    f'Error bars: 90 % CL upper limits.')
    body += sd.row(P.svg('contract'), right, gap=64)
    D.slide('contract', body, '''
<p>Real triggers are re-reconstructed with each chain and matched to the frozen full pass, on the <b>same triggers</b> for both chains, both with x/y re-pairing (which alone moves ~5 % of unsplit events against the frozen pass, which was reconstructed without it).</p>
<p>The contract is clean single muons split ≤ 0.66 % (production’s own rate on C) and no event losing a track. The sample is small: 0.66 % of C’s 1 057 clean singles is 7 events, so neighbouring thresholds are not statistically distinct.</p>
<p>C’s fixed chain at F = 2400 splits far fewer real events than production (207 vs 913) and recovers twice as many of production’s own tracks: inside the contract with room to spare. That room is what the F rescan measures. Real triggers have no truth: split-ab bounds the cost of a threshold, while the gain comes only from the overlay bench.</p>''',
            foot='contract_fixed_vs_current.csv · condor cluster 4334051 · seven file tags of run_145 stat090_0000.',
            short='Contract')


def s_operating(D, L):
    Dl = R.ladder_table(L)
    Sc = L['scan'][L['scan'].variant == 'fixed_f0']
    bench = Sc.groupby(['arm', 'F']).apply(lambda g: pd.Series(dict(
        fsr=g[g.n_true == 1].split.mean(), eff=g[g.n_true == 2].found.mean(),
        n1=int((g.n_true == 1).sum()), n2=int((g.n_true == 2).sum()))), include_groups=False).reset_index()
    bench = bench[(bench.F >= 300) & (bench.F <= 4800)]
    matched = {'A': 1200, 'C': 2400}
    merged = len(Dl) > 0
    panels = []
    FX = [(300, '300'), (600, '600'), (1000, '1000'), (2400, '2400'), (4800, '4800')]
    floor = 1e-3
    for arm in ('A', 'C'):
        g = bench[bench.arm == arm].sort_values('F')
        Ps = sd.Plot(810, 300, x=(300, 4800, 'log'), y=(floor, 0.3, 'log'), title=f'chamber {arm}',
                     ylabel='singles split' if arm == 'A' else '', margin=(10, 30, 40, 104))
        Ps.xticks([(f, '') for f, _l in FX]).yticks([(1e-3, '0.1%'), (1e-2, '1%'), (1e-1, '10%')])
        Ps.line(list(g.F), [max(v, floor) for v in g.fsr], GREY, 3, '10 7', r=4.5, tip='bench: single donors split',
                tips=[f'bench, F = {f:g}\nsingle donors split {100 * v:.1f} % of {n}'
                      + (' (0: drawn at the floor)' if v == 0 else '') for f, v, n in zip(g.F, g.fsr, g.n1)])
        Ps.hline(CONTRACT, RED, '10 7', 2, label='contract 0.66 %', tip=G['contract'], anchor='start', where='below')
        Ps.vline(matched[arm], MUT, '4 5', 2, label=f'matched {matched[arm]}',
                 tip='The F used for the bench and contract runs.')
        if merged and arm in set(Dl.arm):
            h = Dl[Dl.arm == arm].sort_values('F')
            Ps.band(list(h.F), [max(v, floor) for v in h.frac_clean_split], list(h.clean_split_ul90), RED, 0.12,
                    tip='90 % CL upper limit (Clopper–Pearson)')
            Ps.line(list(h.F), [max(v, floor) for v in h.frac_clean_split], RED, 4,
                    tip='real triggers: clean singles split',
                    tips=[f'real triggers, F = {f:g}\nclean singles split {int(k)}/{int(n)} '
                          f'({100 * v:.2f} %, UL90 {100 * u:.2f} %)\n' + ('passes' if ok else 'fails') + ' the contract'
                          for f, k, n, v, u, ok in zip(h.F, h.clean_singles_split, h.clean_singles,
                                                       h.frac_clean_split, h.clean_split_ul90, h.passes)])
        Pe = sd.Plot(810, 300, x=(300, 4800, 'log'), y=(0, 0.8), xlabel='split threshold F',
                     ylabel='pairs resolved' if arm == 'A' else '', margin=(14, 30, 84, 104))
        Pe.xticks(FX).yticks([(v / 100, f'{v}%') for v in (0, 40, 80)])
        Pe.line(list(g.F), list(g.eff), GREEN, 4, tip='bench: real pairs resolved',
                tips=[f'bench, F = {f:g}\npairs resolved {100 * e:.0f} % (n = {n})' for f, e, n in zip(g.F, g.eff, g.n2)])
        Pe.vline(matched[arm], MUT, '4 5', 2)
        panels.append(sd.col(Ps.svg(f'singles split {arm}'), Pe.svg(f'pairs resolved {arm}'), gap=0, w=810))
    if merged:
        pick = R.ladder_pick(Dl)
        state = ', '.join(f'{a} F = {int(f)}' for a, f in sorted(pick.items()))
        sub = f'Lowest F meeting the contract on real triggers: {state}. Bench curves for the gain.'
        legend = [('bench: pairs resolved', GREEN), ('bench: single donors split', GREY, 'dash'),
                  ('real triggers: clean singles split', RED)]
        status = sd.callout(f'Picked by point estimate, with the 90 % CL upper limit in the tooltip. '
                            'Neighbouring F values are not statistically distinct.', RED, 24)
    else:
        sub = (f'The real-trigger rescan (condor cluster {LADDER_CLUSTER}) is still running; '
               'the bench shows what a lower F would buy.')
        legend = [('bench: pairs resolved', GREEN), ('bench: single donors split', GREY, 'dash')]
        status = sd.callout(f'<b>Pending.</b> One split-ab pass replays every F from 300 to 4800 exactly. '
                            'This slide fills in when it is merged: the real-trigger false-split curve goes '
                            'on top of these, against the 0.66 % contract.', GOLD, 24)
    body = sd.title('Open question: is C’s threshold too strict?', sub)
    body += sd.legend(legend) + sd.row(*panels, gap=44) + status
    D.slide('operating', body, f'''
<p>Sources: bench, <code>r3_scan.parquet</code> (fixed_f0); real triggers, <code>split_ab_ladder_{{A,C}}_7tags/summary_ladder.csv</code>.</p>
<p>Bench donors split more readily than clean singles on real triggers (C at 2400: 1.2 % vs 0.47 %). The matched thresholds were set on the bench, so they may be stricter than the real data needs, and the real-trigger rescan, not the bench, should set F.</p>
<p>How one pass covers every threshold: neither the split attempts nor their fstat depend on F, only acceptance does. <code>wft.reco.two_track_ladder</code> keeps both children of every attempt and, for each F, rebuilds the candidate lists, applies the worker’s lost-track revert and runs the selector. The replay was verified exact, event by event, against primary runs at F = 2400 and 300 before submission.</p>
<p>Check before believing it: each chamber’s ladder row at its matched F must reproduce <code>split_ab_fixed_&lt;arm&gt;_7tags/summary.csv</code> (events split, gaining a track, not recovered).</p>''',
            short='Operating point')


def s_close(D):
    items = [
        ('One sub-run, two chambers', 'run_145 stat090_0000, A and C. B and D are untested. All on the post-23-July (noisy) configuration.'),
        ('Real triggers have no truth', 'split-ab bounds the cost of a threshold. The gain comes only from overlays of clean donors, and real pairs are busier.'),
        ('Parallel co-located tracks', 'Degenerate at any Δt. They stay lost and must be carried as inefficiency.'),
        ('The toy is one plane', 'Equal charges, tied t0, no second view: the x/y pairing loss is not in R1–R2.'),
        ('Donor truth is a fit', 'A donor whose single-track fit is wrong gives a wrong label, and its twin inherits it.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:16px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:28px;font-weight:600;width:470px">{a}</p>'
                    f'<p style="font-size:24px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    dec = [('Operating F per chamber', 'From the rescan: the lowest F meeting the contract. A threshold choice, not a measurement.'),
           ('Ship or not', 'Bundles with profc pairing, the fixed-chain switches and F in the condor environment, the rescue floor, and a full re-pass.'),
           ('Flagged, outside this task', 'q_sum &gt; 10⁶ on 12–33 % of stage-3 tracks from unconstrained depth bins: their charge-derived quantities are meaningless.')]
    drows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:16px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:28px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:24px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in dec)
    body = (f'<div style="display:flex;gap:80px">'
            f'<div style="flex:1.25;display:flex;flex-direction:column;gap:8px">'
            f'<h2 style="font-size:52px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:8px">'
            f'<h2 style="font-size:52px;font-weight:600">Decisions</h2>{drows}</div></div>')
    D.slide('close', body, '''
<p>Handoff: <code>sept26_prelim_analysis/TWO_TRACK_LIMIT_RESUME.md</code>. Full record: <code>TWO_TRACK_FIT_LOG.md</code> (2026-09-29 → 2026-10-01). Long-form report with every table: <code>~/x17/sept26_prelim/two_track_limit/report/report.html</code>.</p>
<p>Dropped, with reasons in the log: a fractional model error ε (no gain at a fixed false-split rate) and a looser 1.2 mm guard (it blocks 42 % of false splits on real singles).</p>''',
            dark=True, short='Caveats & decisions')


# --------------------------------------------------------------------------- #
def build(out: Path) -> Path:
    F = figdir()
    L = R.load()
    BC = bench_counts()
    B = BC.pivot_table(index=['arm', 'variant'], columns='band', values='eff').reset_index()
    from sept26_prelim_analysis import intra_bench as ib
    C = pd.read_csv(ib.out_dir() / 'contract_fixed_vs_current.csv')
    C = C[C.tag == 'all']
    D = sd.Deck('Two-Track Limit',
                'Two tracks in one micro-TPC plane: the physical limit, the real-track limit, production, '
                'and an opt-in chain that passes the real-trigger contract (run_145, chambers A and C).')
    s_cover(D, B, C)
    s_problem(D)
    s_r1(D, F)
    s_synth(D, F)
    s_null(D, F)
    s_real(D, F)
    s_roc(D, F)
    s_chain(D)
    s_bench(D, BC)
    s_contract(D, C)
    s_operating(D, L)
    s_close(D)
    meta = dict(title='Two tracks in one plane: the real limit, and how close we get',
                summary='Physical limit ~1 strip pitch, real-track limit 2–3 mm; an opt-in fixed chain triples A’s '
                        'close-pair efficiency and passes the real-trigger contract. F rescan pending.',
                tags='X17,reconstruction,two-track', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by '
                                     'nTof_x17/sept26_prelim_analysis/make_two_track_deck.py with slidedoc.py.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (R.src_dir() / 'report' / 'two-track-limit.html')
    print('wrote', build(out))


if __name__ == '__main__':
    main()
