#!/usr/bin/env python3
"""
make_same_chamber_slides.py -- the context and the broadened-scope slides of the
two-track deck.

`make_two_track_deck` tells the core story: the physical limit (toy), how
production compares, what explains the gap. This module adds what happened
around it, from 2026-09-10 to 2026-10-07:

* where the question came from (the data symptoms, `intra_vertex`);
* a timeline of both threads and a map of the reconstruction chain;
* the overlay bench's first decomposition (2026-09-14) and its noise artefact
  (2026-09-29);
* the progression of the event-level bench over every step;
* x/y pairing;
* the single-track thread on `beam-off-cosmics` (T2): cosmic truth, the
  in-situ bundle, the 3-strip seeder, the beam angle scale;
* the join: what has to ride the one re-pass.

Inputs are the analyses' own outputs. The T2 products live with the other
branch's checkout, its work dir and its output dir; override with ``X17_T2_RESULTS``,
``X17_INSITU_WORK`` and ``X17_NTOF_COSMICS_OUT``. A slide whose input is missing is skipped with a message.
Numbers that exist only in a log (never written to a table) are quoted in
`LOGGED` with their source, so the record stays traceable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import slidedoc as sd
from slidedoc import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT, RULE

from sept26_prelim_analysis import paths

ARM_COL = dict(A=BLUE, C=ORANGE, D=PURPLE)
PCT_TICKS = [(v / 100, f'{v}%') for v in (0, 25, 50, 75, 100)]
T2_RESULTS = Path(os.path.expanduser(os.environ.get(
    'X17_T2_RESULTS', '~/PycharmProjects/nTof_x17/ntof_cosmics/results')))
INSITU = Path(os.path.expanduser(os.environ.get('X17_INSITU_WORK', '~/scratch/ntof_insitu')))
NC_OUT = Path(os.path.expanduser(os.environ.get('X17_NTOF_COSMICS_OUT', '/media/dylan/data/x17/ntof_cosmics')))

#: numbers that were measured but only written into a log, with where they are
LOGGED = dict(
    # TWO_TRACK_FIT_LOG.md 2026-09-29 (scratch selfnoise.py): 175 clean donors per arm,
    # an empty trigger's waveforms added on the donor's own region
    selfnoise={'A': dict(t0=(0.0, 0.65, 0.58), found=(1.00, 0.83, 0.86)),
               'C': dict(t0=(0.0, 0.62, 0.66), found=(0.994, 0.77, 0.77))},
    # TWO_TRACK_FIT_LOG.md 2026-09-30: held-out keep/swap decisions on stat090_0000
    # coincident donor pairs, calibrated on 0001 (A 2 437, C 695 pairs)
    pairing={'A': (0.867, 0.839, 0.922), 'C': (0.892, 0.901, 0.967)},
    # HANDOFF_TRACKING_2026-10-06 §10c: wall edges per arm (scint-stack fit_pointing, λ·k)
    wall={'A': (0.81, 0.89), 'C': (1.01, 1.06), 'D': (1.04, 1.20)},
)


def _have(*fs) -> bool:
    miss = [str(f) for f in fs if not Path(f).exists()]
    if miss:
        print('  skipped a slide, missing:', ', '.join(miss))
    return not miss


def pct(k: int, n: int) -> str:
    """k/n in whole per cent, rounded half up (the cover and the slides must agree)."""
    return f'{int(100 * k / n + 0.5 + 1e-9)}'


def _lt(s: str) -> str:
    return s.replace('&lt;', '<').replace('&ge;', '≥')


# --------------------------------------------------------------------------- #
# 1. where it started
# --------------------------------------------------------------------------- #
def s_origin(D):
    pv = paths.out('pair_vertex')
    fm, fs = pv / 'intra_multiplicity.csv', pv / 'intra_twotrack_separation.csv'
    if not _have(fm, fs):
        return
    M = pd.read_csv(fm)
    S = pd.read_csv(fs)
    cats = ['1 track in the chamber', '2 tracks', '3 or more tracks']
    labs = ['1 track', '2 tracks', '3 or more']
    P = sd.Plot(700, 560, x=(-0.5, 2.5), y=(0, 200), title='y at the capsule depth, robust σ',
                ylabel='[mm]', margin=(24, 30, 80, 90))
    P.yticks([(v, str(v)) for v in (0, 50, 100, 150, 200)])
    for i, (c, lab) in enumerate(zip(cats, labs)):
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 34, lab, 22, INK))
        for j, arm in enumerate(('A', 'C')):
            r = M[(M.arm == arm) & (M['sample'] == c)].iloc[0]
            P.vbar((P.X(i) + (j - 0.5) * 70,), r.rsig_y_at_capsule, 62, ARM_COL[arm], label=f'{r.rsig_y_at_capsule:.0f}',
                   tip=f'chamber {arm}, {lab}: {int(r.n):,} tracks\ny at capsule depth σ {r.rsig_y_at_capsule:.0f} mm, '
                       f'x {r.rsig_x_at_capsule:.0f} mm\nmedian strips y {r.med_y_strips:.0f}, x {r.med_x_strips:.0f}\n'
                       f'median χ²/dof y {r.med_chi2dof_y:.1f}, x {r.med_chi2dof_x:.1f}')
    # right: tracks in two-track chambers, by distance to the partner (y view), per mm
    Q = sd.Plot(900, 560, x=(1, 400, 'log'), y=(0.1, 300, 'log'), title='tracks in two-track chambers, by partner distance (y)',
                xlabel='distance to the other track on the strip plane [mm]', ylabel='tracks per mm',
                margin=(24, 30, 92, 100))
    Q.xticks([(t, str(t)) for t in (1, 4, 12, 24, 100, 400)]).yticks(
        [(0.1, '0'), (1, '1'), (10, '10'), (100, '100')])
    Q.band([1, 12], [0.1, 0.1], [300, 300], RED, 0.08,
           tip='Inside the 12 mm seed gap two tracks share one seed cluster and one window.')
    Q.text(1.15, 200, 'inside the 12 mm seed gap', 21, RED)
    for arm in ('A', 'C'):
        g = S[(S.arm == arm) & (S.view == 'y')].sort_values('sep_lo')
        lo = np.maximum(g.sep_lo.to_numpy(), 1.0)
        mid = np.sqrt(lo * g.sep_hi.to_numpy())
        dens = g.n.to_numpy() / (g.sep_hi - g.sep_lo).to_numpy()
        tips = [f'chamber {arm}, {a:g}–{b:g} mm: {int(n):,} tracks' + (' (0: drawn at the floor)' if n == 0 else '')
                for a, b, n in zip(g.sep_lo, g.sep_hi, g.n)]
        Q.line(list(mid), [max(v, 0.1) for v in dens], ARM_COL[arm], 4, None, r=6, tips=tips, tip=f'chamber {arm}')
    a12 = int(S[(S.arm == 'A') & (S.view == 'y') & (S.sep_hi <= 12)].n.sum())
    na = int(M[(M.arm == 'A') & (M['sample'] == '2 tracks')].n.iloc[0])
    body = sd.title('It started in the data: a second track in the chamber spoils both, and close pairs vanish',
                    'Campaign stage-3 tracks (33 runs), gated, pointing at the capsule. intra_vertex, 2026-09-13.')
    body += sd.legend([('chamber A', BLUE, 'box'), ('chamber C', ORANGE, 'box')]) + sd.row(P.svg('origin multiplicity'),
                                                                                          Q.svg('origin separation'), gap=64)
    D.slide('origin', body, f'''
<p>Same-chamber pairs (A–A, C–C, D–D) are about a third of the two-track sample, 19 707 of ~62 700 real pairs at the published 30 mm cut. They are the only pairs where both legs are measured by one chamber on one calibration, so the natural place to test whether two tracks share a vertex, and where small-opening-angle backgrounds (conversions, low-angle IPC) live. In September all three event-mixed vertex tests on them came out blind.</p>
<p><b>Left:</b> a track that shares its chamber with another is measured 3–4× worse in y at the capsule, its fit takes 2–3× the strips at 6–10× the χ²/dof. <b>Right:</b> of {na:,} tracks in two-track A chambers, {a12} has its partner within 12 mm in y. Close pairs are not reconstructed as two tracks at all. On the detector-A intra control, 0 real intra-A pairs below 20 mm (and 56 below 40 mm) were seen, where event mixing expects ~12 400.</p>
<p>The damage also <i>grew</i> with separation (A y: 79 mm at 16–24 mm, 142–146 mm beyond 24 mm), which is not what overlapping charge does. Four hypotheses followed (HANDOFF_INTRA_TWO_TRACK_RECO §4): H1 x/y swaps of time-degenerate tracks, H2a merging under the seed gap, H2b windows spanning both tracks, H3 two-track events are intrinsically messier. Only a truth sample separates them, which is why the overlay bench was built.</p>
<p>Source: <code>ntof_athens_26/pair_vertex_imaging/intra_vertex.py --multiplicity</code> → <code>&lt;out&gt;/pair_vertex/intra_multiplicity.csv</code>, <code>intra_twotrack_separation.csv</code>.</p>''',
            foot='intra_multiplicity.csv, intra_twotrack_separation.csv. Hover a bar or point for n, strips and χ²/dof.',
            short='Where it started')


# --------------------------------------------------------------------------- #
# 2. timeline
# --------------------------------------------------------------------------- #
TIMELINE = [
    # (date, thread, headline, detail)
    ('09-13', 'T1', 'Symptoms measured',
     'A second track spoils both (y σ ×3–4); 1 of 25 292 two-track A tracks has its partner within 12 mm.'),
    ('09-14', 'T1', 'Overlay bench; x/y pairing + rescue floor',
     'Production finds both tracks ≥ 24 mm in 47 % (A) / 39 % (C). Causes: x/y swaps, plane-wide floor, 12 mm seed gap. '
     'Pairing + rescue floor → 71 / 66 %, no production track lost.'),
    ('09-16', 'T1', 'Joint two-track fit (off by default)',
     'First pairs below 12 mm: 0 → 18 % (A), 37 % (C). Clean muons split ≤ 0.66 %. Compute declared not a constraint.'),
    ('09-29', 'T1', 'The bench doubled the noise',
     'Summing two triggers doubled the second donor’s noise; --overlay replace lifts ≥ 24 mm 73 → 85 % (A), 68 → 86 % (C).'),
    ('09-30', 'T1', 'The limit ladder, the fixed chain, profile pairing',
     'Physical limit ~1 strip pitch; real-track limit 2–3 mm (model mismatch). Four algorithmic losses, each fixed opt-in.'),
    ('10-01', 'T1', 'Contract passed, A and C',
     'split-ab on 7 tags (condor 4334051): clean singles split 0.50 / 0.47 %, no event loses a track.'),
    ('10-02', 'T1·T3', 'Pair angle · late tracks',
     'Close common-vertex pairs are parallel (divergence 0.13 d). Late tracks (t0 > 300 ns, 17 %) have unconstrained depth bins.'),
    ('10-06', 'T2', 'Cosmic truth for single-track angles',
     'run_149 A–C line: cosmic response ≠ beam; the bundle’s v = 42.6 prior is the bulk scale error; head-on loss = 5-strip seeder.'),
    ('10-07', 'T1', 'F rescan merged',
     'A 1200 / C 2400 are already the lowest F meeting the contract; one step lower fails.'),
    ('10-07', 'T2', 'Seeder 3 on beam · in-situ bundles · capsule k refuted',
     '+45/52/40 % scintillator-confirmed tracks; in-situ A closes on cosmics; walls give D_eff ≈ 330 mm. Beam vs cosmic gap open.'),
    ('10-07', 'T2', 'Beam/cosmic gap explained: electron scattering',
     'Cosmics read 1.15 at A’s wall (= the A–C line); in-beam muons read like beam-off ones. Geant4, ideal reco: the wall '
     'estimator reads 0.59 on beam electrons, 1.00 on muons. The cosmic scale stands.'),
]


def s_timeline(D):
    col = {'T1': BLUE, 'T2': GREEN, 'T1·T3': PURPLE}

    def card_(date, th, head, det):
        c = col[th]
        return (f'<div{sd.tipattr(det)} style="background:{sd.CARD};border:2px solid {c};border-radius:12px;'
                f'padding:9px 16px;display:flex;gap:16px;align-items:baseline">'
                f'<p style="font-size:22px;font-weight:600;color:{c};width:70px;flex:none">{date}</p>'
                f'<div><p style="font-size:23px;font-weight:600;line-height:1.2">{head}</p>'
                f'<p style="font-size:18px;color:{MUT};line-height:1.25">{det}</p></div></div>')
    left = [card_(*t) for t in TIMELINE if t[1] == 'T1']
    right = [card_(*t) for t in TIMELINE if t[1] != 'T1']
    hl = (f'<p style="font-size:26px;font-weight:600;color:{BLUE}">T1 · two-track separation '
          f'<span style="font-weight:400;color:{MUT};font-size:22px">branch two-track-joint-fit</span></p>')
    hr = (f'<p style="font-size:26px;font-weight:600;color:{GREEN}">T2 · single-track truth, T3 · late tracks '
          f'<span style="font-weight:400;color:{MUT};font-size:22px">branch beam-off-cosmics</span></p>')
    q = sd.callout('The starting question, Sept 2026: <b>what is the best that can be done in principle, how far '
                   'is the reconstruction from it, and what explains the difference?</b> T1 answered it for the '
                   'two-track step. The answer turned out to rest on the single-track reconstruction underneath, '
                   'which T2 now tests against cosmic truth.', GOLD, 22)
    body = sd.title('Two threads: find two tracks, then make each one right',
                    'Dated steps, 2026-09-13 → 10-07. Hover a card for the full numbers.')
    body += sd.row(sd.col(hl, *left, gap=8, w=880), sd.col(hr, *right, q, gap=8, w=740), gap=44)
    D.slide('timeline', body, '''
<p>The study began as one question about same-chamber coincident pairs: <b>what is the best that could be done in principle, how far is the reconstruction from it, and what explains the difference?</b> The first two weeks answered that for the two-track step itself (T1). That answer turned out to rest on things outside the two-track step: the single-track angle calibration, the seeder, and the depth profile of late tracks. Those became separate threads (T2 on the <code>beam-off-cosmics</code> branch, T3 late tracks) that now constrain T1.</p>
<p>Records: T1 <code>TWO_TRACK_FIT_LOG.md</code> and <code>TWO_TRACK_LIMIT_RESUME.md</code>; T2 <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §7–10; the map of both <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code> (on both branches); T3 <code>qsum_runaway.py</code>, OCTOBER_2026 O10.</p>''',
            short='Timeline')


# --------------------------------------------------------------------------- #
# 3. the reconstruction chain, stage by stage
# --------------------------------------------------------------------------- #
PIPE = [
    dict(stage='Seed', where='wft.seed · wft_beam',
         prod='hits clustered with a 12 mm gap; significance floor 10 % of the plane’s brightest strip; ≥ 5 strips',
         loss='close tracks share one window · a bright partner erases a faint track (~16 %) · head-on tracks (3–4 strips) never seeded',
         fix='rescue floor ✓ · 3-strip minimum ✓ beam-checked · seed split at 6/8 mm ✗',
         state=GREEN, tip='Rescue floor: WFT_SIG_FLOOR_LOCAL_MM=16, mode rescue (0 production tracks lost). '
                          'Seeder: WFT_BEAM_MIN_STRIPS=3. Splitting seed clusters lost 11–27 % of production tracks.'),
    dict(stage='One-track fit', where='forward model per window',
         prod='bench kernel with the v = 42.6 µm/ns Magboltz prior',
         loss='raw angles 4–12 % shallow against cosmic truth; the prior replaced the bench-fitted v',
         fix='in-situ v + robust kw per plane: A closes to ±3 % ✓; C built',
         state=GREEN, tip='wft_beam.make_bundle keeps the kernel but swaps the bench v (fitted with it) for the prior. '
                          'In situ: A ≈ 38, C ≈ 28 µm/ns.'),
    dict(stage='Two-track decision', where='wft.reco',
         prod='trigger → children seeded from the parent → fstat (scale χ²₁/dof) ≥ 300',
         loss='trigger fires on ~0 % of pairs < 12 mm · “X” basins · fstat diluted by the 2nd track',
         fix='fixed chain at F = 1200 (A) / 2400 (C) ✓ contract',
         state=GREEN, tip='WFT_TWO_TRACK_SCALE=two, WFT_TWO_TRACK_SEARCH=grid, no trigger, every candidate.'),
    dict(stage='x/y pairing', where='select_tracks',
         prod='greedy by χ² improvement among x/y candidates within ±120 ns',
         loss='swaps when both tracks are prompt: 24 % (A) / 23 % (C) of pairs',
         fix='charge pairing ✓ → + constrained depth profile ✓ (92 / 97 % correct)',
         state=GREEN, tip='xy_pairing bundle field; xy_pairing_{A,C}_profc.json.'),
    dict(stage='Depth profile', where='NNLS q(depth)',
         prod='unregularised; 18 bins of 60 ns after t0',
         loss='late tracks (t0 > 300 ns, 17 %): bins outside the window take unbounded charge, geometry unreliable',
         fix='depth-grid fix proposed, not built',
         state=GOLD, tip='qsum_runaway.py; OCTOBER_2026 O10. Also defeats charge-based x/y pairing (q_sum > 1e6 on 12–33 % of tracks).'),
    dict(stage='Angle scale', where='stage 3 · k_arm',
         prod='k from capsule pointing (A 1.27), assumes tracks from the beam axis at 234.6 mm',
         loss='capsule k refuted (walls: D_eff ≈ 330 mm); the walls read electron scattering, not angle truth',
         fix='adopt the cosmic in-situ scale (A–C line) for beam: proposed',
         state=GOLD, tip='Cosmic truth for A: 1.11 × production raw tan (A–C line), 1.15 at A’s wall. '
                         'Beam walls 0.89 and capsule 1.24–1.29 are set by the electrons’ scattering and population.'),
]


def s_pipeline(D):
    n = len(PIPE)
    cell = lambda inner, c, bg=sd.CARD, tip=None: (  # noqa: E731
        f'<div{sd.tipattr(tip)} style="background:{bg};border:2px solid {c};border-radius:12px;padding:12px 14px;'
        f'display:flex;flex-direction:column;gap:6px">{inner}</div>')
    hdr = [cell(f'<p style="font-size:26px;font-weight:600">{s["stage"]}</p>'
                f'<p style="font-size:18px;color:{MUT}">{s["where"]}</p>', INK, '#eef1f6', s['tip']) for s in PIPE]
    pr = [cell(f'<p style="font-size:19px;line-height:1.3">{s["prod"]}</p>', GREY) for s in PIPE]
    lo = [cell(f'<p style="font-size:19px;line-height:1.3">{s["loss"]}</p>', RED, '#fbf0f1') for s in PIPE]
    fx = [cell(f'<p style="font-size:19px;line-height:1.3;font-weight:600;color:{s["state"]}">{s["fix"]}</p>',
               s['state']) for s in PIPE]
    lab = lambda t, c: f'<p style="font-size:22px;font-weight:600;color:{c};align-self:center;text-align:right">{t}</p>'  # noqa: E731
    grid = (f'<div style="display:grid;grid-template-columns:150px repeat({n},1fr);gap:12px 12px">'
            + lab('stage', INK) + ''.join(hdr) + lab('production today', GREY) + ''.join(pr)
            + lab('found to cost', RED) + ''.join(lo) + lab('opt-in fix', GREEN) + ''.join(fx) + '</div>')
    body = sd.title('Every stage of the chain was examined; four have a validated fix, two a proposed one',
                    'How a chamber’s tracks are reconstructed now, what each stage was measured to cost a same-chamber '
                    'pair, and the state of its fix. Hover a stage for its switch.')
    body += grid + sd.legend([('validated, opt-in', GREEN, 'box'), ('proposed', GOLD, 'box'), ('open', RED, 'box')], 22)
    D.slide('pipeline', body, '''
<p>This is the map of the note. Left to right is the order in which an event is reconstructed (<code>wft</code>, then stage 3). Rows: what production does today, what this study measured it to cost, and the opt-in replacement.</p>
<p><b>Nothing here is in production.</b> Every fix is behind a switch that, when off, reproduces production output bit for bit. They are meant to ride a single campaign re-pass together (last slides), after one combined validation.</p>
<p>The first four stages are T1 and T2’s fixes; the last two are why the broadened scope matters: a two-track efficiency is not useful if the depth profile of a late track is unconstrained, and an opening angle is only as good as the angle scale under it. That scale is now understood (the cosmic in-situ scale is the truth; the beam walls read electron scattering) but not yet adopted.</p>''',
            short='The chain, stage by stage')


# --------------------------------------------------------------------------- #
# 4. the first decomposition on the overlay bench (09-14)
# --------------------------------------------------------------------------- #
def s_decomp(D):
    f = paths.out('intra_bench') / 'compare.csv'
    if not _have(f):
        return
    C = pd.read_csv(f)
    C = C[(C.variant == 'production') & (C.cls == 'coincident')]
    bands = ['<12 mm', '12-24 mm', '>=24 mm']
    blab = {'<12 mm': '< 12 mm', '12-24 mm': '12–24 mm', '>=24 mm': '≥ 24 mm'}
    cats = [('both_found', 'both found, paired', GREEN), ('merged', 'merged into one window', GREY),
            ('swapped', 'x/y swapped', RED), ('seed_lost', 'seed erased by the floor', GOLD)]
    # schematic of the bench
    o = []
    def wf(x, y, c, lab):
        o.append(f'<rect x="{x}" y="{y}" width="150" height="110" fill="#eef1f6" stroke="{RULE}"/>')
        o.append(sd.line(x + 40, y + 10, x + 80, y + 100, c, 5))
        o.append(sd.T(x + 75, y + 140, lab, 20, INK))
    wf(20, 30, BLUE, 'donor a (clean)')
    wf(20, 230, ORANGE, 'donor b (clean)')
    o.append(sd.T(95, 410, 'same chamber, file tag,', 18, MUT))
    o.append(sd.T(95, 432, 'trigger phase', 18, MUT))
    o.append(sd.arrow(180, 85, 250, 180, MUT) + sd.arrow(180, 285, 250, 200, MUT))
    o.append(f'<rect x="255" y="135" width="150" height="110" fill="#eef1f6" stroke="{RULE}"/>')
    o.append(sd.line(295, 145, 335, 235, BLUE, 5) + sd.line(320, 145, 360, 235, ORANGE, 5))
    o.append(sd.T(330, 275, 'overlay', 20, INK, tip='b’s waveforms placed on a’s trigger over b’s strip region '
                                                    '(--overlay replace since 09-29).'))
    o.append(sd.arrow(330, 290, 330, 350, MUT))
    o.append(f'<rect x="255" y="355" width="150" height="60" rx="10" fill="{sd.CARD}" stroke="{INK}" stroke-width="2"/>')
    o.append(sd.T(330, 392, 'reconstruct', 21, INK, weight=600))
    o.append(sd.arrow(330, 420, 330, 470, MUT))
    o.append(sd.T(330, 500, 'compare with', 20, INK))
    o.append(sd.T(330, 524, 'each donor’s own fit', 20, INK, tip='Truth is each donor’s single-track fit '
                                                                '(frozen pass). A wrong donor fit is a wrong label.'))
    schem = sd.svg(430, 560, ''.join(o), 'overlay bench')
    panels = []
    for arm in ('A', 'C'):
        P = sd.Plot(580, 540, x=(-0.5, 2.5), y=(0, 1), title=f'chamber {arm}, production',
                    ylabel='fraction of pairs' if arm == 'A' else '', margin=(24, 16, 70, 90 if arm == 'A' else 30))
        P.yticks(PCT_TICKS if arm == 'A' else [(t, '') for t, _ in PCT_TICKS])
        for i, b in enumerate(bands):
            r = C[(C.arm == arm) & (C.band == b)].iloc[0]
            P.raw(sd.T(P.X(i), P.y0 + P.ph + 34, blab[b], 21, INK))
            for j, (k, lab, c) in enumerate(cats):
                v = float(r[k])
                P.vbar((P.X(i) + (j - 1.5) * 36,), v, 32, c,
                       tip=f'chamber {arm}, {blab[b]}, production\n{lab}: {100 * v:.0f} % of {int(r.n_events)} pairs')
        panels.append(P.svg(f'decomp {arm}'))
    pa = C[(C.arm == 'A') & (C.band == '>=24 mm')].both_found.iloc[0]
    pc = C[(C.arm == 'C') & (C.band == '>=24 mm')].both_found.iloc[0]
    body = sd.title('A truth bench split production’s loss into mechanisms, none of them physical',
                    f'Production reco on overlays of two clean single tracks (run_145 stat090_0000), coincident pairs. '
                    f'Even ≥ 24 mm apart, both are found in only {100 * pa:.0f} % (A) / {100 * pc:.0f} % (C).')
    body += sd.legend([(lab, c, 'box') for _k, lab, c in cats], 22) + sd.row(schem, *panels, gap=28)
    D.slide('decomp', body, '''
<p>The bench (2026-09-14, <code>intra_bench.py</code>) overlays the decoded waveforms of two clean single-track triggers of the same chamber, file tag and trigger phase, then reconstructs the overlay with the production chain and compares against each donor’s own fit. Pairs are coincident (|Δt0| &lt; 30 ns), offset, or between. The harness reproduces production’s fits bit for bit on the donors themselves.</p>
<p>What it found, against the four hypotheses of the previous slide:</p>
<ol>
<li><b>H1 x/y swaps, confirmed.</b> When both tracks are prompt, <code>select_tracks</code> pairs the strongest x with the strongest y. Wrong in ~75 % of cases where the views rank the tracks differently.</li>
<li><b>H2a merging, confirmed.</b> Below 12 mm the seed clusterer makes one window, so at most one track.</li>
<li><b>A mechanism not on the list:</b> the seed significance floor is 10 % of the brightest strip of the <i>whole plane</i>, so a bright partner anywhere erases a fainter track (~16 % of tracks at ≥ 24 mm).</li>
<li><b>H2b not reproduced:</b> found tracks fit like singles. The data’s widened fits come from busier real events (H3).</li>
</ol>
<p>The categories overlap (a pair can have one track swapped and the other seed-lost), so the bars do not sum to 100 %. This is the first bench, on <code>--overlay add</code>; see the next slide for why its ≥ 24 mm numbers are pessimistic.</p>
<p>Source: <code>&lt;out&gt;/intra_bench/compare.csv</code>, variant production. Record: <code>wft/MULTITRACK_2026-09-14.md</code>, handoff §10.</p>''',
            foot='intra_bench compare.csv (production, coincident). Bars overlap in meaning, so they do not add to 100 %.',
            short='Truth bench')


# --------------------------------------------------------------------------- #
# 5. the bench noise artefact (09-29)
# --------------------------------------------------------------------------- #
def _event_eff(variant_dir: Path, arm: str, cls='coincident'):
    from sept26_prelim_analysis import intra_bench as ib
    base = ib.out_dir()
    d = base / variant_dir
    M = pd.read_parquet(d / 'overlays.parquet')
    C = pd.read_parquet(d / 'candidates.parquet')
    Dn = pd.read_parquet(d / 'donors.parquet' if (d / 'donors.parquet').exists() else base / 'donors.parquet')
    S = ib.score(M, C, Dn)
    o = S[(S['mode'] == 'overlay') & (S.cls == cls) & (S.arm == arm)]
    ev = o.groupby('oid').agg(sx=('sep_x', 'first'), sy=('sep_y', 'first'), both=('track_found', 'all'))
    sep = np.minimum(ev.sx, ev.sy)
    out = {}
    for lo, hi, b in ((0, 12, '&lt; 12 mm'), (12, 24, '12–24 mm'), (24, 1e9, '&ge; 24 mm')):
        m = (sep >= lo) & (sep < hi)
        out[b] = (int(ev.both[m].sum()), int(m.sum()))
    return out


def s_noise(D, prog):
    sn = LOGGED['selfnoise']
    rows = []
    for arm in ('A', 'C'):
        t0, fd = sn[arm]['t0'], sn[arm]['found']
        for k, lab in enumerate(('single donor', '+ a 2nd trigger’s noise', '+ noise, σ×√2 in the fit')):
            rows.append((f'{arm} · {lab}', t0[k], ARM_COL[arm] if k else GREY,
                         f'chamber {arm}, {lab}: t0 moves > 30 ns in {100 * t0[k]:.0f} % of fits, '
                         f'track found {100 * fd[k]:.0f} % (175 donors)'))
    left = sd.col(sd.p('t0 jumps by one 60 ns sample when a clean single track carries a second trigger’s noise', 24, INK, 600),
                  sd.hbars(rows, 1.0, width=300, h=28, label_w=380, size=22, fmt=lambda v: f'{100 * v:.0f} %'),
                  sd.p('…and the jump breaks the ±120 ns x/y coincidence, so the track is lost. Position, slope and '
                       'charge do not move.', 22, MUT), gap=16, w=820)
    # right: >= 24 mm before/after replace, computed from the two bench variants
    P = sd.Plot(700, 560, x=(-0.5, 1.5), y=(0, 1), title='coincident pairs ≥ 24 mm, both found',
                margin=(24, 30, 70, 90))
    P.yticks(PCT_TICKS)
    for i, arm in enumerate(('A', 'C')):
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 36, f'chamber {arm}', 23, ARM_COL[arm], weight=600))
        for j, (key, c, lab) in enumerate((('joint', GREY, 'add (summed)'), ('replace', ARM_COL[arm], 'replace'))):
            k, n = prog[arm][key]['&ge; 24 mm']
            P.vbar((P.X(i) + (j - 0.5) * 92,), k / n, 84, c, label=f'{100 * k / n:.0f}',
                   tip=f'chamber {arm}, overlay {lab}: {k}/{n} pairs both found and paired')
    right = sd.col(sd.legend([('overlay add', GREY, 'box'), ('overlay replace', INK, 'box')], 22), P.svg('noise fix'),
                   gap=6, w=720)
    body = sd.title('The first bench was pessimistic: it doubled the noise',
                    'Found 2026-09-29 on clean single donors; corrected with --overlay replace (a bench change, '
                    'not a reconstruction change).')
    body += sd.row(left, right, gap=80)
    D.slide('noise', body, '''
<p>The first bench summed donor b’s whole waveforms onto donor a’s over b’s region, so b’s strips carried two triggers’ noise. The fit is not told, and telling it (σ×√2) does not help. Measured directly by adding an empty trigger’s waveforms on a clean donor’s <i>own</i> region, with no second track (scratch <code>selfnoise.py</code>, numbers in <code>TWO_TRACK_FIT_LOG.md</code> 2026-09-29).</p>
<p><b>Fix (bench only, opt-in):</b> <code>build --overlay replace</code>: on b’s region outside a’s, b’s waveforms replace a’s. Only strips both regions share still carry two triggers’ noise, which is unavoidable there.</p>
<p>Consequences: every overlay number before 09-29 was low at ≥ 12 mm (below 12 mm the regions overlap anyway). What was left at ≥ 24 mm was x/y pairing (11.7 / 7.2 % swapped donor tracks, A / C), not a “flat algorithmic plateau”. And t0 is fragile at the one-sample level on real noise, so the ±120 ns x/y gate has little margin, in real events too.</p>''',
            foot='Left: TWO_TRACK_FIT_LOG 2026-09-29 (175 donors per arm). Right: intra_bench pairing_rescue16_two_final vs …_replace.',
            short='Bench noise')


# --------------------------------------------------------------------------- #
# 6. the progression on the event-level bench
# --------------------------------------------------------------------------- #
PROG_STEPS = [('production', '.', 'production', ''),
              ('rescue', 'pairing_rescue16_ranked', '+ pairing', '+ rescue floor'),
              ('joint', 'pairing_rescue16_two_final', '+ joint', 'fit (F 300)'),
              ('replace', 'pairing_rescue16_two_final_replace', 'bench fix:', 'overlay replace'),
              ('fixed', 'fixed_{arm}_replace', 'fixed', 'chain'),
              ('profc', 'fixed_{arm}_replace_profc', '+ profile', 'pairing')]
PROG_DATES = ['', '09-14', '09-16', '09-29', '09-30', '09-30']


def progression():
    from sept26_prelim_analysis import intra_bench as ib
    base = ib.out_dir()
    out = {}
    for arm in ('A', 'C'):
        out[arm] = {}
        for key, tmpl, *_ in PROG_STEPS:
            d = tmpl.format(arm=arm)
            if (base / d / 'overlays.parquet').exists():
                out[arm][key] = _event_eff(Path(d), arm)
    return out


def s_progress(D, prog):
    bands = [('&lt; 12 mm', RED), ('12–24 mm', GOLD), ('&ge; 24 mm', GREEN)]
    panels = []
    keys = [k for k, *_ in PROG_STEPS]
    for arm in ('A', 'C'):
        P = sd.Plot(810, 560, x=(-0.5, len(keys) - 0.5), y=(0, 1), title=f'chamber {arm}',
                    ylabel='pairs found and x/y-paired' if arm == 'A' else '', margin=(24, 30, 100, 104))
        P.yticks(PCT_TICKS)
        ir = keys.index('replace')
        P.band([ir - 0.4, ir + 0.4], [0, 0], [1, 1], GREY, 0.10,
               tip='Not a reconstruction change: the bench stopped doubling the second donor’s noise (09-29).')
        for i, (_k, _t, l1, l2) in enumerate(PROG_STEPS):
            P.raw(sd.T(P.X(i), P.y0 + P.ph + 30, l1, 19, INK))
            P.raw(sd.T(P.X(i), P.y0 + P.ph + 52, l2, 19, INK))
            if PROG_DATES[i]:
                P.raw(sd.T(P.X(i), P.y0 + P.ph + 76, PROG_DATES[i], 18, MUT))
        for b, c in bands:
            xs, ys, tips = [], [], []
            for i, k in enumerate(keys):
                if k not in prog[arm]:
                    continue
                kk, n = prog[arm][k][b]
                xs.append(i)
                ys.append(kk / n)
                tips.append(f'chamber {arm}, {_lt(b)}\n{PROG_STEPS[i][2]} {PROG_STEPS[i][3]}: {kk}/{n} = {100 * kk / n:.0f} %')
            P.line(xs, ys, c, 4, None, r=6.5, tips=tips, tip=_lt(b))
        panels.append(P.svg(f'progress {arm}'))
    a0, a1 = prog['A']['production']['&lt; 12 mm'], prog['A']['profc']['&lt; 12 mm']
    c0, c1 = prog['C']['production']['&lt; 12 mm'], prog['C']['profc']['&lt; 12 mm']
    body = sd.title(f'On the bench, pairs under 12 mm went from {pct(*a0)} to {pct(*a1)} % (A) '
                    f'and {pct(*c1)} % (C)',
                    'The same coincident overlays through every configuration, in the order they were built. '
                    'Both donors found <i>and</i> correctly x/y-paired.')
    body += sd.legend([(_lt(b), c) for b, c in bands]) + sd.row(*panels, gap=44)
    D.slide('progress', body, '''
<p>Each point is the same set of coincident overlays (run_145 stat090_0000, stratified by separation; hover for k/n) reconstructed with one more change. Separation is the smaller of the x and y mesh separations.</p>
<ol>
<li><b>+ pairing + rescue floor</b> (09-14): x/y pairing by charge for time-degenerate tracks; a local significance floor that adds clusters the plane-wide floor erased, ranked below production’s. Lifts ≥ 24 mm; nothing below 12 mm, where one window holds both.</li>
<li><b>+ joint fit</b> (09-16): a two-track fit inside one window, at production’s F = 300 / 120 with its trigger. The first pairs below 12 mm.</li>
<li><b>Bench fix</b> (09-29, grey column): the overlay stopped doubling the noise. Same reconstruction.</li>
<li><b>Fixed chain</b> (09-30): no trigger, every candidate, grid search, two-track noise scale, at F matched to production’s false-split rate (A 1200, C 2400).</li>
<li><b>+ profile pairing</b>: the x/y cost gains a constrained depth-profile term (<code>xy_pairing_*_profc.json</code>).</li>
</ol>
<p>C is held back below 12 mm by its stricter threshold: the overlay scan says C resolves 58 % at F = 2400, 69 % at 1200. The real-trigger rescan says F = 2400 is nonetheless the lowest that meets the contract (Operating point). Parallel, co-located tracks are degenerate at any Δt and stay lost.</p>
<p>Variants: <code>production</code> (bench root), <code>pairing_rescue16_ranked</code>, <code>pairing_rescue16_two_final</code>, <code>…_replace</code>, <code>fixed_{A,C}_replace</code>, <code>fixed_{A,C}_replace_profc</code> under <code>&lt;out&gt;/intra_bench/</code>. The first three are on <code>--overlay add</code>.</p>''',
            foot='intra_bench variants, scored by intra_bench.score. Grey column: a bench correction, not a reconstruction change.',
            short='Bench progression')


# --------------------------------------------------------------------------- #
# 7. x/y pairing
# --------------------------------------------------------------------------- #
def s_pairing(D):
    pr = LOGGED['pairing']
    labs = [('charge ratio (production)', GREY), ('constrained charge', '#9fb7d6'), ('+ depth profile', GREEN)]
    P = sd.Plot(820, 560, x=(-0.5, 1.5), y=(0.8, 1.0), title='coincident donor pairs: correct keep/swap decision',
                margin=(24, 30, 70, 100), ylabel='decisions correct')
    P.yticks([(v / 100, f'{v}%') for v in (80, 85, 90, 95, 100)])
    for i, arm in enumerate(('A', 'C')):
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 36, f'chamber {arm}', 23, ARM_COL[arm], weight=600))
        for j, (lab, c) in enumerate(labs):
            v = pr[arm][j]
            P.vbar((P.X(i) + (j - 1) * 96,), v, 86, c, base=0.8, label=f'{100 * v:.1f}',
                   tip=f'chamber {arm}, {lab}: {100 * v:.1f} % correct (held out on stat090_0000, calibrated on 0001)')
    side = sd.col(
        sd.p('Two prompt tracks give two x and two y candidates, and timing cannot tell which x goes with which y. '
             'Both views see the <i>same</i> drifting charge, so each track’s depth profile q(t) should match '
             'across views.', 24),
        sd.p('Compared bin by bin in each view’s own t0 frame, profiles <b>fail</b> (72 / 80 %): t0 slides by whole '
             'bins between views. On a common absolute-time grid they carry real information.', 24),
        sd.callout('The floor for C: some tracks disagree with themselves across views (x/y charge 2069/515). '
                   'No pairing built on shared charge can place them.', ORANGE, 23),
        sd.callout('Open: the 3-strip seeder re-pairs 5–13 % of production’s tracks in busy events. '
                   'Validate it together with this pairing.', RED, 23),
        gap=20, w=740)
    body = sd.title('Pairing x with y by the shared depth profile removes a third of the wrong pairings',
                    'The second limit: a pair is only found if both views resolve it <i>and</i> the views are paired correctly.')
    body += sd.legend([(l_, c, 'box') for l_, c in labs], 22) + sd.row(P.svg('pairing'), side, gap=64)
    D.slide('pairing', body, '''
<p>Decision: for two coincident clean donors, keep or swap the y partners; the lower summed cost wins. Production’s cost is the x/y charge ratio (<code>lq</code>) plus time. The <b>constrained</b> versions zero the depth bins the window cannot see before summing (<code>wft.reco.constrained_charge</code>, feature <code>lqc</code>), because on 12–33 % of stage-3 tracks per view those bins hold runaway charge (q_sum &gt; 10⁶) and the charge ratio saturates. The profile term is the distance between the two views’ constrained, normalised profiles on an absolute-time grid, smoothed by 1–2 bins.</p>
<p>Held-out numbers (calibrated on stat090_0001, tested on 0000; <code>TWO_TRACK_FIT_LOG.md</code> 2026-09-30). On the event-level bench at ≥ 24 mm, A’s swapped donor tracks fall 12.2 → 7.2 % and efficiency rises 85 → 91 %; C barely moves, for the reason in the callout.</p>
<p>The x/y charge ratio is flat with position (median log ratio within ±0.05 across the chamber), so a gain map would not help. Its per-track spread is 0.48 (A) / 0.67 (C) on all gated tracks against 0.12–0.20 on clean donors.</p>
<p>Not done: a joint x–y fit with each track’s profile shared between views (handoff §1.3), the only idea that adds information rather than a better cost.</p>''',
            foot='TWO_TRACK_FIT_LOG 2026-09-30 (scratch profile_pairing.py): A 2 437, C 695 coincident donor pairs.',
            short='x/y pairing')


# --------------------------------------------------------------------------- #
# 8. the threads and how they constrain each other
# --------------------------------------------------------------------------- #
def s_threads(D):
    W, H = 1664, 700
    o = []
    BW = 600
    boxes = [(0, 20, 'T1 · two-track separation', BLUE, 'branch two-track-joint-fit',
              ['Two tracks found when there are two, and only then?',
               'Physical limit ~1 pitch; real-track limit 2–3 mm.',
               'Fixed chain passes the contract at F 1200 / 2400,',
               'set on the v = 42.6 bundles.']),
             (W - BW, 20, 'T2 · single-track angle truth', GREEN, 'branch beam-off-cosmics',
              ['Is each angle right, near normal included?',
               'Truth: the A–C line on run_149 cosmics.',
               'v prior was the scale error; seeder 3 finds head-on',
               'tracks; beam/cosmic gap = electron scattering.']),
             ((W - BW) / 2, 470, 'T3 · late tracks', PURPLE, 'branch beam-off-cosmics',
              ['Is the depth profile observable for t0 > 300 ns?',
               'No: NNLS fills bins the window cannot see (17 %).'])]
    for x, y, t, c, br, lines in boxes:
        w, h = BW, 92 + 32 * len(lines)
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{sd.CARD}" stroke="{c}" stroke-width="3"/>')
        o.append(sd.T(x + 22, y + 40, t, 28, c, 'start', 600))
        o.append(sd.T(x + 22, y + 68, br, 18, MUT, 'start'))
        for i, l_ in enumerate(lines):
            o.append(sd.T(x + 22, y + 106 + 32 * i, l_, 21, INK, 'start'))
    links = [
        ((BW, 75), (W - BW, 75), 'shared seeder', 'T2 changes the minimum 5 → 3. T1’s rescue floor, split seeding and the five candidate '
                                                   'slots act on the same list; min 3 re-pairs 5–13 % of tracks in busy events. '
                                                   'Run both through the split-ab contract together.'),
        ((W - BW, 130), (BW, 130), 'bundles move F', 'T1’s Δχ² and F were calibrated on v = 42.6 bundles. T2 changes v (and C’s kernel). '
                                                    'Re-derive F on the new bundles; do not carry 1200 / 2400 across. Part of T1’s '
                                                    'model mismatch may be the same mismatch: a hypothesis to test.'),
        ((BW, 185), (W - BW, 185), 'near-normal pairs need both', 'T1’s oracle and bench sit at tan ≈ 0.3; capsule pairs near the axis are '
                                                                 'near normal, where production loses the track before any pair logic runs.'),
        ((W - BW, 240), (BW, 240), 'angle errors, cosmic donors', 'T1 should use σ(|tan|) from T2’s resolution.csv, not the constant tan_err '
                                                                 '(pulls 9–16 near normal). run_149 through-goers could be bench donors with true angles.'),
    ]
    for (x1, y1), (x2, y2), lab, tip in links:
        o.append(sd.arrow(x1, y1, x2 - 6 if x2 > x1 else x2 + 6, y2, INK, 2.5, 14))
        o.append(sd.T((x1 + x2) / 2, y1 - 10, lab, 22, INK, 'middle', 600, tip=tip))
    o.append(sd.arrow(600, 470, 300, 262, PURPLE, 2.5, 14) + sd.arrow(1064, 470, 1364, 262, PURPLE, 2.5, 14))
    o.append(sd.T(440, 440, 'contaminates both:', 21, PURPLE, 'end', 600,
                  tip='Flag t0 > 300 ns before quoting a pair angle or a two-track efficiency.'))
    o.append(sd.T(440, 466, 'flag t0 > 300 ns', 21, PURPLE, 'end'))
    body = sd.title('Two tracks need two right single tracks: the question split into three threads',
                    'What each thread asks and has established, and how they constrain each other. Hover a link.')
    body += sd.svg(W, H, ''.join(o), 'threads')
    D.slide('threads', body, '''
<p>The goal is unchanged: reconstruct two tracks from one vertex in <b>one</b> chamber, with correct angles. T1 answered “can we separate them”. Doing so showed it rests on the single-track reconstruction, which T2 tests against an independent truth: a cosmic muon crossing chambers A and C is one straight line, so each chamber’s own track must lie along the line joining the two (run_149, 87 sub-runs, 1.9 M triggers, beam off).</p>
<p>The two branches share no code; they meet in the seeder and the calibration bundles. Map: <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code> (identical on both branches).</p>''',
            short='Three threads')


# --------------------------------------------------------------------------- #
# 9. T2: the in-situ bundle against cosmic truth
# --------------------------------------------------------------------------- #
RESP_BINS = [(0.08, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.6)]


def insitu_response():
    T = pd.read_parquet(INSITU / 'truth.parquet')
    out = []
    for lab, arm, name in (('prod_s3', 'A', 'production bundle'), ('is2_s3', 'A', 'in-situ bundle'),
                           ('is2_s3', 'C', 'in-situ bundle')):
        f = INSITU / f'reco_{lab}_{arm}.parquet'
        if not f.exists():
            continue
        R = pd.read_parquet(f)
        M = T[(T.arm == arm) & ~T.train].merge(R, on=['subrun', 'event_id'])
        for ax in 'xy':
            t = M[f'tan_{ax}'].to_numpy()
            r = M[f'{ax}_tan_theta'].to_numpy() / t
            for lo, hi in RESP_BINS:
                m = (np.abs(t) >= lo) & (np.abs(t) < hi) & np.isfinite(r)
                res = M[f'{ax}_tan_theta'].to_numpy()[m] - t[m]
                out.append(dict(label=lab, arm=arm, name=name, ax=ax, lo=lo, hi=hi, n=int(m.sum()),
                                ratio=float(np.median(r[m])) if m.any() else np.nan,
                                sig=float(1.4826 * np.median(np.abs(res - np.median(res)))) if m.any() else np.nan))
    return pd.DataFrame(out)


def s_insitu(D):
    if not _have(INSITU / 'truth.parquet', INSITU / 'reco_is2_s3_A.parquet'):
        return
    R = insitu_response()
    ser = [('prod_s3', 'A', 'A, production bundle', GREY, '10 7'), ('is2_s3', 'A', 'A, in-situ bundle', BLUE, None),
           ('is2_s3', 'C', 'C, in-situ bundle', ORANGE, None)]
    panels = []
    for ax, lab in (('x', 'x view'), ('y', 'y view')):
        P = sd.Plot(590 if ax == 'x' else 520, 500, x=(0.05, 0.62), y=(0.8, 1.15), title=lab,
                    xlabel='true |tan θ| (A–C line)', ylabel='reconstructed / true tan' if ax == 'x' else '',
                    margin=(24, 24, 84, 100 if ax == 'x' else 30))
        P.xticks([(v, f'{v:g}') for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]).yticks(
            [(v, f'{v:.2f}') for v in (0.8, 0.9, 1.0, 1.1)])
        P.band([0.05, 0.62], [0.97, 0.97], [1.03, 1.03], GREEN, 0.10, tip='±3 %')
        P.hline(1.0, INK, '4 5', 1.5)
        for labk, arm, name, c, dash in ser:
            g = R[(R.label == labk) & (R.arm == arm) & (R.ax == ax)]
            xs = list((g.lo + g.hi) / 2)
            tips = [f'{name}, {lo:g}–{hi:g}: median {r:.3f}, σ_tan {s:.3f} (n = {n})'
                    for lo, hi, r, s, n in zip(g.lo, g.hi, g.ratio, g.sig, g.n)]
            P.line(xs, list(g.ratio), c, 4, dash, r=6, tips=tips, tip=name)
        panels.append(P.svg(f'insitu {ax}'))
    pa = R[(R.label == 'prod_s3') & (R.arm == 'A') & (R.lo >= 0.15)].ratio
    side = sd.col(
        sd.p(f'Production’s raw angles read <b>{100 * (1 - pa.max()):.0f}–{100 * (1 - pa.min()):.0f} % shallow</b> '
             'against the cosmic line, and fall with angle.', 24),
        sd.p(f'Cause: the bundle keeps the bench kernel but swaps the bench-fitted '
             f'{sd.term("v", "Drift velocity. Bench det3 36.6, det6 26.7 µm/ns, fitted together with the kernel; in situ A ≈ 38, C ≈ 28.")} '
             'for the 42.6 µm/ns prior. 42.6/36.6 ≈ 1.16 and 42.6/26.7 ≈ 1.60 are most of k_arm.', 24),
        sd.callout('In-situ recipe: geometric v from free fits, robust kw per plane, seeder 3. '
                   'A y flat at 1.00, A x within ±3 %.', GREEN, 23),
        gap=18, w=500)
    body = sd.title('On cosmic truth the in-situ bundle reads true angles; production’s were shallow',
                    'run_149 beam-off through-goers, held-out third (truth = line joining chambers A and C, 469 mm lever). '
                    'Seeder 3 for both.')
    body += sd.legend([(n, c, 'dash' if d else 'line') for _l, _a, n, c, d in ser], 22) + sd.row(*panels, side, gap=24)
    D.slide('insitu', body, '''
<p>Truth (<code>ntof_cosmics/insitu_calib.py truth</code>): 815 clean A–C crossings per arm from 14 run_149 sub-runs; a third trains the calibration, the rest is held out. The lever between chambers is 469 mm, so the joined line’s slope is known to ~0.002.</p>
<p>Bundles: production’s run_145 bundle (<code>prod_s3</code>, same seeder 3), and <code>is2_A</code> (production kernel, v 38.0, kw x 1.020 / y 0.993 from <code>insitu_calib.py kwmed</code>, a robust median over |tan| 0.15–0.45). <code>is2_C</code>: the r06 det7 kernel, v 28.7, kw x 1.001 / y 0.957. A least-squares kw (<code>w0kw</code>) is pulled by the tails and over-corrects A y by 7 %.</p>
<p>Ruled out as the cause, with bench M3 truth: S/N ÷ 8 (n_TOF S/N is 7–10× below bench), cropping to 20 samples, real n_TOF noise; and the template, ZS (off), the fit window, the w-scan range. A ref-pinned in-situ hyper fit is <b>not</b> the fix: its v = 39.2 is the χ²(v)-valley bias, so v comes from free fits against truth.</p>
<p>Chamber C’s 8–13 % core tails are not from the kernel; candidates are x/y noise, dead or hot strips, and an outward-going sign effect.</p>
<p>What this does not do: production then multiplies raw angles by the stage-3 capsule k (A 1.27), so production’s <i>final</i> angles read steep, not shallow. That k is the subject of the beam-scale slide.</p>''',
            foot='~/scratch/ntof_insitu truth.parquet, reco_{prod_s3,is2_s3}_{A,C}.parquet, test split. Hover a point for σ_tan and n.',
            short='T2: in-situ bundle')


# --------------------------------------------------------------------------- #
# 10. T2: the seeder
# --------------------------------------------------------------------------- #
def s_seeder(D):
    f3, f5 = INSITU / 'seedtest_m3.parquet', INSITU / 'seedtest_m5.parquet'
    fs = T2_RESULTS / 'seed_beam' / 'data' / 'scint_confirmation.csv'
    if not _have(f3, f5, fs):
        return
    bins = [(0, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 0.2), (0.2, 0.3), (0.3, 0.45)]
    S = {}
    for k, f in ((5, f5), (3, f3)):
        d = pd.read_parquet(f)
        d = d[(d.n_A == 1) & (d.n_C == 1)]
        t = d.jx.abs().to_numpy()
        core = (t > 0.12) & (t < 0.45)
        kk = float(np.median(d.A_raw_x[core] / d.jx[core]))
        rows = []
        for lo, hi in bins:
            m = (t >= lo) & (t < hi)
            r = (d.A_raw_x.to_numpy()[m] / kk) - d.jx.to_numpy()[m]
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                             sig=float(1.4826 * np.median(np.abs(r - np.median(r)))) if m.sum() >= 3 else np.nan))
        S[k] = (len(d), pd.DataFrame(rows))
    col = {5: GREY, 3: GREEN}
    lab = {5: 'production, ≥ 5 strips', 3: '≥ 3 strips'}
    # counts per |tan| bin (bar pairs)
    P = sd.Plot(560, 480, x=(-0.5, len(bins) - 0.5), y=(0, 400), title='one-track A–C crossings',
                ylabel='crossings', margin=(24, 16, 92, 92))
    P.yticks([(v, str(v)) for v in (0, 100, 200, 300, 400)])
    for i, (lo, hi) in enumerate(bins):
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 30, f'{lo:g}–', 17, INK) + sd.T(P.X(i), P.y0 + P.ph + 50, f'{hi:g}', 17, INK))
        for j, k in enumerate((5, 3)):
            n = int(S[k][1].n.iloc[i])
            P.vbar((P.X(i) + (j - 0.5) * 28,), n, 26, col[k], tip=f'{lab[k]}, true |tan x| {lo:g}–{hi:g}: {n} crossings')
    P.raw(sd.T(P.x0 + P.pw / 2, P.y0 + P.ph + 80, 'true |tan θ| in x', 20, MUT))
    # resolution per bin
    Q = sd.Plot(560, 480, x=(-0.5, len(bins) - 0.5), y=(0.01, 1, 'log'), title='chamber A x: σ_tan against truth',
                ylabel='σ_tan (robust)', margin=(24, 16, 92, 100))
    Q.yticks([(0.01, '0.01'), (0.03, '0.03'), (0.1, '0.1'), (0.3, '0.3'), (1, '1')])
    for i, (lo, hi) in enumerate(bins):
        Q.raw(sd.T(Q.X(i), Q.y0 + Q.ph + 30, f'{lo:g}–', 17, INK) + sd.T(Q.X(i), Q.y0 + Q.ph + 50, f'{hi:g}', 17, INK))
    Q.raw(sd.T(Q.x0 + Q.pw / 2, Q.y0 + Q.ph + 80, 'true |tan θ| in x', 20, MUT))
    for k in (5, 3):
        g = S[k][1]
        ok = g.n >= 3
        Q.line([i for i in range(len(g)) if ok.iloc[i]], list(g.sig[ok]), col[k], 4, None, r=6,
               tips=[f'{lab[k]}, |tan| {lo:g}–{hi:g}: σ_tan {s:.3f} (n = {n})'
                     for lo, hi, s, n in zip(g.lo[ok], g.hi[ok], g.sig[ok], g.n[ok])], tip=lab[k])
    # beam: scintillator-confirmed excess
    C = pd.read_csv(fs).set_index(['arm', 'sample'])
    rows, gains = [], []
    for arm in ('A', 'C', 'D'):
        p0 = int(C.loc[(arm, 'production'), 'wall_excess_n'])
        p3 = int(C.loc[(arm, 'min 3, all'), 'wall_excess_n'])
        gains.append(p3 / p0 - 1)
        rows.append((f'{arm} production', p0, GREY, f'chamber {arm}, production: {p0:,} wall-confirmed tracks above accidentals'))
        rows.append((f'{arm} ≥ 3 strips', p3, ARM_COL[arm], f'chamber {arm}, seeder 3: {p3:,} (+{100 * (p3 / p0 - 1):.0f} %)'))
    beam = sd.col(sd.p('Beam, run_145: tracks confirmed by the SiPM wall they point at', 22, INK, 600),
                  sd.hbars(rows, max(r[1] for r in rows), width=250, h=24, label_w=190, size=20), gap=12, w=520)
    n5, n3 = S[5][0], S[3][0]
    gtxt = '/'.join(f'{100 * g:.0f}' for g in gains)
    body = sd.title(f'The head-on loss was the seeder: 3 strips instead of 5 gives ×{n3 / n5:.1f} cosmic crossings, '
                    f'+{gtxt} % confirmed beam tracks',
                    'Same triggers, same bundle and fit; only the beam seeder’s minimum cluster size changes '
                    '(opt-in WFT_BEAM_MIN_STRIPS=3).')
    body += sd.legend([(lab[5], GREY, 'box'), (lab[3], GREEN, 'box')], 22) + sd.row(P.svg('seed counts'), Q.svg('seed sigma'),
                                                                                  beam, gap=24)
    D.slide('seeder', body, f'''
<p>At n_TOF signal-to-noise a near-normal track puts only 3–4 strips over threshold, and the beam seeder (<code>MIN_STRIPS_BEAM = 5</code>, set against single-strip junk) never seeds it; the bench seeder uses 3. Cosmic test (<code>ntof_cosmics/seed_test.py</code>): one-track A–C crossings {n5} → {n3}; below |tan| 0.08 the crossings that exist are badly measured at 5 strips and well measured at 3. Only |tan| &lt; 0.02 stays poor.</p>
<p>Beam test (<code>seed_beam_test.py</code>, run_145 stat090_0000, all 7 tags, every trigger; report <code>ntof_cosmics/results/seed_beam/report.html</code>): confirmation = the track extrapolates to the SiPM-wall group that fired, minus the accidental rate from a shifted control. <b>No particle is lost:</b> of production tracks without a min-3 match, nearly all reappear with one view re-paired to another partner; only 22 (A) / 5 (C) vanish, in busy, junk-heavy events. Gained tracks are later and less often confirmed, so real but dirtier.</p>
<p>For same-chamber pairs: every two-track event is still ≥ 24 mm apart (median ~210 mm). The seeder neither makes close fake pairs nor recovers close pairs; it does re-pair 5–13 % of production tracks in busy events, which is T1’s x/y pairing problem. Validate both together.</p>''',
            foot='~/scratch/ntof_insitu seedtest_m{5,3}.parquet (σ: raw tan rescaled by its core median ratio) · seed_beam scint_confirmation.csv.',
            short='T2: seeder')


# --------------------------------------------------------------------------- #
# 11. T2: the beam angle scale
# --------------------------------------------------------------------------- #
def _kbeam():
    f = INSITU / 'kbeam_AC.txt'
    out = {}
    if not f.exists():
        return out
    for ln in f.read_text().splitlines():
        p_ = ln.split()
        if len(p_) == 5 and p_[0] in ('prod', 'is2', 'm3') and p_[1] in 'AC':
            out[(p_[0], p_[1])] = (float(p_[3]), float(p_[4]))
    return out


def _g4_wall():
    """Geant4, ideal reconstruction: what the data's wall estimator reads per population (truth = 1)."""
    fn, fs = NC_OUT / 'g4_angle' / 'neutrons_wall_estimator.csv', NC_OUT / 'g4_angle' / 'single_summary.csv'
    if not _have(fn, fs):
        return []
    N = pd.read_csv(fn).set_index('sample').true_over_raw
    S = pd.read_csv(fs)
    mu = S[(S.particle == 'mum') & (S.tan_gun > 0)].wall_over_gun.median()
    return [('muons (1 GeV guns)', float(mu), BLUE, 'μ⁻ guns at tan 0.1–0.5: wall tan / gun tan.'),
            ('electrons > 4 MeV', float(N['A+C, KE > 4 MeV']), GREEN,
             'n_TOF capture campaign, A+C, electrons above 4 MeV at the gap.'),
            ('electrons 2–4 MeV', float(N['A+C, KE 2-4 MeV']), GOLD, 'Same, 2–4 MeV at the gap.'),
            ('all beam electrons', float(N['A+C, all']), RED,
             'Same, all gap electrons (median ~3 MeV): the population the beam wall test sees.')]


def s_beamscale(D):
    fw = T2_RESULTS / 'wall_edge_scale' / 'wall_edge_scale.csv'
    fk = T2_RESULTS / 'tracking' / 'pooled' / 'k_true_by_borrowed_k.csv'
    fc = NC_OUT / 'cosmic_wall_scale' / 'summary.json'
    if not _have(fw, fk, fc):
        return
    W = pd.read_csv(fw)
    wa = W[(W.arm == 'A') & (W['sample'] == 'all') & (W.boundary.str.startswith('outer'))].iloc[0]
    K = pd.read_csv(fk)
    cw = json.loads(fc.read_text())
    kb = _kbeam()
    rows = []   # (y, label, lo, hi, point, colour, tip)
    for arm in ('A', 'C'):
        ck = K[(K.arm == arm) & (K.k_from == 'run_147')].set_index('axis').k_true
        wl = LOGGED['wall'][arm]
        wp = float(wa.true_over_raw) if arm == 'A' else sum(wl) / 2
        rows.append((arm, 'SiPM-wall edges (beam)', wl[0], wl[1], wp, GREEN,
                     f'chamber {arm}: scintillator wall boundaries, beam tracks.\n'
                     + (f'u-binned, offset-free: {wp:.3f}, D_eff {wa.D_eff:.0f} mm ({int(wa.n):,} tracks)\n' if arm == 'A' else '')
                     + f'range over estimators (λ·k, pass 1–2): {wl[0]:.2f}–{wl[1]:.2f}'))
        if arm == 'A':
            rows.append((arm, 'SiPM-wall edges (cosmics)', cw['boot_lo'], cw['boot_hi'], cw['s_best'], PURPLE,
                         f'chamber A: run_149 cosmics that fire A’s wall, edge likelihood (fit_wall_u).\n'
                         f'best {cw["s_best"]:.2f}, bootstrap {cw["boot_lo"]:.2f}–{cw["boot_hi"]:.2f}, '
                         f'{cw["n_single_x"]:,} tracks; the beam value is excluded by ΔNLL {cw["dnll_at_beam"]:.0f}.'))
        rows.append((arm, 'cosmic A–C line', min(ck), max(ck), float(ck.mean()), BLUE,
                     f'chamber {arm}: run_149 through-goers, k_true x {ck["x"]:.3f}, y {ck["y"]:.3f}'))
        if ('prod', arm) in kb:
            b, t = kb[('prod', arm)]
            rows.append((arm, 'capsule pointing (k_arm)', min(b, t), max(b, t), (b + t) / 2, RED,
                         f'chamber {arm}: capsule estimators on run_145, band {b:.2f}, track {t:.2f}. '
                         'Assumes tracks from the beam axis at 234.6 mm.'))
    na = sum(r[0] == 'A' for r in rows)
    P = sd.Plot(960, 600, x=(0.7, 1.9), y=(-0.5, len(rows) - 0.5), xlabel='true tan / production raw tan',
                margin=(24, 30, 84, 330))
    P.xticks([(v, f'{v:.1f}') for v in (0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9)])
    P.vline(1.0, MUT, '4 5', 1.5)
    for i, (arm, lab, lo, hi, pt, c, tip) in enumerate(rows):
        y = len(rows) - 1 - i
        P.raw(sd.T(P.x0 - 20, P.Y(y) + 8, f'{arm} · {lab}', 22, ARM_COL[arm] if lab.startswith('SiPM') else INK, 'end',
                   600 if lab.startswith('SiPM') else 400))
        P.raw(sd.line(P.X(lo), P.Y(y), P.X(hi), P.Y(y), c, 6))
        P.points([pt], [y], c, r=11, tips=[tip])
    P.raw(sd.line(P.x0 - 320, P.Y(len(rows) - na - 0.5), P.x0 + P.pw, P.Y(len(rows) - na - 0.5), RULE, 1.5))
    g4 = _g4_wall()
    side = sd.col(
        sd.p(f'<b>Cosmics read {cw["s_best"]:.2f} at A’s wall</b>, the A–C line’s {cw["ac_line"]:.2f}; beam reads '
             f'{cw["beam_all"]:.2f}. In-beam muons read like beam-off ones. So the wall only disagrees for beam '
             'electrons.', 22),
        sd.p('Geant4, ideal reconstruction (the gap’s true line): what the same wall estimator reads, truth = 1', 22,
             INK, 600),
        sd.hbars(g4, 1.1, width=250, h=26, label_w=230, size=20, fmt=lambda v: f'{v:.2f}') if g4 else '',
        sd.callout('<b>The gap is electron scattering</b> between the gap and the wall, not the reconstruction. '
                   'The wall (and the capsule k) are not angle truth for beam; the cosmic in-situ scale stands. '
                   'Adopting it for beam is the next decision.', GREEN, 22),
        gap=14, w=640)
    body = sd.title('The beam angle scale: the walls read the electrons’ scattering, and the cosmic scale stands',
                    'What multiplies production’s raw tan to give the true one, by four methods (left); '
                    'why the beam walls disagree (right).')
    body += sd.row(P.svg('beam scale'), side, gap=36)
    D.slide('beamscale', body, f'''
<p><b>Why this matters for pairs:</b> an opening angle is only as good as the angle scale under it, and the stage-3 k (capsule pointing) multiplies every production angle. Until 7 Oct three truths disagreed for A: walls 0.89, cosmic A–C line 1.11, capsule 1.24–1.29.</p>
<p><b>Wall edges, beam</b> (<code>ntof_cosmics/wall_edge_scale.py</code>, campaign, 31 runs, ~1 M single-track single-group events): where the fired wall group switches across a surveyed boundary, the median true tan at that strip position is known geometrically. The two outer boundaries give 0.893 with D_eff ≈ 330 mm, which refutes capsule pointing (234.6 mm). Sub-samples ruled out t0, partial tracks, fit quality and charge. By time after the flash: 0.77 at 10–15 ms, 0.91–0.93 after 20 ms.</p>
<p><b>Wall edges, cosmics</b> (<code>cosmic_wall_scale.py</code>, run_149 put on the n_TOF clock): {cw["n_single_x"]:,} x-plane tracks firing A’s wall read {cw["s_best"]:.2f} (bootstrap {cw["boot_lo"]:.2f}–{cw["boot_hi"]:.2f}), the A–C line’s value; the beam value is excluded. <b>In-beam muons</b> (<code>inbeam_through_goers.py</code>): A–C through-goers in beam runs read within 2–5 % of beam-off muons although their gain is ~1/3 lower, so gain, beam noise and space charge are not the cause.</p>
<p><b>Geant4</b> (<code>ntof_cosmics/g4_angle/</code>, condor 4402864/5): the data’s wall estimator applied to the gap’s <i>true</i> line. Muons read 1.00; capture electrons read 0.59 overall and 0.92 above 4 MeV. The wall sees where the electron arrives after scattering through the chamber and the material before the wall, not the direction it had in the gap. The data’s exact 0.89–0.92 is not yet reproduced: that needs a <code>wft</code>-digitised forward model of the Geant4 tracks and the ambient thermal-neutron population behind the late triggers, which no simulation has yet.</p>
<p><b>Consequence:</b> never quote the wall scale or the capsule k_arm as beam angle truth. The cosmic in-situ scale (bundles is2_A, is2_C) is the calibration to adopt; that changes stage 3 / k_arm on the campaign and needs a decision. A related check: the ideal gap line itself reads electrons shallower than their emission (gap/gun 0.73 at 5 MeV), so the X17 opening-angle simulation must come from Geant4 hits, not truth directions.</p>''',
            foot='wall_edge_scale.csv (A) · scint-stack λ·k (C) · cosmic_wall_scale/summary.json · pooled k_true_by_borrowed_k.csv · '
                 'kbeam_AC.txt · g4_angle/neutrons_wall_estimator.csv, single_summary.csv.',
            short='T2: beam angle scale')


# --------------------------------------------------------------------------- #
# 12. the join: one re-pass
# --------------------------------------------------------------------------- #
JOIN = [
    ('seeder minimum 3 (WFT_BEAM_MIN_STRIPS=3)', 'T2', 'cosmic truth; beam purity on run_145', GOLD,
     'passed (+40–52 % confirmed); re-pairs 5–13 % → validate with T1’s pairing'),
    ('in-situ v + robust kw per plane', 'T2', 'held-out cosmic truth', GOLD,
     'A closes (is2_A); C built (is2_C); B, D need another truth'),
    ('chamber C kernel', 'T2', 'free-fit closure vs truth', GOLD, 'r06 det7 adopted; tails not kernel-driven'),
    ('beam angle scale', 'T2', 'cosmic A–C line; wall test; Geant4', GOLD,
     'gap = electron scattering; adopt the cosmic in-situ scale (needs a decision)'),
    ('x/y pairing + rescue floor', 'T1', 'split-ab, bench', GREEN, 'passed'),
    ('fixed two-track chain, F per chamber', 'T1', 'split-ab + F ladder', GOLD,
     'passed at A 1200 / C 2400 on the old bundles; re-derive on the new'),
    ('late-t0 depth grid', 'T3', 'refit vs external pointing', RED, 'proposed'),
]


def s_join(D):
    rows = [[f'<b>{c}</b>', th, how, f'<span style="color:{col};font-weight:600">●</span> {st}']
            for c, th, how, col, st in JOIN]
    tab = sd.table(['change', 'thread', 'validated by', 'state'], rows, size=22,
                   widths=[520, 110, 420, 614], align=['left', 'left', 'left', 'left'])
    steps = [dict(label='Merge the branches', sub='docs-only conflict', color=GREY,
                  tip='Merge base 8ca9ed7; they overlap only in HANDOFF.md, STATUS.md and OCTOBER_2026.md.'),
             dict(label='T2 bundles', sub='in-situ v, kw, C kernel', color=GREEN),
             dict(label='T1 on them', sub='bench + F ladder again', color=BLUE,
                  tip='Do not carry F = 1200 / 2400 across a bundle change.'),
             dict(label='One combined split-ab', sub='seeder 3 + pairing + fixed chain', color=PURPLE,
                  tip='The contract: clean single muons split ≤ 0.66 %, no event loses a track.'),
             dict(label='Re-pass', sub='campaign, condor (O4)', color=INK),
             dict(label='Downstream', sub='vertex tests, angles', color=GREY)]
    body = sd.title('Everything rides one re-pass, after one combined validation',
                    'What must be validated before the campaign is re-reconstructed, and its state today (2026-10-07).')
    body += tab + sd.flow(steps, size=22)
    D.slide('join', body, '''
<p>OCTOBER_2026.md §4 makes the campaign re-pass (O4) the single join point: it costs ~8–12 h of condor plus a day of chain, so it runs once, and everything that wants to ride it must be validated first. All the changes are opt-in; with every switch off the output is today’s production.</p>
<p>Shipping also needs plumbing that does not exist yet: bundles carrying the profc <code>xy_pairing</code> (<code>wft_beam.make_bundle</code> has no merge step for it), the switches and F in the condor environment, and a smoke gate that expects <i>more</i> tracks (today’s demands event counts identical to the August pass).</p>
<p>Source: <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code>, “The join”.</p>''',
            short='The join')
