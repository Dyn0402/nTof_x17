#!/usr/bin/env python3
"""
make_y_orientation_deck.py -- the evidence that stage-3 global y is mirrored,
as a slide note for dylan-neff.web.cern.ch/notes (y-orientation).

The evidence slides show the MIRRORED stored frame, so they read the frozen
pre-fix copies in ntof_cosmics/results/y_orientation/pre_fix/ (taken 2026-10-14
before Y_FIX_REGEN.sh overwrote the originals); the "fixed" slide reads the
regenerated originals below.

Reads (current = after the fix):
  ntof_cosmics/results/y_orientation/ac149.parquet   (y_orientation_check.py cosmic)
  ntof_cosmics/results/y_orientation/pmt_timing.csv  (y_orientation_check.py pmt)
  /media/dylan/data/x17/scint_stack/ana/eff_map.parquet (liquid firing maps)
  <out>/capsule_tof/{tof,g4}.csv                      (source height, data and Geant4)
  mx17_m1_map.csv                                     (FEU channel -> strip position)

    python ntof_cosmics/make_y_orientation_deck.py [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug y-orientation --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd                                                        # noqa: E402
from slidedoc import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT, RULE, DBLUE, DRED, DGREEN  # noqa: E402
from sept26_prelim_analysis import paths                                    # noqa: E402

RES = REPO / 'ntof_cosmics' / 'results' / 'y_orientation'
PRE = RES / 'pre_fix'                    # the mirrored-frame inputs, frozen before the regen
EM = Path('/media/dylan/data/x17/scint_stack/ana/eff_map.parquet')
CT = paths.out('capsule_tof')
ARM_COL = dict(A=BLUE, B=GREEN, C=ORANGE, D=PURPLE)


def ac_lines(path: Path) -> pd.DataFrame:
    R = pd.read_parquet(path).dropna(subset=['tA', 'tC'])
    R['dy'], R['dt'] = R.yA - R.yC, R.tA - R.tC
    return R[(R.dev1 < 15) & (R.dev2 < 15) & (R.dt.abs() < 20)].copy()


def load(pre: bool = True):
    """Deck inputs: pre=True the frozen mirrored-frame copies, False the regenerated ones."""
    d = PRE if pre else RES
    G = ac_lines(d / 'ac149.parquet')
    P = pd.read_csv(d / 'pmt_timing.csv')
    M = pd.read_parquet(PRE / 'eff_map.parquet' if pre else EM)
    M = M[(M.layer == 'liq_full') & (M['sample'] == 'all_late')]
    T = pd.read_csv((PRE if pre else CT) / 'tof.csv')
    g4 = pd.read_csv((PRE if pre else CT) / 'g4.csv')
    mp = pd.read_csv(REPO / 'mx17_m1_map.csv')
    return G, P, M, T, g4, mp


def s_cover(D, G):
    up, dn = G[G.dy > 20], G[G.dy < -20]
    p = mannwhitneyu(up.dt, dn.dt).pvalue
    body = (sd.kicker(f'n_TOF EAR2 · X17 · coordinate audit · {dt.date.today():%-d %b %Y}')
            + sd.title('<span style="color:#eef0f3">Stage-3 global y is upside down: every stored height is '
                       'mirrored, every angle between tracks is fine</span>')
            + sd.row(
                sd.bignum(f'{sd.pct((up.dt > 0).mean())} vs {sd.pct((dn.dt > 0).mean())}',
                          'cosmic muons hit the stored-higher arm LATER', DBLUE,
                          f'beam-off run_149, {len(G)} straight A–C muons, p = {sd.sci(p, 0)}; muons go down',
                          size=62),
                sd.bignum('8 / 8', 'plastic bars fire later at stored-high y', DGREEN,
                          'their PMTs are on top, so stored-high is the far end'),
                sd.bignum('−31 mm', 'corrected capsule source height', GOLD,
                          'Geant4 nose: −28 mm; stored: +31 (A), +35 (C)'),
                gap=48))
    notes = """
<p>The sign of the y strip coordinate was set in software on 2026-09-07 (<code>build_tracks.IN_PLANE_SIGN_Y = −1</code>,
applied to the POSITION). The test used then, corr(y, tan y) &gt; 0, is satisfied equally by flipping the position or
the tan, and the two choices differ by a mirror of the vertical axis. Nothing in that test fixes which one is right.
This note collects five independent pieces of evidence that the position should not have been flipped.</p>
<p>Hover the dotted terms and points for values and counts.</p>"""
    D.slide('cover', body, notes, dark=True)


def s_mechanism(D, mp):
    W, H = 1600, 560
    s = []
    # left: the chamber with strips, channel numbers
    x0, y0, h = 120, 60, 420
    s.append(f'<rect x="{x0}" y="{y0}" width="240" height="{h}" fill="#f3ead8" stroke="{GOLD}" stroke-width="3"/>')
    for i in range(12):
        yy = y0 + 20 + i * (h - 40) / 11
        s.append(sd.line(x0 + 10, yy, x0 + 230, yy, GREY, 1.5))
    s.append(sd.T(x0 + 120, y0 - 16, 'y view of one chamber (as mounted)', 21, INK, weight=600))
    s.append(sd.T(x0 - 14, y0 + 26, 'ch 511', 20, INK, 'end'))
    s.append(sd.T(x0 - 14, y0 + h - 12, 'ch 0', 20, INK, 'end'))
    s.append(sd.arrow(x0 + 300, y0 + h - 20, x0 + 300, y0 + 30, INK, 3))
    s.append(sd.T(x0 + 316, y0 + h / 2, 'up = +y = beam', 21, INK, 'start', 600))
    s.append(sd.T(x0 + 120, y0 + h + 36, 'wiring: low FEU channels at the bottom (Dylan)', 20, MUT))
    ys = mp[mp.axis == 'y'].sort_values('channel_num').y_position_mm
    s.append(sd.T(x0 + 120, y0 + h + 62, f'strip map: y = {ys.iloc[1] - ys.iloc[0]:.2f} mm × channel (0 → {ys.max():.1f})', 20, MUT))
    # right: the chain
    bx = 720
    steps = [('raw strip coordinate y_p0', 'increases upward (wiring + map)', INK),
             ('build_tracks: yl = −(y_p0 − 199.29)', 'IN_PLANE_SIGN_Y = −1, applied to the POSITION', RED),
             ('DetTransform: local y → global +y', 'rotation about y only', INK),
             ('stage-3 p0_y, d_y, target_y …', 'global y points DOWN', RED)]
    for i, (a, b, c) in enumerate(steps):
        yy = 70 + i * 120
        s.append(f'<rect x="{bx}" y="{yy}" width="760" height="86" rx="14" fill="#fffdf9" stroke="{c}" stroke-width="2.5"/>')
        s.append(sd.T(bx + 24, yy + 36, a, 24, INK, 'start', 600))
        s.append(sd.T(bx + 24, yy + 66, b, 20, MUT, 'start'))
        if i < 3:
            s.append(sd.arrow(bx + 380, yy + 88, bx + 380, yy + 118, MUT, 2.5))
    body = (sd.title('How it happened: the sign test could not tell a flipped position from a flipped tan',
                     'corr(y, tan y) < 0 raw ⇒ flip one of them; 2026-09-07 flipped the position, which mirrors y')
            + sd.svg(W, H, ''.join(s), 'wiring and sign chain'))
    notes = """
<p><b>Why x got it right and y did not.</b> In x the pinwheel puts each plane centre ~16 mm off the beam axis, so
flipping the tan instead of the position would have displaced opposing arms' images in opposite directions; the target
image chose the position flip (2026-08-20). The y plane has no such offset, so the same choice was copied by analogy.
For y it is a global mirror: both choices give a self-consistent detector, and only an external fact (wiring, gravity,
a PMT position, the capsule geometry) can tell them apart.</p>
<p><b>The reco's tan convention.</b> Since x needed a position flip but no tan flip, the reconstruction's raw tan is
tan_raw = −d(strip)/d(depth) in the outward frame. Applied to a y view whose strips count upward, that means the tan
must be flipped and the position left alone. The correct line is
<code>yl = +(y_p0 − 199.29)</code> with <code>tan_y = −tan_y_raw · k</code>.</p>"""
    D.slide('mechanism', body, notes, foot='mx17_m1_map.csv · sept26_prelim_analysis/build_tracks.py · ntof_tracking/reco/geometry.py')


def s_cosmic(D, G):
    up, dn = G[G.dy > 20], G[G.dy < -20]
    P = sd.Plot(1000, 600, x=(-260, 340), y=(-9, 7), xlabel='y_A − y_C as stored (mm)',
                ylabel='t(wall A) − t(wall C) (ns)', title='beam-off cosmic muons, straight A–C lines')
    P.xticks([(v, str(v)) for v in (-200, -100, 0, 100, 200, 300)])
    P.yticks([(v, str(v)) for v in (-8, -4, 0, 4)])
    P.hline(0, GREY, dash=None, w=1)
    P.vline(0, GREY, dash=None, w=1)
    P.points(list(G.dy), list(G.dt.clip(-8.8, 6.8)), BLUE, r=6,
             tips=[f'event {e}: Δy {a:+.0f} mm, Δt {b:+.1f} ns' for e, a, b in zip(G.event_id, G.dy, G.dt)])
    for sub, x0, x1 in ((up, 20, 330), (dn, -250, -20)):
        m = float(sub.dt.median())
        P.fore.append(sd.line(P.X(x0), P.Y(m), P.X(x1), P.Y(m), RED, 4))
        P.text(0.5 * (x0 + x1), 6.0 if m > 0 else -8.0, f'median {m:+.1f} ns', 22, RED, 'middle', 600)
    col = sd.col(
        sd.callout('Cosmic muons travel down, so the truly higher arm is hit first.', INK, 25),
        sd.callout(f'In the data the arm that is stored higher is hit <b>later</b>: '
                   f'{sd.pct((up.dt > 0).mean())} of muons when A is stored higher, '
                   f'{sd.pct((dn.dt > 0).mean())} when it is stored lower.', RED, 25),
        sd.callout('The step is ±2.3 ns: 66 cm between the walls at the speed of light. Stored y points down.', BLUE, 25),
        w=560, gap=24)
    body = (sd.title('Cosmic muons reach the stored-higher arm later: stored y points down',
                     f'run_149, {len(G)} A–C muons (both tracks within 15° of the joining line), wall times on the n_TOF clock')
            + sd.row(P.svg('cosmic timing'), col, gap=56))
    notes = f"""
<p><b>Timing.</b> Each wall group is read at both ends; the mean of the two end times does not depend on where along the
bar the muon crossed, so it is the muon's arrival time at that wall (plus a per-arm cable constant, which shifts both
medians together and cannot make a step). Beam-off DREAM triggers are on the n_TOF clock through
<code>ntof_cosmics/clock_match.py</code>; WALA and WALC hits within ±150 ns of the matched trigger.</p>
<p><b>Selection.</b> run_149 pairs table (<code>results/tracking/k_run_147/pairs_*</code>), pair AC, each track within 15°
of the line joining them, |Δt| &lt; 20 ns: {len(G)} muons from the 35 clock-matched sub-runs. Mann–Whitney between
Δy &gt; +20 and Δy &lt; −20: p = {sd.sci(mannwhitneyu(up.dt, dn.dt).pvalue, 0)}.</p>
<p><b>Why not in-beam through-goers.</b> The same test on late in-beam A–C straight lines gives no step: those are
mostly simultaneous particles from the capsule (the calorimetry study found beam through-goers non-penetrating), which
carry no direction.</p>
<p>Code: <code>python ntof_cosmics/y_orientation_check.py cosmic</code>.</p>"""
    D.slide('cosmic', body, notes, foot='needs no hardware fact except that cosmic muons come from above')


def s_plastic(D, P):
    rows = []
    for arm in 'ABCD':
        for el in ('p1', 'p2'):
            v = P[(P.arm == arm) & (P.el == el) & (P.coord == 'v_mm')].iloc[0]
            u = P[(P.arm == arm) & (P.el == el) & (P.coord == 'u_mm')].iloc[0]
            rows.append((f'{arm} bar {el[1]}', v, u))
    Pl = sd.Plot(1000, 600, x=(0.5, 8.5), y=(-0.2, 0.6), xlabel='',
                 ylabel='Δt slope (ns per 100 mm)', title='plastic time − wall time, per bar')
    Pl.xticks([(i + 1, r[0]) for i, r in enumerate(rows)])
    Pl.yticks([(v, f'{v:.1f}') for v in (-0.2, 0, 0.2, 0.4, 0.6)])
    Pl.hline(0, GREY, dash=None, w=1)
    for i, (lab, v, u) in enumerate(rows):
        Pl.vbar(i + 1 - 0.18, v.slope_ns_per_100mm, 34, RED,
                tip=f'{lab}: along stored y {v.slope_ns_per_100mm:+.3f} ± {v.err:.3f} ns/100 mm (n {int(v.n):,})')
        Pl.vbar(i + 1 + 0.18, u.slope_ns_per_100mm, 34, GREY, opacity=0.6,
                tip=f'{lab}: across the bar (u) {u.slope_ns_per_100mm:+.3f} ± {u.err:.3f} ns/100 mm')
    col = sd.col(sd.legend([('along stored y', RED, 'box'), ('across the bar (control)', GREY, 'box')], size=22),
                 sd.callout('Every bar fires later when the track is stored higher, at 5–40σ, and is flat across the bar.',
                            RED, 25),
                 sd.callout('The plastic PMTs are on top (all arms). Light from the true top arrives first, so the '
                            'stored top is the true bottom.', INK, 25),
                 sd.callout('Expected size: a few tenths of a ns per 100 mm for light in PVT, as seen.', GREY, 23),
                 w=560, gap=22)
    body = (sd.title('All 8 plastic bars fire later at stored-high y: their PMTs are at stored-low y',
                     'beam runs, tracks ≥ 12 ms, plastic time minus the wall group\'s mean end time, '
                     'per-group offsets removed')
            + sd.row(Pl.svg('plastic timing'), col, gap=56))
    notes = """
<p><b>Method.</b> For each track in the scint-stack table (<code>ntof_scint_stack</code>, every second run), the
plastic bar's time minus the mean end time of the wall group with the largest two-ended signal. The wall mean removes
the trigger and the particle's own arrival; what is left is the plastic's light path to its single PMT. Regressed on the
track's stored v at the strip plane within each wall group (fixed effects), residuals clipped at 10 ns, bootstrap
errors. The across-bar (u) slope is the control: the PMT sits at one end of the bar, not at a side.</p>
<p><b>Caveat.</b> "PMTs on top" is Dylan's report (2026-10-06); the Geant4 model has bare bars. Amplitude trends are too
weak and trigger-biased to use. The liquid scintillator timing is too noisy (n = 0.7–17 k) to add anything.</p>
<p>Code: <code>python ntof_cosmics/y_orientation_check.py pmt</code>.</p>"""
    D.slide('plastic', body, notes, foot='ntof_cosmics/results/y_orientation/pmt_timing.csv')


def s_liquid_capsule(D, M, T, g4):
    def prof(arm, coord):
        m = M[M.arm == arm]
        g = m.groupby(coord).apply(lambda s: (s.k_on.sum() - s.k_off.sum()) / max(s.n.sum(), 1),
                                   include_groups=False)
        return g[g.index.to_series().abs() < 220]
    Pl = sd.Plot(800, 520, x=(-220, 220), y=(0, 0.16), xlabel='position on the liquid, as stored (mm)',
                 ylabel='net firing', title='liquids: firing vs position')
    Pl.xticks([(v, str(v)) for v in (-200, -100, 0, 100, 200)])
    Pl.yticks([(v, f'{100 * v:.0f}%') for v in (0, 0.04, 0.08, 0.12, 0.16)])
    for arm, coord, c, lab in (('A', 'x', BLUE, 'A along u (PMT at +u)'), ('D', 'x', PURPLE, 'D along u (PMT at +u)'),
                                ('B', 'y', GREEN, 'B along stored y (PMT on top)')):
        g = prof(arm, coord)
        Pl.line(list(g.index), list(g.values), c, tips=[f'{lab}: {k:.0f} mm → {100 * v:.2f} %' for k, v in g.items()])
    leg = sd.legend([('A, along u (control)', BLUE), ('D, along u (control)', PURPLE), ('B, along stored y', GREEN)], size=21)
    late = T[T.lo >= 20]
    yA = float(np.average(late[late.arm == 'A'].y_cross, weights=1 / late[late.arm == 'A'].y_err ** 2))
    yC = float(np.average(late[late.arm == 'C'].y_cross, weights=1 / late[late.arm == 'C'].y_err ** 2))
    gy = float(g4[g4.kind.str.startswith('reconstructed')].v_cross.mean())
    tab = sd.table(['source height (mm)', 'value'], [
        ['Geant4 nose-first capsule, same estimator', f'{gy:.0f}'],
        ['data, as stored (A / C)', f'+{yA:.0f} / +{yC:.0f}'],
        ['data, corrected (A / C)', f'<b>−{yA:.0f} / −{yC:.0f}</b>'],
        ['July hits analysis (no position flip)', '"~3 cm low"'],
    ], size=22, widths=[460, 160])
    body = (sd.title('The liquids and the capsule agree: the PMT side and the nose are at stored-low y',
                     'left: late-trigger whole-face maps (scint stack) · right: y band crossing, ≥ 20 ms')
            + sd.row(sd.col(Pl.svg('liquid profiles'), leg, gap=6, w=800),
                     sd.col(tab, sd.callout('Beam neutrons are absorbed in the first 0.15 mm of 500 bar ³He, so beam '
                                            'captures in aluminium sit at the upstream (lower) end: the nose. '
                                            'Stored, the source is at the valve end.', GOLD, 23), gap=28), gap=48))
    notes = """
<p><b>Liquids.</b> Net firing (on − off window) of the liquid behind tracks tagged by the wall at both ends. A and D lie
horizontally with the PMT at +u: they answer 10–20× more on the PMT half, so the method sees light collection. B stands
vertically with its PMT on top (17–18 July survey, in the Geant4 model): it answers ×2 more at stored −y. C has too few
counts. This was the open "vertical liquids do not answer near their PMT" question of the scint-stack note, and the
mirror answers it.</p>
<p><b>Capsule.</b> <code>sept26_prelim_analysis/capsule_tof.py</code>: the y band crossing is scale-free; Geant4 through
the real reconstruction (g4_digi) reproduces its truth (−28). The 2026-09-14 conclusion that the CAD height is wrong by
~30 mm (HANDOFF_CAPSULE_Y.md) is this mirror.</p>"""
    D.slide('liquid', body, notes, foot='scint stack eff_map (liq_full, all_late) · capsule_tof tof.csv / g4.csv')


def _summary(G, P, M, T, g4) -> dict:
    """The five orientation facts, by the same estimators, for one vintage of inputs."""
    up, dn = G[G.dy > 20], G[G.dy < -20]
    v = P[(P.coord == 'v_mm') & P.el.isin(['p1', 'p2'])]            # the 8 plastic bars
    late = T[T.lo >= 20]
    yc = {a: float(np.average(late[late.arm == a].y_cross, weights=1 / late[late.arm == a].y_err ** 2))
          for a in 'AC'}
    b = M[M.arm == 'B']
    net = lambda s: (s.k_on.sum() - s.k_off.sum()) / max(s.n.sum(), 1)
    return dict(cos_up=float((up.dt > 0).mean()), cos_dn=float((dn.dt > 0).mean()),
                cos_p=float(mannwhitneyu(up.dt, dn.dt).pvalue), cos_n=len(G),
                pmt_pos=int((v.slope_ns_per_100mm > 0).sum()), pmt_n=len(v),
                pmt_lo=float(v.slope_ns_per_100mm.min()), pmt_hi=float(v.slope_ns_per_100mm.max()),
                liqB_top=net(b[b.y > 50]), liqB_bot=net(b[b.y < -50]),
                capA=yc['A'], capC=yc['C'],
                g4=float(g4[g4.kind.str.startswith('reconstructed')].v_cross.mean()))


def s_fixed(D, B, A, regen):
    """Before (mirrored, frozen pre_fix inputs) vs after (regenerated) on the same estimators."""
    f = lambda x: f'{x:+.0f}'
    rows = [
        ['cosmic muons: A later when A stored higher / lower',
         f"{sd.pct(B['cos_up'])} / {sd.pct(B['cos_dn'])}", f"<b>{sd.pct(A['cos_up'])} / {sd.pct(A['cos_dn'])}</b>",
         'stored-higher arm hit first'],
        ['plastic bars later at stored-high y',
         f"{B['pmt_pos']} / {B['pmt_n']}", f"<b>{A['pmt_pos']} / {A['pmt_n']}</b>", 'PMTs on top: 0 / 8'],
        ['liquid B net firing, stored y > +50 / < −50',
         f"{100 * B['liqB_top']:.1f} / {100 * B['liqB_bot']:.1f} %",
         f"<b>{100 * A['liqB_top']:.1f} / {100 * A['liqB_bot']:.1f} %</b>", 'PMT on top: top > bottom'],
        ['capsule source height, ≥ 20 ms (A / C, mm)',
         f"{f(B['capA'])} / {f(B['capC'])}", f"<b>{f(A['capA'])} / {f(A['capC'])}</b>", 'nose: Geant4 −28'],
        ['Geant4 through the digitiser + reco (mm)', f(B['g4']), f"<b>{f(A['g4'])}</b>", 'truth −28'],
    ]
    tab = sd.table(['orientation fact', 'before (stored, 09-07 → 10-13)', 'after (regenerated)', 'external fact'],
                   rows, size=22, widths=[520, 300, 260, 300])
    body = (sd.title('Fixed: y keeps its position, tan y flips. Every check now points up',
                     f'code 13 Oct (main 9ea8b72), tables regenerated 14 Oct · same estimators as the slides above, '
                     f'before = the frozen pre-fix inputs')
            + sd.col(tab, sd.callout(regen, GREY, 22), gap=28))
    notes = """
<p><b>The fix.</b> <code>build_tracks.local_and_global</code> writes <code>y_local = +(y_p0 − 199.29)</code> and
<code>tany = −tan_y_raw · k</code>. Against the 09-07 tables it is an exact global y mirror: <code>p0_y</code>,
<code>d_y</code>, <code>tany</code>, <code>target_y_mm</code>, <code>y_local</code> change sign, and nothing else.
Every stage-3 table was rebuilt from its own sidecar and checked row by row against the old one before it replaced
it (<code>regen_stage3.py</code>). Sidecars carry <code>y_convention: "up"</code>. A guard in every build refuses a
table whose late capsule y crossing is not below zero.</p>
<p><b>What moved.</b> Heights only: the capsule sits at the nose where the CAD and Geant4 put it (to ~5 mm; the "CAD
is 30 mm off" result was the mirror), the vertical liquids rise toward their PMT, vertex-y images flip. Angles between
tracks, |tan|, x and z are unchanged by construction. Full list: <code>Y_FLIP_AUDIT.md</code>.</p>"""
    D.slide('fixed', body, notes, foot='Y_FLIP_AUDIT.md · Y_FIX_REGEN.sh · ntof_cosmics/results/y_orientation/pre_fix/')


def s_close(D):
    li = lambda t: f'<li style="margin-bottom:14px">{t}</li>'
    left = ('<ul style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li('<b>Unchanged:</b> every angle between two tracks (opening angles, X17 and IPC), |tan|, x and z '
                 'positions, the x/z imaging, track counts and efficiencies that do not cut on y.')
            + li('<b>Mirrored:</b> stored p0_y, d_y, tany sign, target_y_mm, angle_to_beam_deg, vertex heights, the '
                 'capsule height and the "CAD is 30 mm off" result, CAPSULE_V / the 32.2 mm Geant4 shift, scintillator '
                 'v-predictions and maps, and anything comparing heights with Geant4 or CAD.')
            + li('<b>Possibly affected:</b> cuts on y windows (in-bore along y, capsule y ranges), the y-binned '
                 'shear, any y-asymmetric acceptance or efficiency correction.') + '</ul>')
    right = ('<ol style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:32px">'
             + li('<b>Done 13 Oct:</b> <code>build_tracks</code> keeps the position and flips tan y; a guard in '
                  'every build and a test pinned to the cosmic timing.')
             + li('<b>Done 14 Oct:</b> every stage-3 set regenerated and proved an exact y mirror; consumers audited.')
             + li('<b>Done 14 Oct:</b> capsule height, same-chamber vertices, scint maps re-derived.') + '</ol>')
    body = (sd.title('<span style="color:#eef0f3">What the mirror changes, and the fix</span>')
            + sd.row(sd.col(sd.p('scope', 26, '#a3abb9'), left, w=860),
                     sd.col(sd.p('the fix', 26, '#a3abb9'), right, w=740), gap=56))
    D.slide('close', body, '<p>Handoff: <code>HANDOFF_Y_FLIP_FIX.md</code> in the nTof_x17 repo root; the rule is in '
                           '<code>COORDINATES.md</code> and CLAUDE.md.</p>', dark=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/media/dylan/data/x17/ntof_cosmics/y_orientation/y-orientation.html'))
    ap.add_argument('--regen-note', default='', help='one line on the regeneration for the "fixed" slide')
    a = ap.parse_args()
    G, P, M, T, g4, mp = load(pre=True)
    after = _summary(*[x for i, x in enumerate(load(pre=False)) if i != 5])
    before = _summary(G, P, M, T, g4)
    D = sd.Deck('Stage-3 y is mirrored', 'evidence that the stored global y axis points down')
    s_cover(D, G)
    s_mechanism(D, mp)
    s_cosmic(D, G)
    s_plastic(D, P)
    s_liquid_capsule(D, M, T, g4)
    s_fixed(D, before, after, a.regen_note)
    s_close(D)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    D.write(a.out, note_meta=dict(
        title='Stage-3 global y is mirrored: the evidence',
        summary='Cosmic-muon timing, plastic and liquid PMT positions, the strip wiring and the capsule nose all said '
                'stored y pointed down; fixed 13 Oct, regenerated 14 Oct, every check now points up.',
        tags='X17,coordinates,tracking', date=dt.date.today().isoformat()))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
