#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_deck.py -- the scintillator-stack slide note (dylan-neff.web.cern.ch/notes).

Figure-first write-up of `ana`: the track calibration it uses (the single-track
imaging's per-run k, with the extrapolation's coefficients calibrated on the
SiPM wall's group boundaries), then a heat map of efficiency and of response
over the face of every scintillator on all four arms.  Reads only
``<scint>/ana``; every number in a title or a tooltip is computed here.

    python -m ntof_scint_stack.make_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        <scint>/deck/scint-stack.html --slug scint-stack --force --deploy
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402
from ntof_scint_stack.ana import (  # noqa: E402
    CAPSULE_XYZ, LATE_MS, LS_HALF_U, LS_HALF_V, PLAS_HALF_U, PLAS_HALF_V,
    WALL_EDGES, WALL_HALF_V)

ARMS = ('A', 'B', 'C', 'D')
#: Where each liquid's PMT is, in the map frame (+u = right when looking from
#: behind the wall toward the target, +v = up).  From the Geant4 model, which
#: carries the 2026-07-17/18 survey (`MX17_Full_Geant/include/SimConfig.hh`,
#: `ls_rot_deg` = -90 on A/D: horizontal, neck + PMT along +u; 0 on B/C:
#: vertical, PMT up).  The beam data agree independently: A and D respond
#: 10-20x more on their +u half, B and C show no u gradient.  The vessel is a
#: 451 mm slab, a 90 mm funnel down to a 50 mm neck, PMT half-inserted.
LIQ_PMT = {'A': '+u', 'B': '+v', 'C': '+v', 'D': '+u'}
#: The plastic bars' PMTs: on top (+v) on every arm, as reported by Dylan
#: 2026-10-06.  NOT in the Geant model (it has bare bars) and not testable on
#: this data: the response is flat in v to +-8 % and trigger-cut.
PLAS_PMT = '+v'
#: liquid vessel funnel (from the slab edge) and neck, mm
LS_FUNNEL, LS_NECK_HALF = 90.0, 25.0
ARM_COL = {'A': sd.BLUE, 'B': sd.GREY, 'C': sd.GREEN, 'D': sd.PURPLE}
#: the comparison chains, one colour each on every slide
VAR_COL = {'k_only': sd.GREY, 'capsule': sd.ORANGE, 'used': sd.BLUE,
           'old': sd.GOLD}
VAR_NAME = {'k_only': 'bare imaging k', 'capsule': 'k, shrunk to the capsule',
            'used': 'k, calibrated on the wall',
            'old': 'free scale (2 Oct)'}
#: which sample each map is drawn from, and why (repeated in the Details).
#: The main maps are the ``_full`` whole-face ones on every late trigger: the
#: arm's own trigger biases their level (and stamps the plastic's outline on
#: them) but they carry ~30x the statistics of the unbiased sample, which has
#: its own slide.
MAP_SAMPLE = {('wall_full', 'eff'): 'all_late', ('wall_full', 'gain'): 'all_late',
              ('plas_full', 'eff'): 'all_late', ('plas_full', 'gain'): 'all_late',
              ('liq_full', 'eff'): 'all_late', ('liq_full', 'gain'): 'all_late'}
SAMPLE_WORDS = {
    'unbiased': f'events another arm triggered, > {LATE_MS:.0f} ms after the flash',
    'all_late': f'every trigger > {LATE_MS:.0f} ms after the flash'}


def ana() -> Path:
    return paths.spell('scint') / 'ana'


def rd(name):
    return pd.read_parquet(ana() / f'{name}.parquet')


def geo():
    return pd.read_csv(paths.spell('scint') / 'geometry.csv').groupby('arm').first()


def pc(x, d=0):
    return sd.pct(x, d)


def fnum(x, d=2):
    return '–' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


# --------------------------------------------------------------------------- #
# heat-map panels
# --------------------------------------------------------------------------- #
def binw(M):
    xs = np.sort(M.x.unique())
    return float(np.min(np.diff(xs))) if len(xs) > 1 else 50.0


def rebin_eff(M, f=2, min_n=20):
    """Merge f x f cells of an efficiency map by summing their counts.

    The unbiased sample is a few thousand tags per arm, too few for 50 mm
    cells; the maps are built on grids aligned to the channel edges, so 2 x 2
    merging gives 100 mm cells (one wall group wide).  The small correction
    for accidental TAGS (a few per mille past 10 ms) is not re-applied."""
    if not len(M):
        return M
    b = binw(M)
    x0 = float((M.x - (M.ix + 0.5) * b).median())
    y0 = float((M.y - (M.iy + 0.5) * b).median())
    R = (M.assign(jx=M.ix // f, jy=M.iy // f)
         .groupby(['jx', 'jy'])[['n', 'k_on', 'k_off']].sum().reset_index())
    n = R.n.astype(float)
    p_on, p_off = R.k_on / n, R.k_off / n
    R['eff'] = (p_on - p_off) / (1 - p_off)
    R['err'] = np.sqrt(np.clip(p_on * (1 - p_on), 1e-6, None) / n)
    R.loc[R.n < min_n, ['eff', 'err']] = np.nan
    R['x'] = x0 + (R.jx + 0.5) * b * f
    R['y'] = y0 + (R.jy + 0.5) * b * f
    return R


PMT_FILL = '#8f98a3'


def _pmt_glyph(P, edge, half_w, length, neck_half, axis, tip, u0=0.0):
    """A light guide / funnel from a detector edge, narrowing to the PMT.

    ``axis`` '+u' or '+v'; ``edge`` the detector's edge coordinate on that
    axis, ``half_w`` its half-width across it.  Drawn as far as the panel."""
    if axis == '+u':
        xs = [edge, edge + length, edge + length + 60, edge + length + 60,
              edge + length, edge]
        ys = [half_w, neck_half, neck_half, -neck_half, -neck_half, -half_w]
    else:
        ys = [edge, edge + length, edge + length + 60, edge + length + 60,
              edge + length, edge]
        xs = [-half_w, -neck_half, -neck_half, neck_half, neck_half, half_w]
        xs = [x + u0 for x in xs]
    xr, yr = P.xs, P.ys
    xs = [min(max(x, xr[0]), xr[1]) for x in xs]
    ys = [min(max(y, yr[0]), yr[1]) for y in ys]
    P.raw(sd.poly([P.X(x) for x in xs], [P.Y(y) for y in ys], PMT_FILL,
                  fill=PMT_FILL, tip=tip))
    mid = min(edge + 0.6 * length, 0.5 * (edge + xr[1] if axis == '+u'
                                          else edge + yr[1]))
    if axis == '+u':
        P.raw(sd.T(P.X(mid) + 6, P.Y(0), 'PMT', 17, '#ffffff', weight=700,
                   rot=-90, tip=tip))
    else:
        P.raw(sd.T(P.X(0.5 * (xs[2] + xs[3])), P.Y(mid) + 6, 'PMT', 17,
                   '#ffffff', weight=700, tip=tip))


def outline(P, layer, g, arm=None):
    """Detector edges and channel boundaries in the map's own frame."""
    layer = layer.split('_')[0]
    if layer == 'wall':
        P.rect(WALL_EDGES[0], WALL_EDGES[-1], -WALL_HALF_V, WALL_HALF_V,
               sd.INK, 2, tip='SiPM wall: 16 bars of 25 mm, read out as four '
               'groups of four (dashed), a SiPM at each end')
        for e in WALL_EDGES[1:-1]:
            P.raw(sd.line(P.X(e), P.Y(-WALL_HALF_V), P.X(e), P.Y(WALL_HALF_V),
                          sd.INK, 1.5, '5 4'))
    elif layer == 'plas':
        for b in (1, 2):
            c = g[f'plas_u_{b}']
            side = 'L' if b == 1 else 'R'
            _pmt_glyph(P, PLAS_HALF_V, PLAS_HALF_U, 60, 22, PLAS_PMT,
                       f'plastic bar {b}: light guide and PMT on top (as '
                       'reported; not in the Geant model)', u0=c)
            P.rect(c - PLAS_HALF_U, c + PLAS_HALF_U, -PLAS_HALF_V, PLAS_HALF_V,
                   sd.INK, 2, tip=f'plastic bar {b} ({side}): 200 x 300 x 20 mm '
                   'PVT, one PMT, on top')
    else:
        ax = LIQ_PMT.get(arm, '+v')
        _pmt_glyph(P, LS_HALF_U if ax == '+u' else LS_HALF_V,
                   LS_HALF_V if ax == '+u' else LS_HALF_U, LS_FUNNEL,
                   LS_NECK_HALF, ax,
                   f'liquid {arm}: 90 mm funnel to the 50 mm neck and the PMT, '
                   + ('to the right (+u) looking from behind'
                      if ax == '+u' else 'on top') + ' (Geant4 / 17 July survey)')
        P.rect(-LS_HALF_U, LS_HALF_U, -LS_HALF_V, LS_HALF_V, sd.INK, 2,
               tip='liquid cell, 451 x 451 mm, one PMT')
        # the plastic's footprint on the cell, at its surveyed u (unprojected)
        lo = g['plas_u_1'] - PLAS_HALF_U - g['u_ls']
        hi = g['plas_u_2'] + PLAS_HALF_U - g['u_ls']
        P.rect(lo, hi, -PLAS_HALF_V - g['v_ls'], PLAS_HALF_V - g['v_ls'],
               '#ffffff', 2.4, '8 5',
               tip='the plastic bars in front of the cell: inside this box a '
               'particle crossed 20 mm of PVT first')


def heat_panel(M, col, layer, arm, g, vmin, vmax, cmap, fmt_tip, w, h, ml,
               show_x, show_y, title):
    # the liquid panels reach further, to show the funnel and the PMT
    rng = 360 if layer.startswith('liq') else 262
    P = sd.Plot(w, h, x=(-rng, rng), y=(-rng, rng),
                margin=(4, 6, 40 if show_x else 6, ml), title=title)
    if show_x:
        P.xticks([(-200, '−200'), (0, '0'), (200, '200')])
    if show_y:
        P.yticks([(-200, '−200'), (0, '0'), (200, '200')])
    b = binw(M) if len(M) else 50.0
    cells = []
    for r in M.itertuples():
        v = getattr(r, col)
        if not np.isfinite(v):
            continue
        cells.append((r.x - b / 2, r.x + b / 2, r.y - b / 2, r.y + b / 2,
                      float(v), fmt_tip(r)))
    P.cells(cells, vmin, vmax, cmap, gap=0.6)
    outline(P, layer, g, arm)
    return P.svg(f'{layer} map, chamber {arm}')


def heat_slide(D, sid, short, ttl, sub, layer, G, EM, GM, notes, eff_max=1.0,
               eff_ticks=None, eff_label='efficiency', gain_rel=(0.6, 1.4),
               eff_min=0.0):
    """Two rows x four arms: efficiency (top), response relative to the arm
    median (bottom), each row with its colour bar."""
    side, ml = 262, 62
    se = MAP_SAMPLE[(layer, 'eff')]
    sg = MAP_SAMPLE[(layer, 'gain')]
    top, bot, info = [], [], {}
    for i, arm in enumerate(ARMS):
        g = G.loc[arm]
        lm = ml if i == 0 else 10
        me = EM[(EM.arm == arm) & (EM.layer == layer) & (EM['sample'] == se)]
        if se == 'unbiased':
            me = rebin_eff(me)

        def tip_e(r, arm=arm):
            return (f'chamber {arm} · u {r.x:+.0f}, v {r.y:+.0f} mm\n'
                    f'efficiency {100 * r.eff:.1f} ± {100 * r.err:.1f} %\n'
                    f'{int(r.k_on)} of {int(r.n)} tagged tracks lit it '
                    f'({int(r.k_off)} in the pre-trigger window)')
        top.append(heat_panel(me, 'eff', layer, arm, g, eff_min, eff_max,
                              sd.VIRIDIS, tip_e, side + lm + 6, side + 44, lm,
                              False, i == 0, f'chamber {arm}'))
        mg = GM[(GM.arm == arm) & (GM.layer == layer) & (GM['sample'] == sg)
                & ~GM.quantity.str.endswith('_through')]
        med = float(np.nanmedian(mg.med)) if len(mg) else np.nan
        unit = mg.unit.iloc[0] if len(mg) else ''
        mg = mg.assign(rel=mg.med / med)
        info[arm] = (med, unit)

        def tip_g(r, arm=arm, unit=unit, med=med):
            return (f'chamber {arm} · u {r.x:+.0f}, v {r.y:+.0f} mm\n'
                    f'median {r.med:,.0f} {unit} = {r.rel:.2f} × the face '
                    f'median ({med:,.0f} {unit})\n{int(r.n)} hits')
        if unit == 'keVee' and np.isfinite(med) and med >= 1000:
            mt = f'{arm} · median {med / 1e3:.2f} MeVee'
        else:
            mt = (f'{arm} · median {med:,.0f} {unit}' if np.isfinite(med)
                  else f'{arm} · no data')
        bot.append(heat_panel(mg, 'rel', layer, arm, g, *gain_rel, sd.DIVERGE,
                              tip_g, side + lm + 6, side + 84, lm, True,
                              i == 0, mt))
    et = eff_ticks or [(0, '0'), (0.5, '0.5'), (1, '1')]
    cb1 = sd.colorbar(170, side + 30, eff_min, eff_max, et, eff_label)
    gt = [(gain_rel[0], f'{gain_rel[0]:.1f}'), (1, '1'),
          (gain_rel[1], f'{gain_rel[1]:.1f}')]
    cb2 = sd.colorbar(170, side + 30, *gain_rel, gt, 'response ÷ median',
                      cmap=sd.DIVERGE)
    body = (sd.title(ttl, sub)
            + '<div style="display:flex;flex-direction:column;gap:6px">'
            + sd.row(*top, cb1, gap=4, align='end')
            + sd.row(*bot, cb2, gap=4, align='center') + '</div>')
    D.slide(sid, body, notes, short=short)
    return info


def samples_slide(D, FN, EM, eff):
    """Where the statistics go, and the two checks behind the sample choice
    (`checks.py`): the trigger emulation and the time cut."""
    ck = paths.spell('scint') / 'checks'
    TA = (pd.read_csv(ck / 'trigger_arms.csv')
          if (ck / 'trigger_arms.csv').exists() else None)
    LS = (pd.read_csv(ck / 'late_scan.csv')
          if (ck / 'late_scan.csv').exists() else None)
    steps = list(dict.fromkeys(FN.step))
    trs = []
    for st in steps:
        r = [st.replace('LATE_MS', f'{LATE_MS:.0f} ms')]
        for a in ARMS:
            n = FN[(FN.arm == a) & (FN.step == st)].n
            r.append(f'{int(n.iloc[0]):,}' if len(n) else '–')
        trs.append(r)
    nmap = {a: int(FN[(FN.arm == a) & FN.step.str.startswith('>')].n.iloc[0])
            for a in ARMS}
    # the unbiased TAGS the whole-wall map is built on (not the events)
    nunb = {a: eff(a, 'wall', 'wany_tol_full', 'unbiased')[2] for a in ARMS}
    tab = sd.table(['', *ARMS], trs, 22,
                   align=['left'] + ['right'] * 4)
    one = (f'{pc(TA.frac_1_arms.iloc[0], 1)}' if TA is not None else '~98 %')
    two = (f'{pc(TA.frac_2_arms.iloc[0], 1)}' if TA is not None else '~0.2 %')
    side = []
    if LS is not None:
        L = LS[(LS['sample'] == 'all') & (LS.layer == 'wall')]
        bins = sorted(L.lo_ms.unique())
        P = sd.Plot(640, 440, x=(-0.5, len(bins) - 0.5), y=(0.3, 1.0),
                    xlabel='time since the flash, ms (bin start)',
                    ylabel='wall | plastic, net',
                    title='Earlier than 10 ms the efficiency has not converged')
        P.xticks([(i, f'{b:g}') for i, b in enumerate(bins)])
        P.yticks([(v, f'{v:.1f}') for v in (0.4, 0.6, 0.8, 1.0)])
        for a in ARMS:
            q = L[L.arm == a].set_index('lo_ms').reindex(bins)
            ok = q.n.to_numpy() >= 100
            xs = [i for i in range(len(bins)) if ok[i]]
            ys = [float(q.eff.iloc[i]) for i in xs]
            P.line(xs, ys, ARM_COL[a], w=2.5, r=5,
                   tips=[f'{a}, {bins[i]:g} ms: {100 * q.eff.iloc[i]:.1f} %, '
                         f'{int(q.n.iloc[i]):,} tags, accidental tags '
                         f'{100 * q.c.iloc[i]:.0f} %' for i in xs])
        P.vline(bins.index(LATE_MS) - 0.5, sd.MUT, '6 5', 2, 'cut')
        side.append(P.svg('wall efficiency against time since the flash'))
        side.append(sd.legend([(a, ARM_COL[a]) for a in ARMS], 20))
    body = (sd.title(
        f'The maps use every late trigger, {min(nmap.values()) / 1e3:.0f}k–'
        f'{max(nmap.values()) / 1e3:.0f}k tracks per arm; the unbiased sample '
        f'has {min(nunb.values()) / 1e3:.1f}k–{max(nunb.values()) / 1e3:.1f}k tags',
        f'{one} of triggers fire exactly one arm and {two} two, so “another arm '
        'triggered” is rare by nature, not by the emulation')
        + sd.row(sd.col(tab, gap=10), sd.col(*side, gap=6), gap=40))
    D.slide('samples', body, f"""
<p><b>Coverage.</b> Every reconstructed track of the stage-3 full pass is read:
all production runs but the two before the 27 July access (run_79, run_81).
The table follows one arm's tracks down to the maps, each step cumulative; the
indented line is the late sample inside the plastic's footprint, which the
plastic and liquid maps need.</p>
<p><b>The unbiased sample.</b> The trigger is wall-sum AND plastic in any arm,
so on an event an arm triggered itself its own wall and plastic fired by
construction. A trigger-free measurement needs another arm to have fired the
trigger, and the beam's triggers are almost all single-arm: on two sub-runs of
every run the emulation gives {one} of triggers to exactly one arm,
{two} to two and the rest to none. Arm D's chamber reconstructs far
more tracks on events the other arms triggered (about a third of its late
tracks, against 4–18 % on A, B, C), which is why its unbiased sample is larger;
whether those are real particles is not yet checked.</p>
<p><b>The trigger emulation is measured, not assumed.</b> On triggers no other
arm could have fired, an arm's wall-group sum and plastic must sit above the real
discriminator; the 0.5 % low edge of each is taken as the threshold. The edges
are flat to ~1 mV across the campaign. The run_79 read-back values had the plastic
3–6 mV too high and C's wall 4 mV too low.</p>
<p><b>The time cut stays at {LATE_MS:.0f} ms.</b> Right: the wall's efficiency
given the plastic, net of accidental channels and of accidental tags, in bins of
time since the flash. Before {LATE_MS:.0f} ms up to half the tags are accidental,
the corrected value has not converged, and the bins hold under 15 % of the late
statistics, so moving the cut earlier would add little and bias what it adds.
Even 10–20 ms reads a few per cent below the rest. Source: `checks.py`.</p>""",
            short='samples')


def unbiased_slide(D, G, EM, eff, weak):
    """The trigger-free shape: wall and plastic, unbiased sample, 100 mm."""
    side, ml = 262, 62
    rows = []
    for layer, label in (('wall_full', 'wall'), ('plas_full', 'plastic')):
        pans = []
        for i, arm in enumerate(ARMS):
            lm = ml if i == 0 else 10
            me = rebin_eff(EM[(EM.arm == arm) & (EM.layer == layer)
                              & (EM['sample'] == 'unbiased')])

            def tip_e(r, arm=arm):
                return (f'chamber {arm} · u {r.x:+.0f}, v {r.y:+.0f} mm\n'
                        f'efficiency {100 * r.eff:.1f} ± {100 * r.err:.1f} %\n'
                        f'{int(r.k_on)} of {int(r.n)} tagged tracks lit it')
            pans.append(heat_panel(me, 'eff', layer, arm, G.loc[arm], 0, 1,
                                   sd.VIRIDIS, tip_e, side + lm + 6,
                                   side + (84 if layer == 'plas_full' else 44),
                                   lm, layer == 'plas_full', i == 0,
                                   f'{arm} · {label}'))
        rows.append(pans)
    cb = sd.colorbar(170, side + 30, 0, 1, [(0, '0'), (0.5, '0.5'), (1, '1')],
                     'efficiency')
    weak_txt = (', '.join(f'{a} g{k} ({pc(e)})' for e, a, k in weak)
                or 'none below 75 %')
    pu = [eff(a, 'plas', 'pm', 'unbiased')[0] for a in ARMS]
    body = (sd.title(f'Trigger-free: weak wall groups {weak_txt}; the plastic '
                     f'answers {pc(min(pu))}–{pc(max(pu))}',
                     f'{SAMPLE_WORDS["unbiased"]} · 100 mm cells · '
                     'top: wall given a plastic bar · bottom: plastic given the wall')
            + '<div style="display:flex;flex-direction:column;gap:6px">'
            + sd.row(*rows[0], cb, gap=4, align='end')
            + sd.row(*rows[1], gap=4, align='center') + '</div>')
    D.slide('map-unbiased', body, f"""
<p>The same whole-face maps as the previous slides, on the events another arm
triggered: there this arm's wall and plastic were not required by the trigger,
so a weak group or a corner that does not answer shows at its true level.
It is a few thousand tracks per arm (D more: see the samples slide), hence
100 mm cells, one wall group wide; cells with fewer than 20 tracks are empty.</p>
<p><b>Wall</b> (top): of tracks for which a plastic bar fired, the share that
lit the predicted wall group. The weak-group list in the title uses the interior
of each group (2 σ from its boundaries) and arms A, C, D.</p>
<p><b>Plastic</b> (bottom): of tracks that lit the wall at both ends, the share
that lit the predicted bar. Without the trigger this is ~55 %: sub-MeV electrons
stop in the 3 mm wall and its wrapping. So the level is the beam's energy
spectrum, and the map's <i>shape</i> is the detector.</p>""",
            short='unbiased')


def stack_diagram(G, lam_a):
    """One arm from the side, to scale: capsule, chamber, wall, plastic,
    liquid; a track and its extrapolation."""
    g = G.loc['A']
    s = 1.12                      # px per mm
    x0, yc = 70, 300
    W = lambda w: x0 + s * w      # noqa: E731
    Y = lambda u: yc - s * u      # noqa: E731
    ws = g['w_strip']
    lay = [('MM chamber', ws - 30, ws, -190, 190, '#c9d6e8',
            'the micromegas TPC: drift gap in front of the strip plane; the '
            'track is fitted to its waveforms (u, v at the strip plane, and two '
            'slopes)'),
           ('SiPM wall', ws + g['L_wall'], ws + g['L_wall'] + 3, WALL_EDGES[0],
            WALL_EDGES[-1], '#e8c9a8',
            f'3 mm scintillator bars, four read-out groups; {g["L_wall"]:.1f} '
            'mm past the strip plane'),
           ('plastic', ws + g['L_plas'], ws + g['L_plas'] + 20,
            g['plas_u_1'] - PLAS_HALF_U, g['plas_u_2'] + PLAS_HALF_U,
            '#d6c7e6', f'two 20 mm PVT bars (L, R); {g["L_plas"]:.0f} mm past '
            'the strip plane'),
           ('liquid', ws + g['L_ls'], ws + g['L_ls'] + 60,
            g['u_ls'] - LS_HALF_U, g['u_ls'] + LS_HALF_U, '#cfe3d4',
            f'one 451 x 451 mm liquid cell; {g["L_ls"]:.0f} mm past the strip '
            'plane (depth drawn nominal)')]
    o = []
    o.append(sd.line(W(-20), Y(0), W(560), Y(0), sd.RULE, 1.5, '6 6'))
    o.append(f'<circle cx="{W(0):.1f}" cy="{Y(CAPSULE_XYZ[0]):.1f}" r="{s * 10:.1f}" '
             f'fill="#f3d2c9" stroke="{sd.RED}" stroke-width="2"'
             f'{sd.tipattr("the ³He capsule, Ø20 mm; its position is the single-track imaging measurement")}/>')
    o.append(sd.T(W(0), Y(CAPSULE_XYZ[0]) + 52, 'capsule', 22, sd.RED))
    for name, w0, w1, u0, u1, col, tip in lay:
        o.append(f'<rect x="{W(w0):.1f}" y="{Y(u1):.1f}" width="{s * (w1 - w0):.1f}" '
                 f'height="{s * (u1 - u0):.1f}" fill="{col}" stroke="{sd.INK}" '
                 f'stroke-width="1.5"{sd.tipattr(tip)}/>')
        o.append(sd.T(W(0.5 * (w0 + w1)), Y(u1) - 12, name, 21, sd.INK))
    for e in WALL_EDGES[1:-1]:
        o.append(sd.line(W(ws + g['L_wall']) - 6, Y(e), W(ws + g['L_wall'] + 3) + 6,
                         Y(e), sd.INK, 2))
    # one track from the capsule: true line, and the chamber's view of it
    tt = 0.42
    ua = CAPSULE_XYZ[0] + tt * ws
    o.append(sd.line(W(0), Y(CAPSULE_XYZ[0]), W(ws), Y(ua), sd.RED, 3))
    end = ws + g['L_ls'] + 70
    for frac, col, dash, lab, dy in ((1.0, sd.GREY, '10 6', 'bare k', -6),
                                     (lam_a, sd.BLUE, None,
                                      f'{lam_a:.2f} × k', 18)):
        u_l = ua + frac * tt * (end - ws)
        o.append(sd.line(W(ws), Y(ua), W(end), Y(u_l), col, 3, dash))
        o.append(sd.T(W(end) + 8, Y(u_l) + dy, lab, 21, col, 'start'))
    o.append(sd.T(W(ws / 2), Y(-235), f'{ws:.1f} mm to the strip plane', 21))
    return sd.svg(int(W(end) + 110), 600, ''.join(o),
                  'one arm from the side, to scale')


# --------------------------------------------------------------------------- #
def build():
    meta = json.loads((ana() / 'meta.json').read_text())
    G = geo()
    PF = rd('pointing')
    LR = rd('lam_run')
    S = rd('scales')
    E = rd('eff')
    EM = rd('eff_map')
    GS = rd('gain')
    GM = rd('gain_map')
    EP = rd('edge_profile')
    LV = rd('liq_vs_plas')
    RB = rd('by_run')
    ext = json.loads((paths.spell('scint') / 'extract.meta.json').read_text())
    cal = meta['cal']

    def eff(arm, lay, probe, samp):
        r = E[(E.arm == arm) & (E.layer == lay) & (E.probe == probe)
              & (E['sample'] == samp)]
        return ((float(r.eff.iloc[0]), float(r.err.iloc[0]), int(r.n.iloc[0]))
                if len(r) else (np.nan, np.nan, 0))

    def pf(arm, layer, axis, variant, col, fp=2):
        r = PF[(PF.arm == arm) & (PF.layer == layer) & (PF.axis == axis)
               & (PF.variant == variant) & (PF.fit_pass == fp)]
        return r[col].iloc[0] if len(r) else np.nan

    def old(arm, layer, col):
        r = S[(S.arm == arm) & (S.layer == layer) & (S.axis == 'u')
              & (S.fit_pass == 2)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def gain(arm, lay, ch, samp='all_late'):
        r = GS[(GS.arm == arm) & (GS.layer == lay) & (GS.channel == ch)
               & (GS['sample'] == samp)]
        return float(r['median'].iloc[0]) if len(r) else np.nan

    D = sd.Deck('Scintillator stack, mapped by the tracks',
                'Every SiPM wall, plastic and liquid of the n_TOF 2026 station, '
                'mapped in efficiency and response by the MM tracks that point '
                'at them, on the single-track imaging calibration.')

    # ------------------------------------------------------------- 1 cover
    wA, wC, wD = (eff(a, 'wall', 'wany', 'unbiased')[0] for a in 'ACD')
    lam = {a: cal[a]['lam'] for a in ARMS}
    offw = {a: cal[a]['off_w'] for a in ARMS}
    lb = {a: eff(a, 'liq', 'lf_walltag_beside_plastic', 'all_late')[0]
          for a in ARMS}
    lf = {a: eff(a, 'liq', 'lf_walltag_behind_plastic', 'all_late')[0]
          for a in ARMS}
    omax = max(abs(offw[a]) for a in 'ACD')
    lam_rng = f'{min(lam[a] for a in "ACD"):.2f}–{max(lam[a] for a in "ACD"):.2f}'
    lR = {a: eff(a, 'liq', 'lf_behind_bar2', 'all_late')[0] for a in ARMS}
    lL = {a: eff(a, 'liq', 'lf_behind_bar1', 'all_late')[0] for a in ARMS}
    cover = (sd.kicker('n_TOF 2026 · X17 station · scintillator stack')
             + sd.p('On the imaging calibration, the tracks map all twelve '
                    'counters — and the liquids answer on one side only',
                    64, sd.DINK, 600, 'line-height:1.1;letter-spacing:-1px')
             + sd.p(f'{ext["n_tracks"]:,} MM tracks over {ext["n_runs"]} runs, '
                    'extrapolated from the per-run k that imaged the capsule '
                    'through the SiPM wall, the plastic and the liquid of each '
                    'arm; the wall’s group boundaries calibrate the lever.',
                    30, sd.DMUT)
             + sd.row(
                 sd.bignum(f'≤ {omax:.0f} mm',
                           'wall group edges vs the survey', sd.DBLUE,
                           f'A, C, D. The best predictor carries {lam_rng} '
                           'of k·tan forward: the scintillators prefer a '
                           'shallower slope than k',
                           tip='Common offset of the three internal wall-group '
                           'boundaries from their surveyed positions, with the '
                           'extrapolation calibrated on them.'),
                 sd.bignum(f'{pc(wA)} · {pc(wC)} · {pc(wD)}',
                           'wall efficiency, A · C · D', sd.DGREEN,
                           'given the plastic fired; events another arm '
                           'triggered, > 10 ms'),
                 sd.bignum(f'{pc(lR["A"])} vs {pc(lL["A"])}',
                           'liquid A behind the R bar vs the L bar', sd.DRED,
                           f'D the same ({pc(lR["D"])} vs {pc(lL["D"])}); C '
                           f'{pc(eff("C", "liq", "lf", "all_late")[0], 1)} '
                           'everywhere'),
                 gap=56))
    D.slide('cover', cover, '', dark=True, short='cover')

    # ------------------------------------------------------------- 2 setup
    hint = sd.p('Hover any dotted term, map cell or data point for its value, '
                'its counts and where it comes from.', 22, sd.MUT)
    steps = [dict(label='MM track', sub='u, v and two slopes at the strip plane',
                  color=sd.BLUE, tip='Gated tracks from the campaign full pass '
                  '(stage-3 tables), single-track arms only for the maps.'),
             dict(label='trust cuts', sub='own t0 in time · chamber cell '
                  'confirmed', color=sd.GREY,
                  tip='About a third of tracks crossed the chamber at another '
                  'time than the trigger; some chamber cells reconstruct tracks '
                  'nothing confirms (most on D). Both are removed.'),
             dict(label='extrapolate', sub='imaging k, coefficients from the '
                  'wall boundaries', color=sd.BLUE),
             dict(label='tag & probe', sub='net of a same-width pre-trigger '
                  'window', color=sd.GREEN,
                  tip='Efficiency of a layer = P(it fired | another layer '
                  'says a particle passed), minus the accidental rate in '
                  '(−560, −400) ns.'),
             dict(label='maps', sub='efficiency · median response',
                  color=sd.PURPLE)]
    body = (sd.title('Every track is walked outward through three counters',
                     'one arm from the side, to scale (arm A survey); the same '
                     'stack sits behind each of the four chambers')
            + sd.row(stack_diagram(G, lam['A']),
                     sd.col(sd.flow(steps[:3], size=22),
                            sd.flow(steps[3:], size=22),
                            sd.callout('Grey dashed: the bare imaging slope. '
                                       'Blue: the slope the wall boundaries '
                                       f'prefer on arm A, {lam["A"]:.2f} of it '
                                       '(slide 3).', sd.BLUE, 24),
                            hint, gap=24, w=700), gap=40, align='center'))
    D.slide('setup', body, f"""
<p>Geometry is read from each run's <code>run_config.json</code> and asserted
identical in all {ext['n_runs']} runs: levers past the strip plane of
{G.L_wall.iloc[0]:.1f} mm (wall), {G.L_plas.min():.0f}–{G.L_plas.max():.0f} mm
(plastic) and {G.L_ls.min():.0f}–{G.L_ls.max():.0f} mm (liquid).</p>
<p>The hit side is the n_TOF slim: every wall end, both plastic bars and the
liquid in the prompt window (−100, +60) ns and in a same-width window before the
trigger, (−560, −400) ns, which sets the accidental floor of every rate in this
note.</p>
<p>Pipeline: <code>ntof_scint_stack/extract.py</code> → <code>ana.py</code> →
<code>make_deck.py</code> (this note) and <code>make_report.py</code> (the long
report). Outputs: <code>/media/dylan/data/x17/scint_stack/</code>.</p>""",
            short='setup')

    # ------------------------------------------------------------- 3 calibration
    P1 = sd.Plot(810, 560, x=(0, 4), y=(0, 30), xlabel='',
                 ylabel='wall edge width, mm',
                 title='How sharply each extrapolation finds the wall groups')
    P1.yticks([(v, str(v)) for v in (0, 10, 20, 30)])
    P1.xticks([(i + 0.5, a) for i, a in enumerate(ARMS)])
    order = ('old', 'k_only', 'capsule', 'used')
    for i, a in enumerate(ARMS):
        for j, var in enumerate(order):
            if var == 'old':
                sg, n = old(a, 'wall', 'sigma'), old(a, 'wall', 'n')
                tip = (f'{a}: free raw-tan scale with one offset per boundary '
                       f'(the 2 Oct analysis)\nedge width {sg:.1f} mm')
            else:
                sg = pf(a, 'wall', 'u', var, 'sigma')
                al, lm = (pf(a, 'wall', 'u', var, c) for c in ('alpha', 'lam'))
                off = pf(a, 'wall', 'u', var, 'offset')
                tip = (f'{a}: {VAR_NAME[var]}\nalpha {al:.3f}, lam {lm:.3f}, '
                       f'offset {off} mm\nedge width {sg:.1f} mm')
            P1.vbar(i + 0.17 + 0.22 * j, sg, 36, VAR_COL[var], tip=tip)
    # right: effective scale per run
    rk = LR[LR.arm.isin(['A', 'C', 'D'])]
    P2 = sd.Plot(810, 560, x=(80, 165), y=(0.6, 1.9), xlabel='run',
                 ylabel='slope scale × raw tan',
                 title='Per run: the imaging k (open) and what the wall uses')
    P2.xticks([(r, str(r)) for r in (80, 100, 120, 140, 160)])
    P2.yticks([(v, f'{v:.1f}') for v in (0.6, 1.0, 1.4, 1.8)])
    P2.band([128, 147], [0.6, 0.6], [1.9, 1.9], sd.GOLD, 0.12,
            tip='the 3–5 Aug k block, runs 128–147: k rises on every arm')
    for a in ('A', 'C', 'D'):
        r = rk[rk.arm == a].sort_values('rn')
        tk = [f'{a} {x.run}: k {x.k:.3f}' + (' (campaign median: no own k)'
                                             if x.k_fill else '')
              for x in r.itertuples()]
        te = [f'{a} {x.run}: lam·k = {x.eff_scale:.3f} (lam {x.lam:.3f} ± '
              f'{x.lam_err:.3f}, k {x.k:.3f})\nwall offset {x.offset:+.1f} mm, '
              f'{x.n:,} edge tracks' for x in r.itertuples()]
        P2.line(r.rn.tolist(), r.k.tolist(), ARM_COL[a], 1.5, dash='4 4',
                markers=True, r=5, tips=tk)
        P2.line(r.rn.tolist(), r.eff_scale.tolist(), ARM_COL[a], 3, tips=te)
    # the k-block test: how much of k's rise the wall's effective scale follows
    blk = {}
    for a in ('A', 'C', 'D'):
        r = rk[rk.arm == a]
        kin, kout = r[r.k_block].k.median(), r[~r.k_block].k.median()
        ein, eout = r[r.k_block].eff_scale.median(), r[~r.k_block].eff_scale.median()
        blk[a] = (kin / kout - 1, ein / eout - 1)
    blk_txt = '; '.join(f'{a}: k {100 * x:+.1f} %, λ·k {100 * y:+.1f} %'
                        for a, (x, y) in blk.items())
    pb = {a: json.loads(pf(a, 'wall', 'u', 'per_boundary', 'offset'))
          for a in ARMS}
    pb_txt = '; '.join(f'{a} ' + ' / '.join(f'{v:+.0f}' for v in pb[a])
                       for a in ARMS)
    leg = sd.legend([(VAR_NAME[v], VAR_COL[v], 'box') for v in order]
                    + [(f'arm {a}', ARM_COL[a]) for a in 'ACD'], 22)
    ttl = (f'The wall groups pin the extrapolation: k as input, {lam_rng} of it '
           'carried forward')
    body = (sd.title(ttl, 'single in-time tracks in confirmed chamber cells, '
                     'wall boundaries held at the survey') + leg
            + sd.row(P1.svg('edge widths'), P2.svg('scale per run'), gap=44))
    rows = ''.join(
        f'<tr><td>{a}</td><td>{G.loc[a].w_strip:.1f}</td>'
        f'<td>{pf(a, "wall", "u", "used", "alpha"):.3f}</td>'
        f'<td>{pf(a, "wall", "u", "used", "lam"):.3f} ± {pf(a, "wall", "u", "used", "lam_err"):.3f}</td>'
        f'<td>{cal[a]["off_w"]:+.1f}</td><td>{pf(a, "wall", "u", "per_boundary", "offset")}</td>'
        f'<td>{pf(a, "wall", "u", "capsule", "lam"):.3f}</td>'
        f'<td>{cal[a]["off_p"]:+.1f}</td><td>{cal[a]["lam_v"]:.2f}</td>'
        f'<td>{meta["k_median"][a]:.3f}</td><td>{pc(meta["frac_k_fill"][a])}</td></tr>'
        for a in ARMS)
    D.slide('calibration', body, f"""
<p><b>What was asked.</b> The single-track imaging of the capsule (the Athens
side view and the 33-run band crossings) fixes a per-run angle scale k, and the
question was whether to use it here. It is used: every slope in this note is
k × tan_raw (the stage-3 <code>tanx</code>), with each run's own k from
<code>&lt;out&gt;/kcal</code>.</p>
<p><b>Correction to the 2 October analysis.</b> That pass compared its fitted
scale with 1/k (0.81 on A). The track tables define tanx = k × tan_raw, so the
imaging scale is k itself (1.27 on run_145 A). Its "A scale is k × 1.10"
statement compared against the wrong number.</p>
<p><b>The predictor.</b> A track's crossing at a layer is predicted as
u + L (α a + λ m) − δ, with m = k·tan_raw its imaging slope and a = (u − u<sub>c</sub>)/
{G.w_strip.iloc[0]:.1f} the slope it would have from the capsule centre
(u<sub>c</sub> from the imaging: X = {CAPSULE_XYZ[0]}, Z = {CAPSULE_XYZ[2]} mm). α, λ and
the alignment δ are fitted to which wall group fires, on single tracks lighting
exactly one group, at the three internal boundaries held at the survey. Three
special cases are fitted beside the free fit: the bare imaging k (α = 0, λ = 1),
the imaging k with the slope's noise shrunk toward the capsule (α = 1 − λ, the
Bayes predictor for a pointing population), and one offset per boundary (the
check).</p>
<p><b>What it finds.</b> α ≈ 0 and λ ≈ {lam_rng} on A, C and D. With that, the
wall sits at the survey (common offset ≤ {omax:.0f} mm) and the edges are sharper
than with any special case. Freeing one offset per boundary (−125 / −25 / +75 mm)
gives {pb_txt} mm: on A the three agree, on B, C and D the middle boundary sits
12–16 mm high — the 2 October free fit saw the same, so it is the wall (or the
chamber centre), not this model. The capsule
form, the one that would say "k is right and the slopes are just noisy", is
clearly worse. Split by the track's own slope error the λ is the same in the
four best fifths and collapses to ≈ 0 (with α → 1) in the worst fifth: junk slopes
are best replaced by the capsule direction, which is the expected behaviour.</p>
<p><b>What it does not decide.</b> Read as a scale, λ·k is what the scintillators
prefer: on A, 1/λ ≈ {1 / lam["A"]:.2f}, the same size as <code>det_a_scint</code>'s
"A tangents 33 % too large" (2026-09-10). But a boundary fit regresses on a noisy
slope, and noise that does not pull toward the capsule (for instance a non-pointing
admixture in the sample) would also shrink λ. So this is a <i>predictor</i>
calibration, sufficient for mapping, and a pointer for the October angle-scale
work, not a measurement of the angle scale.</p>
<p><b>The k block</b> (right panel; runs 128–147, block median against the rest):
{blk_txt}. The wall follows about half of k's rise on C and D: neither "the block
is real pointing" (λ·k would rise with k) nor "the block is a pure k artefact"
(λ·k would stay flat) on its own.</p>
<table><tr><th>arm</th><th>w_strip</th><th>α</th><th>λ</th><th>wall δ</th>
<th>per-boundary δ</th><th>λ (capsule form)</th><th>plastic δ</th><th>λ_v</th>
<th>median k</th><th>k filled</th></tr>{rows}</table>
<p>The plastic has a single boundary (its L/R gap), which cannot separate α
from λ, so it takes the wall's and fits only its offset. On B, C and D that
offset is ~25 mm: the gap between the two bars sits away from where
<code>run_config.json</code> puts it, by about the same amount on three arms,
under every predictor tried. A survey question, recorded, not corrected. λ_v is
scanned against the wall's own ln(top/bottom) position (no MM input). Arm B has
no drift-field rings and k in only a few runs; its maps use the campaign-median
B k and are the least sharp.</p>""",
            short='calibration')

    # ------------------------------------------------------------- 4 edges
    pans = []
    for i, a in enumerate(ARMS):
        e = EP[(EP.arm == a) & (EP.layer == 'wall')]
        P = sd.Plot(810, 300, x=(-260, 200), y=(0, 1),
                    xlabel='predicted u at the wall, mm' if i >= 2 else '',
                    ylabel='share', margin=(10, 20, 70 if i >= 2 else 30, 90),
                    title=f'chamber {a}')
        P.yticks([(0, '0'), (0.5, '.5'), (1, '1')])
        if i >= 2:
            P.xticks([(v, f'{v:+d}' if v else '0') for v in (-200, -100, 0, 100, 200)])
        for ed in WALL_EDGES:
            P.vline(ed, sd.INK, '4 5', 1.5,
                    tip=f'surveyed group edge, u = {ed:+.0f} mm')
        for gg, col in zip(range(4), (sd.BLUE, sd.ORANGE, sd.GREEN, sd.PURPLE)):
            q = e[e.what == f'g{gg}'].sort_values('x')
            tips = [f'{a} at u {x.x:+.0f} mm: group {gg} is {100 * x.p:.0f} % '
                    f'of {x.n:,} one-group events' for x in q.itertuples()]
            P.line(q.x.tolist(), q.p.tolist(), col, 3, markers=False, tips=None)
            P.points(q.x.tolist()[::3], q.p.tolist()[::3], col, 4,
                     tips=tips[::3])
        pans.append(P.svg(f'wall edges {a}'))
    e = EP[(EP.arm == 'D') & (EP.layer == 'wall') & (EP.x < -170)]
    d_g2 = float(np.average(e[e.what == 'g2'].p, weights=e[e.what == 'g2'].n))
    body = (sd.title('…and the wall groups switch at their surveyed edges — '
                     f'except D’s group 0: beyond −170 mm, {pc(d_g2)} light group 2',
                     'which group fired, against where the calibrated track '
                     'crosses the wall; late single tracks, exactly one group lit')
            + sd.legend([(f'group {g}', c) for g, c in
                         zip(range(4), (sd.BLUE, sd.ORANGE, sd.GREEN,
                                        sd.PURPLE))], 22)
            + '<div style="display:grid;grid-template-columns:810px 810px;gap:8px 44px">'
            + ''.join(pans) + '</div>')
    D.slide('edges', body, """
<p><b>Chamber D below −170 mm.</b> Tracks the chamber puts there light group 2
(u −25 to +75 mm) about half the time and group 0 only 10–30 %; group 0 answers
properly only between −165 and −125 mm. That is either chamber D misplacing
tracks near its −u edge or part of group 0's light reaching group 2's channel
(cabling); the scintillator data alone cannot tell which. It is also why D g0's
efficiency is low on slide 5.</p>
<p>Each curve is the share of one-group events in which that group fired, in
10 mm bins of the predicted crossing. With a perfect predictor each would be a
step at the dotted lines; the width of the transition is the extrapolation's
resolution at the wall (edge widths on slide 3), and its position is the
alignment. Chamber D's centre and its u &lt; −180 mm region are distorted by the
chamber (the trust mask removes most of its tracks), and B's slopes are the
poorest. The plastic and liquid equivalents are in the long report.</p>""",
            short='wall edges')

    # ------------------------------------------------------------- 5 samples
    samples_slide(D, rd('funnel'), EM, eff)

    # ------------------------------------------------------------- 6-8 maps
    gr = {}
    for a in ARMS:
        gg = np.array([gain(a, 'wall', f'g{k}') for k in range(4)])
        gr[a] = gg / np.nanmedian(gg)
    lo_a, lo_k = min(((a, k) for a in ARMS for k in range(4)),
                     key=lambda x: gr[x[0]][x[1]])
    nw = {a: eff(a, 'wall', 'wany_tol_full', 'all_late') for a in ARMS}
    heat_slide(
        D, 'map-wall', 'wall maps',
        f'SiPM walls over the whole face: {lo_a} g{lo_k} answers at '
        f'{gr[lo_a][lo_k]:.2f} of its neighbours; '
        f'{min(n[2] for n in nw.values()) / 1e3:.0f}k–'
        f'{max(n[2] for n in nw.values()) / 1e3:.0f}k tracks per arm',
        'top: P(predicted group fired | a plastic bar fired), every late trigger, '
        '25 mm cells · bottom: MIP response √(top × bottom)·cosθ ÷ the face median',
        'wall_full', G, EM, GM, f"""
<p><b>Sample.</b> {SAMPLE_WORDS['all_late']}, single tracks, in time and in a
trusted chamber cell. On most of these events the arm triggered itself, which
needs a wall group and a plastic bar above threshold: the level is biased toward
1 and the plastic's 400 × 300 mm footprint is printed on the map (above and below
it, |v| &gt; 150 mm, a wall hit had to come with a plastic hit from elsewhere).
What the map shows reliably is a channel that does not answer where its
neighbours do. The trigger-free shape is on the unbiased slide.</p>
<p><b>Efficiency</b> (top): the predicted wall group fired at either end, net of
the pre-trigger rate and of accidental tags. Within 2 σ of an internal group
boundary either neighbour counts, so the map covers the whole wall instead of
losing a ±{2 * cal['A']['sig_w']:.0f} mm band around every boundary. 25 mm cells;
empty cells have fewer than 30 tracks.</p>
<p><b>Response</b> (bottom): the geometric mean of the two ends times cosθ, from
the group that answered. For an exponentially attenuating bar this does not
depend on where along the bar the light was made, so it is the scintillator's
own MIP response. Both ends lit, neither saturated. Medians (the response is a
Landau). Each face is divided by its own median, printed in the panel title,
so the colour shows uniformity; the arms differ in absolute gain (WALA ~30 %
below the others, as on the July bench).</p>""",
        eff_min=0.5, eff_ticks=[(0.5, '0.5'), (0.75, '0.75'), (1, '1')])
    heat_slide(
        D, 'map-plas', 'plastic maps',
        f'Plastic over the whole face: no holes, ≥ '
        f'{100 * min(eff(a, "plas", "pm_full", "all_late")[0] for a in ARMS):.1f} % '
        'on every late trigger',
        'top: P(bar fired | wall fired at both ends), every late trigger, 25 mm · '
        'bottom: deposit keVee·cosθ ÷ the face median',
        'plas_full', G, EM, GM, f"""
<p><b>Sample.</b> {SAMPLE_WORDS['all_late']}. The trigger needs a plastic bar,
so on self-triggered events the level reads near 1 by construction; the map can
still show a corner or a light-guide shadow that does not answer. The unbiased
level (~55 %, particles that stop in the wall) is on the unbiased slide.</p>
<p><b>Efficiency</b>: of tracks whose predicted wall group fired at both ends
(either neighbour near a group boundary), the share that also lit the predicted
plastic bar. Within 1.5 edge widths of the L/R gap either bar counts. No margin
from the outer edges: the fall-off there is the extrapolation's resolution
(σ ≈ {min(cal[a]['sig_p'] for a in ARMS):.0f}–{max(cal[a]['sig_p'] for a in ARMS):.0f} mm
at the plastic).</p>
<p><b>Response</b>: the predicted bar's amplitude × cosθ, in keVee from the
28 July two-source calibration. The trigger needs a plastic bar above ~0.9 MIP,
so on self-triggered events the spectrum is cut just below its own median: the
level is biased up, roughly evenly over the face, and the map is divided by its
own median. A bar brighter near its PMT shows as a gradient along u.</p>""",
        eff_min=0.5, eff_ticks=[(0.5, '0.5'), (0.75, '0.75'), (1, '1')])
    def halves(arm):
        """Liquid net firing on the PMT half of the cell against the far
        half (|coordinate| > 75 mm), from the whole-face late map."""
        M = EM[(EM.arm == arm) & (EM.layer == 'liq_full')
               & (EM['sample'] == 'all_late')]
        c = M.x if LIQ_PMT[arm] == '+u' else M.y

        def r(m):
            n = M.n[m].sum()
            return (M.k_on[m].sum() - M.k_off[m].sum()) / max(n, 1)
        return r(c > 75), r(c < -75)
    hv = {a: halves(a) for a in ARMS}
    hor = [a for a in ARMS if LIQ_PMT[a] == '+u']
    ver = [a for a in ARMS if LIQ_PMT[a] != '+u']
    rat = {a: hv[a][0] / hv[a][1] if hv[a][1] > 0 else np.inf for a in ARMS}
    li = heat_slide(
        D, 'map-liq', 'liquid maps',
        f'Liquids: {" and ".join(hor)} (PMT to the side) answer '
        f'{min(rat[a] for a in hor):.0f}–{max(rat[a] for a in hor):.0f}× more '
        f'on their PMT half; {" and ".join(ver)} (PMT on top) do not',
        'tagged by the wall alone (both ends), so the whole cell is mapped · '
        'white dashed: the plastic in front · every trigger > 10 ms, 50 mm cells',
        'liq_full', G, EM, GM, f"""
<p>The liquid is in no trigger, so the late sample is not biased by it. Tag:
the predicted wall group lit at both ends (either neighbour near a group
boundary). Probe: the liquid fired. The tag does not need the plastic, so the
map covers the whole 451 × 451 mm cell — inside the dashed box a particle crossed
20 mm of PVT first (which stops electrons below ~4 MeV), outside it did not.
Coordinates are relative to the cell centre.</p>
<p>The <b>response</b> row is the liquid's amplitude ÷ its face median, in keVee
where the 28 July calibration has a liquid edge (A, B, D) and in mV for C, which
has none. No path-length correction: what reaches the liquid is not a straight
continuation of the MM track.</p>
<p><b>The PMTs</b> (grey: the 90 mm funnel narrowing to the 50 mm neck that
holds the PMT) are drawn where the Geant4 model puts them, from the 17–18 July
survey: vessels horizontal with the PMT to the right (+u, looking from behind
the wall) on {' and '.join(hor)}, vertical with the PMT on top on
{' and '.join(ver)}. The data agree independently.</p>
<p><b>Horizontal vessels answer near their PMT.</b> Net firing on the PMT half
of the cell (|u| &gt; 75 mm) against the far half:
{'; '.join(f'{a} {pc(hv[a][0], 1)} vs {pc(hv[a][1], 1)}' for a in hor)} — a smooth
rise toward the PMT, not a step at the plastic's bar gap. This is the
one-sided response the July source runs saw, now explained by where the PMT is.
<b>The vertical vessels do not:</b> top half against bottom half,
{'; '.join(f'{a} {pc(hv[a][0], 2)} vs {pc(hv[a][1], 2)}' for a in ver)}, if anything
lower near the PMT. Why light collection falls so steeply with distance on
the horizontal vessels and not on the vertical ones is open.</p>""",
        eff_max=0.2, eff_ticks=[(0, '0'), (0.1, '0.1'), (0.2, '0.2')],
        gain_rel=(0.5, 1.5))

    # ------------------------------------------------------------- 9 unbiased
    weak = sorted(((eff(a, 'wall', f'wany_g{k}', 'unbiased')[0], a, k)
                   for a in 'ACD' for k in range(4)))
    weak = [w for w in weak if w[0] < 0.75]
    unbiased_slide(D, G, EM, eff, weak)

    # ------------------------------------------------------------- 8 liquid
    rowsb = []
    for a in ARMS:
        b_, f_ = (eff(a, 'liq', f'lf_walltag_{w}', 'all_late')
                  for w in ('beside_plastic', 'behind_plastic'))
        l1, l2 = (eff(a, 'liq', f'lf_behind_bar{k}', 'all_late')
                  for k in (1, 2))
        rowsb.append((a, b_, f_, l1, l2))
    P1 = sd.Plot(760, 560, x=(0, 4), y=(0, 0.12), ylabel='liquid fired (net)',
                 title='Where the particle reaches the cell')
    P1.yticks([(v, f'{100 * v:.0f} %') for v in (0, 0.04, 0.08, 0.12)])
    P1.xticks([(i + 0.5, a) for i, a in enumerate(ARMS)])
    cols4 = (('beside the plastic', sd.GREEN), ('behind the plastic', sd.GREY),
             ('behind bar L', sd.BLUE), ('behind bar R', sd.ORANGE))
    for i, (a, *vals) in enumerate(rowsb):
        for j, ((nm, c), v) in enumerate(zip(cols4, vals)):
            P1.vbar(i + 0.17 + 0.22 * j, max(v[0], 0) if np.isfinite(v[0]) else 0,
                    34, c, tip=f'{a}, {nm}: {100 * v[0]:.1f} ± {100 * v[1]:.1f} % '
                    f'of {v[2]:,} tagged tracks')
    ymax = float(np.ceil(LV[LV['sample'] == 'all_late'].eff.max() * 10 + 0.5) / 10)
    P2 = sd.Plot(860, 560, x=(500, 20000, 'log'), y=(0, ymax),
                 xlabel='energy left in the plastic, keVee',
                 ylabel='liquid fired (net)',
                 title='Against what the plastic saw (wall + plastic tag)')
    P2.xticks([(1000, '1 MeV'), (3000, '3'), (10000, '10 MeV')])
    P2.yticks([(v, f'{100 * v:.0f} %') for v in np.linspace(0, ymax, 4)])
    for a in ARMS:
        q = LV[(LV.arm == a) & (LV['sample'] == 'all_late')].sort_values('e_lo')
        xm = np.sqrt(np.clip(q.e_lo, 500, None) * np.clip(q.e_hi, None, 20000))
        tips = [f'{a}: plastic {x.e_lo / 1e3:.1f}–' + (f'{x.e_hi / 1e3:.1f}' if x.e_hi < 1e8 else '∞')
                + ' MeVee (drawn at the bin’s geometric centre, last bin at 15.5)\n'
                f'liquid {100 * x.eff:.1f} ± {100 * x.err:.1f} % of {x.n:,}'
                for x in q.itertuples()]
        P2.line(xm.tolist(), q.eff.tolist(), ARM_COL[a], 3, tips=tips)
    lt = (f'A liquid answers by where it is hit, not by what is in front: '
          f'A {pc(lR["A"], 1)} behind the R bar, {pc(lL["A"], 1)} behind the L, '
          f'{pc(lb["A"], 1)} beside the plastic')
    body = (sd.title(lt, 'late single tracks; left tagged by the wall (beside '
                     '/ behind) or wall and plastic (by bar); right binned in '
                     'the plastic deposit')
            + sd.legend([(n, c, 'box') for n, c in cols4]
                        + [(f'arm {a}', ARM_COL[a]) for a in ARMS], 22)
            + sd.row(P1.svg('liquid by region'), P2.svg('liquid vs plastic'),
                     gap=44))
    D.slide('liquid', body, """
<p>Left: the liquid's firing probability for wall-tagged tracks whose predicted
crossing is beside the plastic (no PVT in front) and behind it, and, behind it,
split by which bar. If the plastic were simply absorbing what the liquid would
see, "beside" would be the highest bar. It is not: on A and D the liquid fires
where it is struck near its +u edge (behind the R bar) and hardly anywhere else,
plastic or no plastic — the map on the previous slide shows the same thing as a
gradient. "Beside" is mostly the cell's top and bottom margins (|v| &gt; 150 mm)
and its far −u edge, the dim parts of the gradient. Right: with the wall and the plastic both lit, against the
energy the plastic recorded. A through-going minimum-ionising particle leaves
~4 MeV in 20 mm of PVT; the liquid answers to that and to the high tail, which
is why its response is a punch-through probability for this beam and not a pure
detector efficiency. LIQ C needs deposits above ~8 MeVee in the plastic before it
answers at all — alive, but with a threshold or gain far from the others.</p>""",
            short='liquid')

    # ------------------------------------------------------------- 9 summary
    trs, tips = [], []
    for a in ARMS:
        w, wl = eff(a, 'wall', 'wany', 'unbiased'), eff(a, 'wall', 'wany', 'all_late')
        p_, pl = eff(a, 'plas', 'pm', 'unbiased'), eff(a, 'plas', 'pm', 'all_late')
        lq = eff(a, 'liq', 'lf_walltag', 'all_late')
        trs.append([f'<b>{a}</b>', f'{pc(w[0], 1)} ± {pc(w[1], 1)}', pc(wl[0], 1),
                    f'{pc(p_[0], 1)} ± {pc(p_[1], 1)}', pc(pl[0], 1),
                    pc(lq[0], 1), f'{gain(a, "wall", "g1"):.0f}',
                    f'{gain(a, "plas", "bar1", "unbiased") / 1e3:.2f} / '
                    f'{gain(a, "plas", "bar2", "unbiased") / 1e3:.2f}'])
        tips.append(f'{a}: {w[2]:,} unbiased wall tags, {p_[2]:,} unbiased '
                    f'plastic tags, {lq[2]:,} wall-tagged liquid probes')
    tab = sd.table(['arm', 'wall | plastic<br>unbiased', 'self-<br>triggered',
                    'plastic | wall<br>unbiased', 'self-<br>triggered',
                    'liquid | wall', 'wall g1<br>MIP, mV',
                    'plastic L / R<br>MeVee'], trs, 26, tips=tips)
    body = (sd.title('The numbers behind the maps',
                     'net of accidentals; unbiased = another arm triggered, '
                     '> 10 ms; liquid on every late trigger')
            + tab
            + sd.row(sd.callout('The trigger bias is large: a wall that the '
                                'arm’s own trigger required reads 89–98 % '
                                'on its own events. Quote the unbiased column.',
                                sd.ORANGE, 24),
                     sd.callout('Plastic | wall is a response probability for '
                                'this beam: two in five particles that cross '
                                'the 3 mm wall stop before the plastic.',
                                sd.BLUE, 24), gap=40))
    D.slide('summary', body, """
<p>Every efficiency is (P_on − P_off)/(1 − P_off), with P_off the same probe in
the pre-trigger window on the same tracks, corrected for the fraction of tags
that are themselves accidental (measured, a few per mille past 10 ms). Errors
are binomial on the prompt count. The unbiased tag samples are a few thousand per
arm (fewer on B), so the B row is rough. The wall MIP column is group 1's
median √(a1·a2)·cosθ; the plastic column the unbiased median deposit per bar.</p>""",
            short='numbers')

    # ------------------------------------------------------------- 10 stability
    P1 = sd.Plot(810, 520, x=(80, 165), y=(0.7, 1.3), xlabel='run',
                 ylabel='÷ campaign median', title='Wall MIP response, run by run')
    P2 = sd.Plot(810, 520, x=(80, 165), y=(0.7, 1.3), xlabel='run',
                 ylabel='÷ campaign median', title='Plastic deposit, run by run')
    for P in (P1, P2):
        P.xticks([(r, str(r)) for r in (80, 100, 120, 140, 160)])
        P.yticks([(v, f'{v:.1f}') for v in (0.7, 0.85, 1.0, 1.15, 1.3)])
        P.band([128, 147], [0.7, 0.7], [1.3, 1.3], sd.GOLD, 0.12,
               tip='the 3–5 Aug k block')
        P.hline(1.0, sd.MUT, '6 6', 1.5)
    spread = {}
    for a in ARMS:
        r = RB[RB.arm == a].sort_values('rn')
        for P, c, u in ((P1, 'wall_gm_mV', 'mV'), (P2, 'plas_kevee', 'keVee')):
            m = float(np.nanmedian(r[c]))
            y = (r[c] / m).tolist()
            tp = [f'{a} {x.run}: {getattr(x, c):,.0f} {u} ({getattr(x, c) / m:.3f} '
                  f'of the median)' for x in r.itertuples()]
            P.line(r.rn.tolist(), y, ARM_COL[a], 3, tips=tp, r=5)
            q = r[c] / m
            spread[(a, c)] = float(q.quantile(.9) - q.quantile(.1))
    sw = max(spread[(a, 'wall_gm_mV')] for a in ARMS)
    sp = max(spread[(a, 'plas_kevee')] for a in ARMS)
    body = (sd.title(f'The responses hold through the campaign: wall within '
                     f'{100 * sw:.0f} %, plastic within {100 * sp:.0f} % (p10–p90)',
                     'per-run medians, late single tracks, each arm ÷ its own '
                     'campaign median; post-access runs, all on the 23 July '
                     'noisy configuration')
            + sd.legend([(f'arm {a}', ARM_COL[a]) for a in ARMS], 22)
            + sd.row(P1.svg('wall per run'), P2.svg('plastic per run'), gap=44))
    D.slide('stability', body, """
<p>Every run here is after the 23 July readout-clock change (run_84 onward) and
after the 27 July access, so none of them straddles either condition in CLAUDE.md;
runs 79 and 81 are excluded. The k block (shaded) is where every arm's imaging k
rises together; nothing in the scintillator responses moves with it.</p>""",
            short='stability')

    # ------------------------------------------------------------- 11 closing
    items = [
        ('The angle scale itself.', 'The wall prefers about '
         f'{lam_rng} × k·tan with no pull toward the capsule. The cosmic '
         'review of 6 Oct points the same way and is smaller (true k on A '
         '≈ 1.11 against the beam’s 1.22–1.24, and angle-dependent). A '
         'boundary fit regresses on a noisy slope, so it cannot separate a k '
         'that reads steep from a non-pointing admixture. This calibrates the '
         'extrapolation; it does not measure the angle scale.'),
        ('Chamber D’s −u edge.', 'Tracks D puts beyond −170 mm light wall '
         'group 2, not group 0: chamber-D position near its edge, or wall '
         'cabling.'),
        ('The plastic L/R gap is 20–30 mm off on B, C, D.', 'Same size on three '
         'arms under every predictor (6 mm on A): a survey or config entry, '
         'not a fit artefact. Worth a check of the as-built drawing.'),
        ('Arm D’s extra unbiased tracks.', 'A third of D’s late tracks sit on '
         'events another arm triggered (A–C: 4–18 %). Real particles or chamber '
         'pick-up is not yet checked; D’s unbiased maps lean on them.'),
        ('Liquids: why only the horizontal ones?', 'A and D (PMT to the side) '
         'answer near their PMT; B and C (PMT on top) do not. The cause is open. '
         'Liquid “efficiency” is punch-through × efficiency for this beam.'),
        ('Plastic PMT positions are as reported.', 'On top for every bar; the '
         'Geant model has bare bars and this data cannot place them.'),
        ('MIP-clean efficiencies need the cosmics.', 'Run_149 is now '
         'reconstructed (87 sub-runs); joining it to the scintillators is the '
         'next pass.'),
    ]
    closing = (sd.kicker('What this does not rule out')
               + ''.join(sd.p(f'<b>{h}</b> {t}', 27, sd.DINK)
                         for h, t in items)
               + sd.p('Long report with every table: '
                      '<code>/media/dylan/data/x17/scint_stack/report.html</code>',
                      22, sd.DMUT))
    D.slide('open', '<div style="display:flex;flex-direction:column;gap:22px">'
            + closing + '</div>', '', dark=True, short='open')
    return D


def main() -> int:
    D = build()
    out = paths.spell('scint') / 'deck' / 'scint-stack.html'
    D.write(out, note_meta=dict(
        title='Scintillator stack, mapped by the tracks',
        summary='Efficiency and response heat maps of every SiPM wall, plastic '
                'and liquid on the four arms, from MM tracks on the imaging '
                'calibration.',
        tags='n_TOF, X17, scintillators, calibration',
        date=dt.date.today().isoformat()))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
