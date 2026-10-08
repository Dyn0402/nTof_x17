#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_deck.py -- the calorimetry + MM dE/dx slide note, built from the tables
the analysis wrote (`scint_ecal`, `liquid_salvage`, `mm_dedx_cosmics`,
`g4_response`, `pair_sensitivity`).  Re-run after any of them changes.

    PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        /media/dylan/data/x17/calorimetry/deck/calorimetry.html --slug calorimetry --force --deploy
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

from ntof_calorimetry import landau as LD  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from ntof_scint_stack.extract import PLAS_THR  # noqa: E402

ARMC = {'A': sd.BLUE, 'B': sd.ORANGE, 'C': sd.GREEN, 'D': sd.PURPLE}
BARS = ['PSSA1', 'PSSA2', 'PSSC1', 'PSSC2', 'PSSD1', 'PSSD2']
SRCCAL = REPO / 'mx_july_beam_qa' / 'calib' / 'srccal_energy_calib.json'


def load() -> dict:
    t = {}
    t['R'] = pd.read_csv(OUT / 'c1' / 'mip_per_bar.csv')
    t['V'] = pd.read_csv(OUT / 'c1' / 'calib_variants.csv')
    t['P'] = pd.read_csv(OUT / 'c1' / 'path_check.csv')
    t['spec'] = pd.read_parquet(OUT / 'c1' / 'mip_spectra.parquet')
    t['cal'] = json.loads((OUT / 'c1' / 'calib_plastic_e.json').read_text())
    t['LO'] = pd.read_csv(OUT / 'c4' / 'overall.csv')
    t['LS'] = pd.read_csv(OUT / 'c4' / 'mip_scale.csv')
    t['LP'] = pd.read_csv(OUT / 'c4' / 'profiles.csv')
    t['LM'] = pd.read_csv(OUT / 'c4' / 'maps.csv')
    t['PN'] = pd.read_csv(OUT / 'c4' / 'penetration.csv')
    t['liq'] = pd.read_parquet(OUT / 'c4' / 'liquid_tagged.parquet')
    t['MR'] = pd.read_csv(OUT / 'm2' / 'resolution.csv')
    t['MT'] = pd.read_csv(OUT / 'm2' / 'two_mip.csv')
    t['MS'] = json.loads((OUT / 'm2' / 'summary.json').read_text())
    t['msel'] = pd.read_parquet(OUT / 'm2' / 'dedx_selected.parquet')
    t['C2'] = pd.read_csv(OUT / 'c2' / 'response.csv')
    t['C3'] = json.loads((OUT / 'c3' / 'summary.json').read_text())
    t['J'] = json.loads(SRCCAL.read_text())['channels']
    return t


def bar_name(ch):
    return f'{ch[3]}{"L" if ch[4] == "1" else "R"}'


# --------------------------------------------------------------------------- #
def s_cover(D, t):
    Rc = t['R'][t['R']['sample'] == 'cosmic']
    lo, hi = Rc.ratio_line_477_699.min(), Rc.ratio_line_477_699.max()
    LS = t['LS']
    C = t['C3']['results']
    g_m1, g_e0 = C['M1']['smeared']['gain'], C['E0']['smeared']['gain']
    body = (sd.kicker('n_TOF X17 · calorimetry feasibility · 2026-10-08')
            + sd.p('The scintillator stack is a range telescope, and its energy adds almost nothing '
                   'against the IPC background', 60, sd.DINK, 600)
            + sd.p('What the plastic, the liquid and the MM charge can measure, tested on cosmics, '
                   'beam data and Geant4. Two calibration scales were wrong, and the hoped-for '
                   'soft-leg energy discriminant is dead.', 30, sd.DMUT)
            + sd.row(
                sd.bignum(f'{lo:.2f}–{hi:.2f}', 'what the source calibration reads for a muon',
                          sd.DRED, 'of its true 3.41 MeV deposit in 20 mm plastic. Replaced by a '
                          'muon-anchored scale.', tip='production keVee line / Bichsel MPV, per bar, run_149 cosmics'),
                sd.bignum(f'{LS.mpv_mv.min():.0f}–{LS.mpv_mv.max():.0f} mV', 'a muon in the liquid',
                          sd.DGREY, 'against a 16–18 mV readout threshold. The liquids are not '
                          'dead; their gain is ×5–7 too low.', tip='Landau MPV fitted with the ZS threshold as truncation'),
                sd.bignum(f'+{(g_e0 - 1) * 100:.0f}–{(g_m1 - 1) * 100:.0f} %', 'gain in X17 sensitivity',
                          sd.DBLUE, 'from adding the plastic energy to the opening angle. The plan '
                          'needed ≥ 10 %: killed.', tip='Asimov Z² ratio, E0 / M1 IPC, 10⁷-event pair sim'),
                gap=56))
    D.slide('cover', body, dark=True, short='Answer', notes=(
        'Plan: <code>ntof_calorimetry/PLAN.md</code> (with "Results so far"). Long report: '
        '<code>/media/dylan/data/x17/calorimetry/report.html</code>. Every number in this note is '
        'computed by <code>ntof_calorimetry/make_deck.py</code> from that analysis\'s own tables.'))


def s_setup(D, t):
    W, H = 1664, 380
    o = []
    layers = [('capsule', 'He-3 in Al', 90, sd.GREY, 'Capsule wall, He-3 at 500 bar, air: the first part of the ~1.2 MeV a soft electron loses before the plastic (Geant4, C2)'),
              ('MM', '30 mm drift', 150, sd.BLUE, 'Micromegas: 30 mm Ar drift gap + PCB. Tracks and (here) the raw charge for dE/dx'),
              ('wall', '3 mm PVT', 70, sd.GOLD, 'SiPM wall: 3 mm PVT, 4 groups x 2 ends. Part of the trigger'),
              ('plastic', '20 mm PVT', 170, sd.ORANGE, 'Two plastic bars, 200 x 300 x 20 mm (20 mm per the Geant4 geometry; the run_config text says 25), one PMT each. Trigger: wall AND plastic'),
              ('liquid', '18 mm LAB', 150, sd.GREEN, 'One LAB cell, 451 x 450 mm, 21.2 mm vessel, one PMT')]
    x = 140
    o.append(sd.T(60, 200, 'vertex', 22, sd.INK))
    o.append(f'<circle cx="60" cy="160" r="12" fill="{sd.RED}"/>')
    for name, sub, w, c, tp in layers:
        o.append(f'<rect x="{x}" y="70" width="{w}" height="180" rx="10" fill="{c}" fill-opacity="0.18" '
                 f'stroke="{c}" stroke-width="3"{sd.tipattr(tp)}/>')
        o.append(sd.T(x + w / 2, 150, name, 26, sd.INK, weight=600))
        o.append(sd.T(x + w / 2, 182, sub, 21, sd.MUT))
        x += w + 70
    # soft and hard lepton arrows
    o.append(sd.arrow(72, 120, 700, 120, sd.RED, 4))
    o.append(sd.T(400, 40, 'soft leg, 3.9–4.8 MeV (X17 at ≥ 140°): ~half stop before the plastic', 22, sd.RED, 'middle'))
    o.append(sd.arrow(72, 220, 1100, 220, sd.INK, 4))
    o.append(sd.T(600, 290, 'hard leg, 10–16 MeV: crosses everything, leaves a MIP-like ~4 MeV in the plastic',
                  21, sd.INK))
    diag = sd.svg(W, H, ''.join(o), 'one arm of the stack')
    fl = sd.flow([
        dict(label='C1 plastic scale', sub='is keVee right at 3–5 MeV?', color=sd.ORANGE,
             tip='Cosmic muons as a known 3.41 MeV deposit'),
        dict(label='C4 liquids', sub='dead, or below threshold?', color=sd.GREEN,
             tip='Muons certain to cross the cell'),
        dict(label='M1–M2 MM dE/dx', sub='can one gap flag 2 MIPs?', color=sd.BLUE, strike=True,
             tip='Killed: resolution ~ 40-47 %'),
        dict(label='C2 Geant4 response', sub='what reaches the plastic', color=sd.GREY,
             tip='Single e-/e+ 0.5-16 MeV, condor 4409644'),
        dict(label='C3 X17 vs IPC', sub='is the energy worth anything?', color=sd.RED, strike=True,
             tip='Killed: Z^2 gain 5-8 % < 10 %'),
    ], size=23)
    body = (sd.title('One arm, and the five questions asked of it',
                     'Outward from the capsule: tracker, trigger layers, liquid. Struck out = stopped by its kill criterion.')
            + diag + fl
            + sd.p('Hover the dotted words, boxes and data points for definitions, counts and sources.', 22, sd.MUT))
    D.slide('setup', body, short='Setup', notes=(
        '<p>The plan (<code>ntof_calorimetry/PLAN.md</code>) set a measurable kill criterion on every step '
        'before anything ran. X17 kinematics are exact: two-body decay, m = 16.9 MeV, so the softer lepton '
        'carries 3.9-4.8 MeV whenever the opening angle is at or above 140 deg. That is the range a 20 mm plastic '
        'should stop, and the hope was that X17 would show up as a line in (angle, soft-leg energy) where IPC is a '
        'continuum.</p><p>Layer thicknesses: MX17_Full_Geant <code>SimConfig.hh</code> (plastic corrected '
        '2026-07-20 from 2.5 to 2.0 cm; run_config descriptions still say 2.5 -- worth a caliper).</p>'))


def s_ladder(D, t):
    V = t['V'].set_index('ch')
    R = t['R'][t['R']['sample'] == 'cosmic'].set_index('ch')
    P = sd.Plot(1060, 700, x=(0.4, 4.6, 'log'), y=(0.68, 1.12), xlabel='energy, MeV(ee)',
                ylabel='measured mV / production calibration line')
    P.xticks([(0.477, 'Cs 0.48'), (0.699, 'Y 0.70'), (1.612, 'Y 1.61'), (3.41, 'muon 3.41')])
    P.yticks([(v, f'{v:.1f}') for v in (0.7, 0.8, 0.9, 1.0, 1.1)])
    P.hline(1.0, sd.MUT, label='the line itself')
    off = dict(zip(BARS, np.linspace(-0.035, 0.035, len(BARS))))
    for ch in BARS:
        a, b = V.loc[ch, 'line_477_699_a'], V.loc[ch, 'line_477_699_b']
        xs, ys, tips = [], [], []
        for e, pt in sorted(t['J'][ch]['points'].items(), key=lambda kv: float(kv[0])):
            e = float(e) / 1000
            pred = a * e * 1000 + b
            xs.append(e * (1 + off[ch]))
            ys.append(pt['mv'] / pred)
            tips.append(f'{bar_name(ch)} ({ch}) source edge {e:.3f} MeV: {pt["mv"]:.1f} mV, line predicts {pred:.1f}')
        r = R.loc[ch]
        pred = a * r.exp_mev * 1000 + b
        xs.append(r.exp_mev * (1 + off[ch]))
        ys.append(r.mpv / pred)
        tips.append(f'{bar_name(ch)} cosmic muon MPV {r.mpv:.1f} +- {r.mpv_err:.1f} mV (n = {int(r.n)}); '
                    f'line predicts {pred:.0f} mV for {r.exp_mev:.2f} MeV -> reads {r.ratio_line_477_699:.2f}')
        P.line(xs, ys, ARMC[ch[3]], w=2.5, r=8, tips=tips,
               marker='circle' if ch[4] == '1' else 'open')
    lo, hi = R.ratio_line_477_699.min(), R.ratio_line_477_699.max()
    side = sd.col(
        sd.callout(f'The production scale is a straight line through the two lowest source edges, 0.22 MeV apart, '
                   f'and it is used 5× beyond them. At the muon it reads <b>{lo:.2f}–{hi:.2f}</b> of the true deposit.',
                   sd.RED),
        sd.callout('The source\'s own 1.6 MeV edge falls below the line by the same 12–22 % on A and D. '
                   'That edge was treated as untrustworthy, but it was right.', sd.ORANGE),
        sd.callout('Fix: anchor on the muon. ' + sd.term('calib_plastic_e.json',
                   'per bar mV/MeVee = cosmic MPV / Bichsel Delta_p, 4-7 % incl. fit and theory; '
                   'c1/calib_plastic_e.json') + ' gives the scale at the energy that matters.', sd.GREEN),
        sd.legend([('A', sd.BLUE, 'dot'), ('C', sd.GREEN, 'dot'), ('D', sd.PURPLE, 'dot')], 22),
        sd.p('filled = L bar, open = R bar', 22, sd.MUT), w=540, gap=22)
    body = (sd.title(f'The plastic calibration reads a muon {(1 - hi) * 100:.0f}–{(1 - lo) * 100:.0f} % low',
                     'Every calibration point over what the production line predicts there; run_149 beam-off cosmics')
            + sd.row(P.svg('calibration ladder'), side, gap=48))
    D.slide('c1-scale', body, short='Plastic scale', foot=(
        'Muon deposit: Landau–Vavilov–Bichsel MPV in 20 mm PVT, 3.36–3.42 MeV for βγ 5–100; Geant4 gives 3.375. '
        'Arm B: no usable cosmic sample.'), notes=(
        '<p>Sample: run_149 beam-off cosmics put on the n_TOF clock (<code>clock_match.py</code>, n_TOF '
        '224678–87); the n_TOF PSS trees read around the matched time. Selection: one gated track, the predicted bar '
        'and both ends of the predicted wall group fired, &gt; 20 mm from the bar edges, &gt; 15 mm from the L/R gap.</p>'
        '<p>The expected muon deposit is the most probable loss (Bichsel thin-layer formula), not the mean (4.6 MeV). '
        'The full-sim muon run in 2 cm PVT (<code>MX17_Full_Geant/analysis/mip_2cm</code>) peaks at 3.375 MeV.</p>'
        '<p>If the bars are really 25 mm (run_config text), the expected deposit is 4.30 MeV and every ratio here '
        'drops by 21 % -- same direction, bigger offset.</p>'))


def s_spectra(D, t):
    S, R = t['spec'], t['R']
    panels = []
    for ch in BARS:
        arm, bar = ch[3], int(ch[4])
        x = S[(S['sample'] == 'cosmic') & (S.arm == arm) & (S.bar == bar)]
        r = R[(R.ch == ch) & (R['sample'] == 'cosmic')].iloc[0]
        edges = np.linspace(0, 2.6 * r.mpv, 33)
        n, _ = np.histogram(x.e_mv, edges)
        ymax = n.max() * 1.18
        P = sd.Plot(530, 340, x=(0, edges[-1]), y=(0, ymax), margin=(16, 20, 60, 70),
                    title=f'{bar_name(ch)}: {r.mpv:.0f} mV = {r.exp_mev:.2f} MeV')
        step = 100 if edges[-1] < 450 else 200
        P.xticks([(v, f'{v:.0f}') for v in np.arange(0, edges[-1], step)])
        P.yticks([(v, f'{v:.0f}') for v in np.linspace(0, ymax, 4)[:-1].round()])
        xs, ys = sd.step_xy(edges, n)
        P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], sd.INK, 2))
        for i in range(len(n)):
            P.raw(f'<rect x="{P.X(edges[i]):.1f}" y="{P.Y(ymax):.1f}" width="{P.X(edges[i + 1]) - P.X(edges[i]):.1f}" '
                  f'height="{P.Y(0) - P.Y(ymax):.1f}" fill="transparent"'
                  f'{sd.tipattr(f"{edges[i]:.0f}-{edges[i + 1]:.0f} mV: {n[i]} muons")}/>', back=True)
        sc = LD.XI_OVER_MPV_PVT20 * r.mpv
        g = np.linspace(0, edges[-1], 200)
        f = LD._langaus_grid(g, r.mpv - LD.LANDAU_PEAK * sc, sc, r.sigma)
        c = 0.5 * (edges[1:] + edges[:-1])
        k = n[c > r.mpv].sum() / max(np.interp(c[c > r.mpv], g, f).sum() * (edges[1] - edges[0]), 1e-12)
        P.line(list(g), list(k * f * (edges[1] - edges[0])), ARMC[arm], w=3, markers=False,
               tip=f'Landau(x)Gauss: MPV {r.mpv:.1f} +- {r.mpv_err:.1f} mV, sigma {r.sigma:.0f} mV')
        thr = PLAS_THR[arm] * float(x.cos.median())
        P.vline(thr, sd.ORANGE, tip=f'trigger threshold {PLAS_THR[arm]:.0f} mV x median cos = {thr:.0f} mV')
        panels.append(P.svg(ch))
    grid = (sd.row(*panels[0::2], gap=26) + sd.row(*panels[1::2], gap=26))
    body = (sd.title('Clean muon peaks on every bar, but A\'s trigger cuts right under its peak',
                     'Path-corrected amplitude, cosmic muons; curve = fit; orange = trigger threshold')
            + grid)
    D.slide('c1-spectra', body, short='Muon peaks', notes=(
        '<p>The hardware trigger needs one plastic bar above PLAS_THR (112–151 mV). On arm A that is 0.85–0.9 of '
        'the muon peak, so a plain fit to A\'s self-triggered spectrum is biased. The fit '
        '(<code>landau.fit_trunc</code>) truncates each event at its own threshold × cos θ, and only when this bar '
        'had to satisfy the trigger. The Landau width is held at its physics value (ξ/Δ<sub>p</sub> = 0.051), with '
        'a weak prior σ/MPV = 0.18 ± 0.05 taken from the bars whose threshold is far below their peak.</p>'
        '<p>Closure: raising every event\'s threshold to 0.9 and 1.0 of the peak moves the MPV by ≤ 6 % (A2 worst), '
        '≤ 2 % on C and D. Without the prior, the fit slides to a low MPV with a huge σ when the threshold is near '
        'the peak.</p>'))


def s_path(D, t):
    Pc = t['P']
    P = sd.Plot(820, 600, x=(20.5, 29), y=(1.0, 1.5), xlabel='path through the bar, mm',
                ylabel='muon peak / bar average (no cos correction)')
    P.xticks([(v, str(v)) for v in (21, 23, 25, 27, 29)])
    P.yticks([(v, f'{v:.1f}') for v in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)])
    g = np.linspace(20.5, 29, 40)
    P.line(list(g), [LD.mpv(x) / LD.mpv(20) for x in g], sd.MUT, w=2.5, dash='10 7', markers=False,
           tip='Bichsel MPV(path) / MPV(20 mm)')
    for arm, d in Pc.groupby('arm'):
        P.line(list(d.path_mm), list(d.mpv_rel), ARMC[arm], w=2, r=9,
               tips=[f'{arm}: path {a:.1f} mm, peak x{b:.3f} +- {e:.3f} (n = {n}); Bichsel x{x:.3f}'
                     for a, b, e, n, x in zip(d.path_mm, d.mpv_rel, d.err, d.n, d.exp_rel)])
    cal = t['cal']['bars']
    rows = [[bar_name(ch), f'{v["mv_per_mevee"]:.1f}', f'{v["rel_err"] * 100:.0f} %', f'{v["mv_per_mevee_srccal_line"]:.1f}',
             f'{v["resolution_sigma_over_mpv"] * 100:.0f} %', f'{v["trigger_threshold_mevee"]:.1f}']
            for ch, v in cal.items()]
    tab = sd.table(['bar', 'mV/MeV muon', '±', 'mV/MeV old', 'σ at muon', 'trigger, MeV'], rows, size=23)
    body = (sd.title('The response is linear up to ~5 MeV, so the muon scale holds where the soft leg lands',
                     'Muon peak against path length; the new scale per bar (c1/calib_plastic_e.json)')
            + sd.row(P.svg('path check'), sd.col(tab,
                     sd.p('The trigger itself already demands 2.1–2.9 MeV in a bar, at or above the '
                          + sd.term('²⁸Al β endpoint', 'Q_beta 4.64 MeV, beta endpoint 2.86 MeV: the activation background') +
                          '. ADC clipping starts beyond ~15 MeV.', 24), w=760, gap=24), gap=40))
    D.slide('c1-linear', body, short='Linearity', notes=(
        '<p>Steeper muons cross more plastic. The raw-amplitude peak (no cos correction), split into path terciles '
        'and normalised per bar, follows the Bichsel path dependence from 1.0 to 1.45 muon-equivalents, i.e. up to '
        '~5 MeV. That checks the cos correction and the linearity over exactly the soft-leg range.</p>'
        '<p><code>satuflag</code> is never set in this processing; the first repeated (clipped) amplitudes are at '
        '16–23 MeV on the muon scale. 2 V digitisers on a +950 mV baseline.</p>'))


def s_liquid(D, t):
    L, LS, LP = t['liq'], t['LS'].set_index('arm'), t['LP']
    edges = np.arange(0, 101, 4)
    P = sd.Plot(760, 600, x=(0, 100), y=(0, 0.27), xlabel='liquid amplitude, mV', ylabel='fraction of fired')
    P.xticks([(v, str(v)) for v in range(0, 101, 20)])
    P.yticks([(v, f'{v:.2f}') for v in (0, 0.05, 0.1, 0.15, 0.2, 0.25)])
    P.raw(f'<rect x="{P.X(16):.1f}" y="{P.Y(0.27):.1f}" width="{P.X(18) - P.X(16):.1f}" height="{P.Y(0) - P.Y(0.27):.1f}" '
          f'fill="{sd.ORANGE}" fill-opacity="0.3"{sd.tipattr("n_TOF zero-suppression threshold, 16-18 mV (DAQsettings)")}/>', back=True)
    P.text(19, 0.255, 'readout threshold', 21, sd.ORANGE)
    for arm in 'ACD':
        x = L[(L['sample'] == 'cosmic') & (L.arm == arm) & L.lf_on]
        n, _ = np.histogram(x.la, edges)
        f = n / max(n.sum(), 1)
        xs, ys = sd.step_xy(edges, f)
        P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], ARMC[arm], 3,
                      tip=f'{arm}: {int(n.sum())} fired; Landau MPV {LS.loc[arm, "mpv_mv"]:.1f} mV = '
                          f'{LS.loc[arm, "mv_per_mev_mip"]:.1f} mV/MeV'))
    src = float(LS.mv_per_mev_source.dropna().mean())
    P.text(58, 0.12, f'the source scale predicts ~{3.1 * src:.0f} mV', 22, sd.MUT)
    P.raw(sd.arrow(P.X(85), P.Y(0.11), P.X(99), P.Y(0.01), sd.MUT, 2))
    Q = sd.Plot(820, 600, x=(-225, 225), y=(0, 1), xlabel='u on the liquid face, mm (A, D: PMT at +u)',
                ylabel='muon efficiency')
    Q.xticks([(v, str(v)) for v in (-200, -100, 0, 100, 200)])
    Q.yticks([(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    Q.hline(0.5, sd.MUT, label='plan: usable if > 50 %')
    for arm in 'ACD':
        d = LP[(LP.arm == arm) & (LP.axis == 'u') & (LP.n >= 20)]
        Q.line(list(d.mid), list(d.eff), ARMC[arm], w=3, r=7,
               tips=[f'{arm}, u {a:.0f} mm: {e:.0%} +- {s:.0%} (n = {n}), median amp {m:.0f} mV'
                     for a, e, s, n, m in zip(d.mid, d.eff, d.err, d.n, d.amp_med)])
    for arm, (xx, yy) in {'A': (150, 0.92), 'D': (150, 0.6), 'C': (150, 0.17)}.items():
        Q.text(xx, yy, arm, 24, ARMC[arm], weight=600)
    body = (sd.title('The liquids are not dead: a muon sits just above the readout threshold',
                     'Cosmic muons confirmed through the plastic in front; left: amplitude, right: efficiency across the face')
            + sd.row(P.svg('liquid spectrum'), Q.svg('liquid efficiency'), gap=50))
    D.slide('c4-liquid', body, short='Liquids', foot=(
        'Usable as a hard-leg tag: A u ≥ 0 (65–86 %), D u ≥ 50 mm (50–70 %). C &lt; 30 % everywhere; B not measurable. '
        'Pulses under threshold were never recorded.'), notes=(
        '<p>A muon crossing the 18 mm LAB cell deposits ~2.6–3.2 MeV (Bichsel, path-corrected). The LIQ source '
        'calibration (46–48 mV/MeVee) predicts ~145 mV; the liquids record 20–28 mV, i.e. 6–9 mV/MeV. The '
        'source "edges" were not Compton edges: LIQA\'s Cs-137 and Y-88 edges sit at the same amplitude.</p>'
        '<p>So the n_TOF zero-suppression threshold (16–18 mV) is 1.8–2.8 MeV on the muon scale, and the '
        'efficiency is set by how much of the Landau clears it. On A and D the efficiency and the amplitude rise '
        'toward the PMT (light attenuation). C, with its PMT on top, loses efficiency toward the top (14 % → 2 %), '
        'which attenuation cannot do -- consistent with a bubble or under-fill at the top, for C only.</p>'
        '<p>Not a pulse-fit artefact: amp_0/amp = 1.03.</p>'))


def s_maps(D, t):
    M = t['LM']
    M = M[(M['sample'] == 'cosmic') & (M.n >= 8)]
    panels = []
    for arm in 'ACD':
        d = M[M.arm == arm]
        P = sd.Plot(470, 500, x=(-225, 225), y=(-225, 225), margin=(16, 16, 70, 80), title=f'liquid {arm}',
                    xlabel='u, mm', ylabel='v, mm' if arm == 'A' else '')
        P.xticks([(v, str(v)) for v in (-200, 0, 200)])
        P.yticks([(v, str(v)) for v in (-200, 0, 200)])
        cells = [(r.u - 25, r.u + 25, r.v - 25, r.v + 25, max(r.eff, 0),
                  f'{arm} cell ({r.u:.0f}, {r.v:.0f}) mm: {r.eff:.0%} +- {r.err:.0%}, n = {int(r.n)}')
                 for r in d.itertuples()]
        P.cells(cells, 0, 1, gap=1.5)
        P.rect(-225, 225, -225, 225, sd.RULE, 2)
        panels.append(P.svg(f'liquid {arm} map'))
    cb = sd.colorbar(150, 480, 0, 1, [(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)], 'muon eff.')
    body = (sd.title('Where each liquid still works', 'Muon efficiency per 50 mm cell, run_149 cosmics; blank = no muons through it')
            + sd.row(*panels, cb, gap=30, align='center'))
    D.slide('c4-maps', body, short='Liquid maps', notes=(
        'Cells with ≥ 8 confirmed muons. Net of accidentals using the same-width window before the trigger (its '
        'rate is ≤ 0.2 %). The cosmic sample only covers where muons cross both a chamber and the stack, hence the '
        'blank edges.'))


def s_through(D, t):
    PN = t['PN'].set_index('arm')
    rows = []
    for arm in 'AC':
        r = PN.loc[arm]
        rows.append((f'chamber {arm}', float(r.ratio), ARMC[arm],
                     f'{arm}: observed {r.observed:.0f} liquid hits, expected {r.expected:.1f} from the cosmic '
                     f'efficiency of each event\'s cell (n = {int(r.n)})',
                     f'±{r.ratio_err:.2f}'))
    rows.append(('cosmic muons', 1.0, sd.GREY, 'by construction: the expectation is the cosmic map', ''))
    bars = sd.hbars(rows, 1.0, width=900, h=44, label_w=260, fmt=lambda v: f'{v:.2f}')
    body = (sd.title('Most in-beam "through-goers" never reach the liquid behind the plastic',
                     'A–C track pairs in beam data, ≥ 10 ms after the flash: liquid hits observed / expected for penetrating muons')
            + bars
            + sd.row(sd.callout('If they were cosmic muons crossing both stacks, the liquid would fire as it does for '
                                'cosmics. It fires at 4–10 % of that.', sd.RED),
                     sd.callout('So they stop inside the stack: collinear pairs from a vertex off the beam axis, '
                                'not penetrating muons. The through-going background is not purely cosmic.', sd.BLUE),
                     gap=40))
    D.slide('through', body, short='Through-goers', foot=(
        'Checked: the beam liquid hits are in time (−30 to −10 ns in a −100/+60 ns window) and their amplitudes '
        'match cosmics (medians 26–30 mV), so neither timing nor gain explains it.'), notes=(
        '<p>A by-product with consequences beyond calorimetry. The in-beam A–C "through-goers" (one gated track in '
        'each, lines within 40 mm, joined line &gt; 60 mm from the beam axis) at ≥ 10 ms fire the predicted plastic '
        'bar 70–77 % of the time, but the liquid behind it only 0.10 ± 0.04 (A) and 0.04 (C) of the position-matched '
        'cosmic expectation. Before 10 ms the pairs are mostly flash junk (the plastic fires 22–33 %).</p>'
        '<p>Consequence for C1: they cannot calibrate the plastic, and the beam-period plastic peak that rises with '
        'time after the flash cannot be read as a gain change. Which particles they are (conversion pairs, Compton '
        'pairs, chance alignments) is open.</p>'))


def s_dedx(D, t):
    S, R, T = t['msel'], t['MR'], t['MT']
    rng = np.random.default_rng(3)
    m = float(R[(R.arm == 'A') & (R.estimator == 'q_whole')].mpv.iloc[0])
    x = S[(S.arm == 'A') & S.q_whole.notna()]
    v = x.q_whole.to_numpy() / m
    Q, pth = x.q_whole.to_numpy() * x.path_mm.to_numpy(), x.path_mm.to_numpy()
    i, j = rng.integers(0, len(v), (2, 20000))
    two = (Q[i] + Q[j] * pth[i] / pth[j]) / pth[i] / m
    edges = np.linspace(0, 5, 51)
    n1, _ = np.histogram(v, edges)
    n2, _ = np.histogram(two, edges)
    f1, f2 = n1 / n1.sum(), n2 / n2.sum()
    P = sd.Plot(1000, 640, x=(0, 5), y=(0, 0.085), xlabel='road charge per mm of path / most probable value',
                ylabel='fraction')
    P.xticks([(k, str(k)) for k in range(6)])
    P.yticks([(k, f'{k:.2f}') for k in (0, 0.02, 0.04, 0.06, 0.08)])
    for f, c, lab in ((f1, sd.BLUE, 'one muon'), (f2, sd.INK, 'two muons (best case)')):
        xs, ys = sd.step_xy(edges, f)
        P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], c, 3, tip=lab))
    cut = float(np.quantile(v, 0.9))
    e10 = float((two > cut).mean())
    P.vline(cut, sd.ORANGE, label='10 % of single muons above',
            tip=f'cut at {cut:.2f} x MPV keeps 10 % of one-muon tracks and {e10:.0%} of two-muon tracks')
    P.text(0.25, 0.075, 'one muon', 24, sd.BLUE, weight=600)
    P.text(2.3, 0.03, 'two muons', 24, sd.INK, weight=600)
    rA = R[(R.arm == 'A') & (R.estimator == 'q_whole')].iloc[0]
    rT = R[(R.arm == 'A') & (R.estimator == 'q_trunc_t')].iloc[0]
    side = sd.col(
        sd.bignum(f'{rA.fwhm_over_mpv:.2f}', 'FWHM / MPV of one 30 mm gap', sd.BLUE,
                  f'σ-equivalent {rA.sigma_eq:.0%}; truncated mean {rT.sigma_eq:.0%} -- it does not help', dark=False, size=64),
        sd.bignum(f'{e10:.0%}', 'of two-muon tracks flagged', sd.RED,
                  'at 10 % single-muon mis-tag, before any reconstruction or ZS loss', dark=False, size=64),
        sd.p('The plan\'s kill criterion (resolution ≳ 45 %) is met: no 2-MIP flag. The waveform overlay (M3) '
             'would only be worse, so it was not run.', 24), w=520, gap=28)
    body = (sd.title('One Micromegas gap cannot tell one minimum-ionising track from two',
                     f'Raw road charge on run_149 cosmics, chamber A ({len(x):,} tracks); two-muon = charges of random real pairs added')
            + sd.row(P.svg('dE/dx'), side, gap=50))
    D.slide('dedx', body, short='MM dE/dx', notes=(
        '<p>Estimator (<code>mm_charge.py</code>): the sum over the strips on each track\'s own corridor (from the '
        'waveform reco: p0 + depth × tan, depth −3 to 33 mm, ± 5 mm), all 20 samples, pedestal-subtracted. The '
        'common mode is taken from each 64-channel block\'s channels <i>outside</i> the road; the stock CNS would '
        'subtract part of a steep track\'s own charge. An off-road control road averages −0.06 % of the signal. '
        'Never the reco\'s q_sum/q_total (the NNLS runaway).</p>'
        '<p>Chamber C drifts at 28 µm/ns, so its 30 mm gap takes 1.1 µs and its deep charge falls off the 1.2 µs window '
        'on ~94 % of tracks. A plateau estimator (charge per sample inside the drift) covers it: σ-equivalent 56 %. '
        'The 40 mm gain map spreads 19 % (A) and 39 % (C) cell to cell; the charge is a fine gain monitor.</p>'
        '<p>A first depth profile suggested electron attachment in C (×3 fall with depth). It came from normalising '
        'each track on C\'s minority of in-window tracks; the unbiased all-track profile shows none.</p>'))


def s_c2(D, t):
    R = t['C2']
    R = R[(R.theta == 90) & (R.phi == 0) & (R.particle == 'e-')].sort_values('T')
    P = sd.Plot(800, 600, x=(0, 16), y=(0, 1), xlabel='electron energy at the capsule, MeV', ylabel='probability')
    P.xticks([(v, str(v)) for v in range(0, 17, 2)])
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    P.raw(f'<rect x="{P.X(3.9):.1f}" y="{P.Y(1):.1f}" width="{P.X(4.8) - P.X(3.9):.1f}" height="{P.Y(0) - P.Y(1):.1f}" '
          f'fill="{sd.RED}" fill-opacity="0.12"{sd.tipattr("X17 soft leg at opening angle >= 140 deg: 3.9-4.8 MeV")}/>', back=True)
    P.line(list(R['T']), list(R.p_plas), sd.ORANGE, w=3, r=6,
           tips=[f'{a:g} MeV: {b:.0%} put > 0.3 MeV in a plastic bar' for a, b in zip(R['T'], R.p_plas)])
    P.line(list(R['T']), list(R.p_liq), sd.GREEN, w=3, r=6,
           tips=[f'{a:g} MeV: {b:.0%} put > 1 MeV in the liquid' for a, b in zip(R['T'], R.p_liq)])
    P.text(9, 0.98, 'reaches the plastic', 22, sd.ORANGE, weight=600)
    P.text(10.4, 0.55, 'reaches the liquid', 22, sd.GREEN, weight=600)
    Q = sd.Plot(800, 600, x=(0, 16), y=(0, 7), xlabel='electron energy at the capsule, MeV',
                ylabel='MeV left in the plastic (median, 16–84 %)')
    Q.xticks([(v, str(v)) for v in range(0, 17, 2)])
    Q.yticks([(v, str(v)) for v in range(0, 8)])
    ok = R.plas_med.notna() & (R['T'] >= 2.5)
    d = R[ok]
    Q.band(list(d['T']), list(d.plas_q16), list(d.plas_q84), sd.ORANGE, 0.18)
    Q.line(list(d['T']), list(d.plas_med), sd.ORANGE, w=3, r=6,
           tips=[f'{a:g} MeV: median {b:.2f} MeV ({c:.2f}-{e:.2f}); energy lost before it: {m:.2f} MeV'
                 for a, b, c, e, m in zip(d['T'], d.plas_med, d.plas_q16, d.plas_q84, d.miss_med)])
    Q.line([1.2, 8.2], [0, 7], sd.MUT, w=2, dash='8 6', markers=False, tip='T - 1.2 MeV')
    Q.line([0, 7], [0, 7], sd.RULE, w=2, dash='4 6', markers=False, tip='deposit = T')
    Q.text(5.6, 6.5, 'T − 1.2 MeV', 21, sd.MUT)
    Q.text(10, 3.0, 'muon-like plateau', 21, sd.MUT)
    body = (sd.title('Half of the X17 soft legs stop before they reach the plastic',
                     'Geant4: single electrons from the capsule into one arm (production geometry, 20k per point)')
            + sd.row(P.svg('reach'), Q.svg('deposit'), gap=60))
    D.slide('c2', body, short='Geant4 response', foot=(
        'Condor 4409644, 68 jobs; e⁺ behave the same. A leg that does reach the plastic loses ~1.2 MeV first '
        '(σ ~0.3 MeV): capsule, chamber, wall, wrapping.'), notes=(
        '<p>Built with the same <code>mx17_full_sim</code> build that made the pair sim (nose-first capsule, '
        '2 cm plastic, 21.2 mm LS vessel). Points at θ = 90°, φ = 0 (normal to arm 0); φ = 20° agrees. '
        'Prompt deposits only (time &lt; 10⁸ ns; radioactive decay is on in the sim).</p>'
        '<p>The plan had assumed 0.5–1 MeV of upstream loss and a sharp reach threshold; both were optimistic. '
        'Legs that do reach the plastic below 6 MeV stop there (≥ 95 %), reading E ≈ T − 1.2 MeV.</p>'))


def s_c3(D, t):
    C = t['C3']['results']
    labs = [('true_soft_T', 'ideal: true soft-leg energy'),
            ('true_wall_plus_plastic', 'perfect wall + plastic'),
            ('true_deposit', 'true plastic deposit'),
            ('smeared', 'realistic plastic')]
    rows = []
    for k, lab in labs:
        for kind, c in (('M1', sd.BLUE), ('E0', sd.PURPLE)):
            g = C[kind][k]
            rows.append((f'{lab} · {kind}', g['gain'], c,
                         f'{lab}, IPC = pure {kind}: Z^2 gain {g["gain"]:.3f} (MC halves {g["gain_half_a"]:.3f} / {g["gain_half_b"]:.3f})'))
    vmax = max(r[1] for r in rows) * 1.05
    bars = sd.hbars(rows, vmax, width=520, h=30, label_w=400, fmt=lambda v: f'×{v:.2f}', size=22)
    f2 = C['M1']['frac_both_plastic_x17']
    side = sd.col(
        sd.callout('The information is real: knowing the soft-leg energy would multiply the X17 sensitivity '
                   'by 2.5–4.', sd.GREEN),
        sd.callout(f'The stack throws it away. Only {f2:.0%} of the wide-angle X17 pairs have both legs in the plastic. '
                   'With the real resolution the gain is 5–8 %, under the plan\'s 10 % line.', sd.RED),
        sd.callout('Lesson for the next build: measure the soft leg before ~1 MeV of material.', sd.BLUE),
        w=600, gap=22)
    body = (sd.title('The plastic energy buys 5–8 % over the opening angle alone: not worth it',
                     'Asimov Z² gain (angle × softer-leg energy over angle only), X17 vs IPC; 10⁷-event pair sim')
            + sd.row(sd.col(bars, sd.legend([('IPC = pure M1', sd.BLUE, 'box'), ('IPC = pure E0', sd.PURPLE, 'box')], 22),
                            gap=24), side, gap=40))
    D.slide('c3', body, short='X17 vs IPC', foot=(
        'IPC reweighted from the sim\'s 1/M ansatz to ipc_born M1 and E0 (the two thermal channels; their mix is '
        'unknown). Condor 4409645 reduced all 100 files.'), notes=(
        '<p>Event model: the two leptons point into two different arms and both drift gaps see charge; the trigger '
        'is wall ≥ 0.3 MeV AND plastic ≥ 2.5 MeV in some arm; plastic resolution 0.18√(E·3.41 MeV) (the measured '
        'muon width); angle smeared by 4°; θ ≥ 100°. Figure of merit: Z² = Σ s²/b in (θ, E<sub>low</sub>) over Σ s²/b '
        'in θ, which does not depend on normalisation when S ≪ B.</p>'
        '<p>A both-legs plastic threshold keeps X17 and IPC at the same rate (~20 % at 0.5 MeV) and costs 80 % of '
        'the signal; the low-energy backgrounds (²⁸Al β, capture Comptons) are already cut on the triggering arm '
        'by its 2.1–2.9 MeV threshold.</p>'
        '<p>Gotcha met on the way: a fine 2D grid on ~30k effective IPC MC events made the <i>smeared</i> energy '
        'look worth more than the true one. Coarse bins, background bins with &lt; 10 MC events pooled, and the two '
        'MC halves agree to ± 0.01.</p>'))


def s_end(D, t):
    items = [
        ('Plastic bar thickness', '20 mm (Geant4) or 25 mm (run_config text)? 25 mm shifts the muon scale by 21 %. A caliper settles it.'),
        ('Plastic gain after the flash', 'The beam peak rises ~10 % from 10 to 80 ms, but the beam sample does not penetrate. Needs an in-beam energy reference (H-capture Compton edge?).'),
        ('What the beam through-goers are', 'Non-penetrating and collinear: conversion pairs, Compton pairs or chance alignments -- open, and it matters for the background.'),
        ('Liquid C', 'Top-side efficiency loss fits a bubble; a dead photocathode region would look the same.'),
        ('Arm B', 'No usable cosmic sample for either the plastic or the liquid.'),
        ('Below the muon', 'Linearity between 0.7 and 3.4 MeV is only constrained by source edges that disagree with each other.'),
    ]
    cards = ''.join(sd.card(sd.p(a, 28, sd.DINK, 600) + sd.p(b, 23, sd.DMUT), bg='#1f2533') for a, b in items[:3])
    cards2 = ''.join(sd.card(sd.p(a, 28, sd.DINK, 600) + sd.p(b, 23, sd.DMUT), bg='#1f2533') for a, b in items[3:])
    body = (sd.kicker('What this does not rule out')
            + sd.p('The calorimetry line ends here for this dataset: no invariant mass, no soft-leg line. '
                   'What remains are corrected scales, a liquid hard-leg tag on half of A and D, and a design lesson.',
                   40, sd.DINK, 600)
            + f'<div style="display:flex;gap:28px">{cards}</div>'
            + f'<div style="display:flex;gap:28px">{cards2}</div>')
    D.slide('end', body, dark=True, short='Open', notes=(
        'Code: <code>nTof_x17/ntof_calorimetry/</code> on branch beam-off-cosmics. Run order in its README. '
        'Condor scripts: <code>ntof_calorimetry/condor/</code>.'))


def main() -> int:
    t = load()
    D = sd.Deck('Calorimetry feasibility: n_TOF X17',
                'What energy the scintillator stack and the Micromegas charge can measure, and whether it helps against IPC.')
    for f in (s_cover, s_setup, s_ladder, s_spectra, s_path, s_liquid, s_maps, s_through, s_dedx, s_c2, s_c3, s_end):
        f(D, t)
    out = OUT / 'deck' / 'calorimetry.html'
    D.write(out, note_meta=dict(
        title='Calorimetry feasibility: what the stack can measure',
        summary='Plastic scale 3-24 % low at the muon (fixed), liquids threshold-limited, MM dE/dx and the '
                'soft-leg energy discriminant both killed (+5-8 % against IPC).',
        tags='n_TOF, X17, scintillators, calibration, Geant4',
        date=dt.date.today().isoformat()))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
