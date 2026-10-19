#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_wp_deck.py -- the wall + plastic calorimetry check (wall_plastic.py) as a
slide note: calibrate each layer on cosmic muons, then test both on the beam's
electrons against Geant4.  Built only from wall_plastic's tables.

    PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.wall_plastic
    PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_wp_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        /media/dylan/data/x17/calorimetry/wp/deck/wall-plastic.html --slug scint-calorimetry --force --deploy
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
from ntof_calorimetry import wall_plastic as WPm  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402

WP = OUT / 'wp'
ARMC = {'A': sd.BLUE, 'C': sd.ORANGE, 'D': sd.PURPLE}
G4C = sd.INK


def load():
    t = {}
    for n in ('wall_mip', 'wall_along_bar', 'wall_path', 'spectra', 'wall_vs_plastic', 'scale_fits', 'anchors',
              'rate_profile'):
        t[n] = pd.read_csv(WP / f'{n}.csv')
    t['S'] = json.loads((WP / 'summary.json').read_text())
    t['cos'] = pd.read_parquet(WP / 'wall_cosmic.parquet')
    T = pd.read_parquet(WP / 'beam_data.parquet', columns=['run', 'arm', 'e_wall', 'e_plas'])
    t['runs'] = (T.groupby(['arm', 'run']).agg(n=('e_wall', 'size'), w=('e_wall', 'median'),
                                               p=('e_plas', 'median')).reset_index())
    t['pc'] = WPm.plastic_cal()
    return t


def fitrow(t, arm, var, subset=None):
    F = t['scale_fits']
    m = (F.arm == arm) & (F['var'] == var)
    m &= F.subset.isna() if subset is None else (F.subset == subset)
    return F[m].iloc[0]


LATE = '>= 40 ms'


def bars(t):
    """[(name, fit row)] per plastic bar on the rate-free plateau."""
    return [(f'{a}{b}', fitrow(t, a, 'e_plas', f'bar {b}, {LATE}')) for a in 'AC' for b in (1, 2)]


# --------------------------------------------------------------------------- #
def s_cover(D, t):
    B = bars(t)
    good = [f for n, f in B if abs(f.scale - 1) < 0.08]
    odd = [(n, f) for n, f in B if abs(f.scale - 1) >= 0.08]
    wA, wC = fitrow(t, 'A', 'e_wall_fit', LATE), fitrow(t, 'C', 'e_wall_fit', LATE)
    R = t['runs']
    R = R[R.n > 5000]
    rw = max(R[R.arm == a].w.std() / R[R.arm == a].w.mean() for a in 'AC')
    rp = max(R[R.arm == a].p.std() / R[R.arm == a].p.mean() for a in 'AC')
    rr = t['rate_profile']
    sag = []
    for (a, b), g in rr.groupby(['arm', 'bar']):
        sag.append(1 - g.plas_med.iloc[0] / g[g.lo >= 40].plas_med.mean())
    odd_txt = ', '.join(f'{n} at {f.scale:.2f}' for n, f in odd) or 'none'
    body = (sd.kicker(f'n_TOF X17 · scintillator calorimetry · {dt.date.today():%-d %b %Y}')
            + sd.title('<span style="color:#eef0f3">Yes, with one odd channel: calibrated only on cosmic muons, '
                       'the wall and the plastic measure beam electrons at Geant4\'s scale to ~5 %</span>')
            + sd.row(
                sd.bignum(f'{len(good)} of {len(B)}', 'plastic bars within '
                          f'{100 * max(abs(f.scale - 1) for f in good):.0f} % of Geant4', sd.DGREEN,
                          f'3–8 MeV electrons, rate-free plateau, shapes agree. Odd one: {odd_txt}.',
                          tip='each bar against its geometrically matched Geant4 bar, >= 40 ms after the flash'),
                sd.bignum(f'{wA.scale:.2f} / {wC.scale:.2f}', 'wall (A / C) relative to Geant4', sd.DBLUE,
                          'the same ~5 % in both arms and at all times; shapes agree with the wall\'s own '
                          'smearing fitted'),
                sd.bignum(f'{100 * rp:.1f} / {100 * rw:.1f} %', 'run-to-run spread, plastic / wall', sd.GOLD,
                          '25 production runs, one calibration, nothing corrected'),
                sd.bignum(f'−{100 * min(sag):.0f} to −{100 * max(sag):.0f} %', 'plastic gain just after the flash',
                          sd.DRED, 'at 12–15 ms vs the plateau after ~40 ms: the PMTs sag at high rate; the wall '
                                   'does not', size=60),
                gap=40))
    notes = """
<p>Question (Dylan, 14 Oct): as a proof of concept for LNL or another facility, does the scintillator stack measure
energy? Just the SiPM wall and the plastic. Answer it with an internal cross-check, not with the n_TOF physics.</p>
<p>The test: calibrate each layer once, on cosmic muons (a MIP in 3 mm and in 20 mm of PVT, run_149, beam off),
and then predict a different population — the beam's 1–7 MeV Compton and pair electrons — with no further tuning.
Geant4 provides the prediction for both layers jointly. If the calibrations transfer, Geant4 needs no rescaling.</p>
<p>Data: every gated single MM track ≥ 12 ms after the flash whose own arm fired the trigger, chambers A and C,
production runs (all after the 23 July DREAM change). Geant4: the 10⁹-neutron thermal campaign (97 of 100 files),
re-reduced for total scintillator deposits per strip and bar (<code>MX17_Full_Geant/scripts/stack_edep.py</code>,
condor 4453978), and 1 GeV muon guns for the anchors (condor 4454427).</p>
<p>Verdict in one line: the method works (scale transfers from muons to stopping electrons, linear, stable); the
problems it found are one channel (C1) and a rate effect on the PMTs, both identifiable and both things a
facility set-up should monitor.</p>"""
    D.slide('cover', body, notes, dark=True, short='Answer')


def s_setup(D, t):
    W, H = 1664, 330
    o = []
    layers = [('MM', 'track, angle', 160, sd.GREY, 'Micromegas: the track gives the path through the layers'),
              ('wall', '3 mm PVT · SiPM', 80, sd.GOLD, 'SiPM wall: 4 groups x 2 ends; energy = sum of the two ends'),
              ('plastic', '20 mm PVT · PMT', 190, sd.ORANGE, 'two plastic bars, one PMT each; energy = the larger bar')]
    x = 300
    o.append(f'<circle cx="120" cy="165" r="14" fill="{sd.RED}"/>')
    o.append(sd.T(120, 210, 'capsule', 21, sd.MUT))
    for name, sub, w, c, tp in layers:
        o.append(f'<rect x="{x}" y="60" width="{w}" height="210" rx="10" fill="{c}" fill-opacity="0.18" '
                 f'stroke="{c}" stroke-width="3"{sd.tipattr(tp)}/>')
        o.append(sd.T(x + w / 2, 160, name, 26, sd.INK, weight=600))
        o.append(sd.T(x + w / 2, 192, sub, 19, sd.MUT))
        x += w + 90
    o.append(sd.arrow(1520, 40, 360, 300, sd.GREEN, 4))
    o.append(sd.T(1560, 30, 'cosmic μ', 22, sd.GREEN, 'end', 600))
    o.append(sd.T(1540, 320, 'calibrate: MIP peak in each layer', 21, sd.GREEN, 'end'))
    o.append(sd.arrow(134, 165, 820, 150, sd.RED, 4))
    o.append(sd.T(470, 120, 'beam electron, 1–7 MeV', 22, sd.RED, 'middle', 600))
    o.append(sd.T(470, 300, 'test: both energies vs Geant4, nothing refitted', 21, sd.RED, 'middle'))
    fl = sd.flow([
        dict(label='1 · wall MIP', sub='new: mV → MeV for the SiPM wall', color=sd.GOLD,
             tip='run_149 cosmics, truncated Landau x Gauss fit'),
        dict(label='2 · plastic MIP', sub='from 8 Oct (C1), unchanged', color=sd.ORANGE,
             tip='c1/calib_plastic_e.json'),
        dict(label='3 · stability', sub='the same electrons, run by run', color=sd.GREY),
        dict(label='4 · beam vs Geant4', sub='plastic, wall, wall vs plastic', color=sd.RED,
             tip='Geant4 totals per strip/bar, the data trigger in MeV, the measured resolutions'),
    ], size=23)
    body = (sd.title('The test: calibrate on muons, predict electrons',
                     'one arm, outward from the capsule; the liquid behind the plastic is left out here')
            + sd.svg(W, H, ''.join(o), 'the calibration and the test') + fl
            + sd.p('Hover the dotted words, boxes and data points for definitions, counts and sources.', 22, sd.MUT))
    notes = """
<p><b>Why this is a real test.</b> A muon is minimum-ionising and crosses both layers. The beam electrons are
different: 1–7 MeV, mostly stopping in the plastic, entering at an angle, often with their own delta rays and
bremsstrahlung in the same counter. If either layer's light output were non-linear, position-dependent or
mis-calibrated, the muon-calibrated energies would disagree with Geant4's prediction for the electrons.</p>
<p><b>Anchoring.</b> The fitter reports the Landau component's peak with the Landau width fixed, and that number
moves with the Gaussian smearing (+2–3 % on the wall from Geant4's bare width to the data's). So each counter's MeV
anchor is Geant4's 1 GeV muons, smeared until the same fitter returns that counter's measured σ/MPV, and then
fitted: wall {aw}; plastic {ap}. Unsmeared Geant4 muons give 0.462 / 3.398 MeV, i.e. Bichsel's values; the 0.475 /
3.349 of <code>analysis/mip_2cm</code> are another definition.</p>
<p><b>Geant4 selection</b>, matched to the data: one lepton crossing ≥ 20 mm of the arm's gap, the arm's own
trigger (wall group sum ≥ its threshold AND a plastic bar ≥ its threshold) applied to the smeared deposits in
MeV, using the thresholds measured in mV and converted with the same calibrations.</p>"""
    A = t['anchors']
    aw = ', '.join(f'{r.ch} {r.anchor:.3f}' for r in A[A.layer == 'wall'].itertuples())
    ap = ', '.join(f'{r.ch[3:]} {r.anchor:.3f}' for r in A[A.layer == 'plastic'].itertuples())
    notes = notes.replace('{aw}', aw + ' MeV').replace('{ap}', ap + ' MeV')
    D.slide('setup', body, notes, short='The test')


def s_wall(D, t):
    C, F = t['cos'], t['wall_mip']
    A = t['anchors']
    aw = A[A.layer == 'wall'].set_index('ch').anchor.to_dict()
    Fa = F[F.grp == -1].set_index('arm')
    edges = np.linspace(0, 1.6, 41)
    P = sd.Plot(820, 560, x=(0, 1.6), y=(0, 0.16), xlabel='wall energy × cos θ (MeV, muon-calibrated)',
                ylabel='fraction per 40 keV', title='cosmic muons, 3 mm wall')
    P.xticks([(v, f'{v:.1f}') for v in (0, 0.4, 0.8, 1.2, 1.6)])
    P.yticks([(v, f'{v:.2f}') for v in (0, 0.04, 0.08, 0.12, 0.16)])
    for arm in ('A', 'C', 'D'):
        x = C[C.arm == arm]
        e = x.s_mv / (Fa.loc[arm, 'mpv'] / aw[arm])
        n, _ = np.histogram(e, edges)
        f = n / n.sum()
        xs, ys = sd.step_xy(list(edges), list(f))
        P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], ARMC[arm], 3,
                      tip=f'{arm}: {len(x)} muons; MPV {Fa.loc[arm, "mpv"]:.1f} ± {Fa.loc[arm, "mpv_err"]:.1f} mV '
                          f'= {Fa.loc[arm, "mpv"] / aw[arm]:.1f} mV/MeV; σ_gauss/MPV {Fa.loc[arm, "sigma"] / Fa.loc[arm, "mpv"]:.2f}'))
    am = float(np.mean(list(aw.values())))
    P.vline(am, sd.MUT, label='Geant4 MIP', tip=f'Geant4 muons in the 3 mm wall, fitted as the data: {am:.3f} MeV')
    Pa = t['wall_path']
    Q = sd.Plot(760, 560, x=(3.0, 4.4), y=(1.0, 1.5), xlabel='path through the wall (mm)',
                ylabel='MPV / MPV at all angles', title='linear in path, as Bichsel predicts')
    Q.xticks([(v, f'{v:.1f}') for v in (3.0, 3.4, 3.8, 4.2)])
    Q.yticks([(v, f'{v:.1f}') for v in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)])
    xx = np.linspace(3.05, 4.35, 30)
    Q.line(list(xx), [LD.mpv(v, 30) / LD.mpv(3.0, 30) for v in xx], sd.MUT, w=2.5, dash='8 6', markers=False,
           tip='Bichsel most probable loss vs thickness, normalised at 3 mm')
    for arm in ('A', 'C', 'D'):
        g = Pa[Pa.arm == arm]
        Q.points(list(g.path_mm), list(g.mpv_rel), ARMC[arm], r=9,
                 tips=[f'{arm} path {p:.2f} mm: {m:.3f} (Bichsel {e:.3f}), n = {n}' for p, m, e, n in
                       zip(g.path_mm, g.mpv_rel, g.exp_rel, g.n)])
    grp = F[F.grp >= 0]
    spread = {a: grp[grp.arm == a].mpv.std() / grp[grp.arm == a].mpv.mean() for a in grp.arm.unique()}
    leg = sd.legend([('A', sd.BLUE), ('C', sd.ORANGE), ('D', sd.PURPLE)], size=22)
    body = (sd.title('The SiPM wall has a clean muon peak, linear in path and the same in every group',
                     f'run_149 beam-off cosmics, single track inside a wall group, both ends fired · sum of the two '
                     f'ends × cos θ')
            + sd.row(sd.col(P.svg('wall MIP'), leg, gap=4), Q.svg('path'), gap=40)
            + sd.p(f'Group-to-group spread of the MIP peak: A {100 * spread["A"]:.0f} %, C {100 * spread["C"]:.0f} %, '
                   f'D {100 * spread.get("D", np.nan):.0f} %. Trigger threshold on this scale: '
                   f'{WPm.WALL_THR["A"] / (Fa.loc["A", "mpv"] / aw["A"]):.2f} (A) – '
                   f'{WPm.WALL_THR["C"] / (Fa.loc["C", "mpv"] / aw["C"]):.2f} (C) MeV, ~0.6 MIP.', 23, sd.MUT))
    rows = ''.join(f'<tr><td>{r.arm}</td><td>{"all" if r.grp < 0 else int(r.grp)}</td><td>{int(r.n_sel)}</td>'
                   f'<td>{r.mpv:.1f} ± {r.mpv_err:.1f}</td><td>{r.sigma / r.mpv:.2f}</td></tr>' for r in F.itertuples())
    prof = t['wall_along_bar']
    notes = f"""
<p><b>Fit.</b> Unbinned Landau ⊗ Gauss with a per-event lower truncation where the arm's own trigger needed the
wall (<code>landau.fit_trunc</code>), the Landau width fixed at ξ/Δp = {WPm.XI_OVER_MPV_W:.3f} for 3 mm PVT, the
Gaussian free (photostatistics, the light-collection U-shape, path spread). Thresholds sit at ~0.55 of the peak, so
the truncation barely matters (truncated fraction &lt; 3 %).</p>
<table><tr><th>arm</th><th>group</th><th>n</th><th>MPV (mV)</th><th>σ/MPV</th></tr>{rows}</table>
<p><b>Along the bars</b> each end's response changes by ±25 % (attenuation), but their sum varies by
{100 * (prof.sum_rel.max() - prof.sum_rel.min()) / 2:.0f} % (half-range) and is highest near the bar ends — a
mild U-shape that the sum keeps in the resolution. The geometric mean would cancel attenuation exactly; the sum
is used because it is what the hardware discriminates.</p>
<p>Arm B has too few cosmic tracks (42) and is left out, as in C1.</p>"""
    D.slide('wall', body, notes, short='Wall MIP')


def s_stability(D, t):
    R = t['runs']
    R = R[R.n > 5000].copy()
    R['rn'] = R.run.str.replace('run_', '').astype(int)
    P = sd.Plot(1500, 520, x=(84, 164), y=(0.96, 1.04), xlabel='run', ylabel='electron median / campaign mean')
    P.xticks([(v, str(v)) for v in range(85, 165, 10)])
    P.yticks([(v, f'{v:.2f}') for v in (0.96, 0.98, 1.0, 1.02, 1.04)])
    P.band([84, 164], [0.99, 0.99], [1.01, 1.01], sd.GREY, 0.10, tip='±1 %')
    for arm, dx in (('A', -0.3), ('C', 0.3)):
        g = R[R.arm == arm].sort_values('rn')
        for var, dash, lab in (('p', None, 'plastic'), ('w', '6 6', 'wall')):
            v = g[var] / g[var].mean()
            P.line(list(g.rn + dx), list(v), sd.BLUE if arm == 'A' else sd.ORANGE, w=2.5, dash=dash, r=6,
                   tips=[f'{arm} {lab} {r}: median {m:.3f} MeV ({x:.3f} of mean), n = {n:,}' for r, m, x, n in
                         zip(g.run, g[var], v, g.n)])
    leg = sd.legend([('A plastic', sd.BLUE), ('A wall', sd.BLUE, 'dash'), ('C plastic', sd.ORANGE),
                     ('C wall', sd.ORANGE, 'dash')], size=22)
    body = (sd.title('Both layers hold their scale to under 1 % across the whole production campaign',
                     'median energy of the self-triggered beam electrons, per run, with ONE calibration (run_149, end '
                     'of campaign) · runs with > 5,000 tracks')
            + P.svg('stability') + leg)
    notes = """
<p>Nothing is corrected run by run: the cosmic calibration from run_149 (beam off, after the last beam run) is
applied to every run. The plastic median moves by 0.5–0.6 % rms and the wall by 0.8 % rms over 25 runs from
run_86 (23 July onwards) to run_162. Part of that is the beam itself (population), so the gain stability is at
least this good.</p>
<p>All these runs are on the noisy side of the 23 July DREAM change (CLAUDE.md), as is the calibration run.</p>"""
    D.slide('stability', body, notes, short='Stability')


def _spec_panel(t, arm, var, xr, xt, w, h, title, xlabel, fit_lo):
    Sp = t['spectra']
    g = Sp[(Sp.arm == arm) & (Sp['var'] == var)]
    c = 0.5 * (g.lo + g.hi)
    m = (c >= fit_lo) & (c <= xr[1])
    d = g.data / g.data[m].sum()
    s = g.g4 / g.g4[m].sum()
    ymax = float(max(d[m].max(), s[m].max()) * 1.25)
    P = sd.Plot(w, h, x=xr, y=(0, ymax), xlabel=xlabel, ylabel='fraction per bin', title=title)
    P.xticks(xt)
    step = 0.05 if ymax < 0.25 else 0.1
    P.yticks([(v, f'{v:.2f}') for v in np.arange(0, ymax, step)])
    gi = g[(g.lo >= xr[0]) & (g.hi <= xr[1])]
    si = gi.g4 / g.g4[m].sum()
    xs, ys = sd.step_xy(list(gi.lo) + [float(gi.hi.iloc[-1])], list(si))
    P.raw(sd.poly([P.X(a) for a in xs], [P.Y(min(b, ymax)) for b in ys], G4C, 3,
                  tip=f'Geant4, {int(g.g4.sum()):,} events'))
    err = np.sqrt(g.g4) / g.g4[m].sum()
    for x, y, e in zip(c[m], s[m], err[m]):
        P.fore.append(sd.line(P.X(x), P.Y(max(y - e, 0)), P.X(x), P.Y(min(y + e, ymax)), G4C, 1.5))
    P.points(list(c[m]), list(d[m]), ARMC[arm], r=6,
             tips=[f'{arm} {lo:.2f}-{hi:.2f} MeV: data {a:,}, Geant4 {b}' for lo, hi, a, b in
                   zip(g.lo[m], g.hi[m], g.data[m], g.g4[m])])
    return P


def s_plastic(D, t):
    T = pd.read_parquet(WP / 'beam_data.parquet', columns=['arm', 'bar', 'e_plas', 'p_sat', 't_since_flash_ns'])
    S = pd.read_parquet(WP / 'g4_sample.parquet', columns=['armL', 'pL', 'pR', 'e_plas'])
    edges = np.arange(2.0, 9.01, 0.25)
    c = 0.5 * (edges[1:] + edges[:-1])
    win = (c >= WPm.EP_FIT[0]) & (c < 9.0)
    panels = []
    for arm in 'AC':
        row = []
        for b in (1, 2):
            d = T[(T.arm == arm) & (T.bar == b) & ~T.p_sat & (T.t_since_flash_ns >= 40e6)].e_plas
            s_ = S[S.armL == arm]
            isL = WPm.G4_BAR_IS_L[(arm, b)]
            g = s_[(s_.pL >= s_.pR) == isL].e_plas
            nd, _ = np.histogram(d, edges)
            ng, _ = np.histogram(g, edges)
            fd, fg = nd / nd[win].sum(), ng / ng[win].sum()
            ymax = 0.16
            f = fitrow(t, arm, 'e_plas', f'bar {b}, {LATE}')
            P = sd.Plot(800, 330, x=(2, 9), y=(0, ymax), xlabel='plastic energy (MeV)' if arm == 'C' else '',
                        ylabel='fraction' if b == 1 else '', margin=(16, 20, 70 if arm == 'C' else 40, 90),
                        title=f'{arm}{b} vs Geant4 {"L" if isL else "R"} · scale {f.scale:.3f} ± {f.err:.3f}')
            P.xticks([(v, str(v)) for v in range(2, 10)])
            P.yticks([(v, f'{v:.2f}') for v in (0, 0.05, 0.1, 0.15)])
            xs, ys = sd.step_xy(list(edges), list(np.minimum(fg, ymax)))
            P.raw(sd.poly([P.X(x) for x in xs], [P.Y(y) for y in ys], G4C, 3,
                          tip=f'Geant4 bar {"L" if isL else "R"}, {len(g)} events, NOT rescaled'))
            P.points(list(c[c >= 2.5]), list(fd[c >= 2.5]), ARMC[arm], r=5,
                     tips=[f'{arm}{b} {x - 0.125:.2f}-{x + 0.125:.2f} MeV: {n:,} tracks' for x, n in
                           zip(c[c >= 2.5], nd[c >= 2.5])])
            row.append(P.svg(f'plastic {arm}{b}'))
        panels.append(sd.row(*row, gap=40))
    leg = sd.legend([('data, ≥ 40 ms (A blue, C orange)', sd.BLUE, 'dot'),
                     ('Geant4, matched bar, same trigger and resolution, not rescaled', G4C)], size=21)
    body = (sd.title('Each plastic bar\'s electron spectrum against Geant4, on the muon calibration alone',
                     'self-triggered single tracks on the rate-free plateau (≥ 40 ms) · shapes normalised over 3–9 MeV')
            + sd.col(*panels, gap=6) + leg)
    fA, fC = fitrow(t, 'A', 'e_plas'), fitrow(t, 'C', 'e_plas')
    notes = f"""
<p><b>What is plotted.</b> Data: the larger of the two bars, mV → MeV with that bar's cosmic MIP (C1) and the
matched Geant4 anchor. Geant4: the TOTAL prompt deposit in the geometrically matched bar (the lepton, its delta rays,
its bremsstrahlung and anything else of the event in that bar), smeared with the bar's measured σ/MPV × √(E·MIP),
the arm's trigger applied in MeV. Geant4 is drawn at scale 1: no refit.</p>
<p><b>The scales</b> (Geant4 × s matches the data's mean over 3–8 MeV; bootstrap errors): see the next slide. Pooling
both bars of an arm and all times: A {fA.scale:.3f} ± {fA.err:.3f}, C {fC.scale:.3f} ± {fC.err:.3f}.</p>
<p><b>What it tests.</b> Most of these electrons STOP in the plastic (below ~6 MeV), so their deposit is their
energy, not a dE/dx. Agreement says the light output is linear from the muon's 3.4 MeV (path-normalised) to the
stopping electrons' 3–8 MeV, and the scale is the same for a through-going and a stopping particle.</p>
<p><b>Plastic thickness.</b> A 25 mm bar (the run-config text) would make a muon deposit 4.30 MeV, not 3.40: the
data energies would all rise by ×1.27 against a 20 mm Geant4, which three bars at 1.02–1.05 do not allow. The 20 mm
of the Geant4 geometry is right (a caliper would close it).</p>
<p><b>Not included:</b> Birks quenching (Geant4 scores energy, the bars make light). For electrons that stop, the
dense end of the track is quenched relative to a MIP, which would make data read LOW by a few %; the data read high,
so quenching is not what is being seen.</p>"""
    D.slide('plastic', body, notes, short='Plastic vs G4')


def s_wallbeam(D, t):
    panels = []
    for arm in ('A', 'C'):
        f = fitrow(t, arm, 'e_wall_fit')
        P = _spec_panel(t, arm, 'e_wall_fit', (0.2, 1.6), [(v, f'{v:.1f}') for v in (0.2, 0.6, 1.0, 1.4)], 790, 540,
                        f'chamber {arm} · scale {f.scale:.3f} ± {f.err:.3f} · KS p {f.ks_p_fit:.2f}',
                        'wall energy (MeV, muon-calibrated)', 0.2)
        panels.append(P.svg(f'wall {arm}'))
    fA, fC = fitrow(t, 'A', 'e_wall_fit'), fitrow(t, 'C', 'e_wall_fit')
    gA, gC = fitrow(t, 'A', 'e_wall'), fitrow(t, 'C', 'e_wall')
    sc = 0.5 * (fA.scale + fC.scale)
    leg = sd.legend([('data', sd.BLUE, 'dot'), ('Geant4 × scale, with the fitted smearing', G4C)], size=22)
    title = (f'The wall reads beam electrons at {sc:.2f} of Geant4 in both arms; the shape agrees'
             if abs(sc - 1) > 0.02 else 'The wall reads beam electrons at Geant4\'s scale; the shape agrees')
    body = (sd.title(title, 'same events, plastic ≥ 3 MeV · the wall group with the largest sum · Geant4 smeared by '
                            f'{100 * fA.r_fit:.0f} % (A) / {100 * fC.r_fit:.0f} % (C) at the MIP, fitted · '
                            'shown × the fitted scale')
            + sd.row(*panels, gap=40) + leg)
    notes = f"""
<p><b>The scale.</b> A {fA.scale:.3f} ± {fA.err:.3f}, C {fC.scale:.3f} ± {fC.err:.3f} (bootstrap), the same on the
rate-free plateau (next to it on the per-bar slide). It does not depend on the smearing (the same to 0.003 over the
whole scan, <code>wall_res_scan.csv</code>).</p>
<p><b>The smearing.</b> The muon fit's Gaussian (σ/MPV ≈ 0.16–0.18) cannot be applied to Geant4 for a 3 mm layer:
with the Landau width fixed at ξ it also absorbs the real straggling beyond a Landau, which Geant4 already simulates,
and on the cosmics it also carries the path-correction error. Applied anyway, Geant4 comes out too wide (KS p ≤ 0.006;
A {gA.scale:.3f}, C {gC.scale:.3f}). Fitted on the beam electrons instead, the extra smearing is
{100 * fA.r_fit:.0f} % (A) and {100 * fC.r_fit:.0f} % (C) at one MIP, σ ∝ √E — photostatistics plus the light
collection along the bar — and the shapes agree (KS p = {fA.ks_p_fit:.2f} / {fC.ks_p_fit:.2f}).</p>
<p><b>The ~5 %.</b> The same in both arms, at all times after the flash, and both halves of the wall see it.
Candidates: Geant4's thin-layer straggling vs the real one (the anchor carries Geant4's own), a few % of SiPM
non-linearity between a muon at normal incidence and electrons at the median angle, or the group sum collecting
less of a neighbour's light in the data. It is at the level of the method's own systematics and does not change how
the wall is used (a dE/dx tag).</p>"""
    D.slide('wallbeam', body, notes, short='Wall vs G4')


def s_bars(D, t):
    B = bars(t)
    rows = [(n, f.scale, sd.GREEN if abs(f.scale - 1) < 0.08 else sd.RED,
             f'{n}: {f.scale:.3f} +- {f.err:.3f}; KS p {f.ks_p_fit:.2f} after scaling; {int(f.n_data):,} tracks, '
             f'{int(f.n_g4)} Geant4') for n, f in B]
    for a in 'AC':
        w = fitrow(t, a, 'e_wall_fit', LATE)
        rows.append((f'wall {a}', w.scale, sd.BLUE, f'wall {a}: {w.scale:.3f} +- {w.err:.3f}'))
    hb = sd.hbars(rows, 1.25, width=560, label_w=150, fmt=lambda v: f'{v:.3f}', size=25)
    Q = sd.Plot(760, 520, x=(0.05, 0.99), y=(0.85, 1.2), xlabel='quantile of the bar\'s electron spectrum',
                ylabel='bar 1 / bar 2 energy', title='bar 1 / bar 2, data vs Geant4')
    Q.xticks([(v, f'{v:.1f}') for v in (0.1, 0.3, 0.5, 0.7, 0.9)])
    Q.yticks([(v, f'{v:.2f}') for v in (0.9, 1.0, 1.1, 1.2)])
    T = pd.read_parquet(WP / 'beam_data.parquet', columns=['arm', 'bar', 'e_plas', 'p_sat', 't_since_flash_ns'])
    S = pd.read_parquet(WP / 'g4_sample.parquet', columns=['armL', 'pL', 'pR', 'e_plas'])
    qs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    for arm in 'AC':
        d = T[(T.arm == arm) & ~T.p_sat & (T.t_since_flash_ns >= 40e6)]
        r = np.quantile(d[d.bar == 1].e_plas, qs) / np.quantile(d[d.bar == 2].e_plas, qs)
        s_ = S[S.armL == arm]
        L, Rr = s_[s_.pL >= s_.pR].e_plas, s_[s_.pL < s_.pR].e_plas
        g1, g2 = (L, Rr) if arm == 'A' else (Rr, L)
        rg = np.quantile(g1, qs) / np.quantile(g2, qs)
        Q.line(list(qs), list(r), ARMC[arm], w=3, r=6, tips=[f'{arm} data q{q:.2f}: {v:.3f}' for q, v in zip(qs, r)])
        Q.line(list(qs), list(rg), ARMC[arm], w=2.5, dash='8 6', r=4,
               tips=[f'{arm} Geant4 q{q:.2f}: {v:.3f}' for q, v in zip(qs, rg)])
    leg = sd.legend([('A data', sd.BLUE), ('A Geant4', sd.BLUE, 'dash'), ('C data', sd.ORANGE),
                     ('C Geant4', sd.ORANGE, 'dash')], size=21)
    c1 = dict(B)['C1']
    body = (sd.title(f'Bar by bar: three plastic bars and both walls within 5 % of Geant4; bar C1 reads {c1.scale:.2f}',
                     'scale that makes Geant4 match each counter, rate-free plateau (≥ 40 ms), 3–8 MeV · right: the '
                     'two bars of an arm against each other, quantile by quantile')
            + sd.row(sd.col(sd.p('data / Geant4 (1 = the muon calibration transfers exactly)', 23, sd.MUT), hb,
                            w=880, gap=16),
                     sd.col(Q.svg('bar ratio'), leg, gap=4), gap=40))
    notes = f"""
<p><b>Which Geant4 bar is which.</b> Pinned on A: the data's bar-1/bar-2 ratio is 1.06 at every quantile, as is
Geant4's left/right; A's bars sit at different distances from the source, so their spectra differ by that much.
C faces A, so its left and right mirror: Geant4's left is C's softer bar, as data bar 2 is.</p>
<p><b>C1.</b> The C ratio is flat in energy (a pure scale, not a non-linearity) at ~1.12 against Geant4's ~1.06,
and C1 alone sits {100 * (c1.scale - 1):.0f} % above Geant4 (its shape also disagrees at low energy: KS p
{c1.ks_p_fit:.3f}). Its muon calibration (C1 in the 8 Oct work, ±4 %) and the source calibration agree on C1/C2 to
3 %, so the channel answers electrons differently from muons and sources. Not resolved; candidates: a light-collection
or PMT-coupling difference that matters for the electrons' depth and position profile (they stop near the front face,
on the bar's inner half), or a gain step between run_149 and the beam runs on that tube alone.</p>
<p><b>Face maps.</b> The response across each bar, from the electrons' predicted crossings, is flat to ~5 %
(<code>face_map.csv</code>), but those crossings are smeared by ~50 mm of scattering, so a sharper non-uniformity is
not excluded.</p>"""
    D.slide('bars', body, notes, short='Per bar')


def s_rate(D, t):
    rr = t['rate_profile']
    P = sd.Plot(1400, 520, x=(12, 75), y=(0.88, 1.03), xlabel='time since the γ flash (ms)',
                ylabel='median / plateau (≥ 40 ms)')
    P.xticks([(v, str(v)) for v in (12, 20, 30, 40, 50, 60, 70)])
    P.yticks([(v, f'{v:.2f}') for v in (0.88, 0.92, 0.96, 1.0)])
    for (a, b), g in rr.groupby(['arm', 'bar']):
        x = list(0.5 * (g.lo + g.hi))
        pp = g.plas_med / g[g.lo >= 40].plas_med.mean()
        ww = g.wall_med / g[g.lo >= 40].wall_med.mean()
        P.line(x, list(pp), ARMC[a], w=3 if b == 1 else 1.8, r=5 if b == 1 else 3,
               tips=[f'plastic {a}{b} {lo:g}-{hi:g} ms: {v:.3f} ({n:,})' for lo, hi, v, n in zip(g.lo, g.hi, pp, g.n)])
        if b == 1:
            P.line(x, list(ww), ARMC[a], w=2.5, dash='3 6', markers=False, tip=f'wall {a}')
    leg = sd.legend([('A plastic (thick bar 1, thin bar 2)', sd.BLUE), ('C plastic', sd.ORANGE),
                     ('wall, dotted (A, C)', sd.MUT, 'dash')], size=22)
    body = (sd.title('The plastic PMTs lose 3–8 % gain at high rate after the flash; the SiPM wall moves < 2 %',
                     'median plastic and wall energy of the self-triggered electrons vs time since the flash, each '
                     'normalised to its own plateau')
            + P.svg('rate') + leg)
    notes = """
<p>From 12–15 ms to the plateau after ~40 ms the plastic median rises by 3 % (A) and 5–8 % (C); the wall moves the
other way by 1–2 %. The electron population is the same at all these times (late capture γs), so this is the PMTs'
gain recovering as the rate after the flash falls — the effect C1.4 saw on 8 Oct, now measured per bar. The
cosmic calibration is at zero rate, so the comparison with Geant4 uses the plateau.</p>
<p>For a facility set-up: a rate-dependent PMT gain is a known effect with known cures (tapered or active bases,
lower gain with more amplification after the tube, an LED/pulser monitor through the run).</p>"""
    D.slide('rate', body, notes, short='Rate')


def s_profile(D, t):
    Pr = t['wall_vs_plastic']
    panels = []
    for arm in ('A', 'C'):
        g = Pr[Pr.arm == arm]
        c = list(0.5 * (g.lo + g.hi))
        P = sd.Plot(790, 520, x=(2, 7.5), y=(0.3, 0.9), xlabel='plastic energy (MeV)',
                    ylabel='wall energy (MeV)', title=f'chamber {arm}')
        P.xticks([(v, str(v)) for v in range(2, 8)])
        P.yticks([(v, f'{v:.1f}') for v in (0.3, 0.5, 0.7, 0.9)])
        P.band(c, list(g.wall_q16_g4), list(g.wall_q84_g4), G4C, 0.08, tip='Geant4 16-84 %')
        P.band(c, list(g.wall_q16_data), list(g.wall_q84_data), ARMC[arm], 0.15, tip='data 16-84 %')
        P.line(c, list(g.wall_med_g4), G4C, w=3, r=5,
               tips=[f'Geant4 {lo:g}-{hi:g} MeV: median {m:.3f} (n = {n})' for lo, hi, m, n in
                     zip(g.lo, g.hi, g.wall_med_g4, g.n_g4)])
        P.line(c, list(g.wall_med_data), ARMC[arm], w=3, r=5,
               tips=[f'data {lo:g}-{hi:g} MeV: median {m:.3f} (n = {n:,})' for lo, hi, m, n in
                     zip(g.lo, g.hi, g.wall_med_data, g.n_data)])
        P.hline(0.472 / 0.85, sd.MUT, label='a MIP at the median angle', anchor='end',
                tip='0.472 MeV (the wall anchor) / cos 0.85')
        panels.append(P.svg(f'profile {arm}'))
    body = (sd.title('Wall against plastic: flat, as for minimum-ionising electrons, in data and Geant4 alike',
                     'median and 16–84 % band of the wall energy in bins of plastic energy · lines: data (colour), '
                     'Geant4 (ink)')
            + sd.row(*panels, gap=40)
            + sd.callout('The wall does not measure an electron\'s energy: from 1 to 7 MeV an electron loses '
                         'about the same 0.5 MeV in 3 mm. It measures dE/dx (one MIP vs two), and the plastic '
                         'measures the energy. Both behave as predicted.', sd.INK, 24))
    notes = """
<p>This is the "estimate in one, compare with the other" test in the form physics allows. Electrons from 1 to
7 MeV are close to minimum-ionising (collision dE/dx in PVT changes by &lt; 5 % over that range), so the wall
deposit should not follow the plastic energy. It does not, in either arm, and the slight rise at the lowest
plastic energies (slower, more steeply entering electrons) is in both.</p>
<p>For LNL this is the useful division of labour: the thin layer tags MIP vs 2-MIP (an unresolved pair or a
conversion in front of the stack) and the thick layer measures the energy of what stops in it.</p>"""
    D.slide('profile', body, notes, short='Wall vs plastic')


def s_close(D, t):
    B = dict(bars(t))
    wA, wC = fitrow(t, 'A', 'e_wall_fit', LATE), fitrow(t, 'C', 'e_wall_fit', LATE)
    li = lambda x: f'<li style="margin-bottom:14px">{x}</li>'
    left = ('<ul style="font-size:25px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li(f'Plastic: a muon calibration predicts 3–8 MeV stopping electrons in A1, A2 and C2 at '
                 f'{B["A1"].scale:.2f}, {B["A2"].scale:.2f} and {B["C2"].scale:.2f} of Geant4, linear, shapes '
                 'agreeing.')
            + li(f'Wall: a clean MIP peak, linear in path; electrons at {wA.scale:.2f} / {wC.scale:.2f} of Geant4, '
                 'shapes agreeing; it measures dE/dx, as it should.')
            + li('Stable to < 1 % over the campaign with one calibration; the plastic is 20 mm.')
            + li('For LNL: this stack works as an energy measurement for what stops in the plastic, and as a dE/dx '
                 'tag in the wall.') + '</ul>')
    right = ('<ul style="font-size:25px;line-height:1.35;color:#eef0f3;padding-left:28px">'
             + li(f'Bar C1 reads {B["C1"].scale:.2f} of Geant4 for electrons while muons and sources calibrate it: '
                  'unexplained, channel-specific.')
             + li('PMT gain sags 3–8 % in the 25 ms after the flash: needs a rate-proof base or a pulser monitor.')
             + li('Muons alone set the scale: add an in-situ electron reference (e.g. a conversion-electron source) '
                  'per channel at the next facility.')
             + li('Only 3–8 MeV tested (the trigger sets the floor); and the 8 Oct lesson stands: ~1.2 MeV is lost '
                  'before the plastic.') + '</ul>')
    body = (sd.title('<span style="color:#eef0f3">The scintillator calorimetry worked; two things to fix before '
                     'relying on it</span>')
            + sd.row(sd.col(sd.p('what the data show', 26, sd.DGREEN), left, w=800),
                     sd.col(sd.p('what to fix, and what this does not rule out', 26, sd.DRED), right, w=820), gap=56))
    D.slide('close', body, '', dark=True, short='Verdict')


def main() -> int:
    t = load()
    D = sd.Deck('Scintillator calorimetry: wall and plastic',
                'calibrated on cosmic muons, tested on beam electrons against Geant4')
    for f in (s_cover, s_setup, s_wall, s_stability, s_plastic, s_bars, s_rate, s_wallbeam, s_profile, s_close):
        f(D, t)
    out = WP / 'deck' / 'wall-plastic.html'
    D.write(out, note_meta=dict(
        title='Scintillator calorimetry: muon-calibrated wall and plastic vs beam electrons and Geant4',
        summary='Calibrated once on cosmic muons, the SiPM wall and three of four plastic bars describe the beam '
                'electrons at Geant4\'s scale to ~5 %, stable to <1 %; bar C1 reads 15 % high and the PMTs sag '
                '3-8 % at high rate.',
        tags='n_TOF, X17, scintillators, calorimetry, Geant4', date=dt.date.today().isoformat()))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
