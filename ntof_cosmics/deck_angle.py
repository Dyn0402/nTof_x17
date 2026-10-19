#!/usr/bin/env python3
"""
deck_angle.py -- the 2026-10-07 slides of the beam-off-cosmics note: the beam
angle scale (cosmic wall test, beam environment, Geant4) and Dylan's four
questions (gain, the A-C cosmic rate, source production after ~1 ms,
short-lived isotopes).  Called from make_deck.build; reads only the analyses'
outputs, so rerunning them moves the slides.

Inputs (all written by this package):
  /media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale/   cosmic_wall_scale.py ana
  /media/dylan/data/x17/ntof_cosmics/inbeam_through_goers inbeam_through_goers.py
  /media/dylan/data/x17/ntof_cosmics/g4_angle/            g4_angle/analyze.py
  results/clock_match/summary_*.json                      clock_match.py
  results/ac_rate/summary.json                            ac_cosmic_rate.py
  results/late_clock/summary.json                         late_trigger_clock.py
  results/activation/                                     activation_bound.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import slidedoc as sd
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,  # noqa: F401
                      INK, MUT, DBLUE, DRED, DGREEN, DMUT)

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
DATA = Path('/media/dylan/data/x17/ntof_cosmics')
STACK = Path('/media/dylan/data/x17/scint_stack/tracks')

#: one colour per population, on every slide of this section
C_COS, C_BEAM, C_SIM, C_MU = GREEN, ORANGE, PURPLE, BLUE

G = dict(
    wall=('The SiPM trigger wall behind each chamber: 16 bars of 25 mm in four groups of four, '
          'read out per group. The lever from the strip plane to the wall is L = 97.4 mm.'),
    ubin=('u-binned edge test: where the fired wall group switches across a surveyed boundary U_b, '
          'half the tracks cross on each side, so at that strip position u_b the median TRUE tan is '
          '(U_b − u_b)/L. Compared with the median reconstructed tan of the same tracks. Needs '
          'tracks whose angle is correlated with position (a source), so it works on beam.'),
    edge=('Edge-likelihood fit (ntof_scint_stack.ana.fit_wall_u): the scale s in u + L·s·tan, set by '
          'how sharp the three internal group boundaries become. Needs a spread of angles at fixed '
          'position, so it works on cosmics; on beam it is degenerate with the boundary offsets.'),
    raw='Production raw tan: the full-pass reconstruction\'s tan before any k is applied.',
    acline=('The A–C line: for a cosmic crossing both chambers, the straight line through the two '
            'chambers\' track points; its slope in A is the truth the in-situ calibration uses.'),
    sep='sep: closest distance between the A and C track lines; small = one straight particle.',
    qlen='q_per_len: fitted track charge per mm of path in the gap, ADC/mm; a gain proxy at fixed particle.',
    ideal=('Ideal reconstruction: an energy-weighted straight line through every true ionisation step '
           'in the 30 mm drift gap (Geant4 DriftGas hits). No digitisation, no drift, no noise.'),
)


# --------------------------------------------------------------------------- #
def load() -> dict:
    o = {}
    cw = DATA / 'cosmic_wall_scale'
    o['cw_sum'] = json.loads((cw / 'summary.json').read_text())
    o['cw_prof'] = pd.read_csv(cw / 'profile.csv')
    o['cw_scales'] = pd.read_csv(cw / 'scales.csv')
    o['beam_ref'] = pd.read_csv(cw / 'beam_reference.csv')
    o['inbeam'] = pd.read_csv(DATA / 'inbeam_through_goers' / 'scales.csv')
    g4 = DATA / 'g4_angle'
    o['g4_n'] = pd.read_csv(g4 / 'neutrons_wall_estimator.csv')
    o['g4_s'] = pd.read_csv(g4 / 'single_summary.csv')
    o['cm'] = pd.DataFrame([json.loads(Path(f).read_text()) for f in
                            sorted(glob.glob(str(RES / 'clock_match' / 'summary_run_149_*.json')))])
    o['ac'] = json.loads((RES / 'ac_rate' / 'summary.json').read_text())
    o['lc'] = json.loads((RES / 'late_clock' / 'summary.json').read_text())
    o['act'] = pd.read_csv(RES / 'activation' / 'subruns.csv')
    o['act_sum'] = json.loads((RES / 'activation' / 'summary.json').read_text())
    # charge per length: beam particles at A's wall (late), run_149 cosmics (all gated A)
    T = []
    for f in sorted(glob.glob(str(STACK / 'stack_run_*.parquet'))):
        x = pd.read_parquet(f, columns=['arm', 'n_trk', 'q_per_len', 't_since_flash_ns'])
        T.append(x[(x.arm == 'A') & (x.n_trk == 1)])
    b = pd.concat(T)
    ms = b.t_since_flash_ns / 1e6
    o['q_beam_late'] = float(b.q_per_len[(ms >= 20) & (ms < 80)].median())
    o['q_beam_early'] = float(b.q_per_len[(ms >= 10) & (ms < 12.5)].median())
    c = pd.concat(pd.read_parquet(f, columns=['arm', 'gated', 'q_per_len'])
                  for f in glob.glob(str(RES / 'tracking' / 'k_run_147' / 'tracks_run_149_*.parquet')))
    o['q_cos_all'] = float(c[(c.arm == 'A') & c.gated.astype(bool)].q_per_len.median())
    return o


# --------------------------------------------------------------------------- #
def s_section(D, o):
    S, B = o['cw_sum'], o['beam_ref']
    pl = B[B['sample'].isin(['20-30 ms', '30-45 ms', '45-80 ms'])].true_over_raw
    ib = o['inbeam']
    q_off = float(ib[(ib['sample'] == 'run_149 (beam off)') & (ib.sep_max == 60)].q_A.iloc[0])
    q_on = float(ib[(ib['sample'] == 'beam runs') & (ib.ms == '20-80') & (ib.sep_max == 60)].q_A.iloc[0])
    g4 = o['g4_n'].set_index('sample').true_over_raw
    nums = ''.join([
        sd.bignum(f'{S["s_best"]:.2f}', "cosmics at A's own wall", DGREEN,
                  f'true tan ÷ raw tan; the cosmic A–C line says 1.11. Beam: {S["beam_all"]:.2f} '
                  f'(after 20 ms {pl.min():.2f}–{pl.max():.2f}).',
                  tip=f'run_149, {S["n_single_x"]} single x-plane tracks, edge-likelihood profile; '
                      f'sub-run bootstrap {S["boot_median"]:.2f} (68 % {S["boot_lo"]:.2f}–{S["boot_hi"]:.2f}).'),
        sd.bignum(f'−{100 * (1 - q_on / q_off):.0f} %', 'gain under beam, same muons', '#f0a36b',
                  'yet muons crossing A–C in beam runs read the cosmic scale within 2–5 %: not gain, not noise.'),
        sd.bignum(f'{g4["A+C, all"]:.2f}', 'Geant4, PERFECT reconstruction', '#c39bd3',
                  f'the wall test on simulated beam electrons; above 4 MeV {g4["A+C, KE > 4 MeV"]:.2f}, '
                  'muons 1.00. The gap is electron scattering.')])
    body = (sd.kicker('Section 2 · the beam angle scale · 7 Oct 2026')
            + '<h1 style="font-size:76px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1640px">'
              'Beam tracks read ~20 % shallower at the SiPM wall than cosmics because few-MeV electrons '
              'scatter, not because the reconstruction differs</h1>'
            + '<div style="flex:1"></div>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('angle-cover', body, '''
<p><b>The question.</b> The in-situ calibration against the cosmic A–C line closes on cosmics (true ≈ 1.11 × production raw for A), but the scintillator walls say beam tracks are 0.89 × raw. Either beam really reconstructs differently, or the walls/lever arms/cosmic truth are off.</p>
<p><b>The chain of tests.</b> (1) cosmics on the same wall → the wall agrees with the A–C line, so the gap is real; (2) muons crossing A–C during beam runs → the beam environment (gain −34 %, beam noise) leaves the scale alone; (3) Geant4 with an ideal reconstruction → the wall estimator itself reads well below 1 for scattering few-MeV electrons.</p>
<p>Full record: <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §10d–10g. Reports: <code>/media/dylan/data/x17/ntof_cosmics/{cosmic_wall_scale,g4_angle}/report.html</code>.</p>''',
            dark=True, short='2 · Angle scale')


def s_method(D):
    W, H = 1060, 640
    o = []
    # side view: capsule left, chamber gap, wall with four groups (vertical = u)
    cx, cy = 90, 330
    o.append(f'<circle cx="{cx}" cy="{cy}" r="26" fill="{GREY}" fill-opacity=".35" stroke="{MUT}" stroke-width="2"/>')
    o.append(sd.T(cx, cy + 62, 'capsule', 21))
    gx0, gx1 = 430, 500                      # drift gap
    o.append(f'<rect x="{gx0}" y="60" width="{gx1 - gx0}" height="540" fill="{BLUE}" fill-opacity=".10" stroke="{BLUE}" stroke-width="2"/>')
    o.append(sd.T((gx0 + gx1) / 2, 40, 'drift gap 30 mm', 21, BLUE))
    wx = 860                                 # wall
    gy = [80, 210, 340, 470, 600]
    cols = [GREEN, GOLD, GREEN, GOLD]
    for i in range(4):
        o.append(f'<rect x="{wx}" y="{gy[i] + 3}" width="26" height="{gy[i + 1] - gy[i] - 6}" fill="{cols[i]}" fill-opacity=".55"'
                 + sd.tipattr(f'wall group {i}: 4 bars × 25 mm') + '/>')
        o.append(sd.T(wx + 60, (gy[i] + gy[i + 1]) / 2 + 8, f'g{i}', 21))
    o.append(sd.T(wx + 13, 40, 'SiPM wall', 21, INK))
    ub = gy[2]
    o.append(sd.line(wx - 30, ub, wx + 100, ub, RED, 2, '6 5'))
    o.append(sd.T(wx + 104, ub - 10, 'U_b', 22, RED, 'start'))
    # lever arrow
    o.append(sd.arrow(gx1, 625, wx, 625, MUT, 2))
    o.append(sd.T((gx1 + wx) / 2, 615, 'L = 97.4 mm', 21))
    # tracks from capsule through gap to the wall around the boundary
    for yy, c in [(318, ORANGE), (362, ORANGE), (338, ORANGE)]:
        t = (yy - cy) / (wx - cx)
        o.append(sd.line(cx, cy, wx, cy + t * (wx - cx) + (yy - cy) * 0, c, 2.5))
    # u_b on the gap
    o.append(f'<circle cx="{gx1}" cy="{cy + (338 - cy) * (gx1 - cx) / (wx - cx):.1f}" r="7" fill="{RED}"/>')
    o.append(sd.T(gx1 + 12, 300, 'u_b: half cross each side', 21, RED, 'start'))
    # a scattered electron
    o.append(sd.poly([gx0, gx1, 640, 760, wx], [470, 480, 470, 430, 404], PURPLE, 3, '8 6',
                     tip='A few-MeV electron changes direction in the gas, mesh, PCB, air and SiPM container: '
                         'its wall crossing no longer lies on its gap line.'))
    o.append(sd.T(650, 500, 'scattered e⁻', 21, PURPLE))
    diag = sd.svg(W, H, ''.join(o), 'how the wall measures an angle')
    right = sd.col(
        sd.p('The wall gives a capsule-free truth: a track\'s straight line from the strip plane must reach '
             'the group that fired.', 26),
        sd.callout('<b>Beam:</b> ' + sd.term('u-binned edge test', G['ubin']) + ' — the median true tan where '
                   'the fired group switches, against the median reconstructed tan there.', C_BEAM, 24),
        sd.callout('<b>Cosmics:</b> ' + sd.term('edge-likelihood fit', G['edge']) + ' — the scale that makes the '
                   'group boundaries sharpest.', C_COS, 24),
        sd.p('Both assume the particle goes straight from the gap to the wall. A muon does; a few-MeV '
             'electron does not.', 24, MUT),
        sd.p('Hover dotted terms and points for definitions and counts.', 21, MUT),
        gap=20, w=560)
    body = sd.title('The wall is angle truth only if the particle flies straight',
                    'Side view of one arm (not to scale). Group boundaries are surveyed; L from the strip plane.')
    body += sd.row(diag, right, gap=44)
    D.slide('angle-method', body, '''
<p>Two estimators are needed because the samples differ. Beam tracks come from the capsule, so tan ≈ (u − u_c)/D: at fixed position the angle is fixed, which the u-binned medians use, and which leaves no tan spread for the edge likelihood (there, s is degenerate with the per-boundary offsets — it returns 0.52 at late times where the u-binned test gives 0.92). Cosmics have no position–angle correlation, so the edge likelihood is the right tool and the u-binned test has no lever.</p>
<p>Code: <code>ntof_cosmics/wall_edge_scale.py</code> (u-binned), <code>ntof_scint_stack/ana.py</code> <code>fit_wall_u</code> (edge likelihood), <code>ntof_cosmics/cosmic_wall_scale.py</code> (cosmics).</p>''',
            short='Wall method')


def s_clockall(D, o):
    cm = o['cm']
    cm = cm[cm.efficiency_denominator > 0].sort_values(['subrun', 'ntof']).reset_index(drop=True)
    n = len(cm)
    P = sd.Plot(1100, 560, x=(-1, n), y=(80, 100), xlabel='(sub-run, n_TOF run) pair, in time order',
                ylabel='matched within ±50 ns  [%]')
    P.yticks([(v, str(v)) for v in range(80, 101, 5)])
    P.xticks([(i, cm.subrun.iloc[i][-4:]) for i in range(0, n, 6)])
    for i, r in cm.iterrows():
        tip = (f'{r.subrun} ↔ {r.ntof}: {r.matched_loo} matched of {r.efficiency_denominator} in-window '
               f'triggers ({100 * r.efficiency_loo:.1f} %); core MAD {r.res_core_mad_ns:.1f} ns')
        col = C_COS if r.efficiency_denominator > 400 else GREY
        P.vbar(i, 100 * r.efficiency_loo, 16, col, tip=tip, base=80)
    tot_m, tot_d = int(cm.matched_loo.sum()), int(cm.efficiency_denominator.sum())
    right = sd.col(
        sd.p(f'<b>{cm.subrun.nunique()} sub-runs × {cm.ntof.nunique()} n_TOF runs</b> ({n} pairs) — every '
             f'run_149 sub-run n_TOF recorded.', 26),
        sd.p(f'{tot_m:,} of {tot_d:,} in-window triggers matched '
             f'(<b>{100 * tot_m / tot_d:.1f} %</b>); core MAD {cm.res_core_mad_ns.min():.0f}–'
             f'{cm.res_core_mad_ns.max():.0f} ns.', 26),
        sd.callout('Two changes made it scale: DREAM timestamps from a 4 MB lxplus extract (not 11 GB of '
                   'decoded files), and sub-runs that straddle two n_TOF runs are trimmed to each run\'s span.',
                   C_COS, 23),
        sd.p('Grey: short overlaps (< 400 testable triggers).', 21, MUT),
        gap=20, w=540)
    body = sd.title(f'Clock match on all of run_149\'s n_TOF overlap: {100 * tot_m / tot_d:.0f} % matched',
                    'Leave-one-out efficiency per (DREAM sub-run, n_TOF run) pair.')
    body += sd.row(P.svg('clock match, all pairs'), right, gap=40)
    D.slide('clock-all', body, '''
<p>run_149 cos_0000–0034 overlap n_TOF 224678–224687 (8.6 h of n_TOF recording with no protons). n_TOF records 80 ms of every 0.5 s, so only ~16 % of DREAM triggers can be matched at all; that duty cycle, not the match, sets the cosmic sample size for anything using scintillator times.</p>
<p>Timestamps: <code>/media/dylan/data/x17/beam_july/dream_ts/run_149/ts_&lt;sub&gt;.npz</code> (eventId + timestamp of FEU 01). <code>clock_match.py</code> falls back to them when no decoded file is local, and keeps only DREAM triggers within <code>SPAN_PAD_S</code> = 15 s of the n_TOF run's bunches.</p>''',
            foot='run_149 cosbounce_cos_0000–0034 ↔ n_TOF 224678–224687 · results/clock_match/summary_*.json',
            short='Clock: all')


def s_cosmic_wall(D, o):
    P_, S, R = o['cw_prof'], o['cw_sum'], o['cw_scales']
    P = sd.Plot(1000, 600, x=(0.7, 1.5), y=(0, 210), xlabel='wall scale s  (true tan = s × raw tan)',
                ylabel='−log L, relative')
    P.xticks([(v, f'{v:.1f}') for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)])
    P.yticks([(v, str(v)) for v in range(0, 201, 50)])
    P.band([0.89, 0.93], [0, 0], [210, 210], C_BEAM, 0.18,
           tip=f'beam, u-binned: campaign {S["beam_all"]:.2f}, 20–80 ms plateau 0.91–0.93')
    P.vline(1.11, C_COS, '8 6', 3, label='cosmic A–C line', tip='in-situ truth from the A–C line (§7): 1.11')
    P.line(P_.s.tolist(), P_.dnll.tolist(), INK, 3.5, markers=False,
           tips=None, tip=f'cosmic profile, minimum at s = {S["s_best"]:.2f}; Δ at 0.89 = {S["dnll_at_beam"]:.0f}')
    P.text(0.905, 195, 'beam', 22, C_BEAM, 'middle')
    x = R[R.selection.str.startswith('x plane')].set_index('sample')
    rows = [[k, f'{int(x.loc[k, "n"]):,}', f'{x.loc[k, "s"]:.2f} ± {x.loc[k, "s_err"]:.2f}']
            for k in ('all', 'A-triggered', '|tan| < 0.15', '|tan| 0.15-0.3', '|tan| 0.3-0.6')]
    right = sd.col(
        sd.p(f'Cosmics read <b>{S["s_best"]:.2f}</b> at A\'s own wall (bootstrap {S["boot_lo"]:.2f}–'
             f'{S["boot_hi"]:.2f}), matching the ' + sd.term('A–C line', G['acline']) + ' (1.11).', 26),
        sd.table(['sample', 'n', 's'], rows, size=22),
        sd.callout(f'The beam value {S["beam_all"]:.2f} is disfavoured by Δ(−log L) = {S["dnll_at_beam"]:.0f}: '
                   'wall survey, lever arms and cosmic truth agree, so the beam/cosmic gap is real.', C_COS, 23),
        gap=20, w=600)
    body = sd.title(f'Cosmics at A\'s wall read {S["s_best"]:.2f}, beam {S["beam_all"]:.2f}: the gap is real', f'run_149, {S["n_single_x"]:,} single x-plane tracks clock-matched to '
                                       'n_TOF; edge likelihood profiled over width, offsets, floors.')
    body += sd.row(P.svg('cosmic wall profile'), right, gap=40)
    D.slide('cosmic-wall', body, f'''
<p>For each matched trigger, A's wall channels (n_TOF tree WALA) are read in ±50 ns around the matched singles time — on triggers from ANY arm, since raw tof turned out to be on a common zero within 2 ns for these trees. Tracks: the cosmic full-pass tables (raw tans are k-independent), u in the wall's structure frame.</p>
<p>Cross-checks: the binned version (the u where the fired group switches, against tan in narrow bins; slope −L·s, offsets cancel) gives ~1.13 over the central tans. The beam <code>gated</code> selection gives 1.13 ± 0.03. The scale rises with |tan|, as the A–C line also showed.</p>
<p>Only {S["n_truth"]} tracks carry A–C truth AND a matched trigger, too few for a truth-scale wall fit. Code: <code>ntof_cosmics/cosmic_wall_scale.py</code>.</p>''',
            short='Cosmic wall')


def s_beam_time(D, o):
    B = o['beam_ref']
    b = B[B['sample'] != 'all'].copy()
    b['lo'] = b['sample'].str.extract(r'(\d+)-').astype(float)
    b['hi'] = b['sample'].str.extract(r'-(\d+)').astype(float)
    b['mid'] = 0.5 * (b.lo + b.hi)
    S = o['cw_sum']
    P = sd.Plot(1060, 600, x=(5, 85), y=(0.6, 1.3), xlabel='time since the flash  [ms]',
                ylabel='true tan ÷ raw tan at the wall')
    P.xticks([(v, str(v)) for v in (10, 20, 30, 40, 50, 60, 70, 80)])
    P.yticks([(v, f'{v:.1f}') for v in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)])
    P.band([5, 85], [S['boot_lo']] * 2, [S['boot_hi']] * 2, C_COS, 0.18,
           tip=f'cosmics at the same wall, 68 % bootstrap {S["boot_lo"]:.2f}–{S["boot_hi"]:.2f}')
    P.text(8, S['boot_hi'] + 0.03, 'cosmics (same wall)', 22, C_COS)
    tips = [f'{r.sample}: {r.true_over_raw:.3f} (n {int(r.n):,} single-group tracks); D_eff {r.D_eff:.0f} mm'
            for r in b.itertuples()]
    P.line(b.mid.tolist(), b.true_over_raw.tolist(), C_BEAM, 3, tips=tips, r=9)
    P.text(8, 0.64, 'before 10 ms the edges are too wide to fit', 20, MUT)
    plateau = b[b.lo >= 20].true_over_raw
    right = sd.col(
        sd.p(f'After 20 ms the beam scale is flat at <b>{plateau.min():.2f}–{plateau.max():.2f}</b> while the '
             'trigger rate falls ×5. One population, one scale.', 26),
        sd.p(f'At 10–20 ms it is lower ({b.true_over_raw.iloc[0]:.2f} → {b.true_over_raw.iloc[1]:.2f}), with '
             f'D_eff {b.D_eff.iloc[0]:.0f} → {b.D_eff.iloc[1]:.0f} mm: a change of population, not recovery '
             '(next slide).', 26),
        sd.callout('The campaign\'s 0.89 averages the transient in. The plateau still sits ~20 % below cosmics.',
                   C_BEAM, 23),
        gap=22, w=560)
    body = sd.title(f'Beam at the wall: {plateau.mean():.2f} after 20 ms, lower at 10–20 ms',
                    'Arm A, single tracks with one wall group lit, all campaign runs; u-binned outer boundary pair.')
    body += sd.row(P.svg('beam scale vs time'), right, gap=40)
    D.slide('beam-time', body, '''
<p>D_eff is the effective source distance from the same two edges (tan-free): a point source on the beam axis predicts 234.6 mm. Beam tracks at A's wall are less divergent than radial (D_eff ≈ 310–330 mm on the plateau), which is why the capsule-pointing k (k_arm) carries a ×1.4 factor and is refuted.</p>
<p>Source: <code>cosmic_wall_scale.beam_reference()</code> on the scint-stack per-track tables (<code>/media/dylan/data/x17/scint_stack/tracks/</code>).</p>''',
            short='Beam vs time')


def s_gain(D, o):
    ib = o['inbeam']
    qrows = [('run_149 cosmics, all A', o['q_cos_all'], C_COS, 'all gated A tracks in the 87 run_149 sub-runs'),
             ('run_149 A–C muons', float(ib[(ib['sample'] == 'run_149 (beam off)') & (ib.sep_max == 60)].q_A.iloc[0]),
              C_COS, 'A–C through-goers, sep < 60 mm'),
             ('A–C muons in beam runs', float(ib[(ib['sample'] == 'beam runs') & (ib.ms == '20-80') & (ib.sep_max == 60)].q_A.iloc[0]),
              C_MU, 'same selection, beam runs, 20–80 ms after the flash'),
             ('beam particles, 20–80 ms', o['q_beam_late'], C_BEAM, 'single A tracks at the wall'),
             ('beam particles, 10–12 ms', o['q_beam_early'], C_BEAM, 'single A tracks at the wall')]
    bars = sd.hbars([(a, v, c, f'{t}: median q_per_len {v:.0f}') for a, v, c, t in qrows],
                    vmax=260, width=330, h=30, label_w=330, fmt=lambda v: f'{v:.0f}', size=22)
    drop = 1 - qrows[2][1] / qrows[1][1]
    P = sd.Plot(760, 520, x=(0, 70), y=(0.9, 1.15), xlabel='A–C line separation cut  [mm]',
                ylabel='A–C tan ÷ raw tan (median)')
    P.xticks([(v, str(v)) for v in (10, 20, 60)]).yticks([(v, f'{v:.2f}') for v in (0.9, 0.95, 1.0, 1.05, 1.1, 1.15)])
    for lab, ms, col in [('run_149 (beam off)', '-', C_COS), ('beam runs', '20-80', C_MU), ('beam runs', '40-80', GREY)]:
        x = ib[(ib['sample'] == lab) & (ib.ms == ms)].sort_values('sep_max')
        tips = [f'{lab} {ms} ms, sep < {int(r.sep_max)} mm: {r.median_ratio:.3f} ± {r.err:.3f} (n {int(r.n)}); '
                f'q_A {r.q_A:.0f}' for r in x.itertuples()]
        P.line(x.sep_max.tolist(), x.median_ratio.tolist(), col, 3, tips=tips, r=8)
    lg = sd.legend([('beam off (run_149)', C_COS), ('beam runs, 20–80 ms', C_MU), ('beam runs, 40–80 ms', GREY)], 21)
    left = sd.col(sd.p('<b>Charge per mm</b> (' + sd.term('q_per_len', G['qlen']) + ')', 24), bars,
                  sd.callout(f'Same muon selection: {qrows[1][1]:.0f} → {qrows[2][1]:.0f} under beam. Beam-particle charge '
                             f'is higher early ({o["q_beam_early"]:.0f} at 10–12 ms) than late ({o["q_beam_late"]:.0f}): '
                             'the wrong way for gain recovery.', C_MU, 23),
                  sd.p('Right: the muons\' angle scale against their own A–C line. Tightening the separation cut removes '
                       'accidental A–C coincidences; in beam runs the scale climbs to within 2–5 % of beam-off.', 23, MUT),
                  gap=22, w=820)
    right = sd.col(P.svg('in-beam muons'), lg, gap=10)
    body = sd.title(f'Gain is {100 * drop:.0f} % lower under beam; muons there keep the cosmic scale',
                    'Your question: is the angle scale a function of gain (recovering after the flash)?')
    body += sd.row(left, right, gap=44)
    D.slide('q-gain', body, '''
<p><b>Answer: no, not at the ~20 % level.</b> Three tests, chamber A.</p>
<ol><li>The same muon selection loses about a third of its charge per mm under beam, but that charge does not recover between 10 and 80 ms (in-beam muons flat at 115–121). Beam-particle charge moves the wrong way for recovery (higher at 10–12 ms than after 20 ms; numbers on the slide).</li>
<li>On beam, the wall scale is flat across q_per_len quintiles: 0.915–0.933 at 20–80 ms over a factor ~3 in q; no monotonic trend at 10–20 ms; run by run corr(q, scale) = 0.4 over q 93–105.</li>
<li>Muons crossing A and C during beam runs (one gated track each, lines within sep, joined line > 60 mm from the axis, ≥ 20 ms after the flash) read within 2–5 % of beam-off muons with identical truth and estimator. The beam-run values climb toward run_149 as sep tightens: accidental A–C coincidences dilute the loose cuts. So gain, beam-on noise and steady space charge are excluded as the cause; a ≤ 5 % environment effect is allowed.</li></ol>
<p>Code: <code>ntof_cosmics/inbeam_through_goers.py</code>.</p>''',
            short='Q: gain?')


def s_geant(D, o):
    g = o['g4_n'].set_index('sample')
    rows = [('A (sim arm 2)', g.loc['A (sim arm 2)', 'true_over_raw'], C_SIM),
            ('C (sim arm 3)', g.loc['C (sim arm 3)', 'true_over_raw'], C_SIM),
            ('A+C, KE > 4 MeV', g.loc['A+C, KE > 4 MeV', 'true_over_raw'], PURPLE),
            ('A+C, KE 2–4 MeV', g.loc['A+C, KE 2-4 MeV', 'true_over_raw'], PURPLE),
            ('A+C, KE < 2 MeV', max(g.loc['A+C, KE < 2 MeV', 'true_over_raw'], 0.0), PURPLE),
            ('μ⁻ 1 GeV', 1.0, C_MU)]
    tips = {r: f'{r}: {g.loc[r, "true_over_raw"]:.3f}, n {int(g.loc[r, "n"]):,}, D_eff {g.loc[r, "D_eff"]:.0f} mm'
            for r in g.index}
    hb = sd.hbars([(a, v, c, tips.get(a.replace('–', '-'), 'single muons: gap line = wall direction exactly'))
                   for a, v, c in rows], vmax=1.1, width=600, h=48, label_w=280,
                  fmt=lambda v: f'{v:.2f}', size=24)
    s = o['g4_s']
    piv = s[s.tan_gun >= 0.1].groupby(['particle', 'E']).gap_over_gun.median()
    right = sd.col(
        sd.p('The data\'s wall estimator on simulated beam-capture electrons, with an '
             + sd.term('ideal reconstruction', G['ideal']) + ':', 25),
        sd.callout(f'<b>{g.loc["A+C, all", "true_over_raw"]:.2f}</b> instead of 1 — and strongly energy dependent. '
                   'Muons: exactly 1.00. Data: beam 0.89–0.92 vs muons 1.10–1.15, a ratio ~0.80, inside the '
                   'simulated range.', C_SIM, 24),
        sd.p('So the wall is not angle truth for beam electrons without a forward model, and the cosmic '
             'in-situ scale stands as the reconstruction\'s scale.', 24),
        sd.p(f'X17 flag: the ideal gap line already reads electrons shallower than their emission direction '
             f'(median gap/gun {piv.get(("em", 5.0), float("nan")):.2f} at 5 MeV, {piv.get(("em", 8.0), float("nan")):.2f} '
             'at 8 MeV, over gun angles).', 22, MUT),
        gap=18, w=640)
    body = sd.title(f'Perfect reco in Geant4: the wall reads {g.loc["A+C, all", "true_over_raw"]:.2f} for beam e⁻, 1.00 for μ',
                    f'neutrons_thermal_trig_2cm_nose (10⁹ neutrons), {int(g.loc["A+C, all", "n"]):,} A+C gap tracks '
                    'reaching the wall; virtual boundaries at u = ±100 mm, L = 97.4 mm.')
    body += sd.row(sd.col(hb, gap=0, w=1000), right, gap=30)
    D.slide('geant4', body, '''
<p><b>Method.</b> <code>ntof_cosmics/g4_angle/reduce_gap_wall.py</code> (condor on lxplus, LCG_106) reduces each HitTree: per (event, arm), an edep-weighted line u(w), v(w) through all prompt DriftGas steps (t &lt; 10⁸ ns; RadioactiveDecay is on), the dominant track, and where that track first enters the SiPM wall. <code>analyze.py</code> then runs the data's outer-pair u-binned estimator with boundaries placed at u = ±100 mm on the wall plane.</p>
<p><b>Population.</b> 91 % e⁻ (rest e⁺), KE in the gap quartiles 2.0 / 3.1 / 4.6 MeV — Compton electrons of capture γs, mostly the 7.7 MeV Al line from the capsule nose. The dominant-track line alone gives the same 0.59: delta rays are not the cause.</p>
<p><b>Single particles</b> (capsule centre → arm A, fixed gun angles, 20k each): wall/gap per gun angle 0.93 at 8 MeV, 0.75 at 5 MeV, muons 1.000; these carry wall-edge truncation (fixed guns aim near the wall's outer edge at large angle), so they show the trend only.</p>
<p><b>Not settled:</b> the data's exact 0.92 needs the real population (late triggers are ambient hall-neutron captures, absent from the sim) and the real reconstruction's weighting of scattered charge (a <code>wft</code>-digitised forward model).</p>''',
            short='Geant4')


def s_rate(D, o):
    A = o['ac']
    e = np.array(A['zen_bins'])
    hm = np.array(A['expected']['zen_hist_hz'])
    ho = np.array(A['observed']['zen_hist'], float)
    c = 0.5 * (e[1:] + e[:-1])
    fm, fo = hm / hm.sum(), ho / ho.sum()
    P = sd.Plot(1000, 560, x=(50, 90), y=(0, max(fm.max(), fo.max()) * 1.15),
                xlabel='zenith angle of the A–C line  [deg]', ylabel='fraction per 2.5°')
    P.xticks([(v, str(v)) for v in range(50, 91, 10)])
    P.yticks([(v, f'{v:.2f}') for v in np.arange(0, P.ys[1], 0.05)])
    ex, ey = sd.step_xy(e.tolist(), fm.tolist())
    P.raw(sd.poly([P.X(a) for a in ex], [P.Y(b) for b in ey], C_SIM, 3,
                  tip='Monte Carlo, flux-weighted, after acceptance, gate and trigger'))
    P.points(c.tolist(), fo.tolist(), C_COS, 8,
             tips=[f'{a:.1f}°: {int(n)} pairs ({f:.3f})' for a, n, f in zip(c, ho, fo)])
    exp_h = A['expected']['rate_gated_trig'] * 3600
    obs_h = A['observed']['rate_sep60_hz'] * 3600
    nums = sd.col(
        sd.bignum(f'{exp_h:.0f} /h', 'expected', PURPLE, 'open-sky muon flux, 100 % efficient chambers', dark=False, size=60,
                  tip=f'I₀ = {A["I0"]} m⁻²s⁻¹sr⁻¹ with the Chirkin large-zenith correction; A and C active areas; '
                      '|tan| < 0.6 both planes; wall∧plastic on A or C.'),
        sd.bignum(f'{obs_h:.0f} /h', 'observed, run_149', GREEN, f'{A["observed"]["n_sep60"]:,} clean A–C '
                  f'through-goers in {A["observed"]["hours"]:.1f} h', dark=False, size=60),
        sd.callout(f'ε_A·ε_C ≈ <b>{obs_h / exp_h:.2f}</b> → ~{math.sqrt(obs_h / exp_h):.2f} per chamber, as on '
                   'the bench (40–65 %). Zenith medians '
                   f'{A["expected"]["zen_median"]:.1f}° vs {A["observed"]["zen_median"]:.1f}°.', C_COS, 23),
        gap=18, w=560)
    body = sd.title('The A–C cosmic rate matches the muon flux at ~60 % efficiency per chamber',
                    'Your question. At EAR2 the beam is vertical, so an A–C through-goer is near-horizontal.')
    body += sd.row(P.svg('A-C zenith'), nums, gap=44)
    lg = sd.legend([('Monte Carlo (shape)', C_SIM), ('run_149 data', C_COS, 'dot')], 21)
    body += lg
    D.slide('q-rate', body, f'''
<p><b>Monte Carlo</b> (<code>ntof_cosmics/ac_cosmic_rate.py</code>): directions from I₀cos²θ* with Chirkin's large-zenith cos θ*, I₀ = 70 m⁻²s⁻¹sr⁻¹ (E ≳ 1 GeV, open sky); lines uniform on a 0.9 m disk; accepted if crossing A's and C's measured active areas (398.6 × 362 mm), passing the gate's |tan| &lt; 0.6 in both local planes of both chambers, and firing wall AND plastic on A or C (run_149 geometry from run_config.json).</p>
<p><b>Observed:</b> run_149, events with exactly one gated track in A and in C whose lines pass within 60 mm ({A["observed"]["n_pairs"]:,} single-single pairs in all, {A["observed"]["rate_pairs_hz"] * 3600:.0f} /h).</p>
<p>The building and the EAR2 bunker attenuate the flux somewhat (more vertically than horizontally), which would raise the implied efficiencies. The shape agreement is the stronger statement: these are cosmic muons.</p>''',
            short='Q: A–C rate')


def _clock_curve(p, t):
    A, n, B, T, C = p
    return A * (t / 30) ** (-n) + B * 2 ** (-(t - 30) / T) + C


def s_source(D, o):
    L = o['lc']['C']
    e = np.array(L['edges'])
    h = np.array(L['hist'], float)
    c = 0.5 * (e[1:] + e[:-1])
    ok = h > 0.2 * np.median(h[h > 0])
    P = sd.Plot(1060, 600, x=(0, 82), y=(1e1, 1e6, 'log'), xlabel='time since the flash  [ms]',
                ylabel='triggers with a gated C track, per ms')
    P.xticks([(v, str(v)) for v in range(0, 81, 10)]).yticks(sd.log_ticks(1, 6))
    P.points(c[ok].tolist(), h[ok].tolist(), INK, 5,
             tips=[f'{a:.1f} ms: {int(v):,} triggers' for a, v in zip(c[ok], h[ok])])
    pf = L['free']['p']
    tt = np.linspace(20, 80, 61)
    P.line(tt.tolist(), [_clock_curve(pf, t) for t in tt], RED, 3, markers=False,
           tip=f'fit 20–80 ms: exponential T½ = {pf[3]:.1f} ± {L["free"]["err"][3]:.1f} ms + power + const; '
               f'χ²/ndf {L["free"]["chi2"]:.0f}/{L["free"]["ndf"]}')
    # beam captures in a thin 1/v absorber: ~t^-4, normalised at 20 ms
    y20 = float(np.interp(20.5, c[ok], h[ok]))
    t4 = np.linspace(20, 80, 61)
    P.line(t4.tolist(), [y20 * (t / 20.5) ** -4 for t in t4], GREY, 3, dash='10 7', markers=False,
           tip='direct capture of beam neutrons arriving at t (thin 1/v absorber): ~t⁻⁴, normalised at 20 ms')
    P.text(24, y20 * (40 / 20.5) ** -4 * 0.5, 'beam neutrons ∝ t⁻⁴', 21, GREY)
    P.text(48, _clock_curve(pf, 48) * 2.2, f'T½ = {pf[3]:.1f} ms', 22, RED)
    right = sd.col(
        sd.p('Agreed: production on the target should not change after ~1 ms. But after ~30 ms the triggers are '
             '<b>not</b> target production at all.', 26),
        sd.p(f'Beam neutrons arriving that late (E ≲ 2.5 meV; the evaluated EAR2 flux has ~nothing there) would '
             f'give a capture rate falling ~t⁻⁴. The data fall as one exponential, T½ = <b>{pf[3]:.1f} ± '
             f'{L["free"]["err"][3]:.1f} ms</b> (A: {o["lc"]["A"]["free"]["p"][3]:.1f} ms).', 25),
        sd.callout('Best reading: thermalised neutrons dying away in the EAR2 hall (τ ≈ 34 ms; ¹⁴N capture in '
                   'air alone gives ~60 ms) and captured all around the setup — a population the Geant4 beam '
                   'campaign does not contain.', C_BEAM, 23),
        gap=20, w=580)
    body = sd.title(f'After ~30 ms the triggers are not beam captures: one {pf[3]:.1f} ms exponential',
                    'Your question: source production after ~1 ms. Arm C (the cleanest fit), all campaign runs.')
    body += sd.row(P.svg('late trigger clock'), right, gap=36)
    D.slide('q-source', body, f'''
<p>Fit: R(t) = A·t⁻ⁿ + B·2^(−t/T) + C on 1 ms bins, 20–80 ms, comb-emptied bins masked (<code>ntof_cosmics/late_trigger_clock.py</code>). The exponential carries ~100 % of the rate at 40 ms. The local half-life is constant at 23–24.5 ms from 35 to 75 ms, where any power law's would grow linearly with t.</p>
<p>Before ~20 ms the rate is steeper (local T½ ~12 ms at 15–25 ms): beam captures still dominate there. That is also where the beam angle scale dips (0.77 at 10–15 ms) and D_eff changes — the population change of the beam-time slide.</p>
<p>Consequence: the late beam "sample" in A's wall maps and the 20–80 ms plateau is ambient-capture background, not capsule production. Simulating it needs an ambient thermal-neutron source in Geant4 (capture vertices in the arms' structure), not the beam.</p>''',
            short='Q: production')


def s_isotopes(D, o):
    a = o['act']
    S = o['act_sum']
    P = sd.Plot(860, 580, x=(1, 3000, 'log'), y=(23.5, 27), xlabel='minutes since the last n_TOF proton pulse',
                ylabel='beam-off trigger rate  [Hz]')
    P.xticks([(v, str(v)) for v in (1, 3, 10, 30, 100, 300, 1000, 3000)])
    P.yticks([(v, f'{v:.1f}') for v in (24, 25, 26, 27)])
    a = a[(a.min_since_beam > 1) & a.rate_hz.between(23.5, 27)]
    P.points(a.min_since_beam.tolist(), a.rate_hz.tolist(), C_COS, 6,
             tips=[f'{r.run} {r.subrun}: {r.rate_hz:.2f} Hz, {r.min_since_beam:.0f} min after beam'
                   for r in a.itertuples()])
    lc = o['lc']
    rows = []
    for arm in ('A', 'C'):
        f, x = lc[arm]['free'], lc[arm]['fixed_12B']
        rows.append([arm, f'{f["p"][3]:.1f} ± {f["err"][3]:.1f}', f'{f["chi2"]:.0f}/{f["ndf"]}',
                     f'{x["chi2"]:.0f}/{x["ndf"]}'])
    iso = ', '.join(f'{k} {v["delta_rate_hz"]:+.2f}' for k, v in S.items())
    right = sd.col(
        sd.p('<b>Minutes to hours</b> (²⁸Al, ⁶⁶Cu, ⁴¹Ar from the argon, ⁵⁶Mn, ²⁴Na): sub-runs 3–7 min after the beam '
             'stops trigger at the same 25 Hz as 20 h later. Bound ≲ 0.2 Hz against 110–600 Hz of late beam triggers.', 24),
        sd.p('<b>Sub-second:</b> ¹²B (T½ 20.2 ms, ¹²C(n,p) by flash neutrons on every scintillator and the capsule\'s '
             'CFRP) is the natural candidate for the late clock — but fixing T½ at 20.2 ms fits clearly worse:', 24),
        sd.table(['arm', 'free T½ [ms]', 'χ² free', 'χ² T½ = 20.2'], rows, size=21),
        sd.p('Fit 20–80 ms: exponential + power law + constant; 1 ms bins.', 20, MUT),
        gap=16, w=700)
    body = sd.title('No activation after the beam stops; ¹²B is not the late clock',
                    'Your question: short-lived isotope decays (Geant4 has RadioactiveDecay on, but every analysis cuts t < 100 ms).')
    body += sd.row(P.svg('beam-off rate after beam'), right, gap=40)
    D.slide('q-isotopes', body, f'''
<p><b>Is it simulated?</b> Yes: MX17_Full_Geant runs RadioactiveDecayPhysics and the neutron campaigns contain ²⁸Al β decays (half of DriftGas charged hits, at 10⁹–10¹³ ns). Every reduction cuts hit time &lt; 10⁸ ns, so no study uses them — correctly for the prompt analysis, but decays within the 80 ms window from isotopes made by the flash (sub-second half-lives) would need fast-neutron (flash) primaries, which the thermal/epithermal campaigns do not fire.</p>
<p><b>Activation bound</b> (<code>ntof_cosmics/activation_bound.py</code>): per beam-off sub-run at the production point with no pulse inside it, the trigger rate against a decay-weighted proton history Σ P_i e^(−Δt/τ) from the per-minute slow-control logs. Rate change across each index's range: {iso} Hz; correlations ≤ 0.10. The gas is Ar/iso 90/10, so no ²⁰F.</p>
<p><b>Sub-second:</b> a ¹²B component of up to ~20–30 % is not excluded (a two-exponential fit is not constrained by these data). How to pin it: the ¹²B β spectrum reaches 13.4 MeV, but the 2 cm LS only samples ~4 MeV of a crossing electron and is calibrated at one point; a Geant4 run with flash (> 13.6 MeV) neutrons and RadioactiveDecay kept to 100 ms would predict its rate directly.</p>''',
            short='Q: isotopes')


def s_close(D):
    items = [
        ('The data\'s exact 0.92', 'Needs the real late population (ambient hall-neutron captures, not in the sim) and the real fit\'s weighting of scattered charge (wft-digitised forward model).'),
        ('The 10–20 ms dip (0.77)', 'Read as a change of population/energy, consistent with the Geant4 energy dependence; not shown.'),
        ('A small environment effect', 'In-beam muons read 2–5 % low; ≤ 5 % is allowed.'),
        ('Sub-second activation', '¹²B up to ~20–30 % of the late clock is not excluded.'),
        ('Chamber C and D', 'The in-beam muon test and the wall test were done on A only (the Geant4 result covers A and C).'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:14px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:27px;font-weight:600;width:380px">{a}</p>'
                    f'<p style="font-size:23px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    nxt = [('X17 opening angle', 'The ideal gap line is ~10 % compressed against the emission direction at 8 MeV (~20 % at 5 MeV): check the pair simulation includes it.'),
           ('Calibration decision', 'Use the cosmic in-situ scale (A 1.11, C in-situ) for beam angles; retire the wall/capsule k as truth. Then T1 on the in-situ bundles with seeder min 3.'),
           ('Ambient-neutron source in Geant4', 'Capture vertices in the arms\' structure, τ ≈ 34 ms: the late population for backgrounds and for the wall check.'),
           ('Optional', 'wft-digitised forward model; Geant4 flash run for ¹²B.')]
    nrows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:14px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:27px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:23px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in nxt)
    body = (f'<div style="display:flex;gap:72px">'
            f'<div style="flex:1.15;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:50px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:50px;font-weight:600">Next</h2>{nrows}</div></div>')
    D.slide('close', body, '''
<p>Entry points: <code>ntof_cosmics/README.md</code>; the full record and next steps are in <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §10–11 and the top-level <code>HANDOFF.md</code> (branch <code>beam-off-cosmics</code>). The clock-match caveats of section 1 (single-trigger bunches, straight drift line, 92 % vs 96 %) still stand; "match all of it" and "track the cosmic runs" are done for run_149.</p>''',
            dark=True, short='Caveats & next')
