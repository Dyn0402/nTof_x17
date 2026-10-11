#!/usr/bin/env python3
"""
make_insitu_deck.py -- slide note: chambers A and C on the in-situ angle
calibration (is2), how good it is, what is still open, and what to check
before the is2_v1 campaign re-pass.

Reads the analyses' own outputs, so re-running them moves the slides:
  results/repass_readiness/*.csv          repass_readiness.py cosmic / beam
  results/seed_beam/data/summary.csv      seed_beam_test.py compare
  /media/dylan/data/x17/ntof_cosmics/inbeam_through_goers/pooled_norm.csv
  /media/dylan/data/x17/ntof_cosmics/g4_digi/compare_data.csv
  /media/dylan/data/x17/sept26_prelim/kcal_is2_v1/k_arm_run_*.json
Numbers that exist only in the record (HANDOFF_TRACKING_2026-10-06.md) are in
LOGGED, each with its section.

    .venv/bin/python ntof_cosmics/make_insitu_deck.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        ntof_cosmics/results/deck/ac-insitu-angles.html --slug ac-insitu-angles --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,  # noqa: E402,F401
                      INK, MUT, DBLUE, DRED, DGREEN, DMUT, DINK)

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
RR = RES / 'repass_readiness'
DATA = Path('/media/dylan/data/x17/ntof_cosmics')
KCAL = Path('/media/dylan/data/x17/sept26_prelim/kcal_is2_v1')

#: one colour per chain / population, on every slide
C_PROD, C_IS2, C_M3, C_MU, C_SIM, C_BEAM = RED, GREEN, BLUE, BLUE, PURPLE, ORANGE

#: numbers that live only in the written record, with where they come from
LOGGED = dict(
    v=dict(src='§8 (insitu_calib free fits against the A–C line)',
           bench=dict(A=36.6, C=26.7), geom=dict(A=37.5, C=25.8), prior=42.6, is2=dict(A=38.0, C=28.7)),
    nearnormal=dict(src='§9 (seed_test.py, run_149, same triggers, min 5 vs min 3)',
                    ac_pairs=(945, 1570), A_x_n=(34, 205), A_x_sigma=('0.18–0.22', '0.03–0.06'),
                    C_x_sigma=('0.24–0.44', '0.06–0.08'), below002=0.15),
    # §13 Result 1: x reco / ideal gap line, per |tan| band
    digi=dict(src='§13 Result 1 (g4_digi/analyse.py)', bins=['0.10–0.30', '0.35–0.45', '0.45–0.55'], x=[0.2, 0.4, 0.5],
              A_mu=[0.98, 0.96, 0.95], A_e=[0.975, 0.965, 0.94], C_mu=[1.00, 0.99, 0.98], C_e=[0.995, 0.985, 0.97]),
    # A's beam angle scale (true / production raw) from each estimator
    estimators=dict(src='§10c–10g, §13', rows=[
        ('Capsule pointing (k_arm, production)', 1.27, RED,
         'k_arm band: assumes every track comes from a point on the beam axis 234.6 mm away. Production applies this.'),
        ('Cosmic A–C line (truth on cosmics)', 1.11, GREEN,
         'run_149 through-going muons: the straight line through A and C. §7a.'),
        ('Cosmics at A\'s own wall', 1.15, GREEN,
         'edge likelihood on 3,088 single x-plane cosmic tracks; bootstrap 1.12–1.19. §10d.'),
        ('In-beam muons (A–C, sep < 10)', 1.09 * 0.992, BLUE,
         'A–C muons inside beam runs, 20–80 ms, gas-normalised to run_145: 0.992 × the run_149 value 1.10. §13.'),
        ('Beam particles at A\'s wall (u-binned)', 0.89, ORANGE,
         'SiPM-wall group boundaries binned in strip position, 990 k tracks, 31 runs. Few-MeV electrons scatter '
         'between the gap and the wall: §10g.'),
        ('Geant4 electrons, perfect reco, same wall test', 0.60 * 1.11, PURPLE,
         'the wall estimator reads 0.59–0.60 of the true gap angle on simulated capture electrons (×1.11 to put it on '
         'this axis); 0.92 above 4 MeV, 1.00 for muons. §10g.')]),
    source_width=dict(src='§13 residual_checks.txt (sim band vs the true source rms at the capsule plane)',
                      rms=[3, 6, 11, 15, 20, 26], A=[1.005, 1.012, 1.023, 1.038, 1.058, 1.068],
                      C=[0.997, 0.995, 1.002, 1.016, 1.033, 1.048],
                      core=dict(A=(1.135, 1.070), C=(1.117, 1.032)),
                      y=dict(A=(1.332, 1.101), C=(1.316, 1.033))),
    fp145=dict(src='§13 production-bundle reruns (cluster 4405795)', A=1.018, C=1.103),
)

G = dict(
    is2=('is2: the in-situ bundles. A = production kernel, v 38.0 µm/ns, per-plane robust kw (x 1.020, y 0.993). '
         'C = r06 det7 kernel, v 28.7, kw x 1.001 / y 0.957. Both with the seeder at 3 strips. '
         'Built from run_149 beam-off cosmics against the A–C line (HANDOFF_TRACKING §10b–c).'),
    acline=('The A–C line: a cosmic muon crossing both chambers is one straight line; its slope in each chamber '
            'is the truth. Lever 469 mm, so its own error is ~0.002 in tan.'),
    k=('k: the stage-3 factor, tan = k × raw tan. Production: k_arm from capsule pointing (A 1.27, C 1.62 on run_145). '
       'is2_v1: muon norm × band(run)/band(run_145).'),
    raw='raw tan: the fit\'s own tan, w/v with the bundle\'s v and kw, before stage 3 applies k.',
    m3='Production bundle with only the seeder changed to a 3-strip minimum: the §10a beam-purity test.',
    tanmax=('wft.reco.TAN_MAX = 0.6: a fit is "plausible" only if |raw tan| < 0.6 (and 250 ≤ q_uend ≤ 1100 ns). '
            'Implausible fits lose the candidate ranking and fail the x/y pairing gate.'),
    sep='sep: the closest distance between the A and C track lines; small = one straight particle.',
    band=('Capsule band: k_arm\'s regression of the track\'s lever (position relative to the pinwheel foot) on tan, '
          'assuming all tracks come from the capsule axis. It depends on the source distribution as much as on the scale.'),
    confirmed=('Scintillator-confirmed: the track extrapolated to its arm\'s SiPM wall and plastic hits the bar that '
               'fired (det_a_scint.match_run), minus the same-width off-time control window (accidentals).'),
    digi=('g4_digi: Geant4 DriftGas steps → electrons → the bundle\'s own kernel and template → added to real quiet '
          'run_145 raw ADC → pedestal + common-mode → emulated hits → the unchanged seeder and wft fit.'),
)


# --------------------------------------------------------------------------- #
def load() -> dict:
    o = dict(cc=pd.read_csv(RR / 'cosmic_closure.csv'), shift=pd.read_csv(RR / 'beam_shift.csv'),
             gate=pd.read_csv(RR / 'gate_tanmax.csv'), scan=pd.read_csv(RR / 'gate_tanmax_scan.csv'),
             scint=pd.read_csv(RR / 'scint_confirmation.csv'),
             seed=pd.read_csv(RES / 'seed_beam' / 'data' / 'summary.csv'),
             norm=pd.read_csv(DATA / 'inbeam_through_goers' / 'pooled_norm.csv'),
             digi=pd.read_csv(DATA / 'g4_digi' / 'compare_data.csv'))
    rows = []
    for f in glob.glob(str(KCAL / 'k_arm_run_*.json')):
        d = json.loads(Path(f).read_text())
        r = int(re.findall(r'run_(\d+)', f)[0])
        a = d['arms']
        rows.append(dict(run=r, A=d['apply'].get('A'), C=d['apply'].get('C'),
                         gA=a.get('A', {}).get('gas_ratio'), gC=a.get('C', {}).get('gas_ratio')))
    o['k'] = pd.DataFrame(rows).sort_values('run').reset_index(drop=True)
    return o


def _cc(o, arm, ax, chain):
    c = o['cc']
    return c[(c.arm == arm) & (c.axis == ax) & (c.chain == chain)].sort_values('lo')


def _core(o, arm, chain):
    """median |ratio − 1| over 0.08–0.6, both axes."""
    c = o['cc']
    s = c[(c.arm == arm) & (c.chain == chain) & (c.lo >= 0.08)]
    return float(np.median(np.abs(s.ratio - 1)))


# --------------------------------------------------------------------------- #
def s_cover(D, o):
    n = o['norm'][o['norm'].sep_max == 10]
    dev = (1 - n.norm).abs()
    g = o['gate'].set_index(['arm', 'chain'])
    sc = o['scint'].set_index(['arm', 'chain'])
    pA = _core(o, 'A', 'production')
    nums = ''.join([
        sd.bignum(f'{100 * dev.min():.0f}–{100 * dev.max():.0f} %', 'is2 against in-beam muons', DGREEN,
                  'muons crossing A and C during beam runs, the only angle truth that is not a model of the source.',
                  tip='pooled_norm.csv, sep < 10 mm: ' + ', '.join(
                      f'{r.arm} {r.view} {r.norm:.3f} ± {r.norm_err:.3f} (n {int(r.pooled_n)})' for r in n.itertuples())),
        sd.bignum(f'{100 * pA:.0f} %', 'production\'s typical angle error, A', DRED,
                  'production tans are too steep on cosmic truth; is2 closes A to '
                  f'{100 * _core(o, "A", "is2_v1"):.0f} % and C to {100 * _core(o, "C", "is2_v1"):.0f} %.',
                  tip='median |delivered/true − 1| over |tan| 0.08–0.6, x and y, held-out run_149 A–C tracks '
                      '(repass_readiness.py cosmic).'),
        sd.bignum(f'{g.loc[("C", "production"), "true_tan_reach"]:.2f} → {g.loc[("C", "is2_v1"), "true_tan_reach"]:.2f}',
                  'C\'s angular reach under the re-pass', '#f0c060',
                  'a fixed raw-tan cut in the fit changes meaning with v. Decide it before launching.',
                  tip=f'TAN_MAX = 0.6 raw × k. Confirmed C tracks on run_145: m3 {sc.loc[("C", "m3"), "wall_excess"]:,} '
                      f'→ is2_v1 {sc.loc[("C", "is2_v1"), "wall_excess"]:,}.')])
    body = (sd.kicker('Chambers A and C · in-situ angle calibration · 8 Oct 2026')
            + '<h1 style="font-size:72px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1660px">'
              'The in-situ calibration measures A and C angles to a few per cent; one cut in the fit '
              'has to be settled before the campaign re-pass</h1>'
            + '<div style="flex:1"></div>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, '''
<p><b>What this note is.</b> A review of the chamber A and C angle work on branch <code>beam-off-cosmics</code> (6–8 Oct), written before spending ~4300 CPU-h on the <code>is2_v1</code> campaign re-pass. It says how good the in-situ calibration is, which problems are still open, and which small checks are worth doing first. One new check, made for this note, found a cut in the fit whose meaning changes with the bundle (slide 11).</p>
<p><b>Sample.</b> Cosmic truth is run_149 (beam-off, 87 sub-runs, August), held-out split. Beam checks use run_145 stat090_0000 (the re-pass smoke sub-run, all 7 tags) and pooled in-beam muons from 36 production runs. Every production run is on the post-23-July (noisy) readout configuration, so the noise-floor boundary does not split this work.</p>
<p>Full record: <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §7–13. Code for the new checks: <code>ntof_cosmics/repass_readiness.py</code>.</p>''',
            dark=True, short='Cover')


def s_setup(D, o):
    prod = [dict(label='Hits → seeder', sub='cluster ≥ 5 strips', color=RED,
                 tip='MIN_STRIPS_BEAM = 5. Head-on tracks have only 3–4 strips at n_TOF S/N and are lost (§9).'),
            dict(label='wft fit', sub='bench kernel, v = 42.6 prior', color=RED,
                 tip='wft_beam.make_bundle keeps the bench kernel but swaps the bench v for the Magboltz 42.6.'),
            dict(label='plausible?', sub='|raw tan| < 0.6', color=GREY, tip=G['tanmax']),
            dict(label='stage 3 × k', sub='capsule k: A 1.27, C 1.62', color=RED,
                 tip='k_arm: capsule pointing. It corrects the v error, but its assumption of a point source on the '
                     'axis is wrong for the real population (§10c).'),
            dict(label='consumers', sub='opening angle, 170° cut, slope_reliable, T1 F', color=GREY)]
    is2 = [dict(label='Hits → seeder', sub='cluster ≥ 3 strips', color=GREEN,
                tip='WFT_BEAM_MIN_STRIPS=3: head-on recovered, confirmed tracks +45 % (A) +52 % (C) on run_145.'),
           dict(label='wft fit', sub='in-situ bundle, geometric v, kw', color=GREEN, tip=G['is2']),
           dict(label='plausible?', sub='|raw tan| < 0.6 — now ≈ 0.58 true', color=GOLD, tip=G['tanmax']),
           dict(label='stage 3 × k', sub='muon norm × gas ratio ≈ 0.84–0.98', color=GREEN,
                tip='k_insitu.py: k = norm(arm) × band(run)/band(run_145); norm from in-beam muons (A 0.976, C 0.962).'),
           dict(label='consumers', sub='unchanged code; B, D angles null', color=GREY)]
    body = sd.title('The re-pass changes three things in the A and C chain, and leaves a fourth alone',
                    'Production (top) against is2_v1 (bottom). Hover the boxes, the dotted terms and every chart point.')
    body += sd.p('<b style="color:%s">production</b> — September full pass' % RED, 26)
    body += sd.flow(prod, size=24)
    body += sd.p('<b style="color:%s">is2_v1</b> — staged on lxplus, not submitted' % GREEN, 26)
    body += sd.flow(is2, size=24)
    body += sd.callout('The gold box is the one the re-pass did not change, and it is the problem of slide 11: the '
                       'plausibility cut is written in <i>raw</i> tan, and the in-situ bundles change what raw tan means.',
                       GOLD, 25)
    D.slide('setup', body, f'''
<p><b>The package.</b> lxplus <code>~/sept26_stage2_is2_v1</code> (copy <code>/media/dylan/data/x17/sept26_prelim/pkg_is2_v1</code>): 6466 jobs, 293 sub-runs, arms A and C, built from afa9fcb with a clean tree, ≈ 4300 CPU-h, ~14 GB to <code>/eos/user/d/dneff/x17/sept26_fullpass_is2_v1</code>. Stage 3 writes <code>stage3_is2_v1</code>, never the production tables.</p>
<p><b>What does not ride it.</b> T1's two-track chain (x/y pairing, rescue floor, joint fit; branch <code>two-track-joint-fit</code>) and T3's depth-grid fix for late tracks. OCTOBER_2026 §4 plans one combined re-pass (O4) after those. is2_v1 is therefore an interim, angle-only pass for A and C.</p>
<p>run_126 has no k in production either (k_arm never certified its one sub-run), so its angles stay null in both.</p>''',
            short='What changes')


def s_v(D, o):
    v = LOGGED['v']
    rows = []
    for arm in 'AC':
        rows += [(f'{arm} bench fit', v['bench'][arm], GREY, f'June bench, fitted together with the kernel ({arm}).'),
                 (f'{arm} in-situ, A–C line', v['geom'][arm], GREEN, 'free fits against the cosmic A–C line, x plane (§8).'),
                 (f'{arm} production', v['prior'], RED, 'the Magboltz prior that make_bundle substitutes.')]
    left = sd.hbars(rows, 46, width=640, h=34, label_w=330, fmt=lambda x: f'{x:.1f} µm/ns', size=24)
    kA, kC = v['prior'] / v['bench']['A'], v['prior'] / v['bench']['C']
    right = sd.col(
        sd.p('Raw tan is w / v. Putting the 42.6 prior in place of the speed the kernel was fitted with shrinks '
             f'every raw tan by v_true/42.6, so production needs k ≈ {kA:.2f} (A) and {kC:.2f} (C) just to undo it.', 26),
        sd.p('Production\'s capsule k (A 1.27, C 1.62) is mostly that factor. The rest is the capsule '
             'estimator\'s own bias, which is why production\'s delivered tans are still off on cosmics (next slide).', 26),
        sd.callout('The in-situ bundles put the geometric v back (A 38.0, C 28.7), so their raw tan is already close '
                   'to true, and k drops to ≈ 0.85–0.98.', GREEN, 25), gap=22, w=640)
    body = sd.title('Production\'s angle error is a substituted drift speed, not physics',
                    'Drift speed per chamber: what the kernel was fitted with, what the cosmic A–C line measures, '
                    'and what production uses.')
    body += sd.row(left, right, gap=56)
    D.slide('why-v', body, f'''
<p>Established 6 Oct (§8): free fits against the A–C line give v = 37.5 (x) / 35.7 (y) for A and 25.8 / 25.4 for C, i.e. the bench values. The fit finds its χ² minimum, but under the production bundle that minimum is off the truth by 28–220 scaled χ² units: a systematic mismatch, not noise.</p>
<p>The ref-pinned in-situ refit (v 39.2) is NOT the fix: it is the known χ²(v)-valley bias (ANALYSIS_STATE S8). Never use the ref-pinned v. Source: {LOGGED["v"]["src"]}.</p>''',
            short='Why v')


def s_closure(D, o):
    panels = []
    for arm in 'AC':
        for ax in 'xy':
            P = sd.Plot(800, 330, x=(0.0, 0.62), y=(0.8, 1.4), ylabel=f'{arm} {ax}: delivered / true',
                        xlabel='|true tan|' if arm == 'C' else '', margin=(18, 24, 70 if arm == 'C' else 34, 104))
            P.xticks([(t, f'{t:.1f}') for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
            P.yticks([(t, f'{t:.1f}') for t in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)])
            P.band([0, 0.62], [0.97, 0.97], [1.03, 1.03], GREEN, 0.10, tip='±3 %')
            P.hline(1.0, MUT, '6 5', 1.5)
            for chain, col in (('production', C_PROD), ('is2_v1', C_IS2)):
                c = _cc(o, arm, ax, chain)
                c = c[c.lo >= 0.08]
                xs = ((c.lo + c.hi) / 2).tolist()
                tips = [f'{chain}, {arm} {ax}, |tan| {r.lo:.2f}–{r.hi:.2f}: {r.ratio:.3f} ± {r.ratio_err:.3f}, '
                        f'σ_tan {r.sigma:.3f}, n {r.n} (k {r.k:.3f})' for r in c.itertuples()]
                P.line(xs, c.ratio.clip(0.8, 1.4).tolist(), col, 3.5, tips=tips)
            panels.append(P.svg(f'closure {arm}{ax}'))
    grid = (f'<div style="display:grid;grid-template-columns:800px 800px;gap:8px 48px">{"".join(panels)}</div>')
    c = o['cc'][o['cc'].lo >= 0.08]
    ia = c[(c.arm == 'A') & (c.chain == 'is2_v1')].ratio
    pa = c[(c.arm == 'A') & (c.chain == 'production')].ratio
    body = sd.title(f'On cosmic truth is2 delivers A to {100 * (ia.min() - 1):+.0f}/{100 * (ia.max() - 1):+.0f} %; '
                    f'production reads A {100 * (pa.min() - 1):.0f}–{100 * (pa.max() - 1):.0f} % steep',
                    'Held-out run_149 A–C tracks: tan as delivered (raw × each chain\'s k on run_145) ÷ the A–C line.')
    body += sd.legend([('production (v 42.6, capsule k)', C_PROD), ('is2_v1 (in-situ, k = muon norm)', C_IS2),
                       ('±3 %', GREEN, 'box')], 22)
    body += grid
    D.slide('closure', body, '''
<p><b>How.</b> <code>repass_readiness.py cosmic</code>: the truth table of §8 (815 A–C tracks per arm from 14 run_149 sub-runs; the third used to set kw is excluded), joined to the local reconstructions with the production bundle at seeder 3 (<code>reco_prod_s3_A</code>, <code>reco_prod_C</code>) and the is2 bundles (<code>reco_is2_s3_*</code>). Production is multiplied by run_145's k_arm (A 1.266, C 1.616), is2 by kcal_is2_v1's run_145 k (A 0.976, C 0.962), so both are what the campaign would deliver for that run.</p>
<p><b>Reading.</b> A is flat to −4/+3 % between |tan| 0.08 and 0.6 in both views. C y is flat to 2 %. <b>C x is not linear</b>: 1.03 at 0.1 falling to 0.89 at 0.5 (a 14 % span), as it was before the kernel swap (§9: C is the problem chamber, its 8–13 % core tails are not from the kernel). Production's error is large and angle-dependent on both arms.</p>
<p>Below |tan| 0.08 (not drawn) both chains read 13–67 % high: near-normal fits are pushed away from zero, an additive offset of ~0.01–0.04 in tan, not a scale (§7d). See the resolution slide.</p>''',
            short='Cosmic closure')


def s_resolution(D, o):
    big = o['cc'][o['cc'].lo == 0.45].groupby(['arm', 'chain']).sigma.mean()
    fa, fc = big[('A', 'production')] / big[('A', 'is2_v1')], big[('C', 'production')] / big[('C', 'is2_v1')]
    P = sd.Plot(940, 600, x=(0.0, 0.62), y=(0, 0.17), xlabel='|true tan|', ylabel='σ_tan (MAD of delivered − true)')
    P.xticks([(t, f'{t:.1f}') for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
    P.yticks([(t, f'{t:.2f}') for t in (0, 0.04, 0.08, 0.12, 0.16)])
    for arm, dash in (('A', None), ('C', '10 6')):
        for chain, col in (('production', C_PROD), ('is2_v1', C_IS2)):
            c = pd.concat([_cc(o, arm, 'x', chain), _cc(o, arm, 'y', chain)]).groupby(['lo', 'hi'], as_index=False).agg(
                sigma=('sigma', 'mean'), n=('n', 'sum'))
            xs = ((c.lo + c.hi) / 2).tolist()
            tips = [f'{chain} {arm}, |tan| {r.lo:.2f}–{r.hi:.2f}: σ {r.sigma:.3f} (mean of x, y), n {r.n}'
                    for r in c.itertuples()]
            P.line(xs, c.sigma.tolist(), col, 3.5, dash=dash, tips=tips, marker='circle' if arm == 'A' else 'open')
    nn = LOGGED['nearnormal']
    s = o['seed'].set_index('arm')
    right = sd.col(
        sd.legend([('production', C_PROD), ('is2_v1', C_IS2), ('A solid, C dashed', MUT, 'dash')], 22),
        sd.p(f'is2: σ_tan ≈ 0.02–0.04 in the core; at |tan| 0.45–0.6 A goes {big[("A", "production")]:.3f} → '
             f'{big[("A", "is2_v1")]:.3f}, C {big[("C", "production")]:.3f} → {big[("C", "is2_v1")]:.3f}.', 25),
        sd.p(f'<b>Head-on was the seeder, not the fit.</b> With the 3-strip minimum, A x tracks at true |tan| '
             f'0.02–0.08 go {nn["A_x_n"][0]} → {nn["A_x_n"][1]} and σ_tan {nn["A_x_sigma"][0]} → {nn["A_x_sigma"][1]} '
             f'(C {nn["C_x_sigma"][0]} → {nn["C_x_sigma"][1]}).', 25),
        sd.callout(f'Still poor: |tan| < 0.02 (σ ≈ {nn["below002"]}), and a positive bias below 0.08 '
                   f'(both chains). On beam, the seeder adds near-normal tracks ×{s.loc["A", "nearnormal_m3"] / s.loc["A", "nearnormal_prod"]:.1f} (A).',
                   GOLD, 24), gap=20, w=600)
    body = sd.title(f'Large-angle resolution improves ×{fa:.1f} (A) and ×{fc:.1f} (C), and head-on tracks are back',
                    'Same held-out cosmic tracks: σ_tan per |tan| bin, x and y averaged.')
    body += sd.row(P.svg('resolution'), right, gap=48)
    D.slide('resolution', body, f'''
<p>Head-on numbers: {nn["src"]}. One-track A–C pairs on the same triggers go {nn["ac_pairs"][0]} → {nn["ac_pairs"][1]} with the 3-strip seeder. Beam (run_145, §10a): near-normal gated tracks A {s.loc["A", "nearnormal_prod"]:,} → {s.loc["A", "nearnormal_m3"]:,}, C {s.loc["C", "nearnormal_prod"]:,} → {s.loc["C", "nearnormal_m3"]:,}.</p>
<p>Still open near normal: <code>slope_reliable</code> (|raw tan| ≥ 0.08) is set on the reconstructed tan; the fit pushes near-normal tracks away from zero, so 64–90 % of truly near-normal tracks are flagged reliable (§7d). <code>tan_err</code> is a constant (0.022/0.026): pulls are 1.3 in the core and 9–16 near normal. Both feed T1 and <code>det_a_intra</code>.</p>''',
            short='Resolution')


def s_truth(D, o):
    pn = o['norm'][(o['norm'].sep_max == 10) & (o['norm'].arm == 'A') & (o['norm'].view == 'x')].iloc[0]
    rows = [(a, (float(pn.pooled) if a.startswith('In-beam') else b), c,
             (f'A–C muons in beam runs, 20–80 ms, gas-normalised to run_145, pooled: {pn.pooled:.3f} ± {pn.pooled_err:.3f} '
              f'(n {int(pn.pooled_n)}); beam-off run_149 {pn.run149:.3f}. §13.' if a.startswith('In-beam') else d))
            for a, b, c, d in LOGGED['estimators']['rows']]
    bars = sd.hbars(rows, 1.4, width=480, h=40, label_w=540, fmt=lambda x: f'{x:.2f}', size=24)
    right = sd.col(
        sd.p('Each bar: true tan ÷ production raw tan for chamber A, by a different estimator.', 23),
        sd.p('<b style="color:%s">Muons</b> agree wherever they are measured: on cosmics, at the wall, inside '
             'beam runs.' % GREEN, 23),
        sd.p('<b style="color:%s">The wall</b> reads low on beam: few-MeV electrons scatter between gap and wall. '
             'Geant4 with a <i>perfect</i> reconstruction does the same.' % ORANGE, 23),
        sd.p('<b style="color:%s">The capsule k</b> assumes a point source on the axis; the real population is '
             'not that (D_eff ≈ 330 mm, not 234.6).' % RED, 23), gap=14, w=500)
    body = sd.title('Only muons are angle truth for beam; the wall and the capsule measure the population',
                    'Chamber A\'s beam angle scale from six estimators (§10c–10g, §13). Hover a bar for its sample.')
    body += sd.row(bars, right, gap=40)
    D.slide('truth', body, f'''
<p>This is the chain that closed 7 Oct: (1) cosmics read the same at A's wall as on the A–C line, so the wall survey and the A–C truth are consistent; (2) A–C muons inside beam runs read the cosmic scale within 2–5 % with 34 % less gain and full beam-on noise, so the beam environment is not the cause; (3) Geant4 with an ideal reconstruction gives the beam-electron wall reading (0.59 of the gap angle, energy-dependent).</p>
<p>Consequence: neither the wall nor the capsule band is beam angle truth without a forward model. The in-beam muons are. Source: {LOGGED["estimators"]["src"]}.</p>''',
            short='Beam truth')


def s_digi(D, o):
    L = LOGGED['digi']
    P = sd.Plot(980, 600, x=(0.15, 0.55), y=(0.90, 1.02), xlabel='|tan| band (centre)',
                ylabel='x: reco ÷ ideal gap line')
    P.xticks([(x, b) for x, b in zip(L['x'], L['bins'])])
    P.yticks([(t, f'{t:.2f}') for t in (0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02)])
    for arm, mk in (('A', 'circle'), ('C', 'open')):
        P.line(L['x'], L[f'{arm}_mu'], C_MU, 3.5, dash='8 6', marker=mk,
               tips=[f'{arm}, synthetic straight muons through the digitiser: {v:.3f} ({b})' for v, b in zip(L[f'{arm}_mu'], L['bins'])])
        P.line(L['x'], L[f'{arm}_e'], C_SIM, 3.5, marker=mk,
               tips=[f'{arm}, Geant4 beam-capture electrons through the digitiser: {v:.3f} ({b})' for v, b in zip(L[f'{arm}_e'], L['bins'])])
    P.text(0.52, 0.985, 'C', 24, INK)
    P.text(0.52, 0.948, 'A', 24, INK)
    right = sd.col(
        sd.legend([('muons (gate)', C_MU, 'dash'), ('G4 beam electrons', C_SIM)], 23),
        sd.p('Geant4 beam-capture electrons are digitised into real quiet run_145 waveforms and reconstructed by '
             'the unchanged production code (' + sd.term('g4_digi', G['digi']) + ').', 25),
        sd.callout('Against their own ideal gap line, electrons reconstruct like muons, within 1–2 %. A muon '
                   'calibration transfers to electrons.', C_SIM, 25),
        sd.p('What it cannot test: a chamber that differs from the fit model. The digitiser\'s response <i>is</i> the '
             'fit model; that part is calibrated on cosmics.', 23, MUT), gap=20, w=560)
    body = sd.title('Electrons reconstruct like muons: the reconstruction does not compress beam angles',
                    'Digitised Geant4 through the production reco (is2 bundles): x reco ÷ ideal line, per |tan| band.')
    body += sd.row(P.svg('digitiser'), right, gap=40)
    D.slide('digi', body, f'''
<p>Source: {L["src"]}; HANDOFF_TRACKING §13. This overturned §12's "the reco compresses beam electrons by 20–27 %": most of the falling capsule response was the estimator on this population (the reco-charge window selects angle-correlated tracks; plus angular spread and failed fits). Even the ideal line, in k_arm's selection, falls to 0.87 on A.</p>
<p>The muon gate itself reads 0.98 (A x, = 1/kw_x) easing to 0.95 at |tan| 0.5, and 18–23 % wrong-sign x fits at |tan| > 0.35, in sim and data alike.</p>''',
            short='Electrons')


def s_muons(D, o):
    n = o['norm']
    P = sd.Plot(980, 600, x=(0.5, 4.5), y=(0.88, 1.06), ylabel='true ÷ is2 (norm)')
    P.xticks([(i + 1, f'{r.arm} {r.view}') for i, r in enumerate(n[n.sep_max == 10].itertuples())])
    P.yticks([(t, f'{t:.2f}') for t in (0.88, 0.92, 0.96, 1.00, 1.04)])
    P.hline(1.0, MUT, '6 5', 1.5, label='is2 exactly right', anchor='end')
    for sep, dx, col, mk in ((10, -0.1, C_MU, 'circle'), (6, 0.1, GREY, 'open')):
        s = n[n.sep_max == sep].reset_index(drop=True)
        for i, r in s.iterrows():
            x = i + 1 + dx
            P.raw(sd.line(P.X(x), P.Y(r.norm - r.norm_err), P.X(x), P.Y(r.norm + r.norm_err), col, 3))
        P.points([i + 1 + dx for i in range(len(s))], s.norm.tolist(), col, r=9, marker=mk,
                 tips=[f'{r.arm} {r.view}, sep < {sep} mm: {r.norm:.3f} ± {r.norm_err:.3f}; pooled n {int(r.pooled_n)}; '
                       f'beam-off run_149 {r.run149:.3f}' for r in s.itertuples()])
    kA = json.loads((KCAL / 'k_arm_run_145.json').read_text())['arms']
    right = sd.col(
        sd.legend([('sep < 10 mm', C_MU, 'dot'), ('sep < 6 mm', GREY, 'dot')], 23),
        sd.p('Cosmic muons that cross A and C during beam runs (20–80 ms after the flash), each scaled to run_145\'s '
             'gas with the capsule band ratio, then pooled over 36 runs.', 24),
        sd.callout(f'is2 reads them right to 1–4 %, tans slightly too large if anything. The re-pass normalises with '
                   f'the mean of x and y: A {kA["A"]["norm"]:.3f}, C {kA["C"]["norm"]:.3f} '
                   f'(± {kA["A"]["norm_err"]:.3f} / {kA["C"]["norm_err"]:.3f}).', C_MU, 24),
        sd.p(f'The x–y difference (A {kA["A"]["norm_view_spread"]:.3f}, C {kA["C"]["norm_view_spread"]:.3f}) '
             'is carried as a systematic: one k per arm and run.', 23, MUT), gap=18, w=560)
    body = sd.title('In-beam muons, the only beam truth: is2 is right to 1–4 %',
                    'A–C through-going muons in beam runs, gas-normalised to run_145, relative to run_149 (beam off).')
    body += sd.row(P.svg('in-beam muons'), right, gap=40)
    D.slide('muons', body, '''
<p><b>How</b> (<code>inbeam_through_goers.py pooled</code> → <code>pooled_norm.csv</code>): one gated track each in A and C, lines within sep, joined line > 60 mm from the axis, 20–80 ms. Each muon's production raw tan is scaled by k_run/k_145 (the capsule band ratio), then true/raw is taken relative to run_149. That ratio is what is2 reads on run_145-period muons.</p>
<p><b>The step this leaves inferred.</b> The muons were reconstructed with the <i>production</i> bundle; the norm assumes is2 raw = production raw × a constant. On cosmics that holds to the closure slide's precision, not exactly (production's response falls with angle, is2's does not). The re-pass itself contains these muons reconstructed with is2, so the norm can be re-derived directly from its output at stage 3, which costs nothing. Errors are per bin; the sep cut interacts with the scale, so only equal-cut comparisons mean anything.</p>''',
            short='In-beam muons')


def s_gas(D, o):
    k = o['k'].dropna(subset=['A'])
    k = k.reset_index(drop=True)
    P = sd.Plot(1100, 560, x=(-1, len(k)), y=(0.82, 1.0), xlabel='production run', ylabel='is2_v1 k  (tan = k × raw)')
    lab = [(i, str(r)) for i, r in enumerate(k.run) if i % 3 == 0]
    P.xticks(lab)
    P.yticks([(t, f'{t:.2f}') for t in (0.82, 0.86, 0.90, 0.94, 0.98)])
    for arm, col in (('A', BLUE), ('C', ORANGE)):
        P.line(list(range(len(k))), k[arm].tolist(), col, 3,
               tips=[f'run_{r.run}: k_{arm} {getattr(r, arm):.3f} (gas ratio {getattr(r, "g" + arm):.3f})' for r in k.itertuples()])
    i145 = int(k.index[k.run == 145][0])
    P.vline(i145, MUT, '4 5', 1.5, label='run_145 (reference)')
    spr = {a: 100 * (k[a].max() / k[a].min() - 1) for a in 'AC'}
    right = sd.col(
        sd.legend([('A', BLUE), ('C', ORANGE)], 23),
        sd.p(f'The scale moves with the gas: across the campaign k spans {spr["A"]:.0f} % in A and {spr["C"]:.0f} % in C, '
             'peaking in the 48-hour block of runs 128–147.', 24),
        sd.p('Only the capsule band\'s <i>ratio</i> between runs is used. In-beam muons confirm it where they can: '
             'C\'s 128–147 / 150–162 ratio is 1.08 ± 0.05 in muons, 1.09 in the band.', 24),
        sd.callout('B and D get no k: their angles stay null in the re-pass rather than borrowing one.', GOLD, 23),
        gap=18, w=500)
    body = sd.title(f'Per-run k follows the gas: C moves {spr["C"]:.0f} % across the campaign, A {spr["A"]:.0f} %',
                    'kcal_is2_v1: k = muon norm × capsule band(run) / band(run_145), one value per arm and run.')
    body += sd.row(P.svg('per-run k'), right, gap=40)
    D.slide('gas', body, '''
<p><code>sept26_prelim_analysis/k_insitu.py --version is2_v1</code> → <code>/media/dylan/data/x17/sept26_prelim/kcal_is2_v1/</code> (36 runs; run_126 empty, as in production). Production's own band varies per run by A ±1.2 %, C ±3.8 %, D ±4.3 %, coherently (§13).</p>
<p>The band is a source-distribution estimator as well as a scale one (next slide), so using its run-to-run ratio assumes the source distribution is stable between runs. The muon ratio check above supports that at ± 5 %; per-period muon statistics are thin (n ≈ 50–250).</p>''',
            short='Per-run k')


def s_residual(D, o):
    L = LOGGED['source_width']
    P = sd.Plot(940, 580, x=(0, 28), y=(0.98, 1.16), xlabel='true source rms at the capsule plane (mm), simulation',
                ylabel='capsule band, x (sim reco, is2)')
    P.xticks([(t, str(t)) for t in (0, 5, 10, 15, 20, 25)])
    P.yticks([(t, f'{t:.2f}') for t in (1.00, 1.04, 1.08, 1.12, 1.16)])
    for arm, col in (('A', BLUE), ('C', ORANGE)):
        P.line(L['rms'], L[arm], col, 3.5, tips=[f'{arm} sim band at source rms {r} mm: {v:.3f}' for r, v in zip(L['rms'], L[arm])])
        d, s = L['core'][arm]
        P.hline(d, col, '8 6', 2, label=f'{arm} data (pointing core) {d:.3f}', anchor='start',
                tip=f'{arm} data band with |miss| < 40 mm: {d:.3f}; full sim, same cut: {s:.3f}')
    right = sd.col(
        sd.p('With the true source position held fixed, the simulated band reads 1.00: the reconstruction is right. '
             'The band then rises with the width of the source.', 24),
        sd.p(f'Data sit {100 * (L["core"]["A"][0] / L["core"]["A"][1] - 1):.0f} % (A) and '
             f'{100 * (L["core"]["C"][0] / L["core"]["C"][1] - 1):.0f} % (C) above the full simulation. The muons '
             'say the scale is right, so the excess belongs to the source and population model.', 24),
        sd.callout(f'<b>Open: y.</b> The y band residual is ~3× x (data A {L["y"]["A"][0]:.2f} vs sim {L["y"]["A"][1]:.2f}; '
                   f'C {L["y"]["C"][0]:.2f} vs {L["y"]["C"][1]:.2f}). Likely an under-modelled source along the '
                   'capsule\'s long axis. In-beam muons read y like x.', GOLD, 23), gap=18, w=600)
    body = sd.title('The capsule band\'s data/sim gap is the source model, not the angle scale',
                    'Simulated capsule band against the true source width; data in the pointing core (|miss| < 40 mm).')
    body += sd.row(P.svg('band vs source'), right, gap=40)
    D.slide('residual', body, f'''
<p>Source: {L["src"]}. Ruled out as causes of the residual (§13): time since the flash (no trend), an excess non-pointing population (the data's capsule-plane miss is narrower than the sim's), charge (flat across quintiles), the kernel (C's ~10 % is the same under r06-det7 and lp: production-bundle reruns give data/sim A {LOGGED["fp145"]["A"]:.3f}, C {LOGGED["fp145"]["C"]:.3f}).</p>
<p>A chamber stand-off common to both arms (~15–20 mm on D = 234.6) would also fit x alone; it would hit y equally, and y is 3× larger, so it is disfavoured but not excluded.</p>''',
            short='Band residual')


def _scan_plot(o):
    sc = o['scan']
    P = sd.Plot(640, 330, x=(0.5, 2.0), y=(8, 18), xlabel='true-tan reach of the cut (0.6…1.2 raw × k)',
                ylabel='candidates (k)', margin=(14, 20, 70, 84))
    P.xticks([(t, f'{t:.1f}') for t in (0.5, 1.0, 1.5, 2.0)])
    P.yticks([(t, str(t)) for t in (8, 10, 12, 14, 16, 18)])
    for arm, col in (('A', BLUE), ('C', ORANGE)):
        for chain, dash, mk in (('production', '8 6', 'open'), ('is2_v1', None, 'circle')):
            q = sc[(sc.arm == arm) & (sc.chain == chain)].sort_values('tan_max_raw')
            P.line(q.true_tan_reach.tolist(), (q.candidates / 1000).tolist(), col, 3, dash=dash, marker=mk,
                   tips=[f'{arm} {chain}: TAN_MAX {r.tan_max_raw:.1f} raw = {r.true_tan_reach:.2f} true → '
                         f'{r.candidates:,} candidates passing everything else' for r in q.itertuples()])
    P.text(1.95, 17.6, 'A is2', 19, BLUE, 'end')
    P.text(1.95, 14.0, 'C is2', 19, ORANGE, 'end')
    P.text(1.95, 8.6, 'dashed: production', 19, MUT, 'end')
    return P.svg('candidates vs reach')


def s_gate(D, o):
    g = o['gate'].set_index(['arm', 'chain'])
    sc = o['scint'].set_index(['arm', 'chain'])
    rows = []
    for arm in 'AC':
        for chain, col in (('production', C_PROD), ('is2_v1', C_IS2)):
            r = g.loc[(arm, chain)]
            rows.append((f'{arm} {chain}', float(r.true_tan_reach), col,
                         f'TAN_MAX 0.6 raw × k {r.k:.3f} = {r.true_tan_reach:.2f} true. Quality-OK candidates failing '
                         f'only on tan: {int(r.fail_tan_only):,} of {int(r.quality_ok):,} ({100 * r.fail_tan_frac:.0f} %).'))
    bars = sd.hbars(rows, 1.05, width=420, h=34, label_w=260, fmt=lambda x: f'|tan| < {x:.2f}', size=24)
    trs = []
    for arm in 'AC':
        trs.append([arm] + [f'{int(sc.loc[(arm, c), "gated"]):,}' for c in ('production', 'm3', 'is2_v1')]
                   + [f'{int(sc.loc[(arm, c), "wall_excess"]):,}' for c in ('production', 'm3', 'is2_v1')])
    tab = sd.table(['arm', 'gated prod', 'm3', 'is2', 'confirmed prod', 'm3', 'is2'], trs, size=22)
    lost = 1 - sc.loc[('C', 'is2_v1'), 'wall_excess'] / sc.loc[('C', 'm3'), 'wall_excess']
    left = sd.col(sd.p('True-angle reach of the plausibility cut', 24, INK, 600), bars,
                  sd.p(sd.term('Scintillator-confirmed', G['confirmed']) + ' tracks, run_145 stat090_0000 (' +
                       sd.term('m3', G['m3']) + ' = production bundle with the 3-strip seeder):', 23), tab, gap=16, w=900)
    right = sd.col(
        sd.p(sd.term('TAN_MAX', G['tanmax']) + ' = 0.6 is written in raw tan. On the v = 42.6 bundles that meant '
             '0.76 (A) and 0.97 (C) true; on the in-situ bundles, 0.58.', 24),
        sd.p(f'C feels it most: {100 * g.loc[("C", "is2_v1"), "fail_tan_frac"]:.0f} % of its quality-OK candidates now '
             f'fail on angle alone. Gated tracks fall {100 * (1 - sc.loc[("C", "is2_v1"), "gated"] / sc.loc[("C", "m3"), "gated"]):.0f} % '
             f'against m3, confirmed tracks {100 * lost:.0f} %.', 24),
        sd.callout('The cut is now what the bench meant (≈ 0.6 true), but the acceptance in angle shrinks, and '
                   'nothing has tested the reconstruction above |tan| 0.6. Decide the reach, then validate it.', GOLD, 23),
        _scan_plot(o), gap=18, w=640)
    body = sd.title('A fixed raw-tan cut in the fit halves C\'s angular reach under the in-situ bundles',
                    'wft.reco plausibility (|raw tan| < 0.6) in true-tan units, and what it does to tracks on the smoke sub-run.')
    body += sd.row(left, right, gap=48)
    D.slide('gate', body, '''
<p><b>Found while writing this note</b> (<code>repass_readiness.py beam</code>). The smoke test compared condor with the local is2 reco (identical); nobody had compared is2's yield with production's. The seeder test of §10a (+45/52 % confirmed) used the <i>production</i> bundle.</p>
<p><b>Why it bites.</b> <code>wft.reco._candidate_score</code>: plausible = 250 ≤ q_uend ≤ 1100 ns and |fit.tan_theta| < 0.6, where tan_theta is the raw tan of the bundle. An implausible fit loses the candidate ranking (key = (plausible, Δχ²)) and fails the x/y pairing gate. Raw tan ∝ 1/v, so the cut's true-angle meaning is 0.6 × k.</p>
<p><b>Are the lost tracks real?</b> Partly. Their arm-level scintillator coincidence (<code>coinc_this_arm</code>) is 34 % against 40–64 % for gated 0.3–0.6 tracks, and 26–40 % have t0 > 300 ns (the late class with unreliable geometry, T3). Large-angle tracks also miss the wall more often, so the wall test is partly blind to them. Candidates passing everything but the cut, raw 0.6 / 0.8 / 1.0 / 1.2: C is2 10,159 / 12,121 / 14,077 / 14,801.</p>
<p><b>Options.</b> (a) keep 0.6 raw ≈ 0.58 true and record it as the acceptance (the cosmic truth covers only |tan| < 0.6); (b) make TAN_MAX a true-angle cut carried by the bundle (e.g. 0.9–1.0) and validate 0.6–1.0 with digitised Geant4 muons/electrons at those angles plus scintillator confirmation, then rerun the smoke sub-run. Other raw-tan constants: <code>TAN_MIN_SLOPE</code> 0.08 (slope_reliable) and <code>FLOOR_TAN</code> 0.018 shift by the same factor, which matters less.</p>''',
            short='Raw-tan cut')


def s_shift(D, o):
    s = o['shift']
    P = sd.Plot(980, 580, x=(0.0, 0.62), y=(0.6, 1.1), xlabel='|production tan|',
                ylabel='is2_v1 tan ÷ production tan, same track')
    P.xticks([(t, f'{t:.1f}') for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
    P.yticks([(t, f'{t:.1f}') for t in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)])
    P.hline(1.0, MUT, '6 5', 1.5)
    for arm, col in (('A', BLUE), ('C', ORANGE)):
        for ax, dash, mk in (('x', None, 'circle'), ('y', '8 6', 'open')):
            q = s[(s.arm == arm) & (s.axis == ax) & (s.n >= 30)].sort_values('lo')
            xs = ((q.lo + q.hi) / 2).tolist()
            P.line(xs, q.is2_over_prod.tolist(), col, 3.5, dash=dash, marker=mk,
                   tips=[f'{arm} {ax}, |prod tan| {r.lo:.2f}–{r.hi:.2f}: {r.is2_over_prod:.3f}, n {r.n}' for r in q.itertuples()])
    right = sd.col(
        sd.legend([('A', BLUE), ('C', ORANGE), ('x solid, y dashed', MUT, 'dash')], 23),
        sd.p('What the re-pass does to the same particle (p0 within 2 mm in both planes), run_145 smoke sub-run.', 24),
        sd.p('A: every tan shrinks by ≈ 15 % (x) and 12 % (y), flat in angle. C x: ≈ 10 % shallower. C y: unchanged.', 24),
        sd.callout('Every opening angle, the 170° back-to-back cut, T1\'s F thresholds and slope_reliable sit on these '
                   'numbers. A–B / A–D / C–B / C–D pairs disappear from is2-only tables, since B and D have no k.', GOLD, 23),
        gap=18, w=560)
    body = sd.title('For the same track the re-pass makes A\'s tans 12–15 % shallower, C\'s up to 10 %',
                    'Matched gated tracks, run_145 stat090_0000: is2_v1 delivered tan ÷ production delivered tan.')
    body += sd.row(P.svg('shift'), right, gap=40)
    D.slide('shift', body, '''
<p>Consistent with the cosmic closure: production delivered A ~14 % steep and C x ~5–20 % steep, C y about right. Bins with fewer than 30 matched tracks are not drawn.</p>
<p>A single sub-run holds only ~20 two-arm inter-chamber pairs, so the effect on opening-angle spectra can only be measured on the full pass.</p>''',
            short='Per-track shift')


def s_problems(D, o):
    W = dict(stop=RED, before=GOLD, after=GREEN, open=GREY)
    items = [
        ('stop', 'Raw-tan plausibility cut', 'stage 2', 'C reach 0.97 → 0.58 true; −10 % confirmed C tracks', 'slide 11'),
        ('before', 'is2 bundles never yield-tested on beam', 'stage 2', 'the §10a purity test used the production bundle', 'slide 11'),
        ('before', 'Min-3 re-pairs 5–13 % of busy events', 'stage 2', 'x/y pairing arbitration is T1\'s xy_pairing; not in is2_v1', '§10a'),
        ('open', 'Late tracks (t0 > 300 ns)', 'stage 2', '17 % of gated tracks have wrong geometry; 31 % of min-3 gains are late', 'T3 / O10'),
        ('open', 'C x non-linearity, C core tails', 'stage 2', '1.03 → 0.89 across |tan|; tails 8–13 %, not the kernel', '§9'),
        ('open', 'Near normal', 'stage 2 / flags', '|tan| < 0.02 σ 0.15; positive bias < 0.08; slope_reliable, tan_err', '§7d, §9'),
        ('after', 'Muon norm inferred via production', 'stage 3', 're-derive from the re-pass\'s own in-beam muons', 'slide 8'),
        ('after', 'x–y norm spread, per-period stats', 'stage 3', 'A 3 % x–y; n ≈ 50–250 muons per period', 'slide 8'),
        ('open', 'y band residual (3× x)', 'model', 'source extent along the capsule\'s long axis', 'slide 10'),
        ('open', 'B and D', 'all', 'no in-situ truth yet; angles null in is2_v1', 'next study'),
    ]
    head = ('<div style="display:grid;grid-template-columns:40px 470px 190px 1fr 170px;gap:0 20px;'
            'font-size:21px;color:%s;padding:0 0 8px 0;border-bottom:2px solid %s">'
            '<p></p><p>problem</p><p>where it acts</p><p>size / what is known</p><p>record</p></div>') % (MUT, sd.RULE)
    rows = ''.join(
        f'<div style="display:grid;grid-template-columns:40px 470px 190px 1fr 170px;gap:0 20px;align-items:center;'
        f'padding:9px 0;border-bottom:1px solid {sd.RULE}">'
        f'<div style="width:22px;height:22px;border-radius:50%;background:{W[w]}"></div>'
        f'<p style="font-size:24px;font-weight:600">{a}</p><p style="font-size:22px;color:{MUT}">{b}</p>'
        f'<p style="font-size:22px">{c}</p><p style="font-size:20px;color:{MUT}">{d}</p></div>'
        for w, a, b, c, d in items)
    body = sd.title('One item blocks the launch; the rest either ride a later pass or are free afterwards',
                    'Open problems in the A/C angle chain, by where a fix would have to act.')
    body += sd.legend([('blocks the launch', RED, 'dot'), ('cheap to settle first', GOLD, 'dot'),
                       ('free after the re-pass (stage 3)', GREEN, 'dot'), ('open, needs a later stage-2 pass', GREY, 'dot')], 22)
    body += f'<div style="display:flex;flex-direction:column">{head}{rows}</div>'
    D.slide('problems', body, '''
<p><b>The sorting rule.</b> Anything that changes stage 2 (the fit) costs a re-pass; anything in stage 3 (k, gates, flags) is a re-run of <code>campaign_tracks</code> over the stage-2 output, minutes to hours. The k itself (norm, gas ratio, x–y treatment) is all stage 3, so it can keep improving after launch.</p>
<p><b>The interim question.</b> is2_v1 does not contain T1's x/y pairing and two-track chain, nor a fix for late tracks; OCTOBER_2026 §4 wants one combined re-pass (O4). Compute is not a constraint, so an interim pass is cheap. The real cost is building downstream products twice. Label it interim, and keep the consumers (opening angle, 170° cut, T1 F) on production until both have been compared.</p>''',
            short='Open problems')


def s_studies(D, o):
    cards = [
        ('1 · Settle the plausibility cut', RED, 'hours',
         'Make TAN_MAX a true-angle cut carried by the bundle (or decide 0.58 is the acceptance). Validate |tan| 0.6–1.0: '
         'g4_digi muon guns at tan 0.6/0.8/1.0, plus scintillator confirmation on the smoke sub-run. Re-run the smoke.'),
        ('2 · Yield gate, real bundles', GOLD, '< 1 h',
         'seed_beam_test scint/compare with is2_A / is2_C on 2–3 sub-runs from other periods (run_86, run_110, run_156): '
         'confirmed tracks, near-normal yield, late fraction. Make it a standing smoke gate.'),
        ('3 · A pilot pass', GOLD, '~1 h on condor',
         'One sub-run per run (36 sub-runs, ~800 jobs) through the package. Gives per-run yields, k sanity and the in-beam '
         'muons reconstructed with is2 directly, before 6466 jobs.'),
        ('4 · Decide interim vs combined', GOLD, 'decision',
         'Either launch is2_v1 labelled interim, or first merge T1\'s xy_pairing (validated, opt-in) and run the combined '
         'split-ab contract on is2 bundles, then launch once.'),
        ('5 · Near-normal bias', GREY, '< 1 h',
         'Quantify the < 0.08 overread on the 180° back-to-back peak (cosmic pairs), as a stage-3 correction or a flag.'),
        ('6 · After launch, free', GREEN, 'stage 3',
         'Re-derive the muon norm from is2-reconstructed in-beam muons; binned response for C x; σ(|tan|) for tan_err.'),
    ]
    html = ''.join(
        f'<div style="background:{sd.CARD};border:1px solid {sd.RULE};border-top:6px solid {c};border-radius:14px;'
        f'padding:22px 24px;display:flex;flex-direction:column;gap:10px">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline"><p style="font-size:27px;font-weight:600">{t}</p>'
        f'<p style="font-size:21px;color:{MUT}">{cost}</p></div>'
        f'<p style="font-size:22px;line-height:1.35;color:{INK}">{d}</p></div>' for t, c, cost, d in cards)
    body = sd.title('Before 4300 CPU-hours: three small checks and one decision',
                    'In order. Items 1–3 use the existing smoke products and digitiser; none needs the full pass.')
    body += f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px">{html}</div>'
    D.slide('studies', body, '''
<p><b>Why a pilot.</b> The smoke test proved condor reproduces the local reco; it did not test yields against production, other run periods or the k tables. One sub-run per run catches a per-period surprise (gas, noise, the run_79 connector-8 mask on A) at 1/8 of the cost and gives the is2-reconstructed in-beam muons that the norm currently infers.</p>
<p><b>What is not on the list.</b> The y band residual (a simulation source-model question, no effect on stage 2), the late-t0 fix (T3, a separate study), and the C core tails: all real, none cheap, and none would change the decision to run an interim pass.</p>''',
            short='Small studies')


def s_close(D):
    items = [
        ('Angles above |tan| 0.6', 'No truth there: cosmic A–C covers |tan| < 0.6. Whatever TAN_MAX becomes, that range rests on simulation.'),
        ('Electrons ≠ muons in the real chamber', 'The digitiser tests noise, seeding and ionisation shape, not a chamber that differs from the fit model.'),
        ('A source-model error in y', 'The y band residual is 3× x; attributed to the source, not shown.'),
        ('Run-condition drift beyond gas', 'is2 was calibrated on run_149 (August). Kernel or diffusion changes would not be caught by a scalar k.'),
        ('B and D', 'No in-situ truth; null angles. Perpendicular-pair topologies are absent from is2 tables.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:13px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:26px;font-weight:600;width:400px">{a}</p>'
                    f'<p style="font-size:22px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    nxt = [('Decide the plausibility cut', 'Keep 0.58 true, or move to a bundle-carried true-angle cut and validate 0.6–1.0.'),
           ('Interim or combined pass', 'is2_v1 alone now, or after merging T1\'s xy_pairing.'),
           ('Then a pilot', 'One sub-run per run before 6466 jobs.'),
           ('B and D', 'Feasibility of in-situ truth for D (B–D lines) and of B as a hit-mode chamber: next study.')]
    nrows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:13px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:26px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:22px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in nxt)
    body = (f'<div style="display:flex;gap:72px">'
            f'<div style="flex:1.15;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:48px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:48px;font-weight:600">Decisions</h2>{nrows}</div></div>')
    D.slide('close', body, '''
<p>Entry points: <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> (§13 for the re-pass), the root <code>HANDOFF.md</code>, and <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code> for how this joins the two-track work. Regenerate this note with <code>ntof_cosmics/repass_readiness.py all</code> then <code>ntof_cosmics/make_insitu_deck.py</code>.</p>''',
            dark=True, short='Decisions')


def build(out: Path) -> Path:
    o = load()
    D = sd.Deck('A and C angles: in-situ calibration before the re-pass',
                'Chambers A and C on the in-situ (is2) angle calibration: how good it is, what is open, and what to check '
                'before the campaign re-pass.')
    for f in (s_cover, s_setup, s_v, s_closure, s_resolution, s_truth, s_digi, s_muons, s_gas, s_residual,
              s_gate, s_shift, s_problems, s_studies):
        f(D, o)
    s_close(D)
    g = o['gate'].set_index(['arm', 'chain'])
    meta = dict(title='A and C angles: in-situ calibration before the re-pass',
                summary=('The in-situ (is2) bundles close A within ±4 % on cosmic truth and read in-beam muons right to '
                         '1–4 %; production was 10–20 % steep. Before the is2_v1 re-pass: a raw-tan cut in the fit cuts '
                         f'C\'s angular reach from {g.loc[("C", "production"), "true_tan_reach"]:.2f} to '
                         f'{g.loc[("C", "is2_v1"), "true_tan_reach"]:.2f}; open problems and the small checks to do first.'),
                tags='X17,tracking,calibration', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by nTof_x17/ntof_cosmics/'
                                     'make_insitu_deck.py with slidedoc.py.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=RES / 'deck' / 'ac-insitu-angles.html')
    a = ap.parse_args()
    print('wrote', build(a.out))


if __name__ == '__main__':
    main()
