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
def s_cover(D, o, v):
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
        sd.bignum(f'×{_sum_ratio(v, "A", "confirmed_acc"):.2f} / ×{_sum_ratio(v, "C", "confirmed_acc"):.2f}',
                  'wall-confirmed tracks, is2_v2 ÷ production', DGREEN,
                  'A / C over the whole campaign, |true tan| < 0.6: the two-sided re-pass.',
                  tip='pilot_compare.py --version is2_v2: sums over run/arms both chains calibrate.'),
        sd.bignum(f'×{_offset(v, "A").s_best:.2f}', 'what the capsule asks of is2 angles', '#f0c060',
                  'on both arms, after the 9.5 mm source offset. The muons ask for ×1.00. Unresolved.',
                  tip=f'joint (s, dx) focus fit, 20–80 ms, wall fired, |tan| 0.1–0.5: A {_offset(v, "A").s_best:.2f}, '
                      f'C {_offset(v, "C").s_best:.2f}. Geant4 puts a correct scale at 0.975–1.025.')])
    body = (sd.kicker('Chambers A and C · in-situ angle calibration · updated 12 Oct 2026')
            + '<h1 style="font-size:72px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1660px">'
              'The is2_v2 re-pass confirms 1.5–1.8× production\'s tracks; muons and the capsule '
              'still disagree on the angle scale by 12 %</h1>'
            + '<div style="flex:1"></div>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, '''
<p><b>Update, 12 Oct.</b> Slides 13–21 are new: the is2_v2 full pass against production, the capsule focus scan, the steep tracks, the population and offset tests of the focus scale, the muon-vs-capsule tension with what has been ruled out, the physical-kernel test, and the updated problems and next steps (HANDOFF_TRACKING §19–22). Slides 2–12 are the 8 Oct pre-launch review, kept as the record; slide 11's cut has since been resolved.</p>
<p><b>What this note was.</b> A review of the chamber A and C angle work on branch <code>beam-off-cosmics</code> (6–8 Oct), written before spending ~4300 CPU-h on the <code>is2_v1</code> campaign re-pass. It says how good the in-situ calibration is, which problems are still open, and which small checks are worth doing first. One new check, made for this note, found a cut in the fit whose meaning changes with the bundle (slide 11).</p>
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
    body = sd.title('A fixed raw-tan cut halved C\'s reach; fixed by TAN_MAX 1.0 in the fit and a true-angle cut',
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


# --------------------------------------------------------------------------- #
# is2_v2 full pass, the capsule focus and the scale tension (§20–22, 10-11/12)
# --------------------------------------------------------------------------- #
FP = RES / 'fullpass_is2_v2'
FOCUS = RES / 'capsule_focus'
FSPLIT = RES / 'focus_split'

#: §22's physical-kernel test (insitu_paired.py, 547 held-out run_149 events); the
#: CSVs live in ~/scratch/ntof_insitu, outside the repo, so the table is logged here.
RC = dict(src='§22 (recipe_rc.sh + insitu_paired.py; each label at its own TRAIN scale)', rows=[
    ('A · rcm3 (det3 constants)', '+0.0003 ± 0.0016', '−0.0020 ± 0.0012 (−7 %)', '−0.011 / −0.015', '−0.101 → −0.062', '−1.9 %'),
    ('C · rcm7 (det7 constants)', '+0.0007 ± 0.0016', '+0.0039 ± 0.0016 (+10 %)', '−0.017 / −0.025', '−0.107 → −0.060', '−0.7 %'),
    ('C · rcm6 (det6 constants)', '+0.0002 ± 0.0015', '+0.0006 ± 0.0021', '+0.002 / −0.021', 'unchanged', '+0.7 %')])

#: Geant4 calibration of the focus estimator (§21): sharpest s with the true / reco sim angles
G4FOCUS = dict(true=(0.975, 1.025), reco=(1.00, 1.025), floor=(15, 16))

#: capsule geometry used by stage 3 (beam axis to chamber), and the geometric x-reach of a capsule track
D_AXIS, TANX_GEOM = 234.6, 0.64


def load_v2() -> dict:
    o = dict(pc=pd.read_csv(FP / 'pilot_compare.csv'), cbt=pd.read_csv(FP / 'confirm_by_tan.csv'),
             st=pd.read_csv(FP / 'same_track_confirm.csv'),
             pc08=pd.read_csv(RES / 'fullpass_is2_v2_acc08' / 'pilot_compare.csv'))
    o['focus'] = pd.concat([pd.read_csv(FOCUS / f'focus_{c}.csv') for c in ('prod', 'is2_v2', 'is2_v2_acc08')])
    o['split'] = pd.read_csv(FSPLIT / 'focus_split_is2_v2.csv')
    o['off'] = pd.read_csv(FSPLIT / 'focus_offset_is2_v2.csv')
    return o


def _v2_ratio(o, arm, col='confirmed_acc_vs_prod'):
    pc = o['pc']
    q = pc[(pc.chain != 'prod') & (pc.arm == arm) & pc.calibrated]
    prod = pc[(pc.chain == 'prod') & (pc.arm == arm) & pc.calibrated]
    q = q[q.run.isin(prod.run)]
    return q.sort_values('run')


def _sum_ratio(o, arm, col):
    pc = o['pc']
    both = set(pc[(pc.chain == 'prod') & (pc.arm == arm) & pc.calibrated].run) & \
        set(pc[(pc.chain != 'prod') & (pc.arm == arm) & pc.calibrated].run)
    q = pc[(pc.arm == arm) & pc.run.isin(both)]
    return q[q.chain != 'prod'][col].sum() / q[q.chain == 'prod'][col].sum()


def _offset(o, who, lo=0.1, hi=0.5):
    f = o['off']
    return f[(f.who == who) & np.isclose(f.tan_lo, lo) & np.isclose(f.tan_hi, hi)].iloc[0]


def s_fullpass(D, o, v):
    rA, rC = _sum_ratio(v, 'A', 'confirmed_acc'), _sum_ratio(v, 'C', 'confirmed_acc')
    runs = sorted(set(_v2_ratio(v, 'A').run) | set(_v2_ratio(v, 'C').run), key=lambda r: int(r.split('_')[1]))
    nx = len(runs) + 1
    P = sd.Plot(1080, 600, x=(0, nx), y=(0.8, 2.1), ylabel='is2_v2 ÷ production, wall-confirmed tracks',
                xlabel='run (calibrated in both chains)', margin=(20, 24, 96, 104))
    P.yticks([(t, f'{t:.1f}') for t in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)])
    P.hline(1.0, MUT, '6 5', 1.5, label='production', where='below')
    ix = {r: i + 1 for i, r in enumerate(runs)}
    P.xticks([(ix[r], r.split('_')[1]) for r in runs[::4]])
    for arm, col, dx in (('A', BLUE, -0.18), ('C', ORANGE, 0.18)):
        q = _v2_ratio(v, arm)
        P.points([ix[r] + dx for r in q.run], q.confirmed_acc_vs_prod.tolist(), col, r=7,
                 tips=[f'{arm} {r.run}: confirmed in acceptance {int(r.confirmed_acc):,} = '
                       f'×{r.confirmed_acc_vs_prod:.2f} production; gated ×{r.gated_acc_vs_prod:.2f}; '
                       f'wall rate {r.wall_rate:.2f}' for r in q.itertuples()])
    P.text(nx - 0.5, 2.05, f'C ×{rC:.2f} overall', 24, ORANGE, 'end', 600)
    P.text(nx - 0.5, 1.3, f'A ×{rA:.2f} overall', 24, BLUE, 'end', 600)
    gA, gC = _sum_ratio(v, 'A', 'gated_acc'), _sum_ratio(v, 'C', 'gated_acc')
    st = v['st']
    right = sd.col(
        sd.p(f'Every run, both arms, above 1. Gated tracks in acceptance rise only ×{gA:.2f} (A) and ×{gC:.2f} (C), '
             'so the gain is purity as much as yield.', 24),
        sd.p('Two changes made it: in-situ bundles (the right v, so the right k) and a ' +
             sd.term('two-sided search', 'wft.reco tries both drift directions (t0 early and late) and keeps the better '
                     'χ². Fixes the mirror basins; ~20× CPU. §17–18.') + '. The pilot\'s A deficit at |tan| 0.3–0.6 '
             'disappeared with the second.', 24),
        sd.callout('Same tracks, is2_v2 ÷ production tan: A 0.86 x / 0.87 y, C 0.90 / 1.01. That is the k difference, '
                   'and it is what the rest of this note argues about.', GOLD, 23),
        gap=18, w=540)
    body = sd.title(f'The is2_v2 re-pass confirms ×{rA:.2f} (A) and ×{rC:.2f} (C) more tracks on the wall than production',
                    'Full campaign, 292 sub-runs, |true tan| < 0.6 in both chains: scintillator-confirmed tracks per run.')
    body += sd.row(P.svg('fullpass ratio'), right, gap=44)
    D.slide('fullpass', body, '''
<p><b>The pass.</b> is2_v2 = is2 bundles, seeder minimum 3, two-sided search, TAN_MAX 1.0 in the fit, stage 3 with <code>kcal_is2_v2</code> (= kcal_is2_v1 × A 1.009, C 1.003, the two-sided rescale from the cosmic closure) and a true-angle cut. Condor 4410513, 6,464 of 6,466 jobs; the two missing (run_104 stat090_0016 A/C) have empty decoded input and production has no A/C there either. Stage 3: 292 sub-runs, 15.7 M segments, 5.41 M gated at 0.6.</p>
<p><b>Ratios</b> are sums over run/arms that both chains calibrate (production has no k for 9 run/arms). Per |true tan| bin, is2_v2 is above production in all 24 bin/views up to 0.6. Where only one chain's direction confirms a matched track, is2_v2 wins A 12,458 : 6,114, C 7,832 : 3,227.</p>
<p>Report: <code>ntof_cosmics/results/fullpass_is2_v2/report.html</code> (<code>make_fullpass_report.py --version is2_v2</code>). Record: HANDOFF_TRACKING §20.</p>''',
            foot='pilot_compare.py --version is2_v2 · wall-confirmed = SiPM wall/plastic hit minus off-time control',
            short='is2_v2 vs production')


def _focus_curve(o, chain, arm, lo, null=False):
    f = o['focus']
    q = f[(f.chain == chain) & (f.arm == arm) & np.isclose(f.tan_lo, lo) & (f.null == null)]
    q = q[~np.isclose(q.tan_hi - q.tan_lo, 0.6) & ~np.isclose(q.tan_hi - q.tan_lo, 0.8)]
    return q.sort_values('scale')


def s_focus(D, v):
    panels = []
    for arm, col in (('A', BLUE), ('C', ORANGE)):
        P = sd.Plot(800, 520, x=(0.75, 1.35), y=(14, 34), xlabel='scale s applied to every x tan',
                    ylabel=f'{arm}: median miss at the beam axis [mm]' if arm == 'A' else '',
                    title=f'chamber {arm}, |tan| 0.2–0.3', margin=(44, 24, 92, 96))
        P.xticks([(t, f'{t:.2f}') for t in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)])
        P.yticks([(t, str(t)) for t in (15, 20, 25, 30)])
        P.band([G4FOCUS['true'][0], G4FOCUS['true'][1]], [14, 14], [34, 34], PURPLE, 0.10,
               tip='Geant4, true sim angles: the focus is sharpest at s 0.975–1.025 (|tan| 0.1–0.4). The estimator\'s '
                   'own bias with the simulated source.')
        P.hline(G4FOCUS['floor'][0], PURPLE, '3 5', 1.5, label='sim floor 15–16 mm', anchor='start', where='above')
        P.vline(1.0, MUT, '6 5', 1.5)
        for chain, c, dash, name in (('prod', C_PROD, '8 6', 'production'), ('is2_v2', C_IS2, None, 'is2_v2')):
            q = _focus_curve(v, chain, arm, 0.2)
            q = q[(q.med_miss <= 34) & (q.scale >= 0.75) & (q.scale <= 1.35)]
            P.line(q.scale.tolist(), q.med_miss.tolist(), c, 3.5, dash=dash, markers=False,
                   tip=f'{name}, {arm}, |own tan| 0.2–0.3: median miss vs s (n {int(q.n.iloc[0]):,})')
            b = q.loc[q.med_miss.idxmin()]
            P.points([b.scale], [b.med_miss], c, r=9, marker='open',
                     tips=[f'{name} {arm}: sharpest at s {b.scale:.3f}, median miss {b.med_miss:.1f} mm; '
                           f'at s = 1: {float(q[np.isclose(q.scale, 1.0)].med_miss.iloc[0]):.1f} mm'])
        panels.append(P.svg(f'focus {arm}'))
    rows = []
    for arm in 'AC':
        for chain, name in (('prod', 'production'), ('is2_v2', 'is2_v2')):
            cells = []
            for lo in (0.1, 0.2, 0.3, 0.4):
                q = _focus_curve(v, chain, arm, lo)
                cells.append(f'{q.loc[q.med_miss.idxmin(), "scale"]:.3f}')
            rows.append([f'{arm} {name}'] + cells)
    tab = sd.table(['sharpest s', '0.1–0.2', '0.2–0.3', '0.3–0.4', '0.4–0.5'], rows, size=21)
    body = sd.title('is2_v2 images the capsule more sharply than production, but sharpest at s ≈ 1.05–1.12',
                    'Rescale every gated track\'s x tan by s; median distance from the beam axis at the capsule plane. '
                    'Post-access runs.')
    body += sd.row(*panels, gap=44)
    body += sd.legend([('production', C_PROD, 'dash'), ('is2_v2', C_IS2), ('Geant4: estimator bias (true angles)', PURPLE),
                       ('open circle: sharpest s', MUT, 'dot')], 22)
    D.slide('focus', body, f'''
<p><b>Why a focus scan.</b> The standard capsule image (the band\'s zero crossing) is scale-free by design, so it cannot test k. Rescaling every track\'s tan by s and asking where the image is sharpest can. <code>ntof_cosmics/capsule_focus.py</code>, s 0.6–1.4 in 0.025, per |own tan| bin, with a shuffled-angle null (median miss 56–100 mm).</p>
<p><b>Calibrated on Geant4</b> (g4_*_is2_full): with the true sim angles the focus is sharpest at s 0.975–1.025, with the sim reco angles 1.00–1.025; the floor is a 15–16 mm median miss (source extent + scattering). Unlike the band (biased 1.005 → 1.09 by source width, §13), the focus is close to unbiased in the sim.</p>
<p><b>Sharpest s per bin</b> (0.025 grid; finer fits on the next slide):</p>{tab}
<p>Production\'s own focus says production is ~7.5 % steep on A and right on C. With the same-track ratios (A 0.86, C 0.90) that predicts is2_v2 at A 1.08, C 1.11, as measured: the two chains agree with each other about what the capsule wants.</p>''',
            foot='capsule_focus.py · results/capsule_focus/focus_{prod,is2_v2}.csv',
            short='Capsule focus')


def s_steep(D, v):
    f = v['focus']
    P = sd.Plot(1040, 600, x=(0.0, 0.8), y=(0, 150), xlabel='|own x tan| bin (is2_v2, gated at 0.8 true)',
                ylabel='median miss at the beam axis, s = 1 [mm]', margin=(20, 24, 96, 104))
    P.xticks([(t, f'{t:.1f}') for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)])
    P.yticks([(t, str(t)) for t in (0, 25, 50, 75, 100, 125, 150)])
    P.band([TANX_GEOM, 0.8], [0, 0], [150, 150], GREY, 0.12,
           tip=f'A track from the capsule cannot reach |tan x| > {TANX_GEOM} on these chambers '
               f'(active edge ~150 mm at D {D_AXIS} mm).')
    P.text(0.72, 8, 'beyond capsule reach', 20, MUT, 'middle')
    for arm, col in (('A', BLUE), ('C', ORANGE)):
        xs, ys, ns, tips, tipn = [], [], [], [], []
        for lo in np.arange(0.0, 0.8, 0.1):
            q = f[(f.chain == 'is2_v2_acc08') & (f.arm == arm) & np.isclose(f.tan_lo, lo) & np.isclose(f.tan_hi, lo + 0.1)]
            a = q[~q.null & np.isclose(q.scale, 1.0)]
            n = q[q.null]
            if not len(a):
                continue
            xs.append(lo + 0.05)
            ys.append(float(a.med_miss.iloc[0]))
            ns.append(float(n.med_miss.median()))
            tips.append(f'{arm} |tan| {lo:.1f}–{lo + 0.1:.1f}: median miss {ys[-1]:.1f} mm (n {int(a.n.iloc[0]):,})')
            tipn.append(f'{arm} |tan| {lo:.1f}–{lo + 0.1:.1f}, shuffled-angle null: {ns[-1]:.1f} mm')
        P.line(xs, ys, col, 3.5, tips=tips)
        P.line(xs, ns, col, 2.5, dash='8 6', marker='open', tips=tipn)
    pcs = v['pc08']
    right = sd.col(
        sd.legend([('A', BLUE), ('C', ORANGE), ('dashed: shuffled-angle null', MUT, 'dash')], 22),
        sd.p('Below 0.4 the image sits 15–25 mm from the axis, near the simulated floor. At 0.5–0.6 it is already '
             '64–72 mm (sim reco: 24 mm); above 0.6 it is <b>worse than random</b>.', 24),
        sd.p('Those steep tracks are real particles: the wall still confirms 16–29 % of them (0.60–0.75 in the core). '
             'They are not from the capsule.', 24),
        sd.callout('Decision (10-11): no wider stage-2 pass. The analysis cut stays 0.6 true, or tighter; '
                   'stage3_is2_v2 keeps the 0.8 superset as gated.', GREEN, 23),
        gap=18, w=560)
    body = sd.title('Tracks steeper than |tan| 0.5 do not point at the capsule; above 0.6, worse than random',
                    'is2_v2 rebuilt with the true-angle cut at 0.8: median x miss at the beam axis per |tan| bin, s = 1.')
    body += sd.row(P.svg('steep'), right, gap=44)
    D.slide('steep', body, '''
<p><b>The 0.8 rebuild</b> is lossless: <code>gated</code> at 0.6 equals <code>gated_reco</code> &amp; both |tan| < 0.6. Opening to 0.8 adds gated A +15.5 %, C +18.6 %, but wall-confirmed only A +5.3 % (1,542,705 → 1,623,837) and C +7.6 % (1,257,534 → 1,352,526).</p>
<p><b>Split by view.</b> Tracks steep only in y (x fine) still miss by 85–89 mm (core 26.5 mm) and sit at the chamber edges. y could reach ~0.75 geometrically, x only 0.64. So 0.6–0.8 is background (non-capsule particles) or badly mis-measured, not signal acceptance. 0.8 true is the widest the current stage 2 allows (C run_88 reach 0.843).</p>
<p>Record: HANDOFF_TRACKING §21; wall tables <code>results/fullpass_is2_v2_acc08/</code>.</p>''',
            foot='capsule_focus.py on stage3_is2_v2 at --tan-max-true 0.8',
            short='Steep tracks')


def s_split(D, v):
    sp = v['split']
    keep = ['all', 'ms 10-20', 'ms 20-80', 'y in capsule', 'y outside capsule', 'wall fired', 'wall not fired',
            'chi2 < median', 'chi2 > median', 'q/len < median', 'q/len > median', '|tany| < 0.15', '|tany| > 0.15',
            'tanx > 0', 'tanx < 0']
    label = {'all': 'all gated', 'ms 10-20': '10–20 ms', 'ms 20-80': '20–80 ms', 'y in capsule': 'y inside capsule height',
             'y outside capsule': 'y outside', 'wall fired': 'wall fired', 'wall not fired': 'wall not fired',
             'chi2 < median': 'χ² < median', 'chi2 > median': 'χ² > median', 'q/len < median': 'charge/length low',
             'q/len > median': 'charge/length high', '|tany| < 0.15': '|tan y| < 0.15', '|tany| > 0.15': '|tan y| > 0.15',
             'tanx > 0': 'tan x > 0', 'tanx < 0': 'tan x < 0'}
    rows = keep + ['fit_pos', 'fit_neg', 'fit']
    label.update(fit_pos='tan x > 0, with dx fitted', fit_neg='tan x < 0, with dx fitted', fit='joint (s, dx) fit')
    n = len(rows)
    P = sd.Plot(1060, 700, x=(0.9, 1.32), y=(n + 0.5, 0.3), xlabel='sharpest s, |tan| 0.1–0.5', ylabel='',
                margin=(10, 24, 80, 330))
    P.xticks([(t, f'{t:.2f}') for t in (0.9, 1.0, 1.1, 1.2, 1.3)])
    P.band([G4FOCUS['reco'][0] - 0.025, G4FOCUS['true'][1]], [0.3, 0.3], [n + 0.5, n + 0.5], PURPLE, 0.10,
           tip='Geant4: where the estimator puts a correct scale (s 0.975–1.025)')
    P.vline(1.0, MUT, '6 5', 1.5)
    for i, r in enumerate(rows):
        P.text(0.893, i + 1 + 0.15, label[r], 21, INK if not r.startswith('fit') else GREEN, 'end',
               600 if r.startswith('fit') else 400)
        if i in (len(keep) - 2, len(keep)):
            P.raw(sd.line(P.x0 - 320, P.Y(i + 0.5), P.x0 + P.pw, P.Y(i + 0.5), sd.RULE, 1.5))
    for arm, col, dy in (('A', BLUE, -0.15), ('C', ORANGE, 0.15)):
        for i, r in enumerate(rows):
            if r.startswith('fit'):
                o = _offset(v, arm)
                s = {'fit': o.s_best, 'fit_pos': o.s_pos_at_dx, 'fit_neg': o.s_neg_at_dx}[r]
                t = f'{arm}, 20–80 ms, wall fired: s {s:.2f} at dx {o.dx_best:+.1f} mm (median miss {o.med_miss:.1f} mm, n {int(o.n):,})'
                P.points([s], [i + 1 + dy], col, r=9, marker='diamond', tips=[t])
                continue
            q = sp[(sp.split == r) & (sp.arm == arm) & np.isclose(sp.tan_lo, 0.1) & np.isclose(sp.tan_hi, 0.5)]
            if not len(q):
                continue
            q = q.iloc[0]
            P.raw(sd.line(P.X(q.s_best - q.s_err), P.Y(i + 1 + dy), P.X(q.s_best + q.s_err), P.Y(i + 1 + dy), col, 2.5))
            P.points([q.s_best], [i + 1 + dy], col, r=8,
                     tips=[f'{arm} {label[r]}: s {q.s_best:.3f} ± {q.s_err:.3f}, median miss {q.med_miss:.1f} mm, n {int(q.n):,}'])
    oA, oC = _offset(v, 'A'), _offset(v, 'C')
    right = sd.col(
        sd.legend([('A', BLUE), ('C', ORANGE), ('Geant4 unbiased band', PURPLE)], 22),
        sd.p('Time since the flash, capsule height, wall, χ², charge, y angle: the same s, 1.08–1.13.', 24),
        sd.p('Only the sign of tan x splits it, oppositely on A and C. Since A and C face opposite ways in global x, '
             f'that is one displacement of the source: a joint fit gives dx = {oA.dx_best:+.1f} mm (A), '
             f'{oC.dx_best:+.1f} mm (C), as the band-crossing image also finds (−9.3).', 24),
        sd.callout(f'With dx fitted the halves agree and the miss falls to {oA.med_miss:.1f} mm, below the simulated '
                   f'floor. But s stays {oA.s_best:.2f} (A) and {oC.s_best:.2f} (C).', RED, 23),
        gap=18, w=520)
    body = sd.title(f'The capsule wants s ≈ {oA.s_best:.2f} on both arms, in every population, after the source offset',
                    'Sharpest s by population (focus_split.py) and the joint (s, dx) fit (focus_offset.py), is2_v2.')
    body += sd.row(P.svg('split'), right, gap=40)
    D.slide('split', body, f'''
<p><b>Populations</b> are |own tan| 0.1–0.5, all post-access runs; s_err is the curvature error of a parabola near the minimum. 1–10 ms after the flash does not focus at all (median miss 58–82 mm: flash junk) and is not drawn; neither is "other arms too" (n ≈ 25 k, miss 72–76 mm).</p>
<p><b>The joint fit</b> uses 20–80 ms, wall fired (1.86 M tracks), s and dx on a grid. Per |tan| bin: A 1.11 / 1.13 / 1.12 / 1.08, C 1.15 / 1.12 / 1.13 / 1.07 at 0.1–0.2 … 0.4–0.5. The fitted dx is stable at −8 … −10.5 mm.</p>
<p><b>Side finding, not fixed:</b> <code>build_tracks.pointing()</code> computes <code>dca_axis_mm</code>, <code>target_*</code> and <code>in_bore</code> against the nominal axis (0, y, 0). The source sits ≈ −9.5 mm in x, about one capsule radius, so any cut on <code>in_bore</code> / <code>dca_axis_mm</code> selects off-centre.</p>
<p>Record: HANDOFF_TRACKING §22. Results: <code>results/focus_split/</code>.</p>''',
            short='Scale by population')


def s_tension(D, o, v):
    oA, oC = _offset(v, 'A'), _offset(v, 'C')
    n = o['norm'][o['norm'].sep_max == 10].set_index(['arm', 'view'])
    cc = o['cc']
    rows = [
        ('cosmic', 'Cosmic A–C line (run_149, held out)', 'closure of the bundle itself, |tan| 0.08–0.6'),
        ('muons', 'In-beam muons, A–C (20–80 ms)', 'norm applied in k; inferred through production'),
        ('g4', 'Geant4: focus estimator, simulated source', 'true / reco sim angles'),
        ('focus', 'Capsule focus, joint (s, dx)', '20–80 ms, wall fired, |tan| 0.1–0.5'),
    ]
    P = sd.Plot(1080, 520, x=(0.85, 1.2), y=(len(rows) + 0.5, 0.4), xlabel='factor the truth asks for on is2 tans '
                '(1 = is2 right)', margin=(10, 24, 84, 40))
    P.xticks([(t, f'{t:.2f}') for t in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)])
    P.vline(1.0, MUT, '6 5', 1.5)
    for i, (key, lab, sub) in enumerate(rows):
        P.text(0.853, i + 1 - 0.30, lab, 22, INK, 'start', 600)
        P.text(0.853, i + 1 - 0.05, sub, 19, MUT, 'start')
    for arm, col, dy in (('A', BLUE, 0.17), ('C', ORANGE, 0.32)):
        y = 1 + dy
        c = cc[(cc.arm == arm) & (cc.chain == 'is2_v1') & (cc.lo >= 0.08)]
        vals = 1 / c.ratio
        lo_, hi_ = float(vals.quantile(0.1)), float(vals.quantile(0.9))
        P.raw(sd.line(P.X(lo_), P.Y(y), P.X(hi_), P.Y(y), col, 3))
        P.points([float(vals.median())], [y], col, r=8,
                 tips=[f'{arm}: true ÷ delivered over |tan| 0.08–0.6, x and y bins: median {vals.median():.3f}, '
                       f'10–90 % {lo_:.3f}–{hi_:.3f} (is2_v1 bundle; is2_v2 rescale ×1.009 A, ×1.003 C)'])
        y = 2 + dy
        e = float(np.mean([n.loc[(arm, ax), 'norm_err'] for ax in 'xy']))
        P.raw(sd.line(P.X(1 - e), P.Y(y), P.X(1 + e), P.Y(y), col, 3))
        P.points([1.0], [y], col, r=8, tips=[f'{arm}: the pooled in-beam muon norm is built into k, so muons read 1 '
                                             f'by construction; error ±{e:.3f} (x {n.loc[(arm, "x"), "norm"]:.3f}, '
                                             f'y {n.loc[(arm, "y"), "norm"]:.3f} of run_149, n {int(n.loc[(arm, "x"), "pooled_n"])}/'
                                             f'{int(n.loc[(arm, "y"), "pooled_n"])})'])
        y = 4 + dy
        s = oA.s_best if arm == 'A' else oC.s_best
        P.points([s], [y], col, r=10, marker='diamond', tips=[f'{arm}: s {s:.2f} at dx {(oA if arm == "A" else oC).dx_best:+.1f} mm'])
    P.raw(sd.line(P.X(G4FOCUS['true'][0]), P.Y(3.25), P.X(G4FOCUS['true'][1]), P.Y(3.25), PURPLE, 6))
    P.points([1.0], [3.25], PURPLE, r=8, tips=['Geant4: sharpest s 0.975–1.025 with true angles, 1.00–1.025 with the '
                                               'digitised reco angles (|tan| 0.1–0.4). §21.'])
    P.raw(f'<rect x="{P.X(1.0):.1f}" y="{P.Y(4.02):.1f}" width="{P.X(1.12) - P.X(1.0):.1f}" height="{P.Y(4.47) - P.Y(4.02):.1f}" '
          f'fill="{RED}" fill-opacity="0.08"/>')
    P.text(1.06, 4.33, '≈ 12 %', 24, RED, 'middle', 600)
    ex = [('Geometry (distances)', 'common error cancels; differential is antisymmetric A vs C, data symmetric'),
          ('Source offset', f'real, dx {oA.dx_best:+.1f} mm; fixes the tan-sign split, not s'),
          ('Depth origin (t0)', 'a 26 mm slide needs charge at 40 mm; measured mid-gap 12–14 mm: ≲ 1 %'),
          ('Population', 'same s in every split (previous slide)'),
          ('Sharing kernel', 'the physical RC kernel moves the raw scale < 2 % (next slide)')]
    ex_html = ''.join(f'<div style="display:flex;gap:14px;align-items:baseline;padding:7px 0;border-top:1px solid {sd.RULE}">'
                      f'<p style="font-size:22px;font-weight:600;width:210px;color:{GREEN}">✗ {a}</p>'
                      f'<p style="font-size:20px;color:{MUT};flex:1;line-height:1.3">{b}</p></div>' for a, b in ex)
    right = sd.col(sd.p('Ruled out as the cause', 24, INK, 600), f'<div>{ex_html}</div>',
                   sd.callout('Left: a difference between beam electrons and muons, in the <b>vertex distribution</b> '
                              '(data vs the simulated source) or the <b>angle response</b>.', RED, 23), gap=12, w=560)
    body = sd.title('The open problem: muons say is2 is right, the capsule says its angles are 12 % too shallow',
                    'What each angle truth asks of the is2 scale, chambers A and C.')
    body += sd.row(P.svg('tension'), right, gap=40)
    body += sd.legend([('A', BLUE), ('C', ORANGE), ('Geant4', PURPLE)], 22)
    D.slide('tension', body, f'''
<p><b>Reading the plot.</b> Each row is a truth, plotted as the factor it would apply to is2\'s delivered tans. The cosmic A–C line and the in-beam muons are both muons; the capsule focus is beam electrons (few MeV, mostly Compton electrons from capture gammas). The Geant4 row is the focus estimator run on simulated electrons through the same reco: it reads a correct scale as 1.00 ± 0.03.</p>
<p><b>Equivalent statement.</b> s = 1.12 is the same as the tracks converging {(oA.s_best - 1) * D_AXIS:.0f} mm beyond the beam axis as seen from each chamber. Because that is on opposite sides for A and C, no single displaced source does it (that is the antisymmetric dx, which was fitted out); it would take a vertex distribution that is "far" from both chambers at once, or an electron angle response ~12 % shallower than the muons\'.</p>
<p><b>History of the sign.</b> The band residual (§13) already had A +3.6 %, C +10 % on the same side, but the band is source-width biased. The focus is not (in the sim), so the residual is now a measurement, not an artefact of the estimator. The data image after the offset fit (11.6 mm) is tighter than the simulated floor (15–16 mm): the simulated source is wider than the real one, which is itself a data/sim difference.</p>
<p><b>Why it matters.</b> Every opening angle, the 170° cut and T1\'s F thresholds scale with k. A 12 % scale error on both arms moves a 140° pair by several degrees.</p>''',
            short='The tension')


def s_kernel(D, o):
    tab = sd.table(['arm (constants)', 'Δσ x, 0.08–0.6', 'Δσ y, 0.08–0.6', 'head-on Δσ x / y', 'x S-shape', 'raw scale'],
                   [list(r) for r in RC['rows']], size=20, align=['left'] + ['right'] * 5,
                   widths=[270, 170, 230, 160, 170, 100])
    bars = sd.hbars([('A · rcm3', 1.9, BLUE, 'raw scale moves −1.9 %'),
                     ('C · rcm7', 0.7, ORANGE, 'raw scale moves −0.7 %'),
                     ('C · rcm6', 0.7, ORANGE, 'raw scale moves +0.7 %'),
                     ('needed by the capsule', 12.0, RED, 'the focus asks for ×1.12')],
                    13, width=620, h=30, label_w=300, fmt=lambda x: f'{x:.1f} %', size=23)
    left = sd.col(sd.p('Change against the is2 bundle, paired bootstrap, 547 held-out run_149 A–C events '
                       '(negative = better):', 24), tab,
                  sd.p('|Change of raw angle scale| against what the capsule asks', 24, INK, 600), bars, gap=18, w=1060)
    right = sd.col(
        sd.p('The cloud_basics head-on constants (σ0, Dp, RC diffusion in y) dropped into the is2 bundles, no refit.', 24),
        sd.p('A reproduces the bench: Y resolution −7 %, X neutral, a third of the x S-shape removed.', 24),
        sd.p('C does not get det6\'s bench gain (−18 to −22 %): n_TOF signal-to-noise is 7–10× worse and C\'s gas wetter.', 24),
        sd.callout('The detector model matters at n_TOF for A\'s y and head-on resolution and the S-shape, '
                   'not for the scale question.', GREEN, 23), gap=18, w=540)
    body = sd.title('The physical (RC) kernel sharpens A\'s y by 7 % and moves the angle scale by under 2 %',
                    'Opt-in RC kernel from the detector-model work, on the n_TOF cosmic truth.')
    body += sd.row(left, right, gap=48)
    D.slide('kernel', body, f'''
<p><b>Arms.</b> rcm3 / rcm7 / rcm6 = the is2 bundle plus cloud_basics §12 head-on constants of det3 / det7 / det6; seeder 3, TAN_MAX 1.0, two-sided. Each label is evaluated at its own TRAIN scale, so the comparison is of shape and resolution, and the "raw scale" column is how much the kernel moves the fit\'s v-scale.</p>
<p><b>Not tested:</b> n_TOF-gas Dp (Magboltz 90/10 at the n_TOF fields), det6\'s own template (the rcm6 arm runs on is2_C\'s det7 template), the joint X+Y Dd, and any refit. Source: {RC["src"]}; scripts in <code>~/scratch/ntof_insitu/</code> (recipe_rc.sh), kernel code brought from <code>mx17-paper-status</code> (opt-in, off by default).</p>''',
            short='RC kernel')


def s_problems(D, o):
    W = dict(key=RED, cheap=GOLD, done=GREEN, open=GREY)
    items = [
        ('key', 'Angle scale: in-beam muons vs capsule', 'stage 3 (k)', 'capsule asks ×1.12 on both arms; muons 1.00 ± 0.02', '§21–22'),
        ('cheap', 'Muon norm still inferred via production', 'stage 3', 're-derive from is2_v2\'s own in-beam A–C muons', '§19'),
        ('cheap', 'Source offset not in pointing()', 'stage 3', 'in_bore / dca_axis_mm use x = 0; source at −9.5 mm', '§22'),
        ('done', 'Raw-tan plausibility cut', 'stage 2', 'TAN_MAX 1.0 in the fit; true-angle cut 0.6 (0.8 kept)', '§16, §21'),
        ('done', 'is2 yield on beam; mirror basins', 'stage 2', 'two-sided search; confirmed ×1.49 A, ×1.82 C', '§17–20'),
        ('open', 'Late tracks (t0 > 300 ns)', 'stage 2', 'wrong geometry on ~17 % of gated tracks', 'T3 / O10'),
        ('open', 'C x non-linearity, C core tails', 'stage 2', '1.03 → 0.89 across |tan|; tails 8–13 %', '§9'),
        ('open', 'Near normal', 'stage 2 / flags', '|tan| < 0.02 σ 0.15; bias below 0.08', '§7d, §9'),
        ('open', 'RC kernel (A y −7 %, S-shape −⅓)', 'stage 2', 'opt-in; ride the next stage-2 pass', '§22'),
        ('open', 'B and D', 'all', 'no in-situ truth; D bundle, B HV in run_149', '§20'),
    ]
    head = ('<div style="display:grid;grid-template-columns:40px 520px 190px 1fr 170px;gap:0 20px;'
            'font-size:21px;color:%s;padding:0 0 8px 0;border-bottom:2px solid %s">'
            '<p></p><p>problem</p><p>where it acts</p><p>size / what is known</p><p>record</p></div>') % (MUT, sd.RULE)
    rows = ''.join(
        f'<div style="display:grid;grid-template-columns:40px 520px 190px 1fr 170px;gap:0 20px;align-items:center;'
        f'padding:9px 0;border-bottom:1px solid {sd.RULE}">'
        f'<div style="width:22px;height:22px;border-radius:50%;background:{W[w]}"></div>'
        f'<p style="font-size:24px;font-weight:600{";color:" + MUT if w == "done" else ""}">{a}</p>'
        f'<p style="font-size:22px;color:{MUT}">{b}</p>'
        f'<p style="font-size:22px">{c}</p><p style="font-size:20px;color:{MUT}">{d}</p></div>'
        for w, a, b, c, d in items)
    body = sd.title('One problem decides the k on both arms; the launch blockers are closed',
                    'Open problems in the A/C angle chain after the is2_v2 pass, by where a fix would act.')
    body += sd.legend([('the open question', RED, 'dot'), ('cheap, stage 3', GOLD, 'dot'),
                       ('done', GREEN, 'dot'), ('open, needs a later stage-2 pass', GREY, 'dot')], 22)
    body += f'<div style="display:flex;flex-direction:column">{head}{rows}</div>'
    D.slide('problems', body, '''
<p><b>The sorting rule.</b> Anything that changes stage 2 (the fit) costs a re-pass (~20× CPU now, with the two-sided search); anything in stage 3 (k, gates, flags, pointing) is a re-run of <code>campaign_tracks</code> over the stage-2 output, minutes to hours. The scale question is stage 3: whichever truth wins, it changes k, not the fit.</p>
<p><b>Closed since 8 Oct.</b> TAN_MAX was raised to 1.0 raw in the fit with a stage-3 true-angle cut (§16); the pilot and the full pass tested the is2 bundles on beam (§19–20); the two-sided search fixed the mirror basins (§17–18). The 8 Oct version of this slide had the plausibility cut as the one launch blocker.</p>''',
            short='Open problems')


def s_next(D, o, v):
    oA = _offset(v, 'A')
    cards = [
        ('1 · Muons through is2_v2 directly', RED, '< 1 h, stage 3',
         'The in-beam A–C muons are in the is2_v2 stage-2 output. Measure their scale there instead of inferring it '
         'through production. If they still read 1.00, the tension is real; if they read ~1.1, it was the inference.'),
        ('2 · Focus on the digitised sim, with the offset', RED, 'hours',
         'g4_digi through the is2 bundles and the two-sided reco, then the joint (s, dx) fit with the data\'s −9.5 mm. '
         'Does the estimator still read 1.00 there? Compare image widths: data 11.6 mm is tighter than the sim floor.'),
        ('3 · Vary the simulated source', GOLD, '~1 day',
         f'What vertex distribution makes a muon-correct reco focus at 1.12, i.e. converge ~{(oA.s_best - 1) * D_AXIS:.0f} mm '
         'beyond the axis from both sides? Capsule walls, gamma conversions in the holder, structures behind the axis.'),
        ('4 · An electron-specific truth', GOLD, 'days',
         'Two-arm A–C pairs from one vertex: the depth where their lines cross moves with s, independent of the '
         'source\'s x extent. Statistics are the risk (~20 inter-chamber pairs per sub-run).'),
        ('5 · Meanwhile: carry it as a systematic', GREY, 'decision',
         'Keep k from the muons (the only truth not a model of the source) and quote the capsule scale as an '
         'alternate: ±6 % about the midpoint on opening angles, the 170° cut and T1\'s F.'),
        ('6 · Cheap fixes, any time', GREEN, 'stage 3',
         'Shift pointing()\'s axis to the measured source x; park the RC kernel and the late-t0 fix for the next '
         'stage-2 pass rather than a pass of their own.'),
    ]
    html = ''.join(
        f'<div style="background:{sd.CARD};border:1px solid {sd.RULE};border-top:6px solid {c};border-radius:14px;'
        f'padding:22px 24px;display:flex;flex-direction:column;gap:10px">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px"><p style="font-size:26px;font-weight:600">{t}</p>'
        f'<p style="font-size:20px;color:{MUT};white-space:nowrap">{cost}</p></div>'
        f'<p style="font-size:22px;line-height:1.35;color:{INK}">{d}</p></div>' for t, c, cost, d in cards)
    body = sd.title('Where to go next: two cheap checks decide whether the 12 % is real',
                    'In order. 1 and 2 test the two inferences each side rests on; 3 and 4 only if both hold.')
    body += f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px">{html}</div>'
    D.slide('next', body, '''
<p><b>Why this order.</b> Each side of the disagreement rests on one inferred step. The muon side: the norm was measured with the <i>production</i> bundle and carried to is2 assuming is2 raw = production raw × a constant, which the cosmic closure supports only to its own precision. The capsule side: "the focus is unbiased" was shown in Geant4 with the simulated source and without the offset the data turned out to have. Checks 1 and 2 remove those two steps, for an hour and a few hours.</p>
<p><b>If both hold</b>, the difference is physics of the data, not of the estimators: either the vertex distribution (3) or electrons responding differently from muons in the real chamber, which only an electron truth (4) can show. Until then, option 5 keeps downstream work moving without choosing.</p>
<p><b>What not to do.</b> Do not refit k on the capsule: that is the k_arm logic production used (A 1.27), and it is what the in-situ calibration replaced. A wider stage-2 pass is also off the table (§21: steep tracks are background).</p>''',
            short='Where next')


def s_close(D):
    items = [
        ('Which scale is right on A and C', 'Muons and the capsule disagree by ~12 % on both arms; neither has been shown wrong.'),
        ('Electrons ≠ muons in the real chamber', 'The digitiser reproduces muon-like response for electrons in the model, not in the chamber.'),
        ('The source model', 'The data image is tighter than the simulated one and sits 9.5 mm off axis.'),
        ('Angles above |tan| 0.6', 'No truth there; data at 0.5–0.6 already points poorly.'),
        ('B and D', 'No in-situ truth; null angles.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:13px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:26px;font-weight:600;width:400px">{a}</p>'
                    f'<p style="font-size:22px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    nxt = [('Run checks 1 and 2', 'is2_v2 in-beam muons directly; (s, dx) focus on the digitised sim.'),
           ('Then decide k', 'Muons, capsule, or a bracket carried as a systematic downstream.'),
           ('Fix pointing()', 'Use the measured source x for in_bore / dca_axis_mm.'),
           ('Next stage-2 pass', 'Bundle the RC kernel (A), the late-t0 fix and T1\'s xy pairing into one.')]
    nrows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:13px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:26px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:22px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in nxt)
    body = (f'<div style="display:flex;gap:72px">'
            f'<div style="flex:1.15;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:48px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:48px;font-weight:600">Decisions</h2>{nrows}</div></div>')
    D.slide('close', body, '''
<p>Entry points: <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> (§19–22 for this update), the root <code>HANDOFF.md</code>, and <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code> for how this joins the two-track work. Regenerate with <code>ntof_cosmics/make_insitu_deck.py</code> after <code>repass_readiness.py all</code>, <code>pilot_compare.py … --version is2_v2</code>, <code>capsule_focus.py</code>, <code>focus_split.py</code> and <code>focus_offset.py</code>.</p>''',
            dark=True, short='Decisions')


def build(out: Path) -> Path:
    o = load()
    v = load_v2()
    D = sd.Deck('A and C angles: in-situ calibration and the is2_v2 re-pass',
                'Chambers A and C on the in-situ (is2) angle calibration: how good it is, what the is2_v2 campaign '
                're-pass gained, and the open disagreement between in-beam muons and the capsule on the angle scale.')
    s_cover(D, o, v)
    for f in (s_setup, s_v, s_closure, s_resolution, s_truth, s_digi, s_muons, s_gas, s_residual, s_gate, s_shift):
        f(D, o)
    s_fullpass(D, o, v)
    s_focus(D, v)
    s_steep(D, v)
    s_split(D, v)
    s_tension(D, o, v)
    s_kernel(D, o)
    s_problems(D, o)
    s_next(D, o, v)
    s_close(D)
    oA, oC = _offset(v, 'A'), _offset(v, 'C')
    meta = dict(title='A and C angles: in-situ calibration and the is2_v2 re-pass',
                summary=(f'The two-sided is2_v2 re-pass confirms ×{_sum_ratio(v, "A", "confirmed_acc"):.2f} (A) and '
                         f'×{_sum_ratio(v, "C", "confirmed_acc"):.2f} (C) more tracks on the wall than production. '
                         f'Open: the capsule focus asks for is2 angles ×{oA.s_best:.2f} (A) / ×{oC.s_best:.2f} (C) while '
                         'in-beam muons read them right; geometry, source offset, depth origin and the kernel are ruled out. '
                         'Next steps to settle it.'),
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
