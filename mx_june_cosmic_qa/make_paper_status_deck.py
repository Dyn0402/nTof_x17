#!/usr/bin/env python3
"""
make_paper_status_deck.py -- where the MX17 detector paper stands, as a slide note.

The paper covers the June cosmic bench and the SPS H4 beam test (det4); the
n_TOF results are a separate paper. This deck is the 2026-10-02 re-audit of
PAPER_STATUS.md (whose tables predate the waveform-first rebase landing and the
c2 > c1 retirement): what exists, which topics are ready, which numbers have to
be re-quoted, and where those numbers physically live.

Reads, so that rerunning after any of them moves the slides:
  * mx_june_wft/FLEET_DIGEST.md                 -- the r06 fleet numbers (wft vs hits)
  * mx_june_wft/RETIRE_C2GTC1_2026-08-21.md     -- the retired c2/c1 ratios
  * sps_beam_test_26/analysis/sharing_kernel/{fit_kernel,bench_kernel_y}.json
  * sps_beam_test_26/analysis/spatial_resolution/results.json
  * sps_beam_test_26/analysis/angled_kernel/results.json

    python mx_june_cosmic_qa/make_paper_status_deck.py [--out PATH]
    python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug mx17-detector-paper-status --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402
                      INK, MUT, RULE, DBLUE, DRED, DGREEN, DMUT)

FLEET = REPO / 'mx_june_wft' / 'FLEET_DIGEST.md'
RETIRE = REPO / 'mx_june_wft' / 'RETIRE_C2GTC1_2026-08-21.md'
SPS = REPO / 'sps_beam_test_26' / 'analysis'

DETS = ['sat_det3', 'o22_long_det2', 'g_det4', 'g_det6_long', 'g_det7_long']
DLAB = {'sat_det3': 'det3', 'o22_long_det2': 'det2', 'g_det4': 'det4',
        'g_det6_long': 'det6', 'g_det7_long': 'det7'}

# Status colours, used on every slide that carries a status.
READY, REQUOTE, OPEN, RETIRED = GREEN, BLUE, GOLD, GREY
HITS, WFT = GREY, GREEN

# Numbers the SPS report states in prose (spatial_resolution/make_report.py),
# not in its results.json. The scattering figure is arithmetic, not a measurement.
M3_POINT_MM = (0.21, 0.24)
MS_MRAD, MS_LEVER_MM = 1.1, 558
# The bench residual the SPS report compares with: det3 σ68 at θ < 5°, x / y
# (PAPER_STATUS topic 9 on the χ² < 1 recipe; the wft core σ|r| is a narrower core fit).
BENCH_S68 = (0.61, 0.72)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_fleet():
    """FLEET_DIGEST.md first table -> {metric: {key: (new, old|None)}}, plus bundles."""
    txt = FLEET.read_text()
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('| quantity'))
    keys = [c.strip() for c in lines[i].strip('|').split('|')][1:]
    out = {}
    for l in lines[i + 2:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip('|').split('|')]
        row = {}
        for k, c in zip(keys, cells[1:]):
            m = re.match(r'([+-]?[\d.]+)\s*(?:\(was ([+-]?[\d.]+))?', c)
            if m:
                row[k] = (float(m.group(1)), float(m.group(2)) if m.group(2) else None)
        out[cells[0]] = row
    bund = dict(re.findall(r'^\| (\w+) \| `(calib_bundle\w*)` \|', txt, flags=re.M))
    return out, bund


def load_retired():
    """The 'was' column of RETIRE §2: [(label, ratio)] for every inverted bundle."""
    rows = []
    for l in RETIRE.read_text().splitlines():
        m = re.match(r'\| (mx17_\d)(?: \(([\w]+)\))? \| .*?\| \*\*([\d.]+)\*\* \|', l)
        if m:
            rows.append((m.group(1).replace('mx17_', 'det') + (f' {m.group(2)}' if m.group(2) else ''),
                         float(m.group(3))))
    return rows


def load_sps():
    fk = json.loads((SPS / 'sharing_kernel' / 'fit_kernel.json').read_text())
    beam = []
    for run, r in fk['y'].items():
        par, err = r['delay']['par'], r['delay']['err']
        q = par['c2'] / par['c1']
        e = q * ((err['c2'] / par['c2']) ** 2 + (err['c1'] / par['c1']) ** 2) ** 0.5
        beam.append((run, r['field_Vcm'], r['n_events'], q, e))
    bk = json.loads((SPS / 'sharing_kernel' / 'bench_kernel_y.json').read_text())
    bp, be = bk['delay']['par'], bk['delay']['err']
    bq = bp['c2'] / bp['c1']
    bench = (bq, bq * ((be['c2'] / bp['c2']) ** 2 + (be['c1'] / bp['c1']) ** 2) ** 0.5, bk['n_events'])
    res = json.loads((SPS / 'spatial_resolution' / 'results.json').read_text())
    ang = json.loads((SPS / 'angled_kernel' / 'results.json').read_text())
    return dict(beam=beam, bench=bench, res=res, ang=ang)


# --------------------------------------------------------------------------- #
# glossary
# --------------------------------------------------------------------------- #
G = dict(
    wft=('Waveform-first reconstruction (wft/): a forward model of every strip waveform, including the '
         'resistive sharing, fitted per track. The basis for any position, angle or depth since 2026-07-28.'),
    hits=('The original chain: geometry from per-strip combined_hits times. Each hit time mixes in delayed '
          'neighbour charge, which compresses the drift ladder 20–30 % and reads ~4° too steep.'),
    r06=('calib_bundle_r06: the shipped calibration since 2026-08-19, with c2 pinned to 0.6 × c1. '
         'Every earlier bundle had c2 > c1, which no resistive film can produce.'),
    kernel=('The sharing kernel: how much of a strip’s charge appears on its ±1 and ±2 neighbours (c1, c2), '
            'and how late. c2 must be < c1: the ±2 strip is reached only through the ±1.'),
    sigt='σ_θ: angular resolution, core fit of reconstructed − M3-reference track angle, per readout plane.',
    w5='Fraction of M3 reference rays with a reconstructed track within 5 mm (|r| < 5 mm).',
    core='Core σ of the radial residual |r| to the M3 reference ray. Includes the reference’s own error.',
    pitch='Strip pitch 0.78 mm. A one-strip (binary) readout would give pitch/√12 = 225 µm.',
)


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, F, S):
    st = F['sigma_theta X']
    best = min(v[0] for v in st.values())
    worst = max(max(F['sigma_theta X'][k][0], F['sigma_theta Y'][k][0]) for k in DETS)
    sig = S['res']['fit']
    tip_r = '\n'.join(f'{DLAB[k]}: X {F["sigma_theta X"][k][0]:.2f}° (hits {F["sigma_theta X"][k][1]:.2f}), '
                      f'Y {F["sigma_theta Y"][k][0]:.2f}° (hits {F["sigma_theta Y"][k][1]:.2f})' for k in DETS)
    nums = ''.join([
        sd.bignum('0', 'manuscript pages', DRED,
                  'Ten topics audited and seven analysis plans written, all in this repo. '
                  'No outline or draft has been started.',
                  tip='mx_june_cosmic_qa/PAPER_STATUS.md (7-10, last edit 7-29) and paper_plans/PLAN_37–47.'),
        sd.bignum(f'{best:.2f}–{worst:.1f}°', 'cosmic angular resolution, five chambers', DGREEN,
                  'Waveform-first on the r06 calibration: better than the hits chain on every plane, '
                  'halved on det3.',
                  tip=tip_r),
        sd.bignum(f'{1000 * sig["sigma_det4"]:.0f} µm', 'det4 intrinsic, SPS H4', DBLUE,
                  f'Normal incidence, telescope fitted out: {sig["f_back"]:.2f} × pitch. '
                  'The bench’s 0.6–0.7 mm is mostly reference and scattering.',
                  tip=f'σ = {1000 * sig["sigma_det4"]:.0f} ± {1000 * sig["sigma_det4_err"]:.0f} µm, '
                      f'run 53, {S["res"]["run53"]["n"]:,} tracks.')])
    body = (sd.kicker('MX17 detector paper · cosmic bench + SPS H4 · status 2 Oct 2026')
            + '<h1 style="font-size:80px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1640px">'
              'The analyses for the detector paper are done. Its plan is two calibrations out of date</h1>'
            + '<div style="flex:1"></div>'
            + f'<p style="font-size:28px;color:{DMUT}">Scope: the June cosmic bench (det2, 3, 4, 6, 7) and det4 '
              'in the SPS H4 beam. n_TOF results are a separate paper.</p>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, '''
<p>This note re-audits <code>mx_june_cosmic_qa/PAPER_STATUS.md</code>, the 10-topic readiness audit of 2026-07-10. Its last edit was on 7-29. Since then two things changed every geometric number in it. The waveform-first reconstruction replaced the hits chain (decided 7-28). Then the c2 &gt; c1 sharing kernels were retired and everything was re-reconstructed on <code>calib_bundle_r06</code> (8-19 to 8-21).</p>
<p>So the topic list and the narrative are still useful, but none of the position, angle or drift-velocity values in PAPER_STATUS's tables can be quoted. Current values come from <code>mx_june_wft/FLEET_DIGEST.md</code> and the SPS reports, both read directly by this deck.</p>
<p>Not in scope: the n_TOF campaign (separate paper) and the agent-as-control-layer paper draft (<code>notes/x17-agent-paper-draft.html</code>), which is unrelated.</p>''',
            dark=True, short='Answer')


def s_timeline(D):
    W, H = 1664, 640
    t0, t1 = dt.date(2026, 6, 20), dt.date(2026, 10, 12)

    def X(d):
        return 40 + (W - 80) * (d - t0).days / (t1 - t0).days

    o = []
    ya, yb = 210, 420
    # stale span
    x29, xnow = X(dt.date(2026, 7, 29)), X(dt.date(2026, 10, 2))
    o.append(f'<rect x="{x29:.0f}" y="{ya - 70}" width="{xnow - x29:.0f}" height="{yb - ya + 140}" '
             f'fill="{GOLD}" fill-opacity="0.10"{sd.tipattr("PAPER_STATUS.md has not been edited since 7-29.")}/>')
    o.append(sd.T((x29 + xnow) / 2, ya - 82, 'PAPER_STATUS.md unedited for 65 days', 22, GOLD, weight=600))
    for y, lab in ((ya, 'paper plan'), (yb, 'what moved under it')):
        o.append(sd.line(250, y, W - 40, y, RULE, 3))
        o.append(sd.T(40, y + 8, lab, 22, MUT, 'start', 600))
    for m in range(7, 11):
        d = dt.date(2026, m, 1)
        o.append(sd.line(X(d), yb + 104, X(d), yb + 116, MUT, 2))
        o.append(sd.T(X(d), yb + 146, d.strftime('%b'), 22, MUT))
    o.append(sd.line(40, yb + 104, W - 40, yb + 104, RULE, 1))
    plan = [
        (dt.date(2026, 7, 10), 'audit: 10 topics', READY, 'PAPER_STATUS.md written; PLAN_37–41 + 42 drafted.', -30),
        (dt.date(2026, 7, 12), 'PLAN 38/39/42 done', READY, 'Charge balance, spark dead time (null), time resolution.', 44),
        (dt.date(2026, 7, 29), 'last edit: wft numbers', OPEN,
         'Fleet numbers from the waveform-first chain added, on bundles later found inverted.', -30),
    ]
    moved = [
        (dt.date(2026, 7, 28), 'hits → waveforms', BLUE,
         'RECONSTRUCTION_BASIS.md: never reconstruct geometry from combined_hits times.', 44),
        (dt.date(2026, 8, 5), 'SPS closed', BLUE, 'det4 H4 campaign audit: 8 unlisted runs, CF4 ladders, gain-invariance.', -30),
        (dt.date(2026, 8, 12), 'MPGD26 freeze', BLUE, 'Frozen chain + 214-row June condor campaign (FREEZE_MPGD26).', 44),
        (dt.date(2026, 8, 19), 'kernel measured', BLUE, 'SPS head-on, model-free: c2/c1 = 0.45.', -30),
        (dt.date(2026, 8, 21), 'c2 > c1 retired', RED, 'All bundles → r06; 156 condor jobs re-run (RETIRE_C2GTC1).', 80),
        (dt.date(2026, 9, 1), '“TPC”, not µTPC', GREY, 'Naming decision on the MPGD26 slides.', -30),
        (dt.date(2026, 10, 2), 'today', INK, 'This re-audit.', 44),
    ]
    for lane, y in ((plan, ya), (moved, yb)):
        for d, lab, c, tip, side in lane:
            x = X(d)
            ly = y + side
            o.append(f'<circle cx="{x:.0f}" cy="{y}" r="11" fill="{c}"{sd.tipattr(d.strftime("%d %b") + ": " + tip)}/>')
            o.append(sd.T(x, ly, lab, 21, c if c != GREY else MUT, weight=600, tip=tip))
    body = sd.title('The plan stopped 29 July; the reco changed twice',
                    'Paper-plan milestones (top) against the changes that invalidated its numbers (bottom). '
                    'Hover the dots and the dotted terms throughout this note.')
    body += sd.svg(W, H, ''.join(o), 'timeline')
    D.slide('timeline', body, '''
<p>Everything paper-related is in this repository, not on another machine: <code>mx_june_cosmic_qa/PAPER_STATUS.md</code>, <code>mx_june_cosmic_qa/paper_plans/</code> (PLAN_37–42 and PLAN_47), and the clone in <code>~/fleetcheck</code>, which is identical. Step 6 of the audit's order of work, "start the paper skeleton", was never begun.</p>
<p>The newest synthesis of the detector story is the MPGD26 deck (<code>mpgd26/</code>, slides finalised early September). Its <code>make_*.py</code> scripts already build the setup renders, the sharing cartoon, the efficiency loss budget and the angle-resolution figures, so it is the natural starting point for paper figures.</p>''',
            foot='Sources: git history of mx_june_cosmic_qa/, mx_june_wft/, sps_beam_test_26/, mpgd26/.',
            short='Timeline')


TOPICS = [
    # (section, status, one-liner, tooltip)
    ('Detector & resistive design', READY, 'Chamber, pixel layer, strips',
     'X/Y charge balance (PLAN_38): f = 0.487 / 0.531 det3/det2, σ68 0.07, flat in position and angle. '
     'QA-level, unaffected by the rebase.'),
    ('Sharing kernel, measured head-on', READY, 'SPS: c2/c1 = 0.45, ±1 delay',
     'sps_beam_test_26/analysis/sharing_kernel + angled_kernel. Shape decided (cascade beats delay). '
     'Open: τ and c2 are bounds, the window truncates the tail.'),
    ('Waveform forward-model reco', READY, 'Methods section exists',
     'WAVEFORM_FIRST_THREADING.md + THREADING_DISPLAYS. Open: production runs the delay branch, '
     'SPS prefers lp; per-plane c2/c1 ratio not built.'),
    ('Cosmic performance', REQUOTE, 'Angles, position, efficiency',
     'Values exist in FLEET_DIGEST.md (r06), but the r06 products are not on this machine. '
     'Re-quote from re-pulled products, never from PAPER_STATUS.'),
    ('Intrinsic resolution (SPS)', READY, 'det4: 176 µm at normal incidence',
     'spatial_resolution/report.html. Replaces PLAN_37: the bench residual is reference + multiple scattering.'),
    ('Gas: v(E), attachment, gap', OPEN, 'Drift scan not on r06',
     'Six det3 drift-scan rows (tier B) were never re-run on r06. PLAN_40 skeptic tests not marked done.'),
    ('Timing', OPEN, '33 ns, but from hits',
     'PLAN_42 measured detector σ_t = 33 ns from hit times. The waveform port is pending.'),
    ('Operations: HV, sparks, edge', READY, 'Fringe-field angle needs a re-check',
     'HV optima, sparks non-propagating, no post-spark dead time, waveform anatomy: all QA-level. '
     'The −3° edge tilt is a hits-derived angle.'),
]


def s_storyline(D):
    cards = []
    for i, (sec, c, line, tip) in enumerate(TOPICS):
        cards.append(
            f'<div{sd.tipattr(tip)} style="width:388px;height:330px;background:#fffdf9;border:1px solid {RULE};'
            f'border-top:8px solid {c};border-radius:14px;padding:22px 24px;display:flex;flex-direction:column;gap:10px">'
            f'<p style="font-size:20px;color:{MUT};font-weight:600">§{i + 1}</p>'
            f'<p style="font-size:28px;font-weight:600;line-height:1.15">{sec}</p>'
            f'<p style="font-size:22px;color:{INK};line-height:1.3;font-weight:600">{line}</p>'
            f'<p style="font-size:19px;color:{MUT};line-height:1.3">{tip}</p></div>')
    grid = f'<div style="display:flex;flex-wrap:wrap;gap:24px 37px;width:1664px">{"".join(cards)}</div>'
    body = sd.title('A cosmic + SPS paper in eight sections: five are ready to write',
                    'Proposed storyline. Colour is the state of the numbers each section needs.')
    body += sd.legend([('ready to write', READY, 'box'), ('values exist, must be re-quoted', REQUOTE, 'box'),
                       ('analysis still open', OPEN, 'box')])
    body += grid
    D.slide('storyline', body, '''
<p>The old narrative arc in PAPER_STATUS was: sharing measured → it breaks time-based tracking → unsharing and a geometry estimator fix it → the hybrid makes it uniform → sharing repaid as sub-pitch position. The waveform-first forward model absorbs the middle three steps, and the hybrid tracker is superseded (it should appear, if at all, as the motivation).</p>
<p>The proposed arc leads with the physics of the resistive layer, measured directly in the beam, then the reconstruction that models it, then the performance it gives on cosmics, with the SPS intrinsic resolution as the anchor. Gas and operations follow.</p>
<p>The ordering of §2 before §4 matters: the beam measurement is what justifies the kernel ordering the cosmic reconstruction ships with.</p>''',
            short='Storyline')


def s_board(D):
    rows = [
        ('1 · Charge spreading X vs Y', 'measured (hits constants)', 'replace with the SPS kernel', OPEN),
        ('2 · Unsharing correction', 'done + benchmarked', 'absorbed by the forward model', RETIRED),
        ('3 · Hybrid tracking', 'fleet-wide, 1.75–4.1°', 'superseded: wft 1.15–2.5°', RETIRED),
        ('4 · X/Y charge balance', 'measured', 'stands (QA-level)', READY),
        ('5 · Fringe field / edge', 'measured (det3)', 'efficiency stands; angle tilt re-check', OPEN),
        ('6 · HV scans + sparks', 'complete', 'stands (detection-level)', READY),
        ('7 · Drift gap / moisture', 'closure-based', 'PLAN_40 hardening still open', OPEN),
        ('8 · Drift velocity scan', 'done (hits)', 'tier-B rows not on r06', OPEN),
        ('9 · Spatial resolution', 'convolved only', 'SPS 176 µm answers it', READY),
        ('10 · Time resolution', 'measured (hits)', 'waveform port pending', OPEN),
    ]
    lab = {READY: 'ready', OPEN: 'open', RETIRED: 'retired', REQUOTE: 're-quote'}
    trs = [[a, f'<span style="color:{MUT}">{b}</span>', c,
            f'<span style="color:{col};font-weight:600">● {lab[col]}</span>'] for a, b, c, col in rows]
    body = sd.title('Of the ten audited topics, four stand, two are retired, and four are open',
                    'PAPER_STATUS.md topics, July verdict against today’s.')
    body += sd.table(['topic', 'July audit', 'today', 'state'], trs, size=25,
                     widths=[470, 420, 560, 180], align=['left', 'left', 'left', 'left'])
    D.slide('board', body, '''
<p><b>Retired</b> means the topic disappears as a paper section: the forward model does what the unsharing correction and the hybrid regression approximated. They can be mentioned as what the hits chain needed.</p>
<p><b>Open, in priority order</b> for next week: (8) run the six det3 drift-scan rows locally on r06 (they need the hits-chain alignment and event cache, so they cannot go to condor); (10) port the time resolution to waveforms; (5) re-measure the edge angle tilt with wft angles; (7) PLAN_40's three skeptic tests; (1) quote the SPS kernel instead of the hits sharing constants.</p>
<p>Two more things the cosmic-performance text must handle: det6's X plane is <i>not</i> dead (the old table says it is; wft gives σ_θ 2.2°), and det4's poor bench numbers come from 62 % of its area not amplifying, in fixed ~35 mm stripes (<code>sps_beam_test_26/det4_sps_assessment/</code>).</p>''',
            short='Topic board')


def s_angles(D, F):
    P = sd.Plot(1060, 620, x=(-0.6, 4.6), y=(0, 4), ylabel='σ_θ [deg]')
    P.yticks([(v, str(v)) for v in range(0, 5)])
    P.xticks([(i, DLAB[k]) for i, k in enumerate(DETS)])
    off = {('X', 'hits'): -0.3, ('X', 'wft'): -0.1, ('Y', 'hits'): 0.1, ('Y', 'wft'): 0.3}
    for i, k in enumerate(DETS):
        for pl in ('X', 'Y'):
            new, old = F[f'sigma_theta {pl}'][k]
            b = F[f'bias {pl} deg'][k][0]
            P.vbar(i + off[(pl, 'hits')], old, 36, HITS, opacity=0.55 if pl == 'Y' else 1,
                   tip=f'{DLAB[k]} {pl}, hits chain: σ_θ {old:.2f}°')
            P.vbar(i + off[(pl, 'wft')], new, 36, WFT, opacity=0.55 if pl == 'Y' else 1,
                   tip=f'{DLAB[k]} {pl}, waveform-first (r06): σ_θ {new:.2f}°, bias {b:+.2f}°',
                   label=f'{new:.1f}')
    ratios = [F[f'sigma_theta {pl}'][k][1] / F[f'sigma_theta {pl}'][k][0] for k in DETS for pl in 'XY']
    side = sd.col(
        sd.legend([('hits chain', HITS, 'box'), ('waveform-first, r06', WFT, 'box')]),
        sd.p('Solid bars are the X plane, pale bars Y.', 22, MUT),
        sd.p(f'Every plane improves, by ×{min(ratios):.2f} to ×{max(ratios):.1f}: halved on det3, least on '
             f'det6 Y and det4 Y. The bias is consistent with zero everywhere.'),
        sd.p(f'The improvement is not a tuning gain: the {sd.term("hits chain", G["hits"])} is biased by '
             'construction, and the forward model removes the bias.'),
        sd.callout('det3 and det2 are the headline chambers; det6 and det7 carry board-C/D sharing; det4 is '
                   'gain-limited and better shown through SPS.', GREY, 24),
        gap=22, w=520)
    body = sd.title('Waveform-first improves σ_θ on every plane',
                    f'{sd.term("σ_θ", G["sigt"])} per plane against the M3 reference, golden run per detector, '
                    f'{sd.term("r06", G["r06"])} calibration (det4: lp_t0p, det6: lp).')
    body += sd.row(P.svg('sigma theta per detector'), side, gap=72)
    D.slide('angles', body, '''
<p>Source: <code>mx_june_wft/FLEET_DIGEST.md</code> (written by <code>mx_june_wft/digest.py</code>, last committed 2026-08-19 with the r06 promotion), which compares both chains on the same events with the same M3 recipe (χ² &lt; 1, NClus = 4).</p>
<p>The bundles differ by detector: r06 (c2 = 0.6 c1) on det2, det3 and det7; <code>calib_bundle_lp_t0p</code> on det4 and <code>calib_bundle_lp</code> on det6, whose ratios were already below 1. Columns built on different bundles are not interchangeable; the paper table has to say which bundle each row used.</p>
<p>The r06 cost, from the paired bootstrap in <code>R06_GATE_2026-08-19.md</code>: det3 Y +0.06°, det7 X +0.08°, det7 Y +0.13° relative to the inverted kernel. The data prefer the inverted kernel <i>inside this model</i>; the proposed per-plane ratio is the identified fix and has not been built.</p>''',
            foot='Source: mx_june_wft/FLEET_DIGEST.md (r06 promotion, 2026-08-19).', short='Angles')


def s_position(D, F):
    panels = []
    P = sd.Plot(810, 520, x=(-0.6, 4.6), y=(0, 100), ylabel='rays with a track within 5 mm [%]',
                title='efficiency within 5 mm')
    P.yticks([(v, f'{v}') for v in (0, 25, 50, 75, 100)])
    P.xticks([(i, DLAB[k]) for i, k in enumerate(DETS)])
    for i, k in enumerate(DETS):
        new, old = F['within 5 mm %'][k]
        n = F['rays'][k][0]
        P.vbar(i - 0.17, old, 44, HITS, tip=f'{DLAB[k]} hits chain: {old:.1f} % of {n:.0f} rays')
        P.vbar(i + 0.17, new, 44, WFT, tip=f'{DLAB[k]} waveform-first: {new:.1f} % of {n:.0f} rays',
               label=f'{new:.0f}')
    panels.append(P.svg('within 5 mm'))
    Q = sd.Plot(810, 520, x=(-0.6, 4.6), y=(0, 1.2), ylabel='|r| to the M3 ray [mm]',
                title='position residual (includes the reference)')
    Q.yticks([(v / 10, f'{v / 10:.1f}') for v in range(0, 13, 2)])
    Q.xticks([(i, DLAB[k]) for i, k in enumerate(DETS)])
    Q.hline(0.225, PURPLE, label='binary, pitch/√12', tip=G['pitch'], anchor='end', where='below')
    for i, k in enumerate(DETS):
        c = F['core sigma r mm'][k][0]
        m = F['median r mm'][k][0]
        Q.vbar(i - 0.17, c, 44, WFT, tip=f'{DLAB[k]} core σ|r| {c:.2f} mm (wft)', label=f'{c:.2f}')
        Q.vbar(i + 0.17, m, 44, WFT, opacity=0.45, tip=f'{DLAB[k]} median |r| {m:.2f} mm (wft)')
    panels.append(Q.svg('core sigma'))
    body = sd.title('Position holds on det2/3 and recovers det4/6/7',
                    f'{sd.term("Within 5 mm", G["w5"])} (left) and the {sd.term("core σ", G["core"])} and median of |r| '
                    '(right, solid and pale). Same events as the angle slide.')
    body += sd.legend([('hits chain', HITS, 'box'), ('waveform-first', WFT, 'box')]) + sd.row(*panels, gap=44)
    gains = ', '.join(f'{DLAB[k]} +{F["within 5 mm %"][k][0] - F["within 5 mm %"][k][1]:.0f}'
                      for k in ('g_det4', 'g_det6_long', 'g_det7_long'))
    D.slide('position', body, f'''
<p>Gains in points of within-5 mm: {gains}. det3 and det2 were already at ~92–93 % and stay there. det4's low value is its geometry: 62 % of its active area does not amplify, in fixed stripes (<code>DET4_SPS_ASSESSMENT.md</code>).</p>
<p>All residuals here are detector ⊕ reference. The next slide is why the bench cannot measure the chamber's intrinsic position resolution, and what does.</p>
<p>Spark fractions in the same digest (det3 8 %, det2 10 %, det4 10 %, det6 22 %, det7 37 %) are identical between chains, as they should be: detection is still hits-defined.</p>''',
            short='Position')


def s_sps_res(D, S):
    res = S['res']
    fit = res['fit']
    P = sd.Plot(1000, 700, x=(-0.6, 5.6), y=(0, 0.8), ylabel='σ [mm]')
    P.yticks([(v / 10, f'{v / 10:.1f}') for v in range(0, 9, 2)])
    zl = []
    i = 0
    for coord, zs in res['zones'].items():
        for z in zs:
            zl.append((i, f'{coord[-1]} {z["pitch"]:g}'))
            P.vbar(i, z['sigma_res'], 34, GREY, opacity=0.55,
                   tip=f'{coord}, uRWELL pitch {z["pitch"]} mm: residual σ {1000 * z["sigma_res"]:.0f} µm, n = {z["n"]:,}')
            P.points([i], [z['sigma_det4']], BLUE, r=10,
                     tips=[f'{coord} pitch {z["pitch"]}: det4 after removing the pointing term '
                           f'({1000 * z["pointing"]:.0f} µm): {1000 * z["sigma_det4"]:.0f} µm'])
            i += 1
    bench = (sum(x * x for x in M3_POINT_MM) / 2) ** 0.5
    ms = MS_MRAD * MS_LEVER_MM / 1000
    tot = (fit['sigma_det4'] ** 2 + bench ** 2 + ms ** 2) ** 0.5
    P.xticks([(a, b) for a, b in zl] + [(5, 'bench')])
    P.vbar(5, tot, 34, ORANGE, opacity=0.35,
           tip=f'176 µm ⊕ M3 pointing {1000 * bench:.0f} µm ⊕ scattering {1000 * ms:.0f} µm = {1000 * tot:.0f} µm '
               f'(arithmetic, not a fit)')
    P.points([5, 5], list(BENCH_S68), ORANGE, r=10,
             tips=[f'det3 bench σ68 {v:.2f} mm, θ < 5° ({c}), the residual the SPS report compares with'
                   for v, c in zip(BENCH_S68, 'xy')])
    P.hline(fit['sigma_det4'], BLUE, label=f'det4 fit {1000 * fit["sigma_det4"]:.0f} ± '
                                            f'{1000 * fit["sigma_det4_err"]:.0f} µm', anchor='end', where='below')
    P.hline(0.225, PURPLE, label='binary 225 µm', tip=G['pitch'], anchor='end')
    side = sd.col(
        sd.legend([('residual to the uRWELL', GREY, 'box'), ('det4, reference removed', BLUE, 'dot'),
                   ('bench', ORANGE, 'dot')], size=22),
        sd.p(f'The uRWELL back plane has three pitches, so the reference’s own term is fitted out, not modelled: '
             f'det4 = {1000 * fit["sigma_det4"]:.0f} µm = {fit["f_back"]:.2f} × '
             f'{sd.term("pitch", G["pitch"])}.'),
        sd.p(f'The bench reference points with a core σ of {M3_POINT_MM[0]:.2f}/{M3_POINT_MM[1]:.2f} mm, without '
             f'scattering. ~{MS_MRAD} mrad over {MS_LEVER_MM} mm adds {1000 * ms:.0f} µm and closes the bench residual.'),
        sd.callout('PLAN_37 asked for this deconvolution. The beam answers it, but the scattering term is '
                   'inferred, not measured.', GOLD, 24),
        gap=20, w=580)
    body = sd.title(f'det4 resolves {1000 * fit["sigma_det4"]:.0f} µm; the bench sees its reference',
                    f'det4 in SPS H4, run 53, {res["run53"]["n"]:,} tracks at normal incidence, by uRWELL zone '
                    '(view, pitch in mm).')
    body += sd.row(P.svg('sps resolution'), side, gap=64)
    D.slide('sps-resolution', body, '''
<p>Source: <code>sps_beam_test_26/analysis/spatial_resolution/</code> (<code>resolution.py</code> → <code>results.json</code>; report <code>report.html</code>). The five zones are independent: three uRWELL pitches on X, two on Y. The joint fit over the back plane's pitches gives σ_det4 = 176 ± 10 µm with χ² = 2.8.</p>
<p>The bench bar is arithmetic: 176 µm ⊕ the M3 pointing core (quadrature mean of 0.21 and 0.24 mm, from the M3 self-resolution study, which excludes multiple scattering by construction) ⊕ 1.1 mrad × 558 mm. The SPS report's own caveat: <i>the scattering explanation is arithmetic, not a measurement</i>. A paper would want either a scattering simulation of the bench stack or the M3 inter-doublet kink (2.6 mrad measured) used directly.</p>
<p>Caveat for the paper: this is det4 at normal incidence in its live bands, in Ar/CF₄/iso. It establishes what the chamber design can do, not det3's number at cosmic angles.</p>''',
            short='SPS resolution')


def s_kernel(D, S, retired):
    rows = [(lab, q, 0.0, RED, f'{lab}: retired 2026-08-21, c2/c1 = {q:.2f}') for lab, q in retired]
    rows.append(('det6 (lp)', 0.82, 0.0, GREY, 'det6 calib_bundle_lp: c2/c1 = 0.82, physical, kept'))
    rows.append(('shipped r06', 0.60, 0.0, GREEN, 'calib_bundle_r06: c2 pinned to 0.6 c1 on det2, det3, det7'))
    bq, be, bn = S['bench']
    rows.append(('det3 bench, measured', bq, be, ORANGE,
                 f'near-vertical det3 cosmics, delay form: c2/c1 = {bq:.2f} ± {be:.2f}, n = {bn}'))
    for run, fld, n, q, e in S['beam']:
        rows.append((f'SPS {fld:.0f} V/cm', q, e, BLUE,
                     f'SPS H4 det4 head-on, {run}, Y view, delay form: c2/c1 = {q:.3f} ± {e:.3f}, n = {n:,}'))
    n = len(rows)
    P = sd.Plot(1040, 640, x=(0, 2.4), y=(n - 0.4, -0.6), xlabel='c2 / c1', margin=(24, 30, 92, 290))
    P.xticks([(v / 2, f'{v / 2:g}') for v in range(0, 5)] + [(2.4, '')])
    P.yticks([(i, r[0]) for i, r in enumerate(rows)])
    P.raw(f'<rect x="{P.X(1):.1f}" y="{P.y0}" width="{P.X(2.4) - P.X(1):.1f}" height="{P.ph}" '
          f'fill="{RED}" fill-opacity="0.06"/>', back=True)
    P.vline(1, RED, tip='The ±2 strip is reached only through the ±1, so it cannot receive more.')
    P.text(1.06, n - 1.6, 'c2 > c1:', 22, RED, weight=600)
    P.text(1.06, n - 1.0, 'not a resistive film', 22, RED)
    for i, (lab, q, e, c, tip) in enumerate(rows):
        if e:
            P.raw(sd.line(P.X(q - e), P.Y(i), P.X(q + e), P.Y(i), c, 3))
        P.points([q], [i], c, r=10, tips=[tip])
    beams = [q for _r, _f, _n, q, _e in S['beam']]
    fl = S['ang']['flat700']['x']
    side = sd.col(
        sd.p(f'Head-on in the beam, the cross-relation n₀∗W_d = n_d∗W₀ cancels the unknown drive signal, so the '
             f'{sd.term("kernel", G["kernel"])} is measured with no deconvolution.'),
        sd.p(f'c2/c1 = {min(beams):.2f}–{max(beams):.2f} at three drift fields; det3 bench cosmics agree within '
             f'errors. Every shipped bundle before 8-19 sat on the wrong side of 1.'),
        sd.p(f'The ±1 delay is {fl["pm1_shift_ns"]:.0f} ns on the flat mount and survives a 25.6° rotation.', 24, MUT),
        gap=20, w=560)
    body = sd.title(f'Beam: c2/c1 = {sum(beams) / len(beams):.2f}; every retired bundle had c2 > c1',
                    'Sharing ratio c2/c1: the retired bundles, the shipped one, and the two direct measurements.')
    body += sd.row(P.svg('kernel ratio'), side, gap=56)
    D.slide('kernel', body, '''
<p>Sources: <code>sps_beam_test_26/analysis/sharing_kernel/fit_kernel.json</code> (beam, Y view, delay form fitted on RAW run 71 at 243/156/95 V/cm) and <code>bench_kernel_y.json</code> (det3 near-vertical cosmics, tan θ &lt; 0.05). Retired ratios from <code>mx_june_wft/RETIRE_C2GTC1_2026-08-21.md</code> §2. Errors are the fits' bootstrap errors.</p>
<p>What the beam did <i>not</i> pin: the absolute time constant walks 629 → 1040 ns as the fit window grows, because 44–52 % of the central-strip amplitude is still present at the last sample. τ and c2 are bounds. The cascade (lp) form fits the cross-relation at 2.1 % residual against the shipped delay form's 4.2 %, but transplanted onto det3 it costs σ_θ(Y) 1.14 → 1.54°, so it is not adopted.</p>
<p>The cosmic χ² prefers c2 ≥ c1 inside the current model (R06_GATE). The paper should present that as evidence about the model, with the beam as the arbiter.</p>''',
            short='Kernel')


def s_where(D, F, bund):
    items = [
        (GREEN, 'In git', 'mx_june_wft/FLEET_DIGEST.md',
         'The only written record of the r06 golden numbers: every value on slides 5–6. Committed with the '
         'r06 promotion, 2026-08-19.'),
        (BLUE, 'On lxplus', '~/wft_campaign_r06/',
         '152 condor result tarballs of 2026-08-21 (events.parquet per sub-run × detector), 165 MB. The full '
         'June campaign on r06. Not yet pulled back.'),
        (GOLD, 'On this machine', '~/x17/cosmic_bench/Analysis',
         'r06 calibration bundles (8-19) are here, but the reco products, angles and efficiency JSONs are '
         'from 8-05, on the earlier bundles.'),
        (RED, 'Not found anywhere checked', 'golden r06 products',
         'The per-key angles/efficiency JSONs behind FLEET_DIGEST and the SUPERSEDED manifest. They were '
         'made on another machine.'),
    ]
    cards = ''.join(
        f'<div style="flex:1;background:#fffdf9;border:1px solid {RULE};border-top:8px solid {c};'
        f'border-radius:14px;padding:26px;display:flex;flex-direction:column;gap:14px">'
        f'<p style="font-size:30px;font-weight:600;color:{c}">{h}</p>'
        f'<p style="font-size:23px;font-family:IBM Plex Mono,monospace">{path}</p>'
        f'<p style="font-size:24px;color:{MUT};line-height:1.35">{txt}</p></div>' for c, h, path, txt in items)
    steps = [
        dict(label='pull', sub='rsync wft_campaign_r06 back; collect_results.py --promote', color=BLUE),
        dict(label='re-run 01–04', sub='alignment, efficiency, angles, maps on the five golden keys', color=BLUE),
        dict(label='digest', sub='digest.py must reproduce FLEET_DIGEST.md', color=GREEN,
             tip='If it does not, the paper numbers have no reproducible source. Settle this before writing.'),
        dict(label='tier-B rows', sub='six det3 drift-scan v-refits, locally on r06', color=GOLD),
    ]
    bl = ', '.join(f'{DLAB[k]} {bund.get(k, "?").replace("calib_bundle_", "")}' for k in DETS)
    body = sd.title('Cosmic numbers have a record, not local products',
                    'Where the r06 reconstruction lives, checked 2 Oct 2026.')
    body += f'<div style="display:flex;gap:28px">{cards}</div>'
    body += sd.p('Before quoting anything: make the record reproducible on this machine.', 28, INK, 600)
    body += sd.flow(steps, size=24)
    D.slide('where', body, f'''
<p>Checked on 2026-10-02: <code>find /media/dylan/data/x17/cosmic_bench/Analysis -path '*wft*' -newermt 2026-08-18</code> finds no angle, efficiency or events metadata files; sat_det3's <code>events_lp.meta.json</code> points at <code>calib_bundle_lp_sp0free</code> (8-05). <code>~/x17</code> is the same disk. On lxplus, <code>~/wft_campaign</code> holds the pre-fix 8-13 tarballs and <code>~/wft_campaign_r06</code> the 8-21 re-run; <code>/eos/user/d/dneff/x17</code> has no wft campaign output.</p>
<p>Bundles named in FLEET_DIGEST: {bl}.</p>
<p>The campaign packager has two known traps (<code>RETIRE_C2GTC1_2026-08-21.md</code> §6a–6b): it ran det4 without the t0 prior once ("t0p-adopted dets: none yet" is the tell), and a v-refit seeded from an r06 bundle used to write c2 = 0 with no ratio (fixed, tested). Expect bit-identity on det2/3/4/6; det7 differed on 3 of 8,082 events.</p>''',
            short='Where the numbers are')


def s_close(D):
    items = [
        ('The bench resolution is not decomposed', 'The scattering term behind the bench residual is inferred '
         'arithmetic. A bench-stack scattering simulation or the measured M3 kink would make it a result.'),
        ('One calibration, one model', 'r06 is better physics and a worse fit (Y up to +0.13°). The per-plane '
         'ratio and the lp-vs-delay branch are both unresolved and both would re-open the gates.'),
        ('The kernel tail is outside every window', 'τ and c2 are bounds. No beam for three years.'),
        ('Golden runs only', 'One run per detector is reconstructed to paper standard; HV-scan rows are '
         'stamped off-conditions (trend grade).'),
        ('det4 speaks through SPS', '62 % of its area does not amplify; its bench numbers are geometry.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:14px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:27px;font-weight:600;width:440px">{a}</p>'
                    f'<p style="font-size:23px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    dec = [('Freeze r06 as it stands?', 'Or build the per-plane ratio first. Write on r06 unless the ratio is '
            'a few days’ work.'),
           ('Which chambers in the main table', 'det3 + det2 headline; det6/7 as the board-C/D contrast; det4 via SPS.'),
           ('Journal', 'NIM A or JINST: length and figure budget follow from it.'),
           ('Outline source', 'Start from the MPGD26 running order and this storyline; its figure scripts exist.')]
    drows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:14px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:27px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:23px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in dec)
    body = (f'<div style="display:flex;gap:80px">'
            f'<div style="flex:1.25;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:50px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:50px;font-weight:600">Decisions before writing</h2>{drows}</div></div>')
    D.slide('close', body, '''
<p>Record of this audit: <code>mx_june_cosmic_qa/PAPER_STATUS.md</code>, section "2026-10-02 re-audit" at the top. The July audit below it is kept as written; its numbers are superseded.</p>
<p>Key inputs for writing: <code>RECONSTRUCTION_BASIS.md</code> (migration table), <code>mx_june_wft/FLEET_DIGEST.md</code>, <code>mx_june_wft/RETIRE_C2GTC1_2026-08-21.md</code>, <code>sps_beam_test_26/analysis/README.md</code> and its <code>sharing_kernel</code>, <code>angled_kernel</code> and <code>spatial_resolution</code> reports, <code>mpgd26/slides/RUNNING_ORDER.md</code>.</p>''',
            dark=True, short='Caveats & decisions')


# --------------------------------------------------------------------------- #
def build(out: Path) -> Path:
    F, bund = load_fleet()
    S = load_sps()
    retired = load_retired()
    D = sd.Deck('MX17 Paper Status',
                'Where the MX17 detector paper (June cosmic bench + SPS H4) stands on 2 Oct 2026: '
                'ready sections, numbers to re-quote, open analyses, and where the r06 products live.')
    s_cover(D, F, S)
    s_timeline(D)
    s_storyline(D)
    s_board(D)
    s_angles(D, F)
    s_position(D, F)
    s_sps_res(D, S)
    s_kernel(D, S, retired)
    s_where(D, F, bund)
    s_close(D)
    meta = dict(title='MX17 detector paper: where it stands',
                summary='Re-audit before writing: the cosmic + SPS analyses are done, but the July plan predates '
                        'the waveform-first rebase and the c2 > c1 retirement; five sections ready, the r06 '
                        'numbers need re-pulling.',
                tags='X17,detector,paper', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by '
                                     'nTof_x17/mx_june_cosmic_qa/make_paper_status_deck.py with slidedoc.py.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path,
                    default=Path(os.path.expanduser('~/x17/paper_status/mx17-detector-paper-status.html')))
    a = ap.parse_args()
    print('wrote', build(a.out))


if __name__ == '__main__':
    main()
