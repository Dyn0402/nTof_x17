#!/usr/bin/env python3
"""
make_deck.py -- the beam-off cosmic work as a figure-first slide note.

Reads only this package's outputs (inventory.py's CSVs, clock_match.py's
summary / pairs / drift / delta_b files, the beam-on DAQ-log excerpt) and the
pulse-match cache, so rerunning after the analysis moves numbers, charts and
claims together. Built with dylan-cern-site/scripts/slidedoc.py; meant to grow
as the cosmic work does.

    .venv/bin/python ntof_cosmics/make_deck.py [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH \\
        --slug beam-off-cosmics --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK)

RES = HERE / 'results'
CM = RES / 'clock_match'
STEM = 'run_149_cosbounce_cos_0000_224678'

#: one colour per thing, used on every slide
C_DREAM, C_NTOF = BLUE, ORANGE
C_CLEAN, C_EDGE, C_PRE = GREEN, GOLD, GREY
C_S1, C_S2, C_S3, C_S4 = GOLD, BLUE, PURPLE, GREEN

G = dict(
    singles=('Scintillator SINGLES: per arm, the analogue sum of a wall bar\'s two ends over '
             'threshold AND a plastic bar over threshold within 20 ns. This is what triggers '
             'DREAM; n_TOF records the same hits, so it can be rebuilt offline (fast_singles).'),
    psTime=('PKUP.psTime: the n_TOF time stamp of each bunch (UTC, ns, float64 -- so quantised '
            'to 256 ns at 1.8e18). Filled for every bunch even with no protons on target.'),
    tof='tof: time of a hit since the start of its n_TOF acquisition window, ns.',
    log=('The DREAM DAQ log line "Subrun started: <name>" -- the PC clock, to the millisecond. '
         'DREAM\'s timestamp counter starts at "go", a few seconds later.'),
    k=('k: the DREAM oscillator rate against psTime, between bunches. Fitted across the '
       'sub-run.'),
    kappa=('κ: the DREAM clock rate against the n_TOF digitiser clock, inside an 80 ms window. '
           'The beam-on calibration (run_79 ↔ 224572) calls it K = 110.4 ppm.'),
    delta=('δ_b: where bunch b\'s acquisition window really starts, relative to its psTime. '
           'With beam it is pinned by the gamma flash; without beam it has to be measured '
           'from the bunch\'s own matched triggers.'),
    loo=('Leave-one-out: each trigger is predicted from δ_b fitted on the OTHER triggers of '
         'its bunch, never itself, so the residual cannot be flattered by the fit.'),
    prod=('Production operating point: run_79\'s HV (resist A540/B540/C525/D520, drift 700 V) '
          'and readout (no ZS, latency 27, 20 samples × 60 ns). Only the trigger differs.'),
    edge=('Beam at the edges only: n_TOF pulses fall in the first or last 2 minutes of a '
          'sub-run -- beam leaving or returning around the automatically substituted cosmic '
          'run. An event-level veto on pulse time removes them.'),
)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load():
    run = pd.read_csv(RES / 'cosmic_runs.csv')
    S = json.loads((CM / f'summary_{STEM}.json').read_text())
    pairs = pd.read_csv(CM / f'pairs_{STEM}.csv')
    drift = pd.read_csv(CM / f'drift_pairs_{STEM}.csv')
    db = pd.read_csv(CM / f'delta_b_{STEM}.csv')
    return run, S, pairs, drift, db


def beam_on_latency() -> pd.DataFrame:
    """DAQ start latency on beam sub-runs: file-name anchor + pulse-match offset
    - log line - 0.829 s NXCALS lag  =  DREAM go on the psTime clock, after the
    log line. Sources: results/clock_match/beam_on_log_starts.txt (dream_daq.log
    lines and combined-file names, pulled from EOS 2026-10-02) and
    ntof_july_analysis/cache_pulse_match/."""
    lines = (CM / 'beam_on_log_starts.txt').read_text().splitlines()
    logs, anch, run = {}, {}, None
    for ln in lines:
        m = re.match(r'(run_\d+) (\S+ \S+) INFO: Subrun started: (\S+)', ln)
        if m:
            run = m[1]
            logs[(m[1], m[3])] = dt.datetime.strptime(m[2], '%Y-%m-%d %H:%M:%S,%f').timestamp()
            continue
        m = re.match(r'Mx17_(\S+)_datrun_(\d\d)(\d\d)(\d\d)_(\d\d)H(\d\d)_000', ln)
        if m:
            anch[(run, m[1])] = dt.datetime(2000 + int(m[2]), int(m[3]), int(m[4]),
                                            int(m[5]), int(m[6])).timestamp()
    rows = []
    cache = REPO / 'ntof_july_analysis' / 'cache_pulse_match'
    for key, t in sorted(logs.items()):
        f = cache / f'{key[0]}_{key[1]}.json'
        if not f.exists() or key not in anch:
            continue
        off = json.loads(f.read_text()).get('offset_s')
        if off is None:
            continue
        rows.append(dict(run=key[0], subrun=key[1], L=anch[key] + off - t - 0.829))
    d = pd.DataFrame(rows)
    d['ok'] = d.L.between(0, 20)
    return d


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, run, S):
    prod = run[run.era == 'production_point']
    tip_h = (f'{len(prod)} runs (run_80–159) at the production operating point; '
             f'{prod.events.sum() / 1e6:.2f} M triggers; all decoded on EOS.\n'
             f'{run[run.category == "production point, clean"].hours.sum():.1f} h never saw an n_TOF pulse.')
    tip_e = (f'{S["matched_loo"]} of {S["efficiency_denominator"]} DREAM triggers that fall inside an '
             f'n_TOF window match a wall×plastic singles within ±{S["window_ns"]:g} ns.\n'
             f'Beam-on reference: 96 %.')
    tip_r = (f'Leave-one-out residual, core MAD {S["res_core_mad_ns"]:.1f} ns. '
             f'Accidentals: {S["sideband"]} trigger in the 1–300 µs sideband → '
             f'{S["accidental_expected"]:.4f} expected in ±{S["window_ns"]:g} ns.')
    nums = ''.join([
        sd.bignum(f'{prod.hours.sum():.1f} h', 'of beam-off cosmics at the production point', DGREEN,
                  f'{len(prod)} runs, {prod.events.sum() / 1e6:.1f} M triggers, never analysed until now.',
                  tip=tip_h),
        sd.bignum(f'{100 * S["efficiency_loo"]:.0f}%', f'matched to n_TOF within ±{S["window_ns"]:g} ns',
                  DBLUE, 'of in-window triggers, with no beam and no flash to lock on (beam-on: 96 %).',
                  tip=tip_e),
        sd.bignum(f'{S["res_core_mad_ns"]:.0f} ns', 'match resolution (MAD)', '#f0a36b',
                  'leave-one-out, so nothing validates itself; accidentals unmeasurably small.',
                  tip=tip_r)])
    body = (sd.kicker(f'n_TOF EAR2 · X17 · beam-off cosmic runs · {dt.date.today():%-d %b %Y}')
            + '<h1 style="font-size:84px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1640px">'
              'Beam-off cosmics can be put on the n_TOF clock to ~10 ns — so straight-through '
              'particles can be studied with their scintillator times</h1>'
            + '<div style="flex:1"></div>'
            + f'<p style="font-size:28px;color:{DMUT}">First sub-run: run_149 cosbounce_cos_0000 ↔ n_TOF 224678. '
              'This note grows as the rest of the cosmic sample is matched.</p>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, '''
<p><b>Why cosmics now.</b> The ILL feasibility study (<code>x17_facility_search/ill/</code>) found cosmics a substantial background there, with two handles: the arm-to-arm time of flight and collinearity. At n_TOF, <code>sept26_prelim_analysis/tight_coincidence.py</code> already vetoes opposing pairs above 170° (<code>BACK_TO_BACK_DEG</code>), 40 % of the tight opposing sample, but nothing used the timing direction, and the beam-off runs were cut at stage 0 and never read.</p>
<p><b>What exists so far.</b> An inventory of all 47 beam-off runs (<code>ntof_cosmics/inventory.py</code>) and a clock match of DREAM cosmic triggers to n_TOF without beam (<code>ntof_cosmics/clock_match.py</code>), demonstrated on one 15-minute sub-run. Long-form reports: <code>ntof_cosmics/results/report.html</code> and <code>results/clock_match/report.html</code>.</p>''',
            dark=True, short='Answer')


def s_background(D):
    W, H = 1000, 720
    cx, cy = 500, 360
    o = []
    # arms: A left, C right (opposing), B top, D bottom
    arms = dict(A=(cx - 330, cy, 'v'), C=(cx + 330, cy, 'v'), B=(cx, cy - 290, 'h'), D=(cx, cy + 290, 'h'))
    for a, (x, y, ori) in arms.items():
        if ori == 'v':
            o.append(f'<rect x="{x - 14}" y="{y - 120}" width="28" height="240" fill="#dfe6f1" stroke="{RULE}"'
                     f'{sd.tipattr("Chamber " + a + ": micro-TPC (drift gap + x/y strips)")}/>')
            sx = x - 40 if a == 'A' else x + 28
            o.append(f'<rect x="{sx}" y="{y - 120}" width="12" height="240" fill="{GOLD}" fill-opacity=".55"'
                     f'{sd.tipattr("Arm " + a + " scintillators: wall + plastic. They give the time.")}/>')
            o.append(sd.T(x, y - 136, a, 30, INK, 'middle', 600))
        else:
            o.append(f'<rect x="{x - 120}" y="{y - 14}" width="240" height="28" fill="#dfe6f1" stroke="{RULE}"'
                     f'{sd.tipattr("Chamber " + a)}/>')
            sy = y - 40 if a == 'B' else y + 28
            o.append(f'<rect x="{x - 120}" y="{sy}" width="240" height="12" fill="{GOLD}" fill-opacity=".55"/>')
            o.append(sd.T(x + 140, y + 10, a, 30, INK, 'start', 600))
    o.append(f'<circle cx="{cx}" cy="{cy}" r="34" fill="#f3e3c9" stroke="{ORANGE}" stroke-width="2.5"'
             f'{sd.tipattr("³He capsule: where neutron captures, and so any real e⁺e⁻ pair, originate.")}/>')
    o.append(sd.T(cx + 10, cy - 46, '³He capsule', 22, ORANGE))
    # real pair: from capsule to A and C at ~150 deg
    for ang in (172, 22):
        r = math.radians(ang)
        o.append(sd.arrow(cx, cy, cx + 340 * math.cos(r), cy - 340 * math.sin(r), GREEN, 4, 16))
    o.append(sd.T(cx - 215, cy - 50, 'pair from the capsule', 22, GREEN, 'middle', 600,
                  tip='A real pair: two tracks from one vertex in the capsule, opening angle set by the '
                      'physics (X17: above 109°), arriving at both arms at the same time.'))
    # straight-through: one line across A and C, passing just below the capsule
    yl = lambda x: 490 - 160 * (x - 30) / 940            # noqa: E731
    o.append(sd.line(30, yl(30), 900, yl(900), RED, 4, '14 8'))
    o.append(sd.arrow(900, yl(900), 975, yl(975), RED, 4, 16))
    o.append(sd.T(150, yl(150) + 52, 'one particle straight through', 22, RED, 'start', 600,
                  tip='A cosmic, or a beam-related particle punching through: one straight line crossing '
                      'two opposing arms. It reads as a pair at ~180°.'))
    # handles
    def badge(x, y, n, tip):
        return (f'<circle cx="{x}" cy="{y}" r="22" fill="{INK}"{sd.tipattr(tip)}/>'
                + sd.T(x, y + 8, str(n), 24, '#fff', 'middle', 600))
    o.append(badge(650, yl(650) + 40, 1, 'Collinearity: one particle → opening angle ≈ 180°.'))
    o.append(badge(cx, yl(cx) + 2, 2, 'Capsule vertex: the line\'s closest approach to the capsule is '
                                          'flat across the acceptance, not peaked on it.'))
    o.append(badge(cx + 400, cy - 160, 3, 'Time of flight: the second arm fires ~2–3 ns after the first, '
                                          'always in the same order for cosmics.'))
    schem = sd.svg(W, H, ''.join(o), 'set-up schematic with a pair and a straight-through particle')
    right = sd.col(
        sd.p('<b>One background, whatever its source.</b> A single particle crossing the set-up in a '
             'straight line, cosmic or beam punch-through, is the same category for X17.', 28),
        sd.callout('<b>1 · Collinearity</b>, opening angle ≈ 180°. In use already: '
                   + sd.term('back_to_back', 'tight_coincidence.BACK_TO_BACK_DEG = 170°: 40 % of the tight '
                                             'opposing sample, inside the X17 region.')
                   + '. The cosmic runs can set the cut from data.', RED, 25),
        sd.callout('<b>2 · Capsule vertex</b>: through-goers pass the capsule only as often as the '
                   'acceptance allows; real pairs come from it.', ORANGE, 25),
        sd.callout('<b>3 · Time-of-flight sign</b>: a ~2–3 ns lag, below the ~7 ns Δt resolution '
                   'per event, but visible as a shift on the >170° sample. '
                   '<b>Needs the clock match in this note.</b>', BLUE, 25),
        sd.p(f'Hover the {sd.term("dotted terms", "Like this one. Charts work the same way: hover points, bars and lines.")}'
             ' and chart points for definitions and numbers.', 22, MUT),
        gap=22, w=600)
    body = sd.title('A straight-through particle mimics a back-to-back pair; three handles separate it',
                    'Schematic top view, not to scale. A and C oppose; B and D are perpendicular.')
    body += sd.row(schem, right, gap=56)
    D.slide('background', body, '''
<p>The framing (Dylan, 2026-10-02): it does not matter whether a through-going track is a cosmic or beam background. If a single particle crosses from one side to the other in a straight line, it is one category, and the beam-off runs are the clean, high-statistics sample of that topology. They give its signature and the efficiency of any veto, not a separate background to subtract.</p>
<p>Handle 2 should use pointing estimators (the closest approach of the line to the capsule), not the shape of a position distribution: CLAUDE.md, "Take positions from pointing".</p>
<p>Handle 3 needs scintillator times for cosmic events, which only n_TOF records. Hence the rest of this note.</p>''',
            short='The background')


def s_inventory(D, run):
    t0 = dt.datetime(2026, 7, 18).timestamp()
    run = run.copy()
    run['day'] = [(dt.datetime.strptime(s, '%Y-%m-%d %H:%M').timestamp() - t0) / 86400 for s in run.start]
    col = {'production point, clean': C_CLEAN, 'production point, beam at the edges only': C_EDGE,
           'pre-production (HV scan/ladder)': C_PRE, 'pre-production (fixed HV)': C_PRE, 'empty': RULE}
    P = sd.Plot(1080, 600, x=(0, 23), y=(0, 24), xlabel='date (2026)', ylabel='hours per run')
    P.xticks([(d, (dt.date(2026, 7, 18) + dt.timedelta(days=d)).strftime('%-d %b')) for d in range(0, 24, 3)])
    P.yticks([(v, str(v)) for v in (0, 4, 8, 12, 16, 20, 24)])
    P.vline((dt.datetime(2026, 7, 26, 18).timestamp() - t0) / 86400, MUT,
            label='run_79: production starts',
            tip='run_79, 26 July: first long run at the final configuration. Cosmic runs after it '
                'share its HV and readout.')
    for r in run.itertuples():
        tip = (f'run_{r.run} · {r.start}\n{r.category}\n{r.hours:.2f} h, {r.events:,} triggers '
               f'({r.rate_hz:.1f} Hz)\nn_TOF beam pulses: {r.ntof_beam} '
               f'(inside the running: {r.ntof_beam_interior})\nn_TOF recording, no protons: '
               f'{r.ntof_quiet_h:.2f} h')
        P.vbar(r.day, max(r.hours, 0.15), 9, col[r.category], tip=tip)
    for rn in (149, 133, 89, 103):
        r = run[run.run == rn].iloc[0]
        P.text(r.day, r.hours + 0.6, f'run_{rn}', 20, INK, 'middle')
    cat = run.groupby('category').hours.sum()
    rows = [('clean', cat.get('production point, clean', 0), C_CLEAN, G['prod'] + '\nNo n_TOF pulse at all.'),
            ('beam at edges', cat.get('production point, beam at the edges only', 0), C_EDGE, G['edge']),
            ('pre-production', cat.filter(like='pre-production').sum(), C_PRE,
             'run_54–74: other HV and timing (latency 35, 32 samples), mostly resist-HV ladders.')]
    bars = sd.hbars(rows, vmax=40, width=300, label_w=200, fmt=lambda v: f'{v:.1f} h', size=24)
    inside = run[run.ntof_beam_interior > 0].run.tolist()
    right = sd.col(sd.p('<b>Hours by category</b>', 26), bars,
                   sd.callout(f'Beam inside the running: only {", ".join(f"run_{r}" for r in inside)}. '
                              'Everywhere else the pulses sit at run edges.', C_EDGE, 24),
                   sd.callout(f'Trigger: {sd.term("scintillator singles", G["singles"])}, veto open, '
                              f'~{run[run.era == "production_point"].events.sum() / run[run.era == "production_point"].hours.sum() / 3600:.0f} Hz. '
                              'One arm is enough to fire it.', BLUE, 24),
                   gap=20, w=560)
    prod = run[run.era == 'production_point']
    body = sd.title(f'{len(prod)} cosmic runs at the production point: {prod.hours.sum():.1f} h, '
                    'all decoded, never analysed',
                    'Every beam-off run of the campaign; bar height = hours, colour = category.')
    body += sd.legend([('production point, clean', C_CLEAN, 'box'), ('beam at the edges only', C_EDGE, 'box'),
                       ('pre-production', C_PRE, 'box')]) + sd.row(P.svg('cosmic run timeline'), right, gap=40)
    D.slide('inventory', body, '''
<p>Source: <code>ntof_cosmics/inventory.py</code> → <code>results/cosmic_runs.csv</code> (47 rows) and <code>cosmic_subruns.csv</code> (273). Inputs, none typed by hand: the frozen EOS survey (<code>dylan-cern-site/data/x17-runs.json</code>), each run's <code>run_config.json</code>, the per-minute <code>beam_class</code> slow-control log (n_TOF-destined pulses only), the match ledger and the n_TOF completed ledger.</p>
<p>The cosmic runs were substituted automatically whenever the beam dropped, which is why they are many and short, and why beam appears at their edges. The beam flag has one-minute resolution: a sub-run shorter than ~6 min has almost no "interior", so run_157 (0.031 Hz residual beam across 7 minutes) is classed "edges". Any analysis must veto by pulse time, not by this flag.</p>
<p>Conditions: everything at the production point is after the 23-July noise change. run_80 still has chamber A x connector 8 dead (live from run_83). run_148 is empty. PLAN §S4 had named run_83 and run_146 for the cosmic rate: 0.6 h between them. run_149 alone is 21.8 h.</p>''',
            foot='All 47 beam-off runs; hours from each sub-run\'s run_time, beam from n_TOF\'s per-minute beam_class log.',
            short='Inventory')


def s_quiet(D, run, S):
    # timing diagram, illustrative: real period / window / trigger rate
    W, H = 1664, 330
    x0, x1 = 60, 1620
    span = 2.6                                         # seconds shown
    X = lambda t: x0 + (x1 - x0) * t / span            # noqa: E731
    o = []
    yN, yD = 90, 230
    o.append(sd.T(x0 - 10, yN + 8, 'n_TOF', 24, C_NTOF, 'end', 600))
    o.append(sd.T(x0 - 10, yD + 8, 'DREAM', 24, C_DREAM, 'end', 600))
    o.append(sd.line(x0, yN + 30, x1, yN + 30, MUT, 1.5) + sd.line(x0, yD + 30, x1, yD + 30, MUT, 1.5))
    per, win = 0.5009, 0.080
    starts = [0.12 + i * per for i in range(6) if 0.12 + i * per < span]
    for s in starts:
        o.append(f'<rect x="{X(s):.1f}" y="{yN - 20}" width="{X(s + win) - X(s):.1f}" height="50" '
                 f'fill="{C_NTOF}" fill-opacity=".35" stroke="{C_NTOF}"'
                 f'{sd.tipattr("One n_TOF acquisition window: 80 ms, starting near its psTime. Free-running, 0.5009 s apart.")}/>')
        o.append(sd.line(X(s), yN - 34, X(s), yN + 30, C_NTOF, 2.5))
    o.append(sd.T(X(starts[0]), yN - 42, 'psTime', 20, C_NTOF, 'middle', tip=G['psTime']))
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.exponential(1 / 25.0, 120))
    t = t[t < span]
    for v in t:
        inside = any(s <= v <= s + win for s in starts)
        c = C_DREAM if inside else '#9db6d6'
        o.append(sd.line(X(v), yD - 18, X(v), yD + 30, c, 4 if inside else 2.5))
        if inside:
            o.append(sd.line(X(v), yN + 30, X(v), yD - 18, C_DREAM, 1.2, '4 5'))
    for k in range(6):
        o.append(sd.T(X(k * 0.5), yD + 62, f'{k * 0.5:.1f} s', 20, MUT))
    o.append(sd.T(x1, 22, 'illustration: real period, window and trigger rate', 20, MUT, 'end'))
    diag = sd.svg(W, H, ''.join(o), 'n_TOF free-running windows vs DREAM triggers')

    q = run[run.ntof_quiet_h > 0].sort_values('ntof_quiet_h', ascending=False)
    rows = [(f'run_{r.run}', r.ntof_quiet_h, C_NTOF,
             f'run_{r.run}: n_TOF recording with no protons for {r.ntof_quiet_h:.2f} h of its {r.hours:.2f} h')
            for r in q.itertuples()]
    bars = sd.hbars(rows[:6], vmax=9, width=520, h=28, label_w=140, fmt=lambda v: f'{v:.2f} h', size=23)
    duty = 0.080 / 0.5009
    txt = sd.col(
        sd.p(f'With no protons, n_TOF <b>free-runs</b>: an 80 ms window every 0.5009 s, a '
             f'<b>{100 * duty:.0f} % duty cycle</b>. PKUP still stamps every window with a '
             f'{sd.term("psTime", G["psTime"])}.', 26),
        sd.p(f'So about {100 * duty:.0f} % of DREAM\'s 25 Hz cosmic triggers fall inside a window, and '
             'n_TOF records the same particle in its walls and plastics: the only place a cosmic gets a '
             'scintillator time.', 26),
        gap=18, w=820)
    body = sd.title(f'n_TOF kept recording with no beam for {run.ntof_quiet_h.sum():.0f} h, '
                    'and catches ~16 % of the cosmic triggers',
                    'Top: how the two DAQs overlap. Bottom right: where n_TOF was recording, per cosmic run.')
    body += diag + sd.row(txt, sd.col(bars, gap=8), gap=60)
    D.slide('quiet', body, f'''
<p>n_TOF's own run log does not say "random trigger"; it shows as runs whose bunches carry no beam: <code>PulseIntensity</code> = 0, <code>user</code> and <code>lsaCycle</code> = −1, and ~50 kB per bunch against ~10 MB with beam. <code>inventory.py</code> classes an n_TOF run as recording-without-protons below 0.5 MB/bunch, a gap of two decades. Runs 224678–687 (with run_149) and 224608–613 (with run_103) are the bulk.</p>
<p>The window length is read from the data: the largest wall <code>tof</code> in 224678 is 79.999 ms. The period is the spacing of consecutive psTime values, 0.50077–0.50099 s.</p>
<p>{S["n_bunches_no_pstime"]} of {S["n_bunches"] + S["n_bunches_no_pstime"]} bunches of 224678 have psTime = 0 and are dropped for now; the period is regular enough to recover them from their neighbours.</p>''',
            short='n_TOF without beam')


def s_clocks(D, S):
    W, H = 1664, 420
    o = []
    yD, yN = 110, 290
    x0 = 160
    o.append(sd.T(x0 - 20, yD + 8, 'DREAM', 26, C_DREAM, 'end', 600))
    o.append(sd.T(x0 - 20, yN + 8, 'n_TOF', 26, C_NTOF, 'end', 600))
    o.append(sd.line(x0, yD, 1640, yD, C_DREAM, 2) + sd.line(x0, yN, 1640, yN, C_NTOF, 2))
    # DREAM: log line, go, trigger
    xl, xg, xt = 220, 520, 1240
    o.append(sd.line(xl, yD - 40, xl, yD + 14, MUT, 3))
    o.append(sd.T(xl, yD - 50, 'log line', 22, MUT, tip=G['log']))
    o.append(sd.line(xg, yD - 40, xg, yD + 14, C_DREAM, 3))
    o.append(sd.T(xg, yD - 50, 'go (timestamp = 0)', 22, C_DREAM))
    o.append(sd.arrow(xl, yD - 16, xg, yD - 16, GOLD, 3, 12))
    o.append(sd.T((xl + xg) / 2, yD + 40, 'S ≈ 6.45 s', 24, GOLD, 'middle', 600,
                  tip='Translation S: start-up latency plus any PC-to-n_TOF clock offset. Per sub-run.'))
    o.append(sd.arrow(xg, yD - 16, xt, yD - 16, C_S2, 3, 12))
    o.append(sd.T((xg + xt) / 2, yD + 40, 'td · (1 + k)', 24, C_S2, 'middle', 600, tip=G['k']))
    o.append(f'<circle cx="{xt}" cy="{yD}" r="12" fill="{C_DREAM}"/>')
    o.append(sd.T(xt, yD - 30, 'cosmic trigger', 22, C_DREAM, 'middle', 600))
    # n_TOF: psTime, window start, hit
    xp, xw = 820, 900
    o.append(sd.line(xp, yN - 14, xp, yN + 40, C_NTOF, 3))
    o.append(sd.T(xp, yN + 66, 'psTime_b', 22, C_NTOF, tip=G['psTime']))
    o.append(f'<rect x="{xw}" y="{yN - 26}" width="680" height="52" fill="{C_NTOF}" fill-opacity=".14" '
             f'stroke="{C_NTOF}"/>')
    o.append(sd.T(xw + 670, yN - 36, '80 ms window', 22, C_NTOF, 'end'))
    o.append(sd.arrow(xp, yN + 14, xw, yN + 14, C_S4, 3, 10))
    o.append(sd.T((xp + xw) / 2 - 4, yN - 40, 'δ_b', 26, C_S4, 'middle', 600, tip=G['delta']))
    o.append(sd.arrow(xw, yN - 4, xt, yN - 4, C_S3, 3, 12))
    o.append(sd.T((xw + xt) / 2, yN - 40, 'tof · (1 − κ)', 24, C_S3, 'middle', 600, tip=G['kappa']))
    o.append(f'<circle cx="{xt}" cy="{yN}" r="12" fill="{C_NTOF}"/>')
    o.append(sd.T(xt + 20, yN + 66, 'wall × plastic singles', 22, C_NTOF, 'start', 600, tip=G['singles']))
    o.append(sd.line(xt, yD + 14, xt, yN - 14, INK, 2, '6 6'))
    o.append(sd.T(xt + 14, (yD + yN) / 2 + 8, 'same particle: same instant', 22, INK, 'start'))
    diag = sd.svg(W, H, ''.join(o), 'the two clocks and the map between them')
    steps = [
        dict(label='1 · Coarse', sub='S on a 60 s slice; ±60 s search, 1 ms bins', color=C_S1,
             tip='Histogram every n_TOF singles − DREAM trigger difference. 60 s is short enough that the '
                 'rate cannot smear a 1 ms peak.'),
        dict(label='2 · Drift', sub='S + k·td across the sub-run; windows 5 ms → 200 µs', color=C_S2, tip=G['k']),
        dict(label='3 · κ', sub='bunches with two matched triggers: δ_b cancels', color=C_S3, tip=G['kappa']),
        dict(label='4 · δ_b', sub='per bunch, leave-one-out; accept ±50 ns', color=C_S4,
             tip=G['delta'] + '\n\n' + G['loo']),
    ]
    body = sd.title('Two clocks, one particle: the beam-on map with the flash replaced by psTime',
                    'What is unknown, and the order the stages solve it in — each opens a window only as wide '
                    'as the previous one\'s error.')
    body += diag + sd.flow(steps, size=25)
    D.slide('clocks', body, '''
<p><b>DREAM.</b> Absolute time = the go instant + <code>timestamp</code> × 10 ns, from FEU 1's <code>decoded_root</code> (not <code>combined_hits</code>, which keeps only events with Micromegas activity). Go is not logged; the DAQ log's "Subrun started" line (PC clock, ms) is the anchor and S absorbs the difference.</p>
<p><b>n_TOF.</b> Absolute time = psTime of the bunch + δ_b + tof. The stored <code>tflash</code> is meaningless without beam and is not used, which has a side effect: raw tof has no common zero across detector trees (next-to-last slide).</p>
<p><b>The beam-on map</b> (<code>ntof_dream_merge/DREAM_NTOF_CALIBRATION.md</code>) is t_nTOF = t_DREAM(1 + K) + T0 + a_arm + δa_b + δk_b·t, in time since the flash. Here the flash is replaced by psTime, and δa_b (6.5 ns RMS with beam) becomes δ_b (tens of µs).</p>
<pre><code>t_DREAM_on_nTOF = t_log + S + td·(1 + k)
off             = t_DREAM_on_nTOF − psTime_b = δ_b + tof·(1 − κ)</code></pre>''',
            short='Two clocks')


def s_coarse(D, S, lat):
    h = S['coarse']['hist_1ms']
    full = np.array(h['full_100ms'])
    x = (h['full_lo_ns'] + 1e8 * (np.arange(full.size) + 0.5)) / 1e9
    P = sd.Plot(1000, 560, x=(-60, 60), y=(450, 900), xlabel='n_TOF − DREAM − log line  [s]',
                ylabel='pairs per 100 ms')
    P.xticks([(v, f'{v:+d}' if v else '0') for v in range(-60, 61, 20)])
    P.yticks([(v, str(v)) for v in (500, 600, 700, 800, 900)])
    good = lat[lat.ok]
    P.band([good.L.min(), good.L.max()], [450, 450], [900, 900], GOLD, 0.15,
           tip=f'Beam-on prior: {len(good)} sub-runs put DREAM go {good.L.min():.1f}–{good.L.max():.1f} s '
               'after the log line, on the psTime clock.')
    xs, ys = sd.step_xy(list(np.r_[x - 0.05, x[-1] + 0.05]), list(full))
    P.raw(sd.poly([P.X(v) for v in xs], [P.Y(v) for v in ys], INK, 1.6))
    S1 = S['coarse']['S_ns'] / 1e9
    k = int(np.argmax(full))
    P.points([x[k]], [full[k]], RED, r=9,
             tips=[f'peak bin: {full[k]} pairs per 100 ms at {x[k]:+.2f} s\n1 ms scan: '
                   f'{S["coarse"]["steps"][0]["peak"]} pairs over a median of '
                   f'{S["coarse"]["steps"][0]["median_bin"]:.0f}'])
    P.text(x[k] + 2, full[k] - 5, f'S = {S1:+.3f} s', 22, RED, 'start', 600)
    left = P.svg('coarse scan')
    z = np.array(h['zoom'])
    xz = (h['zoom_lo_ns'] + 1e6 * (np.arange(z.size) + 0.5)) / 1e6 - S['coarse']['S_ns'] / 1e6
    Q = sd.Plot(560, 560, x=(-20, 20), y=(0, 240), xlabel='offset from S  [ms]', ylabel='pairs per 1 ms')
    Q.xticks([(v, str(v)) for v in (-20, -10, 0, 10, 20)]).yticks([(v, str(v)) for v in (0, 60, 120, 180, 240)])
    for xv, zv in zip(xz, z):
        Q.vbar(xv, zv, 11, C_S1 if abs(xv) < 0.6 else GREY, tip=f'{xv:+.0f} ms: {zv} pairs')
    right = Q.svg('coarse zoom')
    # latency strip
    W2 = 1664
    o = [sd.T(0, 26, 'beam-on DAQ start latency, per sub-run:', 22, INK, 'start', 600)]
    X = lambda v: 520 + (v + 110) / 140 * 1100           # noqa: E731
    o.append(sd.line(X(-110), 60, X(30), 60, MUT, 1.5))
    for v in (-100, -60, -20, 0, 10, 20, 30):
        o.append(sd.T(X(v), 92, f'{v:+d} s' if v else '0', 19, MUT))
    for r in lat.itertuples():
        c = GOLD if r.ok else GREY
        o.append(f'<circle cx="{X(r.L):.1f}" cy="60" r="9" fill="{c}" fill-opacity=".85"'
                 f'{sd.tipattr(f"{r.run} {r.subrun}: L = {r.L:+.2f} s" + ("" if r.ok else chr(10) + "pulse-match lock off by a supercycle step (cache predates the 2026-08-12 fix)"))}/>')
    o.append(f'<circle cx="{X(S1):.1f}" cy="60" r="11" fill="none" stroke="{RED}" stroke-width="3"'
             f'{sd.tipattr(f"this cosmic sub-run: S = {S1:+.3f} s")}/>')
    strip = sd.svg(W2, 100, ''.join(o), 'latency strip')
    body = sd.title(f'One translation stands out of ±60 s: S = {S1:+.3f} s, where beam-on runs predict',
                    'First 60 s of run_149 cos_0000: every n_TOF singles minus every DREAM trigger.')
    body += sd.row(left, right, gap=60) + strip
    D.slide('coarse', body, f'''
<p>{S["coarse"]["n_dream_slice"]} DREAM triggers against every singles candidate within ±60 s: the flat floor is the 0.5 s n_TOF comb against uncorrelated triggers; the slow shape is the overlap of the slice with the run's span. The peak survives a zoom from 1 ms to 10 µs to 100 ns bins before the rate starts to smear it.</p>
<p><b>The prior.</b> On beam sub-runs DREAM's start is known from the pulse lock: file-name minute + <code>pulse_match</code> offset − log line − 0.829 s (the NXCALS publication lag) puts go on the psTime clock. {int(lat.ok.sum())} sub-runs give {lat[lat.ok].L.min():.1f}–{lat[lat.ok].L.max():.1f} s; {int((~lat.ok).sum())} land at −100 s or +21–24 s, which are supercycle mislocks in a cache that predates the lock fix (<code>ntof_processing/join_mislock/</code>). Grey dots on the strip.</p>
<p>Sources: <code>results/clock_match/beam_on_log_starts.txt</code>, <code>ntof_july_analysis/cache_pulse_match/</code>.</p>''',
            short='1 · Coarse')


def s_drift(D, S, drift):
    k = S['drift']['k']
    dS = (S['drift']['S_ns'] - S['coarse']['S_ns']) / 1e6
    d = drift[drift.r_us.abs() < 5000]
    P = sd.Plot(1120, 640, x=(0, 920), y=(-5, 5), xlabel='DREAM time since go  [s]',
                ylabel='n_TOF − DREAM − S  [ms]')
    P.xticks([(v, str(v)) for v in range(0, 901, 150)]).yticks([(v, f'{v:+d}' if v else '0') for v in range(-5, 6)])
    P.raw(''.join(f'<circle cx="{P.X(a):.1f}" cy="{P.Y(b / 1e3):.1f}" r="2.6" fill="{GREY}" fill-opacity=".55"/>'
                  for a, b in zip(d.t_dream_s, d.r_us)), back=True)
    tt = [0, 900]
    P.line(tt, [dS + k * t * 1e3 for t in tt], C_S2, 4, markers=False,
           tip=f'S + k·t: S = {S["drift"]["S_ns"] / 1e9:+.6f} s, k = {k * 1e6:+.3f} ppm')
    steps = S['drift']['steps']
    rows = [[f'±{s["half_ns"] / 1e3:.0f} µs', f'{s["n_core"]:,}', f'{s["k"] * 1e6:+.3f}',
             f'{s["core_sigma_ns"] / 1e3:.0f} µs'] for s in steps]
    tab = sd.table(['window', 'pairs in core', 'k [ppm]', 'core σ'], rows, size=23)
    right = sd.col(
        sd.p(f'With S fixed, a ±5 ms window around every trigger follows a <b>straight line</b>: DREAM runs '
             f'<b>{abs(k) * 1e6:.1f} ppm fast</b> against psTime, '
             f'{abs(k) * 900 * 1e3:.1f} ms over the 15-minute sub-run.', 26),
        tab,
        sd.callout(f'The core stops narrowing at ~{steps[-1]["core_sigma_ns"] / 1e3:.0f} µs: '
                   'that floor is not the rate. It is the window-start jitter of slide 9.', C_S4, 24),
        gap=22, w=520)
    body = sd.title(f'The drift is {abs(k) * 1e6:.1f} ppm, not the in-window 110 ppm: a line carries S '
                    'across the sub-run', 'Every pair within ±5 ms of the coarse translation, all 900 s.')
    body += sd.row(P.svg('drift'), right, gap=40)
    D.slide('drift', body, f'''
<p>The windows shrink in three steps (5 ms, 1 ms, 200 µs); at each the robust line (3σ clipping, four passes) is refitted on the pairs inside it. The fit converges by the second step.</p>
<p>Two rates, not one: between bunches DREAM drifts {k * 1e6:+.2f} ppm against psTime; inside a window it runs {S["kappa"]["kappa"] * 1e6:.0f} ppm against the n_TOF digitiser (κ, next slide). The beam-on calibration only ever saw the second, because with beam each burst is re-locked to its own flash.</p>
<p>The line is not the whole story: the per-bunch offsets (slide 9) show a slow wander of ~{S["delta_b_slow_range_ns"] / 1e3:.0f} µs, the curvature a straight line misses. A smooth drift model is on the list.</p>''',
            foot='run_149 cosbounce_cos_0000 ↔ n_TOF 224678 · grey: every (DREAM trigger, singles) pair within ±5 ms.',
            short='2 · Drift')


def s_kappa(D, S, pairs):
    two = pairs[pairs.n_b == 2].sort_values(['bunch', 'tof'])
    g = two.groupby('bunch')
    f1, f2 = g.nth(0).set_index('bunch'), g.nth(1).set_index('bunch')
    du = (f2.u - f1.u).to_numpy() / 1e3
    dtof = (f2.tof - f1.tof).to_numpy() / 1e6
    kap = S['kappa']['kappa']
    m = np.abs(du + kap * dtof * 1e3) < 2.0
    P = sd.Plot(1000, 640, x=(0, 80), y=(-10, 1), xlabel='tof separation of the two triggers  [ms]',
                ylabel='difference in (off − tof)  [µs]')
    P.xticks([(v, str(v)) for v in range(0, 81, 10)]).yticks([(v, str(v)) for v in range(-10, 2, 2)])
    P.line([0, 80], [0, -110.4e-6 * 80e3], GREY, 3, '10 8', markers=False,
           tip='Beam-on K = 110.4 ppm (run_79 ↔ 224572, DREAM_NTOF_CALIBRATION.md)')
    P.line([0, 80], [0, -kap * 80e3], C_S3, 3.5, markers=False, tip=f'fitted κ = {kap * 1e6:.1f} ppm')
    P.raw(''.join(f'<circle cx="{P.X(a):.1f}" cy="{P.Y(b):.1f}" r="4" fill="{C_S3 if ok else RED}" '
                  f'fill-opacity=".6"/>' for a, b, ok in zip(np.abs(dtof), du * np.sign(dtof), m)))
    P.text(46, -kap * 46e3 - 1.3, f'κ = {kap * 1e6:.1f} ppm', 23, C_S3, 'start', 600)
    P.text(40, -110.4e-6 * 40e3 + 0.7, 'beam-on 110.4', 21, GREY, 'start')
    mad = S['kappa']['mad_by_sep']
    rows = [(k.replace('ms', ' ms'), v, C_S3, f'{k}: MAD of the two-trigger residual about the κ line = {v:.1f} ns')
            for k, v in mad.items()]
    bars = sd.hbars(rows, vmax=16, width=240, label_w=150, fmt=lambda v: f'{v:.0f} ns', size=23)
    right = sd.col(
        sd.p(f'Two triggers in the <b>same</b> bunch share δ_b and psTime, so their difference sees only '
             f'the in-window rate: {S["kappa"]["n_bunches"]} such bunches give '
             f'<b>κ = {kap * 1e6:.1f} ppm</b>.', 26),
        sd.p(f'About them, the pair difference has σ = {S["kappa"]["sigma_pair_ns"]:.0f} ns, '
             f'i.e. ~{S["kappa"]["sigma_pair_ns"] / math.sqrt(2):.0f} ns per trigger.', 26),
        sd.p('<b>Residual MAD by separation</b>', 24), bars,
        sd.callout('The mild growth with separation is the per-bunch rate jitter (~1 ppm with beam), '
                   'not yet fitted.', C_S3, 23),
        gap=18, w=600)
    body = sd.title(f'Inside a window the clocks differ by κ = {kap * 1e6:.1f} ppm, within '
                    f'{abs(kap * 1e6 - 110.4):.0f} ppm of beam-on',
                    'Bunches holding exactly two matched triggers: their difference against their tof separation.')
    body += sd.row(P.svg('kappa'), right, gap=50)
    D.slide('kappa', body, f'''
<p>off − tof for a real pair is δ_b − κ·tof. Taking the difference of the two triggers of one bunch removes δ_b, and the slope against their tof separation is −κ. Robust line, 3σ clipping; red points are the clipped ones (accidental partners).</p>
<p>The beam-on calibration notes that K is per (DREAM run, n_TOF processing) pair and must be re-fitted, not transported; a {kap * 1e6 - 110.4:+.1f} ppm difference between run_79 ↔ 224572 and run_149 ↔ 224678 is consistent with that. At 80 ms it is a {abs(kap - 110.4e-6) * 8e7:.0f} ns effect, so transporting K would have cost resolution at late tof.</p>''',
            short='3 · κ')


def s_delta(D, S, db):
    db = db.sort_values('bunch_t_s')
    t = (db.bunch_t_s - db.bunch_t_s.min()).to_numpy()
    v = db.delta_b_ns.to_numpy() / 1e3
    sm = np.array([np.median(v[np.abs(t - x) < S['smooth_s'] / 2]) for x in t])
    P = sd.Plot(1100, 640, x=(0, 920), y=(-300, 220), xlabel='bunch time  [s]',
                ylabel='δ_b: window start − psTime  [µs]')
    P.xticks([(x, str(x)) for x in range(0, 901, 150)]).yticks([(y, str(y)) for y in range(-300, 201, 100)])
    P.raw(''.join(f'<circle cx="{P.X(a):.1f}" cy="{P.Y(b):.1f}" r="3.6" fill="{C_S4}" fill-opacity=".55"/>'
                  for a, b in zip(t, v)), back=True)
    P.line(list(t[::8]), list(sm[::8]), INK, 3.5, markers=False,
           tip=f'running median over {S["smooth_s"]:.0f} s: the slow part, spanning '
               f'{S["delta_b_slow_range_ns"] / 1e3:.0f} µs')
    right = sd.col(
        sd.p(f'Without beam, nothing pins where each 80 ms window starts. Measured per bunch from its '
             f'own triggers, {sd.term("δ_b", G["delta"])} has two parts:', 26),
        sd.callout(f'<b>slow</b>: a wander of {S["delta_b_slow_range_ns"] / 1e3:.0f} µs over the sub-run, '
                   'the curvature the straight drift line misses. It can be followed.', INK, 24),
        sd.callout(f'<b>fast</b>: {S["delta_b_fast_mad_ns"] / 1e3:.0f} µs MAD, uncorrelated bunch to bunch '
                   f'(neighbour step {S["delta_b_step_mad_ns"] / 1e3:.0f} µs ≈ √2 × it). It cannot.', C_S4, 24),
        sd.p(f'So a trigger alone in its bunch ({S["singletons"]} here) is placed only to ~30 µs. '
             f'Two or more, and δ_b is measured.', 25, MUT),
        gap=20, w=520)
    body = sd.title('Each free-running window starts ~30 µs off its psTime, at random, on a slow wander',
                    'Per-bunch window offset from the bunch\'s matched triggers (median), run_149 cos_0000 ↔ 224678.')
    body += sd.row(P.svg('delta_b'), right, gap=44)
    D.slide('delta', body, f'''
<p>δ_b = median over the bunch's matched triggers of off − tof·(1 − κ), where off is the DREAM trigger time on the drift map minus psTime. {len(db)} bunches carry at least one matched trigger.</p>
<p>With beam the equivalent per-burst offset δa_b is 6.5 ns RMS, because every burst re-locks to its gamma flash. Here the window start is tied to psTime only through whatever free-running trigger n_TOF used, and the scatter is ~5000× larger. It is a property of n_TOF's beam-off acquisition, not of DREAM.</p>
<p>Not yet tried: a smooth (spline) drift model in stage 2 would absorb the slow part and let the stage-4 association window shrink from ±300 µs.</p>''',
            short='4 · δ_b')


def s_result(D, S, pairs):
    r = pairs.res.dropna().to_numpy()
    W = S['window_ns']
    edges = np.arange(-200, 201, 5)
    h, _ = np.histogram(r, bins=edges)
    P = sd.Plot(1060, 640, x=(-200, 200), y=(0, 400), xlabel='leave-one-out residual  [ns]',
                ylabel='triggers per 5 ns')
    P.xticks([(v, str(v)) for v in range(-200, 201, 50)]).yticks([(v, str(v)) for v in range(0, 401, 100)])
    P.band([-W, W], [0, 0], [400, 400], C_S4, 0.10, tip=f'accept window ±{W:g} ns')
    for lo, n in zip(edges[:-1], h):
        P.vbar(lo + 2.5, n, P.pw / 80 - 2, INK, tip=f'{lo:+.0f} to {lo + 5:+.0f} ns: {n} triggers')
    P.text(W + 6, 370, f'±{W:g} ns', 22, C_S4, 'start', 600)
    arm = pairs[pairs.res.abs() < W].arm.value_counts().sort_index()
    arow = [(f'arm {a}', int(arm.get(i, 0)), sd.BLUE if a in 'AC' else ORANGE,
             f'arm {a}: {int(arm.get(i, 0))} matched triggers') for i, a in enumerate('ABCD')]
    right = sd.col(
        sd.row(sd.card(sd.p(f'{100 * S["efficiency_loo"]:.1f}%', 52, C_S4, 600)
                       + sd.p(f'of {S["efficiency_denominator"]:,} in-window triggers', 22, MUT), pad=24,
                       tip='Denominator: every DREAM trigger that falls inside a window whose δ_b is known '
                           'from other triggers, matched or not.'),
               sd.card(sd.p(f'{S["res_core_mad_ns"]:.1f} ns', 52, INK, 600) + sd.p('core MAD', 22, MUT), pad=24,
                       tip=G['loo']), gap=20),
        sd.p(f'<b>Accidentals:</b> {S["sideband"]} trigger in the 1–300 µs sideband → '
             f'{S["accidental_expected"]:.4f} expected inside ±{W:g} ns. Too small to measure.', 24),
        sd.p('<b>Matched triggers per arm</b>', 24),
        sd.hbars(arow, vmax=max(a[1] for a in arow) * 1.15, width=260, label_w=110, size=23),
        gap=18, w=560)
    body = sd.title(f'{S["matched_loo"]:,} cosmic triggers matched to ±{W:g} ns: '
                    f'{100 * S["efficiency_loo"]:.0f} % of those n_TOF could see',
                    'Each trigger predicted from its bunch\'s OTHER triggers only — nothing validates itself.')
    body += sd.row(P.svg('residual'), right, gap=44)
    D.slide('result', body, f'''
<p><b>Efficiency, honestly counted.</b> Numerator: triggers whose leave-one-out residual is within ±{W:g} ns. Denominator: every DREAM trigger that falls, on the drift map, inside a bunch window (100 µs clear of either edge) whose δ_b is known from <i>other</i> triggers, whether or not it found a candidate. The first version of this number counted only triggers that already had a candidate within 300 µs, which inflated it to 92.9 %.</p>
<p><b>Compared with beam-on.</b> The beam-on match (run_79 ↔ 224572) reaches 96.0 % (wall AND plastic) with a 6 ns 68 % half-width. The ~4-point gap here is not yet understood; candidates are the unfitted per-bunch rate and the 418 single-trigger bunches, which leave the denominator but could bias which bunches enter it.</p>
<p><b>Accidentals.</b> The nearest-candidate residual 1–300 µs from the peak is accidental by construction; scaled to ±{W:g} ns it predicts {S["accidental_expected"]:.4f}. Analytically: 27 Hz of in-window singles × 100 ns × {S["efficiency_denominator"]} triggers ≈ 0.007.</p>''',
            foot='run_149 cosbounce_cos_0000 ↔ n_TOF 224678 · clock_match.py → results/clock_match/pairs_*.csv',
            short='Result')


def s_gotchas(D, S):
    sh = S['pss_shift_ns']
    rows = [(f'PSS{a}', sh[a], ORANGE, f'arm {a}: plastic raw tof sits {sh[a]:.1f} ns later than the beam-on '
                                       f'wall–plastic timing; shifted back before the 20 ns AND')
            for a in 'ABCD']
    cards = [
        sd.card(sd.p('<b>combined_hits ≠ every trigger</b>', 28)
                + sd.p('It keeps only events with Micromegas activity. The clock match reads FEU 1\'s '
                       '<code>decoded_root</code> <code>timestamp</code> (10 ns ticks): all 22 760 triggers.', 24, MUT)),
        sd.card(sd.p('<b>Raw tof has no common zero</b>', 28)
                + sd.p('With beam, each tree\'s own tflash absorbs its cable delay. Without, the plastics sit '
                       '35–42 ns late, outside the 20 ns coincidence: 0 singles until shifted.', 24, MUT)
                + sd.hbars(rows, vmax=48, width=260, h=24, label_w=90, fmt=lambda v: f'{v:.1f} ns', size=22)),
        sd.card(sd.p('<b>psTime = 0 on 7.5 % of bunches</b>', 28)
                + sd.p(f'{S["n_bunches_no_pstime"]} of {S["n_bunches"] + S["n_bunches_no_pstime"]} in 224678. '
                       'Dropped for now; the 0.5009 s period is regular to ~0.1 ms, so they are recoverable.',
                       24, MUT)),
        sd.card(sd.p('<b>The cached lock has mislocks</b>', 28)
                + sd.p('7 of 21 beam-on pulse-match offsets in the local cache are supercycle steps off. '
                       'Fine as a prior here; never as a constant.', 24, MUT)),
    ]
    body = sd.title('Four things that would have silently broken the match',
                    'Each found on the way and handled in clock_match.py.')
    body += sd.row(sd.col(cards[0], cards[2], gap=28), sd.col(cards[1], cards[3], gap=28), gap=36)
    D.slide('gotchas', body, '''
<p>The plastic shift is measured in situ per arm (<code>clock_match.measure_pss_shift</code>): the peak of plastic − wall-sum times within ±2 µs, median within ±30 ns, then moved to the beam-on peak (A −6.8, B −3.8, C −6.3, D −8.8 ns, <code>DREAM_NTOF_CALIBRATION.md</code> §2b) so the emulated AND sees the timing the hardware did.</p>
<p><code>fast_singles</code> is reused unchanged; only its reader is swapped (<code>_raw_tof_reader</code>) to hand it raw tof instead of tof − tflash.</p>''',
            short='Gotchas')


def s_close(D):
    items = [
        ('One sub-run', 'run_149 cos_0000 ↔ 224678 only. S is per sub-run; k and κ should transfer but are not yet shown to.'),
        ('Single-trigger bunches', '418 triggers alone in their window are placed only to ~30 µs, and leave the denominator.'),
        ('A straight drift line', 'The slow δ_b wander is curvature it misses; a smooth model would tighten stage 4.'),
        ('92 % against 96 %', 'The gap to the beam-on efficiency is not yet understood.'),
        ('No tracking yet', 'Nothing here says how many cosmic triggers carry a track, or a track in two arms.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:16px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:28px;font-weight:600;width:400px">{a}</p>'
                    f'<p style="font-size:24px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    nxt = [('Match all of it', 'Sub-runs spanning two n_TOF runs; psTime=0 recovery; then run_149 and run_103 (~13 h with n_TOF).'),
           ('Track the cosmic runs', 'Production tracking on run_149 first: two-arm crossings, opening angle, capsule DCA.'),
           ('The three handles', 'Calibrate the 170° cut on data; the capsule-DCA shape of through-goers; the arm-to-arm Δt sign.')]
    nrows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:16px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:28px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:24px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in nxt)
    body = (f'<div style="display:flex;gap:80px">'
            f'<div style="flex:1.2;display:flex;flex-direction:column;gap:8px">'
            f'<h2 style="font-size:52px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:8px">'
            f'<h2 style="font-size:52px;font-weight:600">Next</h2>{nrows}</div></div>')
    D.slide('close', body, '''
<p>Entry point: <code>ntof_cosmics/README.md</code>. The October list carries this as item O9 (<code>sept26_prelim_analysis/OCTOBER_2026.md</code>).</p>
<p>Data staged locally for the first sub-run: <code>/media/dylan/data/x17/beam_july/runs/run_149/cosbounce_cos_0000/decoded_root/*_01.root</code>, <code>.../ntof_data/run224678.parts/</code>, <code>run_149/*/n1081b_config.json</code> and <code>run_149/dream_daq.log</code>.</p>''',
            dark=True, short='Caveats & next')


# --------------------------------------------------------------------------- #
def build(out: Path) -> Path:
    run, S, pairs, drift, db = load()
    lat = beam_on_latency()
    D = sd.Deck('Beam-off cosmics on the n_TOF clock',
                'The beam-off cosmic runs of the n_TOF X17 campaign: what exists, why straight-through '
                'particles matter, and matching DREAM cosmic triggers to n_TOF without beam.')
    s_cover(D, run, S)
    s_background(D)
    s_inventory(D, run)
    s_quiet(D, run, S)
    s_clocks(D, S)
    s_coarse(D, S, lat)
    s_drift(D, S, drift)
    s_kappa(D, S, pairs)
    s_delta(D, S, db)
    s_result(D, S, pairs)
    s_gotchas(D, S)
    s_close(D)
    meta = dict(title='Beam-off cosmics on the n_TOF clock',
                summary=(f'57.6 h of production-point cosmics, never analysed; matched to n_TOF without beam: '
                         f'{100 * S["efficiency_loo"]:.0f} % of in-window triggers within ±{S["window_ns"]:g} ns, '
                         f'{S["res_core_mad_ns"]:.0f} ns core. First sub-run; growing.'),
                tags='X17,backgrounds,cosmics,timing', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by nTof_x17/ntof_cosmics/'
                                     'make_deck.py with slidedoc.py.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=RES / 'deck' / 'beam-off-cosmics.html')
    a = ap.parse_args()
    print('wrote', build(a.out))


if __name__ == '__main__':
    main()
