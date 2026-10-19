#!/usr/bin/env python3
"""make_deck.py -- MX17 status and the future of the X17 search (informal meeting, 19 Oct 2026).

    .venv/bin/python talks/future_oct26/make_deck.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        /media/dylan/data/x17/future_oct26/mx17-future-x17.html --slug mx17-future-x17 --force --deploy

A short slide note (slidedoc) for collaborators: what the detectors do (cosmic bench, SPS),
what n_TOF showed, whether the apparatus can search for X17 elsewhere (LNL first; the
neutron facilities), and the papers.  Main slides first, backup after.

Every chart is read from the owning analysis's own outputs, so a rerun upstream moves the
numbers here too:

  bench efficiency maps   ~/x17/cosmic_bench/Analysis/bench_efficiency_maps/<key>_rays.csv
                          (mx_june_wft/report/make_bench_effmaps.py)
  det3 angle resolution   mpgd26/data/angle_resolution.json (mpgd26/make_resolution.py)
  det4 SPS resolution     sps_beam_test_26/analysis/spatial_resolution/results.json
  scintillator stack      /media/dylan/data/x17/scint_stack/figures/*.csv (ntof_scint_stack)
  capsule image           /media/dylan/data/x17/sept26_prelim/capsule_pose/{image,data}.csv
  plastic energy          /media/dylan/data/x17/calorimetry/wp/{spectra,scale_fits}.csv
  LNL reach               ~/PycharmProjects/x17_facility_search/lnl/out/talk/numbers.json
  plastic size scan       .../lnl/out/plastics/days_vs_size.csv (lnl/plastic_size.py)

Numbers that live only in a write-up (no table to read) are constants below, each with the
file it comes from.  Detector fleet numbers are the waveform-reconstruction (r06) ones of
mx_june_wft/FLEET_DIGEST.md.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FAC = Path(os.path.expanduser('~/PycharmProjects/x17_facility_search'))
BENCH = Path(os.path.expanduser('~/x17/cosmic_bench/Analysis/bench_efficiency_maps'))
DATA = Path('/media/dylan/data/x17')
OUT = Path(os.environ.get('X17_FUTURE_OUT', DATA / 'future_oct26'))

INK, MUT, RULE = sd.INK, sd.MUT, sd.RULE
BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN = (sd.BLUE, sd.ORANGE, sd.RED, sd.GOLD,
                                                 sd.PURPLE, sd.GREY, sd.GREEN)
G4C = '#3b3f4a'
# one colour per chamber, the same on every slide.  n_TOF arm = bench detector.
CH = {'A': BLUE, 'B': PURPLE, 'C': ORANGE, 'D': GREEN, 'E': GOLD}
DET = {'A': 'det3', 'B': 'det2', 'C': 'det6', 'D': 'det7', 'E': 'det4'}
KEY = {'A': 'g_det3_wknd', 'B': 'g_det2', 'C': 'g_det6_long', 'D': 'g_det7_long', 'E': 'g_det4'}

# --- constants with provenance (no machine-readable table upstream) -------------------
# mx_june_wft/FLEET_DIGEST.md (waveform reco, r06/lp bundles), the five golden keys
FLEET = {  # within-5 mm %, core sigma_r mm, sigma_theta X, Y deg, spark %
    'A': (93.3, 0.45, 1.15, 1.18, 8.2), 'B': (92.0, 0.44, 1.21, 1.57, 9.7),
    'C': (74.9, 0.46, 2.24, 2.51, 22.3), 'D': (57.1, 0.63, 1.94, 1.87, 37.4),
    'E': (41.6, 0.56, 2.09, 2.28, 9.8)}
VERDICT = {
    'A': ('good', 'arm A at n_TOF; the reference chamber'),
    'B': ('good; drift to fix', 'arm B; tagging-only at n_TOF'),
    'C': ('weak spots', 'arm C; 14 dead X channels (connector pattern); re-bulk'),
    'D': ('poor', 'arm D; ~130/512 X channels dead at n_TOF; re-bulk'),
    'E': ('62 % not amplifying', 'SPS; non-amplifying stripes (det4_sps_assessment)')}
# n_TOF campaign: mpgd26/make_timeline.py docstring
CAMPAIGN = dict(runs=162, subruns=2705, tb=17.9, events=41.8, pulses=436373)


def rd(p):
    return pd.read_csv(p)


# =========================================================================== #
# shared drawings
# =========================================================================== #
def timeline_svg(w=1664, h=250):
    import datetime as dt
    t0, t1 = dt.date(2025, 11, 1), dt.date(2026, 11, 1)
    x0, x1 = 40, w - 20
    X = lambda d: x0 + (d - t0).days / (t1 - t0).days * (x1 - x0)  # noqa: E731
    o = []
    yax = h - 44
    o.append(sd.line(x0, yax, x1, yax, MUT, 2))
    for m in range(13):
        y, mo = 2025 + (10 + m) // 12, (10 + m) % 12 + 1
        d = dt.date(y, mo, 1)
        o.append(sd.line(X(d), yax, X(d), yax + 8, MUT, 1.5))
        if m < 12:
            lab = d.strftime('%b') + (f' {y}' if mo in (1, 11) and m in (0, 2) else '')
            o.append(sd.T(X(d) + (X(dt.date(y + (mo == 12), mo % 12 + 1, 1)) - X(d)) / 2, yax + 32, lab, 20))
    D = dt.date
    bars = [  # (start, end, row, colour, label, tip[, label side])
        (D(2025, 11, 27), D(2025, 12, 5), 0, BLUE, 'EAR2 test 1',
         'n_TOF EAR2 test beam, 27 Nov - 5 Dec 2025: 86 DREAM runs; detectors, targets, SiPM behaviour'),
        (D(2026, 2, 2), D(2026, 3, 1), 0, BLUE, 'EAR2 test 2',
         'n_TOF EAR2, 2 Feb - 1 Mar 2026: 143 runs; gases, drift gaps, frames, targets'),
        (D(2026, 5, 6), D(2026, 5, 18), 0, BLUE, 'EAR2 test 3',
         'n_TOF EAR2, 6-18 May 2026: 72 runs, commissioning of the four-arm station'),
        (D(2026, 6, 6), D(2026, 6, 27), 1, GREY, 'cosmic bench',
         'Saclay cosmic bench, 6-27 June 2026: five MX17 chambers (det2, 3, 4, 6, 7) against the M3 telescope', 'left'),
        (D(2026, 6, 28), D(2026, 8, 10), 0, BLUE, 'n_TOF physics run',
         f'n_TOF EAR2 physics run: arrival 28 June, first beam 2 July, last beam 10 Aug 09:10. '
         f'{CAMPAIGN["runs"]} DREAM runs, {CAMPAIGN["subruns"]:,} sub-runs, {CAMPAIGN["tb"]} TB, '
         f'{CAMPAIGN["events"]} M events, {CAMPAIGN["pulses"]:,} pulses. The dataset is final.'),
        (D(2026, 7, 31), D(2026, 8, 3), 2, ORANGE, 'SPS H4',
         'det4 parasitic in the P2 uRWELL test beam at SPS H4, 31 July - 3 Aug 2026'),
        (D(2026, 8, 10), D(2026, 10, 19), 1, GREY, 'analysis, facility studies, papers',
         'Aug-Oct: waveform reconstruction of the fleet, n_TOF full pass and re-pass, scintillator maps and '
         'calorimetry, Geant4 facility studies (ILL, LNL, NFS), paper outlines', 'inside'),
    ]
    for a, b, r, c, lab, tip, *side in bars:
        y = 8 + r * 46
        xa, xb = X(a), max(X(b), X(a) + 8)
        o.append(f'<rect x="{xa:.1f}" y="{y}" width="{xb - xa:.1f}" height="32" rx="6" fill="{c}"'
                 f' fill-opacity="0.9"{sd.tipattr(tip)}/>')
        sd_ = side[0] if side else 'right'
        if sd_ == 'inside':
            o.append(sd.T((xa + xb) / 2, y + 23, lab, 20, '#ffffff', 'middle', 600))
        elif sd_ == 'left':
            o.append(sd.T(xa - 10, y + 23, lab, 21, INK, 'end', 600))
        else:
            o.append(sd.T(xb + 10, y + 23, lab, 21, INK, 'start', 600))
    return sd.svg(w, h, ''.join(o), 'project timeline')


def bench_svg(w=520, h=430):
    """The Saclay cosmic bench, side view, z to scale (scenes_bench.py)."""
    zs = [(-110, 'scint', 'trigger scintillator 60×60 cm'), (24, 'm3', 'M3 reference'),
          (144, 'm3', 'M3 reference'), (232, 'mx', 'test slot P1 (MX17)'), (702, 'mx', 'test slot P2 (MX17)'),
          (1185, 'm3', 'M3 reference'), (1302, 'm3', 'M3 reference'), (1420, 'scint', 'trigger scintillator')]
    Y = lambda z: 22 + (1460 - z) / 1610 * (h - 44)  # noqa: E731
    cx, half = 150, 120
    o = []
    for z, kind, lab in zs:
        y = Y(z)
        c, th = {'scint': (GOLD, 10), 'm3': (GREY, 8), 'mx': (BLUE, 12)}[kind]
        o.append(f'<rect x="{cx - half}" y="{y - th / 2:.1f}" width="{2 * half}" height="{th}" rx="3" fill="{c}"/>')
        o.append(sd.T(cx + half + 16, y + 7, lab, 19, INK if kind == 'mx' else MUT, 'start',
                      600 if kind == 'mx' else 400))
    o.append(sd.line(cx - 60, Y(1440), cx + 40, Y(-130), RED, 3, '10 6'))
    o.append(sd.T(cx - 66, Y(1440) + 2, 'μ', 24, RED, 'end', 600))
    o.append(sd.T(8, h / 2, '1.5 m', 18, MUT, 'middle', rot=-90))
    return sd.svg(w, h, ''.join(o), 'cosmic bench')


def sps_svg(w=600, h=300):
    """SPS H4: det4 in the P2 uRWELL telescope, z to scale (scenes_sps.py)."""
    st = [(0, 'urw', 'uRWELL'), (320, 'p2', 'P2'), (630, 'p2', 'P2'), (940, 'p2', 'P2'),
          (1155, 'mx', 'det4'), (1370, 'urw', 'uRWELL')]
    X = lambda z: 70 + z / 1370 * (w - 110)  # noqa: E731
    cy = (h - 40) / 2 + 4
    o = [sd.arrow(4, cy, X(1370) + 34, cy, RED, 3, 14), sd.T(6, cy - 14, 'beam', 20, RED, 'start', 600)]
    for z, kind, lab in st:
        x = X(z)
        c, wd, hh = {'urw': (GREY, 10, 120), 'p2': ('#9aa3b2', 8, 100), 'mx': (BLUE, 14, 140)}[kind]
        o.append(f'<rect x="{x - wd / 2:.1f}" y="{cy - hh / 2}" width="{wd}" height="{hh}" rx="3" fill="{c}"/>')
        o.append(sd.T(x, cy + hh / 2 + 26, lab, 19, INK if kind == 'mx' else MUT, 'middle',
                      600 if kind == 'mx' else 400))
    o.append(sd.T(w / 2, h - 8, 'reference tracks from the two uRWELLs; det4 at 0° and 25.6°', 19, MUT))
    return sd.svg(w, h, ''.join(o), 'SPS H4 telescope')


def ntof_plan_svg(w=600, h=600):
    """The EAR2 station seen along the beam: capsule, four arms, each MM → SiPM wall →
    plastic → liquid.  Distances from SimConfig / scint geometry (approximate, no pinwheel)."""
    s = 0.5
    cx, cy = w / 2, (h - 120) / 2 + 4
    o = []
    for k, (arm, ang) in enumerate(zip('ACBD', (0, 180, 90, 270))):
        g = []
        g.append(f'<rect x="{204 * s:.1f}" y="{-200 * s:.1f}" width="{30 * s:.1f}" height="{400 * s:.1f}" '
                 f'fill="{CH[arm]}" fill-opacity="0.35" stroke="{CH[arm]}" stroke-width="2"/>')
        g.append(f'<rect x="{350 * s:.1f}" y="{-250 * s:.1f}" width="{6 * s + 2:.1f}" height="{500 * s:.1f}" fill="{GOLD}"/>')
        g.append(f'<rect x="{410 * s:.1f}" y="{-200 * s:.1f}" width="{20 * s + 2:.1f}" height="{400 * s:.1f}" fill="{PURPLE}" fill-opacity="0.75"/>')
        g.append(f'<rect x="{470 * s:.1f}" y="{-225 * s:.1f}" width="{21 * s + 2:.1f}" height="{450 * s:.1f}" fill="{GREEN}" fill-opacity="0.45"/>')
        o.append(f'<g transform="translate({cx},{cy}) rotate({ang})">{"".join(g)}</g>')
        r = 160 * s
        lx = cx + r * math.cos(math.radians(ang)) * 1.0
        ly = cy + r * math.sin(math.radians(ang)) * 1.0 + 8
        o.append(sd.T(lx, ly, arm, 30, CH[arm], 'middle', 700))
    o.append(f'<circle cx="{cx}" cy="{cy}" r="16" fill="#c9ced8" stroke="{INK}" stroke-width="2"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{RED}"/>')
    o.append(sd.T(cx, cy + 44, '³He capsule', 18, INK))
    o.append(sd.T(cx, cy + 64, 'beam ⊙ (up)', 17, RED))
    leg = [(BLUE, 'Micromegas TPC, 30 mm gap'), (GOLD, 'SiPM wall, 3 mm'), (PURPLE, 'plastic, 2 × 20×30×2 cm'),
           (GREEN, 'liquid (LAB), 45×45 cm')]
    for i, (c, t) in enumerate(leg):
        lx, ly = 20 + (i % 2) * 290, h - 100 + (i // 2) * 30
        o.append(f'<rect x="{lx}" y="{ly}" width="18" height="18" fill="{c}" fill-opacity="0.8"/>')
        o.append(sd.T(lx + 26, ly + 15, t, 18, INK, 'start'))
    o.append(sd.T(w / 2, h - 18, 'seen along the beam, schematic (no pinwheel shifts)', 17, MUT))
    return sd.svg(w, h, ''.join(o), 'n_TOF station plan')


def eff_map(arm, w, h, cell=20.0, xlabel=False, ylabel=False, title=None, rel=True, nmin=8):
    d = rd(BENCH / f'{KEY[arm]}_rays.csv')
    b = json.load(open(BENCH / f'{KEY[arm]}_box.json'))
    x = d.x - b['x0']
    y = d.y - b['y0']
    W, H = b['x1'] - b['x0'], b['y1'] - b['y0']
    nx, ny = int(math.ceil(W / cell)), int(math.ceil(H / cell))
    ix = np.clip((x // cell).astype(int), 0, nx - 1)
    iy = np.clip((y // cell).astype(int), 0, ny - 1)
    g = pd.DataFrame({'ix': ix, 'iy': iy, 'w': d.within.astype(int)}).groupby(['ix', 'iy']).w.agg(['sum', 'count'])
    cells = []
    for (i, j), r in g.iterrows():
        if r['count'] < nmin:
            continue
        e = r['sum'] / r['count']
        cells.append((i * cell, (i + 1) * cell, j * cell, (j + 1) * cell, e,
                      f'{DET[arm]} ({arm}) cell {i * cell:.0f}-{(i + 1) * cell:.0f} × {j * cell:.0f}-{(j + 1) * cell:.0f} mm:\n'
                      f'{int(r["sum"])}/{int(r["count"])} reconstructed within 5 mm = {100 * e:.0f} %'))
    P = sd.Plot(w, h, x=(0, 360), y=(0, 400), xlabel='x (mm)' if xlabel else '',
                ylabel='y (mm)' if ylabel else '', title=title,
                margin=(8, 8, 56 if xlabel else 30, 70 if ylabel else 18))
    P.xticks([(v, str(v)) for v in (0, 100, 200, 300)])
    P.yticks([(v, str(v) if ylabel else '') for v in (0, 100, 200, 300, 400)])
    P.cells(cells, 0.0, 1.0)
    P.rect(0, W, 0, H, INK, 1.5)
    return P, float(d.within.mean()), len(d)


# =========================================================================== #
# slides
# =========================================================================== #
def s_cover(D, L):
    big = sd.row(
        sd.bignum('176 µm', 'intrinsic resolution', sd.DBLUE,
                  'det4 in the SPS H4 beam: 0.30 × the strip pitch',
                  tip='sigma_det4 = 176 ± 10 µm from the uRWELL residuals at three pointing errors '
                      '(spatial_resolution/results.json)'),
        sd.bignum('93 % · 1.2°', 'best chamber on cosmics', sd.DBLUE,
                  'det3: reconstructed within 5 mm; angle σ68, flat from 0 to 18°'),
        sd.bignum(f'{L["bp1"]["days_3sigma"]:.1f} d', 'to 3σ at the ATOMKI ratio, LNL', sd.DGREEN,
                  f'with four big plastics at 1 µA; {L["ab1"]["days_3sigma"]:.0f} d with the n_TOF stack as built'),
        sd.bignum('⅛', 'Al pairs seen / expected at n_TOF', sd.DRED,
                  'singles agree with Geant4 to 15–25 %; the pair deficit is open and pair-specific'),
        gap=48)
    body = (sd.kicker('MX17 · status and the future of the X17 search · informal meeting, 19 Oct 2026')
            + sd.title('<span style="color:#eef0f3">The trackers work, and the scintillators measure energy. '
                       'X17 was out of reach at n_TOF; at LNL the same apparatus, with bigger plastics, '
                       'reaches the ATOMKI level in days</span>')
            + big
            + sd.flow([
                dict(label='1 · Detectors', sub='cosmic bench, SPS: resolution, efficiency, the fleet', color=sd.DBLUE),
                dict(label='2 · n_TOF', sub='what to expect, scintillators, singles vs pairs, energy', color=sd.DBLUE),
                dict(label='3 · Where next', sub='LNL, the big plastics, the neutron facilities', color=sd.DGREEN),
                dict(label='4 · Papers', sub='two detector papers, an EAR2 demonstrator', color=sd.DGREY)],
                dark=True, size=24))
    notes = """
<p><b>The question for the meeting.</b> The n_TOF run cannot see X17 (thermal neutrons only, after the γ-flash
dead time; backup B2). Can this detector system search for X17 somewhere else, and what does it need?</p>
<p><b>Short answer.</b> The Micromegas TPCs track well (best chamber 93 % efficient, 1.2° angles, 176 µm
intrinsic) and, in the beam, single tracks agree with Geant4 and image the target. The scintillators hold a muon
energy scale that predicts electron spectra to 2–5 %. At LNL (⁷Li(p,e⁺e⁻)⁸Be, the ATOMKI reaction) the Geant4
reach is 17 days to 3σ at the ATOMKI ratio as built and about a day with four large plastics. The neutron
facilities are much harder (slide 12).</p>
<p><b>What is still open.</b> At n_TOF the same-chamber Al e⁺e⁻ pairs come in at ⅛ of the Geant4 prediction
while the singles agree; until that is understood, the two-track efficiency for soft, close pairs is not
demonstrated (slide 8). X17 pairs are a far easier topology.</p>
<p>Hover the dotted terms and the data points for values, counts and provenance. Main deck: slides 1–13;
backup: B1 onward.</p>"""
    D.slide('cover', body, notes, dark=True, short='Answer')


def s_setups(D):
    body = (sd.title('Five chambers characterised on a cosmic bench, one in the SPS beam, four at n_TOF',
                     'Nov 2025 – Oct 2026 · three EAR2 tests, the June bench, the n_TOF physics run, det4 at SPS H4')
            + timeline_svg(1664, 196)
            + sd.row(
                sd.card(sd.p('<b>Saclay cosmic bench</b> · June 2026', 26)
                        + sd.row(bench_svg(540, 330),
                                 sd.col(sd.p('Muons tracked by four M3 reference Micromegas (50×50 cm); two MX17 '
                                             'chambers in the test slots.', 22, MUT),
                                        sd.p(f'Five chambers: det2, det3, det4, det6, det7 '
                                             f'({sd.term("7–22 k reference muons each", "rays per golden key: det3 21,953 · det2 19,045 · det4 12,259 · det6 9,628 · det7 9,428")}).',
                                             22, MUT),
                                        gap=14), gap=12), w=900, pad=24),
                sd.card(sd.p('<b>SPS H4</b> · 31 Jul – 3 Aug 2026', 26)
                        + sps_svg(660, 250)
                        + sd.p('det4 parasitic in the P2 uRWELL beam: straight high-momentum tracks, so the intrinsic '
                               'resolution and the sharing kernel come out directly.', 22, MUT), pad=24),
                gap=32))
    notes = """
<p><b>Timeline sources.</b> mpgd26/make_timeline.py (the MPGD26 project timeline): EAR2 tests from the run tables
on EOS; bench span from mx_june_cosmic_qa; n_TOF from ntof_run_report (162 runs, 2,705 sub-runs, 17.9 TB,
41.8 M events); SPS from sps_beam_test_26/analysis/RUN_TIMELINE.md.</p>
<p><b>Chamber names.</b> The same chamber has a bench name and an n_TOF arm: det3 = A, det2 = B, det6 = C, det7 = D;
det4 = "E", the SPS chamber.</p>
<p><b>Bench geometry</b> (scenes_bench.py): scintillators at z −110 and 1420 mm, M3 at 24, 144, 1185, 1302 mm,
test slots P1 232 and P2 702 mm. <b>SPS</b> (scenes_sps.py): uRWELL at 0 and 1370 mm, P2 fans at 320/630/940 mm,
det4 in the ~25 cm between the last P2 and the back uRWELL.</p>"""
    D.slide('setups', body, notes, short='Setups')


def s_best(D):
    R = json.load(open(REPO / 'mpgd26/data/angle_resolution.json'))
    P = sd.Plot(560, 560, x=(0, 18), y=(0, 2.5), xlabel='|reference angle| (°)', ylabel='angle σ68 (°)',
                title='det3 angles: flat 1–1.4°', margin=(16, 20, 92, 96))
    P.xticks([(v, str(v)) for v in (0, 4, 8, 12, 16)])
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5, 2.0, 2.5)])
    P.hline(R['floor_deg'], GREY, label='~1° diffusion floor', anchor='start', where='below', tip='physics floor from the toy closure: ~1° from 0.3 mm centroid jitter')
    for pl, c, nm in (('x', BLUE, 'X'), ('y', ORANGE, 'Y')):
        bins = R['planes'][pl]['bins']
        xs = [(b['lo'] + b['hi']) / 2 for b in bins]
        ys = [b['s68_deg'] for b in bins]
        P.line(xs, ys, c, tips=[f'{nm} plane, {b["lo"]:.0f}-{b["hi"]:.0f}°: σ68 {b["s68_deg"]:.2f} ± {b["s68_err_deg"]:.2f}°, '
                                f'bias {b["bias_deg"]:+.2f}°, n = {b["n"]}' for b in bins])
    P.text(1, 2.25, f'X σ68 {R["planes"]["x"]["s68_deg"]:.2f}° · Y {R["planes"]["y"]["s68_deg"]:.2f}° (all angles)', 20, INK)

    S = json.load(open(REPO / 'sps_beam_test_26/analysis/spatial_resolution/results.json'))
    sig = S['fit']['sigma_det4']
    Q = sd.Plot(540, 560, x=(0, 0.42), y=(0, 0.5), xlabel='telescope pointing error (mm)',
                ylabel='det4 − track residual σ (mm)', title=f'det4 at SPS: {1000 * sig:.0f} µm intrinsic',
                margin=(16, 20, 92, 96))
    Q.xticks([(v, f'{v:.1f}') for v in (0, 0.1, 0.2, 0.3, 0.4)])
    Q.yticks([(v, f'{v:.1f}') for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)])
    ps = np.linspace(0, 0.42, 40)
    Q.line(list(ps), list(np.sqrt(ps ** 2 + sig ** 2)), G4C, 3, dash='8 6', markers=False,
           tip=f'√(pointing² + σ_det4²), σ_det4 = {1000 * sig:.0f} ± {1000 * S["fit"]["sigma_det4_err"]:.0f} µm')
    Q.hline(sig, BLUE, label='σ_det4', anchor='end', where='below')
    for coord, c in (('uRW-x', BLUE), ('uRW-y', ORANGE)):
        z = S['zones'][coord]
        Q.points([r['pointing'] for r in z], [r['sigma_res'] for r in z], c, r=9,
                 tips=[f'{coord}, uRWELL pitch {r["pitch"]} mm: residual σ {1000 * r["sigma_res"]:.0f} ± {1000 * r["err"]:.0f} µm, '
                       f'pointing {1000 * r["pointing"]:.0f} µm → det4 {1000 * r["sigma_det4"]:.0f} µm (n = {r["n"]:,})' for r in z])

    M, e3, n3 = eff_map('A', 470, 560, cell=15, xlabel=True, ylabel=True, title='det3 efficiency map')
    body = (sd.title('The best chamber: 93 % efficient within 5 mm, 1.2° angles at every angle, 176 µm intrinsic',
                     'det3 (n_TOF arm A) on the cosmic bench, waveform reconstruction · det4 in the SPS H4 beam')
            + sd.row(P.svg('angle resolution'), M.svg('det3 efficiency'), Q.svg('SPS resolution'), gap=24)
            + sd.row(sd.legend([('X plane / uRWELL x', BLUE, 'dot'), ('Y plane / uRWELL y', ORANGE, 'dot'),
                                ('efficiency 0 → 100 %, 15 mm cells', sd.VIRIDIS[6][1], 'box')], size=22)))
    notes = f"""
<p><b>Left: angles.</b> σ68 of (reconstructed − M3 reference) track angle in bins of |reference angle|, det3
(<code>sat_det3</code>, r06 bundle, waveform forward model). Flat from head-on, where a drift-time ladder has no lever
arm, to 18°. The cosmic σ68 includes the M3 reference's own error and multiple scattering. Source
<code>mpgd26/data/angle_resolution.json</code>.</p>
<p><b>Middle: efficiency.</b> P(reconstructed within 5 mm of the M3 track) per 15 mm cell, <code>g_det3_wknd</code>,
{n3:,} reference muons, overall {100 * e3:.1f} %. Detection alone (any hit) is ~100 %; the 5 mm requirement is what
the tracking needs. Losses are sparks (~8 %) and far reconstructions.</p>
<p><b>Right: intrinsic resolution.</b> det4 residuals to the uRWELL track in zones of different telescope pitch, so of
different pointing error. σ_res² = pointing² + σ_det4², fitted σ_det4 = {1000 * sig:.0f} ± {1000 * S['fit']['sigma_det4_err']:.0f} µm
= 0.30 × the 0.78 mm pitch. On the bench the position residual is ~0.45 mm core, dominated by M3 and scattering.</p>
<p><b>Hits vs waveforms.</b> All geometry here comes from the waveform forward model (Paper II). Per-strip hit times
compress the drift ladder by 20–30 % on resistive strips; hit-based angles were 2.4–2.6° on the same data.</p>"""
    D.slide('best', body, notes, short='Best chamber')


def s_fleet(D):
    maps = []
    tot = {}
    for i, a in enumerate('ABCDE'):
        f = FLEET[a]
        P, e, n = eff_map(a, 300, 330, cell=20, xlabel=False, ylabel=(i == 0),
                          title=f'{a} · {DET[a]} · {f[0]:.0f} %')
        tot[a] = (e, n)
        maps.append(P.svg(f'{DET[a]} map'))
    cb = sd.colorbar(120, 330, 0, 1, [(0, '0'), (0.5, '50'), (1, '100 %')], 'within 5 mm')
    rows = []
    tips = []
    for a in 'ABCDE':
        f = FLEET[a]
        v = VERDICT[a]
        col = CH[a]
        rows.append([f'<b style="color:{col}">{a}</b> {DET[a]}', f'{f[0]:.1f} %', f'{f[1]:.2f} mm',
                     f'{f[2]:.2f}° / {f[3]:.2f}°', f'{f[4]:.0f} %', f'<b>{v[0]}</b>', v[1]])
        tips.append(f'{DET[a]}: rays {tot[a][1]:,}; map overall {100 * tot[a][0]:.1f} % within 5 mm')
    tab = sd.table(['chamber', 'within 5 mm', 'core σ_r', 'σθ X / Y', 'sparks', 'verdict', 'role / issue'],
                   rows, size=22, tips=tips, align=['left', 'right', 'right', 'right', 'right', 'left', 'left'])
    body = (sd.title('The fleet: two good chambers, two that need re-bulking, one that does not amplify over 62 % of its area',
                     'June cosmic bench, waveform reconstruction (r06), efficiency = reconstructed within 5 mm of the M3 track, 20 mm cells')
            + sd.row(*maps, cb, gap=10, align='end')
            + tab)
    notes = """
<p><b>Maps</b>: per-ray tables of <code>mx_june_wft/report/make_bench_effmaps.py</code> (same datasets as the
colleague PDF), binned here in 20 mm cells (cells with &lt; 8 muons left blank), chamber-local frame, origin at the
corner of the active box. <b>Table</b>: <code>mx_june_wft/FLEET_DIGEST.md</code> (r06, five golden keys; det3 there is
<code>sat_det3</code>, 93.3 %, the map uses <code>g_det3_wknd</code>, 93.1 %).</p>
<p><b>What does not work, and why.</b> det6 (C): 14 dead X channels in a connector pattern, 22 % sparking events.
det7 (D): sparks in 37 % of events; at n_TOF ~130 of its 512 X channels were dead. det4 (E): 62 % of the area does
not amplify, in stripes (det4_sps_assessment), and its amplitude-dependent X width is its dead and weak strips
(cloud_basics §35). The budget page carries the per-chamber plan: A good, B drift fix, C and D re-bulk, new
aluminised-mylar drift windows for all four.</p>
<p><b>Angles</b> are 1.2° on the two good chambers and ~2–2.5° on the others (fewer, noisier strips per track).</p>"""
    D.slide('fleet', body, notes, short='Fleet')


def s_expect(D):
    # Geant4 thermal campaign (Athens funnel, x17-status-2026-10-15 slides 3, 9-11)
    flow = sd.flow([
        dict(label='n into the capsule', sub='³He(n,p) 92 %, (n,γ) 0.81 %', color=GREY,
             tip='Geant4 thermal campaign, 10⁹ n, nose-first capsule; per neutron entering the capsule, E < 2 eV'),
        dict(label='capture on the Al vessel', sub='61 % of captures; 7.72 MeV γ + cascades', color=ORANGE),
        dict(label='trigger leg', sub='Compton e⁻ 77 %; e⁺e⁻ made in the capsule 10 %', color=BLUE,
             tip='what makes a trigger leg (wall ∧ plastic): Compton e⁻ of a capsule γ 76.7 %, e⁺e⁻ made in the '
                 'capsule 9.7 %, γ converting in the structure 6.8 %, capture outside 2.7 %, mixed 4.2 %'),
    ], size=22)
    # Al vs X17, per pair or per lepton (Geant4)
    items = [('both legs reach the plastic', 4.9, 76), ('pair fires the trigger itself', 7.7, 45),
             ('leg passes within 20 mm of its vertex', 31, 82), ('lepton reaches the liquid', 2.6, 57)]
    w, h = 960, 330
    o = []
    for i, (lab, al, x17) in enumerate(items):
        y = 18 + i * 78
        o.append(sd.T(0, y + 22, lab, 22, INK, 'start', 600))
        for j, (v, c, nm) in enumerate(((al, ORANGE, 'Al e⁺e⁻'), (x17, BLUE, 'X17'))):
            yy = y + 32 + j * 20
            o.append(f'<rect x="420" y="{yy}" width="{4.6 * v:.1f}" height="16" rx="3" fill="{c}"'
                     f'{sd.tipattr(f"{nm}: {v} % (Geant4) - {lab}")}/>')
            o.append(sd.T(428 + 4.6 * v, yy + 14, f'{v:g} %', 18, c, 'start', 600))
    bars = sd.svg(w, h, ''.join(o), 'Al vs X17')
    body = (sd.title('What n_TOF could show: the Al-capture pairs, the softest and hardest case of the e⁺e⁻ signature',
                     'Geant4 thermal campaign (10⁹ n, nose-first capsule) and the X17/IPC pair simulation')
            + sd.row(
                ntof_plan_svg(600, 640),
                sd.col(flow,
                       sd.row(sd.card(sd.p('<b>~10⁸ Al e⁺e⁻</b> made per 30 days', 26)
                                      + sd.p('a demonstrator sample; ~2 MeV legs, median 25°, same chamber', 21, MUT), pad=20,
                                      tip='Athens contract: external conversion 7.8×10⁷ + internal pairs 1.4×10⁷ per 30 days'),
                              sd.card(sd.p('<b>≈ 2 X17</b> made over the campaign', 26, RED)
                                      + sd.p('thermal neutrons only (γ-flash dead time); legs 4–16 MeV, ≥ 110°, opposite chambers',
                                             21, MUT), pad=20,
                                      tip='ntof_athens_26 README: ≈100 X17 made over 436,373 pulses with the full spectrum, '
                                          '≈2 in the thermal window that survived the flash dead time'), gap=20),
                       sd.legend([('Al e⁺e⁻ (same chamber)', ORANGE, 'box'), ('X17 (opposite chambers)', BLUE, 'box')], 21),
                       bars, gap=20), gap=40))
    notes = """
<p><b>Why the Al pairs.</b> After the 1 ms flash veto every neutron is thermal; the trigger fires on the 7.72 MeV
Al-capture γ of the capsule vessel (its Compton electrons). About one trigger leg in ten is an e⁺e⁻ made in the
capsule. Those pairs are the proof-of-concept sample: ~2 MeV legs that scatter ~9° on the way out (p·β·θ =
0.40 MeV·rad), mostly do not reach the plastic, and land 25° apart in one chamber. That is the hardest two-track case;
X17 legs (8–10 MeV typical) scatter 2.3–2.9°, cross the whole stack and go to opposite chambers.</p>
<p><b>What we can do with the data, for proof of concept:</b> (1) single tracks against Geant4 (rate, source image,
angle scale on cosmic muons); (2) the scintillator response and energy scale track by track; (3) the correlated
same-chamber pairs (rate, opening angle) against Geant4 pushed through the same reconstruction.</p>
<p>Numbers: x17-status-2026-10-15 slides 3, 9–11 (<code>leg_penetration.py</code>,
<code>same_chamber_scatter.py</code>, Athens <code>thermal_accounting</code>). "Pair fires the trigger itself" is the
energy-only fold; with the real bar geometry both legs ≥ 2.5 MeV in plastic is 6 % (X17) vs 0 % (Al).</p>
<p><b>Plan drawing</b> is schematic: arms at 90° without the pinwheel shifts; MM face at 204 mm, SiPM wall ~35 cm,
plastic ~41 cm, liquid ~47 cm from the beam.</p>"""
    D.slide('expect', body, notes, short='Expected at n_TOF')


def s_scint(D):
    F = DATA / 'scint_stack/figures'
    liq = rd(F / 'eff_map_liq_all_late.csv')
    halves = rd(F / 'liquid.halves.csv')
    pmt = {'A': 'PMT at +u (side)', 'B': 'PMT on top', 'C': 'PMT on top', 'D': 'PMT at +u (side)'}
    panels = []
    for i, a in enumerate('ABCD'):
        g = liq[liq.arm == a]
        P = sd.Plot(330 if i == 0 else 280, 320, x=(-250, 250), y=(-250, 250), xlabel='u (mm)', ylabel='v (mm)' if i == 0 else '',
                    margin=(4, 6, 60, 64 if i == 0 else 14))
        P.xticks([(-200, '−200'), (0, '0'), (200, '200')])
        P.yticks([(v, (str(v) if i == 0 else '')) for v in (-200, 0, 200)])
        cells = [(r.x - 25, r.x + 25, r.y - 25, r.y + 25, (None if pd.isna(r.eff) else float(r.eff)),
                  f'liquid {a}, cell ({r.x:.0f}, {r.y:.0f}) mm: {r.k_on}/{r.n} → {100 * r.eff:.1f} % net' if not pd.isna(r.eff) else None)
                 for r in g.itertuples()]
        P.cells(cells, 0, 0.2)
        P.rect(-225, 225, -225, 225, INK, 1.5)
        if 'side' in pmt[a]:
            P.rect(225, 245, -40, 40, RED, 0, fill=RED, tip='PMT position')
        panels.append(sd.col(sd.p(f'<b style="color:{CH[a]}">{a}</b> · {pmt[a]}', 22,
                                  extra=f'padding-left:{64 if i == 0 else 14}px'), P.svg(f'liquid {a}'),
                             gap=2, w=330 if i == 0 else 280))
    cb = sd.colorbar(130, 320, 0, 0.2, [(0, '0'), (0.1, '10'), (0.2, '20 %')], 'liquid eff.')
    # summary table (deck "numbers behind the maps": unbiased wall, plastic|wall, self-triggered plastic)
    eff = rd(F / 'eff_summary.csv')

    def gv(layer, probe, sample, a):
        r = eff[(eff.layer == layer) & (eff.probe == probe) & (eff['sample'] == sample) & (eff.arm == a)]
        return float(r.eff.iloc[0]) if len(r) else float('nan')

    rows, tips = [], []
    for a in 'ABCD':
        w_ = gv('wall', 'wany', 'unbiased', a)
        p_ = gv('plas', 'pm', 'unbiased', a)
        l_ = float(halves[(halves.arm == a) & (halves.probe == 'lf_full') & (halves['sample'] == 'all_late')].eff.iloc[0])
        rows.append([f'<b style="color:{CH[a]}">{a}</b>', sd.pct(w_), sd.pct(p_), '≥ 98 %', f'{100 * l_:.1f} %'])
        tips.append(f'arm {a}: wall unbiased {100 * w_:.1f} %, plastic given wall {100 * p_:.1f} %, liquid (every late trigger) {100 * l_:.2f} %')
    tab = sd.table(['arm', 'SiPM wall', 'plastic | wall', 'plastic, own trig.', 'liquid'], rows, size=21, tips=tips)
    body = (sd.title('Scintillators: SiPM walls 75–91 % and plastics ≥ 98 % efficient; the liquids answer mostly near their PMT',
                     'every gated MM track walked out through wall → plastic → liquid · liquid maps: every trigger > 10 ms, 50 mm cells')
            + sd.row(*panels, cb, gap=10, align='end')
            + sd.row(sd.col(tab, gap=0, w=760),
                     sd.col(sd.callout('Liquids: a muon gives 20–28 mV against a 16–18 mV readout threshold, so they are '
                                       'threshold-limited; A and D answer 12–23× more on their PMT half, C hardly at all.', RED, 22),
                            sd.callout('Plastic | wall is a response probability, not a hole: two in five particles that cross the '
                                       '3 mm wall stop before the plastic.', BLUE, 22), gap=16), gap=40))
    notes = """
<p><b>Efficiencies</b> (<code>ntof_scint_stack</code>, net of accidentals): the unbiased columns use events another
arm triggered, &gt; 10 ms after the flash; a counter that the arm's own trigger required reads 89–98 % on its own
events, so quote the unbiased one. Unbiased samples are a few thousand tags per arm (B is rough).</p>
<p><b>Liquid maps</b>: P(liquid fired | wall fired at both ends), every late trigger, so trigger-biased high. They
are from the 6 October pass, before the stage-3 y fix (COORDINATES.md), so the v axis may be mirrored; the u axis and
the PMT side of A and D (red mark) are not affected. B and C's PMTs sit on top; no mark is drawn for them. A: 9.6 %
behind the right (+u) bar against 1.0 % behind the left. B and C (PMT on top) rise only 1.7–4.5× toward the PMT; why
the vertical liquids show no strong rise is an open board question. Their n_TOF problem was also pile-up (67–76 % of
pulses had another pulse in their own tail).</p>
<p><b>Consequence for the future</b> (trigger_scint/TRIGGER_OPTIONS.md): repairing the liquids is not worth it for
acceptance (LS alone 1.2 % pair acceptance, LS OR plastic 4.3 %, vs ~14 % for big plastics).</p>
<p>Sources: <code>/media/dylan/data/x17/scint_stack/figures/eff_map_liq_all_late.csv</code>,
<code>eff_summary.csv</code>, <code>liquid.halves.csv</code>; note scint-stack.</p>"""
    D.slide('scint', body, notes, short='Scintillators')


def s_singles(D):
    img = rd(DATA / 'sept26_prelim/capsule_pose/image.csv')
    dat = rd(DATA / 'sept26_prelim/capsule_pose/data.csv')
    panels = []
    for i, a in enumerate('AC'):
        g = img[(img.arm == a) & (img.window == 'late')]
        vmax = float(g.excess.max())
        P = sd.Plot(300, 560, x=(-60, 40), y=(-240, 200), xlabel='x (mm)', ylabel='height (mm)' if i == 0 else '',
                    title=f'arm {a}, 20–75 ms', margin=(8, 10, 60, 78 if i == 0 else 18))
        P.xticks([(v, str(v)) for v in (-50, -25, 0, 25)])
        P.yticks([(v, str(v) if i == 0 else '') for v in (-200, -100, 0, 100, 200)])
        P.cells([(r.x - 2, r.x + 2, r.h - 10, r.h + 10, max(float(r.excess), 0) / vmax,
                  f'arm {a}: x {r.x:.0f} mm, height {r.h:.0f} mm, pointing excess {r.excess / vmax:.2f} of max') for r in g.itertuples()],
                0, 1)
        d = dat[(dat.arm == a) & (dat.window == 'late')].iloc[0]
        P.points([d.x_cross], [d.y_cross], '#ffffff', r=7,
                 tips=[f'arm {a} band crossing: x {d.x_cross:.1f} ± {d.x_err:.2f}, height {d.y_cross:.1f} ± {d.y_err:.2f} mm '
                       f'({int(d.n):,} tracks, 20–75 ms)'])
        P.vline(0, '#ffffff', '6 6', 1.5)
        panels.append(P.svg(f'capsule image {a}'))
    # cosmic A-C diagram
    o = [f'<rect x="40" y="40" width="22" height="300" fill="{BLUE}" fill-opacity="0.4" stroke="{BLUE}" stroke-width="2"/>',
         f'<rect x="398" y="40" width="22" height="300" fill="{ORANGE}" fill-opacity="0.4" stroke="{ORANGE}" stroke-width="2"/>',
         sd.T(51, 30, 'A', 26, BLUE, 'middle', 700), sd.T(409, 30, 'C', 26, ORANGE, 'middle', 700),
         f'<circle cx="230" cy="190" r="12" fill="#c9ced8" stroke="{INK}" stroke-width="2"/>',
         sd.line(10, 70, 450, 300, RED, 3, '10 6'), sd.T(14, 62, 'μ', 24, RED, 'start', 600),
         sd.T(230, 370, '470 mm lever arm: the A–C line is the angle truth', 19, MUT)]
    cos = sd.svg(460, 380, ''.join(o), 'cosmic A-C')
    body = (sd.title('Singles work: cosmic muons fix the angle scale to 1–4 %, and beam tracks image the capsule nose',
                     'left: 41 beam-off runs (57.6 h) on the n_TOF clock · right: pointing excess of late beam tracks (is2_v2, y up)')
            + sd.row(
                sd.col(cos,
                       sd.callout('The in-situ calibration (drift velocity, kernel) closes on the A–C muon line: chamber A flat '
                                  'at 1.00 in y, ±3 % in x; in-beam muons read the same scale to 1–4 %.', BLUE, 22),
                       sd.callout(f'Single-track rate {sd.term("×1.25 / ×1.15", "data tracks per trigger over Geant4 pushed through the digitised reconstruction (g4_digi), A / C")} '
                                  f'of Geant4; {sd.term("46 % / 41 %", "late tracks passing within 15 mm of the source minus a shuffled-angle null, ≥ 20 ms; Geant4 through the same reco: 31 % / 34 %")} '
                                  'point back at the source (Geant4 31 / 34 %).', GREEN, 22), gap=18, w=560),
                sd.row(*panels, gap=10),
                sd.col(sd.p('<b>One compact source</b> on the capsule nose at every time of flight', 24),
                       sd.p(f'A and C agree: x ≈ −10 mm, height −31 (A) / −35 (C) mm, where Geant4\'s capsule puts the '
                            f'Al captures (−28 by the same estimator).', 21, MUT),
                       sd.p('The two chambers\' u axes point opposite ways, so only a common shift of the pair against '
                            'the beam could fake it.', 21, MUT),
                       sd.p('The ~10 mm x offset (beam off the frame, or capsule off the beam) is a stated '
                            'limitation; no survey exists.', 21, MUT), gap=14, w=420), gap=28))
    notes = """
<p><b>Cosmic truth.</b> <code>ntof_cosmics</code>: 47 beam-off runs, 41 at the production operating point (57.6 h,
5.2 M triggers); DREAM triggers placed on the n_TOF clock to ~10 ns (<code>clock_match.py</code>). The straight line
joining A and C is the angle truth for each chamber. The beam/cosmic angle-scale gap at the walls is electron
scattering (Geant4 with an ideal reconstruction), not the reconstruction.</p>
<p><b>Source image</b> (<code>capsule_pose</code>): each late track's line, the density of where lines cross the beam
axis plane minus a shuffled null, 4 mm × 20 mm cells, normalised per panel; white dot = the band crossing (pointing
estimator, which is robust to each chamber's efficiency shape). The same spot appears at 4–12, 12–20 and 20–75 ms and
before and after the 27 July access. The late tracks are delayed (hall die-away, 23.5 ms) neutrons captured on the
capsule Al.</p>
<p><b>Why this is the best demonstration that the trackers work in situ:</b> the image needs correct angles and
positions in two independent chambers at once, and it reproduces Geant4's source position and rate.</p>"""
    D.slide('singles', body, notes, short='Singles')


def s_pairs(D):
    rows = [
        ['single tracks per trigger', 'Geant4 × digitised reco', '×1.25 (A) / ×1.15 (C)', '<b style="color:#2f8a5b">✓</b>'],
        ['where they come from', 'one source on the capsule nose', 'compact source, 2–75 ms', '<b style="color:#2f8a5b">✓</b>'],
        ['correlated 2nd track, same chamber', 'yes, rate-independent', '0.15 % (A) / 0.16 % (C), flat', '<b style="color:#2f8a5b">✓</b>'],
        ['its opening angle (A)', 'Geant4 e⁺e⁻ shape', 'χ² 6.6/6 (unrelated: 26/6)', '<b style="color:#c99318">~</b>'],
        ['pairs point to the capsule', 'no: 2 % pass, even perfect tracks', 'no excess', '<b style="color:#c99318">~</b>'],
        ['pair rate', 'Geant4 × digitised reco', '0.13 (A) / 0.12 (C) of it', '<b style="color:#c63d4f">✗ ×8 low</b>'],
    ]
    tab = sd.table(['observable', 'expected', 'data', ''], rows, size=22, align=['left', 'left', 'left', 'center'])
    P = sd.Plot(440, 430, x=(0, 4), y=(0, 1.5), ylabel='late events with a pair (%)', title='pair fraction',
                margin=(12, 12, 60, 90))
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5)])
    P.xticks([(0.9, 'A'), (2.9, 'C')])
    for x0, a, pr, seen in ((0.9, 'A', 1.19, 0.15), (2.9, 'C', 1.26, 0.16)):
        P.vbar(x0 - 0.35, pr, 56, GREY, tip=f'{a}: Geant4 through the real reconstruction predicts {pr} %', label=f'{pr}')
        P.vbar(x0 + 0.35, seen, 56, CH[a], tip=f'{a}: seen {seen} % (correlated part, rate fit 12–75 ms)', label=f'{seen}')
    body = (sd.title('The question for pairs: singles agree with Geant4, the Al pairs arrive at ⅛ — pair-specific and still open',
                     'is2_v2 re-pass at the capsule-focus k, arms A and C, tracks ≥ 12 ms, run_69 onward; Geant4 pushed through the same reconstruction')
            + sd.row(sd.col(tab, sd.legend([('Geant4 prediction', GREY, 'box'), ('data, A', BLUE, 'box'), ('data, C', ORANGE, 'box')], 21), gap=14),
                     P.svg('pair fraction'), gap=32)
            + sd.row(
                sd.card(sd.p('<b>If the deficit is in the detector</b> (two-track loss in busy events), it hits X17 far '
                             'less: X17 legs land in different chambers 470 mm apart, penetrate, fire their own '
                             'trigger (45 % vs 7.7 %) and vertex to ~10 mm.', 22), pad=22),
                sd.card(sd.p('<b>If it is in the simulation</b>: the capsule model (nose Al, wrap), the delayed-neutron '
                             'source population, or the reconstruction version the prediction used (one-sided search).',
                             22), pad=22),
                sd.card(sd.p('<b>Next with the data we have</b>: re-digitise Geant4 pairs with is2_v2; Geant4 with '
                             'delayed thermal neutrons; check the capsule model; look for a two-lepton trigger effect.',
                             22), pad=22), gap=24))
    notes = """
<p><b>Correlated pairs.</b> Accidental same-chamber pairs scale with the instantaneous rate, correlated ones do not.
Fitting the pair fraction f(t) = a + b·rate(t) over 12–75 ms gives a flat a = 0.15 % (A) / 0.16 % (C) while the rate
falls ×3.5: ~2,050 (A) and ~1,660 (C) correlated pairs. Geant4: 75–78 % of reconstructable same-chamber pairs are
one photon's e⁺e⁻.</p>
<p><b>The prediction</b> already includes the opening angle and the close-pair loss: the Geant4 events are digitised
into real quiet triggers and run through the production reconstruction (P(2|2) 0.16 / 0.14). 44 % of Geant4 Al pairs
are &lt; 20 mm apart in one view. Other efficiency routes bracket data/Geant4 at 0.12–0.52.</p>
<p><b>Why the vertex test cannot help on Al pairs:</b> two near-parallel ~2 MeV legs, each bent ~9°, cross ~75 mm from
where they were born, so even perfect tracks pass 2 % of the time (chance 3 %).</p>
<p>Source: x17-status-2026-10-15 slides 4–11; <code>same_chamber_compare.py</code>,
<code>/media/dylan/data/x17/sept26_prelim/same_chamber/cmp_*.csv</code>.</p>"""
    D.slide('pairs', body, notes, short='Singles vs pairs')


def s_energy(D):
    Sp = rd(DATA / 'calorimetry/wp/spectra.csv')
    fits = rd(DATA / 'calorimetry/wp/scale_fits.csv')
    panels = []
    for i, a in enumerate('AC'):
        g = Sp[(Sp.arm == a) & (Sp['var'] == 'e_plas')]
        c = 0.5 * (g.lo + g.hi)
        m = (c >= 3) & (c <= 9)
        dn = g.data / g.data[m].sum()
        sn = g.g4 / g.g4[m].sum()
        P = sd.Plot(470, 330, x=(2, 10), y=(0, 0.14), xlabel='plastic energy (MeV)' if i == 1 else '',
                    ylabel='fraction / 0.25 MeV', title=f'arm {a} plastic vs Geant4 (no refit)',
                    margin=(10, 16, 66 if i == 1 else 34, 96))
        P.xticks([(v, str(v)) for v in range(2, 11, 2)])
        P.yticks([(v, f'{v:.2f}') for v in (0, 0.05, 0.1)])
        sel = (g.lo >= 2) & (g.hi <= 10)
        gi = g[sel]
        xs, ys = sd.step_xy(list(gi.lo) + [float(gi.hi.iloc[-1])], list(sn[sel]))
        P.raw(sd.poly([P.X(x) for x in xs], [P.Y(min(y, 0.14)) for y in ys], G4C, 3,
                      tip=f'Geant4, {int(g.g4.sum()):,} events, same trigger and resolution, scale 1'))
        mm = sel & (g.data > 0)
        P.points(list(c[mm]), list(dn[mm]), CH[a], r=5,
                 tips=[f'{a} {lo:.2f}-{hi:.2f} MeV: data {d:,}, Geant4 {s}' for lo, hi, d, s in zip(g.lo[mm], g.hi[mm], g.data[mm], g.g4[mm])])
        panels.append(P.svg(f'plastic {a}'))
    # bar-by-bar scales, >= 40 ms
    sc = []
    for a in 'AC':
        for b in (1, 2):
            r = fits[(fits.arm == a) & (fits['var'] == 'e_plas') & (fits.subset == f'bar {b}, >= 40 ms')].iloc[0]
            sc.append((f'plastic {a}{b}', r.scale, r.err, CH[a]))
    for a in 'AC':
        r = fits[(fits.arm == a) & (fits['var'] == 'e_wall_fit') & (fits.subset == '>= 40 ms')].iloc[0]
        sc.append((f'wall {a}', r.scale, r.err, GOLD))
    Q = sd.Plot(520, 560, x=(0.85, 1.2), y=(0, len(sc)), xlabel='data ÷ Geant4 energy scale', title='bar by bar, ≥ 40 ms',
                margin=(10, 20, 92, 140))
    Q.xticks([(v, f'{v:.2f}') for v in (0.9, 1.0, 1.1, 1.2)])
    Q.band([0.95, 1.05], [0, 0], [len(sc), len(sc)], GREEN, 0.12, tip='±5 %')
    Q.vline(1.0, G4C, '6 6', 2)
    for k, (lab, s, e, col) in enumerate(sc):
        y = len(sc) - k - 0.5
        Q.raw(sd.line(Q.X(s - e), Q.Y(y), Q.X(s + e), Q.Y(y), col, 3))
        Q.points([s], [y], col, r=9, tips=[f'{lab}: {s:.3f} ± {e:.3f}'])
        Q.raw(sd.T(Q.x0 - 12, Q.Y(y) + 7, lab, 21, INK, 'end'))
    body = (sd.title('The stack measures energy: calibrated on muons alone, the plastics predict electron spectra to 2–5 %',
                     'self-triggered single tracks at n_TOF · spectra: both bars, all times · scales: per bar, ≥ 40 ms · Geant4 with the measured trigger and resolution')
            + sd.row(sd.col(*panels, gap=4), Q.svg('scales'),
                     sd.col(sd.callout('<b>Linear</b> from the muon\'s 3.4 MeV to stopping electrons at 3–8 MeV; '
                                       '<b>stable</b> to 0.5–0.8 % run to run over the production period.', GREEN, 21),
                            sd.callout('Bar C1 reads 1.15: a pure gain offset of unknown cause. Plastic PMTs lose 3–8 % gain '
                                       '12–25 ms after the flash (rate-proof bases or a pulser next time).', ORANGE, 21),
                            sd.callout('At n_TOF the 2 cm plastic stops only the soft leg (a range telescope). With 4–5 cm '
                                       'plates the full stack is a calorimeter: the LNL E_sum and pair-mass fits rest on it. '
                                       'Resolution: to be measured on a prototype plate.', BLUE, 21),
                            sd.callout('MM charge: a gain monitor (19–39 % per 40 mm cell), not a dE/dx '
                                       'or two-particle flag (FWHM/MPV 0.95).', GREY, 21), gap=16, w=560), gap=24))
    notes = """
<p><b>The test</b> (<code>ntof_calorimetry/wall_plastic.py</code>, note scint-calorimetry): calibrate the SiPM wall
and each plastic bar on cosmic muons only (beam-off runs on the n_TOF clock, path from the MM track), then predict the
beam-electron spectra with Geant4 of the full stack (total prompt deposit in the matched bar, smeared with the
measured σ/MPV, the arm's trigger applied in MeV). Geant4 is drawn at scale 1.</p>
<p><b>Scales</b> (Geant4 × s matches the data mean over 3–8 MeV, bootstrap errors): plastic A1/A2/C2 1.05/1.03/1.02,
C1 1.15; wall A/C 0.95/0.96. Most of these electrons stop in the plastic (below ~6 MeV), so their deposit is their
energy. A 25 mm bar (the run-config text) would put every scale at ×1.27 — excluded, so the 20 mm in Geant4 is
right.</p>
<p><b>Reconciling with the 8 Oct "calorimetry closed".</b> That closure was about what the n_TOF <i>stack</i> can do
for X17 pairs: with 20 mm of plastic and ~1.2 MeV lost upstream, the soft-leg energy adds only 5–8 % to the
significance, and the liquids are threshold-limited. The 14 Oct result above is about the <i>scintillators</i>: their
energy scale works. For a future run, 4–5 cm plates contain the leptons and the LNL study's E_sum / m_ee fit assumes
exactly this n_TOF-anchored response (as built it loses the mass fit, backup B4).</p>"""
    D.slide('energy', body, notes, short='Energy')


def lnl_numbers():
    J = json.load(open(FAC / 'lnl/out/talk/numbers.json'))
    c = J['cases']
    return dict(J=J, ab1=c['as built|1'], ab4=c['as built|4'], bp1=c['big plastics|1'], bp4=c['big plastics|4'])


def s_lnl(D, L):
    J = L['J']
    P = sd.Plot(520, 560, x=(0, 4), y=(0.1, 100, 'log'), ylabel='days to 3σ at R(ATOMKI)', title='reach',
                margin=(12, 16, 70, 100))
    P.yticks([(0.1, '0.1'), (1, '1'), (10, '10'), (100, '100')])
    P.xticks([(1, 'as built'), (3, 'big plastics')])
    for x0, k1, k4, c in ((1, 'ab1', 'ab4', GREY), (3, 'bp1', 'bp4', GREEN)):
        for dx, k, cur in ((-0.3, k1, 1), (0.3, k4, 4)):
            v = L[k]['days_3sigma']
            P.vbar(x0 + dx, v, 70, c, base=0.1, opacity=1.0 if cur == 1 else 0.55, label=f'{v:.1f} d',
                   tip=f'{"as built" if k[0] == "a" else "big plastics"}, {cur} µA: {v:.2f} d to 3σ, {L[k]["days_5sigma"]:.1f} d to 5σ; '
                       f'{L[k]["x17_per_day"]:.0f} X17/day in the 3D fit, {L[k]["bkg_per_day"]:.0f} background/day')
    body = (sd.title(f'LNL: repeat ATOMKI on the same apparatus — 3σ at their ratio in {L["ab1"]["days_3sigma"]:.0f} days as built, '
                     f'{L["bp1"]["days_3sigma"]:.0f} with big plastics',
                     '⁷Li(p,e⁺e⁻)⁸Be at the 18.15 MeV resonance · Geant4 of the n_TOF apparatus, MM 15° + TOF, θ × |y| × E_sum fit')
            + sd.row(
                sd.col(sd.flow([dict(label='p, 1.03–1.10 MeV', sub='AN2000 DC 1 µA; CN ~4 µA', color=RED),
                                dict(label='thin ⁷Li target', sub='Li₂O 300 µg/cm²', color=GOLD),
                                dict(label='⁸Be* 18.15 MeV', sub='X17 → e⁺e⁻ vs IPC', color=BLUE)], size=21),
                       sd.table(['', 'as built', 'big plastics'],
                                [['pair trigger (Geant4)', sd.pct(J['pair_tag']['as built'], 1), sd.pct(J['pair_tag']['big plastics'], 1)],
                                 ['X17 / day in the fit, 1 µA', f'{L["ab1"]["x17_per_day"]:.0f}', f'{L["bp1"]["x17_per_day"]:.0f}'],
                                 ['X17 made / day, 1 µA', f'{L["ab1"]["produced_per_day"]:.0f}', f'{L["bp1"]["produced_per_day"]:.0f}'],
                                 ['DREAM live', sd.pct(L['ab1']['live']), sd.pct(L['bp1']['live'])]], size=22),
                       sd.p(f'For scale: ≈ 2 X17 were made in the whole n_TOF thermal window.', 21, MUT), gap=16, w=560),
                sd.col(P.svg('LNL reach'), sd.p('solid: 1 µA (AN2000) · pale: 4 µA (CN)', 20, MUT,
                                                  extra='padding-left:100px'), gap=0, w=520),
                sd.col(sd.p('<b>Why it can work where n_TOF could not</b>', 24),
                       sd.p('· DC beam, no γ flash: the DAQ is 91–98 % live up to 4 µA.', 21),
                       sd.p('· ~900 X17 made per day at 1 µA, at the exact ATOMKI kinematics.', 21),
                       sd.p('· The 3D fit (angle, energy split, E_sum) reads a background error that copies the X17 angle '
                            'spectrum as 13–24 % of an X17, against 100 % for an angle-only fit.', 21),
                       sd.p('<b>Risks</b>: scattering in the target chamber (1.0 d at 4.5° → 6.8 d at the n_TOF capsule\'s '
                            '~14°), cosmics (MM-tracked veto), IPC shape (in-run controls, E_p scan).', 21, MUT),
                       gap=12, w=520), gap=32))
    notes = f"""
<p><b>Source</b>: <code>x17_facility_search/lnl/</code>, FEASIBILITY_SIM.md §12–18, numbers from
<code>lnl/out/talk/numbers.json</code> (Geant4 L1–L6 → <code>reach_geant.csv</code> column <code>days_3d</code>,
pair tag from <code>acceptance_geant.csv</code>). R(ATOMKI) = 5.8×10⁻⁶; 5σ = 3σ × 25/9.</p>
<p><b>Scenarios.</b> "As built": the n_TOF stack (two 20×30×2 cm plastics per arm, liquid read out). "Big plastics":
four 75×75×5 cm plates behind all 20 SiPM bars. The arm energies carry the n_TOF-anchored response
(<code>lnl_calo.py</code>).</p>
<p><b>Pseudo-data check</b> (§17): one simulated week at R(ATOMKI), blinded, with truth differing from the fit model:
r̂ = 0.87 ± 0.14 (6.1σ); errors honest up to a week, a wider fit needed at 30 days.</p>
<p><b>Caveats.</b> Z ∝ √t is optimistic past ~a week unless the wider fit is used. "X17 per day" is inside the 3D fit
after MM 15° + TOF, not just triggered. LNL's own ⁸Be spectrometer ran there in 2023–24.</p>"""
    D.slide('lnl', body, notes, short='LNL')


def s_plastics(D, L):
    d = rd(FAC / 'lnl/out/plastics/days_vs_size.csv')
    places = [(d.placement.str.startswith('where the bars'), BLUE, 'where the bars are now (R 41 cm)'),
              (d.placement.str.startswith('right behind'), GREEN, 'right behind the SiPM wall (R 35 cm)'),
              (d.placement.str.startswith('SiPM wall removed'), ORANGE, 'SiPM wall removed, behind the MM (R 26 cm)')]
    P = sd.Plot(900, 640, x=(20, 120), y=(0.5, 100, 'log'), xlabel='side of a square plastic, 5 cm thick (cm)',
                ylabel='days to 3σ at R(ATOMKI), 1 µA', title='days to 3σ vs plastic size', margin=(12, 20, 92, 104))
    P.xticks([(v, str(v)) for v in (20, 40, 60, 80, 100, 120)])
    P.yticks([(0.5, '0.5'), (1, '1'), (2, '2'), (5, '5'), (10, '10'), (20, '20'), (50, '50'), (100, '100')])
    for m, c, lab in places:
        g = d[m]
        g = g[g.days < 100]
        P.line(list(g.S_mm / 10), list(g.days), c, 3.5, markers=False,
               tip=lab)
        P.points(list(g.S_mm[::2] / 10), list(g.days[::2]), c, r=4,
                 tips=[f'{lab}: {s / 10:.1f} cm → {dd:.2f} d (X17 acceptance in 125–155°: {100 * a:.2f} %)'
                       for s, dd, a in zip(g.S_mm[::2], g.days[::2], g.acc_x17_125_155[::2])])
    P.points([75], [L['bp1']['days_3sigma']], INK, r=9, marker='open',
             tips=[f'Geant4 75×75×5 cm: {L["bp1"]["days_3sigma"]:.2f} d'])
    P.text(76.5, 1.25, 'Geant4 75×75', 20, INK)
    P.hline(L['ab1']['days_3sigma'], GREY, label=f'n_TOF stack as built: {L["ab1"]["days_3sigma"]:.0f} d (Geant4)')
    leg = sd.legend([(lab, c) for _m, c, lab in places], size=20)
    pr = L['J']['price_keur']
    cost = sd.table(['four plates', 'trigger acc.', 'cost (k€)'],
                    [['as built (2×20×30×2 cm)', '2.3 %', '0'],
                     ['70×70×2 PVT, 2×3" PMTs', '14.0 %', '79 (46–121)'],
                     ['78×80×4 PS, 2×3"', '14.4 %', '87 (50–132)'],
                     ['<b>78×80×5 PS, 2×3"</b>', '<b>14.5 %</b>', f'<b>{pr["PS"]["plates_keur"][1]:.0f}</b> ({pr["PS"]["plates_keur"][0]:.0f}–{pr["PS"]["plates_keur"][2]:.0f})'],
                     ['78×80×5 PVT, 2×3"', '14.5 %', f'{pr["PVT"]["plates_keur"][1]:.0f}']], size=21,
                    tips=['', '', 'acceptance only (no calorimetry)', '', 'polystyrene: ~70 % of PVT light, the cheap lever',
                          'cast PVT'])
    body = (sd.title('Bigger plastics: the days to 3σ fall ×17 up to ~65 cm, then flatten; four 5 cm plates cost ~€93k',
                     'geometric toy of the trigger leg anchored to Geant4 (lnl/plastic_size.py) · price model trigger_scint/costs.py')
            + sd.row(sd.col(P.svg('days vs size'), leg, gap=6),
                     sd.col(sd.p('<b>Prototype plate</b> (trigger_scint, optics toy)', 24),
                            sd.p('78 × 80 × 5 cm polystyrene, fishtail guides, a 3" PMT at each end; read all 20 SiPM '
                                 'bars (×1.4 on acceptance); FERS-5200 for the 160 SiPM ends (~€18k).', 21),
                            sd.p('Toy: ~77 pe/MeV, 4 % non-uniformity with two ends, σt ≈ 124 ps at 5 MeV; ~15 %/√E, '
                                 'about 5 % at a 10 MeV deposit.', 21, MUT),
                            cost,
                            sd.callout(f'All-in with the SiPM readout ~€{pr["PS"]["all_in_keur"][1]:.0f}k. Whole LNL run budget: '
                                       'Lean €98k · Self-sufficient €174k · Upgrade €286k.', GREEN, 21),
                            sd.p('Before buying: one plate on the cosmic bench (light yield, uniformity against an MM map, '
                                 'linearity to 12 MeV).', 20, MUT), gap=12, w=660), gap=40))
    notes = """
<p><b>The curve</b> (<code>lnl/plastic_size.py</code>, an analytic stand-in for a Geant4 size scan): straight leptons
from a 2 mm beam spot through the surveyed MM active area, Highland kicks in the MM and the SiPM wall, a leg must cross a
read-out SiPM bar and the plastic face above an energy threshold. Two free numbers are fitted to the Geant4 pair tag as
built and with 75×75 cm plastics. Days to 3σ scale as B/S² in 125–155°, anchored to the Geant4 big-plastics value;
E_sum, MM 15° and TOF efficiencies are held at their Geant4 values, so it is about geometry only. Where the plates
would collide with their neighbours they are pushed back (beyond ~75 cm).</p>
<p><b>Why it flattens:</b> past ~65–75 cm the plate covers the whole cone of leptons that crossed an MM (92 % of
X17 leptons at 75×75); the SiPM wall (83 %) is then the cap. Removing the wall (orange) leaves a plastic-only leg and
gains a little more.</p>
<p><b>Costs</b> (budgetary, EUR 2026, no labour/VAT): scintillator PVT 0.6 €/cm³, PS 0.25 €/cm³; PMT + divider 3" €3.3k;
guides €0.8–1.5k; wrapping €1.5–5k per slab; HV + digitizer €1.4k per PMT. Get quotes (Eljen/Luxium PVT, Nuvia/Epic PS).
Run budget: dylan-neff.web.cern.ch/facilities/budget.html (collaborator comments enabled).</p>"""
    D.slide('plastics', body, notes, short='Big plastics')


def ill_svg(w=1664, h=360):
    """ILL PF1B target region, side view along the beam, to scale (1 mm = s px).  G1 cell + He flight tube.
    Sources: ill/HANDOFF_SIM.md §4 (G1), FEASIBILITY_SIM.md §13 (tube, windows), he4_bag/README.md."""
    s = 1.55
    ox, cy = 640, 200          # x of the cell entrance window (y_w = -21.6 mm), beam axis height
    X = lambda mm: ox + mm * s  # noqa: E731  (mm downstream of the entrance window)
    o = []
    o.append(sd.arrow(4, cy, X(-345) - 8, cy, RED, 4, 14))
    o.append(sd.T(6, cy - 16, 'cold n', 20, RED, 'start', 600))
    o.append(sd.T(6, cy + 30, '~4.9 Å', 18, RED, 'start'))
    # collimator aperture (Ø4 cm spot)
    for y0 in (cy - 100, cy + 31):
        o.append(f'<rect x="{X(-345):.1f}" y="{y0}" width="16" height="69" fill="#5a6170"/>')
    o.append(sd.T(X(-337), cy + 128, 'aperture, Ø4 cm spot', 18, MUT))
    # He flight tube, aperture to cell window
    tr = 40
    o.append(f'<rect x="{X(-325):.1f}" y="{cy - tr}" width="{(325 - 25) * s:.1f}" height="{2 * tr}" fill="#d7ecf7" '
             f'stroke="{GREY}" stroke-width="4"/>')
    for xx in (-325, -25):
        o.append(sd.line(X(xx), cy - tr, X(xx), cy + tr, PURPLE, 3.5))
    o.append(sd.T(X(-175), cy + 7, '⁴He flight tube, ≥ 99 % He', 21, INK, 'middle', 600))
    o.append(sd.T(X(-175), cy - tr - 14, '~300 mm · 1 mm Al wall', 18, MUT))
    o.append(sd.T(X(-175), cy + tr + 26, '25 µm mylar end windows', 18, PURPLE, 'middle', 600))
    # 3He cell
    R, L = 40 * s, 300 * s
    o.append(f'<rect x="{X(0):.1f}" y="{cy - R:.1f}" width="{L:.1f}" height="{2 * R:.1f}" fill="#f6e7c8" '
             f'stroke="{GOLD}" stroke-width="3"{sd.tipattr("G1: 1 bar 3He, R 40 mm, 300 mm of gas, 1.5 bar·L; stops 99.5 % of the H113 beam")}/>')
    for k in range(70):          # capture density, exponential with the G1 median 21.6 mm
        a0 = k * 3.0
        o.append(f'<rect x="{X(a0):.1f}" y="{cy - R + 4:.1f}" width="{3.0 * s + 0.6:.1f}" height="{2 * R - 8:.1f}" '
                 f'fill="{ORANGE}" fill-opacity="{0.6 * math.exp(-a0 / 31.2):.3f}"/>')
    for yy in (cy - R, cy + R):  # carbon rods of the cage, two seen in section
        o.append(sd.line(X(0), yy, X(300), yy, '#333', 3, '3 7'))
    o.append(f'<rect x="{X(0) - 3:.1f}" y="{cy - R - 10:.1f}" width="6" height="{2 * R + 20:.1f}" fill="{BLUE}"/>')
    o.append(f'<rect x="{X(300):.1f}" y="{cy - R - 10:.1f}" width="10" height="{2 * R + 20:.1f}" fill="#2b2f38"/>')
    for y0 in (cy - R - 26, cy + R + 10):
        o.append(f'<rect x="{X(0) - 10:.1f}" y="{y0:.1f}" width="7" height="16" fill="{GREEN}"/>')
    # MM span (active +-180 mm about the median vertex), arms 204 mm off axis above/below
    a0, a1 = X(21.6 - 180), X(21.6 + 180)
    for yb, lab in ((cy - R - 62, 'MM arm above (204 mm off axis)'), (cy + R + 62, 'MM arm below')):
        o.append(f'<rect x="{a0:.1f}" y="{yb - 5:.1f}" width="{a1 - a0:.1f}" height="10" fill="{BLUE}" fill-opacity="0.3" '
                 f'stroke="{BLUE}" stroke-width="1.5"/>')
    o.append(sd.T((a0 + a1) / 2, cy - R - 76, 'Micromegas active length ±180 mm, centred on the median capture '
                  '(arms drawn closer than their 204 mm)', 18, BLUE, 'middle', 600))
    # labels, right side, with leaders
    lx = X(300) + 60
    labs = [(X(0), cy - R - 10, 40, 'Be 0.25 mm entrance window (cell)', BLUE),
            (X(0) - 6, cy + R + 26, 76, '⁶LiF-lined upstream ring (scraper)', GREEN),
            (X(150), cy - R, 112, '³He cell G1: 1 bar, R 40 mm, 300 mm of gas', GOLD),
            (X(230), cy + R, 148, 'skin 12 µm PET + 6–7 µm Al on 6 carbon rods', INK),
            (X(300) + 10, cy, 184, 'CFRP end cap (+278 mm from the window)', INK),
            (X(21.6), cy + 20, 220, 'median capture 21.6 mm in; 90 % by 89 mm, 99 % by 211 mm', ORANGE),
            (X(-25), cy + tr, 256, 'no air on the beam path: removes ~90 % of the ¹⁴N(n,γ)', PURPLE)]
    for x1, y1, ty, t, c in labs:
        o.append(sd.line(x1, y1, lx - 8, ty + 40 - 6, c, 1.3))
        o.append(sd.T(lx, ty + 40, t, 19, c, 'start', 600))
    return sd.svg(w, h, ''.join(o), 'ILL target sketch')


def s_facilities(D):
    rows = [
        ['<b>n_TOF EAR2</b> (done)', 'pulsed spallation, meV–GeV', 'MeV neutrons (79 % of the X17 rate) lost to the γ flash: DREAM '
         'recovers in 5 ms; thermal ³He self-shields, (n,p) 92 %', '≈ 2 X17 made in the thermal window; Al pairs ~10⁸'],
        ['<b>ILL PF1B</b>', 'cold reactor beam, ~10¹⁰ n/s absorbed', 'thermal s-wave only (no 1⁻ resonance); cosmics, MM occupancy, '
         'DREAM 298 µs dead time; IPC floor', '3σ reach 5.3×10⁻³ per 50-day cycle: ×3.2 above the ⁸Be level (×1.3 even with no background)'],
        ['<b>GANIL NFS</b>', 'pulsed fast n, d+Be, 1–40 MeV', 'the sub-2 MeV flux the physics wants is unmeasured (model 8–14 % of the peak '
         'per MeV); ~10⁵ n/cm²/s at 0.2–2 MeV; wall pairs ~50 per X17', 'plan only; first ask: D(d,n) or lower-E deuterons'],
    ]
    tab = sd.table(['facility', 'beam', 'the wall', 'where we stand'], rows, size=19,
                   align=['left', 'left', 'left', 'left'], widths=[220, 300, 680, 460])
    body = (sd.title('Neutron facilities are hard: n_TOF lost its MeV neutrons, the ILL falls ×3 short per cycle, NFS is unmeasured',
                     'x17_facility_search · sketch: the ILL PF1B target region, to scale along the beam')
            + tab + ill_svg(1664, 340))
    notes = """
<p><b>n_TOF</b>: mpgd26 / ntof_athens_26 (the Athens deck): the γ flash saturates DREAM (detector recovers in 2 µs, DREAM
in 5 ms); every volt of gain costs ms of beam; the run moved to thermals, where (n,p) dominates and the ³He
self-shields: 4.4 → 0.07 X17/day in 0.01–0.1 eV, ≈100 → ≈2 X17 made over the campaign's 436,373 pulses. Below ~2 eV
only s-wave channels are open (M1, and E0 with no real photon; CLAUDE.md / ipc_born.py).</p>
<p><b>ILL</b> (ill/FEASIBILITY_SIM.md §13): He flight tube + Be 0.25 mm (or 25 µm mylar) window cut the trigger rate
249 → 8–23 Hz and halve the MM occupancy; with a ceiling cosmic panel and E_sum &gt; 14 MeV the reach is 6.5×10⁻³ at
1.9×10¹⁰ n/s and 5.3×10⁻³ at 6×10¹⁰ with a Ø4 cm spot (×3.2 from ⁸Be). Walls: MM occupancy + DREAM (3.4×10⁻³ if both
cut ×10), then IPC statistics (2.1×10⁻³ at 1.2×10¹¹, ×1.3). Spin selection with polarised n + ³He is the extra lever
(POLARISED.md). ³He: bare mylar loses ~22 L per cycle → the PET + Al laminate skin (he4_bag §6).</p>
<p><b>Sketch</b>: G1 (HANDOFF_SIM §4): 1 bar, 300 mm of gas, R 40 mm, entrance window at y_w = −21.6 mm so the
median capture sits at the MM centre; 99.5 % of the H113 beam stops inside; downstream cap at +278 mm (1.2 % of vertices
beyond the +180 mm MM edge). The ⁴He "pre-chamber" is the flight tube from the aperture to the cell window: ⁴He captures
no neutrons and scatters ~13× less than air; ~90 % of the air ¹⁴N(n,γ) is on that 300 mm path. A balloon around the
whole cell is not worth it (⁴He permeates into the cell and the MM gas). Shading = capture density along the cell
(schematic exponential with the G1 median).</p>
<p><b>NFS</b> (ganil_nfs/README.md §7, §9): d+Be peaks at 17 MeV, measured only above 1.5 MeV; options: lower
deuteron energy, D(d,n)³He monoenergetic (~2.45 MeV, may never have been built), ⁷Li(p,n) near threshold. The X17
shoulder slides to 99° at 2 MeV and 66° at 14 MeV, into the IPC-rich region.</p>"""
    D.slide('facilities', body, notes, short='Neutron facilities')


def s_papers(D):
    def paper(tag, ttl, sub, secs, col):
        items = ''.join(
            f'<div style="display:flex;gap:10px;align-items:baseline"><span style="font-size:17px;font-weight:600;'
            f'color:#fff;background:{c};border-radius:6px;padding:1px 8px;white-space:nowrap">{st}</span>'
            f'<p style="font-size:20px;line-height:1.3">{t}</p></div>' for t, st, c in secs)
        return sd.card(f'<p style="font-size:20px;letter-spacing:2px;color:{col};font-weight:600">{tag}</p>'
                       + sd.p(f'<b>{ttl}</b>', 26) + sd.p(sub, 20, MUT)
                       + f'<div style="display:flex;flex-direction:column;gap:8px">{items}</div>', pad=24)
    R, P_, O, Dc = ('ready', GREEN), ('partial', GOLD), ('open', GREY), ('to plan', PURPLE)
    p1 = paper('PAPER I', 'MX17 resistive Micromegas TPCs: design and performance',
               'cosmic bench (5 chambers) + det4 at SPS H4', [
                   ('design, X/Y charge balance; setups', *R), ('HV, gain, sparks', *R),
                   ('efficiency and maps', *P_), ('gas: v(E), attachment, gap', *P_), ('timing', *O),
                   ('tracking on cosmics (requote on r06)', 'requote', RED), ('176 µm at SPS; det4 stripes', *R)], BLUE)
    p2 = paper('PAPER II', 'Reconstruction on resistive strips: why hit times fail',
               'companion to I, same day, same journal (NIM A / JINST)', [
                   ('the sharing kernel, measured head-on', *R), ('hit-time ladder compression', *P_),
                   ('the waveform forward model', *R), ('hits vs waveforms, same events', *O),
                   ('what a hit readout must record', *O), ('known-angle test at SPS', *O)], BLUE)
    p3 = paper('PAPER III', 'An X17 demonstrator at n_TOF EAR2',
               'the experiment and what it showed, as a demonstrator', [
                   ('apparatus, run, flash, dataset', *Dc), ('singles vs Geant4; capsule image', *P_),
                   ('scintillator stack: maps, energy scale', *P_), ('Al e⁺e⁻ pairs (rate open)', *P_),
                   ('lessons for the next run', *Dc)], ORANGE)
    p4 = paper('PAPER IV (option)', 'Simulation: the apparatus and where it could measure X17',
               'split from III if it grows', [
                   ('full Geant4 of the station', *P_), ('thermal accounting, IPC M1 + E0', *P_),
                   ('LNL reach and the 3D fit', *P_), ('ILL / NFS feasibility', *P_)], GREY)
    body = (sd.title('Papers: two detector companions now, an EAR2 demonstrator next, simulation possibly on its own',
                     'state per section from mx_june_cosmic_qa/PAPER_PLAN.md; LaTeX outlines in papers/paper1, paper2')
            + sd.row(p1, p2, p3, p4, gap=24)
            + sd.p('Order: I and II together (I cites II for every position, angle and drift velocity). III needs the pair '
                   'rate question settled or stated as a limitation; IV carries the facility reach so III stays about the data.',
                   22, MUT))
    notes = """
<p><b>Papers I and II</b> (decision 2026-10-08, PAPER_PLAN.md): the cosmic bench and SPS in both; post to arXiv on the
same day as companions. Biggest open items: hits vs waveforms on the same events (R1–R5), timing on waveforms (D3),
reproducing the r06 numbers for quoting (done for the golden keys, C1), the calibration freeze (on hold).</p>
<p><b>Paper III</b>: written as a demonstrator: the station, the run and its limits (flash, thermal-only, noise step
on 23 July), the single tracks and capsule image against Geant4, the scintillator stack and energy scale, the Al pair
analysis with the ⅛ rate stated as an open result, and the lessons that feed LNL.</p>
<p><b>Paper IV</b> (optional): the full Geant4 of the apparatus, the thermal accounting, the IPC expectation (M1 + E0,
ipc_born.py) and the facility reach (LNL 3D fit, ILL, NFS). Splitting it keeps III about the data.</p>"""
    D.slide('papers', body, notes, short='Papers')


# =========================================================================== #
# backup
# =========================================================================== #
def b_lessons(D):
    rows = [
        ['γ flash saturates DREAM (5 ms recovery); MeV neutrons lost', 'flash-tolerant, low-dead-time readout (DREAM 298 µs)', 'neutron facilities'],
        ['plastics cover 40×30 cm of a ~75×68 cm lepton cone: 2.3 % pair trigger', 'four ~75×75×5 cm plates, all 20 SiPM bars read', 'everywhere: ×7 trigger, calorimetry'],
        ['liquids threshold-limited (MIP 20–28 mV vs 16–18 mV), pile-up', 'drop them; thick plastic instead', 'everywhere'],
        ['16 of 20 SiPM bars ganged, read on facility digitizers', 'FERS-5200 / PETsys for 160 ends (~€18–22k)', 'everywhere (the digitizers stay)'],
        ['capsule wall = 83 % of the budget: ~16° scattering at the X17 peak', 'thin CFRP/mylar target wall, thin MM window: σ68 → ~4°', 'LNL: 6.8 d → 1.0 d'],
        ['~1.2 MeV lost before the plastic; soft-leg energy gone', 'calorimetric first layer, less material upstream', 'pair-mass fit'],
        ['C and D chambers weak (dead channels, sparks)', 're-bulk C and D, fix B drift, new drift windows', 'all'],
        ['capsule vs beam never surveyed (~10 mm offset)', 'survey and photograph the target in the beam', 'all'],
        ['noise doubled on 23 July (readout clock divider)', 'freeze the DAQ configuration before production', 'all'],
    ]
    body = (sd.title('What n_TOF taught us for the next run', 'each row: what we found → what to change → where it matters')
            + sd.table(['found at n_TOF', 'change', 'pays off at'], rows, size=21, align=['left', 'left', 'left'],
                       widths=[700, 620, 340]))
    D.slide('b-lessons', body, """<p>Sources: scint-stack, calorimetry and status notes; trigger_scint/TRIGGER_OPTIONS.md;
scattering-upgrade note; ntof_pedestal_qa (23 July step); x17_facility_search budget page (per-MM status);
ill/FEASIBILITY_SIM.md §13 (DREAM dead time).</p>""", short='B1 Lessons')


def b_scatter(D):
    P = sd.Plot(760, 560, x=(0, 30), y=(0.5, 50, 'log'), xlabel='lepton scattering at the target, σ (°)',
                ylabel='LNL days to 3σ, big plastics', title='LNL reach vs target-region scattering', margin=(12, 20, 92, 104))
    P.xticks([(v, str(v)) for v in (0, 5, 10, 15, 20, 25, 30)])
    P.yticks([(0.5, '0.5'), (1, '1'), (2, '2'), (5, '5'), (10, '10'), (20, '20'), (50, '50')])
    pts = [(4.5, 1.0, 'CFRP 0.4 mm'), (9, 2.5, '9°'), (14, 6.8, 'n_TOF capsule'), (27, 31, '1 mm steel')]
    P.line([p[0] for p in pts], [p[1] for p in pts], RED, tips=[f'{t}: ~{x}° → {d} d (3D fit)' for x, d, t in pts])
    for x, d_, t in pts:
        P.text(x + 0.6, d_ * 0.8, t, 19, INK)
    rows = [['baseline (n_TOF)', '1.8 %', '9.8°', '16°'], ['upgrade: thin wall + MM window', '0.13 %', '2.3°', '3.8°'],
            ['+ ⁴He instead of air', '~0.07 %', '—', '2.5°']]
    body = (sd.title('Scattering is the design driver for the target chamber: 16° → 4° is ×7 in LNL reach',
                     'scattering-upgrade note (ray-traced budget, Geant4 direction error) · lnl_smear_nd.py (3D fit with a floating background)')
            + sd.row(P.svg('scatter'), sd.col(sd.table(['', 'x/X₀', 'per track @ 10 MeV', 'X17 peak σ68'], rows, size=22),
                                               sd.callout('The capsule wall is 83 % of the n_TOF budget; after the upgrade 63 % is air. '
                                                          'The 109° shoulder returns: peak height ×0.22 → ×0.50, FWHM 37° → 11°.', BLUE, 22),
                                               sd.callout('m_ee alone degrades much faster (1.7 → 284 d); the 3D fit is robust.', GREY, 22),
                                               gap=18), gap=40))
    D.slide('b-scatter', body, """<p>LNL FEASIBILITY_SIM.md §15; note scattering-upgrade (2026-10-06).</p>""", short='B3 Scattering')


def b_lnlcalo(D):
    rows = [('big plastics, 3D fit', 1.0, GREEN), ('big plastics, worst structure tried', 1.2, GREEN),
            ('as built, liquid repaired, 3D fit', 17, GREY), ('as built, liquid as at n_TOF', 190, RED),
            ('as built, m_ee alone, worst', 529, RED)]
    body = (sd.title('LNL needs the 5 cm plates for energy: as built, the 2 cm bars stop neither lepton',
                     'days to 3σ at R(ATOMKI), 1 µA, arm energies with the n_TOF-anchored response (lnl_calo.py, lnl_nd.py)')
            + sd.hbars([(a, b, c, f'{a}: {b} d') for a, b, c in rows], 600, width=900, log=True, vmin=0.5, label_w=520,
                       fmt=lambda v: f'{v:g} d')
            + sd.callout('Impostor test: a background error that copies the X17 angle spectrum exactly reads as 100 % of an X17 in an '
                         'angle-only fit, 13–24 % in the θ × |y| × E_sum fit, with χ² ≈ 20 left over.', BLUE, 24))
    D.slide('b-lnlcalo', body, """<p>LNL FEASIBILITY_SIM.md §14 and §16. The "worst structure" rows put the same
background-shape error on the events seen by every analysis.</p>""", short='B4 LNL energy')


def b_ill(D):
    rows = [('§12: CFRP, no panel, E_sum > 13, 1.9×10¹⁰ n/s', 7.4, GREY),
            ('+ He tube + thin window, cosmic panel, E_sum > 14', 4.0, BLUE),
            ('same, Ø4 cm spot, 6×10¹⁰ n/s', 3.2, BLUE),
            ('+ MM occupancy ÷10, 10 µs DAQ, 10¹¹ n/s', 2.1, PURPLE),
            ('IPC-only floor at 1.2×10¹¹ n/s', 1.3, GREEN)]
    body = (sd.title('ILL: one 50-day cycle stays ×1.3–3 above the ⁸Be level even with every upgrade',
                     '3σ reach in X17/IPC(M1), in units of the ⁸Be ratio (1.64×10⁻³) · ill/FEASIBILITY_SIM.md §13.5')
            + sd.hbars([(a, b, c, f'{a}: ×{b} ⁸Be') for a, b, c in rows], 8, width=760, label_w=720, fmt=lambda v: f'×{v:g}')
            + sd.callout('Remaining levers: X17 efficiency (energy containment, ×4 signal with a calorimeter), spin selection '
                         '(polarised n + ³He), more cycles. Feasible only with ~200 ps arm timing and a cosmic veto.', ORANGE, 24))
    D.slide('b-ill', body, """<p>Thermal capture at rest covers the s-wave half of the physics only; the reference ratio
2.5×10⁻² would be a ~6–9σ effect in one cycle with timing + veto (ill/README.md).</p>""", short='B5 ILL reach')


def b_budget(D):
    rows = [[f'<b style="color:{CH[a]}">{a}</b> {DET[a]}', s, e] for a, s, e in (
        ('A', 'good', '92.9 %'), ('B', 'drift fix', '91.3 %'), ('C', 're-bulk', '57.8 %*'), ('D', 're-bulk, worst', '43.1 %*'))]
    body = (sd.title('Running again: the chambers need work, and the run costs €98–286k',
                     'x17_facility_search/budget (live page, collaborator comments) · per-MM status from the June bench')
            + sd.row(sd.col(sd.p('<b>Micromegas</b>: all four get aluminised-mylar drift windows', 24),
                            sd.table(['chamber', 'status', 'bench eff. (hits)'], rows, size=22),
                            sd.p('* hits-based numbers on the budget page; the waveform reconstruction gives 74.9 % (C) and '
                                 '57.1 % (D).', 19, MUT), gap=14, w=760),
                     sd.col(sd.hbars([('Lean', 98, GREY, 'borrow most'), ('Self-sufficient (default)', 174, BLUE, ''),
                                      ('Upgrade', 286, GREEN, 'big plates, full SiPM readout')], 300, width=500,
                                     label_w=330, fmt=lambda v: f'€{v}k'),
                            sd.p('Rented at CERN, so bought now: N1081B, FIFOs. From CEA: DAQ PC, one NIM bin. '
                                 'Plastics and PMTs were lent by a colleague in Italy. DREAM LV ~45 A; SiPM bias ~63 V.',
                                 21, MUT), gap=18), gap=40))
    D.slide('b-budget', body, """<p>Page: dylan-neff.web.cern.ch/facilities/budget.html (items in
<code>budget/items.py</code>). The plates line excludes HV and digitizer, which are separate items.</p>""",
            short='B6 Budget')


def b_twotrack(D):
    rows = [['physical two-track limit', '~1 strip pitch (0.78 mm)'],
            ['real tracks (forward-model mismatch)', '~2–3 mm'],
            ['re-pass is2_v2: second track lost below', '~20 mm in either view (production ~40 mm)'],
            ['joint fit, coincident pairs < 12 mm (overlay)', 'A 19 → 58 %, C 36 → 49 %'],
            ['clean single muons falsely split', '0.50 % (A), 0.47 % (C)'],
            ['close pairs are parallel pairs', 'divergence 0.13 × separation across the gap']]
    body = (sd.title('Two tracks in one chamber: separable down to a few mm in principle, ~20 mm in production today',
                     'two-track-limit note; SAME_CHAMBER_PAIRS.md (T1)')
            + sd.table(['', ''], rows, size=24, align=['left', 'left'])
            + sd.callout('X17 legs never share a chamber (opening ≥ 110°), so this matters for the Al-pair demonstrator and for '
                         'narrow IPC pairs, not for the X17 signal itself.', BLUE, 24))
    D.slide('b-twotrack', body, '', short='B7 Two-track')


def b_flash(D):
    body = (sd.title('Why n_TOF could not see X17: the rate was in the MeV neutrons the flash took away',
                     'mpgd26 / ntof_athens_26 decks; x17_rate_3He (rate tables, thermal-corrected)')
            + sd.row(sd.bignum('79 %', 'of the X17 rate in the MeV', RED, 'p-wave 1⁻ resonance region, E_n 0.17–2 MeV', dark=False),
                     sd.bignum('5 ms', 'DREAM recovery after the γ flash', ORANGE, 'the detector itself recovers in 2 µs', dark=False),
                     sd.bignum('≈100 → ≈2', 'X17 made over the campaign', GREY,
                               'full spectrum vs the thermal window (0.01–0.1 eV: 4.4 → 0.07 X17/day)', dark=False), gap=40)
            + sd.callout('At thermal energies only s-wave channels are open (M1, and E0 with no real photon), (n,p) takes 92 % of the '
                         'neutrons and the ³He self-shields. The ²⁷Al pairs (~10⁸ per 30 days) became the demonstrator.', BLUE, 24))
    D.slide('b-flash', body, '', short='B2 Flash')


def main():
    L = lnl_numbers()
    D = sd.Deck('MX17 and the future X17 search', 'Status of the MX17 detectors and the n_TOF run, and whether the '
                'apparatus can search for X17 at LNL or a neutron facility')
    s_cover(D, L)
    s_setups(D)
    s_best(D)
    s_fleet(D)
    s_expect(D)
    s_scint(D)
    s_singles(D)
    s_pairs(D)
    s_energy(D)
    s_lnl(D, L)
    s_plastics(D, L)
    s_facilities(D)
    s_papers(D)
    b_lessons(D)
    b_flash(D)
    b_scatter(D)
    b_lnlcalo(D)
    b_ill(D)
    b_budget(D)
    b_twotrack(D)
    out = D.write(OUT / 'mx17-future-x17.html', note_meta=dict(
        title='MX17 and the future X17 search (19 Oct 2026)',
        summary='Detectors (bench, SPS, fleet), what n_TOF showed (scintillators, singles vs pairs, energy), LNL and the '
                'big plastics, the neutron facilities, and the papers.',
        tags='X17,status,LNL,ILL,MX17,papers', date='2026-10-19'),
        footer='Built by nTof_x17/talks/future_oct26/make_deck.py from the analyses\' own outputs.')
    print(out)


if __name__ == '__main__':
    main()
