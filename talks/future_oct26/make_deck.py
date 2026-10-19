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


def _effmaps():
    """mx_june_wft/report/make_bench_effmaps: ray tables, detector-frame transform, sliding kernel."""
    for q in ('mx_june_wft/report', 'mpgd26', '', 'mx_june_cosmic_qa'):
        if str(REPO / q) not in sys.path:
            sys.path.insert(0, str(REPO / q))
    import make_bench_effmaps as BE
    return BE


def kernel_png(arm, sigma=4.0):
    """The MPGD26/Athens sliding-kernel map (Gaussian circle, sigma 4 mm, 0.5 mm steps) as a viridis
    PNG with transparent outside the active area; cached next to the deck.  Returns (data URI, extent)."""
    import base64
    cache = OUT / 'img' / f'kernel_{arm}_s{sigma:g}.png'
    meta = cache.with_suffix('.json')
    if not cache.exists() or not meta.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        BE = _effmaps()
        _d, _box, eff, _cnt, extent, st = BE.make_map(KEY[arm], sigma)
        cache.parent.mkdir(parents=True, exist_ok=True)
        rgba = plt.get_cmap('viridis')(np.nan_to_num(eff.T[::-1], nan=0.0))
        rgba[..., 3] = np.isfinite(eff.T[::-1])
        plt.imsave(cache, rgba)
        meta.write_text(json.dumps(dict(extent=[float(e) for e in extent], **{k: st[k] for k in ('n_rays', 'integrated', 'p50')})))
    m = json.loads(meta.read_text())
    return 'data:image/png;base64,' + base64.b64encode(cache.read_bytes()).decode(), m


def kernel_map(arm, w, h, xlabel=False, ylabel=False, title=None):
    uri, m = kernel_png(arm)
    x0, x1, y0, y1 = m['extent']
    P = sd.Plot(w, h, x=(0, 400), y=(10, 390), xlabel='detector x (mm)' if xlabel else '',
                ylabel='detector y (mm)' if ylabel else '', title=title,
                margin=(8, 8, 56 if xlabel else 30, 70 if ylabel else 18))
    P.xticks([(t, str(t)) for t in (0, 100, 200, 300, 400)])
    P.yticks([(t, str(t) if ylabel else '') for t in (100, 200, 300)])
    tip = f'{DET[arm]} ({arm}): {m["n_rays"]:,} reference muons, {m["integrated"]:.1f} % within 5 mm'
    P.raw(f'<image href="{uri}" x="{P.X(x0):.1f}" y="{P.Y(y1):.1f}" width="{P.X(x1) - P.X(x0):.1f}" '
          f'height="{P.Y(y0) - P.Y(y1):.1f}" preserveAspectRatio="none"{sd.tipattr(tip)}/>')
    return P


# =========================================================================== #
# slides
# =========================================================================== #
def s_cover(D, L, AL):
    big = sd.row(
        sd.bignum('176 µm', 'intrinsic resolution', sd.DBLUE,
                  'det4 in the SPS H4 beam: 0.30 × the strip pitch',
                  tip='sigma_det4 = 176 ± 10 µm from the uRWELL residuals at three pointing errors '
                      '(spatial_resolution/results.json)', size=62),
        sd.bignum('93 % · 1.2°', 'best chamber on cosmics', sd.DBLUE,
                  'det3: reconstructed within 5 mm; angle σ68, flat from 0 to 18°', size=62),
        sd.bignum(f'{AL["singles"] / 1e6:.1f} M · {AL["pairs"] / 1e3:.1f}k', 'Al singles · pairs in the data', sd.DGREEN,
                  'reconstructed, n_TOF chambers A + C, ≥ 12 ms; pairs come in at ⅛ of Geant4 (being worked on)',
                  tip=f'single tracks ≥ 12 ms: A {AL["A"]["singles"]:,}, C {AL["C"]["singles"]:,} · correlated '
                      f'same-chamber pairs (rate fit): A {AL["A"]["pairs"]:,.0f}, C {AL["C"]["pairs"]:,.0f}', size=62),
        sd.bignum(f'{L["bp1"]["days_3sigma"]:.1f} d', 'to 3σ at the ATOMKI ratio, LNL', sd.DGREEN,
                  f'with four big plastics at 1 µA; {L["ab1"]["days_3sigma"]:.0f} d with the n_TOF stack as built', size=62),
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
dead time; backup B4). Can this detector system search for X17 somewhere else, and what does it need?</p>
<p><b>Short answer.</b> The Micromegas TPCs track well (best chamber 93 % efficient, 1.2° angles, 176 µm
intrinsic) and, in the beam, single tracks agree with Geant4 and image the target. The scintillators hold a muon
energy scale that predicts electron spectra to 2–5 %. At LNL (⁷Li(p,e⁺e⁻)⁸Be, the ATOMKI reaction) the Geant4
reach is 17 days to 3σ at the ATOMKI ratio as built and about a day with four large plastics. The neutron
facilities are much harder (slide 11).</p>
<p><b>What is still open.</b> At n_TOF the same-chamber Al e⁺e⁻ pairs come in at ⅛ of the Geant4 prediction
while the singles agree; until that is understood, the two-track efficiency for soft, close pairs is not
demonstrated (slide 7). X17 pairs are a far easier topology.</p>
<p>Hover the dotted terms and the data points for values, counts and provenance. Backup slides follow the main deck.</p>"""
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
    D.slide('b-setups', body, notes, short='B3 Setups')


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

    M = kernel_map('A', 440, 560, xlabel=True, ylabel=True, title='det3 efficiency, σ 4 mm kernel')
    km = kernel_png('A')[1]
    e3, n3 = km['integrated'] / 100, km['n_rays']
    body = (sd.title('Micromegas TPC performance: the best chamber is 93 % efficient, 1.2° in angle, 176 µm intrinsic',
                     'det3 (n_TOF arm A) on the Saclay cosmic bench against the M3 telescope · det4 in the SPS H4 beam against uRWELLs')
            + sd.row(P.svg('angle resolution'), sd.row(M.svg('det3 efficiency'), sd.colorbar(96, 420, 0, 1, [(0, '0'), (0.5, '50'), (1, '100 %')], 'within 5 mm'), gap=2, align='center'), Q.svg('SPS resolution'), gap=16)
            + sd.row(sd.legend([('X plane / uRWELL x', BLUE, 'dot'), ('Y plane / uRWELL y', ORANGE, 'dot')], size=22)))
    notes = f"""
<p><b>Left: angles.</b> σ68 of (reconstructed − M3 reference) track angle in bins of |reference angle|, det3
(<code>sat_det3</code>, r06 bundle, waveform forward model). Flat from head-on, where a drift-time ladder has no lever
arm, to 18°. The cosmic σ68 includes the M3 reference's own error and multiple scattering. Source
<code>mpgd26/data/angle_resolution.json</code>.</p>
<p><b>Middle: efficiency.</b> P(reconstructed within 5 mm of the M3 track), Gaussian sliding kernel σ 4 mm in 0.5 mm
steps (the MPGD26 / Athens map, <code>make_bench_effmaps.make_map</code>), <code>g_det3_wknd</code>,
{n3:,} reference muons, overall {100 * e3:.1f} %. Detection alone (any hit) is ~100 %; the 5 mm requirement is what
the tracking needs. Losses are sparks (~8 %) and far reconstructions.</p>
<p><b>Right: intrinsic resolution.</b> det4 residuals to the uRWELL track in zones of different telescope pitch, so of
different pointing error. σ_res² = pointing² + σ_det4², fitted σ_det4 = {1000 * sig:.0f} ± {1000 * S['fit']['sigma_det4_err']:.0f} µm
= 0.30 × the 0.78 mm pitch. On the bench the position residual is ~0.45 mm core, dominated by M3 and scattering.</p>
<p><b>The setups</b> (timeline and drawings: backup). Saclay cosmic bench, June 2026: four M3 reference Micromegas
(50×50 cm) track the muons through two MX17 test slots. SPS H4, 31 Jul – 3 Aug 2026: det4 parasitic in the P2 uRWELL
test beam, straight high-momentum tracks.</p>
<p><b>Hits vs waveforms.</b> All geometry here comes from the waveform forward model (Paper II). Per-strip hit times
compress the drift ladder by 20–30 % on resistive strips; hit-based angles were 2.4–2.6° on the same data.</p>"""
    D.slide('best', body, notes, short='Best chamber')


def map_head(a, extra=''):
    f = FLEET[a]
    flag = ' <b style="color:#c63d4f;font-size:28px">DRIFT</b>' if a == 'B' else ''
    return sd.p(f'<b style="color:{CH[a]}">{a}</b> · {DET[a]} · {f[0]:.0f} %{flag}{extra}', 23)


def s_fleet(D):
    maps = []
    tot = {}
    for i, a in enumerate('ABCDE'):
        P = kernel_map(a, 300 + (70 if i == 0 else 0), 330, xlabel=False, ylabel=(i == 0))
        km = kernel_png(a)[1]
        tot[a] = (km['integrated'] / 100, km['n_rays'])
        maps.append(sd.col(map_head(a), P.svg(f'{DET[a]} map'), gap=2))
    cb = sd.colorbar(120, 330, 0, 1, [(0, '0'), (0.5, '50'), (1, '100 %')], 'within 5 mm')
    rows = []
    tips = []
    for a in 'ABCDE':
        f = FLEET[a]
        v = VERDICT[a]
        col = CH[a]
        verdict = (f'<b>good;</b> <b style="color:#c63d4f">DRIFT to fix</b>' if a == 'B' else f'<b>{v[0]}</b>')
        rows.append([f'<b style="color:{col}">{a}</b> {DET[a]}', f'{f[0]:.1f} %', f'{f[1]:.2f} mm',
                     f'{f[2]:.2f}° / {f[3]:.2f}°', f'{f[4]:.0f} %', verdict, v[1]])
        tips.append(f'{DET[a]}: rays {tot[a][1]:,}; map overall {100 * tot[a][0]:.1f} % within 5 mm')
    tab = sd.table(['chamber', 'within 5 mm', 'core σ_r', 'σθ X / Y', 'sparks', 'verdict', 'role / issue'],
                   rows, size=22, tips=tips, align=['left', 'right', 'right', 'right', 'right', 'left', 'left'])
    body = (sd.title('The fleet: two good chambers, two that need re-bulking, one that does not amplify over 62 % of its area',
                     'June cosmic bench, waveform reco (r06), own detector frame · within 5 mm of the M3 track, σ 4 mm kernel')
            + sd.row(*maps, cb, gap=10, align='end')
            + tab)
    notes = """
<p><b>Maps</b>: per-ray tables of <code>mx_june_wft/report/make_bench_effmaps.py</code> (same datasets as the
colleague PDF), with the sliding kernel of the MPGD26 / Athens efficiency slide (Gaussian, σ 4 mm, 0.5 mm steps;
<code>make_bench_effmaps.make_map</code>). Pixel-to-pixel texture is counting noise (≈ 7 effective muons per
kernel). Each map is in the chamber's own
frame (<code>make_bench_effmaps.to_detector_frame</code>: the chambers sat at θ ≈ 90° on the bench, so detector y is
bench x), which puts dead strips and connector patterns vertical; the outline is the active area. <b>Table</b>: <code>mx_june_wft/FLEET_DIGEST.md</code> (r06, five golden keys; det3 there is
<code>sat_det3</code>, 93.3 %, the map uses <code>g_det3_wknd</code>, 93.1 %).</p>
<p><b>What does not work, and why.</b> det6 (C): 14 dead X channels in a connector pattern, 22 % sparking events.
det7 (D): sparks in 37 % of events; at n_TOF ~130 of its 512 X channels were dead. det4 (E): 62 % of the area does
not amplify, in stripes (det4_sps_assessment), and its amplitude-dependent X width is its dead and weak strips
(cloud_basics §35). det2 (B) tracks well but its drift field needs fixing before it can be used for more than
tagging. The budget page carries the per-chamber plan: A good, B drift fix, C and D re-bulk, new
aluminised-mylar drift windows for all four.</p>
<p><b>Angles</b> are 1.2° on the two good chambers and ~2–2.5° on the others (fewer, noisier strips per track).</p>"""
    D.slide('fleet', body, notes, short='Fleet')


def s_expect(D, AL):
    flow = sd.flow([
        dict(label='n into the capsule', sub='³He(n,p) 92 %, (n,γ) 0.81 %', color=GREY,
             tip='Geant4 thermal campaign, 10⁹ n, nose-first capsule; per neutron entering the capsule, E < 2 eV'),
        dict(label='capture on the Al vessel', sub='61 % of captures; 7.72 MeV γ + cascades', color=ORANGE),
        dict(label='trigger leg', sub='Compton e⁻ 77 %; e⁺e⁻ made in the capsule 10 %', color=BLUE,
             tip='what makes a trigger leg (wall ∧ plastic): Compton e⁻ of a capsule γ 76.7 %, e⁺e⁻ made in the '
                 'capsule 9.7 %, γ converting in the structure 6.8 %, capture outside 2.7 %, mixed 4.2 %'),
    ], size=22)
    cartoon = AL['OC'].pair_diagram().replace('width="560" height="270"', 'width="800" height="386"', 1)
    sg, pg = AL['singles_g4'], AL['pairs_g4']
    B = g4_breakdown(AL)
    M_ = lambda v: f'{v / 1e6:.1f} M' if v >= 1e6 else (f'{v / 1e6:.2f} M' if v >= 1e5 else f'{v / 1e3:.0f} k')  # noqa: E731
    lim = 3 * B['n_late'] / B['n_trig_g4']
    tab_s = sd.table(['single tracks', *[f'<b style="color:{CH[a]}">{a}</b>' for a in 'ABCD']],
                     [['in the gap (Geant4)', *[M_(B['singles'][a]) for a in 'ABCD']],
                      ['reconstructed', M_(B['singles_reco']['A']), '–', M_(B['singles_reco']['C']), '–']],
                     size=21, tips=['Geant4 track-like leptons (≥ 20 mm of gap) entering the gap with |tan| < 0.6, '
                                    'in triggered events, scaled to the late sample (no detector efficiency)',
                                    '× the digitised-reconstruction efficiency (A 0.54, C 0.45); B and D were not '
                                    'reconstructed in production'])
    opp = lambda k: '0' + (f' (&lt; {lim / 1e3:.0f} k)' if k == 'gap' else '')  # noqa: E731
    tab_p = sd.table(['e⁺e⁻ pairs', 'same chamber', 'adjacent', 'opposite'],
                     [['both legs in a gap', M_(B['pairs_gap']['same']), M_(B['pairs_gap']['adjacent']), opp('gap')],
                      ['both reach the SiPM wall', M_(B['pairs_wall']['same']), M_(B['pairs_wall']['adjacent']), opp('wall')]],
                     size=21, tips=[f'Geant4: e⁺e⁻ of one conversion, both legs track-like and in acceptance, in a '
                                    f'triggered event, all four chambers ({B["pairs_gap_n"]["same"]} / {B["pairs_gap_n"]["adjacent"]} / '
                                    f'0 simulated pairs; opposite: 95 % limit)',
                                    'and each leg leaves ≥ ½ MIP in the 3 mm SiPM wall '
                                    f'({B["pairs_wall_n"]["same"]} / {B["pairs_wall_n"]["adjacent"]} / 0 simulated)'])
    nums = sd.col(
        sd.bignum(f'{sg / 1e6:.1f} M', 'single tracks expected, A + C', BLUE,
                  'Geant4 × the reconstruction efficiency × the late triggers', dark=False, size=50,
                  tip=f'Geant4 singles × the digitised-reconstruction efficiency, scaled to the data triggers: '
                      f'A {AL["A"]["singles_g4"]:,.0f}, C {AL["C"]["singles_g4"]:,.0f}'),
        tab_s,
        sd.bignum(f'{pg / 1e3:.0f} k', 'same-chamber pairs expected, A + C', RED,
                  'reconstructed as two tracks; close-pair losses included', dark=False, size=50,
                  tip=f'predicted pair fraction × tracked late events: A {AL["A"]["pairs_g4"]:,.0f}, C {AL["C"]["pairs_g4"]:,.0f}'),
        tab_p,
        gap=14, w=720)
    body = (sd.title('What n_TOF could show: Al-capture e⁺e⁻ pairs, the hardest version of the X17 signature',
                     'Geant4 thermal campaign (10⁹ n, nose-first capsule), digitised through the production reconstruction')
            + sd.row(sd.col(cartoon, sd.p('one arm, side view: a soft Al pair stops in the stack; an X17 leg goes through', 21, MUT),
                            gap=6), nums, gap=40))
    notes = """
<p><b>The chain</b> (Geant4 thermal campaign, 10⁹ n, nose-first capsule): of the neutrons entering the capsule,
³He(n,p) takes 92 % and (n,γ) 0.81 %; captures on the Al vessel are 61 % of the rest (7.72 MeV γ + cascades). A
trigger leg (wall ∧ plastic) is a Compton e⁻ of a capsule γ 76.7 %, an e⁺e⁻ made in the capsule 9.7 %, a γ
converting in the structure 6.8 %, capture outside 2.7 %, mixed 4.2 %.</p>
<p><b>Why the Al pairs.</b> After the 1 ms flash veto every neutron is thermal; the trigger fires on the 7.72 MeV
Al-capture γ of the capsule vessel (its Compton electrons). About one trigger leg in ten is an e⁺e⁻ made in the
capsule. Those pairs are the proof-of-concept sample: ~2 MeV legs that scatter ~9° on the way out (p·β·θ =
0.40 MeV·rad), mostly stop in the wall or the plastic, and land ~25° apart in one chamber. That is the hardest
two-track case; X17 legs (8–10 MeV typical) scatter 2.3–2.9°, cross the whole stack and go to opposite chambers.</p>
<p><b>The expected counts</b> are for the analysed sample: chambers A and C, tracks ≥ 12 ms after the flash, run_69
onward, is2_v2 re-pass. Singles: Geant4 track-like singles per trigger × the efficiency of the digitised
reconstruction, times the data triggers. Pairs: the Geant4 ratio of two reconstructable leptons to one × the measured
P(2 tracks | 2 leptons), so the opening angle and the close-pair loss are in it. About 10⁸ Al e⁺e⁻ are <i>made</i> per
30 days (Athens: external conversion 7.8×10⁷ + internal pairs 1.4×10⁷); these are the ones the chambers can record.</p>
<p><b>What we can do with the data, for proof of concept:</b> (1) single tracks against Geant4 (rate, source image,
angle scale on cosmic muons); (2) the scintillator response and energy scale track by track; (3) the correlated
same-chamber pairs (rate, opening angle) against Geant4 pushed through the same reconstruction.</p>
<p>Sources: x17-status-2026-10-15 (<code>sept26_prelim_analysis/make_status_oct15_deck.py</code>,
<code>same_chamber_compare.py</code> → <code>cmp_expectation.csv</code>, <code>cmp.meta.json</code>), Athens
<code>thermal_accounting</code>.</p>"""
    notes += f"""
<p><b>The tables</b> (<code>g4_breakdown</code>, Geant4 per-lepton truth of <code>same_chamber_g4</code>, all four
arms). They count what reaches the chambers, with no detector efficiency: a lepton is in the gap when it crosses
≥ 20 mm of drift depth heading in with |tan| &lt; 0.6, in an event that fires the trigger (wall ∧ plastic in any arm).
Counts per Geant4 trigger are scaled by {B['n_late'] / 1e6:.1f} M, the late-trigger count that turns Geant4 × the
digitised efficiency into the 2.0 M above. <i>Reconstructed</i> is only known for A and C (digitised Geant4 through the
production chain); B and D were not reconstructed in production. Pairs are the e⁺ and e⁻ of one conversion, by where
the two legs land: same chamber, adjacent chambers (90° apart) or opposite chambers (180°, A–C or B–D). The 29 k is
the reconstructed same-chamber number (any two leptons, 78 % e⁺e⁻, × the measured P(2 tracks | 2 leptons)), so it
is not the first table cell × an efficiency. No Al pair reaches opposite chambers: 0 of {B['n_trig_g4']:,} simulated
triggers (4 untriggered ones without the acceptance cut), &lt; {3 * B['n_late'] / B['n_trig_g4'] / 1e3:.0f} k at 95 %.
That is the topology X17 populates. <i>Reaches the SiPM wall</i>: each leg leaves ≥ ½ MIP in the 3 mm wall. Geant4 has
external conversions only; internal pairs add ~18 % (Athens accounting, below) with the same topology.
Statistics: {B['pairs_gap_n']['same']} / {B['pairs_gap_n']['adjacent']} simulated pairs (±5 % / ±12 %).</p>"""
    D.slide('expect', body, notes, short='Expected at n_TOF')


SCINT_SIGMA = 8.0   # mm; below the 12-17 mm (wall) / 30-60 mm (plastic) pointing resolution


def scint_kernel_png(layer, arm, vmax=1.0, sigma=SCINT_SIGMA, step=2.0, min_w=30.0):
    """The bench maps' sliding kernel on the scintillators: every late single track's crossing
    (ntof_scint_stack.ana.map_tracks, cached in <OUT>/cache) weighted by a Gaussian of sigma mm;
    efficiency = the net (accidental-subtracted) probe rate of ana.net with the tag's c, q.
    Pixels with < min_w effective tracks are transparent.  Returns (data URI, extent, stats)."""
    import base64
    cache = OUT / 'img' / f'scint_{layer}_{arm}_s{sigma:g}.png'
    meta = cache.with_suffix('.json')
    if not cache.exists() or not meta.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import fftconvolve
        from ntof_scint_stack import ana
        src = OUT / 'cache' / 'scint_map_tracks.parquet'
        if not src.exists():
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(4) as ex:
                pd.concat(list(ex.map(ana.map_tracks, 'ABCD')), ignore_index=True).to_parquet(src, index=False)
        T = pd.read_parquet(src)
        t = T[(T.layer == layer) & (T.arm == arm)]
        x0, x1, y0, y1 = -270.0, 270.0, -270.0, 270.0
        ex_ = np.arange(x0, x1 + step / 2, step)
        ey_ = np.arange(y0, y1 + step / 2, step)
        h = lambda w=None: np.histogram2d(t.u, t.v, bins=(ex_, ey_), weights=w)[0]  # noqa: E731
        k = int(np.ceil(4 * sigma / step))
        gx, gy = np.meshgrid(np.arange(-k, k + 1) * step, np.arange(-k, k + 1) * step, indexing='ij')
        g = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
        n = fftconvolve(h(), g, mode='same')
        kon = fftconvolve(h(t.on.astype(float)), g, mode='same')
        koff = fftconvolve(h(t.off.astype(float)), g, mode='same')
        with np.errstate(divide='ignore', invalid='ignore'):
            eff, _err = ana.net(kon, koff, n, float(t.c.iloc[0]), float(t.q.iloc[0]))
        eff = np.where(n >= min_w, eff, np.nan)
        rgba = plt.get_cmap('viridis')(np.clip(np.nan_to_num(eff.T[::-1], nan=0.0) / vmax, 0, 1))
        rgba[..., 3] = np.isfinite(eff.T[::-1])
        cache.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(cache, rgba)
        live = np.isfinite(eff)
        meta.write_text(json.dumps(dict(extent=[x0, x1, y0, y1], n=int(len(t)), sigma=sigma,
                                        p50=float(np.median(eff[live])), c=float(t.c.iloc[0]), q=float(t.q.iloc[0]))))
    m = json.loads(meta.read_text())
    return 'data:image/png;base64,' + base64.b64encode(cache.read_bytes()).decode(), m


def scint_kernel_panel(SD, layer, arm, g, vmax, w, h, ml, title):
    """heat_panel's frame and channel outlines (ntof_scint_stack.make_deck) around the kernel image."""
    rng = 360 if layer.startswith('liq') else 262
    P = sd.Plot(w, h, x=(-rng, rng), y=(-rng, rng), margin=(4, 6, 6, ml), title=title)
    uri, m = scint_kernel_png(layer, arm, vmax)
    x0, x1, y0, y1 = m['extent']
    tip = (f'arm {arm}, {layer.split("_")[0]}: {m["n"]:,} late single tracks, Gaussian kernel σ {m["sigma"]:g} mm; '
           f'median {100 * m["p50"]:.0f} %')
    P.raw(f'<image href="{uri}" x="{P.X(x0):.1f}" y="{P.Y(y1):.1f}" width="{P.X(x1) - P.X(x0):.1f}" '
          f'height="{P.Y(y0) - P.Y(y1):.1f}" preserveAspectRatio="none"{sd.tipattr(tip)}/>')
    SD.outline(P, layer, g, arm)
    return P.svg(f'{layer} kernel map, chamber {arm}')


def s_scint(D):
    """Wall, plastic and liquid efficiency maps per arm, drawn with ntof_scint_stack's own panels."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from ntof_scint_stack import make_deck as SD
    G = SD.geo()
    eff = rd(DATA / 'scint_stack/figures/eff_summary.csv')

    def unb(layer, probe, a):
        r = eff[(eff.layer == layer) & (eff.probe == probe) & (eff['sample'] == 'unbiased') & (eff.arm == a)]
        return float(r.eff.iloc[0]) if len(r) else float('nan')

    groups = []
    side = 236
    for layer, vmax, head in (('wall_full', 1.0, 'SiPM wall'), ('plas_full', 1.0, 'plastic'), ('liq_full', 0.2, 'liquid')):
        cells = [scint_kernel_panel(SD, layer, a, G.loc[a], vmax, side + 10, side + 30, 10, f'arm {a}')
                 for a in 'ABCD']
        grid = sd.col(sd.row(cells[0], cells[1], gap=4), sd.row(cells[2], cells[3], gap=4), gap=2)
        groups.append(sd.col(sd.p(f'<b>{head}</b>', 26), grid, gap=4))
    cb1 = sd.colorbar(96, 2 * side + 40, 0, 1, [(0, '0'), (0.5, '50'), (1, '100 %')], 'wall, plastic')
    cb2 = sd.colorbar(96, 2 * side + 40, 0, 0.2, [(0, '0'), (0.1, '10'), (0.2, '20 %')], 'liquid')
    wr = [unb('wall', 'wany', a) for a in 'ABCD']
    grp = {a: [float(eff[(eff.layer == 'wall') & (eff.probe == f'wany_g{k}') & (eff['sample'] == 'all_late')
                         & (eff.arm == a)].eff.iloc[0]) for k in range(4)] for a in 'ABCD'}
    gmin = {a: min(v) for a, v in grp.items()}
    grp_rng = lambda a: f'{100 * min(grp[a]):.0f}–{100 * max(grp[a]):.0f} %'  # noqa: E731
    body = (sd.title(f'Scintillators: walls {100 * min(wr):.0f}–{100 * max(wr):.0f} % and plastics ≥ 98 % efficient; '
                     f'the liquids answer mostly near their PMT',
                     'MM tracks extrapolated through the stack · > 10 ms, own-trigger biased · σ 8 mm kernel · seen from behind')
            + sd.row(groups[0], groups[1], cb1, groups[2], cb2, gap=14, align='end')
            + sd.p(f'<b>Not the MM efficiency projected back:</b> each map is P(layer fires | an MM track points there), '
                   f'so the chamber\'s own efficiency cancels; the wall\'s vertical bands are its four 100 mm SiPM read-out '
                   f'groups (dashed), which answer {grp_rng("A")} on A but down to {100 * gmin["D"]:.0f} % on D.', 21, INK))
    rows = ''.join(f'<tr><td>{a}</td><td>{sd.pct(unb("wall", "wany", a))}</td><td>{sd.pct(unb("plas", "pm", a))}</td></tr>'
                   for a in 'ABCD')
    notes = f"""
<p><b>Maps</b> (<code>ntof_scint_stack</code>, <code>ana/eff_map.parquet</code>, the <code>*_full</code> layers on
<code>all_late</code>, net of accidentals): P(the layer fired | an MM track points there), drawn with the bench maps'
sliding kernel: every late single track's crossing (<code>ana.map_tracks</code>) weighted by a Gaussian of σ
{SCINT_SIGMA:g} mm, transparent where fewer than 30 effective tracks; the pointing resolution itself is 12–17 mm at the
wall and 30–60 mm at the plastic, so the maps are no sharper than that. Every
trigger &gt; 10 ms after the flash. The arm's own trigger required wall ∧ plastic, so these levels are biased high
(the plastic most of all); they show <i>where</i> each counter works, not its unbiased efficiency. Panels drawn with
<code>ntof_scint_stack/make_deck.heat_panel</code> (channel edges, bars and PMTs as in the scint-stack note).</p>
<p><b>Unbiased efficiencies</b> (events another arm triggered, a few thousand tags per arm; B is rough):</p>
<table><tr><th>arm</th><th>SiPM wall</th><th>plastic | wall</th></tr>{rows}</table>
<p><b>The wall's structure.</b> Each map is conditional on an MM track, so the chamber's own efficiency divides out;
what is left is the wall. Its 16 vertical 25 mm bars are read out as four 100 mm groups (dashed, a SiPM at each end),
and the bands are those groups. Per group (late sample, own-trigger biased), group 0 → 3:
{'; '.join(f'{a} ' + ' / '.join(f'{100 * v:.0f}' for v in grp[a]) + ' %' for a in 'ABCD')}.
A is flat because all four of its groups answer; groups 0 and 2 are the weak ones on B, C and D. The fall at the
outer edges is the pointing resolution (tracks predicted on the wall that in fact missed it).</p>
<p>Plastic | wall is a response probability, not a hole: two in five particles that cross the 3 mm wall stop before
the plastic. Self-triggered, the plastic is ≥ 98 % efficient.</p>
<p><b>Liquids</b>: a muon gives 20–28 mV against a 16–18 mV readout threshold, so they are threshold-limited; A and D
(PMT at the side) answer 12–23× more on their PMT half, B and C (PMT on top) only 1.7–4.5×. They also suffered
pile-up (67–76 % of pulses had another pulse in their own tail). For the future (trigger_scint/TRIGGER_OPTIONS.md):
repairing the liquids is not worth it for acceptance (LS alone 1.2 % pair acceptance, LS OR plastic 4.3 %, vs ~14 % for
big plastics).</p>"""
    D.slide('scint', body, notes, short='Scintillators')


def capsule_outline(P, x0=0.0, color='#ffffff'):
    """The nose-first capsule (Al and CFRP outlines, side view) from capsule_pose's geometry."""
    from sept26_prelim_analysis import make_capsule_pose_deck as CPD
    CPD.capsule_side(P, x0, 0, color, gas=False, tip='capsule outline (Al, CFRP), nose-first, drawn at the frame centre')


def ratio_bars(w, h, vals, ymax, ylabel, title, fmt, yt):
    """Data vs Geant4 for arms A and C: grey = Geant4 through the reconstruction, colour = data."""
    P = sd.Plot(w, h, x=(0, 4), y=(0, ymax), ylabel=ylabel, title=title, margin=(16, 12, 52, 96))
    P.yticks(yt)
    P.xticks([(1, 'A'), (3, 'C')])
    for x0, a in ((1, 'A'), (3, 'C')):
        g4, dat, tg, td = vals[a]
        P.vbar(x0 - 0.38, g4, 62, GREY, tip=tg, label=fmt(g4))
        P.vbar(x0 + 0.38, dat, 62, CH[a], tip=td, label=fmt(dat))
    return P


def s_singles(D, AL):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    N, A = AL['N'], AL['tabs']
    meta = json.load(open(DATA / 'sept26_prelim/same_chamber/cmp.meta.json'))
    G = A['cmp_g4_classes']
    rate, pur = {}, {}
    for a in 'AC':
        acc = float(G[(G.arm == a) & (G.cls == '(singles in acceptance)')].per_trigger_acc.iloc[0])
        g4 = N['singles_eps_dig'][a] * acc
        da = meta['data_all'][a]['tracks'] / meta['data_all'][a]['triggers']
        rate[a] = (100 * g4, 100 * da, f'{a}, Geant4 through the digitised reconstruction: {100 * g4:.2f} tracks per 100 triggers',
                   f'{a}, data: {meta["data_all"][a]["tracks"]:,} tracks / {meta["data_all"][a]["triggers"]:,} triggers '
                   f'= {100 * da:.2f} per 100 (×{da / g4:.2f} Geant4)')
        pd_ = N[f'pur_late_{a}']
        pg = N['g4_pur'][a]
        pur[a] = (100 * pg, 100 * pd_, f'{a}, Geant4 through the same reconstruction: {100 * pg:.0f} % within 15 mm of the source',
                  f'{a}, data ≥ 20 ms: {100 * pd_:.0f} % within 15 mm of the source (minus a shuffled-angle null)')
    R = ratio_bars(470, 330, rate, 12, 'tracks per 100 triggers', 'single-track rate', lambda v: f'{v:.1f}',
                   [(v, str(v)) for v in (0, 4, 8, 12)])
    R.text(0.15, 11.3, f'data ×{N["singles_ratio"]["A"]:.2f} / ×{N["singles_ratio"]["C"]:.2f} Geant4', 22, INK, weight=700)
    Q = ratio_bars(470, 330, pur, 60, 'pointing at the source (%)', 'tracks that point back', lambda v: f'{v:.0f} %',
                   [(v, str(v)) for v in (0, 20, 40, 60)])
    img = rd(DATA / 'sept26_prelim/capsule_pose/image.csv')
    dat = rd(DATA / 'sept26_prelim/capsule_pose/data.csv')
    panels = []
    for i, a in enumerate('AC'):
        g = img[(img.arm == a) & (img.window == 'late')]
        vmax = float(g.excess.max())
        ml = 92 if i == 0 else 16
        P = sd.Plot(420 + ml, 680, x=(-60, 40), y=(-160, 120), xlabel='x at the beam-axis plane (mm)',
                    ylabel='height (mm)' if i == 0 else '', title=f'arm {a}, 20–75 ms', margin=(8, 12, 60, ml))
        P.xticks([(v, str(v)) for v in (-50, -25, 0, 25)])
        P.yticks([(v, str(v) if i == 0 else '') for v in (-150, -100, -50, 0, 50, 100)])
        P.cells([(r.x - 2, r.x + 2, r.h - 10, r.h + 10, max(float(r.excess), 0) / vmax,
                  f'arm {a}: x {r.x:.0f} mm, height {r.h:.0f} mm, pointing excess {r.excess / vmax:.2f} of max')
                 for r in g.itertuples() if -160 <= r.h <= 120], 0, 1)
        capsule_outline(P)
        d = dat[(dat.arm == a) & (dat.window == 'late')].iloc[0]
        P.points([d.x_cross], [d.y_cross], RED, r=8,
                 tips=[f'arm {a} band crossing: x {d.x_cross:.1f} ± {d.x_err:.2f}, height {d.y_cross:.1f} ± {d.y_err:.2f} mm '
                       f'({int(d.n):,} tracks, 20–75 ms)'])
        if i == 0:
            P.text(-56, 100, 'one source,', 24, '#ffffff', weight=700)
            P.text(-56, 86, 'on the nose', 24, '#ffffff', weight=700)
        panels.append(P.svg(f'capsule image {a}'))
    body = (sd.title('Single tracks match Geant4 and image the capsule nose',
                     'n_TOF A and C, run_69 onward · grey: Geant4 through the same reconstruction · right: late tracks at the beam-axis plane')
            + sd.row(sd.col(sd.legend([('Geant4 × reco', GREY, 'box'), ('data A', BLUE, 'box'),
                                       ('data C', ORANGE, 'box')], 20),
                            R.svg('single rate'), Q.svg('pointing'), gap=10),
                     sd.row(*panels, gap=12),
                     sd.col(sd.colorbar(110, 400, 0, 1, [(0, '0'), (0.5, '½'), (1, 'max')], 'excess'),
                            sd.legend([('band crossing', RED, 'dot'), ('capsule (Al, CFRP)', '#9aa3b2', 'line')], 20), gap=20),
                     gap=36, align='end'))
    notes = f"""
<p><b>Rate</b> (left, top): data tracks per trigger over the whole run (A {meta['data_all']['A']['tracks']:,},
C {meta['data_all']['C']['tracks']:,} tracks in {meta['data_all']['A']['triggers']:,} triggers) against Geant4
track-like singles in acceptance per trigger × the efficiency of the digitised reconstruction (Geant4 events overlaid
on real quiet triggers and run through the production reconstruction: {N['singles_eps_dig']['A']:.3f} A,
{N['singles_eps_dig']['C']:.3f} C). Data/Geant4 = ×{N['singles_ratio']['A']:.2f} (A), ×{N['singles_ratio']['C']:.2f} (C).</p>
<p><b>Pointing</b> (left, bottom): tracks whose line passes within 15 mm of the measured source, minus a shuffled-angle
null, ≥ 20 ms (<code>capsule_tof.py</code>). The data are at least as capsule-pure as Geant4.</p>
<p><b>Source image</b> (<code>capsule_pose</code>): each late track's line, the density of where lines cross the beam
axis plane minus a shuffled null, 4 mm × 20 mm cells, normalised per panel; red dot = the band crossing (pointing
estimator, robust to each chamber's efficiency shape). A and C agree: x ≈ −10 mm, height −31 (A) / −35 (C) mm,
where Geant4's capsule puts the Al captures (−28 by the same estimator). The capsule outline is drawn at the frame
centre (x = 0): the ~10 mm x offset (beam off the frame, or capsule off the beam) is a stated limitation, since no
survey exists. The two chambers' u axes point opposite ways, so only a common shift of the pair against the beam
could fake the image. The same spot appears at 4–12, 12–20 and 20–75 ms and before and after the 27 July access.
The late tracks are delayed (hall die-away, 23.5 ms) neutrons captured on the capsule Al.</p>
<p><b>Angle scale on cosmic muons</b> (<code>ntof_cosmics</code>): 41 beam-off runs at the production operating point
(57.6 h, 5.2 M triggers) put on the n_TOF clock to ~10 ns (<code>clock_match.py</code>). The straight line joining
chambers A and C (470 mm apart) is the angle truth for each chamber: the in-situ calibration (drift velocity, kernel)
reproduces it to 1–4 %, in beam-off and in-beam muons alike.</p>
<p>Why this is the best in-situ demonstration: the image needs correct angles and positions in two independent
chambers at once, and it reproduces Geant4's source position and rate.</p>"""
    D.slide('singles', body, notes, short='Singles')


def singles_height(edges):
    """Single tracks per trigger against the height where the track line crosses the beam-axis plane
    (its pointing origin along the beam, y up): data (all is2_v2 cf tracks, capsule_pose.load_data) and
    Geant4 nose-first tracks through g4_digi (capsule_pose.library).  Shapes only; cached."""
    cache = OUT / 'cache' / 'singles_height.csv'
    if cache.exists():
        return rd(cache)
    from sept26_prelim_analysis import capsule_pose as CP
    d = CP.load_data()
    rows = []
    for a in 'AC':
        L = CP.library(a)
        F = CP._frame(a, L.ur, L.vr, L.tur, L.tvr)
        g = d[d.arm == a]
        hd, _ = np.histogram(g.hgt, edges)
        hg, _ = np.histogram(F.hgt, edges)
        hl, _ = np.histogram(g[g.ms >= 12].hgt, edges)
        for k in range(len(edges) - 1):
            rows.append(dict(arm=a, lo=edges[k], hi=edges[k + 1], data=int(hd[k]), data_late=int(hl[k]), g4=int(hg[k]),
                             n_data=len(g), n_g4=len(F)))
    R = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(cache, index=False)
    return R


def s_pairs(D, AL):
    N, A = AL['N'], AL['tabs']
    meta = json.load(open(DATA / 'sept26_prelim/same_chamber/cmp.meta.json'))
    G = A['cmp_g4_classes']
    edges = np.arange(-200.0, 161.0, 10.0)
    H = singles_height(edges)
    P = sd.Plot(800, 600, x=(-200, 160), y=(0, 1.8), xlabel='height where the track points at the beam axis (mm)',
                ylabel='single tracks per 100 triggers per 10 mm', title='singles: where they come from',
                margin=(16, 20, 92, 104))
    P.xticks([(v, str(v)) for v in (-200, -150, -100, -50, 0, 50, 100, 150)])
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.4, 0.8, 1.2, 1.6)])
    for a in 'AC':
        h = H[H.arm == a]
        acc = float(G[(G.arm == a) & (G.cls == '(singles in acceptance)')].per_trigger_acc.iloc[0])
        g4_rate = 100 * N['singles_eps_dig'][a] * acc
        d_rate = 100 * meta['data_all'][a]['tracks'] / meta['data_all'][a]['triggers']
        c = list(0.5 * (h.lo + h.hi))
        g4 = list(g4_rate * h.g4 / h.g4.sum())
        dd = list(d_rate * h.data / h.data.sum())
        xs, ys = sd.step_xy(list(h.lo) + [float(h.hi.iloc[-1])], g4)
        P.raw(sd.poly([P.X(x) for x in xs], [P.Y(y) for y in ys], CH[a], 3, '9 6',
                      tip=f'{a}, Geant4 through the reconstruction ({int(h.n_g4.iloc[0]):,} tracks), scaled to '
                          f'{g4_rate:.2f} tracks per 100 triggers'))
        P.points(c, dd, CH[a], r=5, tips=[f'{a} data {lo:.0f} to {hi:.0f} mm: {v:.3f} per 100 triggers ({n:,} tracks)'
                                          for lo, hi, v, n in zip(h.lo, h.hi, dd, h.data)])
    P.vline(-31, MUT, '4 6', 1.5, label='capsule source', tip='band crossing: −31 (A) / −35 (C) mm; Geant4 −28')
    P.text(30, 1.5, f'data ×{N["singles_ratio"]["A"]:.2f} / ×{N["singles_ratio"]["C"]:.2f} Geant4', 26, INK, weight=700)
    P.text(30, 1.38, 'both peak at the capsule', 22, INK)

    T = A['cmp_tof_fraction']
    T = T[T.lo >= 12]
    E = A['cmp_expectation']
    dig = E[E.eps_source.str.startswith('digitised')].set_index('arm')
    Q = sd.Plot(800, 600, x=(12, 76), y=(0, 1.6), xlabel='time since the γ flash (ms)',
                ylabel='late events with a 2nd same-chamber track (%)', title='pairs: how many', margin=(16, 20, 92, 104))
    Q.xticks([(v, str(v)) for v in (20, 30, 40, 50, 60, 70)])
    Q.yticks([(v, f'{v:.1f}') for v in (0, 0.4, 0.8, 1.2, 1.6)])
    for a in 'AC':
        g = T[T.arm == a]
        pr = 100 * float(dig.loc[a, 'predicted_frac'])
        Q.hline(pr, CH[a], dash='9 6', tip=f'{a}: Geant4 through the same reconstruction predicts {pr:.2f} % '
                                           '(opening angle and close-pair loss included)')
        Q.points(list(0.5 * (g.lo + g.hi)), list(100 * g.frac), CH[a], r=6,
                 tips=[f'{a} {lo:g}-{hi:g} ms: {100 * f:.3f} ± {100 * e:.3f} % ({n2:,} pairs in {n1:,} events), '
                       'accidentals included' for lo, hi, f, e, n1, n2 in zip(g.lo, g.hi, g.frac, g.err, g.n1, g.n2)])
        aa, ae = N['a'][a]
        Q.hline(100 * aa, CH[a], dash='2 5', tip=f'{a}: rate-independent (correlated) part a = {100 * aa:.3f} ± {100 * ae:.3f} %')
    Q.text(24, 1.42, 'Geant4 (dashed): 1.2 %', 22, INK)
    Q.text(24, 0.62, f'data: ⅛ of it ({N["d_over_g4"]["A"]:.2f} / {N["d_over_g4"]["C"]:.2f})', 26, RED, weight=700)
    Q.text(24, 0.5, 'at every time of flight', 22, RED)
    body = (sd.title('Singles match Geant4; same-chamber pairs come in at ⅛',
                     'n_TOF A and C, is2_v2 re-pass, run_69 onward · points: data · dashed: Geant4 pushed through the same reconstruction')
            + sd.row(P.svg('singles vs height'), Q.svg('pairs vs time'), gap=44)
            + sd.legend([('chamber A', BLUE, 'dot'), ('chamber C', ORANGE, 'dot'), ('Geant4 × reconstruction', INK, 'dash'),
                         ('correlated part (data)', MUT, 'dash')], 21))
    notes = f"""
<p><b>Left.</b> Every data track's line extended to the plane through the beam axis (<code>capsule_pose.load_data</code>:
height = y at that plane, y up), all is2_v2 cf tracks, normalised to the measured tracks per trigger. Geant4: the
nose-first neutron campaign digitised into real quiet triggers and reconstructed (<code>capsule_pose.library</code>,
g4_digi), shape normalised to Geant4's own singles per trigger × the digitised efficiency. The data carry an extra
tails and a wider core than Geant4 (68 % half-width ~33 vs ~25 mm on the late sample: beam-time background before
12 ms, and real-detector resolution); the peak sits where the capsule-image band crossing puts the source.</p>
<p><b>Right.</b> Share of tracked events with a second same-chamber track, per ToF bin ≥ 12 ms (points, accidentals
included; dotted: the rate-independent part a from f(t) = a + b·rate(t), the correlated pairs). Dashed: the Geant4
prediction, the ratio of two reconstructable leptons to one × the measured P(2 tracks | 2 leptons) of the digitised
reconstruction (P(2|2) 0.16 / 0.14), so the opening angle and the close-pair loss are already in it. Correlated pairs:
~{N['corr_pairs']['A']:,.0f} (A), ~{N['corr_pairs']['C']:,.0f} (C). Other efficiency routes bracket data/Geant4 at
{N['d_over_g4_bracket'][0]:.2f}–{N['d_over_g4_bracket'][1]:.2f}.</p>
<p><b>Why pairs are not drawn against their pointing origin.</b> The late pair table keeps each leg's position but not its
direction, and the pair vertex of two near-parallel ~2 MeV legs (each bent ~9°) lands ~75 mm from where they were born:
even perfect tracks pass a 20 mm vertex test 2 % of the time (chance 3 %). A per-leg pointing comparison needs the legs
re-joined to their stage-3 lines.</p>
<p><b>Where the deficit could be.</b> In the detector (two-track loss in busy events): it would hit X17 far less, since
X17 legs land in different chambers 470 mm apart, penetrate, fire their own trigger (45 % vs 7.7 %) and vertex to
~10 mm. In the simulation: the capsule model (nose Al, wrap), the delayed-neutron source population, or the
reconstruction version the prediction used (one-sided search). Next with the data we have: re-digitise Geant4 pairs
with is2_v2; Geant4 with delayed thermal neutrons; check the capsule model; look for a two-lepton trigger effect.</p>
<p>Opening angle (A): the correlated excess prefers the Geant4 e⁺e⁻ shape (χ² 6.6/6 vs 26/6 for unrelated pairs).
Source: x17-status-2026-10-15; <code>same_chamber_compare.py</code>, <code>cmp_*.csv</code>.</p>"""
    D.slide('pairs', body, notes, short='Singles vs pairs')


def passport():
    """ntof_calorimetry/make_passport_deck: the track-passport tables (passport/, mm_edep/results/)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from ntof_calorimetry import make_passport_deck as MP
    return MP, MP.load()


def spec_panel(g, lo, hi, w, h, xlabel, title, col, ymax=None, xt=None, ylabel=True, last=False, g4lab='Geant4'):
    """Data points against the Geant4 histogram (no refit), each normalised over [lo, hi]."""
    c = 0.5 * (g.lo + g.hi)
    m = (c >= lo) & (c <= hi)
    dn = g.data / g.data[m].sum()
    sn = g.g4 / g.g4[m].sum()
    ymax = ymax or float(max(dn[m].max(), sn[m].max()) * 1.25)
    P = sd.Plot(w, h, x=(lo, hi), y=(0, ymax), xlabel=xlabel if last else '', ylabel='fraction per bin' if ylabel else '',
                title=title, margin=(10, 14, 62 if last else 30, 92 if ylabel else 30))
    P.xticks(xt)
    P.yticks([(v, f'{v:.2f}' if ylabel else '') for v in np.linspace(0, ymax, 4)[:-1]])
    sel = (g.lo >= lo) & (g.hi <= hi)
    gi = g[sel]
    xs, ys = sd.step_xy(list(gi.lo) + [float(gi.hi.iloc[-1])], list(sn[sel]))
    P.raw(sd.poly([P.X(x) for x in xs], [P.Y(min(y, ymax)) for y in ys], G4C, 3,
                  tip=f'{g4lab}, {int(g.g4.sum()):,} events, scale 1'))
    mm = sel & (g.data > 0)
    P.points(list(c[mm]), list(dn[mm]), col, r=4.5,
             tips=[f'{lo_:.2f}-{hi_:.2f}: data {d:,}, {g4lab} {s_:,}' for lo_, hi_, d, s_ in
                   zip(g.lo[mm], g.hi[mm], g.data[mm], g.g4[mm])])
    return P, ymax


def s_energy(D):
    Sp = rd(DATA / 'calorimetry/wp/spectra.csv')
    fits = rd(DATA / 'calorimetry/wp/scale_fits.csv')
    MP, t = passport()
    bc = t['bc']
    rows = []
    for i, a in enumerate('AC'):
        last = i == 1
        sc_p = float(fits[(fits.arm == a) & (fits['var'] == 'e_plas') & fits.subset.isna()].scale.iloc[0])
        sc_w = float(fits[(fits.arm == a) & (fits['var'] == 'e_wall_fit') & fits.subset.isna()].scale.iloc[0])
        g = Sp[(Sp.arm == a) & (Sp['var'] == 'e_plas')]
        P, ym = spec_panel(g, 2, 10, 560, 290 + (32 if last else 0), 'plastic energy (MeV)', f'arm {a} · plastic', CH[a],
                           xt=[(v, str(v)) for v in range(2, 11, 2)], last=last)
        P.text(6.2, ym * 0.85, f'data ÷ Geant4 = {sc_p:.2f}', 22, INK, weight=700)
        g = Sp[(Sp.arm == a) & (Sp['var'] == 'e_wall_fit')]
        W, ym = spec_panel(g, 0.25, 1.5, 520, 290 + (32 if last else 0), 'SiPM wall energy (MeV)', f'arm {a} · SiPM wall', CH[a],
                           xt=[(v, f'{v:g}') for v in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)], ylabel=False, last=last)
        W.text(0.8, ym * 0.85, f'data ÷ Geant4 = {sc_w:.2f}', 22, INK, weight=700)
        b = bc[a]
        e = np.asarray(b['edges'])
        ymm = float(max(b['data'].max(), b['sim'].max()) * 1.25)
        M = sd.Plot(520, 290 + (32 if last else 0), x=(1.5, 40, 'log'), y=(0, ymm),
                    xlabel='MM gas deposit, keV per 30 mm at normal incidence' if last else '', title=f'arm {a} · Micromegas dE/dx',
                    margin=(10, 14, 62 if last else 30, 30))
        M.xticks([(v, f'{v:g}') for v in (2, 5, 10, 20, 40)])
        M.yticks([(v, '') for v in np.linspace(0, ymm, 4)[:-1]])
        xs, ys = sd.step_xy(list(e), list(b['sim']))
        M.raw(sd.poly([M.X(x) for x in xs], [M.Y(min(y, ymm)) for y in ys], G4C, 3,
                      tip=f'beam simulation (Heed through g4_digi and the reconstruction), {b["n_sim"]:,} tracks, median {b["med_sim"]:.2f} keV'))
        cx = list(np.sqrt(e[1:] * e[:-1]))
        M.points(cx, list(b['data']), CH[a], r=4.5,
                 tips=[f'{lo:.1f}-{hi:.1f} keV: {100 * v:.1f} % of {b["n_data"]:,} data tracks' for lo, hi, v in zip(e[:-1], e[1:], b['data'])])
        M.text(9, ymm * 0.85, f'median {b["med_data"]:.1f} keV = a MIP', 22, INK, weight=700)
        rows.append(sd.row(P.svg(f'plastic {a}'), W.svg(f'wall {a}'), M.svg(f'mm {a}'), gap=20))
    en = lambda f, s_: float(t['EN'][(t['EN'].features == f) & (t['EN'].subset == s_)].w68.iloc[0])  # noqa: E731
    auc = float(t['TPt'][t['TPt'].source == 'Geant4'].auc_multi_mm.mean())
    body = (sd.title('Every layer sits on Geant4\'s energy scale',
                     'self-triggered single tracks at n_TOF (run_145 for the MM) · points: data · line: Geant4 with the measured '
                     'trigger and resolution, not refitted')
            + sd.col(*rows, gap=4))
    notes = f"""
<p><b>Scintillators</b> (<code>ntof_calorimetry/wall_plastic.py</code>, note scint-calorimetry): the SiPM wall and each
plastic bar are calibrated on cosmic muons only (beam-off runs on the n_TOF clock, path from the MM track), then
Geant4 of the full stack predicts the beam-electron spectra (total prompt deposit in the matched bar, smeared with the
measured σ/MPV, the arm's trigger applied in MeV). Drawn at scale 1; the numbers on the plots are the data/Geant4
scales over the whole spectrum. Per bar (≥ 40 ms): plastic A1/A2/C2 1.05/1.03/1.02, C1 1.15 (a pure gain offset of
unknown cause); wall A/C 0.95/0.96. Linear from the muon's 3.4 MeV to stopping electrons at 3–8 MeV; stable to
0.5–0.8 % run to run. A 25 mm bar (the run-config text) would put every scale at ×1.27, so the 20 mm in Geant4 is right.
Plastic PMTs lose 3–8 % gain 12–25 ms after the flash.</p>
<p><b>Micromegas dE/dx</b> (<code>mm_edep_*</code>, the track passport): the charge along each track, corrected for
the readout window and for an in-situ gain map, in keV per 30 mm at normal incidence. The data median (5.6 keV)
matches Heed's MIP; per-track resolution 16 % (A) / 22 % (C) in beam. The simulation is Heed ionisation through the
full digitisation and reconstruction.</p>
<p><b>What the MM adds</b> (Geant4-trained estimators, cross-validated): nothing for fast tracks (the gas is
minimum-ionising), but for electrons below 3 MeV the energy width goes {en('plastic+wall+angle', 'T < 3 MeV'):.2f} →
{en('plastic+wall+angle+MM', 'T < 3 MeV'):.2f} MeV with the MM; for tracks that never reach the plastic it halves the
vertex-energy width of 1–3.5 MeV capsule electrons (1.7 → 0.45–0.8 MeV). It also tags tracks with several particles in
the gap (AUC {auc:.2f}; the wall cannot: 0.46–0.48), which is what a two-track candidate needs. The track angle buys
back the energy of steep tracks ({en('plastic', 'sec >= 1.1'):.1f} → {en('plastic+wall+angle', 'sec >= 1.1'):.1f} MeV).
Event displays of a single-lepton and a several-particle track are in the backup.</p>
<p><b>For the future.</b> At n_TOF the 2 cm plastic stops only the soft leg (a range telescope). With 4–5 cm plates the
full stack is a calorimeter: the LNL E_sum and pair-mass fits rest on this n_TOF-anchored response (as built it loses
the mass fit, backup). Resolution of a thick plate: to be measured on a prototype.</p>"""
    D.slide('energy', body, notes, short='Energy')


def b_passport(D):
    MP, t = passport()
    EX = t['EX']
    first = {}
    for _, r in EX.iterrows():
        first.setdefault(r.example, r)
    cs = [MP._card(t, first[w]) for w in ('through the plastic', 'several particles') if w in first]
    body = (sd.title('One lepton or several: what the stack records for each track',
                     'real run_145 tracks, side view to scale; blue dots: the drift charge along the track; energies on the Geant4 scale')
            + sd.row(*cs, gap=24))
    notes = """
<p>The track passport (<code>ntof_calorimetry/track_passport.py</code>, note track-passport). Left: a single lepton
that goes through the plastic (the liquid fires). Right: a track with a high gas deposit that the Geant4-trained
estimator tags as several particles in the gap, the signature a two-track candidate leaves when the tracks overlap.
Energies and P(several) come from Geant4-trained estimators (model-dependent); the counters carry their measured
resolutions (wall ~16–18 %, plastic ~4–7 % at the MIP), the gas deposit its per-track resolution.</p>"""
    D.slide('b-passport', body, notes, short='B1 Passports', pad='72px 64px 72px')


def al_numbers():
    """Singles and same-chamber pairs at n_TOF, A and C, tracks >= 12 ms, is2_v2 re-pass, with the
    Geant4-through-the-reconstruction expectation (sept26_prelim_analysis: same_chamber_compare via the
    15 Oct status deck's numbers())."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from sept26_prelim_analysis import make_status_oct15_deck as OC
    A = OC.load()
    N = OC.numbers(A)
    meta = json.load(open(DATA / 'sept26_prelim/same_chamber/cmp.meta.json'))
    T = A['cmp_tof_fit'].set_index('arm')
    E = A['cmp_expectation']
    dig = E[E.eps_source.str.startswith('digitised')].set_index('arm')
    out = dict(tabs=A, N=N, OC=OC)
    for a in 'AC':
        tr = meta['data_late'][a]['tracks']
        ev = float(T.loc[a, 'events'])
        out[a] = dict(singles=tr, singles_g4=tr / N['singles_ratio'][a], pairs=float(T.loc[a, 'correlated_pairs']),
                      pairs_g4=float(dig.loc[a, 'predicted_frac']) * ev, raw_pairs=int(T.loc[a, 'pairs']), events=ev,
                      ratio_s=N['singles_ratio'][a], ratio_p=N['d_over_g4'][a])
    for k in ('singles', 'singles_g4', 'pairs', 'pairs_g4'):
        out[k] = out['A'][k] + out['C'][k]
    return out


#: Geant4 sim arm -> n_TOF chamber (A +z, C -z, D +x, B -x: ntof_athens_26 y_image W_X/W_Z)
SIM_ARM = {2: 'A', 3: 'C', 0: 'D', 1: 'B'}


def g4_breakdown(AL):
    """Per chamber and per pair topology, from the Geant4 thermal campaign's per-lepton truth
    (same_chamber_g4.load over MX17_Full_Geant same_arm_pairs.py parts): track-like leptons
    (>= 20 mm of gap) entering the gap with |tan| < 0.6 in a triggered event, and e+e- of one
    conversion by where the two legs land.  Counts per trigger scaled to the late sample:
    n_late = the trigger count that turns Geant4 x the digitised efficiency into the slide's
    expected singles (mean of A and C).  Cached."""
    cache = OUT / 'cache' / 'g4_breakdown.json'
    if cache.exists():
        return json.loads(cache.read_text())
    from sept26_prelim_analysis import same_chamber_g4 as SG
    L, C = SG.load(Path.home() / 'x17_local' / 'g4_same_arm' / 'parts')
    n_trig = C['trig.events']
    T = L[(L.w_max - L.w_min) >= SG.TRACK_W_MM].copy()
    T['acc'] = False
    for arm in SIM_ARM:
        R, _t, _r = SG.frame(L, arm)
        m = (T.arm == arm).to_numpy()
        d = np.c_[T.px0[m], T.py0[m], T.pz0[m]] @ R.T
        T.loc[m, 'acc'] = ((np.abs(d[:, 0] / d[:, 2]) < SG.TAN_MAX) & (np.abs(d[:, 1] / d[:, 2]) < SG.TAN_MAX)
                           & (d[:, 2] > 0))
    T['trig'] = T.trig_arms != 0
    T['wall'] = T.wall_eV >= SG.WALL_HALF_MIP_EV
    J = json.load(open(Path.home() / 'x17_local' / 'g4_same_arm' / 'out' / 'g4_same_arm.json'))
    n_late = float(np.mean([AL[a]['singles_g4'] / (J['summary'][a]['singles_per_trig'] * AL['N']['singles_eps_dig'][a])
                            for a in 'AC']))
    s = T[T.trig & T.acc]
    out = dict(n_trig_g4=n_trig, n_late=n_late,
               singles={SIM_ARM[a]: n_late * int((s.arm == a).sum()) / n_trig for a in SIM_ARM},
               singles_reco={a: AL[a]['singles_g4'] for a in 'AC'})
    rows = []
    for (_f, _e, _p), g in T[T.op == 'conv'].groupby(['file', 'event', 'parent']):
        if len(g) != 2 or set(g.particle) != {'e+', 'e-'}:
            continue
        a1, a2 = (int(v) for v in g.arm)
        kind = 'same' if a1 == a2 else ('opposite' if {a1, a2} in ({0, 1}, {2, 3}) else 'adjacent')
        rows.append(dict(kind=kind, trig=bool(g.trig.iloc[0]), acc=bool(g.acc.all()), wall=bool(g.wall.all())))
    P = pd.DataFrame(rows)
    q = P[P.trig & P.acc]
    for lab, sel in (('gap', q), ('wall', q[q.wall])):
        vc = sel.kind.value_counts()
        out[f'pairs_{lab}'] = {k: n_late * int(vc.get(k, 0)) / n_trig for k in ('same', 'adjacent', 'opposite')}
        out[f'pairs_{lab}_n'] = {k: int(vc.get(k, 0)) for k in ('same', 'adjacent', 'opposite')}
    out['opp_any_gap_n'] = int(((P.kind == 'opposite') & P.trig).sum())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def lnl_numbers():
    J = json.load(open(FAC / 'lnl/out/talk/numbers.json'))
    c = J['cases']
    return dict(J=J, ab1=c['as built|1'], ab4=c['as built|4'], bp1=c['big plastics|1'], bp4=c['big plastics|4'])


def lnl_setup_svg(w=500, h=400):
    """The LNL set-up seen along the beam: the four MX17 arms around a thin Li film in a small CFRP chamber, with an
    X17 pair (after x17_facility_search/lnl/deck/build_deck.setup_svg, big plastics drawn)."""
    W, H = 860, 690
    cx, cy = 430, 330
    o = []
    face, half = 150, 140
    for ang in (0, 90, 180, 270):
        a = math.radians(ang)
        ux, uy = math.cos(a), math.sin(a)
        vx, vy = -uy, ux

        def pt(r, s_, ux=ux, uy=uy, vx=vx, vy=vy):
            return cx + ux * r + vx * s_, cy + uy * r + vy * s_
        for r0, r1, hh, col, tp in ((face, face + 22, half, BLUE, 'Micromegas TPC, 30 mm drift, faces at ±204 mm'),
                                    (face + 30, face + 40, half, GOLD, 'SiPM wall, 20 bars, all read out'),
                                    (face + 50, face + 80, 1.35 * half, PURPLE, 'big plastic, 75×75×5 cm (the upgrade)')):
            q = [pt(r0, -hh), pt(r1, -hh), pt(r1, hh), pt(r0, hh)]
            o.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in q)}" fill="{col}" fill-opacity="0.75"'
                     f'{sd.tipattr(tp)}/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="40" fill="none" stroke="{MUT}" stroke-width="3"'
             f'{sd.tipattr("thin target chamber: CFRP 0.4 mm, r ≈ 25 mm")}/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{RED}"{sd.tipattr("Li₂O film ~300 µg/cm², proton beam into the page")}/>')
    for ang, lab in ((-160, 'e⁺'), (-20, 'e⁻')):
        a = math.radians(ang)
        x2, y2 = cx + 300 * math.cos(a), cy + 300 * math.sin(a)
        o.append(sd.arrow(cx, cy, x2, y2, INK, 4))
        o.append(sd.T(x2 + (20 if x2 > cx else -20), y2 - 8, lab, 32, INK, weight=600))
    o.append(sd.T(cx, cy - 70, 'X17 → e⁺e⁻, ≥ 134°', 30, INK, weight=600))
    o.append(sd.T(cx, H - 8, 'p beam into the page · ⁷Li(p,e⁺e⁻)⁸Be at 18.15 MeV', 28, MUT))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="LNL set-up" style="overflow:visible">{"".join(o)}</svg>')


def s_lnl(D, L):
    J = L['J']
    MS = _lnl_maps()
    O = FAC / 'lnl/out'
    # statistics: X17 over a 30-day run, linear
    d30 = 30
    bars = [('n_TOF', 2.0, GREY, 'X17 made over the whole n_TOF thermal window (Athens rate tables): ≈ 2, before any acceptance'),
            ('LNL as built', L['ab1']['x17_per_day'] * d30, ORANGE,
             f'LNL, n_TOF stack as built, 1 µA: {L["ab1"]["x17_per_day"]:.0f} X17/day in the 3D fit × {d30} days'),
            ('LNL big plastics', L['bp1']['x17_per_day'] * d30, PURPLE,
             f'LNL, four 75×75×5 cm plastics, 1 µA: {L["bp1"]["x17_per_day"]:.0f} X17/day in the 3D fit × {d30} days')]
    ymax = 1.15 * max(b[1] for b in bars)
    S = sd.Plot(500, 290, x=(0, 3), y=(0, ymax), ylabel='X17 at the ATOMKI ratio', title=f'X17 in hand after {d30} days',
                margin=(16, 12, 52, 96))
    S.yticks([(v, f'{v:,.0f}') for v in (0, 1000, 2000, 3000, 4000)])
    S.xticks([(i + 0.5, t_) for i, t_ in enumerate(('n_TOF', 'LNL', 'LNL + plates'))])
    for i, (lab, v, c, tip) in enumerate(bars):
        S.vbar(i + 0.5, v, 90, c, tip=tip, label=f'{v:,.0f}' if v > 10 else f'≈ {v:.0f}')
    # opening-angle expectation, big plastics, one week
    pr = MS['proj']
    days = 7
    th = pr[(pr.hw == 'big plastics') & (pr['var'] == 'theta')]
    b = th[th.comp == 'bkg'].sort_values('lo')
    x = th[th.comp == 'X17'].sort_values('lo')
    edges = list(b.lo) + [float(b.hi.iloc[-1])]
    tot = (b.per_day.to_numpy() + x.per_day.to_numpy()) * days
    bk = b.per_day.to_numpy() * days
    ym = float(tot.max() * 1.15)
    A = sd.Plot(560, 610, x=(90, 180), y=(0, ym), xlabel='opening angle θ (°)', ylabel=f'pairs per 2°, {days} days',
                title='the opening angle, big plastics', margin=(16, 16, 92, 104))
    A.xticks([(v, f'{v}°') for v in (90, 110, 130, 150, 170)])
    st = 100 if ym < 600 else 200
    A.yticks([(v, f'{v:.0f}') for v in np.arange(0, ym, st)])
    xs, ys = sd.step_xy(edges, list(bk))
    A.raw(sd.poly([A.X(a) for a in [xs[0]] + xs + [xs[-1]]], [A.Y(0)] + [A.Y(y) for y in ys] + [A.Y(0)], GREY, 0, fill=GREY,
                  tip=f'background (IPC M1 + E1, γ₁, cosmics), {bk.sum():,.0f} pairs in {days} days'), back=True)
    xs, ys = sd.step_xy(edges, list(tot))
    A.raw(sd.poly([A.X(a) for a in xs], [A.Y(y) for y in ys], RED, 3,
                  tip=f'background + X17 at R(ATOMKI): {x.per_day.sum() * days:,.0f} X17 in {days} days'))
    A.text(122, ym * 0.93, 'X17: a few % on top of IPC', 24, RED, weight=700)
    A.text(122, ym * 0.87, 'so the IPC shape must be', 22, INK)
    A.text(122, ym * 0.81, 'known and subtracted', 22, INK)
    # 2D: S/B in theta x |y|
    m = MS['maps']
    g = m[(m.hw == 'big plastics') & (m.grid == 'fine') & (m.slice == 'window')]
    csb = [(r.theta_lo, r.theta_hi, r.y_lo, r.y_hi, r.X17 / r.bkg if r.bkg * days >= 3 else float('nan'),
            f'{r.theta_lo:.0f}–{r.theta_hi:.0f}°, |y| {r.y_lo:.2f}–{r.y_hi:.2f}: S/B {100 * r.X17 / r.bkg:.0f} % '
            f'({r.X17 * days:.1f} X17, {r.bkg * days:.1f} background in {days} d)') for r in g.itertuples() if r.bkg > 0]
    M = MS['mod'].heat('signal / background with the energy split', csb, 0, 0.4, MS['mod'].CM_SB, w=560, h=610)
    M.text(95, 0.93, 'X17 = a line at fixed mass;', 22, INK, weight=700)
    M.text(95, 0.86, 'a background error = a stripe', 22, INK, weight=700)
    cb = sd.colorbar(100, 400, 0, 0.4, [(0, '0'), (0.2, '20 %'), (0.4, '≥ 40 %')], 'S/B', cmap=MS['mod'].CM_SB)
    body = (sd.title('LNL: enough X17 in days; the IPC shape is the work',
                     '⁷Li(p,e⁺e⁻)⁸Be at the 18.15 MeV resonance, 1 µA · Geant4 of the n_TOF apparatus, MM 15° + TOF, θ × |y| × E_sum fit')
            + sd.row(sd.col(lnl_setup_svg(440, 300),
                            sd.legend([('MM TPC', BLUE, 'box'), ('SiPM wall', GOLD, 'box'), ('big plastic', PURPLE, 'box')], 18),
                            S.svg('statistics'), gap=4),
                     sd.col(sd.legend([('background (IPC, γ₁, cosmics)', GREY, 'box'), ('+ X17', RED, 'line')], 18),
                            A.svg('opening angle'), gap=4),
                     sd.row(M.svg('S/B 2D'), cb, gap=0, align='center'), gap=20, align='end'))
    rows = ''.join(f'<tr><td>{k}</td><td>{L[a]["days_3sigma"]:.1f}</td><td>{L[a]["days_5sigma"]:.1f}</td>'
                   f'<td>{L[a]["x17_per_day"]:.0f}</td><td>{L[a]["bkg_per_day"]:,.0f}</td><td>{sd.pct(L[a]["live"])}</td></tr>'
                   for k, a in (('as built, 1 µA', 'ab1'), ('as built, 4 µA', 'ab4'), ('big plastics, 1 µA', 'bp1'),
                                ('big plastics, 4 µA', 'bp4')))
    notes = f"""
<p><b>Reach</b> (<code>x17_facility_search/lnl/</code>, FEASIBILITY_SIM.md §12–18, <code>lnl/out/talk/numbers.json</code>):
days to 3σ at R(ATOMKI) = 5.8×10⁻⁶ with the θ × |y| × E_sum fit. Pair trigger (Geant4): as built
{sd.pct(J['pair_tag']['as built'], 1)}, big plastics {sd.pct(J['pair_tag']['big plastics'], 1)}. ~900 X17 made per day at 1 µA.</p>
<table><tr><th></th><th>days to 3σ</th><th>to 5σ</th><th>X17/day in fit</th><th>background/day</th><th>DREAM live</th></tr>{rows}</table>
<p><b>Left, bottom.</b> X17 at the ATOMKI ratio after 30 days at 1 µA, inside the fit, against the ≈ 2 X17 made in the whole
n_TOF thermal window (before any acceptance). Why it works where n_TOF could not: DC beam, no γ flash, so the DAQ is
91–98 % live up to 4 µA, at the exact ATOMKI kinematics.</p>
<p><b>Middle.</b> The opening-angle distribution for one week with big plastics: grey = background (IPC M1 and E1,
γ₁ pairs, cosmics after the MM-tracked veto), red = with X17 at R(ATOMKI) (m = 16.7 MeV) (<code>lnl/out/nd/maps/proj.csv</code>).
In 1D the X17 is a few % of the counts at 125–155°: a background-shape error of a few % competes with the signal, so the IPC
(and any other contamination) has to be known and subtracted very well. In-run controls: the E_p scan moves the X17 edge
(~10° over the scanned energies), off-resonance points, symmetric pairs.</p>
<p><b>Right.</b> S/B per cell of opening angle × energy split |y| (E_sum 14–19 MeV, 3° × 0.05 cells, cells with &lt; 3
background pairs per week blank; dashed: where a 16.7 MeV X17 lands). The X17 follows the constant-mass curve; the IPC (a
virtual photon of any mass) fills the plane. Impostor test: a background error that copies the X17 angle spectrum exactly
reads as 100 % of an X17 in an angle-only fit and 13–24 % in the θ × |y| × E_sum fit, with χ² ≈ 20 left over. So the energy
measurement makes the search resilient to background-modelling errors (LNL note, slides "2D counts" and "impostor").</p>
<p><b>Pseudo-data check</b> (§17): one simulated week at R(ATOMKI), blinded, truth differing from the fit model:
r̂ = 0.87 ± 0.14 (6.1σ); errors honest up to a week, a wider fit needed at 30 days. <b>Risks</b>: scattering in the target
chamber (1.0 d at 4.5° → 6.8 d at the n_TOF capsule's ~14°), cosmics, IPC shape. LNL's own ⁸Be spectrometer ran there in
2023–24. The drawing follows the LNL note's set-up slide (transverse, pinwheel offset not drawn); a 3D view is
<code>lnl/viz/lnl_setup_3d.html</code>.</p>"""
    D.slide('lnl', body, notes, short='LNL')


def _lnl_maps():
    """lnl/deck/maps_slides: the theta x |y| maps and projections (lnl/out/nd/maps, lnl_nd_maps.py)."""
    q = str(FAC / 'lnl/deck')
    if q not in sys.path:
        sys.path.insert(0, q)
    import maps_slides as MSm
    T = MSm.load(FAC / 'lnl/out')
    return dict(mod=MSm, **T)


def s_plastics(D, L):
    d = rd(FAC / 'lnl/out/plastics/days_vs_size.csv')
    # only the placement that keeps the SiPM wall (trigger and tracking point) where the bars are now
    places = [(d.placement.str.startswith('where the bars'), BLUE, 'plates where the bars are now (R 41 cm)')]
    P = sd.Plot(980, 640, x=(20, 120), y=(0.5, 100, 'log'), xlabel='side of a square plastic, 5 cm thick (cm)',
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
    P.text(62, 60, f'×{L["ab1"]["days_3sigma"] / L["bp1"]["days_3sigma"]:.0f} faster with 75 cm plates', 26, BLUE, weight=700)
    P.text(62, 45, 'flat beyond ~65 cm: the plate covers the cone', 22, INK)
    pr = L['J']['price_keur']
    cost = sd.table(['four plates', 'trigger acc.', 'cost (k€)'],
                    [['as built (2×20×30×2 cm)', '2.3 %', '0'],
                     ['70×70×2 PVT, 2×3" PMTs', '14.0 %', '79 (46–121)'],
                     ['78×80×4 PS, 2×3"', '14.4 %', '87 (50–132)'],
                     ['<b>78×80×5 PS, 2×3"</b>', '<b>14.5 %</b>', f'<b>{pr["PS"]["plates_keur"][1]:.0f}</b> ({pr["PS"]["plates_keur"][0]:.0f}–{pr["PS"]["plates_keur"][2]:.0f})'],
                     ['78×80×5 PVT, 2×3"', '14.5 %', f'{pr["PVT"]["plates_keur"][1]:.0f}']], size=21,
                    tips=['', '', 'acceptance only (no calorimetry)', '', 'polystyrene: ~70 % of PVT light, the cheap lever',
                          'cast PVT'])
    body = (sd.title(f'Big plastics: ×{L["ab1"]["days_3sigma"] / L["bp1"]["days_3sigma"]:.0f} fewer days; four 5 cm plates ≈ €{pr["PS"]["plates_keur"][1]:.0f}k',
                     'geometric toy of the trigger leg anchored to Geant4 (lnl/plastic_size.py) · price model trigger_scint/costs.py')
            + sd.row(P.svg('days vs size'), sd.col(sd.p('<b>Four plates, budget prices</b>', 24), cost, gap=12, w=620), gap=44, align='center'))
    notes = """
<p><b>The curve</b> (<code>lnl/plastic_size.py</code>, an analytic stand-in for a Geant4 size scan): straight leptons
from a 2 mm beam spot through the surveyed MM active area, Highland kicks in the MM and the SiPM wall, a leg must cross a
read-out SiPM bar and the plastic face above an energy threshold. Two free numbers are fitted to the Geant4 pair tag as
built and with 75×75 cm plastics. Days to 3σ scale as B/S² in 125–155°, anchored to the Geant4 big-plastics value;
E_sum, MM 15° and TOF efficiencies are held at their Geant4 values, so it is about geometry only. Where the plates
would collide with their neighbours they are pushed back (beyond ~75 cm).</p>
<p><b>Why it flattens:</b> past ~65–75 cm the plate covers the whole cone of leptons that crossed an MM (92 % of
X17 leptons at 75×75); the SiPM wall (83 %) is then the cap. Removing the wall (orange) leaves a plastic-only leg and
gains a little more, but the wall stays: it is the trigger and a tracking point, so only the curve with the plates
where the bars are now is drawn.</p>
<p><b>Prototype plate</b> (trigger_scint, optics toy): 78 × 80 × 5 cm polystyrene, fishtail guides, a 3" PMT at each
end; read all 20 SiPM bars (×1.4 on acceptance); FERS-5200 for the 160 SiPM ends (~€18k). Toy: ~77 pe/MeV, 4 %
non-uniformity with two ends, σt ≈ 124 ps at 5 MeV; ~15 %/√E, about 5 % at a 10 MeV deposit. All-in with the SiPM
readout ~€{allin:.0f}k. Whole LNL run budget: Lean €98k · Self-sufficient €174k · Upgrade €286k. Before buying: one
plate on the cosmic bench (light yield, uniformity against an MM map, linearity to 12 MeV).</p>
<p><b>Costs</b> (budgetary, EUR 2026, no labour/VAT): scintillator PVT 0.6 €/cm³, PS 0.25 €/cm³; PMT + divider 3" €3.3k;
guides €0.8–1.5k; wrapping €1.5–5k per slab; HV + digitizer €1.4k per PMT. Get quotes (Eljen/Luxium PVT, Nuvia/Epic PS).
Run budget: dylan-neff.web.cern.ch/facilities/budget.html (collaborator comments enabled).</p>"""
    notes = notes.replace('{allin:.0f}', f'{pr["PS"]["all_in_keur"][1]:.0f}')
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


def s_facilities(D, L):
    def card(name, beam, big, col, note, tip):
        return sd.col(sd.p(f'<b>{name}</b>', 26), sd.p(beam, 20, MUT),
                      sd.bignum(big, '', col, note, dark=False, size=50, tip=tip), gap=0, w=310)
    cards = sd.row(
        card('n_TOF EAR2', 'pulsed spallation, meV–GeV', '≈ 2 X17', RED, 'made in the thermal window: the MeV neutrons were lost to the γ flash',
             '≈100 X17 made over 436,373 pulses with the full spectrum, ≈2 in the thermal window after the flash dead time'),
        card('ILL PF1B', 'cold reactor beam', '> 50 d', ORANGE, 'one cycle stays ×1.3–3 above the ⁸Be level, even with a perfect detector',
             '3σ reach 5.3×10⁻³ per 50-day cycle with every upgrade: ×3.2 above the ⁸Be ratio; ×1.3 with no background'),
        card('Demokritos', 'Tandem, MeV neutrons', '?', GREY, 'MeV neutrons at the right energy, but the rate is expected to be too low',
             'not yet studied in x17_facility_search: no flux fold or reach estimate exists'),
        card('GANIL NFS', 'pulsed fast n, d+Be, 1–40 MeV', 'rate ✓ · E ✗', GOLD, 'good flux, but mostly above the physics window: needs a neutron-energy tag',
             'sub-2 MeV flux ~10⁵ n/cm²/s at 50 µA (model, unmeasured); time-of-flight would have to tag each neutron\'s energy'),
        card('LNL (for scale)', 'p on ⁷Li, DC, 1 µA', f'{L["bp1"]["days_3sigma"]:.0f}–{L["ab1"]["days_3sigma"]:.0f} d', GREEN,
             'to 3σ at the ATOMKI ratio, big plastics / as built', 'Geant4, θ × |y| × E_sum fit'),
        gap=26)
    body = (sd.title('Neutron facilities: statistics are the wall',
                     'x17_facility_search · sketch: the ILL PF1B target region, to scale along the beam')
            + cards + ill_svg(1664, 340).replace('width="1664" height="340"', 'width="1400" height="286"', 1))
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
<p><b>Demokritos</b> (NCSR Demokritos, Athens): a Tandem that makes quasi-monoenergetic MeV neutrons. Not studied in
x17_facility_search yet: no flux fold and no reach, hence the "?"; the expectation is that the rate is too low.</p>
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
            + sd.row(p1, p2, p3, p4, gap=24))
    notes = """
<p><b>Order</b>: I and II together (I cites II for every position, angle and drift velocity). III needs the pair rate
question settled or stated as a limitation; IV carries the facility reach so III stays about the data.</p>
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
ill/FEASIBILITY_SIM.md §13 (DREAM dead time).</p>""", short='B2 Lessons')


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
    D.slide('b-scatter', body, """<p>LNL FEASIBILITY_SIM.md §15; note scattering-upgrade (2026-10-06).</p>""", short='B5 Scattering')


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
background-shape error on the events seen by every analysis.</p>""", short='B6 LNL energy')


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
2.5×10⁻² would be a ~6–9σ effect in one cycle with timing + veto (ill/README.md).</p>""", short='B7 ILL reach')


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
            short='B8 Budget')


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
    D.slide('b-twotrack', body, '', short='B9 Two-track')


def b_flash(D):
    body = (sd.title('Why n_TOF could not see X17: the rate was in the MeV neutrons the flash took away',
                     'mpgd26 / ntof_athens_26 decks; x17_rate_3He (rate tables, thermal-corrected)')
            + sd.row(sd.bignum('79 %', 'of the X17 rate in the MeV', RED, 'p-wave 1⁻ resonance region, E_n 0.17–2 MeV', dark=False),
                     sd.bignum('5 ms', 'DREAM recovery after the γ flash', ORANGE, 'the detector itself recovers in 2 µs', dark=False),
                     sd.bignum('≈100 → ≈2', 'X17 made over the campaign', GREY,
                               'full spectrum vs the thermal window (0.01–0.1 eV: 4.4 → 0.07 X17/day)', dark=False), gap=40)
            + sd.callout('At thermal energies only s-wave channels are open (M1, and E0 with no real photon), (n,p) takes 92 % of the '
                         'neutrons and the ³He self-shields. The ²⁷Al pairs (~10⁸ per 30 days) became the demonstrator.', BLUE, 24))
    D.slide('b-flash', body, '', short='B4 Flash')


def main():
    L = lnl_numbers()
    D = sd.Deck('MX17 and the future X17 search', 'Status of the MX17 detectors and the n_TOF run, and whether the '
                'apparatus can search for X17 at LNL or a neutron facility')
    AL = al_numbers()
    s_cover(D, L, AL)
    s_best(D)
    s_fleet(D)
    s_expect(D, AL)
    s_scint(D)
    s_singles(D, AL)
    s_pairs(D, AL)
    s_energy(D)
    s_lnl(D, L)
    s_plastics(D, L)
    s_facilities(D, L)
    s_papers(D)
    b_passport(D)
    b_lessons(D)
    s_setups(D)
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
