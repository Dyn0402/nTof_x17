#!/usr/bin/env python3
"""
make_same_chamber_deck.py -- same-chamber pairs vs Geant4 as a figure-first
slide note: the time window, why aluminium e+e- pairs cannot vertex (and X17
legs can), where the capsule is, what the expected count already contains, and
chamber A's opening angle against Geant4.

Reads only the analyses' own tables:
  <out>/capsule_tof/{tof,beam_capture,g4,tilt}.csv          capsule_tof.py
  <out>/same_chamber/{scatter_legs,scatter_x17,vertex_truthline,sep_loss}.csv
                                                             same_chamber_scatter.py
  <out>/same_chamber/cmp_*.csv                               same_chamber_compare.py
  <out>/imaging_campaign/per_arm.csv                         campaign_imaging.py
  /media/dylan/data/x17/scint_stack/ana/eff_map.parquet      ntof_scint_stack

    python -m sept26_prelim_analysis.make_same_chamber_deck [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug same-chamber-pairs --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK)
from sept26_prelim_analysis import paths                               # noqa: E402

ARM_COL = dict(A=BLUE, C=ORANGE)
G4C = PURPLE
SC = paths.out('same_chamber')
CT = paths.out('capsule_tof')
IMG = paths.out('imaging_campaign')
POSE = paths.out('capsule_pose')
SCINT_EM = Path('/media/dylan/data/x17/scint_stack/ana/eff_map.parquet')
D_PLANE = 234.6
SIGMA_TAN = 0.028          # pooled cosmic-truth core resolution, |tan| 0.08-0.4


def rd(d, name):
    return pd.read_csv(d / f'{name}.csv')


def errbars(P, xs, ys, es, color, w=2.5):
    for x, y, e in zip(xs, ys, es):
        P.fore.append(sd.line(P.X(x), P.Y(max(y - e, P.ys[0])), P.X(x), P.Y(min(y + e, P.ys[1])), color, w))


# --------------------------------------------------------------------------- #
def load():
    A = {}
    A['tof'] = rd(CT, 'tof')
    A['beam'] = rd(CT, 'beam_capture')
    A['g4pos'] = rd(CT, 'g4')
    A['tilt'] = rd(CT, 'tilt')
    for n in ('scatter_legs', 'scatter_x17', 'vertex_truthline', 'sep_loss', 'cmp_opening',
              'cmp_opening_chi2', 'cmp_expectation', 'cmp_tof_fit', 'cmp_tof_fraction',
              'cmp_late_vertex'):
        A[n] = rd(SC, n)
    A['img'] = rd(IMG, 'per_arm').set_index('arm')
    A['pose_fit'] = rd(POSE, 'fit')
    A['pose_bound'] = rd(POSE, 'bound').set_index('height_prior_mm')
    A['pose_data'] = rd(POSE, 'data')
    A['pose_early'] = rd(POSE, 'early')
    A['pose_alt'] = pd.concat([rd(POSE, 'alternatives'), rd(POSE, 'halo')], ignore_index=True)
    return A


def numbers(A):
    N = {}
    b, t = A['beam'], A['tof']
    N['beam_lt12'] = float(b[b.hi <= 12].capture_share.sum())
    N['beam_peak'] = b.loc[b.per_ms.idxmax()]
    for arm in 'AC':
        g = t[t.arm == arm]
        w = (g.hi - g.lo)
        cap = g.focus_per_ms.clip(lower=0) * w
        N[f'cap_ge12_{arm}'] = float(cap[g.lo >= 12].sum() / cap.sum())
        late = g[g.lo >= 20]
        N[f'pur_late_{arm}'] = float(np.average(late.excess, weights=late.n))
        early = g[(g.lo >= 2) & (g.hi <= 10)]
        N[f'pur_early_{arm}'] = float(np.average(early.excess, weights=early.n))
        N[f'wall_early_{arm}'] = float(np.average(early.wall_frac, weights=early.n))
        N[f'wall_late_{arm}'] = float(np.average(late.wall_frac, weights=late.n))
        N[f'y_late_{arm}'] = float(np.average(late.y_cross, weights=1 / late.y_err ** 2))
    G = A['g4pos']
    rec = G[G.kind.str.startswith('reconstructed')].set_index('arm')
    N['g4_y'] = float(rec.v_cross.mean())
    N['g4_pur'] = rec.excess.to_dict()
    L = A['scatter_legs']
    allr = L[L.ke_hi > 50].iloc[0]
    N['al_miss'], N['al_defl'], N['al_pbt'] = allr.miss_med, allr.defl_med, allr.pbtheta
    N['al_p20'] = allr.p_miss20
    X = A['scatter_x17'].set_index('E_MeV')
    N['x17'] = X
    V = A['vertex_truthline']
    N['vtx_pass'] = float(V[V.open_lo < 0].pass_frac.iloc[0])
    N['vtx_off'] = float(V[V.open_lo < 0].mid_off_med.iloc[0])
    N['vtx_true20'] = float(V[V.open_lo < 0].true_in_r20.iloc[0])
    img = A['img']
    N['x_src'] = float((img.loc['A', 'median_mm'] + img.loc['C', 'median_mm']) / 2)
    N['z_src'] = float((img.loc['B', 'median_mm'] + img.loc['D', 'median_mm']) / 2)
    E = A['cmp_expectation']
    d = E[E.eps_source.str.startswith('digitised')].set_index('arm')
    N['exp'] = d
    S = A['sep_loss']
    N['sep'] = S[S.open_lo < 0].set_index('arm')
    return N


# --------------------------------------------------------------------------- #
def s_cover(D, A, N):
    x10 = N['x17'].loc[10.0]
    body = (sd.kicker(f'n_TOF EAR2 · X17 · same-chamber pairs · {dt.date.today():%-d %b %Y}')
            + sd.title('<span style="color:#eef0f3">The &lt; 12 ms cut is not where the pairs went, '
                       'aluminium e⁺e⁻ cannot vertex, and y now points up</span>')
            + sd.row(
                sd.bignum(sd.pct(N['beam_lt12']), 'of the beam\'s own captures fall before 12 ms',
                          DBLUE, f'but {sd.pct(1 - N["cap_ge12_A"])} (A) / {sd.pct(1 - N["cap_ge12_C"])} (C) of '
                          'the tracks that point at the capsule do: the DAQ is full of flash junk then',
                          tip='EAR2 evaluated flux x thin 1/v absorber, 1-75 ms window; capsule-pointing = '
                              'line within 15 mm of the measured source, minus a shuffled-angle null'),
                sd.bignum(sd.pct(N['vtx_pass'], 1), 'of real Al e⁺e⁻ pass the vertex test',
                          DRED, 'with PERFECT tracks (Geant4 truth gap lines); chance is 2–3 %',
                          tip=f'1,772 same-chamber conversion pairs; midpoint lands {N["vtx_off"]:.0f} mm '
                              '(median) from the true vertex'),
                sd.bignum(f'{N["al_miss"]:.0f} vs {x10.miss_med:.0f} mm', 'miss at the vertex: Al leg vs X17 leg',
                          DGREEN, 'scattering ∝ 1/p: 2 MeV conversion leptons vs ~10 MeV X17 legs',
                          tip=f'p·β·θ = {N["al_pbt"]:.2f} MeV·rad, measured in Geant4 over 0.5-9 MeV'),
                sd.bignum(f'{N["y_late_A"]:+.0f} mm', 'source height vs −28 mm (Geant4 nose)',
                          GOLD, 'stage 3 stored y mirrored from 7 Sep (+31 here) until the 14 Oct '
                                'regeneration; the y strips count upward', size=64),
                gap=44))
    notes = """
<p>Answers to five questions from 2026-10-13, all on the is2_v2 re-pass at the capsule-focus k (stage3_is2_v2_cf),
post-access runs, gated tracks with |tan| &lt; 0.6 and t0 ≤ 300 ns, chambers A and C, unless a slide says otherwise.</p>
<ol>
<li>Why cut the first 12 ms when the thermal peak is there? (slides 2–3)</li>
<li>Does wall scattering really stop Al pairs pointing back, and is X17 different? (slides 4–5)</li>
<li>Where is the capsule, and could its offset or tilt explain the rate? (slides 6–7)</li>
<li>Does the expected count include the opening angle and the &lt; 20 mm loss? (slide 8)</li>
<li>Chamber A's opening angle against Geant4. (slide 9)</li>
</ol>
<p>Hover the dotted terms and the data points for values, counts and provenance.</p>"""
    D.slide('cover', body, notes, dark=True)


def s_tof(D, A, N):
    t, b = A['tof'], A['beam']
    P = sd.Plot(1000, 600, x=(0, 75), y=(0, 1.08), xlabel='time since the γ flash (ms)',
                ylabel='per ms, scaled to its own peak')
    P.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40, 50, 60, 70)])
    P.yticks([(v, f'{v:.2f}') for v in (0, 0.25, 0.5, 0.75, 1.0)])
    mid = 0.5 * (b.lo + b.hi)
    bm = b.per_ms / b.per_ms.max()
    tips = [f'{lo:g}-{hi:g} ms (E {elo:.0f}-{ehi:.0f} meV): {100 * s:.1f} % of beam captures'
            for lo, hi, s, elo, ehi in zip(b.lo, b.hi, b.capture_share, b.e_lo_meV, b.e_hi_meV)]
    P.line(list(mid), list(bm), INK, dash='10 7', tips=tips, r=5)
    P.vline(12, RED, label='12 ms cut', tip='pair work has used tracks >= 12 ms since 2026-10-13')
    for arm in 'AC':
        g = t[t.arm == arm]
        m = 0.5 * (g.lo + g.hi)
        rec = g.per_ms / g.per_ms[g.lo >= 2].max()
        cap = g.focus_per_ms.clip(lower=0) / g.focus_per_ms.max()
        P.line(list(m[g.lo >= 2]), list(rec[g.lo >= 2]), ARM_COL[arm], w=2, dash='3 6', markers=False,
               tip=f'{arm}: all recorded tracks per ms')
        tp = [f'{arm} {lo:g}-{hi:g} ms: {fp:,.0f} capsule-pointing tracks/ms of {pm:,.0f} '
              f'(purity {100 * ex:.0f} %)' for lo, hi, fp, pm, ex in
              zip(g.lo, g.hi, g.focus_per_ms, g.per_ms, g.excess)]
        P.line(list(m), list(cap), ARM_COL[arm], tips=tp)
    leg = sd.legend([('beam captures, EAR2 flux × 1/v', INK, 'dash'),
                     ('tracks pointing at the capsule', BLUE), ('all recorded tracks', BLUE, 'dash')], size=22)
    pk = N['beam_peak']
    col = sd.col(
        sd.callout(f'The beam\'s captures peak at {pk.lo:g}–{pk.hi:g} ms ({pk.e_lo_meV:.0f}–{pk.e_hi_meV:.0f} meV) '
                   f'and {sd.pct(N["beam_lt12"])} of them fall before 12 ms. You are right about where the '
                   'thermal peak is.', INK, 24),
        sd.callout(f'But {sd.pct(N["cap_ge12_A"])} (A) / {sd.pct(N["cap_ge12_C"])} (C) of the tracks that '
                   'point back at the capsule are after 12 ms, and half are after 30 ms, where the direct '
                   'beam has 0.2 % of its captures.', BLUE, 24),
        sd.callout('The recorded rate is roughly flat at 4–30 ms: the DAQ is ' + sd.term(
            'throughput-limited', 'DREAM readout dead time per trigger; the July run_55 study saw the '
            'trigger "comb" (mx_july_beam_qa/HV_SCAN_RUN55_ANALYSIS.md). Recorded tracked events/ms '
            'are 30-36k from 4 to 30 ms while the beam capture rate falls x15.')
                   + ', so early triggers are taken by whatever fires first.', RED, 24),
        w=600, gap=22)
    body = (sd.title('The beam captures at 4–10 ms, but our capsule tracks are recorded after 12 ms',
                     'chambers A and C, is2_v2 cf, post-access runs · each curve scaled to its own peak')
            + sd.row(sd.col(P.svg('capture and track rates vs time'), leg, gap=8, w=1000), col, gap=56))
    notes = f"""
<p><b>Beam curve.</b> The EAR2 evaluated flux the Geant4 gun samples
(<code>fluxEAR2-Ph3_in_different_units.root</code>, n/pulse per bin) times a thin 1/v absorber, at 19.5 m.
The Al nose is thin for neutrons, so this is the time profile of the beam's own capture γ.
Within the 1–75 ms acquisition window: {sd.pct(N['beam_lt12'])} before 12 ms.</p>
<p><b>Capsule-pointing tracks.</b> Per ToF bin, the fraction of tracks whose x line passes within 15 mm of the
late-time band crossing (A −10.2, C −10.7 mm in global x), minus the same with the angles shuffled between tracks
(the chance rate, ~10–12 %), times the tracks per ms. This is a pointing measure, not a shape fit.</p>
<p><b>What the late capsule tracks are.</b> They cannot be direct beam neutrons: the flux has essentially nothing
after 28 ms, and half of the capsule-pointing tracks come after 30 ms with a ~40 ms decay. They are delayed
(thermalised, room-return) neutrons captured in the capsule's aluminium, the same population the late-trigger
clock found (T½ ≈ 24 ms; <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §10). Their source height is the
same as the beam's would be (slide 6), so they are captured at the same end of the capsule.</p>
<p>Code: <code>python -m sept26_prelim_analysis.capsule_tof beam tof</code> → <code>{CT}/tof.csv, beam_capture.csv</code>.</p>"""
    D.slide('tof', body, notes, foot='EAR2 flux: Sabaté-Gilarte et al. (n_TOF Phase 3 evaluation) · '
                                     'capsule pointing: band crossing, shuffled-angle null subtracted')


def s_early(D, A, N):
    t = A['tof']
    F = A['cmp_tof_fraction']
    P1 = sd.Plot(800, 560, x=(0, 75), y=(0, 0.7), xlabel='time since the γ flash (ms)',
                 ylabel='capsule-pointing purity', title='how many tracks point at the capsule')
    P1.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40, 50, 60, 70)])
    P1.yticks([(v, f'{100 * v:.0f}%') for v in (0, 0.2, 0.4, 0.6)])
    for arm in 'AC':
        g = t[t.arm == arm]
        m = 0.5 * (g.lo + g.hi)
        P1.line(list(m), list(g.excess), ARM_COL[arm],
                tips=[f'{arm} {lo:g}-{hi:g} ms: purity {100 * e:.1f} % ({100 * f:.1f} % within 15 mm, '
                      f'null {100 * f0:.1f} %), n = {n:,}; own wall fired {100 * w:.0f} %'
                      for lo, hi, e, f, f0, n, w in zip(g.lo, g.hi, g.excess, g.f_focus, g.f_null, g.n,
                                                        g.wall_frac)])
        P1.hline(N['g4_pur'][arm], ARM_COL[arm], dash='4 6',
                 tip=f'Geant4 through the real reconstruction (g4_digi, {arm}): {100 * N["g4_pur"][arm]:.1f} %'
                     ' - every simulated track comes from the capsule')
    P1.text(45, N['g4_pur']['C'] + 0.03, 'Geant4, capsule-only (dashed)', 20, MUT)
    P1.vline(12, RED)
    P2 = sd.Plot(800, 560, x=(0, 75), y=(5e-4, 0.3, 'log'), xlabel='time since the γ flash (ms)',
                 ylabel='events with a 2nd track in the chamber', title='same-chamber pair fraction')
    P2.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40, 50, 60, 70)])
    P2.yticks([(1e-3, '0.1%'), (1e-2, '1%'), (1e-1, '10%')])
    fit = A['cmp_tof_fit'].set_index('arm')
    for arm in 'AC':
        g = F[(F.arm == arm) & (F.lo >= 1)]
        m = 0.5 * (g.lo + g.hi)
        P2.line(list(m), list(g.frac.clip(lower=6e-4)), ARM_COL[arm],
                tips=[f'{arm} {lo:g}-{hi:g} ms: {n2:,} of {n1:,} events ({100 * f:.2f} %)'
                      for lo, hi, n1, n2, f in zip(g.lo, g.hi, g.n1, g.n2, g.frac)])
        P2.hline(fit.loc[arm, 'a'], ARM_COL[arm], dash='4 6',
                 tip=f'{arm}: rate-independent (correlated) part of the fit over 12-75 ms, '
                     f'a = {100 * fit.loc[arm, "a"]:.3f} ± {100 * fit.loc[arm, "a_err"]:.3f} %')
    P2.vline(12, RED, label='12 ms')
    body = (sd.title('Before 12 ms most tracks do not come from the capsule, and pairs are pile-up',
                     f'left: purity vs time, data and Geant4 · right: pair fraction, dashed = the correlated '
                     f'{100 * fit.loc["A", "a"]:.2f} % (A) / {100 * fit.loc["C", "a"]:.2f} % (C)')
            + sd.row(P1.svg('purity vs time'), P2.svg('pair fraction vs time'), gap=44)
            + sd.legend([('A', BLUE), ('C', ORANGE)], size=22))
    notes = f"""
<p><b>Purity.</b> At 2–10 ms only {sd.pct(N['pur_early_A'])} (A) / {sd.pct(N['pur_early_C'])} (C) of tracks point
at the capsule, against {sd.pct(N['pur_late_A'])} / {sd.pct(N['pur_late_C'])} after 20 ms. The arm's own SiPM wall
fired for {sd.pct(N['wall_early_A'])} / {sd.pct(N['wall_early_C'])} of the early tracks and
{sd.pct(N['wall_late_A'])} / {sd.pct(N['wall_late_C'])} of the late ones. The late data are at least as pure as
Geant4's capsule-only sample through the reconstruction ({sd.pct(N['g4_pur']['A'])} / {sd.pct(N['g4_pur']['C'])}),
so the late denominator is not inflated by non-capsule tracks. That rules out "other sources in the data" as the
factor-8 explanation for t ≥ 12 ms.</p>
<p><b>Pair fraction.</b> It is 1–14 % before 10 ms and 0.2–0.4 % after; the 12–75 ms fit f = a + b·rate gives the
correlated a. Early pairs are mostly pile-up and flash junk; the correlated pairs are there too, in proportion:
with a constant a they would be ~0.15 % of the ~465 k early A events, about 700 pairs, against 15–30 k early
pairs in total. So "junk" means signal-to-background of a few percent, not "no signal".</p>
<p><b>Can the early window be recovered?</b> Only with a cleaner early trigger and a faster DAQ, which is now
moot for this dataset. For the analysis the cut costs about a quarter of the correlated pairs; a purity cut
(lines through the capsule) applied before pairing could win some of it back. The 2026-10-13 vertex artefact
came from mixing on early events, so any early-time pair work needs its own early-time mixing pool.</p>"""
    D.slide('early', body, notes, foot='pair fraction: same_chamber_compare (events_tof) · purity: capsule_tof tof')


def scatter_diagram(N):
    W, H = 980, 600
    cx, cy = 120, 300
    s = []
    s.append(f'<rect x="{cx - 70}" y="{cy - 150}" width="140" height="300" rx="70" fill="#d9dde3" stroke="{GREY}" stroke-width="3"/>')
    s.append(f'<rect x="{cx - 56}" y="{cy - 136}" width="112" height="272" rx="56" fill="#eaf3fb" stroke="none"/>')
    s.append(sd.T(cx, cy + 180, '³He capsule', 21, INK, weight=600))
    s.append(sd.T(cx, cy + 206, 'Al wall + CFRP', 19, MUT))
    gx = 860
    s.append(f'<rect x="{gx}" y="40" width="60" height="500" fill="#f3ead8" stroke="{GOLD}" stroke-width="2"/>')
    s.append(sd.T(gx + 30, 30, 'drift gap', 21, INK, weight=600))
    s.append(sd.T(520, 590, '≈ 23 cm of air and the cathode between the wall and the gap', 20, MUT))
    vx, vy = cx + 63, cy
    legs = (([(vx, vy), (260, 268), (450, 205), (650, 156), (gx, 125)], RED),
            ([(vx, vy), (260, 312), (450, 338), (650, 380), (gx, 440)], BLUE))
    lines = []
    for pts, c in legs:
        s.append(sd.poly([q[0] for q in pts], [q[1] for q in pts], c, 4))
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        k = (y2 - y1) / (x2 - x1)
        lines.append((k, y2))
        xb = 40
        s.append(sd.line(gx, y2, xb, y2 + k * (xb - gx), c, 2.5, '9 7'))
        s.append(sd.line(gx, y2, gx + 60, y2 + k * 60, c, 7))
    (k1, a1), (k2, a2) = lines
    d = (a2 - a1) / (k1 - k2)
    ix, iy = gx + d, a1 + k1 * d
    s.append(f'<circle cx="{ix:.1f}" cy="{iy:.1f}" r="11" fill="none" stroke="{INK}" stroke-width="3"'
             f'{sd.tipattr("where the two measured lines cross: the reconstructed vertex")}/>')
    s.append(sd.T(ix - 4, iy - 22, 'lines cross here', 19, INK, 'middle', 600))
    s.append(f'<circle cx="{vx}" cy="{vy}" r="10" fill="{INK}"{sd.tipattr("true conversion vertex, in the Al wall")}/>')
    s.append(sd.T(vx + 14, vy + 34, 'true vertex', 20, INK, 'start', 600))
    x10 = N['x17'].loc[10.0]
    s.append(sd.T(560, 72, f'Al pair leg (median 2 MeV): bent {N["al_defl"]:.0f}° on the way out', 22, RED, weight=600))
    s.append(sd.T(560, 100, f'its measured line misses the vertex by {N["al_miss"]:.0f} mm (median)', 22, RED))
    s.append(sd.T(560, 500, f'an X17 leg (~10 MeV) bends {x10.defl_med:.1f}° and misses by {x10.miss_med:.0f} mm', 22, GREEN, weight=600))
    s.append(sd.T(560, 528, f'(resolution alone: σ_tan {SIGMA_TAN} × 235 mm ≈ {SIGMA_TAN * D_PLANE:.0f} mm)', 20, MUT))
    s.append(sd.T(gx - 14, 270, 'measured:', 19, MUT, 'end'))
    s.append(sd.T(gx - 14, 294, 'the line in the gap,', 19, MUT, 'end'))
    s.append(sd.T(gx - 14, 318, 'extended back (dashed)', 19, MUT, 'end'))
    return sd.svg(W, H, ''.join(s), 'schematic of a scattered conversion pair')


def s_scatter(D, A, N):
    L = A['scatter_legs']
    L = L[L.ke_hi < 50]
    P = sd.Plot(640, 560, x=(0.5, 12, 'log'), y=(0, 130), xlabel='lepton kinetic energy at the gap (MeV)',
                ylabel='miss at the true vertex (mm)', title='Geant4: miss vs energy')
    P.xticks([(0.5, '0.5'), (1, '1'), (2, '2'), (4, '4'), (8, '8')])
    P.yticks([(v, str(v)) for v in (0, 25, 50, 75, 100, 125)])
    xm = list(np.sqrt(L.ke_lo * L.ke_hi))
    P.band(xm, list(L.miss_med * 0), list(L.miss_p68), RED, 0.12, tip='68 % of legs miss by less than the band top')
    P.line(xm, list(L.miss_med), RED, tips=[
        f'{lo:g}-{hi:g} MeV: median miss {m:.0f} mm (68 %: {p:.0f}), deflection {d:.1f}°, n = {n:,}'
        for lo, hi, m, p, d, n in zip(L.ke_lo, L.ke_hi, L.miss_med, L.miss_p68, L.defl_med, L.n)])
    X = N['x17']
    P.points(list(X.index), list(X.miss_med), GREEN, r=10, marker='diamond',
             tips=[f'X17 leg {e:g} MeV (1/(pβ) scaling): median miss {m:.1f} mm, deflection {d:.1f}°; '
                   f'P(miss < 20 mm) = {100 * p:.0f} %' for e, m, d, p in
                   zip(X.index, X.miss_med, X.defl_med, X.p_miss20)])
    P.hline(SIGMA_TAN * D_PLANE, GREY, label='resolution × 235 mm', anchor='start', tip='pooled cosmic-truth σ_tan ≈ 0.028')
    body = (sd.title('Scattering on the way out, not the reconstruction, breaks the Al pair vertex',
                     f'Geant4 truth, {int(A["scatter_legs"].n.iloc[-1]):,} conversion leptons from the 10⁹-neutron '
                     f'nose campaign · p·β·θ = {N["al_pbt"]:.2f} MeV·rad at every energy')
            + sd.row(scatter_diagram(N), P.svg('miss vs energy'), gap=24))
    notes = f"""
<p><b>What is measured.</b> For each conversion lepton that crosses ≥ 20 mm of a drift gap: the line through its
first gap step along its gap direction, and that line's distance of closest approach to the true conversion point.
The deflection is the angle between the vertex→gap chord and the gap direction. No reconstruction is involved.</p>
<p><b>The law.</b> The median p·β·θ is {N['al_pbt']:.2f} MeV·rad in every bin from 0.5 to 6 MeV (0.31 at 6–9 MeV,
n = 92), which is multiple scattering: θ ∝ 1/(pβ). The path is the capsule wall (0.6 mm Al barrel, up to 5.5 mm at
the nose, 0.9 mm CFRP), then about 23 cm of air and the drift cathode. The median conversion lepton has 2.1 MeV at the
gap, so it arrives {N['al_defl']:.0f}° off its birth direction.</p>
<p><b>X17 legs</b> (8–10 MeV) are scaled with the same law: {X.loc[10.0].defl_med:.1f}–{X.loc[8.0].defl_med:.1f}°,
miss {X.loc[10.0].miss_med:.0f}–{X.loc[8.0].miss_med:.0f} mm, comparable to the angular resolution
(~{SIGMA_TAN * D_PLANE:.0f} mm). {sd.pct(X.loc[10.0].p_miss20)}–{sd.pct(X.loc[8.0].p_miss20)} of X17 legs pass
within 20 mm of their vertex, against {sd.pct(N['al_p20'])} of Al legs.</p>
<p>Code: <code>python -m sept26_prelim_analysis.same_chamber_scatter</code> → <code>{SC}/scatter_legs.csv,
scatter_x17.csv</code>.</p>"""
    D.slide('scatter', body, notes, foot='Geant4 neutrons_thermal_trig_2cm_nose (nose-first capsule, centred beam), '
                                         'same_arm_pairs.py truth per lepton')


def s_vertex(D, A, N):
    V = A['vertex_truthline']
    V = V[V.open_lo >= 0]
    X = N['x17']
    lv = A['cmp_late_vertex']
    lv = lv[lv.chain == 'cf'].set_index('arm')
    rows = [(f'{int(lo)}–{int(hi)}°', float(p), RED,
             f'{n} pairs; midpoint {m:.0f} mm from the true vertex (median, 68 %: {q:.0f}); line DCA {d:.0f} mm')
            for lo, hi, p, n, m, q, d in zip(V.open_lo, V.open_hi, V.pass_frac, V.n, V.mid_off_med,
                                             V.mid_off_p68, V.dca_med)]
    bars = sd.hbars(rows, 0.1, width=520, label_w=150, fmt=lambda v: f'{100 * v:.1f} %', size=24)
    comp = sd.table(['', 'Al e⁺e⁻ (same chamber)', 'X17 (opposite chambers)'], [
        ['lepton energy', '~2 MeV', '~8–10 MeV'],
        ['miss at the vertex, per leg', f'{N["al_miss"]:.0f} mm', f'{X.loc[10.0].miss_med:.0f}–{X.loc[8.0].miss_med:.0f} mm'],
        ['opening angle', 'median ~25°', '≳ 140°'],
        ['geometry of the test', 'two near-parallel lines: error ÷ sin(opening)', 'two lines back to back: each must hit the axis'],
        ['passes the vertex test', sd.pct(N['vtx_pass'], 1) + ' (perfect tracks)',
         f'{sd.pct(X.loc[8.0].p_both20)}–{sd.pct(X.loc[10.0].p_both20)} (both legs within 20 mm)'],
    ], size=22, widths=[300, 360, 380], align=['left', 'left', 'left'])
    body = (sd.title('Al e⁺e⁻ pass the vertex test at the chance rate; X17 pairs would pass it',
                     'left: Geant4 same-chamber e⁺e⁻, TRUE gap lines, vertex test (DCA < 30 mm, midpoint within '
                     '20 mm of the axis), by opening angle')
            + sd.row(sd.col(sd.p('pass fraction with perfect tracks', 24, MUT), bars,
                            sd.p(f'chance (event-mixed, late data): {100 * lv.loc["A", "expected"] / max(lv.loc["A", "n_real"], 1):.1f} % (A)', 22, MUT),
                            w=820, gap=16),
                     sd.col(comp, sd.callout(
                         f'{sd.pct(N["vtx_true20"])} of these pairs really are born within 20 mm of the axis. '
                         'The test cannot see them because two lines at 25° cross far from where they '
                         'were born once each has been bent by ~9°.', RED, 24), gap=28), gap=48))
    notes = f"""
<p><b>Why near-parallel lines are the problem.</b> The intersection of two lines at opening angle α moves by about
δ / sin α along the bisector when one line is displaced by δ. With δ ≈ {N['al_miss']:.0f} mm and α ≈ 25°, the
midpoint lands ~{N['al_miss'] / np.sin(np.radians(25)):.0f} mm away; Geant4 says {N['vtx_off']:.0f} mm (median).
Small angles are worst (0–15°: 1.6 %), and no bin passes more than 3 %.</p>
<p><b>The X17 case is different in all three factors.</b> Its legs carry 4–5× the momentum, so the miss per leg is
~{X.loc[10.0].miss_med:.0f} mm; they go to opposite chambers, so the vertex test is "does each line pass the axis"
rather than "where do two near-parallel lines cross"; and the 470 mm A–C lever arm makes the line itself
well defined. The 67–82 % assumes the same wall and air as the Al pairs (X17 is born in the gas and crosses the same
wall) and ignores resolution, which adds ~{SIGMA_TAN * D_PLANE:.0f} mm in quadrature.</p>
<p><b>So the 2026-10-12 "vertex excess" could never have been the Al pairs:</b> a test that real pairs pass 2 % of
the time and chance passes 2–3 % of the time cannot show a ×2 excess of them. The late-only result (A ×1.00) is what
Geant4 predicts.</p>
<p>Code: <code>same_chamber_scatter.py</code> → <code>vertex_truthline.csv</code>.</p>"""
    D.slide('vertex', body, notes, foot='Geant4 truth, 1,772 same-chamber conversion pairs (A and C); X17 column from the '
                                        'scaled scattering law of the previous slide')


def topdown_svg(N):
    W, H, s0 = 620, 560, 9.0       # px per mm around the axis
    cx, cy = W / 2, H / 2 + 10
    out = []
    # arms (convention: +z right = A, -x bottom = B, -z left = C, +x top = D)
    for lab, (dx, dy) in dict(A=(1, 0), C=(-1, 0), D=(0, -1), B=(0, 1)).items():
        x, y = cx + dx * 250, cy + dy * 240
        out.append(sd.T(x, y + 8, f'{lab}', 30, INK, weight=600))
    out.append(sd.line(cx - 200, cy, cx + 200, cy, RULE, 1.5))
    out.append(sd.line(cx, cy - 200, cx, cy + 200, RULE, 1.5))
    for r in (10, 20):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r * s0}" fill="none" stroke="{RULE}" stroke-dasharray="6 6"/>')
    out.append(sd.T(cx + 20 * s0 * 0.72, cy - 20 * s0 * 0.72 - 6, '20 mm', 18, MUT))
    out.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{INK}"{sd.tipattr("nominal beam axis = chamber-frame centre")}/>')
    out.append(sd.T(cx + 12, cy - 12, 'nominal axis', 18, MUT, 'start'))
    sx, sz = N['x_src'], N['z_src']
    px, py = cx + sz * s0, cy - sx * s0
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{10.6 * s0}" fill="{GOLD}" fill-opacity="0.18" stroke="{GOLD}" '
               f'stroke-width="3"{sd.tipattr("capsule outline (Al OD 21.2 mm) drawn on the measured source")}/>')
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{GOLD}"'
               f'{sd.tipattr(f"measured source: x = {sx:.1f} mm (A, C band crossings), z = {sz:.1f} mm (B, D)")}/>')
    out.append(sd.arrow(cx, cy, px, py, GOLD, 3))
    out.append(sd.T(px - 6, py + 10.6 * s0 + 30, f'source ({sx:.1f}, {sz:.1f}) mm', 21, INK, 'middle', 600))
    out.append(sd.T(W / 2, H - 4, 'top-down, beam out of the page (+z right, +x up)', 18, MUT))
    return sd.svg(W, H, ''.join(out), 'top-down capsule position')


def sideview_svg(N):
    W, H = 520, 560
    out = []
    sc = 3.4                                    # px per mm
    y0 = H / 2

    def Y(v):
        return y0 - v * sc
    xs = 200
    # capsule (nose down) in CAD frame, gas from -29.5 to +50.7, Al tip at -35
    prof = [(-35, 0), (-33, 5.3), (-29, 8.5), (-25, 10.0), (-21, 10.6), (20, 10.6), (25, 10.0), (29, 8.5),
            (33, 5.7), (37, 4.0), (40, 3.5), (51, 3.5)]
    xl = [xs - r * sc for _, r in prof] + [xs + r * sc for _, r in prof[::-1]]
    yl = [Y(y) for y, _ in prof] + [Y(y) for y, _ in prof[::-1]]
    out.append(sd.poly(xl, yl, GREY, fill='#d9dde3'))
    out.append(sd.poly(xl + [xl[0]], yl + [yl[0]], GREY, 2))
    out.append(sd.T(xs, Y(57), 'valve', 18, MUT))
    out.append(sd.T(xs, Y(-44), 'nose (5.5 mm Al)', 18, MUT))
    out.append(sd.arrow(60, Y(-75), 60, Y(-45), INK, 3))
    out.append(sd.T(60, Y(-80), 'beam', 18, INK))
    out.append(sd.line(30, Y(0), W - 20, Y(0), RULE, 1.5))
    out.append(sd.T(W - 20, Y(0) - 8, 'y = 0', 18, MUT, 'end'))
    marks = [(N['g4_y'], PURPLE, f'Geant4 {N["g4_y"]:.0f}'),
             (N['y_late_A'], BLUE, f'data {N["y_late_A"]:+.0f} (A)'),
             (N['y_late_C'], ORANGE, f'data {N["y_late_C"]:+.0f} (C)'),
             (-N['y_late_A'], GREY, f'before the y fix: {-N["y_late_A"]:+.0f}')]
    for (v, c, lab), dy in zip(marks, (-30, -6, 18, 0)):
        out.append(sd.line(xs - 50, Y(v), xs + 70, Y(v), c, 5))
        out.append(sd.T(xs + 80, Y(v) + 7 + dy, lab, 20, c, 'start', 600))
    return sd.svg(W, H, ''.join(out), 'side view source height')


def s_capsule(D, A, N):
    body = (sd.title('The source sits one capsule radius off-axis, toward B and C',
                     'band crossings (pointing, scale-free), late tracks, campaign medians · '
                     'height: Geant4 and data with the same estimator')
            + sd.row(topdown_svg(N), sideview_svg(N),
                     sd.col(sd.callout(f'Transverse: x = {N["x_src"]:.1f} mm (A, C agree to ±1 mm), '
                                       f'z = {N["z_src"]:.1f} mm (B, D). That is {np.hypot(N["x_src"], N["z_src"]):.0f} mm '
                                       'from the chamber-frame centre, toward the B–C corner, mostly B.', GOLD, 24),
                            sd.callout(f'Height: data {N["y_late_A"]:+.0f} (A) / {N["y_late_C"]:+.0f} (C) mm. Geant4\'s '
                                       f'centred, nose-first capsule gives {N["g4_y"]:.0f} mm by the same estimator '
                                       '(truth −28): its captures sit in the nose, and so do the data. Until the 14 Oct '
                                       'regeneration stage 3 stored y mirrored (next slide).', GREEN, 24),
                            w=520, gap=24), gap=28))
    notes = f"""
<p><b>Transverse.</b> From <code>campaign_imaging</code> (per-run band crossings, post-access runs): A −8.3, C −10.3
(global x; their half-difference is the A–C alignment), D −3.5, B −4.1 (global z, B from far fewer tracks).
<code>capsule_tof tilt</code> reproduces x on is2_v2 cf (A −10.2, C −10.7). The sign convention is the one of
<code>GEOMETRY_COORDINATE_CONVENTION.md</code>: A +z, B −x, C −z, D +x. So the source is displaced toward B (−x)
and slightly toward C (−z).</p>
<p><b>What the offset means.</b> The imaging measures where captures happen relative to the chambers, not where the
beam is. The beam is narrow at thermal energies (50 % within r = 2.5 mm, 90 % within 13 mm; Lambda2D). If it is
centred on the chamber frame and the capsule sits 10 mm off it, only ~half the beam meets the capsule, the nose core
gets ~5 %, and ~11 % grazes the barrel wall, where a ray can run 40 mm through aluminium. If beam and capsule are offset
together, the nose is hit as in Geant4. The source height can tell these apart, but most capsule-pointing tracks are
delayed neutrons rather than beam (slide 2), so it is not yet settled. A pose fit is handed off
(<code>HANDOFF_CAPSULE_POSE_FIT.md</code>).</p>
<p><b>Height and the y sign.</b> The y-strip direction was fixed in software on 2026-09-07
(<code>build_tracks.IN_PLANE_SIGN_Y = −1</code>) by requiring corr(y, tan y) &gt; 0. That test cannot tell
"positions flipped" from "tans flipped": both make the correlation positive, and they differ by exactly a mirror of
the vertical axis. For x the pinwheel offset decided it; y has no such offset. The July hits-based analysis, which did
not flip positions, put the source "~3 cm low, the bottom of the capsule". The 2026-09-14 capsule-height handoff then
concluded that the CAD height was wrong by 30 mm. That conclusion and this flip are the same 60 mm.</p>
<p>A global y mirror leaves every angle between tracks unchanged (opening angles, X17 included). It changed absolute
heights: the source height, <code>target_y_mm</code>, the pair-vertex y window (<code>CAPSULE_V</code> in
<code>same_chamber_physics</code>), the shift used to compare with Geant4 (+32.2 mm then, −4 mm now), and every
v-map of the scintillators. Fixed in code on 13 Oct, every table regenerated on 14 Oct; this deck is on the fixed
tables.</p>"""
    D.slide('capsule', body, notes, foot='imaging: campaign_imaging per_arm.csv · Geant4: g4_digi g4_{A,C}_is2_full '
                                         '(nose-first, centred) through the real reconstruction')


def s_yflip(D, A, N):
    T = A['tilt']
    t = T[(T.chain == 'is2_v2_cf') & (T.lo >= -100) & (T.hi <= 40)]
    P = sd.Plot(800, 560, x=(-120, 60), y=(-18, -6), xlabel='track height at the axis (mm, +y up)',
                ylabel='source x, band crossing (mm)', title='source x against height (tilt?)')
    P.xticks([(v, str(v)) for v in (-120, -90, -60, -30, 0, 30, 60)])
    P.yticks([(v, str(v)) for v in (-18, -15, -12, -9, -6)])
    for arm in 'AC':
        g = t[(t.arm == arm) & (t.err < 2)]
        m = 0.5 * (g.lo + g.hi)
        P.line(list(m), list(g.cross), ARM_COL[arm],
               tips=[f'{arm} y {lo:g}-{hi:g}: x = {c:.1f} ± {e:.1f} mm, n = {n:,}' for lo, hi, c, e, n in
                     zip(g.lo, g.hi, g.cross, g.err, g.n)])
        errbars(P, list(m), list(g.cross), list(g.err), ARM_COL[arm])
    P.text(-115, -7, 'Geant4 (centred): flat within ±1 mm over its populated heights', 19, PURPLE)
    # liquid B profile
    R = None
    if SCINT_EM.exists():
        EM = pd.read_parquet(SCINT_EM)
        M = EM[(EM.layer == 'liq_full') & (EM['sample'] == 'all_late') & (EM.arm == 'B')]
        R = (M.groupby('y').apply(lambda s: (s.k_on.sum() - s.k_off.sum()) / max(s.n.sum(), 1),
                                  include_groups=False))
        R = R[R.index.to_series().abs() < 220]
    P2 = sd.Plot(760, 560, x=(-220, 220), y=(0, 0.07), xlabel='height on the liquid (mm, +y up)',
                 ylabel='liquid B net firing', title='liquid B (PMT on top per the survey)')
    P2.xticks([(v, str(v)) for v in (-200, -100, 0, 100, 200)])
    P2.yticks([(v, f'{100 * v:.0f}%') for v in (0, 0.02, 0.04, 0.06)])
    if R is not None:
        P2.line(list(R.index), list(R.values), GREY, tips=[f'y {y:.0f} mm: {100 * v:.2f} %' for y, v in R.items()])
    P2.text(210, 0.064, 'PMT on top →', 19, MUT, 'end')
    lo_x = {a: t[(t.arm == a) & (t.err < 2)].sort_values('lo') for a in 'AC'}
    shear = ' · '.join(f'{a}: {g.cross.iloc[0]:.1f} mm at {g.lo.iloc[0]:+.0f}…{g.hi.iloc[0]:+.0f}, '
                       f'{g.cross.iloc[-1]:.1f} mm at {g.lo.iloc[-1]:+.0f}…{g.hi.iloc[-1]:+.0f}'
                       for a, g in lo_x.items() if len(g))
    body = (sd.title('y was stored upside down until 14 Oct: liquid B now rises toward its PMT; a ~3° x-vs-height shear',
                     'left: A and C band crossings in bins of reconstructed height, ≥ 12 ms · right: '
                     'scint-stack whole-face late map')
            + sd.row(P.svg('x vs height'), P2.svg('liquid B vs height'), gap=56))
    notes = f"""
<p><b>Shear.</b> Both chambers, both chains (is2_v2 cf and production), show the source x moving with height
({shear}). The same estimator on Geant4's untilted capsule is flat
within ~1 mm, so it is not built in by the estimator. The slope is ≈ 0.05 (≈ 3°) and, since height is smeared by the
resolution, the true slope is at least that. Two readings: the capsule is tilted in the x–y plane (the B side), or
both chambers are rotated in their planes by the same angle; A–C cosmics are blind to the second. The top of the
source is further toward −x (B) than the bottom, the opposite sense to a "bottom furthest out" tilt.</p>
<p><b>Liquid B.</b> The horizontal liquids (A, D) answer 10–20× more on their PMT half. B, vertical with its PMT on top
per the 17–18 July survey, rises steadily toward +y, toward its PMT, by ×2, like A and D. Before the fix the map was
mirrored and it seemed to rise away from its PMT: that was the open "why don't the vertical liquids answer near their
PMT" question of the scint-stack note, closed by the fix. C has too few counts to say.</p>
<p><b>Settled by the wiring (Dylan, 2026-10-13):</b> low FEU channels are at the bottom of each chamber.
<code>mx17_m1_map.csv</code> puts y = 0.78 mm × channel, and each chamber's local y is global +y, so the raw strip
coordinate already points up. <code>build_tracks.IN_PLANE_SIGN_Y = −1</code> then mirrors it. The correct fix in
2026-09 was to flip tan y, not the position. Fixed that way on 13 Oct; every stage-3 table was regenerated on
14 Oct and proved an exact y mirror of the old one, row by row (<code>Y_FLIP_AUDIT.md</code>). The source is at
{N['y_late_A']:+.0f} (A) / {N['y_late_C']:+.0f} (C) mm, at the nose.</p>"""
    D.slide('yflip', body, notes, foot='tilt: capsule_tof tilt (tilt.csv) · liquids: ntof_scint_stack eff_map, liq_full, all_late')


def s_pose(D, A, N):
    B = A['pose_bound']
    late = A['pose_data'][A['pose_data'].window == 'late'].set_index('arm')
    E = A['pose_early']
    Al = A['pose_alt']
    Al = Al[Al.window == 'late'].copy()
    Al['x'] = 0.5 * (Al.x_cross_A + Al.x_cross_C)
    Al['y'] = 0.5 * (Al.y_cross_A + Al.y_cross_C)
    xs = 0.5 * (late.loc['A', 'x_cross'] + late.loc['C', 'x_cross'])
    P = sd.Plot(800, 560, x=(-14, 0), y=(-38, -6), xlabel='source x (mm), frame centre = 0',
                ylabel='source height (mm)', title='which pictures reach the data')
    P.xticks([(v, str(v)) for v in (-14, -12, -10, -8, -6, -4, -2, 0)])
    P.yticks([(v, str(v)) for v in (-35, -30, -25, -20, -15, -10)])
    for case, g in Al.groupby('case', sort=False):
        g = g.sort_values('width_scale')
        on = case.startswith('beam on')
        tail = 'tail-lit' in case
        col = INK if on else (GREY if tail else (PURPLE if '-10.3' in case else ORANGE))
        x = g.x + (xs - g.x.iloc[0] if on else 0.0)
        P.line(list(x), list(g.y), col, tips=[f'{case}, flux ×{w:g}: x {xx:+.1f}, y {yy:+.1f}, ³He share {q:.2f}'
                                               for w, xx, yy, q in zip(g.width_scale, x, g.y, g.gas)])
    for arm in 'AC':
        P.points([late.loc[arm, 'x_cross']], [late.loc[arm, 'y_cross']], ARM_COL[arm], r=11,
                 tips=[f'data {arm}, late: x {late.loc[arm, "x_cross"]:+.1f}, y {late.loc[arm, "y_cross"]:+.1f}'])
    P.text(-13.7, -8, 'beam at 0, capsule off: purple 10.3 mm, orange 13, grey 15–20 (tail-lit)', 16, MUT)
    P.text(-13.7, -36.5, 'black: flux on the capsule (moved to data x), ×1…×10 width', 16, MUT)
    d = E[E.kind == 'data']
    P2 = sd.Plot(760, 560, x=(0.3, 1.9), y=(-14, -4), xlabel='time since the flash (ms, log)',
                 ylabel='source x (mm)', title='source x against ToF')
    P2.xticks([(np.log10(v), str(v)) for v in (2, 5, 10, 20, 50)])
    P2.yticks([(v, str(v)) for v in (-14, -12, -10, -8, -6, -4)])
    mix = E[(E.kind == 'mix') & (E.arm == 'A')].set_index('frac_at_plus10').peak_x
    P2.hline(float(mix.loc[0.2]), MUT, label='20 % from a source 10 mm toward 0')
    P2.hline(float(mix.loc[0.5]), MUT, label='50 %')
    for arm in 'AC':
        g = d[d.arm == arm]
        t = np.log10(np.sqrt(g.lo * g.hi))
        P2.line(list(t), list(g.peak_x), ARM_COL[arm],
                tips=[f'{arm} {lo:g}–{hi:g} ms: x {v:+.1f} ± {e:.1f}' for lo, hi, v, e in zip(g.lo, g.hi, g.peak_x, g.peak_err)])
        errbars(P2, list(t), list(g.peak_x), list(g.peak_err), ARM_COL[arm])
    b5 = B.loc[5]
    body = (sd.title('One compact source on the capsule nose, ~10 mm off the frame centre, at every ToF',
                     'capsule_pose: ray trace (EAR2 flux × Lambda2D per energy) × Geant4 tracks, +y up · '
                     'beam off the frame, or capsule off the beam?')
            + sd.row(P.svg('pictures'), P2.svg('x vs ToF'), gap=56))
    notes = f"""
<p><b>Measured.</b> The tracks image one compact source at x {xs:+.1f}, z ≈ −3.6 mm in the chamber frame, at the
nose's height ({late.loc['A', 'y_cross']:+.1f} A / {late.loc['C', 'y_cross']:+.1f} C). Its x is the same from 2 to 75 ms
(the pointing-excess peak, background-robust where the band crossing is not), in every run, and before and after the
27 July access. A and C agree although their u axes point opposite ways, so only a <i>common</i> shift of the pair
against the beam could fake it.</p>
<p><b>What it does not say.</b> The source is where the illuminating neutrons meet Al, not the beam. A compact source
at the nose needs a narrow flux on the nose (within ~{b5.d_tip_max:.0f} mm, capsule within ±5 mm of its drawn height);
grazing lights the 0.6 mm barrel and lifts the source 10–20 mm. A Lambda2D beam centred on the frame reaches x ≈ −10
only with the capsule 13–18 mm off, a flux ×10 wider, or the tail alone, and then puts the source 14–25 mm too high
(capsule that much lower than drawn; 1–15 % of the beam on the ³He). The direct-beam window shows no source near
x = 0 (≤ 10–34 % at 4–6 ms, ≤ 5 % at 6–12 ms), but the share of direct-beam captures there is unknown: the
late, after-the-flux population images the same spot.</p>
<p><b>Two readings.</b> (1) The flux sits on the capsule, ~10 mm toward B from the frame centre: the beam is not
where the frame says, or the frame is not where it was aligned (≥ {b5.gas_min:.0%} of the beam then enters the ³He).
(2) The beam is centred as aligned and the capsule is far off it and lower than drawn (³He yield 1–15 %). Needs a survey
or photo of capsule vs beam, the tip height, and the beam spot at the station. Neither explains the 1 : 8 alone.
Report: <code>&lt;out&gt;/capsule_pose/report.html</code>.</p>"""
    D.slide('pose', body, notes, foot='capsule_pose early.csv, alternatives.csv, halo.csv, bound.csv')


def s_expect(D, A, N):
    S = A['sep_loss']
    rows = []
    for arm in 'AC':
        g = S[(S.arm == arm) & (S.open_lo >= 0) & (S.open_hi <= 60)]
        for lo, hi, a, b_, n in zip(g.open_lo, g.open_hi, g.lt20, g.b20_40, g.n):
            rows.append((arm, lo, hi, a, b_, n))
    P = sd.Plot(820, 520, x=(0, 60), y=(0, 1), xlabel='true opening angle (°)',
                ylabel='share of accepted e⁺e⁻ pairs', title='how close the two legs are in the gap')
    P.xticks([(v, str(v)) for v in (0, 15, 30, 45, 60)])
    P.yticks(PCT_TICKS := [(v / 100, f'{v}%') for v in (0, 25, 50, 75, 100)])
    for arm, off in (('A', -2.5), ('C', 2.5)):
        for _, lo, hi, a, b_, n in [r for r in rows if r[0] == arm]:
            x = 0.5 * (lo + hi) + off
            P.vbar(x, a, 34, ARM_COL[arm], tip=f'{arm} {lo:g}-{hi:g}°: {100 * a:.0f} % within 20 mm in a view (n = {n})')
            P.vbar(x, a + b_, 34, ARM_COL[arm], base=a, opacity=0.35,
                   tip=f'{arm} {lo:g}-{hi:g}°: {100 * b_:.0f} % at 20-40 mm (lost by the one-sided search, kept by is2_v2)')
    e = N['exp']
    sep = N['sep']
    flow = sd.flow([
        dict(label='Geant4 pairs', sub='true opening angles, energies, positions', color=PURPLE,
             tip='events with >= 2 track-like leptons in the gap, |tan| < 0.6, per tracked single'),
        dict(label='real reconstruction', sub=f'P(2|2) {e.loc["A", "eps"] * e.loc["A", "f_two"] * e.loc["A", "eps"]:.2f} / '
                                               f'{e.loc["C", "eps"] * e.loc["C", "f_two"] * e.loc["C", "eps"]:.2f}',
             color=GOLD, tip='g4_digi: Geant4 steps digitised into quiet run_145 triggers, reconstructed; '
                             'includes every close-pair loss of that pass'),
        dict(label='predicted', sub=f'{100 * e.loc["A", "predicted_frac"]:.2f} % / {100 * e.loc["C", "predicted_frac"]:.2f} %',
             color=RED),
        dict(label='seen', sub=f'{100 * e.loc["A", "data_frac"]:.2f} % / {100 * e.loc["C", "data_frac"]:.2f} %', color=BLUE),
    ], size=22)
    col = sd.col(
        sd.callout(f'Yes, both are in. {sd.pct(sep.loc["A", "lt20"])} (A) / {sd.pct(sep.loc["C", "lt20"])} (C) of '
                   'Geant4 e⁺e⁻ pairs are within 20 mm in one view. The prediction does not assume they are found: '
                   'it pushes the Geant4 events through the real reconstruction.', BLUE, 24),
        sd.callout(f'That response is from the one-sided search, which lost pairs out to ~40 mm. Another '
                   f'{sd.pct(sep.loc["A", "b20_40"])} / {sd.pct(sep.loc["C", "b20_40"])} sit at 20–40 mm, '
                   'which is2_v2 keeps. Updating it raises the prediction, so data/Geant4 gets worse.', RED, 24),
        w=620, gap=20)
    body = (sd.title('The expected count already pays for the opening angle and the close-pair loss',
                     'bars: Geant4 e⁺e⁻ in acceptance, nearest view at mid-gap · solid < 20 mm, pale 20–40 mm')
            + sd.row(sd.col(P.svg('separation vs opening'), sd.legend([('A', BLUE, 'box'), ('C', ORANGE, 'box')], size=22),
                            gap=4, w=820), col, gap=48) + flow)
    notes = f"""
<p><b>The direct route</b> (<code>same_chamber_compare.expectation</code>, "digitised Geant4 through the
reconstruction"): r₂ = Geant4 events with ≥ 2 leptons in the gap per event with ≥ 1, times P(2 tracks | 2 leptons) /
P(≥ 1 | 1) measured on g4_digi (A {N['exp'].loc['A','eps']:.2f}, C {N['exp'].loc['C','eps']:.2f} for one lepton).
Because the digitised events carry Geant4's own opening angles and separations, the close-pair loss and the
opening-angle distribution are in the prediction, as they were in that reconstruction pass. Data/Geant4 =
{N['exp'].loc['A','data_over_g4']:.2f} (A), {N['exp'].loc['C','data_over_g4']:.2f} (C).</p>
<p><b>The bracket routes</b> use the measured per-view two-track efficiency of is2_v2 cf
(<code>two_track_efficiency_axis.csv</code>) on Geant4's true separations; they give 0.12–0.52.</p>
<p><b>Next iteration.</b> Getting below 20 mm (T1's joint fit, off in this pass) would add up to ~45 % more pairs in
both data and prediction. It raises the statistics; it does not by itself change the ratio. Re-digitising with is2_v2
is the step that fixes the prediction's reconstruction, and it should be done before the next comparison.</p>"""
    D.slide('expect', body, notes, foot='sep_loss.csv (same_chamber_scatter) · cmp_expectation.csv (same_chamber_compare)')


def s_opening(D, A, N):
    O = A['cmp_opening']
    C = A['cmp_opening_chi2'].set_index(['arm', 'chain', 'model'])
    panels = []
    for arm, w in (('A', 1000), ('C', 620)):
        o = O[(O.arm == arm) & (O.chain == 'cf') & (O.mid < 70)]
        k = o.correlated.to_numpy()
        tot = k.clip(min=0).sum()
        ymax = 0.40
        P = sd.Plot(w, 600 if arm == 'A' else 480, x=(0, 70), y=(0, ymax), xlabel='opening angle (°)',
                    ylabel='fraction per 10°' if arm == 'A' else '',
                    title=f'chamber {arm}' + (' (the comparison asked for)' if arm == 'A' else ' (for reference)'))
        P.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40, 50, 60, 70)])
        P.yticks([(v, f'{100 * v:.0f}%') for v in (0, 0.1, 0.2, 0.3, 0.4)])
        edges = list(np.arange(0, 80, 10))
        for model, color, dash, name in (('g4_ee', PURPLE, None, 'Geant4 e⁺e⁻'), ('g4_all', PURPLE, '10 7', 'Geant4 all pairs'),
                                         ('mixed', GREY, '4 6', 'unrelated tracks (event-mixed)')):
            m = o[model].to_numpy(float)
            v = m / m.sum()
            xs, ys = sd.step_xy(edges[:len(v) + 1], list(v))
            c2 = C.loc[(arm, 'cf', model)]
            P.fore.append(sd.poly([P.X(x) for x in xs], [P.Y(y) for y in ys], color, 3.5, dash,
                                  tip=f'{name}: χ² {c2.chi2:.1f} / {int(c2.ndf)} against the data'))
        P.points(list(o.mid), list(k / tot), ARM_COL[arm], r=8,
                 tips=[f'{arm} {m - 5:g}-{m + 5:g}°: correlated {c:.0f} ± {e:.0f} (real {r}, accidental {a:.0f})'
                       for m, c, e, r, a in zip(o.mid, k, o.err, o.real, o.accidental)])
        errbars(P, list(o.mid), list(k / tot), list(o.err / tot), ARM_COL[arm])
        panels.append(P.svg(f'opening angle {arm}'))
    cA = {m: C.loc[('A', 'cf', m)] for m in ('g4_ee', 'g4_all', 'mixed')}
    leg = sd.legend([('data, correlated (late real − accidental share × mixed)', BLUE, 'dot'),
                     (f'Geant4 e⁺e⁻ · A χ² {cA["g4_ee"].chi2:.1f}/{int(cA["g4_ee"].ndf)}', PURPLE),
                     (f'Geant4 all pairs · {cA["g4_all"].chi2:.1f}/{int(cA["g4_all"].ndf)}', PURPLE, 'dash'),
                     (f'unrelated tracks · {cA["mixed"].chi2:.1f}/{int(cA["mixed"].ndf)}', GREY, 'dash')], size=21)
    body = (sd.title('Chamber A\'s correlated pairs prefer Geant4\'s opening-angle shape over that of unrelated tracks',
                     'tracks ≥ 12 ms, is2_v2 at the capsule-focus k · shapes only, each normalised to 1 · '
                     'Geant4 through the measured response (smearing, |tan| < 0.6, two-track efficiency)')
            + sd.row(panels[0], sd.col(panels[1], leg, gap=20), gap=44))
    notes = f"""
<p><b>Data.</b> Late real same-chamber pairs minus the accidental share (from the rate fit, A
{100 * A['cmp_tof_fit'].set_index('arm').loc['A', 'accidental_share']:.0f} %) times the late event-mixed shape.
Errors include the subtraction.</p>
<p><b>Models.</b> Geant4 e⁺e⁻ = one photon's conversion pair; "all pairs" adds Compton + delta ray and the rest (75–78 %
of it is e⁺e⁻ anyway); "unrelated" = event-mixed late tracks. Geant4 pairs are folded through the measured
resolution, |tan| &lt; 0.6 and the per-view two-track efficiency (<code>same_chamber_g4.response</code>), so the
dip below ~10° is the close-pair loss, in both.</p>
<p><b>Reading.</b> A: Geant4 e⁺e⁻ χ² {cA['g4_ee'].chi2:.1f}/{int(cA['g4_ee'].ndf)}, all pairs
{cA['g4_all'].chi2:.1f}/{int(cA['g4_all'].ndf)}, unrelated {cA['mixed'].chi2:.1f}/{int(cA['mixed'].ndf)}. The
difference comes mostly from the 0–10° and 40–50° bins, so it is a preference, not a separation; the Geant4 histogram
has only ~60 weighted pairs per arm and its own error is in the χ². C does not discriminate (its Geant4 χ² is
the worse one). Medians: data A {C.loc[('A','cf','g4_ee')].median_corr:.0f}°, Geant4
{C.loc[('A','cf','g4_ee')].median_model:.0f}°. A global y flip would not move any of this: opening angles are
mirror-invariant.</p>"""
    D.slide('opening', body, notes, foot='cmp_opening.csv / cmp_opening_chi2.csv (same_chamber_compare, chain cf)')


def s_close(D, A, N):
    li = lambda t: f'<li style="margin-bottom:14px">{t}</li>'
    left = ('<ul style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li('The 1 : 8 rate is still unexplained. The late data are as capsule-pure as Geant4, the expectation '
                 'already pays for opening angle and close-pair loss, and the transverse offset costs percent.')
            + li('The x-vs-height step (~4 mm) is a ~3.6σ hint of a second, higher source; a 10× Geant4 '
                 'digitised sample would settle it.')
            + li('Late capsule captures are delayed neutrons, not the simulated beam. If they land elsewhere on '
                 'the capsule than beam neutrons do, the Al thickness around them, and so the conversion yield '
                 'per trigger, differs from Geant4.')
            + li('A busy-event two-track loss in data that the quiet digitisation lacks is still open.') + '</ul>')
    right = ('<ol style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:32px">'
             + li('Re-digitise the Geant4 pairs with is2_v2 (two-sided search, t0 ≤ 300, |tan| &lt; 0.6 on both legs).')
             + li('Geant4 with delayed thermal neutrons on the capsule (isotropic or from below) to get the '
                  'pairs-per-trigger of the late population.')
             + li('Settle where the beam is: the capsule images ~10 mm off the frame centre at every ToF. '
                  'Survey/photo of capsule vs beam axis, the capsule tip height, and the beam spot at the station.') + '</ol>')
    body = (sd.title('<span style="color:#eef0f3">What this does not rule out, and what to do next</span>')
            + sd.row(sd.col(sd.p('not ruled out', 26, DMUT), left, w=820),
                     sd.col(sd.p('next', 26, DMUT), right, w=780), gap=56))
    D.slide('close', body, '', dark=True)


def build(out: Path) -> Path:
    A = load()
    N = numbers(A)
    D = sd.Deck('Same-chamber pairs vs Geant4',
                'time window, vertexing, capsule position and opening angle of the same-chamber pairs')
    for f in (s_cover, s_tof, s_early, s_scatter, s_vertex, s_capsule, s_yflip, s_pose, s_expect, s_opening,
              s_close):
        f(D, A, N)
    out.parent.mkdir(parents=True, exist_ok=True)
    D.write(out, note_meta=dict(
        title='Same-chamber pairs vs Geant4: time window, vertexing, capsule position',
        summary='Why the < 12 ms window is dropped although the beam captures there; why aluminium e+e- pairs '
                'cannot vertex and X17 pairs can; the capsule offset and the y orientation (fixed 14 Oct); '
                'chamber A opening angle vs Geant4.',
        tags='X17,same-chamber pairs,Geant4', date=dt.date.today().isoformat()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    print('wrote', build(a.out or SC / 'deck' / 'same-chamber-pairs.html'))


if __name__ == '__main__':
    main()
