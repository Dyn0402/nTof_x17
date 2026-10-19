#!/usr/bin/env python3
"""
make_status_oct15_deck.py -- status update for the 2026-10-15 talk: what has
changed since the Athens meeting, and whether the n_TOF set-up saw what it was
expected to see (the ²⁷Al capture pairs, the single tracks), as the input to
"can the set-up be repeated at LNL".

Reads only the analyses' own tables, and reuses slides of the decks that
already answer a question (same-chamber pairs, calorimetry):
  <out>/status_oct15/{penetration,c2_ladder,leg_spectra,singles_origin}.csv
                                                    leg_penetration.py
  <out>/same_chamber/cmp_*.csv, ...                 same_chamber_compare / _scatter
  ntof_athens_26/data/thermal_accounting/F*.csv     the Athens funnel (Geant4)
  calorimetry tables                                ntof_calorimetry

    python -m sept26_prelim_analysis.leg_penetration
    python -m sept26_prelim_analysis.make_status_oct15_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <html> --slug x17-status-2026-10-15 --force --deploy
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
                      INK, MUT, DBLUE, DRED, DGREEN, DMUT)
from sept26_prelim_analysis import paths                               # noqa: E402
from sept26_prelim_analysis import make_same_chamber_deck as SCD       # noqa: E402

ST = paths.out('status_oct15')
SC = paths.out('same_chamber')
ACC = REPO / 'ntof_athens_26' / 'data' / 'thermal_accounting'
ARM_COL = dict(A=BLUE, C=ORANGE)
AL_C, X17_C = RED, GREEN
UP_LOSS = 0.6        # MeV, capsule -> gap, per leg (leg_penetration.UP_LOSS_TO_GAP)


def rd(d, name):
    return pd.read_csv(d / f'{name}.csv')


# --------------------------------------------------------------------------- #
def load():
    A = SCD.load()
    for n in ('penetration', 'c2_ladder', 'leg_spectra', 'singles_origin'):
        A[n] = rd(ST, n)
    for n in ('cmp_g4_classes', 'cmp_wall_double_ratio'):
        A[n] = rd(SC, n)
    A['F'] = pd.concat([rd(ACC, f'F{i}') for i in range(1, 6)], ignore_index=True).set_index('id')
    return A


def numbers(A):
    N = SCD.numbers(A)
    P = A['penetration']

    def pen(pop, kind):
        r = P[(P.population == pop) & (P.kind == kind)]
        return r.iloc[0] if len(r) else None
    N['al_l_e'] = pen('Al conversion lepton', 'per lepton, energy only (aimed at a bar)')
    N['x_l_e'] = pen('X17 lepton', 'per lepton, energy only (aimed at a bar)')
    N['al_l_g'] = pen('Al conversion lepton', 'per lepton')
    N['x_l_g'] = pen('X17 lepton', 'per lepton')
    N['al_p_e'] = pen('Al e⁺e⁻ pair (same chamber)', 'both legs, energy only (aimed at a bar)')
    N['x_p_e'] = pen('X17 pair (opposite chambers)', 'both legs, energy only (aimed at a bar)')
    N['al_p_g'] = pen('Al e⁺e⁻ pair (same chamber)', 'both legs')
    N['x_p_g'] = pen('X17 pair (opposite chambers)', 'both legs')
    N['al_trig'] = pen('Al e⁺e⁻ pair (same chamber)', 'arm can trigger (sum of legs)')
    N['x_trig'] = pen('X17 pair (opposite chambers)', 'either arm can trigger')
    E = A['cmp_expectation']
    dig = E[E.eps_source.str.startswith('digitised')].set_index('arm')
    alls = E[(E.eps_source == 'all singles') & (E.two_track == 'no two-track loss')].set_index('arm')
    N['singles_ratio'] = {a: float(alls.loc[a, 'eps'] / dig.loc[a, 'eps']) for a in 'AC'}
    N['singles_eps_data'] = {a: float(alls.loc[a, 'eps']) for a in 'AC'}
    N['singles_eps_dig'] = {a: float(dig.loc[a, 'eps']) for a in 'AC'}
    N['d_over_g4'] = {a: float(dig.loc[a, 'data_over_g4']) for a in 'AC'}
    br = E[E.eps_source != 'digitised Geant4 through the reconstruction']
    N['d_over_g4_bracket'] = (float(br.data_over_g4.min()), float(br.data_over_g4.max()))
    T = A['cmp_tof_fit'].set_index('arm')
    N['a'] = {a: (float(T.loc[a, 'a']), float(T.loc[a, 'a_err'])) for a in 'AC'}
    N['corr_pairs'] = {a: float(T.loc[a, 'correlated_pairs']) for a in 'AC'}
    N['acc_share'] = {a: float(T.loc[a, 'accidental_share']) for a in 'AC'}
    C = A['cmp_opening_chi2'].set_index(['arm', 'chain', 'model'])
    N['chi'] = {(a, m): C.loc[(a, 'cf', m)] for a in 'AC' for m in ('g4_ee', 'g4_all', 'mixed')}
    G = A['cmp_g4_classes']
    for a in 'AC':
        g = G[(G.arm == a) & (G.cls != '(singles in acceptance)')]
        N[f'ee_share_{a}'] = float(g[g.cls == 'e+e- (one gamma)'].acc.sum() / g.acc.sum())
    W = A['cmp_wall_double_ratio']
    N['wdr'] = W[W.chain == 'cf'].set_index('arm')
    return N


# --------------------------------------------------------------------------- #
def s_cover(D, A, N):
    rA, rC = N['singles_ratio']['A'], N['singles_ratio']['C']
    body = (sd.kicker(f'n_TOF EAR2 · X17 · status since Athens · {dt.date.today():%-d %b %Y}')
            + sd.title('<span style="color:#eef0f3">The chambers see the aluminium-capture γ as Geant4 predicts; '
                       'the Al e⁺e⁻ pairs are there in shape, at ⅛ of the rate</span>')
            + sd.row(
                sd.bignum(f'×{rA:.2f} / ×{rC:.2f}', 'single tracks per trigger, data over expectation (A / C)',
                          DGREEN, 'Geant4 → digitised → the real reconstruction; they image one compact source on '
                                  'the capsule nose',
                          tip=f'data tracks per trigger / Geant4 singles per trigger = {N["singles_eps_data"]["A"]:.2f} '
                              f'(A), {N["singles_eps_data"]["C"]:.2f} (C); the digitised reconstruction predicts '
                              f'{N["singles_eps_dig"]["A"]:.2f} / {N["singles_eps_dig"]["C"]:.2f}'),
                sd.bignum(f'{100 * N["a"]["A"][0]:.2f} %', 'of late events carry a correlated second track',
                          DBLUE, 'flat in time while the rate falls ×3.5 (A and C agree); Geant4 says '
                                 f'{sd.pct(N["ee_share_A"])} of such pairs are e⁺e⁻',
                          tip=f'a = {100 * N["a"]["A"][0]:.3f} ± {100 * N["a"]["A"][1]:.3f} % (A), '
                              f'{100 * N["a"]["C"][0]:.3f} ± {100 * N["a"]["C"][1]:.3f} % (C); '
                              f'~{N["corr_pairs"]["A"]:,.0f} / {N["corr_pairs"]["C"]:,.0f} correlated pairs'),
                sd.bignum(f'{N["d_over_g4"]["A"]:.2f}', 'pairs seen / pairs expected', DRED,
                          f'A {N["d_over_g4"]["A"]:.2f}, C {N["d_over_g4"]["C"]:.2f}; bracket '
                          f'{N["d_over_g4_bracket"][0]:.2f}–{N["d_over_g4_bracket"][1]:.2f}. Pair-specific: '
                          'the singles agree. Unexplained.'),
                sd.bignum(f'{sd.pct(N["x_l_e"].liquid)} vs {sd.pct(N["al_l_e"].liquid, 1)}',
                          'of leptons reach the liquid: X17 vs Al', GOLD,
                          'X17 legs (~4–16 MeV) cross the whole stack; Al legs (~2 MeV at the gap) stop in it',
                          size=60),
                gap=40))
    notes = """
<p>Status for the 15 October talk. The question behind it: did the n_TOF set-up see what it was built to see, so
that repeating it at LNL is realistic?</p>
<ol>
<li>What changed since Athens (slide 2) and what Geant4 said we should see (slide 3).</li>
<li>Did we see the ²⁷Al pairs, and do they look like Al? (slides 4–8)</li>
<li>X17 legs vs Al legs: penetration and what it does to two-track statistics (slides 9–11).</li>
<li>Where the single tracks come from (slides 12–13).</li>
<li>Calorimetry and MM dE/dx (slides 14–15). Verdict and LNL (slide 16).</li>
</ol>
<p>Data: the is2_v2 re-pass at the capsule-focus k (stage3_is2_v2_cf, y up since 14 Oct), chambers A and C,
tracks ≥ 12 ms after the flash, t0 ≤ 300 ns, |tan| &lt; 0.6, production period (run_69 onward, the noisy-DREAM side
of 23 July). Hover the dotted terms and the data points for values, counts and provenance.</p>"""
    D.slide('cover', body, notes, dark=True, short='Answer')


def s_since(D, A, N):
    cards = [
        ('y now points up', '14 Oct', GOLD,
         'Stage 3 had stored y mirrored since 7 Sep. Confirmed by the wiring, cosmic-muon timing (p = 4×10⁻¹¹) and the '
         'plastic PMT timing; every live stage-3 set regenerated. The source now sits on the capsule nose, where Geant4 '
         'puts it.'),
        ('A calibrated re-pass (is2_v2)', '9–12 Oct', BLUE,
         'In-situ cosmic calibration (drift velocity, kernel), the 3-strip seeder (+40–52 % confirmed tracks) and a '
         'two-sided slope search. Wall purity A 0.38 → 0.59, C 0.49 → 0.66; it drops junk pairs and keeps the real ones.'),
        ('Beam-off cosmics as truth', '2–8 Oct', GREEN,
         '41 runs, 57.6 h, put on the n_TOF clock. The A–C muon line calibrates the angle scale; the in-beam '
         'muons read it right to 1–4 %. The beam/cosmic wall gap is electron scattering, not the reconstruction.'),
        ('Geant4 through the real reco', '7–13 Oct', PURPLE,
         'g4_digi injects Geant4 charge into real quiet triggers and runs the production reconstruction, so data and '
         'simulation are compared after the same losses.'),
        ('Same-chamber pairs vs Geant4', '12–13 Oct', RED,
         'Correlated pairs found (0.15 % of late events), shape compatible with Al e⁺e⁻, rate ⅛ of Geant4. The '
         '12 Oct "vertex excess" was an early-time artefact and is retracted.'),
        ('Calorimetry closed', '8 Oct', GREY,
         'Plastic scale fixed on muons; liquids threshold-limited; MM dE/dx and the soft-leg energy discriminant '
         'killed by their own criteria.'),
    ]
    grid = []
    for i in range(0, 6, 3):
        grid.append(sd.row(*[sd.card(sd.p(t, 28, c, 600) + sd.p(d, 21, MUT) + sd.p(x, 22), tip=f'{t} ({d})', pad=26)
                             for t, d, c, x in cards[i:i + 3]], gap=28))
    body = (sd.title('Since Athens: one calibrated re-pass, cosmic truth, and Geant4 through the real reconstruction',
                     'the work between the Athens meeting (end of September) and 15 October')
            + sd.col(*grid, gap=28))
    notes = """
<p><b>Records.</b> y fix: <code>COORDINATES.md</code>, <code>sept26_prelim_analysis/Y_FLIP_AUDIT.md</code>.
Re-pass: <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §17–22. Cosmics: <code>ntof_cosmics/README.md</code>.
Same-chamber: <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code>. Calorimetry: <code>ntof_calorimetry/PLAN.md</code>.</p>
<p><b>Still provisional:</b> the capsule-focus k (is2_v2 × 1.12) used for every pair angle here. In-beam muons prefer
the plain is2_v2 scale; the 12 % tension is open, and every pair result has been checked at both k (it does not move
the conclusions).</p>
<p>Published slide notes with the detail: <code>same-chamber-pairs</code>, <code>calorimetry</code>,
<code>two-track-limit</code>, <code>scint-stack</code> on dylan-neff.web.cern.ch/notes.</p>"""
    D.slide('since', body, notes, short='Since Athens')


def s_funnel(D, A, N):
    F = A['F']
    f = lambda i: float(F.loc[i, 'frac_of_parent'])
    per30 = float(F.loc['F4.external.al27', 'per30d_ratetable'] + F.loc['F4.internal.al27', 'per30d_ratetable'])
    flow = sd.flow([
        dict(label='neutron enters the capsule', sub='E < 2 eV', color=INK,
             tip=f'{F.loc["root", "mc_count"]:,.0f} of 10⁹ simulated thermal neutrons (gun r < 11.5 mm)'),
        dict(label=f'³He(n,p) {sd.pct(f("F1.he3_np"))}', sub=f'(n,γ) {sd.pct(f("F1.ncapture"), 2)}', color=GREY,
             tip='p + t stop in the gas: invisible. (n,γ) anywhere.'),
        dict(label=f'Al vessel {sd.pct(f("F1.ncapture.wall_al"))}', sub='of the captures; 7.72 MeV + cascades',
             color=RED, tip=f'CFRP {sd.pct(f("F1.ncapture.wall_cfrp"), 1)}, scattered out and captured elsewhere '
                            f'{sd.pct(f("F1.ncapture.elsewhere"))} (87 % in the scintillators)'),
        dict(label=f'charge in a gap {sd.pct(f("F2.charged_in_detector"), 1)}', sub='per capture, prompt',
             color=GOLD, tip='≥ 1 charged particle ≥ 1 keV in a Micromegas drift gap'),
        dict(label=f'Compton e⁻ {sd.pct(f("F3.compton_near_capsule"))}', sub=f'e⁺e⁻ {sd.pct(f("F3.pair_near_capsule"))}'
             ' · made in the capsule', color=BLUE, tip='the rest is made in the chamber/air (19 %) or other (1 %)'),
    ], size=23)
    trig = sd.hbars([
        ('Compton e⁻, capsule γ', f('F5.compton_capsule_capture'), BLUE,
         f'{F.loc["F5.compton_capsule_capture", "per_pulse"]:.0f} legs per pulse; 99.8 % of them from ²⁷Al'),
        ('e⁺e⁻ made in the capsule', f('F5.pair_near_capsule'), RED,
         f'{F.loc["F5.pair_near_capsule", "per_pulse"]:.1f} legs per pulse; 99.9 % ²⁷Al'),
        ('γ converting in the structure', f('F5.conv_structure_capsule_capture'), ORANGE,
         'capsule-capture γ converting in a chamber (69 %) or scintillator (25 %)'),
        ('capture outside the capsule', f('F5.capture_outside_capsule'), GREY, 'scattered neutrons'),
        ('mixed', f('F5.mixed'), GREY, ''),
    ], 1.0, width=520, label_w=380, fmt=lambda v: sd.pct(v, 1), size=24)
    body = (sd.title('What Geant4 told Athens to expect: the trigger is the Al γ, and one leg in ten is an e⁺e⁻',
                     'Geant4 thermal campaign (10⁹ n, nose-first capsule) · the Athens funnel, F1–F5')
            + flow
            + sd.row(sd.col(sd.p('what makes a trigger leg (wall ∧ plastic)', 24, MUT), trig, w=980, gap=14),
                     sd.col(sd.callout(f'Athens: ≈ 10⁸ ({per30 / 1e8:.1f}×10⁸) ²⁷Al e⁺e⁻ pairs made per 30 days, '
                                       'a demonstrator for the method. That is pairs <i>made</i>; the next slides '
                                       'ask how many we could see.', RED, 24),
                            sd.callout('Every number here is a Geant4 prediction. The test is whether the data '
                                       'reproduce the singles (the Compton electrons) and the pairs.', INK, 24),
                            w=600, gap=24), gap=60))
    notes = f"""
<p>From <code>ntof_athens_26/data/thermal_accounting/</code> (F1.csv–F5.csv, the Athens contract;
<code>MX17_Full_Geant/scripts/thermal_accounting.py</code>, condor 4278922). Per neutron entering the capsule, E &lt; 2 eV.</p>
<p>Al pairs per 30 days at the rate-table flux: external conversion {F.loc['F4.external.al27', 'per30d_ratetable']:.2e},
internal pair creation (analytic) {F.loc['F4.internal.al27', 'per30d_ratetable']:.2e}. Trigger legs per pulse:
{F.loc['F5.legs', 'per_pulse']:.0f}.</p>
<p>Two-arm events (a leg in each of two arms): 0.11 per pulse, 92 % two γ of one capture, so the facing-chamber pair
sample is mostly two Comptons, not e⁺e⁻. The Al e⁺e⁻ live in the <b>same</b> chamber (median opening ~25°).</p>"""
    D.slide('funnel', body, notes, short='Expected',
            foot='Geant4 neutrons_thermal_trig_2cm_nose · thermal_accounting.py · numbers as shown in Athens')


def s_scorecard(D, A, N):
    ok = f'<span style="color:{GREEN};font-weight:600">✓ as expected</span>'
    cA, cAm = N['chi'][('A', 'g4_ee')], N['chi'][('A', 'mixed')]
    w = N['wdr']
    rows = [
        ['single tracks per trigger', f'Geant4 × digitised reco', f'×{N["singles_ratio"]["A"]:.2f} (A) / ×{N["singles_ratio"]["C"]:.2f} (C)', ok],
        ['where they come from', 'one source on the capsule nose', 'compact source on the nose, 2–75 ms', ok],
        ['a correlated same-chamber 2nd track', 'yes, rate-independent', f'{100 * N["a"]["A"][0]:.2f} % (A) / {100 * N["a"]["C"][0]:.2f} % (C), flat', ok],
        ['its opening angle (A)', 'Geant4 e⁺e⁻ shape', f'χ² {cA.chi2:.1f}/{int(cA.ndf)} vs unrelated {cAm.chi2:.0f}/{int(cAm.ndf)}',
         f'<span style="color:{GOLD};font-weight:600">~ prefers e⁺e⁻</span>'],
        ['both legs reach the wall', f'{w.loc["A", "g4_double_ratio"]:.2f}–{w.loc["C", "g4_double_ratio"]:.2f} (double ratio)',
         f'{w.loc["A", "data_double_ratio"]:.2f} / {w.loc["C", "data_double_ratio"]:.2f}',
         f'<span style="color:{GOLD};font-weight:600">~ agrees, not a test</span>'],
        ['pairs point back to the capsule', 'no: 2 % pass with perfect tracks', 'no excess late', f'<span style="color:{GOLD};font-weight:600">~ as expected</span>'],
        ['pair rate', 'Geant4 × digitised reco', f'{N["d_over_g4"]["A"]:.2f} (A) / {N["d_over_g4"]["C"]:.2f} (C) of it',
         f'<span style="color:{RED};font-weight:600">✗ factor ~8 low</span>'],
    ]
    tbl = sd.table(['observable', 'Geant4 expectation', 'data', 'verdict'], rows, size=24,
                   widths=[420, 440, 460, 300], align=['left', 'left', 'left', 'left'])
    body = (sd.title('Do the pairs look like aluminium? In shape yes; in rate, an eighth',
                     'every test made so far, data against Geant4 pushed through the same reconstruction')
            + tbl
            + sd.callout('Everything that depends on single tracks agrees. The one failure is pair-specific: '
                         'whatever removes 7 of 8 pairs leaves the singles alone.', BLUE, 26))
    notes = f"""
<p><b>Singles.</b> Data tracks per trigger divided by Geant4 track-like singles in acceptance per trigger gives an
effective efficiency {N['singles_eps_data']['A']:.2f} (A) / {N['singles_eps_data']['C']:.2f} (C). The digitised
reconstruction of the same Geant4 events gives {N['singles_eps_dig']['A']:.2f} / {N['singles_eps_dig']['C']:.2f}.
The data/expectation ratio is the cover's ×{N['singles_ratio']['A']:.2f} / ×{N['singles_ratio']['C']:.2f}.</p>
<p><b>Pairs.</b> The prediction is the Geant4 ratio of events with two reconstructable leptons to events with one,
times the measured P(2 tracks | 2 leptons) / P(≥ 1 | 1) from the digitised reconstruction, so the opening angle and
the close-pair loss are in it. The brackets (other efficiency routes) span {N['d_over_g4_bracket'][0]:.2f}–
{N['d_over_g4_bracket'][1]:.2f}.</p>
<p><b>Why the wall double ratio is not a test.</b> P(both legs on the wall | pair) / P(one | single)² is ~0.55–0.6
for event-mixed (unrelated) pairs too.</p>
<p>Source tables: <code>{SC}/cmp_*.csv</code> (<code>same_chamber_compare.py</code>).</p>"""
    D.slide('scorecard', body, notes, short='Scorecard',
            foot='is2_v2 re-pass, capsule-focus k, tracks ≥ 12 ms · Geant4 via same_arm_pairs.py + g4_digi')


def s_correlated(D, A, N):
    T = A['cmp_tof_fraction']
    T = T[(T.lo >= 4) & (T.frac <= 0.012)]
    P = sd.Plot(1040, 600, x=(0, 76), y=(0, 0.012), xlabel='time since the γ flash (ms)',
                ylabel='events with a 2nd same-chamber track')
    P.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40, 50, 60, 70)])
    P.yticks([(v, f'{100 * v:.1f}%') for v in (0, 0.003, 0.006, 0.009, 0.012)])
    for arm in 'AC':
        g = T[T.arm == arm]
        m = 0.5 * (g.lo + g.hi)
        P.line(list(m), list(g.frac), ARM_COL[arm], w=2.5, r=6,
               tips=[f'{arm} {lo:g}-{hi:g} ms: {100 * f:.3f} ± {100 * e:.3f} % ({n2:,} pairs in {n1:,} events)'
                     for lo, hi, f, e, n1, n2 in zip(g.lo, g.hi, g.frac, g.err, g.n1, g.n2)])
        a, e = N['a'][arm]
        P.hline(a, ARM_COL[arm], dash='4 6', label=(f'rate-independent part a = {100 * N["a"]["A"][0]:.2f} (A) / {100 * a:.2f} % (C)'
                                                     if arm == 'C' else None), anchor='end', where='below',
                tip=f'{arm}: fit f = a + b·rate over 12-75 ms: a = {100 * a:.3f} ± {100 * e:.3f} %')
    P.vline(12, RED, label='12 ms', tip='pair work uses tracks >= 12 ms: before, flash and epithermal pile-up')
    side = sd.col(
        sd.callout('Before 12 ms the fraction is pile-up: it follows the rate.', GREY, 24),
        sd.callout(f'After 12 ms the rate falls ×3.5 and the fraction flattens at '
                   f'{100 * N["a"]["A"][0]:.2f} % (A) / {100 * N["a"]["C"][0]:.2f} % (C). A part that does not '
                   'scale with the rate is one particle source making two tracks.', BLUE, 24),
        sd.callout(f'That is ~{N["corr_pairs"]["A"]:,.0f} (A) and ~{N["corr_pairs"]["C"]:,.0f} (C) correlated '
                   f'pairs; accidentals are ~{sd.pct(N["acc_share"]["A"])} of all late pairs.', INK, 24),
        sd.callout(f'Geant4: {sd.pct(N["ee_share_A"])} (A) / {sd.pct(N["ee_share_C"])} (C) of reconstructable '
                   'same-chamber pairs are one photon\'s e⁺e⁻.', PURPLE, 24), w=560, gap=20)
    body = (sd.title('There are correlated same-chamber pairs: 0.15 % of late events, the same in A and C',
                     'share of tracked events with a second same-chamber track, per ToF bin · fit a + b·rate over 12–75 ms')
            + sd.row(P.svg('pair fraction vs time'), side, gap=48))
    notes = f"""
<p><b>The method.</b> Accidental pairs scale with the instantaneous rate, correlated ones do not. Fitting the fraction
f(t) = a + b·rate(t) over 12–75 ms separates them without any event mixing or vertex test. χ²/ndf is
{A['cmp_tof_fit'].set_index('arm').loc['A', 'chi2']:.0f}/{int(A['cmp_tof_fit'].set_index('arm').loc['A', 'ndf'])} (A),
so the statistical error is understated by ~2.5; the conclusion that a ≠ 0 holds at many σ.</p>
<p><b>Why not 0–12 ms.</b> 77 % of chamber A's pairs fall there; they are flash and epithermal pile-up, and the
capsule-pointing purity of tracks is 6–25 % there against ~45 % later. The beam's own captures peak at 5–6 ms, but the
capsule tracks we record are 90 % after 12 ms (delayed neutrons). Details: same-chamber-pairs note, slides 2–3.</p>
<p>Code: <code>same_chamber_compare.py</code> → <code>cmp_tof_fraction.csv</code>, <code>cmp_tof_fit.csv</code>.</p>"""
    D.slide('correlated', body, notes, short='Correlated pairs',
            foot='is2_v2 re-pass, capsule-focus k · A and C, all runs from run_69 onward')


def s_penetration(D, A, N):
    C = A['c2_ladder']
    S = A['leg_spectra']
    P = sd.Plot(1060, 580, x=(0, 16), y=(0, 1.0), xlabel='lepton kinetic energy at the capsule (MeV)',
                ylabel='probability, lepton aimed at a plastic bar')
    P.xticks([(v, str(v)) for v in range(0, 17, 2)])
    P.yticks([(v, f'{100 * v:.0f}%') for v in (0, 0.25, 0.5, 0.75, 1.0)])
    edges = list(S.lo) + [float(S.hi.iloc[-1])]
    for col, c, shift, lab in (('al_gap', AL_C, UP_LOSS, 'Al conversion legs'), ('x17_birth', X17_C, 0.0, 'X17 legs')):
        v = S[col].to_numpy(float)
        v = 0.42 * v / v.max()
        xs, ys = sd.step_xy([e + shift for e in edges], list(v))
        xs = [min(x, 16) for x in [xs[0]] + list(xs) + [xs[-1]]]
        ys = [0.0] + list(ys) + [0.0]
        P.raw(sd.poly([P.X(x) for x in xs], [P.Y(y) for y in ys], c, 1.5, fill=c,
                      tip=f'{lab}: energy spectrum (shape only)').replace('stroke="none"', 'fill-opacity="0.16" stroke="none"'),
              back=True)
    for col, c, dash, lab in (('plastic', ORANGE, None, 'reaches the plastic (> 0.3 MeV)'),
                              ('plastic_trig', INK, '10 7', 'plastic ≥ 2.5 MeV (trigger level)'),
                              ('liquid', PURPLE, None, 'reaches the liquid (> 0.5 MeV)')):
        P.line(list(C['T']), list(C[col]), c, w=3.5, dash=dash, r=5,
               tips=[f'{t:g} MeV: {100 * v:.0f} % {lab}' for t, v in zip(C['T'], C[col])])
    P.text(1.0, 0.47, 'Al legs', 24, AL_C, weight=600)
    P.text(10.5, 0.47, 'X17 legs', 24, X17_C, weight=600)
    leg = sd.legend([('reaches the plastic', ORANGE), ('plastic ≥ 2.5 MeV (trigger)', INK, 'dash'),
                     ('reaches the liquid', PURPLE)], size=22)
    al, x = N['al_l_e'], N['x_l_e']
    bars = sd.hbars([
        ('X17 · trigger level', x.plastic_trig, X17_C, f'n = {x.n:,} X17 legs into a gap'),
        ('Al · trigger level', al.plastic_trig, AL_C, f'n = {al.n:,} Al conversion legs into a gap'),
        ('X17 · liquid', x.liquid, X17_C, ''),
        ('Al · liquid', al.liquid, AL_C, ''),
    ], 1.0, width=300, label_w=250, fmt=lambda v: sd.pct(v, 1 if v < 0.05 else 0), size=23)
    side = sd.col(sd.p('per lepton that reaches a drift gap', 23, MUT), bars,
                  sd.callout('An X17 lepton is a minimum-ionising particle that crosses the stack. An Al '
                             'lepton has ~2 MeV at the gap and stops in the wall or the plastic.', INK, 23),
                  w=560, gap=16)
    body = (sd.title(f'X17 leptons cross the whole stack; Al pair leptons stop in it ({sd.pct(x.liquid)} vs '
                     f'{sd.pct(al.liquid, 1)} reach the liquid)',
                     'curves: Geant4 single leptons from the capsule into one arm · shaded: leg energy spectra')
            + sd.row(sd.col(P.svg('penetration vs energy'), leg, gap=4), side, gap=40))
    notes = f"""
<p><b>Curves</b> (<code>leg_penetration.c2_ladder</code>): the calorimetry C2 Geant4 guns (e⁻ and e⁺ averaged, from
the capsule centre at θ 90°, aimed at a plastic bar, 20k per energy, production geometry), conditional on the
lepton reaching the drift gap. The stack behind each chamber: 3 mm SiPM wall, 20 mm plastic, ~18 mm liquid (LAB).</p>
<p><b>Spectra.</b> Al: conversion leptons from ²⁷Al(n,γ) in the capsule vessel that cross ≥ 20 mm of a gap, Geant4
10⁹-neutron campaign; their gap-entry energy + {UP_LOSS} MeV (half the measured upstream loss of a pair) to put it at
the capsule. X17: both legs of X17 → e⁺e⁻ pairs (pair sim, 10⁷ events) whose legs enter two different arms.</p>
<p><b>Numbers folded with the curves</b> (energy only: the lepton is aimed at the stack):
plastic &gt; 0.3 MeV — X17 {sd.pct(x.plastic)}, Al {sd.pct(al.plastic)}; plastic ≥ 2.5 MeV — X17
{sd.pct(x.plastic_trig)}, Al {sd.pct(al.plastic_trig)}; liquid — X17 {sd.pct(x.liquid)}, Al {sd.pct(al.liquid, 1)}.</p>
<p><b>With the real geometry</b> (each lepton's own Geant4 deposits, so the gaps between bars and the bar size count):
plastic ≥ 2.5 MeV — X17 {sd.pct(N['x_l_g'].plastic_trig)}, Al {sd.pct(N['al_l_g'].plastic_trig, 1)}; liquid — X17
{sd.pct(N['x_l_g'].liquid)}, Al ≈ {sd.pct(N['al_l_g'].liquid, 1)} (Al liquid estimated from the curve). The ratio X17/Al
is ×7 at the trigger level and ×10–20 at the liquid in both views.</p>
<p><b>Caveat for X17.</b> At wide opening angles (≥ 140°) the soft X17 leg has only 3.9–4.8 MeV and about half of
those stop before the plastic (calorimetry note, slide c2). The hard leg always crosses.</p>"""
    D.slide('penetration', body, notes, short='X17 vs Al legs',
            foot='Geant4: calorimetry C2 guns (condor 4409644), 10⁹-n thermal campaign, 10⁷ X17/IPC pair sim')


def s_pairstats(D, A, N):
    def grp(title, items, vmax=1.0):
        rows = []
        for lab, v, c, tip in items:
            rows.append((lab, v, c, tip))
        return sd.col(sd.p(title, 25, INK, 600),
                      sd.hbars(rows, vmax, width=330, label_w=120, h=32,
                               fmt=lambda v: sd.pct(v, 2 if v < 0.01 else 1 if v < 0.1 else 0), size=23), gap=10)
    ae, xe, ag, xg = N['al_p_e'], N['x_p_e'], N['al_p_g'], N['x_p_g']
    g1 = grp('both legs reach the plastic', [
        ('X17', xe.plastic, X17_C, f'energy only; with real geometry {sd.pct(xg.plastic)}'),
        ('Al', ae.plastic, AL_C, f'energy only; with real geometry {sd.pct(ag.plastic, 1)}')])
    g2 = grp('both legs reach the liquid', [
        ('X17', xe.liquid, X17_C, f'energy only; with real geometry {sd.pct(xg.liquid, 1)}'),
        ('Al', ae.liquid, AL_C, 'energy only (product of the two legs\' curve values)')])
    g3 = grp('the pair can fire the trigger itself', [
        ('X17', N['x_trig'].plastic_trig, X17_C, 'either arm: wall ≥ 0.3 MeV and plastic ≥ 2.5 MeV (real geometry)'),
        ('Al', N['al_trig'].plastic_trig, AL_C, 'the one arm, sum of both legs\' plastic ≥ 2.5 MeV (real geometry)')])
    diagram = pair_diagram()
    side = sd.col(
        diagram,
        sd.callout('An Al pair is recorded mostly because something else fired the trigger: a Compton electron '
                   'from the same capture (76 % of trigger legs).', RED, 23),
        sd.callout('Its two tracks then rest on the Micromegas alone, with the scintillators unable to confirm '
                   'either leg: a soft, close, same-chamber pair is the hardest two-track case.', INK, 23),
        sd.callout('An X17 pair fires its own trigger, is confirmed leg by leg on the scintillators, and its legs '
                   'are 470 mm apart in two chambers.', GREEN, 23), w=720, gap=16)
    body = (sd.title('Al pairs are the hard case for two-track statistics; X17 pairs are the easy one',
                     'Geant4, per pair with both legs in drift gaps · Al: same chamber · X17: opposite chambers')
            + sd.row(sd.col(sd.row(g1, g2, gap=40), g3, gap=40, w=880), side, gap=56))
    notes = f"""
<p><b>Numbers</b> (<code>leg_penetration.py</code> → <code>{ST}/penetration.csv</code>). "Energy only" folds the
single-lepton curves of the previous slide over each pair's two leg energies. "Real geometry" uses each lepton's own
Geant4 deposits, so it also pays for leptons that miss a bar.</p>
<table><tr><th></th><th>Al e⁺e⁻ (same chamber)</th><th>X17 (opposite chambers)</th></tr>
<tr><td>both legs: wall, real geometry</td><td>{sd.pct(ag.wall)}</td><td>{sd.pct(xg.wall)}</td></tr>
<tr><td>both legs: plastic &gt; 0.3 MeV, real</td><td>{sd.pct(ag.plastic, 1)}</td><td>{sd.pct(xg.plastic)}</td></tr>
<tr><td>both legs: plastic ≥ 2.5 MeV, real</td><td>{sd.pct(ag.plastic_trig, 1)}</td><td>{sd.pct(xg.plastic_trig)}</td></tr>
<tr><td>both legs: liquid, real</td><td>—</td><td>{sd.pct(xg.liquid)}</td></tr>
<tr><td>pairs</td><td>{int(ag.n):,}</td><td>{int(xg.n):,}</td></tr></table>
<p>The real-geometry X17 numbers are low mainly because the plastic bars cover less than the chamber face (two
200 × 300 mm bars with a gap at u ≈ +7 mm) — that is acceptance, the same for any particle, not penetration.</p>
<p><b>Same-chamber pairs vs two-track separation.</b> 44 % of Geant4 Al pairs are &lt; 20 mm apart in one view at
mid-gap, where the re-pass still loses the second track (same-chamber note, slide "expect"). X17 legs never share a
chamber.</p>"""
    D.slide('pairstats', body, notes, short='Pair statistics',
            foot='Geant4 per-lepton truth (same_arm_pairs.py) and the X17/IPC pair sim; thresholds as the production trigger')


def pair_diagram():
    W, H = 560, 270
    s = []
    cx, cy = 60, 150
    s.append(f'<circle cx="{cx}" cy="{cy}" r="34" fill="#eaf3fb" stroke="{GREY}" stroke-width="3"/>')
    s.append(sd.T(cx, cy + 62, 'capsule', 19, MUT))
    for x0, lab in ((300, 'gap'), (380, 'wall'), (420, 'plastic'), (490, 'liquid')):
        w = {'gap': 40, 'wall': 10, 'plastic': 40, 'liquid': 40}[lab]
        col = {'gap': '#f3ead8', 'wall': '#e3e7ee', 'plastic': '#fde7d4', 'liquid': '#efe3f7'}[lab]
        s.append(f'<rect x="{x0}" y="30" width="{w}" height="240" fill="{col}" stroke="{GREY}" stroke-width="1.5"/>')
        s.append(sd.T(x0 + w / 2, 22, lab, 17, MUT))
    s.append(sd.poly([cx + 30, 200, 310, 352], [cy - 12, cy - 30, cy - 46, cy - 60], RED, 3.5))
    s.append(sd.poly([cx + 30, 200, 330, 395], [cy - 4, cy + 4, cy + 22, cy + 40], RED, 3.5))
    s.append(f'<circle cx="352" cy="{cy - 60}" r="6" fill="{RED}"/><circle cx="395" cy="{cy + 40}" r="6" fill="{RED}"/>')
    s.append(sd.T(180, cy - 52, 'Al e⁺e⁻ ~2 MeV: stop', 19, RED, weight=600))
    s.append(sd.line(cx, cy + 30, 545, cy + 105, GREEN, 3.5))
    s.append(sd.T(220, cy + 100, 'X17 leg ~10 MeV: through', 19, GREEN, weight=600))
    return sd.svg(W, H, ''.join(s), 'same-chamber Al pair vs X17 leg in the stack')


def s_singles(D, A, N):
    S = A['singles_origin']
    n = int(S[S.split == 'summary'].n.iloc[0])

    def rows(split, cols):
        g = S[S.split == split].sort_values('share', ascending=False)
        return [(r.category, r.share, cols.get(r.category, GREY), f'{r.n:,} of {n:,} triggered track-like leptons')
                for r in g.itertuples()]
    proc = rows('process', {'Compton electron': BLUE, 'pair lepton (e⁺e⁻)': RED, 'delta ray': GOLD})
    made = rows('made', {'made in the capsule': RED, 'made in the chamber': ORANGE, 'made in the air': GREY})
    capt = rows('capture', {'capture on the capsule Al': RED})
    hb = lambda r: sd.hbars(r, 1.0, width=300, label_w=330, h=28, fmt=lambda v: sd.pct(v, 1 if v < 0.1 else 0), size=21)
    left = sd.col(sd.p('Geant4: what the single tracks in A and C are', 24, MUT),
                  sd.p('the particle', 22, INK, 600), hb(proc[:3]),
                  sd.p('where it was made', 22, INK, 600), hb(made[:3]),
                  sd.p('which neutron capture', 22, INK, 600), hb(capt[:2]), w=900, gap=10)
    pl = N['pur_late_A'], N['pur_late_C']
    g4p = N['g4_pur']
    right = sd.col(
        sd.bignum(f'×{N["singles_ratio"]["A"]:.2f} / ×{N["singles_ratio"]["C"]:.2f}',
                  'tracks per trigger, data over Geant4-through-reco', GREEN,
                  'A / C. The rate of single tracks is what Geant4 predicts to ~20 %.', dark=False, size=56),
        sd.bignum(f'{sd.pct(pl[0])} / {sd.pct(pl[1])}', 'of late tracks point at the capsule source', BLUE,
                  f'Geant4 through the same reconstruction: {sd.pct(g4p.get("A", np.nan))} / '
                  f'{sd.pct(g4p.get("C", np.nan))}. The data are at least as capsule-pure.', dark=False, size=56),
        sd.callout('Through-going cosmics and beam punch-through are a small, separate class: the A–C cosmic rate '
                   'matches the muon flux, and in-beam "through-goers" rarely reach the liquid.', INK, 22),
        w=640, gap=22)
    body = (sd.title('The single tracks are Compton electrons of the Al-capture γ, made in the capsule',
                     f'Geant4 10⁹-n thermal campaign, triggered events, leptons crossing ≥ 20 mm of the A or C gap '
                     f'(n = {n:,}) · right: data checks')
            + sd.row(left, right, gap=56))
    notes = f"""
<p><b>Geant4 split</b> (<code>leg_penetration.singles</code>): every lepton that crosses ≥ 20 mm of the A or C drift gap
in an event where some arm fired the emulated trigger. Pair leptons are {sd.pct(float(S[(S.split == 'process') & (S.category == 'pair lepton (e⁺e⁻)')].share.iloc[0]))}
of single tracks — more than the 10 % of trigger legs, because a soft pair leg reaches the gap but not the plastic.
The capture: almost all on the capsule's Al vessel, which is what the capsule-nose image (next slide) shows.</p>
<p><b>Data/expectation</b>: data tracks per trigger / Geant4 singles per trigger = {N['singles_eps_data']['A']:.2f}
(A), {N['singles_eps_data']['C']:.2f} (C), against the digitised-reconstruction efficiency
{N['singles_eps_dig']['A']:.2f} / {N['singles_eps_dig']['C']:.2f}.</p>
<p><b>Capsule purity</b>: tracks whose line passes within 15 mm of the measured source, minus a shuffled-angle null,
≥ 20 ms (<code>capsule_tof.py</code>). Geant4's value comes from g4_digi through the same reconstruction.</p>
<p><b>What else is in the late data.</b> Late triggers follow a 23.5 ms exponential (the hall's neutron die-away, not
the beam's own captures), which is why the capsule tracks we record are delayed neutrons captured on the capsule. Cosmic
A–C muons are at the flux rate (~0.6 efficiency per chamber), a tiny share of triggers.</p>"""
    D.slide('singles', body, notes, short='Singles',
            foot='Geant4 neutrons_thermal_trig_2cm_nose · data: is2_v2 re-pass (cf), run_69 onward')


def s_calo(D, A, N):
    st = lambda t, c: f'<span style="color:{c};font-weight:600">{t}</span>'
    rows = [
        ['plastic energy scale', 'production line reads a muon 3–24 % low; MIP-anchored scale per bar, 4–7 %',
         st('fixed', GREEN)],
        ['plastic thickness', '20 mm (Geant4) vs 25 mm (run config): 21 % on the scale', st('measure a bar', GOLD)],
        ['liquids', 'not dead: a muon sits just above the 16–18 mV readout threshold; A/D usable on part of the face',
         st('threshold-limited', GOLD)],
        ['soft-leg energy vs IPC', 'plastic energy adds 5–8 % to Z² over the opening angle alone; true soft-leg T would '
         'give ×2.5–4', st('killed', RED)],
        ['MM dE/dx', 'one 30 mm gap: FWHM/MPV 0.95; 2-MIP flag 40 % efficient at 10 % mis-tag', st('killed', RED)],
        ['gain monitor', 'MM road charge maps gain to 19 % (A) / 39 % (C) per 40 mm cell; no attachment seen',
         st('available', GREEN)],
        ['plastic gain vs flash time', 'peak rises ~10 % 10 → 80 ms; needs an in-beam energy reference', st('open', GOLD)],
    ]
    tbl = sd.table(['item', 'result', 'status'], rows, size=23, widths=[330, 1000, 260],
                   align=['left', 'left', 'left'])
    body = (sd.title('Calorimetry: a range telescope, not a calorimeter; the energy route is closed for this dataset',
                     'ntof_calorimetry, plan C1–C4 and M1–M2, closed 8 October')
            + tbl
            + sd.callout('Lesson for the next design: an electron loses ~1.2 MeV before the plastic (capsule, chamber, '
                         'wall). Measure the soft leg before that material, with a calorimetric first layer, or the '
                         'energy information is gone.', RED, 25))
    notes = """
<p>Full write-up: the <code>calorimetry</code> slide note (dylan-neff.web.cern.ch/notes/calorimetry.html) and
<code>ntof_calorimetry/PLAN.md</code> "Results so far".</p>
<p>The stack behind each chamber is 3 mm PVT (SiPM wall), 20 mm PVT (plastic) and ~18 mm LAB. It stops an electron of
~5 MeV without the liquid and ~8–9 MeV with it, so for an X17 pair it is a range telescope with dE/dx sampling. The
trigger thresholds on the MIP-anchored scale are 2.1–2.9 MeVee, at the ²⁸Al β endpoint, so the planned ²⁸Al
calibration (C1.2) is not possible.</p>
<p>In-beam A–C "through-goers" reach the liquid at 0.10 ± 0.04 (A) / 0.04 (C) of the cosmic-muon expectation: they are
mostly not penetrating muons.</p>"""
    D.slide('calo', body, notes, short='Calorimetry', foot='Use c1/calib_plastic_e.json for any plastic energy; '
                                                           'never the source keVee line above 2 MeV or for the liquids')


def s_close(D, A, N):
    li = lambda t: f'<li style="margin-bottom:14px">{t}</li>'
    left = ('<ul style="font-size:26px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li('Tracking works in the beam: single-track rate within ~20 % of Geant4, a sharp image of the capsule '
                 'nose at every ToF, angle scale checked on cosmic muons to 1–4 %.')
            + li('The trigger fires on what Geant4 says it should (Compton electrons of the Al γ).')
            + li('Same-chamber correlated pairs exist and their shape is compatible with Al e⁺e⁻.')
            + li('X17 pairs would be the easy topology: two chambers, penetrating legs, self-triggering, '
                 'vertexing within ~10 mm.') + '</ul>')
    right = ('<ul style="font-size:26px;line-height:1.35;color:#eef0f3;padding-left:28px">'
             + li('The pair rate is ⅛ of Geant4 and pair-specific. Until it is understood, the two-track efficiency '
                  'for close, soft pairs is not demonstrated.')
             + li('The Al pairs cannot be vertexed (scattering), and e⁺e⁻ vs two Compton electrons is not separable '
                  'without charge or energy.')
             + li('Capsule pose vs beam (~10 mm offset) needs a survey or photo.')
             + li('For LNL: measure the soft leg before ~1 MeV of material; keep the liquid/back layer above its '
                  'readout threshold; a test pair source with X17-like legs would validate two-track efficiency '
                  'directly.') + '</ul>')
    body = (sd.title('<span style="color:#eef0f3">Did the set-up work? Single tracks yes; pairs not yet shown</span>')
            + sd.row(sd.col(sd.p('what the data show', 26, DGREEN), left, w=800),
                     sd.col(sd.p('what this does not rule out, and next', 26, DRED), right, w=820), gap=56))
    notes = """
<p><b>Next steps for the factor 8</b> (same-chamber note): re-digitise the Geant4 pairs with is2_v2 (two-sided search,
t0 ≤ 300, |tan| &lt; 0.6 on both legs); Geant4 with delayed thermal neutrons on the capsule (the late population);
check Geant4's capsule (nose Al thickness, wrap) against the real one; look for a trigger/readout effect specific to
two-lepton events.</p>
<p><b>Why this matters for LNL.</b> The pair deficit could be in the simulation (capsule model, source population)
or in the detector (two-track loss in busy events). Only the second would argue against the set-up, and it would hit
X17 far less, because X17 legs land in different chambers.</p>"""
    D.slide('close', body, notes, dark=True, short='Verdict')


# --------------------------------------------------------------------------- #
def build(out: Path) -> Path:
    A = load()
    N = numbers(A)
    D = sd.Deck('X17 at n_TOF: status since Athens',
                'did the n_TOF set-up see what it was expected to see: Al pairs, single tracks, penetration, calorimetry')
    s_cover(D, A, N)
    s_since(D, A, N)
    s_funnel(D, A, N)
    s_scorecard(D, A, N)
    s_correlated(D, A, N)
    SCD.s_opening(D, A, N)
    SCD.s_expect(D, A, N)
    SCD.s_scatter(D, A, N)
    s_penetration(D, A, N)
    s_pairstats(D, A, N)
    SCD.s_vertex(D, A, N)
    s_singles(D, A, N)
    SCD.s_pose(D, A, N)
    s_calo(D, A, N)
    from ntof_calorimetry import make_deck as CD
    CD.s_dedx(D, CD.load())
    s_close(D, A, N)
    out.parent.mkdir(parents=True, exist_ok=True)
    D.write(out, note_meta=dict(
        title='X17 at n_TOF: status since Athens (15 Oct 2026)',
        summary='Did the set-up see what it should? Singles and the capsule image agree with Geant4; correlated '
                'same-chamber pairs look like Al e+e- but at 1/8 of the rate; X17 legs cross the stack where Al legs '
                'stop; calorimetry closed.',
        tags='X17,status,Geant4,same-chamber pairs', date=dt.date.today().isoformat()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    print('wrote', build(a.out or ST / 'deck' / 'x17-status-2026-10-15.html'))


if __name__ == '__main__':
    main()
