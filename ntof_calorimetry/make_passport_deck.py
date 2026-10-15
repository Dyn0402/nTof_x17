#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_passport_deck.py -- the track passport as a pedagogical slide note:
the physics principle, how it should work in simulation, and how it works in
the real n_TOF detector, with the performance of each measurement.

Built only from the outputs of `mm_edep_ana`, `track_passport`, `wall_plastic`
and the Heed pools, so re-running those and then this updates every number.

    PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.track_passport
    PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_passport_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        /media/dylan/data/x17/calorimetry/passport/deck/track-passport.html --slug track-passport --force --deploy
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

from ntof_calorimetry import track_passport as TP  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402

PP = TP.PP
RES = TP.RES
ARMC = {'A': sd.BLUE, 'C': sd.ORANGE}
G4C = sd.INK
LAYC = {'MM': sd.GREY, 'wall': sd.GOLD, 'plastic': sd.ORANGE, 'liquid': sd.PURPLE}
FEAT_C = {'plastic': '#b9c1cc', 'plastic+wall': sd.GREY, 'plastic+wall+angle': sd.BLUE,
          'plastic+wall+angle+MM': sd.PURPLE}
W_EV = 27.5


# =========================================================================== #
# data
# =========================================================================== #
def _hist(v, edges, norm=True):
    h, _ = np.histogram(np.asarray(v, float), edges)
    return h / max(h.sum(), 1) if norm else h


def beam_cal_hists() -> dict:
    """Beam MM calibration, as distributions: data q (keV / 30 mm at normal
    incidence, after the in-situ map and the data/sim scale) against the
    beam simulation's reconstructed q and its truth.  Cached."""
    cache = PP / 'deck_cache_beamcal.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from ntof_calorimetry import mm_edep_ana as MA
    D = pd.read_parquet(PP / 'passport_data.parquet')
    out = {}
    edges = np.geomspace(1.5, 40, 46)
    for arm in TP.ARMS:
        Sb = MA.load_sim_prof('beam', arm)
        Wb = MA.window_curve(Sb)
        Sq = MA._q(Sb, Wb, Wb.attrs['k'], arm == 'A')
        d = D[(D.arm == arm) & D.mm_ok]
        out[arm] = dict(edges=edges, data=_hist(d.q_mm, edges), sim=_hist(Sq.q, edges),
                        truth=_hist(Sq.E_true / Sq.sec, edges), n_data=len(d), n_sim=len(Sq),
                        med_data=float(d.q_mm.median()), med_sim=float(Sq.q.median()),
                        med_truth=float((Sq.E_true / Sq.sec).median()))
    cache.write_bytes(pickle.dumps(out))
    return out


def load() -> dict:
    t = {}
    t['S'] = json.loads((PP / 'summary.json').read_text())
    t['ana'] = json.loads((RES / 'summary.json').read_text())
    t['phys'] = pd.read_csv(RES / 'physics.csv')
    t['PC'] = pd.read_csv(PP / 'path_length_cosmic.csv')
    t['PL'] = pd.read_csv(PP / 'path_length.csv')
    t['TPt'] = pd.read_csv(PP / 'two_particles.csv')
    t['tp_pts'] = pd.read_csv(PP / 'figures' / 'two_particles.csv')
    t['EN'] = pd.read_csv(PP / 'energy.csv')
    t['EP'] = pd.read_parquet(PP / 'energy_sim_pred.parquet')
    t['OR'] = pd.read_csv(PP / 'origin.csv')
    t['D'] = pd.read_parquet(PP / 'passport_data.parquet')
    t['Sim'] = pd.read_parquet(PP / 'passport_sim.parquet')
    t['G'] = pd.read_parquet(PP / 'g4.parquet')
    t['pts'] = pd.read_parquet(PP / 'two_particles_points.parquet')
    t['EX'] = pd.read_pickle(PP / 'examples.pkl')
    B = pd.read_pickle(TP.EDEP / f'beam_{TP.RUN}.pkl')[['subrun', 'event_id', 'arm', 'track_id', 'v_drift_um_ns']]
    B['subrun'] = B.subrun.astype(str)
    t['EX'] = t['EX'].merge(B, on=['subrun', 'event_id', 'arm', 'track_id'], how='left')
    for a in TP.ARMS:
        t[f'spatial_{a}'] = pd.read_csv(RES / f'spatial_cosmic_{a}.csv')
        t[f'flash_{a}'] = pd.read_csv(RES / f'flash_{a}.csv')
        t[f'win_{a}'] = pd.read_csv(RES / f'window_{a}.csv')
        t[f'simpairs_{a}'] = pd.read_parquet(RES / f'sim_pairs_cosmic_{a}.parquet')
    t['spectra'] = pd.read_csv(OUT / 'wp' / 'spectra.csv')
    t['scale'] = pd.read_csv(OUT / 'wp' / 'scale_fits.csv')
    t['bc'] = beam_cal_hists()
    t['heed'] = {}
    for name in ('e-_4_th0', 'e-_1_th0', 'e-_0.2_th0', 'mu-_3000_th0'):
        f = OUT / 'heed' / f'{name}.npz'
        if f.exists():
            t['heed'][name] = np.load(f)['n_e'].astype(float) * W_EV / 1e3
    return t


def en(t, f, s):
    x = t['EN'][(t['EN'].features == f) & (t['EN'].subset == s)]
    return float(x.w68.iloc[0]) if len(x) else float('nan')


def orr(t, target, feats, samp):
    x = t['OR'][(t['OR']['sample'] == samp) & (t['OR'].target == target) & (t['OR'].features == feats)]
    return float(x.auc.iloc[0])


def step_hist(P, edges, vals, color, w=3, dash=None, tip=None, ymax=None):
    xs, ys = sd.step_xy(list(edges), list(vals))
    if ymax is not None:
        ys = [min(y, ymax) for y in ys]
    P.raw(sd.poly([P.X(x) for x in xs], [P.Y(y) for y in ys], color, w, dash, tip=tip))


def lin_ticks(lo, hi, step, fmt='{:g}'):
    return [(v, fmt.format(v)) for v in np.arange(lo, hi + 1e-9, step)]


# =========================================================================== #
# shared diagram: the stack, side view
# =========================================================================== #
def stack_svg(W, H, track=None, labels=True, deposits=None, prof=None, small=False, tip_layers=True):
    """Side view of one arm: capsule, MM drift gap, wall, plastic, liquid.
    ``track``: (x at the mesh in mm, tan) to draw a track; ``deposits``: text
    per layer; ``prof``: (depths mm, charges) for the MM drift profile."""
    z0, z1 = -40.0, 330.0
    u_lo, u_hi = -230.0, 230.0
    L, R, Tp, Bt = 60, 20, 40 if labels else 14, 40
    zx = lambda z: L + (z - z0) / (z1 - z0) * (W - L - R)  # noqa: E731
    uy = lambda u: Tp + (u_hi - u) / (u_hi - u_lo) * (H - Tp - Bt)  # noqa: E731
    o = []
    lay = [('MM', 0, 30, 'Micromegas TPC: 30 mm drift gap, Ar/iC4H10 90/10. Measures the track (position, angle, '
                         'depth from the drift time) and the ionisation charge.'),
           ('wall', TP.Z_WALL, TP.Z_WALL + 3, 'SiPM wall: 3 mm PVT strips in 4 groups, read at both ends. '
                                              'A thin layer: dE/dx (~0.46 MeV for a MIP at normal incidence).'),
           ('plastic', TP.Z_PLAS, TP.Z_PLAS + 20, 'Plastic: two 20 mm PVT bars, one PMT each. Stops electrons '
                                                  'below ~5-6 MeV at the gap: a calorimeter for them.'),
           ('liquid', TP.Z_LIQ, TP.Z_LIQ + 60, 'Liquid scintillator behind the plastic: fires only if the '
                                               'particle got through the plastic.')]
    for name, a, b, tp in lay:
        c = LAYC[name]
        o.append(f'<rect x="{zx(a):.1f}" y="{uy(u_hi * 0.92):.1f}" width="{max(zx(b) - zx(a), 4):.1f}" '
                 f'height="{uy(u_lo * 0.92) - uy(u_hi * 0.92):.1f}" fill="{c}" fill-opacity="0.22" stroke="{c}" '
                 f'stroke-width="2.5"{sd.tipattr(tp) if tip_layers else ""}/>')
        if labels:
            o.append(sd.T((zx(a) + zx(b)) / 2, Tp - 10, name, 18 if small else 22, sd.INK, weight=600))
    # mesh + capsule
    o.append(sd.line(zx(0), uy(u_hi * 0.92), zx(0), uy(u_lo * 0.92), sd.INK, 3))
    o.append(f'<circle cx="{zx(-30):.1f}" cy="{uy(0):.1f}" r="9" fill="{sd.RED}"'
             f'{sd.tipattr("the capsule (He-3 + Al/CFRP walls), ~1 m away and well off this scale")}/>')
    if labels:
        o.append(sd.T(zx(-30), uy(0) + 34, 'capsule', 18, sd.RED))
        o.append(sd.T(zx(0) - 6, uy(u_lo * 0.92) + 26, 'mesh', 16, sd.MUT, 'end'))
    if track is not None:
        x0, tn = track
        zz = np.array([0.0, z1 - 10])
        o.append(sd.poly([zx(z) for z in zz], [uy(x0 + tn * z) for z in zz], sd.RED, 3.5))
    if prof is not None:
        dz, q = prof
        q = np.clip(np.asarray(q, float) / max(np.max(q), 1e-9), 0, 1)
        for z, w in zip(dz, q):
            if w > 0.03:
                o.append(f'<circle cx="{zx(z):.1f}" cy="{uy(track[0] + track[1] * z):.1f}" r="{3 + 11 * w:.1f}" '
                         f'fill="{sd.BLUE}" fill-opacity="0.55"/>')
    if deposits:
        for name, a, b, _ in lay:
            if deposits.get(name):
                uu = track[0] + track[1] * (a + b) / 2 if track else 0
                o.append(sd.T((zx(a) + zx(b)) / 2, uy(uu) - 22, deposits[name], 20, sd.INK, weight=600))
    return o, zx, uy


# =========================================================================== #
# slides
# =========================================================================== #
def s_cover(D, t):
    c = t['S']['cal']
    g4t = t['TPt'][t['TPt'].source == 'Geant4']
    body = (sd.kicker(f'n_TOF MX17 · detector demonstrator · {dt.date.today():%-d %b %Y}')
            + sd.title('<span style="color:#eef0f3">One particle, five independent measurements: the MX17 stack '
                       'tells what a track is, where it came from and how much energy it carried</span>')
            + sd.p('Built up from the physics, checked in simulation, then shown on 15,000 real beam tracks '
                   '(run_145). The aim is to prove the concept for a future set-up, not to measure X17 here.',
                   28, sd.DMUT)
            + sd.row(
                sd.bignum(f'{100 * c["A"]["sigma_total"]:.0f} / {100 * c["C"]["sigma_total"]:.0f} %',
                          'MM gas dE/dx per track (A / C)', sd.DBLUE,
                          'one 30 mm gap, on beam electrons, after an in-situ gain map',
                          tip='robust sigma of ln(E_hat/E_true) in the beam simulation, added in quadrature to the '
                              'extra spread the forward fold finds in the data'),
                sd.bignum(f'{en(t, "plastic+wall+angle", "all"):.2f} MeV', 'electron energy (68 % width)', sd.GOLD,
                          f'plastic + track angle; steep tracks {en(t, "plastic", "sec >= 1.1"):.1f} → '
                          f'{en(t, "plastic+wall+angle", "sec >= 1.1"):.1f} MeV with the angle',
                          tip='Geant4 triggered tracks, kinetic energy at the gap, cross-validated kNN'),
                sd.bignum(f'AUC {g4t.auc_multi_mm.mean():.2f}', 'two particles in one track', sd.DGREEN,
                          f'only the MM sees them; the wall cannot (AUC {g4t.auc_multi_wall.mean():.2f})',
                          tip='Geant4: dominant particle < 80 % of the gap deposit'),
                sd.bignum(f'AUC {orr(t, "capsule", "tracking (position, direction)", "all gap tracks"):.2f}',
                          'capsule vs elsewhere', sd.DRED, 'from the track geometry alone',
                          tip='Geant4, every track through the gap'),
                gap=36))
    notes = """
<p><b>The question (Dylan, 15 Oct).</b> The MM dE/dx study and the scintillator calorimetry each worked on their
own. Combined, how much can this set-up say about a single track? The answer here is meant as a demonstrator:
what a TPC-plus-scintillator stack can measure, with numbers, so that a future set-up (LNL or elsewhere) can be
designed from it. It is not an X17 analysis.</p>
<p><b>Structure.</b> Part 1, the principle: what each layer measures and why (Heed and Geant4 truth). Part 2, how
it should work, in a full simulation with the measured resolutions. Part 3, the real detector: calibration, what
the real chamber adds, validation against Geant4, example tracks, and the performance table.</p>
<p><b>Sample.</b> run_145 (5 Aug 2026, the post-23-July noisy configuration), chambers A and C, ≥ 12 ms after the
γ-flash, single gated tracks on their own arm's trigger (wall AND plastic). Cosmic checks use run_149 (beam off).
Geant4: the thermal-neutron campaign on the 2 cm capsule nose (66,603 tracks crossing a gap, 5,726 triggered).</p>
<p>Code: <code>ntof_calorimetry/track_passport.py</code>, <code>make_passport_deck.py</code>; MM chain in
<code>mm_edep_*.py</code>, record in <code>ntof_calorimetry/MM_EDEP.md</code>.</p>"""
    D.slide('cover', body, notes, dark=True, short='Answer')


def s_idea(D, t):
    o, zx, uy = stack_svg(1000, 560, track=(40, 0.22),
                          deposits={'MM': '≈ 6 keV', 'wall': '≈ 0.5 MeV', 'plastic': '≤ 4 MeV', 'liquid': '?'})
    o.append(sd.T(zx(15), uy(-245), 'drift time = depth', 18, sd.MUT))
    diag = sd.svg(1000, 560, ''.join(o), 'the MX17 arm, side view')
    items = [
        ('Tracking (MM)', 'direction, position, pointing back to the source; the path length through every layer',
         sd.GREY),
        ('Gas dE/dx (MM)', 'ionisation in 30 mm of gas: how many particles, how fast', sd.BLUE),
        ('Wall dE/dx', '3 mm of plastic: a second, independent dE/dx sample', sd.GOLD),
        ('Plastic energy', '20 mm of plastic stops the electron: its energy', sd.ORANGE),
        ('Liquid', 'fires only if the particle got through the plastic', sd.PURPLE),
        ('Time of flight', 'time since the γ-flash = the neutron\'s energy', sd.RED),
    ]
    col = ''.join(f'<div style="border-left:6px solid {c};padding:2px 0 2px 18px">'
                  f'<p style="font-size:25px;font-weight:600">{a}</p><p style="font-size:22px;color:{sd.MUT};'
                  f'line-height:1.3">{b}</p></div>' for a, b, c in items)
    body = (sd.title('The idea: every layer measures something different about the same particle',
                     'one MX17 arm, side view along the chamber normal, to scale (the capsule is ~1 m away)')
            + sd.row(diag, sd.col(col, gap=14, w=600), gap=48)
            + sd.p('Hover the dotted words, the layers and every data point for definitions, counts and sources.',
                   22, sd.MUT))
    notes = """
<p><b>Geometry</b> (scint-stack <code>geometry.csv</code>): the wall is 97 mm behind the mesh, the plastic 190 mm, the
liquid 250 mm. The MM measures x and y with two strip views and the depth z from the drift time; together they give
the 3D direction of the track in the gap, and so where it crosses each counter and how long its path through it is.</p>
<p><b>Why combine.</b> A single number per layer is ambiguous: a large plastic signal could be a fast electron at a
steep angle or a slow one that stopped. The track angle resolves that. A large gas signal could be a slow electron
or two particles. The plastic resolves that (a slow electron does not reach it). Each layer removes an ambiguity of
another.</p>"""
    D.slide('idea', body, notes, short='The idea')


def s_thin(D, t):
    ph = t['phys']
    e0 = ph[(ph.particle == 'e-') & (ph.theta == 0)].sort_values('T')
    mu = ph[(ph.particle.str.startswith('mu')) & (ph.theta == 0)]
    P = sd.Plot(960, 600, x=(0.08, 20, 'log'), y=(0, 21), xlabel='electron kinetic energy T  (MeV)',
                ylabel='energy left in 30 mm of gas  (keV)', title='Heed: the gas deposit of one electron')
    P.xticks([(v, f'{v:g}') for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20)])
    P.yticks(lin_ticks(0, 18, 3))
    P.band(list(e0['T']), list(e0.q16), list(e0.q84.clip(upper=21)), sd.BLUE, 0.15,
           tip='16th to 84th percentile of the deposit')
    P.line(list(e0['T']), list(e0.median_keV), sd.BLUE,
           tips=[f'T = {a:g} MeV: median {b:.2f} keV, MPV {c:.2f} keV, FWHM/MPV {d:.2f} (n = {n:,})'
                 for a, b, c, d, n in zip(e0['T'], e0.median_keV, e0.mpv_keV, e0.fwhm_over_mpv, e0.n)])
    m = float(mu.median_keV.iloc[0])
    P.hline(m, sd.GREY, label=f'3 GeV muon: {m:.1f} keV', tip='a cosmic muon, Heed, normal incidence')
    P.raw(f'<rect x="{P.X(1):.1f}" y="{P.y0}" width="{P.X(16) - P.X(1):.1f}" height="{P.ph}" fill="{sd.GOLD}" '
          f'fill-opacity="0.08"/>', back=True)
    P.text(1.15, 19.5, 'n_TOF electrons: 1–16 MeV, "minimum-ionising"', 21, sd.GOLD)
    lo = e0[e0['T'].between(1.0, 16)]
    rise = float(lo.median_keV.max() / lo.median_keV.min() - 1)
    side = sd.col(
        sd.callout(f'Above ~1 MeV an electron leaves <b>{lo.median_keV.min():.1f}–{lo.median_keV.max():.1f} keV</b> '
                   f'in the gap whatever its energy (+{100 * rise:.0f} % from 1 to 16 MeV): '
                   'the gas cannot weigh a fast electron.', sd.BLUE),
        sd.callout('Below ~0.5 MeV the deposit climbs steeply: a <b>slow</b> electron stands out.', sd.RED),
        sd.callout('What the gas <i>can</i> count is particles: two fast electrons leave twice as much.', sd.GREEN),
        sd.p(f'One {sd.term("keV", "W = 27.5 eV per ion pair, so 6 keV is about 220 electrons")} '
             '≈ 36 ion pairs. The MM collects them and the avalanche makes them visible.', 23, sd.MUT),
        w=620, gap=22)
    body = (sd.title('Principle 1: a thin gas layer measures dE/dx, not energy',
                     'ionisation left by one particle crossing 30 mm of Ar/iC4H10 at normal incidence (Heed truth)')
            + sd.row(P.svg('Heed deposit vs T'), side, gap=56))
    notes = """
<p><b>Source.</b> Garfield++/Heed on lxplus (<code>heed_edep.py</code>), 20,000 tracks per point, the n_TOF gas
(Ar/iC4H10 90/10, 730 Torr), 30 mm, straight tracks. Energy = number of ionisation electrons × 27.5 eV.</p>
<p><b>The physics.</b> Bethe-Bloch: dE/dx falls as 1/β² for slow particles, reaches a minimum near βγ ≈ 3 and
then rises slowly (the relativistic rise, about +24 % in the MPV from 1.5 to 16 MeV here, the same as a 3 GeV
muon). A 1 MeV electron already has βγ ≈ 2.8, so every electron the X17 set-up cares about is on the plateau.</p>
<p><b>Consequence for the design.</b> The gas layer is a <i>counter</i> of ionising particles and a slow-particle
tag, not a calorimeter. Energy has to come from a layer thick enough to stop the electron.</p>"""
    D.slide('thin', body, notes, short='1 · dE/dx')


def s_landau(D, t):
    h = t['heed']
    e4 = h.get('e-_4_th0')
    rng = np.random.default_rng(1)
    two = rng.choice(e4, len(e4)) + rng.choice(e4, len(e4))
    edges = np.linspace(0, 30, 61)
    P = sd.Plot(1000, 600, x=(0, 30), y=(0, 0.16), xlabel='energy in the 30 mm gap  (keV)',
                ylabel='fraction of tracks per 0.5 keV', title='Heed: one 4 MeV electron, a muon, and two electrons')
    P.xticks(lin_ticks(0, 30, 5))
    P.yticks(lin_ticks(0, 0.16, 0.04, '{:.2f}'))
    mu = h.get('mu-_3000_th0')
    step_hist(P, edges, _hist(mu, edges), sd.GREY, 3, '8 6', tip=f'3 GeV muon, n = {len(mu):,}')
    step_hist(P, edges, _hist(e4, edges), sd.BLUE, 4, tip=f'one 4 MeV electron, n = {len(e4):,}')
    step_hist(P, edges, _hist(two, edges), sd.GREEN, 4, tip='two 4 MeV electrons in the same gap (sum of two '
                                                           'independent Heed draws)')
    m1, m2 = np.median(e4), np.median(two)
    P.vline(m1, sd.BLUE, label=f'median {m1:.1f}')
    P.vline(m2, sd.GREEN, label=f'{m2:.1f}')
    fw = float(t['phys'][(t['phys'].particle == 'e-') & (t['phys']['T'] == 4) & (t['phys'].theta == 0)].fwhm_over_mpv.iloc[0])
    side = sd.col(
        sd.legend([('one 4 MeV electron', sd.BLUE), ('two of them', sd.GREEN), ('cosmic muon', sd.GREY, 'dash')],
                  22),
        sd.callout(f'A single sample fluctuates a lot: FWHM ≈ {fw:.1f} × the peak, with a long '
                   f'{sd.term("Landau tail", "rare close collisions that hand a delta electron a large energy")}.',
                   sd.BLUE),
        sd.callout('Two particles shift the whole distribution up ×2; they overlap the one-particle tail only in part.',
                   sd.GREEN),
        sd.callout('An electron and a muon look the same: both are minimum-ionising. So cosmic muons can '
                   'calibrate the chamber for beam electrons.', sd.GREY),
        w=560, gap=22)
    body = (sd.title('Principle 2: one dE/dx sample is broad, but two particles still stand out',
                     'distribution of the gap deposit, Heed truth, normal incidence, 30 mm')
            + sd.row(P.svg('Landau'), side, gap=56))
    notes = """
<p><b>Why it is broad.</b> 30 mm of gas sees only ~30 primary collisions; their number is Poisson and each one's
energy has a 1/E² spectrum, so a few close collisions dominate some tracks. The result is the asymmetric Landau-like
shape with the long high-side tail. A thin plastic layer (the wall) has the same shape, narrower, because it sees
thousands of collisions.</p>
<p><b>What it means for "two particles".</b> Separation is statistical, not event-by-event perfect: the 2-particle
distribution peaks at twice the single one but the single one's tail reaches into it. Simulation and data later
quantify it as an AUC.</p>"""
    D.slide('landau', body, notes, short='2 · Landau')


def s_path(D, t):
    W, H = 900, 560
    o = []
    # two tracks through a slab: perpendicular and tilted
    sx0, sx1, sy0, sy1 = 150, 750, 230, 330
    o.append(f'<rect x="{sx0}" y="{sy0}" width="{sx1 - sx0}" height="{sy1 - sy0}" fill="{sd.GOLD}" fill-opacity="0.2" '
             f'stroke="{sd.GOLD}" stroke-width="2.5"/>')
    o.append(sd.T(sx1 + 14, (sy0 + sy1) / 2 + 8, 'thickness d', 21, sd.INK, 'start'))
    o.append(sd.arrow(260, 60, 260, 500, sd.BLUE, 4))
    o.append(sd.T(260, 40, 'θ = 0: path d', 22, sd.BLUE, weight=600))
    o.append(sd.arrow(470, 60, 690, 500, sd.RED, 4))
    o.append(sd.T(560, 40, 'tilted by θ: path d · sec θ', 22, sd.RED, weight=600))
    o.append(sd.line(470, 60, 470, 200, sd.MUT, 1.5, '6 5'))
    o.append(sd.T(492, 160, 'θ', 24, sd.RED, 'start'))
    o.append(sd.T(450, 400, 'deposit ∝ path, so dividing by sec θ', 22, sd.INK))
    o.append(sd.T(450, 432, 'puts every track on the same footing', 22, sd.INK))
    o.append(sd.T(450, 520, 'only true if the particle keeps its direction:', 21, sd.MUT))
    o.append(sd.T(450, 548, 'a muon does; a few-MeV electron scatters', 21, sd.MUT))
    diag = sd.svg(W, H, ''.join(o), 'path length')
    ph = t['phys']
    mu = ph[ph.particle.str.startswith('mu')].sort_values('theta')
    sec = 1 / np.cos(np.radians(mu.theta))
    P = sd.Plot(700, 560, x=(1, 1.5), y=(0.9, 1.6), xlabel='sec θ', ylabel='gas deposit / deposit at θ = 0',
                title='Heed, 3 GeV muon')
    P.xticks(lin_ticks(1, 1.5, 0.1, '{:.1f}'))
    P.yticks(lin_ticks(0.9, 1.6, 0.1, '{:.1f}'))
    P.line([1, 1.5], [1, 1.5], sd.MUT, 2, '8 6', markers=False, tip='∝ sec θ')
    P.line(list(sec), list(mu.median_keV / mu.median_keV.iloc[0]), sd.BLUE,
           tips=[f'θ = {a:.0f}°: median {b:.2f} keV' for a, b in zip(mu.theta, mu.median_keV)])
    body = (sd.title('Principle 3: the track tells every layer how much material the particle crossed',
                     'a slab of thickness d crossed at angle θ: the deposit scales with the path d · sec θ')
            + sd.row(diag, P.svg('Heed vs angle'), gap=48))
    notes = """
<p>This is the simplest link between tracking and calorimetry: the MM measures θ in its gap, and each
counter's expected deposit is (deposit at normal incidence) × sec θ. Heed's muon deposit follows sec θ
slightly faster than linear (the MPV per unit length grows with thickness), slope 1.07 in log–log.</p>
<p>Two places it matters: (1) the gas dE/dx, normalised to normal incidence, is comparable across tracks;
(2) a particle that goes <i>through</i> the 20 mm plastic leaves MIP × sec θ = 3.4 × sec θ MeV, so without the
angle a steep through-going electron looks like a slower stopping one. Part 2 shows the angle buys back energy
resolution for exactly those tracks.</p>"""
    D.slide('path', body, notes, short='3 · Path')


def s_plastic(D, t):
    G = t['G']
    g = G[G.e_plas > 0.05]
    edges = np.arange(1.5, 10.01, 0.5)
    c = 0.5 * (edges[1:] + edges[:-1])
    b = np.digitize(g.lead_ke0, edges)
    med, q1, q3, n = [], [], [], []
    for i in range(1, len(edges)):
        v = g.e_plas[b == i]
        if len(v) >= 30:
            med.append(float(v.median()))
            q1.append(float(v.quantile(.16)))
            q3.append(float(v.quantile(.84)))
            n.append(len(v))
        else:
            med.append(np.nan)
            q1.append(np.nan)
            q3.append(np.nan)
            n.append(len(v))
    P = sd.Plot(1000, 600, x=(0, 10), y=(0, 8), xlabel='electron kinetic energy at the gap  (MeV)',
                ylabel='energy deposited in the plastic  (MeV)', title='Geant4: what the plastic sees')
    P.xticks(lin_ticks(0, 10, 1))
    P.yticks(lin_ticks(0, 8, 1))
    ok = ~np.isnan(med)
    P.band(list(c[ok]), list(np.array(q1)[ok]), list(np.array(q3)[ok]), sd.ORANGE, 0.18, tip='16–84 % of tracks')
    P.line(list(c[ok]), list(np.array(med)[ok]), sd.ORANGE,
           tips=[f'T {a - 0.25:.1f}-{a + 0.25:.1f} MeV: median {m:.2f} MeV (n = {k:,})'
                 for a, m, k in zip(c[ok], np.array(med)[ok], np.array(n)[ok])])
    P.line([1.2, 9.2], [0, 8], sd.MUT, 2, '8 6', markers=False, tip='T - 1.2 MeV: everything left after ~1.2 MeV '
                                                                    'is lost before the plastic (C2)')
    P.text(0.3, 7.4, 'dashed: T − 1.2 MeV, the deposit of an electron that stops', 21, sd.MUT)
    P.hline(3.4, sd.PURPLE, label='through: MIP × sec θ ≈ 3.4 MeV', tip='20 mm PVT, Bichsel MPV 3.40 MeV')
    side = sd.col(
        sd.callout('Up to ~5 MeV the electron <b>stops</b> in the 20 mm plastic: the deposit is its energy, minus '
                   '~1.2 MeV lost in the windows, the wall and the air on the way.', sd.ORANGE),
        sd.callout('Above ~6 MeV it <b>gets through</b>: the deposit saturates at MIP × path (~3.4–4 MeV), '
                   'and the liquid behind it fires.', sd.PURPLE),
        sd.callout('So the plastic measures energy only up to ~6 MeV; above that it says "fast", and the angle '
                   'sets the expected plateau.', sd.INK),
        w=600, gap=22)
    body = (sd.title('Principle 4: the 20 mm plastic is a calorimeter, up to the energy that gets through it',
                     'Geant4 neutron-beam events: plastic deposit (all particles) against the leading lepton\'s T at the gap')
            + sd.row(P.svg('plastic vs T'), side, gap=56))
    notes = """
<p><b>Sample.</b> Geant4 thermal-neutron events on the capsule nose (<code>stack_edep.py</code>), every track
crossing a gap with any energy in the matched plastic bar. The deposit is the total prompt energy in the bar
(the lepton plus its deltas and bremsstrahlung). The band is the 16–84 % range of tracks per bin.</p>
<p><b>Range.</b> The CSDA range of a 4 MeV electron in PVT is about 2 cm; with ~1.2 MeV lost before the plastic
(C2), electrons up to ~5 MeV at the gap stop inside. This is why the plastic is linear 3–8 MeV for stopping
electrons (the scintillator calorimetry note) but cannot measure an X17 leg above ~6 MeV on its own.</p>
<p><b>Design lesson.</b> A future calorimeter for 8–16 MeV legs needs ≥ 5–8 cm of plastic (or a denser crystal),
segmented so that the track picks the cell.</p>"""
    D.slide('plastic', body, notes, short='4 · Calorimeter')


def s_mmchain(D, t):
    ana = t['ana']
    st = ana['A']['sim_cosmic']['stages']
    fl = sd.flow([
        dict(label='ionisation', sub='~200 e⁻ along the track', color=sd.BLUE,
             tip=f'Heed; statistics {100 * st["ionisation"]:.1f} % of the deposit'),
        dict(label='drift', sub='v ≈ 34 (A) / 26 (C) µm/ns, diffusion', color=sd.GREY,
             tip='run_145 drift velocities from the reconstruction'),
        dict(label='avalanche', sub='Polya gain fluctuation', color=sd.GREY,
             tip=f'{100 * st["avalanche"]:.1f} %'),
        dict(label='strips x, y', sub='20 samples × 60 ns = 1200 ns window', color=sd.GREY),
        dict(label='road sum Q', sub='charge in a road around the track', color=sd.GREY,
             tip=f'noise + common mode + road {100 * st["noise_cm_road"]:.1f} %'),
        dict(label='÷ window c(f)', sub='drift cut by the window?', color=sd.ORANGE,
             tip=f'containment {100 * st["containment"]:.1f} % (A)'),
        dict(label='÷ gain map', sub='in-situ, per position', color=sd.GREEN),
        dict(label='keV', sub='÷ sec θ → dE/dx', color=sd.BLUE),
    ], size=21)
    # window diagram
    W, H = 840, 420
    o = []
    x0, x1 = 60, 700
    tx = lambda ns: x0 + ns / 1400 * (x1 - x0)  # noqa: E731
    o.append(sd.line(x0, 330, x1, 330, sd.MUT, 2))
    for ns in range(0, 1401, 200):
        o.append(sd.line(tx(ns), 330, tx(ns), 340, sd.MUT, 2))
        o.append(sd.T(tx(ns), 368, f'{ns}', 19))
    o.append(sd.T((x0 + x1) / 2, 405, 'time after the trigger  (ns)', 21, sd.INK))
    o.append(f'<rect x="{tx(0):.1f}" y="60" width="{tx(1200) - tx(0):.1f}" height="250" fill="{sd.GREY}" '
             f'fill-opacity="0.10" stroke="{sd.GREY}" stroke-dasharray="8 6"/>')
    o.append(sd.T(tx(600), 50, 'readout window: 1200 ns', 21, sd.MUT))
    D_ = t['D']
    for k, (arm, y) in enumerate((('A', 130), ('C', 230))):
        d = D_[D_.arm == arm]
        t0 = float(d.t0.median())
        fmed = float(d.f_obs.median())
        tdrift = (1200 - t0) / fmed
        c = ARMC[arm]
        o.append(f'<rect x="{tx(t0):.1f}" y="{y - 26}" width="{tx(t0 + tdrift) - tx(t0):.1f}" height="52" rx="8" '
                 f'fill="{c}" fill-opacity="0.35" stroke="{c}" stroke-width="2.5"'
                 f'{sd.tipattr(f"chamber {arm}: median t0 {t0:.0f} ns, full drift {tdrift:.0f} ns, f = {fmed:.2f}")}/>')
        o.append(sd.T(tx(t0) - 10, y + 8, arm, 26, c, 'end', 600))
        lab = 'whole drift in' if fmed > 1.3 else 'runs past the end'
        o.append(sd.T(tx(t0 + tdrift) + 12, y + 8, lab, 20, sd.INK, 'start'))
    diag = sd.svg(W, H, ''.join(o), 'readout window')
    wa, wc = t['win_A'], t['win_C']
    P = sd.Plot(760, 420, x=(0.5, 1.5), y=(0.2, 1.1), xlabel='drift fraction inside the window  f',
                ylabel='charge kept  c(f)', margin=(24, 30, 92, 104))
    P.xticks(lin_ticks(0.5, 1.5, 0.25, '{:.2f}'))
    P.yticks(lin_ticks(0.2, 1.0, 0.2, '{:.1f}'))
    for arm, w in (('A', wa), ('C', wc)):
        w = w[w.f < 1.6]
        cc = w.c / w.c.max()        # shown relative to the chamber's own plateau
        P.line(list(w.f), list(cc), ARMC[arm], tips=[f'{arm} f = {a:.2f}: c = {b:.2f} of the plateau (n = {n})'
                                                    for a, b, n in zip(w.f, cc, w.n)])
    body = (sd.title('How the MM turns charge into keV, and why chamber C is harder',
                     'the measurement chain, and the readout window against the drift time of each chamber')
            + fl + sd.row(diag, P.svg('window curve'), gap=40))
    notes = """
<p><b>The chain</b> (<code>mm_edep_sim.py</code>, <code>mm_charge.py</code>, <code>mm_edep_ana.py</code>): the
ionisation drifts to the mesh, is amplified, and is read on x and y strips in 20 samples of 60 ns. The estimator
sums the pedestal- and common-mode-subtracted charge in a road around the reconstructed track (both views), divides
by the share of the charge the window holds, c(f), and by k (ADC per keV, from the simulation).</p>
<p><b>The window.</b> f = (1200 ns − t0) / (30 mm / v) is the fraction of the drift that fits in the readout. In A
(v ≈ 34 µm/ns) the whole drift fits for almost every track; in C (v ≈ 26 µm/ns, a slower gas mixture/field) it
does not, and the last ~20 % of the track is extrapolated. That is the main reason C is less precise (16 % vs 10 %
in simulation). Design lesson: size the window to the slowest drift, with margin for t0.</p>"""
    D.slide('mmchain', body, notes, short='MM chain')


def s_decide(D, t):
    W, H = 1664, 640
    o = []

    def box(x, y, w, h, title, sub, c, tip=None):
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{sd.CARD}" stroke="{c}" '
                 f'stroke-width="3"{sd.tipattr(tip)}/>')
        o.append(sd.T(x + w / 2, y + 38, title, 24, sd.INK, weight=600))
        for k, line_ in enumerate(sub.split('|')):
            o.append(sd.T(x + w / 2, y + 70 + 26 * k, line_, 19, sd.MUT))

    def out(x, y, w, txt, c):
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="56" rx="28" fill="{c}" fill-opacity="0.16" stroke="{c}" '
                 f'stroke-width="2"/>')
        o.append(sd.T(x + w / 2, y + 36, txt, 21, sd.INK, weight=600))

    box(20, 250, 260, 140, 'a track', 'MM: x, y, θ, φ, depth|time since the flash', sd.GREY)
    # four questions
    qs = [(60, 'where from?', 'pointing: DCA to the capsule', sd.GREY, ['capsule', 'elsewhere']),
          (210, 'how many?', 'gas dE/dx ÷ sec θ', sd.BLUE, ['one', 'several / slow']),
          (360, 'stopped?', 'plastic vs MIP × sec θ; liquid', sd.ORANGE, ['stopped: E = plastic', 'through: T > 6 MeV']),
          (510, 'how much?', 'plastic + wall + angle (+ gas)', sd.GOLD, ['T ± 0.9 MeV'])]
    for y, q, how, c, outs in qs:
        o.append(sd.arrow(280, 320, 420, y + 60, sd.MUT, 2.5))
        box(420, y, 420, 120, q, how, c)
        for k, txt in enumerate(outs):
            yy = y + 4 + 62 * k if len(outs) > 1 else y + 32
            o.append(sd.arrow(840, y + 60, 960, yy + 28, c, 2.5))
            out(960, yy, 330, txt, c)
    o.append(sd.T(1330, 110, 'AUC 0.75 (geometry)', 21, sd.MUT, 'start'))
    o.append(sd.T(1330, 260, 'AUC 0.84 (only the gas)', 21, sd.MUT, 'start'))
    o.append(sd.T(1330, 410, 'Geant4-checked on data', 21, sd.MUT, 'start'))
    o.append(sd.T(1330, 575, 'angle: 1.4 → 1.1 MeV', 21, sd.MUT, 'start'))
    o.append(sd.T(1330, 600, 'for steep tracks', 21, sd.MUT, 'start'))
    body = (sd.title('Putting it together: four questions, each answered by a different combination',
                     'how a track is classified in this note; the simulation (next) says how well each works')
            + sd.svg(W, H, ''.join(o), 'decision flow'))
    notes = """
<p>Each question has a natural measurement, and each measurement needs the tracking. The pointing is pure
tracking. The particle count is the gas deposit divided by the path length the track gives. The stopped/through
decision compares the plastic deposit with what a through-going particle would leave along <i>this</i> track's path
(3.4 MeV × sec θ), backed by the liquid. The energy uses everything. In the code
(<code>track_passport.apply_to_data</code>) the last three are k-nearest-neighbour estimators trained on Geant4;
the class labels shown on the passports are simple cuts on their outputs.</p>"""
    D.slide('decide', body, notes, short='Putting it together')


# ----------------------------------------------------------------- simulation
def s_simacc(D, t):
    ana = t['ana']
    edges = np.linspace(-0.8, 0.8, 41)
    P = sd.Plot(900, 560, x=(-0.8, 0.8), y=(0, 0.16), xlabel='ln(measured / true gap deposit)',
                ylabel='fraction of tracks', title='Full simulation, cosmic muons')
    P.xticks([(v, f'{v:+.1f}' if v else '0') for v in np.arange(-0.8, 0.81, 0.4)])
    P.yticks(lin_ticks(0, 0.16, 0.04, '{:.2f}'))
    for arm in TP.ARMS:
        sp = t[f'simpairs_{arm}']
        lr = np.log(sp.E_hat / sp.E_true)
        s = ana[arm]['sim_cosmic']['rsd_used']
        step_hist(P, edges, _hist(lr, edges), ARMC[arm], 4,
                  tip=f'{arm}: robust sigma {s:.3f} (n = {len(sp):,})')
    rows = []
    for arm in TP.ARMS:
        st = ana[arm]['sim_cosmic']['stages']
        for k, lab in (('ionisation', 'ionisation statistics'), ('avalanche', 'avalanche gain'),
                       ('containment', 'window / road containment'), ('noise_cm_road', 'noise + common mode')):
            rows.append((f'{arm} · {lab}', 100 * st[k], ARMC[arm], f'{arm}: {lab} adds {100 * st[k]:.1f} % '
                                                                   '(in quadrature, stage by stage)'))
    side = sd.col(
        sd.p('what each step of the chain adds (%)', 24, sd.MUT),
        sd.hbars(rows, 12, width=300, h=24, label_w=330, fmt=lambda v: f'{v:.1f}', size=21),
        sd.callout(f'Total: <b>{100 * ana["A"]["sim_cosmic"]["rsd_used"]:.0f} %</b> (A) and '
                   f'<b>{100 * ana["C"]["sim_cosmic"]["rsd_used"]:.0f} %</b> (C) per track, against a Landau '
                   'spread of ~60 % FWHM: the chamber measures each track\'s deposit far better than the deposit '
                   'itself varies.', sd.BLUE, 23),
        w=760, gap=14)
    body = (sd.title('In simulation the MM measures each track\'s gas deposit to 10–16 %',
                     'Heed electrons through the digitiser (diffusion, Polya, real noise, production reconstruction)')
            + sd.row(P.svg('sim accuracy'), side, gap=40))
    notes = """
<p><b>Truth</b> is the ionisation in the gap (number of electrons × 27.5 eV). <b>Measured</b> is the full estimator on
the simulated waveforms: diffusion, Polya avalanche, the chamber's signal template and sharing kernel, real
noise taken from quiet run_145 triggers, and the production track reconstruction for the road.</p>
<p><b>Stages</b> are switched on one at a time; each bar is the extra spread in quadrature. On C the window
(containment) dominates, as the previous slide explains. A uses only tracks whose drift fits the window (54 %
of cosmics; ~90 % of beam tracks).</p>"""
    D.slide('simacc', body, notes, short='Sim: MM 10 %')


def s_simcomp(D, t):
    G = t['G']
    g = G[G.trig]
    n = len(g)
    ov = g.lead_ov.astype(str)
    org = [('capsule Al wall', (ov == 'He3Cap_Al').mean(), sd.RED),
           ('capsule CFRP', (ov == 'He3Cap_CFRP').mean(), sd.RED),
           ('MM itself (cathode, gas, mesh, windows)', ov.str.contains('Drift|Micromesh|GasWindow|PCB|Resistive').mean(),
            sd.GREY),
           ('air / elsewhere', (ov == 'World').mean(), sd.GREY)]
    op = g.lead_op.astype(str)
    proc = [('Compton scatter', (op == 'compt').mean(), sd.BLUE), ('pair conversion', (op == 'conv').mean(), sd.PURPLE),
            ('other (ionisation, photo)', (~op.isin(['compt', 'conv'])).mean(), sd.GREY)]
    edges = np.arange(0, 10.01, 0.5)
    P = sd.Plot(760, 520, x=(0, 10), y=(0, 0.16), xlabel='kinetic energy at the gap  (MeV)',
                ylabel='fraction of triggered tracks', title=f'Geant4: {n:,} triggered tracks')
    P.xticks(lin_ticks(0, 10, 2))
    P.yticks(lin_ticks(0, 0.16, 0.04, '{:.2f}'))
    step_hist(P, edges, _hist(g.lead_ke0, edges), sd.INK, 4, tip='leading lepton kinetic energy at the gap')
    step_hist(P, edges, _hist(g[g.multi].lead_ke0, edges) * g.multi.mean(), sd.GREEN, 3, '8 6',
              tip='of which several particles in the gap')
    hb = lambda rows: sd.hbars([(a, 100 * b, c, f'{100 * b:.1f} % of {n:,}') for a, b, c in rows], 100, width=260,  # noqa: E731
                               h=24, label_w=420, fmt=lambda v: f'{v:.0f} %', size=21)
    side = sd.col(sd.p('where the lepton was made', 23, sd.MUT), hb(org),
                  sd.p('how', 23, sd.MUT), hb(proc),
                  sd.p(f'{100 * g.multi.mean():.0f} % carry more than one particle through the gap '
                       '(dominant one &lt; 80 % of the gas deposit)', 23, sd.GREEN), w=820, gap=12)
    body = (sd.title('What the triggered tracks are: Compton electrons from capture γ in the capsule, 3–8 MeV',
                     'Geant4 thermal neutrons on the capsule; the arm\'s trigger (wall AND plastic) applied with the measured thresholds')
            + sd.row(P.svg('G4 T'), side, gap=40))
    notes = """
<p>A thermal neutron captured in the capsule's aluminium emits a γ cascade up to 7.7 MeV; the γ Compton-scatters
or converts in the capsule wall and the lepton flies to the arm. The trigger (a wall group AND a plastic bar above
threshold) keeps leptons that reach the plastic, i.e. above ~2.5–3 MeV at the gap.</p>
<p>These are the truths the measurements will be asked to recover: T (energy), several particles in the gap,
pair conversion vs Compton, capsule vs elsewhere. e⁺ vs e⁻ cannot be recovered at all without a magnetic field.</p>"""
    D.slide('simcomp', body, notes, short='Sim: sample')


def s_simmulti(D, t):
    pts = t['pts']
    p = pts[pts.source == 'Geant4']
    edges = np.geomspace(0.3, 8, 36)
    panels = []
    g4t = t['TPt'][t['TPt'].source == 'Geant4']
    for col, lab, au in (('mm', 'MM gas dE/dx / median', g4t.auc_multi_mm.mean()),
                         ('wall', 'wall dE/dx / median', g4t.auc_multi_wall.mean())):
        P = sd.Plot(800, 560, x=(0.3, 8, 'log'), y=(0, 0.16), xlabel=lab, ylabel='fraction of tracks',
                    title=f'{"MM gas" if col == "mm" else "SiPM wall"}: AUC {au:.2f}')
        P.xticks([(v, f'{v:g}') for v in (0.3, 0.5, 1, 2, 4, 8)])
        P.yticks(lin_ticks(0, 0.16, 0.04, '{:.2f}'))
        for m, nm, c in ((False, 'one lepton', sd.GREY), (True, 'several particles', sd.GREEN)):
            v = p[p.multi == m][col]
            step_hist(P, edges, _hist(v, edges), c, 4, tip=f'{nm}: n = {len(v):,}')
        panels.append(P.svg(col))
    body = (sd.title('Simulation: the gas sees the second particle, the wall 10 cm further on does not',
                     'Geant4 triggered tracks with the measured resolutions; each deposit at normal incidence, '
                     'normalised to its median')
            + sd.row(*panels, gap=44)
            + sd.legend([('one lepton in the gap', sd.GREY), ('several particles in the gap', sd.GREEN)], 22))
    notes = """
<p><b>Why the wall cannot see it.</b> The second particle is usually soft: a delta electron, an Auger/photo
electron, or the low-energy partner of a Compton or pair process. It ranges out within centimetres of gas or
material and never reaches the wall 97 mm away. The MM samples the particles exactly where they are produced
together.</p>
<p><b>AUC</b> is the probability that a random multi-particle track scores higher than a random single one
(0.5 = no information, 1 = perfect). 0.84 means the gas deposit alone ranks multi-particle tracks well but not
perfectly (the single-particle Landau tail overlaps, slide 4).</p>"""
    D.slide('simmulti', body, notes, short='Sim: 2 particles')


def s_simenergy(D, t):
    EP = t['EP']
    panels = []

    def w68(r):
        return 0.5 * float(np.subtract(*np.quantile(r, [.84, .16])))

    for var, edges, xr, xl, ttl in (('sec', [1.0, 1.03, 1.06, 1.1, 1.15, 1.22, 1.32, 1.6], (1, 1.45),
                                     'sec θ of the track', 'vs track angle'),
                                    ('T', [2, 3, 3.75, 4.5, 5.25, 6, 7, 9], (2, 9),
                                     'true T at the gap  (MeV)', 'vs energy')):
        P = sd.Plot(560, 520, x=xr, y=(0, 2.4), xlabel=xl,
                    ylabel='68 % width of T̂ − T  (MeV)' if var == 'sec' else '', title=f'resolution {ttl}')
        P.xticks(lin_ticks(1, 1.4, 0.1, '{:.1f}') if var == 'sec' else lin_ticks(2, 9, 1))
        P.yticks(lin_ticks(0, 2.4, 0.4, '{:.1f}'))
        b_ = np.digitize(EP[var], edges)
        for f in TP.FEATURES:
            xs, ys, tp = [], [], []
            for i in range(1, len(edges)):
                m = b_ == i
                if m.sum() < 60:
                    continue
                r = (EP[f'T_{f}'] - EP['T'])[m]
                xs.append(float(EP[var][m].median()))
                ys.append(w68(r))
                tp.append(f'{f}: {var} {edges[i - 1]:g}-{edges[i]:g}, width {ys[-1]:.2f} MeV (n = {int(m.sum()):,})')
            P.line(xs, ys, FEAT_C[f], 3.5 if 'angle' in f else 2.5, None if 'angle' in f else '8 6', tips=tp)
        panels.append(P.svg(ttl))
    EN = t['EN']
    subs = ['all', 'sec >= 1.1', 'several', 'T < 3 MeV']
    labs = {'all': 'all tracks', 'sec >= 1.1': 'steep (> 25°)', 'several': 'several particles', 'T < 3 MeV': 'T < 3 MeV'}
    B = sd.Plot(520, 520, x=(-0.5, len(subs) - 0.5), y=(0, 1.9), ylabel='68 % width of T̂ − T  (MeV)',
                title='resolution by feature set', margin=(24, 20, 92, 96))
    B.yticks(lin_ticks(0, 1.8, 0.3, '{:.1f}'))
    B.xticks([(i, labs[s].replace(' particles', '').replace('tracks', '')) for i, s in enumerate(subs)])
    feats = list(TP.FEATURES)
    for i, s in enumerate(subs):
        for k, f in enumerate(feats):
            v = float(EN[(EN.features == f) & (EN.subset == s)].w68.iloc[0])
            B.vbar((B.X(i) + (k - 1.5) * 24,), v, 21, FEAT_C[f], tip=f'{labs[s]} · {f}: {v:.2f} MeV')
    body = (sd.title(f'Simulation: the track angle buys back the energy of steep tracks '
                     f'({en(t, "plastic", "sec >= 1.1"):.1f} → {en(t, "plastic+wall+angle", "sec >= 1.1"):.1f} MeV)',
                     'kinetic energy at the gap from the counters, with and without the tracking; Geant4 triggered '
                     'tracks, 5-fold cross-validated kNN')
            + sd.row(*panels, B.svg('bars'), gap=24)
            + sd.legend([(f, FEAT_C[f], 'box') for f in feats], 21))
    notes = f"""
<p><b>Why the angle helps.</b> A steep track that gets through the plastic leaves 3.4 × sec θ MeV, which the plastic
alone cannot tell from a slower electron stopping in it; given θ, the estimator knows which plateau to expect.
Left: without the angle (dashed) the energy width grows with the track angle; with it (solid) it stays flat. Middle: the width against the true energy; above ~6 MeV every feature set loses because the electron leaves the plastic.</p>
<p><b>The MM's share.</b> It adds nothing for fast tracks (the gas is minimum-ionising, slide 3), but helps the slow
ones (T &lt; 3 MeV: {en(t, "plastic+wall+angle", "T < 3 MeV"):.2f} → {en(t, "plastic+wall+angle+MM", "T < 3 MeV"):.2f} MeV)
whose gas deposit is rising. For tracks that never reach the plastic, the MM halves the vertex-energy width of
1–3.5 MeV capsule electrons (1.7 → 0.45–0.8 MeV; <code>mm_edep_combo</code> C2 study).</p>
<p><b>Ceiling.</b> Above ~6 MeV every set loses: the electron leaves the plastic. That is the 20 mm thickness, not the
method.</p>"""
    D.slide('simenergy', body, notes, short='Sim: energy')


def s_simorigin(D, t):
    OR = t['OR']
    rows = []
    cols = {'energies (plastic, wall, MM)': sd.ORANGE, 'tracking (position, direction)': sd.GREY,
            'tracking + energies': sd.BLUE}
    for target, lab in (('capsule', 'made in the capsule'), ('conv', 'pair conversion (not Compton)'),
                        ('multi', 'several particles in the gap')):
        for f, c in cols.items():
            a = orr(t, target, f, 'all gap tracks')
            b = orr(t, target, f, 'triggered')
            rows.append((f'{lab} · {f.split(" (")[0]}', a, c, f'all gap tracks AUC {a:.3f}; triggered only {b:.3f}'))
    P = sd.Plot(1100, 600, x=(0.5, 0.9), y=(-0.5, len(rows) - 0.5), xlabel='AUC (0.5 = no information, 1 = perfect)',
                margin=(24, 30, 92, 520))
    P.xticks(lin_ticks(0.5, 0.9, 0.1, '{:.1f}'))
    for i, (lab, a, c, tp) in enumerate(rows[::-1]):
        y0 = P.Y(i)
        P.raw(f'<rect x="{P.X(0.5):.1f}" y="{y0 - 16:.1f}" width="{P.X(a) - P.X(0.5):.1f}" height="32" rx="5" '
              f'fill="{c}"{sd.tipattr(tp)}/>')
        P.raw(sd.T(P.x0 - 14, y0 + 8, lab, 21, sd.INK, 'end'))
        P.raw(sd.T(P.X(a) + 10, y0 + 8, f'{a:.2f}', 21, sd.INK, 'start', 600))
    side = sd.col(
        sd.callout('<b>Origin</b> is a geometry question: the tracking points back to the capsule.', sd.GREY, 23),
        sd.callout('<b>Several particles</b> is an energy question, answered by the gas.', sd.ORANGE, 23),
        sd.callout('<b>Conversion vs Compton</b> is hard for one arm: a pair\'s partner usually goes elsewhere.', sd.PURPLE,
                   23),
        w=440, gap=20)
    body = (sd.title('Simulation: which measurement answers which question',
                     'Geant4, every track crossing a gap; cross-validated kNN, scored by AUC')
            + sd.row(P.svg('origin'), side, gap=36))
    notes = """
<p>Each question is a Geant4 truth: <b>capsule</b> = the lepton was created in the capsule's Al or CFRP;
<b>conversion</b> = its creator process was γ → e⁺e⁻; <b>several</b> = the dominant particle carries &lt; 80 % of the
gap deposit. Features: tracking = position and direction at the gap; energies = plastic, wall, MM deposits
(smeared with the measured resolutions). Triggered tracks (hover) are 91 % capsule-born already, so the origin AUC
is lower there: little left to separate.</p>
<p>In the data the pointing is used directly as the distance of closest approach to the capsule axis (DCA).</p>"""
    D.slide('simorigin', body, notes, short='Sim: questions')


# ----------------------------------------------------------------- real data
def s_data(D, t):
    Dd = t['D']
    S = t['S']
    panels = []
    eb = np.geomspace(0.3, 15, 31)
    P = sd.Plot(540, 500, x=(0.3, 15, 'log'), y=(0, 0.08), xlabel='neutron energy from the TOF  (meV)',
                ylabel='fraction of tracks', title='when: E_n')
    P.xticks([(v, f'{v:g}') for v in (0.3, 1, 3, 10)])
    P.yticks(lin_ticks(0, 0.08, 0.02, '{:.2f}'))
    for arm in TP.ARMS:
        step_hist(P, eb, _hist(Dd[Dd.arm == arm].E_n_meV, eb), ARMC[arm], 3.5, tip=f'{arm}: {int((Dd.arm == arm).sum()):,} tracks')
    panels.append(P.svg('En'))
    tb = np.arange(0, 56, 2.5)
    P = sd.Plot(540, 500, x=(0, 55), y=(0, 0.14), xlabel='track angle θ to the chamber normal  (°)',
                title='direction')
    P.xticks(lin_ticks(0, 50, 10))
    P.yticks(lin_ticks(0, 0.14, 0.02, '{:.2f}'))
    for arm in TP.ARMS:
        step_hist(P, tb, _hist(Dd[Dd.arm == arm].theta_deg, tb), ARMC[arm], 3.5)
    panels.append(P.svg('theta'))
    db = np.arange(0, 200.1, 10)
    P = sd.Plot(540, 500, x=(0, 200), y=(0, 0.35), xlabel='distance from the capsule axis  (mm)',
                title='pointing: DCA')
    P.xticks(lin_ticks(0, 200, 50))
    P.yticks(lin_ticks(0, 0.35, 0.05, '{:.2f}'))
    for arm in TP.ARMS:
        d = Dd[Dd.arm == arm]
        hh = np.histogram(d.dca_axis_mm, db)[0] / len(d)
        step_hist(P, db, hh, ARMC[arm], 3.5,
                  tip=f'{arm}: {100 * d.to_capsule.mean():.0f} % within {TP.CAPSULE_DCA_MM:.0f} mm, '
                      f'{100 * (d.dca_axis_mm > 200).mean():.1f} % beyond 200 mm (not drawn)')
    panels.append(P.svg('dca'))
    body = (sd.title(f'The real data: {S["n_data"]:,} beam tracks with every layer, '
                     f'{S["n_data_mm"]:,} with a usable gas deposit',
                     'run_145, ≥ 12 ms after the γ-flash, single tracks on their own arm\'s trigger')
            + sd.row(*panels, gap=20) + sd.legend([('chamber A', sd.BLUE), ('chamber C', sd.ORANGE)], 22))
    notes = f"""
<p><b>Selection.</b> Stage-3 fullpass tracks of run_145 (3 sub-runs), gated, the only track of their arm in the
event, ≥ 12 ms after the flash (the MM gain has recovered, next slides), on their own arm's hardware trigger.
The MM deposit is usable when the charge is not saturated, the gain map covers the position, and (A only) the drift
is contained (f &gt; 1.35).</p>
<p><b>Time of flight.</b> At 19.5 m, 12–75 ms means 0.4–14 meV neutrons: everything here is sub-thermal, so the
TOF is recorded per track but there is no resonance structure to show.</p>
<p><b>Pointing.</b> Most tracks pass within a few cm of the capsule axis: the capture γ come from the capsule, as
Geant4 says. The tail is tracks from elsewhere (and A–C differences in acceptance).</p>"""
    D.slide('data', body, notes, short='Data sample')


def s_beamcal(D, t):
    bc = t['bc']
    c = t['S']['cal']
    panels = []
    for arm in TP.ARMS:
        b = bc[arm]
        e = b['edges']
        ymax = 0.10
        P = sd.Plot(800, 540, x=(1.5, 40, 'log'), y=(0, ymax), xlabel='gas deposit per 30 mm at normal incidence  (keV)',
                    ylabel='fraction of tracks' if arm == 'A' else '',
                    title=f'{arm}: data/sim gain {c[arm]["scale"]:.2f} ± {c[arm]["scale_err"]:.2f}, '
                          f'σ {100 * c[arm]["sigma_total"]:.0f} %')
        P.xticks([(v, f'{v:g}') for v in (2, 3, 5, 10, 20, 40)])
        P.yticks(lin_ticks(0, ymax, 0.02, '{:.2f}'))
        step_hist(P, e, b['truth'], sd.GREY, 2.5, '8 6', tip=f'simulation truth (Heed ionisation), median '
                                                            f'{b["med_truth"]:.2f} keV')
        step_hist(P, e, b['sim'], G4C, 3.5, tip=f'simulation, reconstructed, n = {b["n_sim"]:,}, median {b["med_sim"]:.2f}')
        cx = np.sqrt(e[1:] * e[:-1])
        P.points(list(cx), list(b['data']), ARMC[arm], r=5,
                 tips=[f'{a:.1f} keV: {100 * v:.1f} % of {b["n_data"]:,} data tracks' for a, v in zip(cx, b['data'])])
        panels.append(P.svg(f'cal {arm}'))
    body = (sd.title('Real chamber, step 1: calibrated on the beam simulation, the gas reads the MIP value',
                     'run_145 electrons (after the in-situ gain map, ÷ the fitted data/sim gain) against the beam '
                     'simulation\'s reconstruction and its truth')
            + sd.row(*panels, gap=44)
            + sd.legend([('data (A blue, C orange)', sd.BLUE, 'dot'), ('simulation, reconstructed', G4C),
                         ('simulation truth', sd.GREY, 'dash')], 22))
    notes = f"""
<p><b>Calibration.</b> The ADC-per-keV constant k and the window curve come from the beam simulation, generated at
the beam gain (A {c["A"]["k_beam"]:.0f}, C {c["C"]["k_beam"]:.0f} ADC/keV); the earlier cosmic k left a −0.46 bias on
beam. The data/sim gain is a forward fold of the data distribution onto the simulation's (scale × a Gaussian extra
spread σ_x), fitted on the core: A {c["A"]["scale"]:.3f} (σ_x {c["A"]["sigma_x"]:.2f}), C {c["C"]["scale"]:.3f}
(σ_x {c["C"]["sigma_x"]:.2f}).</p>
<p><b>Result.</b> The median deposit is {c["A"]["mip_keV"]:.2f} / {c["C"]["mip_keV"]:.2f} keV per 30 mm in the truth the
data are calibrated to; the data's shape (peak, width, Landau tail) is the simulation's. Per-track resolution =
simulation chain ⊕ σ_x: A {100 * c["A"]["sigma_total"]:.0f} %, C {100 * c["C"]["sigma_total"]:.0f} %.</p>
<p><b>Caveat.</b> This calibration is relative to the simulation's electron spectrum; an absolute check needs a
known source in the gas (e.g. ⁵⁵Fe) — a recommendation for the next set-up.</p>"""
    D.slide('beamcal', body, notes, short='MM calibration')


def s_gainmap(D, t):
    P = sd.Plot(900, 560, x=(0, 150), y=(-0.05, 0.5), xlabel='distance between two tracks on the chamber  (mm)',
                ylabel='correlation of their charge residuals', title='cosmic muons, run_149')
    P.xticks(lin_ticks(0, 150, 25))
    P.yticks(lin_ticks(0, 0.5, 0.1, '{:.1f}'))
    P.hline(0, sd.MUT, dash=None, w=1)
    for arm in TP.ARMS:
        s = t[f'spatial_{arm}']
        for which, dash in (('raw', None), ('corrected', '8 6')):
            g = s[s.which == which]
            x = list(0.5 * (g.d_lo + g.d_hi))
            P.line(x, list(g.r), ARMC[arm], 3.5, dash,
                   tips=[f'{arm} {which}: {a:.0f}-{b:.0f} mm, r = {r:.3f} ({n:,} pairs)'
                         for a, b, r, n in zip(g.d_lo, g.d_hi, g.r, g.n)])
    an = t['ana']
    hv = an['A']['halves_cosmic']
    hb = an['A']['halves_beam']
    rows = [('cosmics, raw', hv['sigma_g_raw'], sd.GREY, 'A, truth-free halves test'),
            ('cosmics, after map', hv['sigma_g_corrected'], sd.BLUE),
            ('cosmics, centre', hv['sigma_g_corrected_centre'], sd.BLUE),
            ('beam, raw', hb['sigma_g_raw'], sd.GREY), ('beam, after beam map', hb['sigma_g_corrected'], sd.GREEN)]
    side = sd.col(
        sd.callout('Nearby tracks share their charge excess: the real chamber\'s gain varies with position '
                   '(a structure along x, in both views: the amplification, not the strips).', sd.BLUE, 23),
        sd.callout('An in-situ map from the tracks themselves removes it (dashed).', sd.GREEN, 23),
        sd.p('what is left: random per-track gain spread σ_g (A)', 22, sd.MUT),
        sd.hbars([(a, b, c) for a, b, c, *_ in rows], 0.5, width=230, h=22, label_w=260,
                 fmt=lambda v: f'{v:.2f}', size=21),
        w=660, gap=16)
    body = (sd.title('Real chamber, step 2: a position-dependent gain, removed by an in-situ map',
                     'correlation of the gas-deposit residuals of track pairs vs their separation; solid raw, dashed '
                     'after the cross-validated map')
            + sd.row(P.svg('spatial'), side, gap=40))
    notes = """
<p><b>The finding.</b> Two tracks within 4 mm of each other have correlated residuals (r = 0.43 A, 0.33 C) and the
correlation dies out over 20–40 mm. Pairs at the same x but far apart in y stay correlated (0.26) in both the x and
y views; pairs at the same y do not (0.01). So it is an amplification-gap structure along x, not channel
electronics. The calibration bundles have all channel gains at exactly 1.0, so the simulation has none of it.</p>
<p><b>The map.</b> For each track, the median residual of its 25 nearest neighbours (cross-validated: never its own
fold). After it, the correlation is gone. The cosmic map does not transfer to beam (different gain), so the beam
needs its own map.</p>
<p><b>Halves test.</b> Split each contained track's drift in two halves; their common excess over the simulation is
a per-track gain fluctuation, measured without any truth. After the map: 0.27 (A cosmics), 0.09 (beam).</p>"""
    D.slide('gainmap', body, notes, short='Gain map')


def s_flash(D, t):
    P = sd.Plot(1100, 560, x=(0, 80), y=(0, 1.3), xlabel='time since the γ-flash  (ms)',
                ylabel='median gas deposit / plateau', title='beam tracks, run_145, all times')
    P.xticks(lin_ticks(0, 80, 10))
    P.yticks(lin_ticks(0, 1.2, 0.2, '{:.1f}'))
    P.raw(f'<rect x="{P.X(0):.1f}" y="{P.y0}" width="{P.X(12) - P.X(0):.1f}" height="{P.ph}" fill="{sd.RED}" '
          f'fill-opacity="0.07"/>', back=True)
    P.text(1, 1.22, 'excluded: < 12 ms', 21, sd.RED)
    P.hline(1, sd.MUT)
    for arm in TP.ARMS:
        f = t[f'flash_{arm}']
        x = list(0.5 * (f.ms_lo + f.ms_hi))
        P.line(x, list(f.gain), ARMC[arm], tips=[f'{arm} {a:g}-{b:g} ms: gain {g:.2f}, x/y {xy:.2f} (n = {n:,})'
                                                for a, b, g, xy, n in zip(f.ms_lo, f.ms_hi, f.gain, f.xy, f.n)])
    side = sd.col(
        sd.callout('The flash leaves the chamber at ~30 % gain for 8 ms; it recovers by 12 ms and is flat (±3 %) '
                   'to 80 ms.', sd.RED, 23),
        sd.callout('Before ~9 ms the x and y views disagree (x/y 0.2–0.7): the charge is not a deposit '
                   'measurement there.', sd.GREY, 23),
        sd.callout('Lesson: a beam set-up needs a flash-proof MM (gating, a lower field during the flash) or a '
                   'monitored gain.', sd.BLUE, 23), w=520, gap=20)
    body = (sd.title('Real chamber, step 3: the γ-flash blinds the gas measurement for ~10 ms',
                     'median gas deposit (normalised to ≥ 12 ms) against time since the flash')
            + sd.row(P.svg('flash'), side, gap=40))
    notes = """
<p>Same estimator, same calibration, every time bin. The gain is relative to the ≥ 12 ms plateau. Times before
12 ms are excluded from everything in this note. The scintillators have their own, smaller rate effect: the plastic
PMTs sag 3–8 % in the first 25 ms (scintillator calorimetry note).</p>"""
    D.slide('flash', body, notes, short='Flash')


def s_pathdata(D, t):
    PC, PL = t['PC'], t['PL']
    panels = []
    P = sd.Plot(800, 560, x=(1, 1.55), y=(0.95, 1.6), xlabel='sec θ (from the MM track)',
                ylabel='deposit / first bin', title='cosmic muons: MM gas and wall')
    P.xticks(lin_ticks(1, 1.5, 0.1, '{:.1f}'))
    P.yticks(lin_ticks(1, 1.6, 0.1, '{:.1f}'))
    P.line([1, 1.55], [1, 1.55], sd.MUT, 2, '8 6', markers=False, tip='∝ sec θ')
    for _, r in PC.iterrows():
        x, y = json.loads(r.sec_med), json.loads(r.dep_med)
        mk = 'circle' if r.layer == 'MM gap' else 'diamond'
        P.line(x, y, ARMC[r.arm], 2 if r.layer == 'wall' else 3, None if r.layer == 'MM gap' else '6 5', r=7,
               marker=mk, tips=[f'{r.arm} {r.layer}: sec {a:.3f} → {b:.3f}' for a, b in zip(x, y)])
    panels.append(P.svg('cosmic path'))
    P = sd.Plot(800, 560, x=(1, 1.55), y=(0.95, 1.6), xlabel='sec θ in the gap', title='beam electrons: the wall')
    P.xticks(lin_ticks(1, 1.5, 0.1, '{:.1f}'))
    P.yticks(lin_ticks(1, 1.6, 0.1, '{:.1f}'))
    P.line([1, 1.55], [1, 1.55], sd.MUT, 2, '8 6', markers=False)
    for _, r in PL[PL.layer == 'wall'].iterrows():
        x, y = np.array(json.loads(r.sec_med)), np.array(json.loads(r.dep_med))
        y = y / y[0]
        k = x <= 1.55
        x, y = list(x[k]), list(y[k].clip(max=1.6))
        if r.source == 'data':
            P.points(x, y, ARMC[r.arm], r=7, tips=[f'{r.arm} data: sec {a:.3f} → {b:.3f}' for a, b in zip(x, y)])
        else:
            P.line(x, y, ARMC[r.arm], 3, '10 6', markers=False, tip=f'{r.arm} Geant4, slope {r.slope:.2f}')
    panels.append(P.svg('beam wall'))
    mm = ', '.join(f'{r.arm} {r.slope:.2f} ± {r.err:.2f}' for _, r in PC[PC.layer == 'MM gap'].iterrows())
    wl = ', '.join(f'{r.arm} {r.slope:.2f} ± {r.err:.2f}' for _, r in PC[PC.layer == 'wall'].iterrows())
    bw = ', '.join(f'{r.arm} {r.source} {r.slope:.2f}' for _, r in PL[PL.layer == 'wall'].iterrows())
    body = (sd.title('Tracking and calorimetry agree on the path: muons follow sec θ, electrons scatter as Geant4 says',
                     f'power-law slope of the median deposit vs sec θ · cosmics: MM {mm} (Heed 1.07), wall {wl} · '
                     f'beam wall: {bw}')
            + sd.row(*panels, gap=44)
            + sd.legend([('MM gas (cosmics)', sd.BLUE, 'dot'), ('wall (cosmics, dashed)', sd.BLUE, 'dash'),
                         ('beam data', sd.ORANGE, 'dot'), ('Geant4 (dashed)', sd.ORANGE, 'dash'),
                         ('∝ sec θ', sd.MUT, 'dash')], 21))
    rows = ''.join(f'<tr><td>{r.arm}</td><td>{r.layer}</td><td>{r.source}</td><td>{r.n:,}</td><td>{r.slope:.2f} ± {r.err:.2f}'
                   f'</td><td>{r.slope_core:.2f}</td></tr>' for _, r in PL.iterrows())
    notes = f"""
<p><b>Cosmics</b> (left) are the clean test: muons do not scatter, so the angle measured in the gap holds at the
wall 10 cm later. Both deposits grow as the path. The wall points are the MPV of a truncated Landau fit in three
path bins (<code>wall_plastic</code>), the MM points the median deposit after the gain map.</p>
<p><b>Beam electrons</b> (right) are not: a few-MeV electron scatters in the MM's own material and 10 cm of air, so
its angle at the wall is only loosely tied to the gap angle; Geant4 predicts almost no dependence and the data show
a weak one. The gas deposit itself does follow the path up to sec θ ≈ 1.25 in both data and Geant4; the steepest
tracks are enriched in slow and multi-particle tracks and jump.</p>
<table><tr><th>arm</th><th>layer</th><th>source</th><th>n</th><th>slope</th><th>slope, sec &lt; 1.3</th></tr>{rows}</table>"""
    D.slide('pathdata', body, notes, short='Path, data')


def s_spectra(D, t):
    Sp = t['spectra']
    panels = []
    for var, xr, xt, lab in (('e_plas', (2, 9), range(2, 10), 'plastic energy  (MeV)'),):
        for arm in TP.ARMS:
            g = Sp[(Sp.arm == arm) & (Sp['var'] == var)]
            if not len(g):
                continue
            c = 0.5 * (g.lo + g.hi)
            m = (c >= xr[0]) & (c <= xr[1])
            d = g.data / g.data[m].sum()
            s = g.g4 / g.g4[m].sum()
            ymax = float(max(d[m].max(), s[m].max()) * 1.2)
            sc = t['scale']
            sr = sc[(sc.arm == arm) & (sc['var'] == ('e_plas' if var == 'e_plas' else 'e_wall_fit')) & sc.subset.isna()]
            P = sd.Plot(800, 600, x=xr, y=(0, ymax), xlabel=lab, ylabel='fraction per bin' if arm == 'A' else '',
                        title=f'plastic, chamber {arm}: data / Geant4 = {float(sr.scale.iloc[0]):.2f}')
            P.xticks([(v, f'{v:g}') for v in xt][::2 if var == 'e_wall' else 1])
            P.yticks([(v, f'{v:.2f}') for v in np.arange(0, ymax, 0.04)])
            gi = g[m]
            step_hist(P, list(gi.lo) + [float(gi.hi.iloc[-1])], list(s[m]), G4C, 3, tip='Geant4, not rescaled')
            P.points(list(c[m]), list(d[m]), ARMC[arm], r=5,
                     tips=[f'{arm} {a:.2f}-{b:.2f} MeV: data {int(x):,}, Geant4 {int(y)}'
                           for a, b, x, y in zip(gi.lo, gi.hi, g.data[m], g.g4[m])])
            panels.append(P.svg(f'{var} {arm}'))
    body = (sd.title('The counters are on Geant4\'s scale: calibrated on muons, the plastic describes the electrons',
                     'beam electrons (points) vs Geant4 with the measured resolution (line), all times, no rescaling · '
                     'the wall: 0.95 / 0.96 of Geant4')
            + sd.row(*panels, gap=44))
    notes = """
<p>The wall reads 0.95 (A) and 0.96 (C) of Geant4, and its shape agrees once its own measured resolution is
fitted (shown in the scintillator calorimetry note, <i>scint-calorimetry</i>).</p>
<p>From the scintillator calorimetry study (<code>wall_plastic.py</code>, note <i>scint-calorimetry</i>): each layer
is calibrated once on cosmic muons and then predicts the beam electrons. Plastic bars A1, A2, C2 land at 1.02–1.05 of
Geant4; C1 reads 1.15 (unexplained, channel-specific), wall 0.95–0.96. The passport divides each counter by its
data/Geant4 scale before applying the simulation-trained estimators, so C1's offset is taken out.</p>"""
    D.slide('spectra', body, notes, short='Counters')


def s_mmwall(D, t):
    R = t['tp_pts']
    P = sd.Plot(960, 580, x=(0.55, 4.2, 'log'), y=(0.88, 1.12), xlabel='MM gas dE/dx / median',
                ylabel='median wall dE/dx / median', title='the wall\'s median in bins of the gas deposit')
    P.xticks([(v, f'{v:g}') for v in (0.6, 0.8, 1, 1.5, 2, 3, 4)])
    P.yticks(lin_ticks(0.9, 1.1, 0.05, '{:.2f}'))
    P.hline(1, sd.MUT)
    for k, arm in enumerate(TP.ARMS):
        for src, f in (('data', 0.985), ('Geant4', 1.015)):
            r = R[(R.arm == arm) & (R.source == src)]
            xs = list(r.mm * f * (1 + 0.02 * k))
            for x, y, e in zip(xs, r.wall_med, r.err):
                P.raw(sd.line(P.X(x), P.Y(max(y - e, 0.88)), P.X(x), P.Y(min(y + e, 1.12)), ARMC[arm], 2))
            P.points(xs, list(r.wall_med.clip(0.88, 1.12)), ARMC[arm], r=7 if src == 'data' else 6,
                     marker='circle' if src == 'data' else 'open',
                     tips=[f'{arm} {src}: gas {a:.2f}, wall {b:.3f} ± {e:.3f} (n = {n:,})'
                           for a, b, e, n in zip(r.mm, r.wall_med, r.err, r.n)])
    TPt = t['TPt']
    dat, g4 = TPt[TPt.source == 'data'], TPt[TPt.source == 'Geant4']
    side = sd.col(
        sd.callout('Flat in data and in Geant4: the gas and the wall see different particles.', sd.BLUE, 23),
        sd.table(['', 'data', 'Geant4'],
                 [['Spearman r', ' / '.join(f'{v:+.3f}' for v in dat.spearman), ' / '.join(f'{v:+.3f}' for v in g4.spearman)],
                  ['both high: lift', ' / '.join(f'{v:.2f}' for v in dat.lift), ' / '.join(f'{v:.2f}' for v in g4.lift)],
                  ['gas high', ' / '.join(f'{100 * v:.0f} %' for v in dat.p_mm_hi), ' / '.join(f'{100 * v:.0f} %' for v in g4.p_mm_hi)]],
                 size=22),
        sd.p('A / C. "High" = above 1.6 × median. Lift = P(both high) / (P(gas high) · P(wall high)).', 20, sd.MUT),
        w=640, gap=16)
    body = (sd.title('Real data behave as the simulation predicts: the gas tail does not show up in the wall',
                     'run_145 tracks with a gas deposit (filled) vs Geant4 with the measured resolutions (open); '
                     'error bars: error of the median')
            + sd.row(P.svg('mm vs wall'), side, gap=40))
    notes = """
<p>This is the data-side check of the simulation's claim (slide on two particles): if the extra gas deposit of
multi-particle tracks also reached the wall, the wall's median would rise with the gas deposit. It does not, in
either. The small joint excess ("lift") is the same in data and Geant4. The fraction of tracks with a high gas
deposit is somewhat lower in data than in Geant4 (12–13 % vs 15–17 %), consistent with Geant4's opt4 gap deposit
having a fatter tail than Heed (MM_EDEP.md).</p>"""
    D.slide('mmwall', body, notes, short='Gas vs wall')


def s_estimators(D, t):
    Dd, S = t['D'], t['Sim']
    G = t['G']
    G = G[G.trig]
    panels = []
    tb = np.arange(0, 10.01, 0.4)
    P = sd.Plot(800, 520, x=(0, 10), y=(0, 0.24), xlabel='estimated T at the gap  (MeV)', ylabel='fraction of tracks',
                title='energy estimate')
    P.xticks(lin_ticks(0, 10, 2))
    P.yticks(lin_ticks(0, 0.24, 0.06, '{:.2f}'))
    pb = np.linspace(0, 1, 21)
    Q = sd.Plot(800, 520, x=(0, 1), y=(0, 0.2), xlabel='P(several particles in the gap)', title='multi-particle tag')
    Q.xticks(lin_ticks(0, 1, 0.2, '{:.1f}'))
    Q.yticks(lin_ticks(0, 0.2, 0.05, '{:.2f}'))
    for arm in TP.ARMS:
        d = Dd[(Dd.arm == arm) & Dd.mm_ok]
        s = S[S.armL == arm]
        for PL_, col, e in ((P, 'T_hat', tb), (Q, 'p_multi', pb)):
            step_hist(PL_, e, _hist(s[col], e), ARMC[arm], 3, '10 6', tip=f'{arm} Geant4 (cross-validated)')
            cx = 0.5 * (e[1:] + e[:-1])
            hd = _hist(d[col], e)
            PL_.points(list(cx), list(hd), ARMC[arm], r=5,
                       tips=[f'{arm} data {a:.2f}: {100 * v:.1f} % of {len(d):,}' for a, v in zip(cx, hd)])
    panels = [P.svg('That'), Q.svg('pmulti')]
    rows = []
    for arm in TP.ARMS:
        d = Dd[Dd.arm == arm]
        g = G[G.armL == arm]
        rows.append([arm, f'{100 * (d.e_plas_g4 < 0.8 * 3.4 * d.sec).mean():.0f} %',
                     f'{100 * (g.e_plas < 0.8 * 3.4 * g.sec).mean():.0f} %',
                     f'{100 * d[d.mm_ok].p_multi.mean():.0f} %', f'{100 * g.multi.mean():.0f} %',
                     f'{100 * d.liq_fired.mean():.1f} %'])
    tab = sd.table(['arm', 'stopped early: data', 'Geant4', 'mean P(multi): data', 'Geant4 truth', 'liquid fired'],
                   rows, size=22)
    body = (sd.title('The simulation-trained estimators applied to real tracks reproduce Geant4\'s distributions',
                     'T̂ and P(several particles) per data track (points) vs the same estimators on Geant4 (dashed)')
            + sd.row(*panels, gap=44) + tab)
    notes = """
<p><b>How.</b> The estimators (k-nearest neighbours in plastic, wall, sec θ, gas deposit) are trained on the Geant4
triggered tracks and applied to each data track after its counter energies are put on Geant4's scale. The Geant4
curves are the same estimators, cross-validated, so they include the estimators' own smearing.</p>
<p><b>Stopped early</b> = plastic deposit below 0.8 × MIP × sec θ: a lepton that stopped in the plastic well below the
through-going plateau. C has many more (its geometry); Geant4 predicts that. The liquid fires behind 4 % of A tracks
and almost never behind C's (C's MIP is only ~1.3 × its zero-suppression threshold).</p>
<p><b>Mean P(multi)</b> in the data vs the Geant4 truth fraction is a consistency check only: P is a probability
from the simulation, so its mean is the simulation's multi-particle rate as seen through the data.</p>"""
    D.slide('estimators', body, notes, short='Estimators')


def _card(t, r) -> str:
    v = float(r.v_drift_um_ns) if np.isfinite(r.v_drift_um_ns) else 30.0
    prof = np.asarray(r.x_prof, float) + np.asarray(r.y_prof, float)
    tt = np.arange(prof.size) * 60.0
    dz = np.clip((tt - 140.0 - r.t0) * v / 1000.0, 0, 30)
    deps = {'MM': f'{r.E_mm:.1f} keV', 'wall': f'{r.e_wall_g4:.2f}', 'plastic': f'{r.e_plas_g4:.2f}',
            'liquid': f'{r.e_liq:.2f}' if r.liq_fired else ''}
    o, zx, uy = stack_svg(770, 380, track=(r.x_local, r.tanx), deposits=deps, prof=(dz, prof), small=True,
                          tip_layers=False)
    diag = sd.svg(770, 380, ''.join(o), 'track')
    pc = 'at the capsule' if r.to_capsule else 'off the capsule'
    lines = [('when', f'{r.t_since_flash_ns / 1e6:.1f} ms → E_n {r.E_n_meV:.2f} meV'),
             ('direction', f'θ {r.theta_deg:.0f}°, φ {r.phi_deg:.0f}°'),
             ('pointing', f'DCA {r.dca_axis_mm:.0f} mm: {pc}'),
             ('gas dE/dx', f'{r.E_mm:.1f} keV ± {100 * r.sigma_mm:.0f} % ({r.q_rel:.2f} × typ.)'),
             ('wall', f'{r.e_wall_g4:.2f} MeV'),
             ('plastic', f'{r.e_plas_g4:.2f} MeV (MIP×sec {3.4 * r.sec:.1f})'),
             ('liquid', f'{r.e_liq:.1f} MeV: through' if r.liq_fired else 'silent: stopped'),
             ('energy', f'T ≈ {r.T_hat:.1f} ± {r.T_sd:.1f} MeV'),
             ('particles', f'P(several) {r.p_multi:.2f}')]
    cell = lambda a, b: (f'<div style="display:flex;gap:10px"><p style="width:105px;flex:none;font-size:19px;'  # noqa: E731
                         f'color:{sd.MUT}">{a}</p><p style="font-size:19px;line-height:1.25">{b}</p></div>')
    tab = ('<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 18px">'
           + ''.join(cell(a, b) for a, b in lines) + '</div>')
    head = (f'<p style="font-size:24px;font-weight:600">{r.example}: {r.arm} · event {int(r.event_id)}</p>'
            f'<p style="font-size:20px;color:{sd.MUT}">class: {r.cls}</p>')
    return sd.card(head + diag + tab, w=810, pad=18)


def s_cards(D, t):
    EX = t['EX']
    want = [('through the plastic', 'stopped in plastic'), ('several particles', 'steep track')]
    first = {}
    for _, r in EX.iterrows():
        first.setdefault(r.example, r)
    for k, pair in enumerate(want):
        cs = [_card(t, first[w]) for w in pair if w in first]
        body = (sd.title('Track passports: ' + (' and '.join(pair)),
                         'real run_145 tracks; side view to scale with the track extrapolated from the MM; blue dots: '
                         'the drift charge profile along the track; numbers on the Geant4 energy scale')
                + sd.row(*cs, gap=24))
        notes = """
<p>Each card is one real track with every measurement the set-up makes. The energy and P(several) come from the
Geant4-trained estimators (model-dependent). The gas deposit's uncertainty is the per-track resolution of its
chamber; the counters' energies carry their measured resolutions (wall ~16–18 %, plastic ~4–7 % at the MIP).</p>
<p>The drift profile: the charge in each 60 ns sample, placed at depth (t − 140 ns − t0) × v; it should lie along
the track and fill the 30 mm for a contained track.</p>"""
        D.slide(f'cards{k}', body, notes, short=f'Passports {k + 1}', pad='72px 64px 72px')


def s_perf(D, t):
    c = t['S']['cal']
    ana = t['ana']
    PC = t['PC']
    g4t = t['TPt'][t['TPt'].source == 'Geant4']
    mm_c = ', '.join(f'{r.arm} {r.slope:.2f}' for _, r in PC[PC.layer == 'MM gap'].iterrows())
    rows = [
        ['direction', 'MM track (x, y, drift depth)', '—', 'k-calibrated angle, per run',
         'path length in each layer; pointing'],
        ['origin', 'pointing (DCA to capsule)', f'AUC {orr(t, "capsule", "tracking (position, direction)", "all gap tracks"):.2f}',
         'DCA distribution as Geant4', 'capsule vs elsewhere'],
        ['gas dE/dx', 'MM charge, window, gain map',
         f'{100 * ana["A"]["sim_cosmic"]["rsd_used"]:.0f} / {100 * ana["C"]["sim_cosmic"]["rsd_used"]:.0f} %',
         f'{100 * c["A"]["sigma_total"]:.0f} / {100 * c["C"]["sigma_total"]:.0f} % (beam)',
         f'particle count (AUC {g4t.auc_multi_mm.mean():.2f}); slow tag'],
        ['path check', 'deposit vs sec θ', 'Heed 1.07', f'cosmics {mm_c}', 'tracks and counters agree'],
        ['wall dE/dx', 'SiPM wall, muon-calibrated', '—', '16–18 % at MIP; 0.95 of Geant4', 'second dE/dx sample'],
        ['energy', 'plastic + angle (+ wall, gas)', f'{en(t, "plastic+wall+angle", "all"):.2f} MeV (68 %)',
         'spectra at 1.02–1.05 of Geant4 (C1 1.15)', 'T up to ~6 MeV'],
        ['through-going', 'liquid fires', '—', 'A 4 %, C ~0 (threshold)', 'T above ~6 MeV'],
        ['time', 'time since γ-flash', '—', '0.4–14 meV at ≥ 12 ms', 'neutron energy'],
    ]
    body = (sd.title('Performance: what each measurement gives, in simulation and on the real detector',
                     'chambers A / C where they differ; run_145 beam electrons and run_149 cosmics')
            + sd.table(['quantity', 'how', 'simulation', 'real detector', 'tells you'], rows, size=23,
                       align=['left', 'left', 'right', 'right', 'left']))
    D.slide('perf', body, '', short='Performance')


def s_future(D, t):
    items = [
        ('Thicker, segmented calorimeter', 'the 20 mm plastic stops only ≤ 5–6 MeV electrons; X17 legs reach 8–16 MeV. '
                                           '≥ 5–8 cm of plastic or a crystal, segmented so the track picks the cell.',
         sd.ORANGE),
        ('Readout window sized to the drift', 'C\'s drift runs past the 1200 ns window: 16 % vs 10 % in simulation. '
                                              'Window ≥ 1.4 × the slowest full drift.', sd.GREY),
        ('Gain uniformity and in-situ calibration', 'a 1D gain structure along x needed a track-based map; an ⁵⁵Fe '
                                                    'or X-ray source in the gas would give an absolute keV scale.',
         sd.BLUE),
        ('Flash / rate robustness', 'the MM is blind for ~10 ms after the flash and the PMTs sag 3–8 %: gate the '
                                    'MM, use rate-proof bases or a pulser monitor.', sd.RED),
        ('Keep the gas dE/dx', 'it is the only layer that counts the particles where they are made '
                               '(AUC 0.84): the tag for a two-track overlap or a δ-ray.', sd.GREEN),
        ('A magnetic field, if charge matters', 'without one, e⁺ and e⁻ are indistinguishable; with one, the '
                                                'TPC gives momentum and sign.', sd.PURPLE),
    ]
    cards = [sd.card(f'<p style="font-size:26px;font-weight:600;color:{c}">{a}</p>'
                     f'<p style="font-size:22px;line-height:1.35">{b}</p>', pad=24) for a, b, c in items]
    body = (sd.title('What this demonstrator says for the next set-up',
                     'each item measured on the n_TOF data or its simulation')
            + sd.row(*cards[:3], gap=24) + sd.row(*cards[3:], gap=24))
    D.slide('future', body, '', short='Next set-up')


def s_close(D, t):
    li = lambda x: f'<li style="margin-bottom:14px">{x}</li>'  # noqa: E731
    left = ('<ul style="font-size:25px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li('Each layer measures what the physics says it should: dE/dx in the gas and the wall, energy in the '
                 'plastic for stopping electrons, penetration in the liquid.')
            + li('The tracking ties them together: path length (muons follow sec θ), energy for steep tracks, '
                 'pointing to the source.')
            + li('The gas is the only particle counter; it works on real data as Geant4 predicts.')
            + li('Per-track gas dE/dx: 16 % (A) / 22 % (C) on beam, after an in-situ gain map.') + '</ul>')
    right = ('<ul style="font-size:25px;line-height:1.35;color:#eef0f3;padding-left:28px">'
             + li('Energies and particle probabilities are Geant4 estimators: a wrong Geant4 scale moves the data with it.')
             + li('"Several particles" is a Geant4 definition; its data rate is inferred, not measured.')
             + li('The MM keV scale is relative to the simulation, not to an absolute source.')
             + li('No liquid in the Geant4 sample; only 3–8 MeV electrons tested (the trigger sets the floor); '
                  'sub-thermal neutrons only.') + '</ul>')
    body = (sd.title('<span style="color:#eef0f3">The combined stack works as a particle-by-particle detector; '
                     'the limits are thickness, window and calibration, not the method</span>')
            + sd.row(sd.col(sd.p('what the data show', 26, sd.DGREEN), left, w=800),
                     sd.col(sd.p('what this does not rule out', 26, sd.DRED), right, w=820), gap=56))
    D.slide('close', body, '', dark=True, short='Verdict')


def main() -> int:
    t = load()
    D = sd.Deck('Track passport: tracking + calorimetry per track',
                'what the MX17 stack measures about one particle: principle, simulation, real n_TOF data')
    for f in (s_cover, s_idea, s_thin, s_landau, s_path, s_plastic, s_mmchain, s_decide,
              s_simacc, s_simcomp, s_simmulti, s_simenergy, s_simorigin,
              s_data, s_beamcal, s_gainmap, s_flash, s_pathdata, s_spectra, s_mmwall, s_estimators, s_cards,
              s_perf, s_future, s_close):
        f(D, t)
    out = PP / 'deck' / 'track-passport.html'
    D.write(out, note_meta=dict(
        title='Track passport: tracking and calorimetry on every track',
        summary='A pedagogical demonstrator: what the MX17 stack (Micromegas TPC, SiPM wall, plastic, liquid) '
                'measures about one particle, from the physics to Geant4 to 15,000 real n_TOF beam tracks: '
                'gas dE/dx to 16-22 %, electron energy to 0.9 MeV, two-particle and origin tags.',
        tags='n_TOF, X17, Micromegas, calorimetry, Geant4', date=dt.date.today().isoformat()))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
