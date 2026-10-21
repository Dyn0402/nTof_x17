#!/usr/bin/env python3
"""
make_early_deck.py -- the slide note of the flash-recovery study, built around the short term:
the earliest time after the γ-flash at which each scintillator reconstructs a pulse reliably, the
corrections it needs then, and its energy, time and efficiency uncertainties after correcting.
The long-time recovery (100 µs - ms) is the appendix.

    .venv/bin/python -m ntof_scint_recovery.make_early_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        <out>/deck/scint-flash-reconstruction.html --slug scint-flash-reconstruction --force --deploy

Reads out_dir('early') (early.py), and for the appendix out_dir('reco'), out_dir('inject'),
out_dir('gain'), out_dir('g4mix') through make_reco_deck's slide functions.
"""
from __future__ import annotations

import datetime as dt
import math
import sys

import numpy as np
import pandas as pd

from ntof_scint_recovery import make_reco_deck as A
from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.early import CRIT, GAIN_FROM, T_HORIZON
from ntof_scint_recovery.headline import WALL_MIP_MEV, energy_scale
from ntof_scint_recovery.make_reco_deck import CC, CN, FCOL, clip, fmt_t, sd

FAMN = {'WAL': 'SiPM wall', 'PSS': 'plastic', 'LIQ': 'liquid'}
#: reference signals: ~1 MIP (plastic, liquid), 2 MIP on the walls (a 1-MIP wall pulse is never reliable at
#: dedicated intensity, see the slides), and the alternatives shown alongside
REF = {'PSS': 200.0, 'WAL': 60.0, 'LIQ': 100.0}
T_LO, T_HI = 0.1, 30.0


def eticks(lo=T_LO, hi=T_HI):
    labs = {0.1: '0.1 µs', 0.3: '0.3 µs', 1: '1 µs', 3: '3 µs', 10: '10 µs', 30: '30 µs'}
    return [(v, s) for v, s in labs.items() if lo <= v <= hi]


def energy_label(fam, mv, ES):
    if fam == 'PSS':
        return f'{mv / ES["pss_mean"]:.1f} MeVee'
    if fam == 'WAL':
        m = mv / ES['wal_mean']
        return f'{m:.1f} MIP ({m * WALL_MIP_MEV:.2f} MeV)'
    return f'{mv:.0f} mV'


def unit(fam):
    return {'PSS': 'MeVee', 'WAL': 'MeV', 'LIQ': 'mV'}[fam]


def acolors(n):
    pal = ['#2b4c7e', '#3f7fb5', '#4fa3a5', '#7cbf6b', '#c9b13b', '#d9822b']
    return pal[:n]


def first(FT, fam, cls, A_, test='reco'):
    r = FT[(FT.fam == fam) & (FT.cls == cls) & (FT.amp_mV == A_) & (FT.test == test)]
    return float(r.t_first_us.iloc[0]) if len(r) else math.nan


def at(T, fam, cls, A_, t, col):
    g = T[(T.fam == fam) & (T.cls == cls) & (T.amp_mV == A_) & (T.t_lo_us <= t) & (T.t_hi_us > t)]
    return float(g[col].iloc[0]) if len(g) else math.nan


def ref_amp(FT, fam, cls):
    """The reference signal: ~1 MIP for plastics and liquids; for the walls the smallest size that is reliable
    within the first 5 µs at this intensity (1 MIP never is at dedicated)."""
    if fam != 'WAL':
        return REF[fam]
    v = smallest_reliable(FT, fam, cls)
    return v if np.isfinite(v) else REF[fam]


def smallest_reliable(FT, fam, cls, t_max=5.0):
    """The smallest injected amplitude that is reliable from <= t_max."""
    g = FT[(FT.fam == fam) & (FT.cls == cls) & (FT.test == 'reco') & (FT.t_first_us <= t_max)]
    return float(g.amp_mV.min()) if len(g) else math.nan


# ------------------------------------------------------------------ main slides
def cover(D, FT, T, TH, ES, NI):
    cards = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        Ar = ref_amp(FT, fam, 'ded')
        t = first(FT, fam, 'ded', Ar)
        tp = first(FT, fam, 'par', Ar)
        st = at(T, fam, 'ded', Ar, t * 1.05, 'sig_t_ns')
        note = (f'{energy_label(fam, Ar, ES)} pulse: ≥ 90 % reconstructed, σ(A) ≤ 25 %, '
                f'σ(t) {st:.1f} ns. Parasitic intensity: {fmt_t(tp)}.')
        if fam == 'PSS':
            note += f' Energy scale known from {GAIN_FROM["PSS"]:g} µs (gain ×{1 / at(T, fam, "ded", REF[fam], 2.2, "gain"):.0f} correction).'
        if fam == 'WAL':
            a2 = smallest_reliable(FT, fam, 'par')
            note += (f' Parasitic: {energy_label(fam, a2, ES)} from {fmt_t(first(FT, fam, "par", a2))}. '
                     '1 MIP: never ≥ 90 % at dedicated intensity (pile-up).')
        cards.append(sd.bignum(fmt_t(t), FAMN[fam], {'PSS': sd.DRED, 'LIQ': sd.DGREEN, 'WAL': sd.DBLUE}[fam], note,
                               tip=f'Criteria: {CRIT["eff"]:.0%} of injected pulses found within ±50 % of the pre-flash '
                                   f'response, σ68(A)/A ≤ {CRIT["sig_res"]:.0%}, σ68(t) ≤ {CRIT["sig_t"][fam]:g} ns, '
                                   f'in every bin from then to {T_HORIZON:g} µs'))
    head = 'Earliest reliable pulse reconstruction after the γ-flash'
    tw = first(FT, 'WAL', 'ded', ref_amp(FT, 'WAL', 'ded'))
    sub = (f'Plastics and liquids reconstruct a ~1-MIP pulse within the first 1–2 µs; the walls need ~{tw:.0f} µs and a '
           f'{ref_amp(FT, "WAL", "ded") / ES["wal_mean"]:.0f}-MIP signal. Early pulses are biased (plastic PMT gain down ×3 at '
           '2–3 µs) but the bias is measured and corrected; what is left is quoted per µs')
    body = (sd.kicker('n_TOF EAR2 · X17 scintillators · the first microseconds after the γ-flash')
            + sd.title(f'<span style="color:{sd.DINK}">{head}</span>', f'<span style="color:{sd.DMUT}">{sub}</span>')
            + sd.row(*cards, gap=56)
            + sd.p(f'Dedicated pulses (8.5×10¹² protons), time since the flash at the detector. Pulses of known size injected '
                   f'into the recorded flash tails of {NI["n_bunch"]:,} bunches ({NI["n_runs"]} production runs) and '
                   'reconstructed; gain from the measured spectra.', 22, sd.DMUT))
    notes = ('<p>The question for future data taking: how early after the flash can each scintillator be used, how much must '
             'its pulses be corrected then, and with what energy, time and efficiency uncertainty after the correction. '
             '"Reliable" here is deliberately loose about bias, since a bias that is measured is corrected: a pulse counts '
             'if it is found and comes out within ±50 % of its normal response; 90 % of pulses must, and the spread '
             f'(σ68/A ≤ {CRIT["sig_res"]:.0%}) and timing (σ68 ≤ 1 ns plastics and liquids, 10 ns walls) must be usable. '
             'Each time quoted holds from then on, not just in one bin.</p>'
             '<p>The full-recovery times (hundreds of µs to ms, the earlier version of this note) are in the appendix.</p>')
    D.slide('cover', body, notes, dark=True, short='cover')


def traces(D, X):
    panels = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        t = X[f'{fam}_t']
        ymax = {'PSS': 2000, 'LIQ': 2000, 'WAL': 1000}[fam]
        P = sd.Plot(540, 600, x=(-0.5, 10.5), y=(-50, ymax), xlabel='time since the flash [µs]',
                    ylabel='signal [mV]' if fam == 'PSS' else '', title=f'{FAMN[fam]} ({X[f"{fam}_channel"]})',
                    margin=(24, 24, 92, 96))
        P.band(t.tolist(), clip(X[f'{fam}_min'], -50, ymax), clip(X[f'{fam}_max'], -50, ymax), FCOL[fam], alpha=0.85)
        for ti, A_, Af, py in zip(X[f'{fam}_inj_t'], X[f'{fam}_inj_A'], X[f'{fam}_fit_A'], X[f'{fam}_peak_y']):
            ok = np.isfinite(Af) and abs(Af / A_ - 1) <= 0.5
            P.points([float(ti)], [min(float(py) + 0.05 * ymax, ymax * 0.98)], sd.GREEN if ok else sd.RED, r=9,
                     marker='diamond', tips=[f'injected {A_:.0f} mV at {ti:.2f} µs → reconstructed '
                                             + (f'{Af:.0f} mV' if np.isfinite(Af) else 'not found')])
        P.xticks([(v, f'{v:g}') for v in (0, 2, 4, 6, 8, 10)]).yticks([(v, f'{v:g}') for v in (0, ymax / 2, ymax)])
        panels.append(P.svg('tr' + fam))
    body = (sd.title('What a pulse has to be found on: the first 10 µs of one dedicated bunch',
                     'Recorded traces (min–max per 8 ns) with ~1-MIP pulses added every µs (diamonds: green = reconstructed '
                     'within ±50 %, red = lost or off); plastic 200 mV, liquid 100 mV, wall end 30 mV')
            + sd.row(*panels, gap=24))
    notes = ('<p>Plastics and liquids: saturated for ~0.2 µs, then a dense train of fast-neutron recoils peaking at ~1 µs '
             '(2 MeV neutrons at 19.5 m), falling to a few tens of mV by 5 µs. A pulse on that slope is found and fitted '
             'with a local baseline. Walls: switched off by the divert gate until ~0.65 µs, then the stored flash charge '
             'clips the amplifier at ~1 V until ~1.6 µs and decays over ~10 µs, with a continuous rain of 10–60 mV pulses '
             'on it. That rain, not the tail, is what spoils the wall reconstruction.</p>'
             '<p>Run 224712, from <code>early.py --examples</code>.</p>')
    D.slide('traces', body, notes, short='first 10 µs')


def method(D, NI):
    steps = [dict(label='real flash trace', color=sd.GREY,
                  sub='every channel, every bunch: the unsuppressed block, 10 µs before to 40–250 µs after the flash'),
             dict(label='+ pulses of known size', color=sd.ORANGE,
                  sub='family template, known amplitude and time; a comb every 0.6–2 µs; clipped at saturation'),
             dict(label='reconstruct', color=sd.BLUE,
                  sub='peak finder → template fit with every neighbour and a local baseline (multi-pulse fit)'),
             dict(label='compare', color=sd.GREEN,
                  sub='found? A_fit / A? t_fit − t₀?  The same 1.5–8 µs before the flash is the reference')]
    body = (sd.title('Method: injection for the reconstruction, the spectra for the gain',
                     f'{NI["n_bunch"]:,} bunches of {NI["n_runs"]} production runs; every number split by beam intensity')
            + sd.flow(steps, size=26)
            + sd.row(sd.card(sd.p('<b>Injection measures</b> the electronics and the algorithm on the real tail: baseline, '
                                  'clipping, pulse shape, and pile-up with the real beam pulses. It gives the efficiency, '
                                  'the reconstruction response and its spread, and the timing.', 24), w=780),
                     sd.card(sd.p('<b>It cannot see</b> a gain change in the PMT/SiPM (the pulse is added after it). The '
                                  'gain comes from the measured amplitude spectra: a gain change scales all their quantiles '
                                  'together, a change of particle mix does not (appendix).', 24), w=780), gap=40))
    notes = ('<p>Amplitudes injected: plastics 25–800 mV (0.45–14 MeVee), wall ends 8–120 mV (0.3–4 MIP), liquids 25–400 mV. '
             'Energy scale: plastics 56 mV/MeVee on average (39–72 per bar, muon calibration and cosmics at 30–32 ms); '
             'wall ends 28 mV per MIP (22–31), a MIP being 0.47 MeV in 3 mm. Liquids: no energy scale.</p>'
             '<p>Code: <code>inject.py</code> (condor), <code>early.py</code>, <code>gain_ref.py</code>.</p>')
    D.slide('method', body, notes, short='method')


def earliest(D, T, FT, ES):
    panels = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        g0 = T[(T.fam == fam) & (T.cls == 'ded') & (T.t_us <= T_HI)]
        x0 = 0.6 if fam == 'WAL' else T_LO
        P = sd.Plot(540, 580, x=(x0, T_HI, 'log'), y=(0, 1), xlabel='time since the flash',
                    ylabel='reconstructed fraction' if fam == 'PSS' else '', title=FAMN[fam], margin=(24, 24, 92, 90))
        P.band([x0, T_HI], [0.9, 0.9], [1, 1], sd.GREEN, alpha=0.10)
        amps = sorted(g0.amp_mV.unique())
        Ar = ref_amp(FT, fam, 'ded')
        for A_, c in zip(amps, acolors(len(amps))):
            g = g0[g0.amp_mV == A_].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.eff, 0, 1), c, 4 if A_ == Ar else 2.2, markers=A_ == Ar, r=4,
                   tips=[f'{energy_label(fam, A_, ES)}, {fmt_t(t)}: {v:.0%}' for t, v in zip(g.t_us, g.eff)]
                   if A_ == Ar else None, tip=energy_label(fam, A_, ES))
        t1 = first(FT, fam, 'ded', Ar)
        if np.isfinite(t1):
            P.vline(t1, sd.INK, '6 5', 2, label=fmt_t(t1), tip=f'reliable from {fmt_t(t1)} ({energy_label(fam, Ar, ES)})')
        P.xticks(eticks(x0, T_HI)).yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
        panels.append(P.svg('e' + fam))
    leg = [(f'{energy_label("PSS", a, ES)}', c, 'line') for a, c in zip((25, 50, 100, 200, 400, 800), acolors(6))]
    aw = ref_amp(FT, 'WAL', 'ded')
    ttl = (f'The earliest time: a ~1-MIP pulse is reliable from {fmt_t(first(FT, "PSS", "ded", REF["PSS"]))} (plastic), '
           f'{fmt_t(first(FT, "LIQ", "ded", REF["LIQ"]))} (liquid); the wall needs {aw / ES["wal_mean"]:.0f} MIP and '
           f'{fmt_t(first(FT, "WAL", "ded", aw))}')
    body = (sd.title(ttl,
                     'Fraction of injected pulses found within ±50 % of their normal response, dedicated; thick = reference '
                     'signal, dashed line = where it becomes reliable. Colours by size (plastic energies; walls 0.3–4 MIP, '
                     'liquids 25–400 mV)')
            + sd.row(*panels, gap=24)
            + sd.legend(leg + [('≥ 90 %', sd.GREEN, 'box')], size=20))
    rows = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        for cls in ('ded', 'par'):
            for A_ in sorted(FT[FT.fam == fam].amp_mV.unique()):
                rows.append(f'{FAMN[fam]} {CN[cls]} {energy_label(fam, A_, ES)}: {fmt_t(first(FT, fam, cls, A_))}')
    notes = ('<p>Before ~0.2 µs the plastics and liquids are saturated; the walls are switched off until ~0.65 µs and clipped '
             'until ~1.6 µs. The plastic dip at 0.2–0.4 µs is the end of the saturation. Bigger pulses are reliable earlier; '
             'small ones are lost under the pile-up (slide 8).</p><p>Earliest reliable time per signal size (criteria on the '
             'cover): ' + '; '.join(rows) + '.</p>')
    D.slide('earliest', body, notes, short='earliest time')


def corrections(D, T, ES, FT):
    panels = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        x0 = 0.6 if fam == 'WAL' else 0.3
        P = sd.Plot(540, 580, x=(x0, T_HI, 'log'), y=(0, 1.6), xlabel='time since the flash',
                    ylabel='measured / true amplitude' if fam == 'PSS' else '', title=f'{FAMN[fam]}, {energy_label(fam, ref_amp(FT, fam, "ded"), ES)}',
                    margin=(24, 24, 92, 90))
        P.hline(1, sd.INK, None, 1.2)
        for cls in ('ded', 'par'):
            g = T[(T.fam == fam) & (T.cls == cls) & (T.amp_mV == ref_amp(FT, fam, 'ded')) & (T.t_us >= x0) & (T.t_us <= T_HI)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.resp, 0, 1.6), CC[cls], 2.2, dash='4 4', markers=False,
                   tip=f'reconstruction response, {CN[cls]}')
            if g.gain.notna().any():
                m = g.gain.notna()
                tot = g.resp * g.gain
                unc = g.resp * g.gain_unc
                P.band(g.t_us[m].tolist(), clip((tot - unc)[m], 0, 1.6), clip((tot + unc)[m], 0, 1.6), CC[cls], alpha=0.25)
                P.line(g.t_us[m].tolist(), clip(tot[m], 0, 1.6), CC[cls], 4, markers=True, r=4,
                       tips=[f'{CN[cls]}, {fmt_t(t)}: response {r:.2f} × gain {gg:.2f} ± {uu:.2f} = {r * gg:.2f}'
                             for t, r, gg, uu in zip(g.t_us[m], g.resp[m], g.gain[m], g.gain_unc[m])])
        if fam == 'PSS':
            P.band([x0, GAIN_FROM['PSS']], [0, 0], [1.6, 1.6], sd.GREY, alpha=0.12, tip='gain not measurable: fast-neutron recoils')
        P.xticks(eticks(x0, T_HI)).yticks([(v, f'{v:.1f}') for v in (0, 0.4, 0.8, 1.2, 1.6)])
        panels.append(P.svg('c' + fam))
    body = (sd.title('What must be corrected: the plastic PMT gain is ×3 low at 2–3 µs; the reconstruction itself is within a few %',
                     'Measured over true amplitude: reconstruction response (dashed) and response × detector gain (thick, band = '
                     'gain uncertainty). Grey: gain not measurable. Liquids: no gain reference')
            + sd.row(*panels, gap=24)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'),
                         ('reconstruction only', sd.MUT, 'dash'), ('× gain', sd.INK, 'line')]))
    notes = ('<p>Reconstruction response: median A_fit/A of injected pulses over the same before the flash. Within ±7 % for the '
             'plastics from 0.4 µs and the liquids from 0.45 µs; the walls read 25–50 % low at 2–3.5 µs (a pulse on the falling '
             'stored-charge tail and under the pile-up) and within 3 % from ~4.5 µs.</p>'
             '<p>Gain: plastics from the 99 % point of the measured spectrum on a fine time grid (×1.12 bins), capped at 1; '
             'uncertainty = half the difference between the singles and the pulses behind a wall hit ⊕ 3 %. Dedicated: '
             '0.33 at 2–3 µs, 0.43 / 0.61 / 0.76 / 0.86 / 0.95 at 3.5 / 4.5 / 5.6 / 7 / 9 µs, 1.0 by 11 µs; parasitic 0.56–0.73 '
             'at 2–3 µs, 0.9 by 4 µs. The sag scales with intensity, as an anode-current-driven PMT gain loss does. Before '
             '2 µs the plastic spectrum is fast-neutron recoils and its shape cannot separate gain from mix (q99 gives only '
             '≥ 0.15–0.3). Walls: no sag resolved (1 ± 3–7 %). Liquids: their spectrum changes ×1.4–2.9 in shape, no reference.</p>')
    D.slide('corrections', body, notes, short='corrections')


def energy_unc(D, T, ES, FT):
    panels = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        x0 = 0.6 if fam == 'WAL' else 0.3
        g0 = T[(T.fam == fam) & (T.cls == 'ded') & (T.t_us >= x0) & (T.t_us <= T_HI)]
        P = sd.Plot(540, 580, x=(x0, T_HI, 'log'), y=(0, 0.6), xlabel='time since the flash',
                    ylabel='σ(E)/E after correction' if fam == 'PSS' else '', title=FAMN[fam], margin=(24, 24, 92, 90))
        amps = sorted(g0.amp_mV.unique())
        Ar = ref_amp(FT, fam, 'ded')
        for A_, c in zip(amps, acolors(len(amps))):
            g = g0[g0.amp_mV == A_].sort_values('t_us')
            y = g.sig_E_rel if fam != 'LIQ' else g.sig_res
            P.line(g.t_us.tolist(), clip(y, 0, 0.6), c, 4 if A_ == Ar else 2.2, markers=A_ == Ar, r=4,
                   tips=[f'{energy_label(fam, A_, ES)}, {fmt_t(t)}: σ/E {v:.0%} (resolution {r:.0%}, correction '
                         f'{s:.0%})' for t, v, r, s in zip(g.t_us, y, g.sig_res, g.sig_sys)] if A_ == Ar else None,
                   tip=energy_label(fam, A_, ES))
            if A_ == Ar and fam != 'LIQ':
                P.line(g.t_us.tolist(), clip(g.sig_sys, 0, 0.6), c, 2, dash='3 4', markers=False,
                       tip='of which: the correction (systematic)')
            ctl = g.ctrl_sig_res.iloc[0] if len(g) else np.nan
            if A_ == Ar and np.isfinite(ctl):
                P.hline(ctl, c, '1 5', 1.5, tip=f'before the flash: {ctl:.1%}')
        if fam == 'PSS':
            P.band([x0, GAIN_FROM['PSS']], [0, 0], [0.6, 0.6], sd.GREY, alpha=0.12, tip='gain not measurable before 2 µs')
        P.xticks(eticks(x0, T_HI)).yticks([(v, f'{v:.0%}') for v in (0, 0.2, 0.4, 0.6)])
        panels.append(P.svg('u' + fam))
    p2 = T[(T.fam == 'PSS') & (T.cls == 'ded') & (T.amp_mV == 200)]
    body = (sd.title('Energy after correction: 6–13 % on a ~1-MIP plastic pulse from 2 µs; walls 13–18 % at 4 MIP from 3.5 µs',
                     'σ68 of the reconstructed amplitude ⊕ the uncertainty of the correction (dashed, reference signal), dedicated; '
                     'liquids: reconstruction only (no gain reference). Thick = reference signal, colours by size')
            + sd.row(*panels, gap=24)
            + sd.legend([(f'{energy_label("PSS", a, ES)}', c, 'line') for a, c in zip((25, 50, 100, 200, 400, 800), acolors(6))]
                        + [('correction part', sd.MUT, 'dash'), ('before the flash', sd.MUT, 'dash')], size=20))
    notes = ('<p>After dividing by response × gain, what is left is (1) the spread of the reconstructed amplitude, which pile-up '
             'with the beam pulses sets and no correction removes, and (2) the uncertainty of the correction: the gain '
             'uncertainty, the statistical error of the response, and half the change of the correction across a time bin. '
             'Plastics, ~1 MIP, dedicated: σ(E)/E '
             + ', '.join(f'{v:.0%} at {fmt_t(t)}' for t, v in zip(p2.t_us, p2.sig_E_rel) if 1.9 < t < 30 and np.isfinite(v))
             + '. The best window is 2–3 µs, between the end of the fast-neutron train and the start of the dense capture-γ '
             'pile-up.</p><p>Walls, dedicated: 4 MIP (2 MeV) per end 18 / 15 / 13 / 10 % at 3.5 / 4.5 / 7 / 20 µs; 2 MIP 40 / 30 / '
             '25 / 20 %; 1 MIP above 40 %. Liquids: no energy scale, the reconstruction spread alone is shown.</p>')
    D.slide('energy', body, notes, short='energy uncertainty')


def timing(D, T, ES, FT):
    panels = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        x0 = 0.6 if fam == 'WAL' else 0.3
        g0 = T[(T.fam == fam) & (T.cls == 'ded') & (T.t_us >= x0) & (T.t_us <= T_HI)]
        P = sd.Plot(540, 580, x=(x0, T_HI, 'log'), y=(0.03, 30, 'log'), xlabel='time since the flash',
                    ylabel='σ68(t) [ns]' if fam == 'PSS' else '', title=FAMN[fam], margin=(24, 24, 92, 90))
        amps = sorted(g0.amp_mV.unique())
        Ar = ref_amp(FT, fam, 'ded')
        for A_, c in zip(amps, acolors(len(amps))):
            g = g0[g0.amp_mV == A_].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.sig_t_ns, 0.03, 30), c, 4 if A_ == Ar else 2.2, markers=A_ == Ar, r=4,
                   tips=[f'{energy_label(fam, A_, ES)}, {fmt_t(t)}: σ {s:.2f} ns, offset {b:+.2f} ns'
                         for t, s, b in zip(g.t_us, g.sig_t_ns, g.t_bias_ns)] if A_ == Ar else None,
                   tip=energy_label(fam, A_, ES))
        P.hline(CRIT['sig_t'][fam], sd.RED, '6 5', 1.5, tip='the reliability limit')
        P.xticks(eticks(x0, T_HI)).yticks([(v, f'{v:g}') for v in (0.1, 1, 10)])
        panels.append(P.svg('t' + fam))
    body = (sd.title('Time: 0.1–0.3 ns on plastics and liquids from ~1 µs; walls 1–2 ns at 4 MIP, 3–6 ns at 2 MIP from 3 µs',
                     'σ68 of t_fit − t₀ after removing the median offset (hover: offset), dedicated; red = the limit used for '
                     '"reliable". Colours by size, thick = reference signal')
            + sd.row(*panels, gap=24))
    p = T[(T.fam == 'PSS') & (T.cls == 'ded') & (T.amp_mV == 200) & (T.t_us < 30)]
    notes = ('<p>The median offset (time bias) is below 0.05 ns for plastics and liquids after 0.4 µs and below 0.3 ns for the '
             'walls after 3 µs; it is corrected the same way as the amplitude. The resolution is the jitter from noise and '
             'pile-up on the leading edge; before the flash it is 0.07 ns (plastics, liquids, 1-ns sampling with a sub-ns fit) and '
             '0.4 ns (walls). Plastic ~1 MIP, dedicated: '
             + ', '.join(f'{s:.2f} ns at {fmt_t(t)}' for t, s in zip(p.t_us, p.sig_t_ns) if 0.4 < t)
             + '.</p><p>The flash time itself (PKUP + per-detector constant) is known to ~10 ns between detector families; '
             'within a family it cancels.</p>')
    D.slide('timing', body, notes, short='time resolution')


def small_signals(D, TH, ES):
    panels = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        x0 = 0.6 if fam == 'WAL' else 0.3
        g = TH[(TH.fam == fam) & (TH.t_us >= x0) & (TH.t_us <= T_HI)]
        if fam == 'LIQ':
            ylab, lo, hi, col = 'smallest signal kept [mV]', 10, 500, 'mV'
        else:
            ylab, lo, hi, col = f'smallest signal kept [{unit(fam)}]', (0.2 if fam == 'PSS' else 0.1), (20 if fam == 'PSS' else 3), 'E'
        P = sd.Plot(540, 580, x=(x0, T_HI, 'log'), y=(lo, hi, 'log'), xlabel='time since the flash',
                    ylabel=ylab, title=FAMN[fam], margin=(24, 24, 92, 96))
        for cls in ('ded', 'par'):
            h = g[g.cls == cls].sort_values('t_us')
            for x, dash, wd in ((90, None, 3.5), (50, '6 5', 2.2)):
                y = h[f'thr{x}_{col}']
                fl = h[f'thr{x}_flag'].fillna('')
                yy = [v if f == '' else None for v, f in zip(y, fl)]
                P.line(h.t_us.tolist(), clip(np.array([np.nan if v is None else v for v in yy], float), lo, hi), CC[cls], wd,
                       dash=dash, markers=x == 90, r=3,
                       tips=[f'{CN[cls]}, {fmt_t(t)}: {x} % kept above {v:.2f} {unit(fam) if fam != "LIQ" else "mV"}'
                             for t, v in zip(h.t_us, y)] if x == 90 else None, tip=f'{x} % efficiency, {CN[cls]}')
        if fam == 'PSS':
            P.band([x0, GAIN_FROM['PSS']], [lo, lo], [hi, hi], sd.GREY, alpha=0.12, tip='gain unknown: energy threshold is a lower bound')
        tk = {'PSS': (0.3, 1, 3, 10), 'WAL': (0.1, 0.3, 1, 3), 'LIQ': (10, 30, 100, 300)}[fam]
        P.xticks(eticks(x0, T_HI)).yticks([(v, f'{v:g}') for v in tk])
        panels.append(P.svg('s' + fam))
    body = (sd.title('Small signals: what is lost early',
                     'Energy (plastics MeVee, walls MeV; liquids mV) above which 90 % (solid) / 50 % (dashed) of pulses are '
                     'reconstructed; gain sag included for the plastics (grey: gain unknown, lower bound)')
            + sd.row(*panels, gap=24)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'), ('90 %', sd.MUT, 'line'),
                         ('50 %', sd.MUT, 'dash')]))
    notes = ('<p>The efficiency of a pulse of a given size (found within ±50 % of its response), interpolated in the size. For the '
             'plastics the energy threshold divides by the gain: a sagging PMT makes every deposit smaller, so at 2–3 µs '
             '(gain 0.33, dedicated) the 90 % threshold is ~2.5 MeVee against ~0.8 MeVee in amplitude terms. Walls: 90 % only '
             'above ~1.2–1.5 MeV (2.5–3 MIP per end) at dedicated intensity up to ~10 µs; half of 1-MIP pulses are kept from '
             '~4 µs. Where a curve is missing the threshold is outside the injected range (below the smallest or above the '
             'largest size).</p>')
    D.slide('small', body, notes, short='small signals')


def summary_table(D, T, TH, FT, ES, cls):
    times = {'PSS': (1, 2, 3, 5, 10, 20), 'LIQ': (1, 2, 3, 5, 10), 'WAL': (2, 3, 5, 10, 20)}
    rows, tips = [], []
    for fam in ('PSS', 'LIQ', 'WAL'):
        A_ = ref_amp(FT, fam, cls)
        t1 = first(FT, fam, cls, A_)
        for t in ([t1 * 1.05] if np.isfinite(t1) and t1 < 0.95 else []) + list(times[fam]):
            eff = at(T, fam, cls, A_, t, 'eff')
            if not np.isfinite(eff):
                continue
            th = TH[(TH.fam == fam) & (TH.cls == cls) & (TH.t_lo_us <= t) & (TH.t_hi_us > t)]
            if fam == 'LIQ':
                thr = f'{th.thr90_mV.iloc[0]:.0f} mV' if len(th) else '—'
            else:
                flag = th.thr90_flag.iloc[0] if len(th) and isinstance(th.thr90_flag.iloc[0], str) else ''
                if fam == 'PSS' and len(th) and not np.isfinite(th.gain.iloc[0]):
                    flag = flag or '≥'                    # gain unknown before 2 µs: a lower bound
                thr = f'{flag}{th.thr90_E.iloc[0]:.2f} {unit(fam)}' if len(th) else '—'
            resp, gn, gu = at(T, fam, cls, A_, t, 'resp'), at(T, fam, cls, A_, t, 'gain'), at(T, fam, cls, A_, t, 'gain_unc')
            corr = f'{resp * gn:.2f} ± {resp * gu:.2f}' if np.isfinite(gn) else f'{resp:.2f} (reco only)'
            sE = at(T, fam, cls, A_, t, 'sig_E_rel') if fam != 'LIQ' else at(T, fam, cls, A_, t, 'sig_res')
            E = A_ / ES['pss_mean'] if fam == 'PSS' else (A_ / ES['wal_mean'] * WALL_MIP_MEV if fam == 'WAL' else np.nan)
            sEs = (f'{sE:.0%} ({sE * E:.2f} {unit(fam)})' if np.isfinite(E) and np.isfinite(sE) else
                   (f'{sE:.0%}' if np.isfinite(sE) else '— (gain unknown)'))
            rel = '✓' if np.isfinite(t1) and t >= t1 else ''
            rows.append([f'{FAMN[fam]} · {energy_label(fam, A_, ES)}', fmt_t(t), rel, f'{eff:.0%}', thr, corr, sEs,
                         f'{at(T, fam, cls, A_, t, "sig_t_ns"):.2f} ns'])
            tips.append(f'resolution part {at(T, fam, cls, A_, t, "sig_res"):.1%}, correction part '
                        f'{at(T, fam, cls, A_, t, "sig_sys"):.1%}; time offset {at(T, fam, cls, A_, t, "t_bias_ns"):+.2f} ns')
    body = (sd.title(f'In numbers, {CN[cls]} pulses: the reference signal over the first 20 µs',
                     'Reconstructed fraction; 90 % efficiency threshold (all sizes, gain included); the correction to divide by '
                     '(measured / true); energy uncertainty after correcting; time resolution')
            + sd.table(['signal', 'time', 'reliable', 'reconstructed', '90 % above', 'correction', 'σ(E) after corr.', 'σ(t)'],
                       rows, size=17, tips=tips))
    notes = ('<p>"reliable" marks the times from which the cover\'s criteria hold. "correction" = reconstruction response × gain, '
             'with the gain uncertainty; divide a measured amplitude by it. Hover a row for the split of σ(E) into the '
             'pile-up spread and the correction uncertainty. Every size and time bin: <code>out_dir("early")/early.csv</code>, '
             '<code>thresholds.csv</code>, <code>first.csv</code>.</p>')
    D.slide(f'table_{cls}', body, notes, short=f'numbers, {CN[cls]}')


def implications(D):
    items = [
        ('<b>Plastics</b> are usable from ~0.5 µs for MIP-sized signals with 0.2–0.5 ns timing. Their energy needs the PMT '
         'gain correction, which the data measure only from 2 µs and which is ×3 at 2–3 µs (dedicated). A gain monitor that '
         'works in the first µs (a pulsed light source on every PMT) would fix the scale before 2 µs and halve the energy '
         'uncertainty at 2–5 µs; a base built for high anode current would shrink the sag itself.'),
        ('<b>Liquids</b> recover fastest in shape (from ~1.6 µs at 100 mV, 25 mV kept from ~5 µs) but have no energy scale in '
         'this data.'),
        ('<b>SiPM walls</b> are limited by pile-up, not by the flash tail: a ~500 ns pulse under a rain of real pulses. At '
         'dedicated intensity a 1-MIP end signal is never reliable in the first 30 µs; 2–4 MIP from ~3 µs. Faster shaping '
         '(a shorter pulse body) cuts the pile-up probability in proportion.'),
        ('<b>Everything early scales with intensity</b>: parasitic pulses (half the protons) halve the plastic gain loss '
         'and make wall 2-MIP signals reliable from ~3 µs instead of never.'),
    ]
    body = (sd.title('What this means for a measurement in the first microseconds',
                     'From the 2026 data; the numbers are for this apparatus at 4 and 8.5×10¹² protons per pulse')
            + ''.join(sd.callout(t, c, 24) for t, c in zip((x for x in items), (sd.RED, sd.GREEN, sd.BLUE, sd.GOLD))))
    D.slide('implications', body, '', short='implications')


def caveats(D):
    items = [
        'The plastic gain before 2 µs is not measured: the fast-neutron recoil spectrum cannot separate gain from mix there '
        '(and Geant4 does not reproduce the data spectrum well enough to serve as a candle). Energies before 2 µs are lower bounds.',
        'The gain method assumes the top of the spectrum is made by the same particles at every time; the two plastic '
        'populations (singles, behind a wall hit) differ by ≤ 0.05–0.1, which is taken as the uncertainty.',
        'Injected pulses use the family template; per-channel shape differences cancel against the pre-flash reference.',
        'Time bins are ×1.27 wide; the correction is interpolated, and half its change across a bin is in the uncertainty.',
        'Pile-up scales with the pulse rate: these numbers hold for this apparatus and these intensities.',
    ]
    body = (sd.kicker('What this does not rule out', dark=True) + ''.join(sd.p(f'· {t}', 25, sd.DINK) for t in items))
    D.slide('limits', body, '', dark=True, short='limits')


# ------------------------------------------------------------------ appendix
def appendix_divider(D, Hd, ES):
    rows = []
    for fam in ('PSS', 'LIQ', 'WAL'):
        for cls in ('ded', 'par'):
            r = Hd[(Hd.fam == fam) & (Hd.cls == cls) & (Hd.amp_mV == A.REF_AMP[fam])]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append([FAMN[fam], CN[cls], r.energy, fmt_t(r.t_headline_us) + ('*' if r.headline_from_model else ''),
                         fmt_t(r.t_gain95_us) if fam == 'PSS' else '—'])
    body = (sd.kicker('Appendix · the long-time recovery', dark=True)
            + sd.p('When does a pulse become "as good as before the flash"? ≥ 90 % within 10 % of its true amplitude '
                   '(pile-up included) and, for the plastics, the gain within 5 %. This is the full-recovery view of the first '
                   'version of this note.', 26, sd.DINK)
            + sd.table(['family', 'intensity', '~1 MIP', 'full recovery', 'gain ≥ 0.95'], rows, size=22)
            + sd.p('* beyond the recorded block, from the pile-up model.', 20, sd.DMUT))
    D.slide('appendix', body, '', dark=True, short='appendix')


def main() -> int:
    eo = out_dir('early')
    T = pd.read_csv(eo / 'early.csv')
    TH = pd.read_csv(eo / 'thresholds.csv')
    FT = pd.read_csv(eo / 'first.csv')
    X = dict(np.load(eo / 'examples.npz'))
    ES = energy_scale()
    z = np.load(out_dir('inject') / 'inject.npz')
    NI = dict(n_bunch=int(z['n_bunch'].sum()), n_runs=int(z['n_files']))
    ro = out_dir('reco')
    Hd = pd.read_csv(ro / 'headline.csv')
    S = pd.read_csv(out_dir('inject') / 'inject_summary.csv')
    Q = pd.read_csv(out_dir('gain') / 'gain_quantiles.csv')
    R = pd.read_csv(out_dir('g4mix') / 'g4_rates.csv')
    D = sd.Deck('Scintillator pulses in the first microseconds after the γ-flash',
                'How early after the n_TOF γ-flash each X17 scintillator reconstructs a pulse, and with what uncertainty')
    cover(D, FT, T, TH, ES, NI)
    traces(D, X)
    method(D, NI)
    earliest(D, T, FT, ES)
    corrections(D, T, ES, FT)
    energy_unc(D, T, ES, FT)
    timing(D, T, ES, FT)
    small_signals(D, TH, ES)
    summary_table(D, T, TH, FT, ES, 'ded')
    summary_table(D, T, TH, FT, ES, 'par')
    implications(D)
    caveats(D)
    # appendix
    appendix_divider(D, Hd, ES)
    A.mix(D, R)
    A.gain_slide(D, Q)
    A.pile_clean(D, S)
    A.good_slide(D, S, A.energy_scale())
    A.model_slide(D, pd.read_csv(ro / 'occupancy_model.csv'), S, Hd)
    A.bias_slide(D, pd.read_csv(ro / 'figures' / 'bias_vs_t.csv'))
    A.table_slide(D, pd.read_csv(ro / 'performance.csv'), A.energy_scale())
    path = D.write(out_dir('deck') / 'scint-flash-reconstruction.html', note_meta=dict(
        title='Scintillator pulses in the first microseconds after the γ-flash',
        summary='How early after the n_TOF γ-flash each X17 scintillator reconstructs a pulse reliably, the corrections it '
                'needs then (reconstruction response, PMT gain), and its energy, time and efficiency uncertainties after '
                'correcting; the long-time recovery in the appendix.',
        tags='X17, n_TOF, scintillators', date=dt.date.today().isoformat()))
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
