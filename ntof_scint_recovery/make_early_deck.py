#!/usr/bin/env python3
"""
make_early_deck.py -- the slide note of the flash-recovery study, built around the short term and around a MIP:
when after the γ-flash each scintillator sees a minimum-ionizing particle (the measured cosmic-muon spectrum x the
measured gain x the injected-pulse efficiency per size), how well it measures its energy and time, and from when
real pulses show the detector and electronics to be linear (the injection assumes it).  Per-size results and the
long-time recovery (100 µs - ms) are the appendix.

    .venv/bin/python -m ntof_scint_recovery.make_early_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        <out>/deck/scint-flash-reconstruction.html --slug scint-flash-reconstruction --force --deploy

Reads out_dir('early') (early.py, mip_eff.py), out_dir('linearity') (linearity.py), and for the appendix
out_dir('reco'), out_dir('inject'), out_dir('gain'), out_dir('g4mix') through make_reco_deck's slide functions.
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
from ntof_scint_recovery.gain_ref import MIP_PLASTIC_MEVEE
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
    ttl = (f'Per size (injection only, best case): a 200 mV plastic pulse is reliable from {fmt_t(first(FT, "PSS", "ded", REF["PSS"]))} (plastic), '
           f'{fmt_t(first(FT, "LIQ", "ded", REF["LIQ"]))} (100 mV liquid); a wall end needs {aw / ES["wal_mean"]:.0f} MIP and '
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
    body = (sd.title('Per size: the corrections, reconstruction response (injection) and × gain (real spectra)',
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
    body = (sd.title('Per size: energy after correction (pile-up spread ⊕ gain uncertainty)',
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
             'pile-up. Note the gain uncertainty is relative: it weighs ×3 more where the gain is 0.33.</p><p>Liquids: no energy '
             'scale, the reconstruction spread alone is shown.</p>')
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
    body = (sd.title('Per size: time resolution (injection only)',
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
    body = (sd.title('Per size: the smallest signal kept',
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


# ------------------------------------------------------------------ the MIP view (main slides)
LIN_TOL = 0.10
FAM2 = ('PSS', 'WAL')
MIPN = {'PSS': 'plastic', 'WAL': 'SiPM wall (one end)'}


def mfirst(MF, fam, cls, col):
    r = MF[(MF.fam == fam) & (MF.cls == cls)]
    return float(r[col].iloc[0]) if len(r) else math.nan


def mat(M, fam, cls, t, col):
    g = M[(M.fam == fam) & (M.cls == cls) & (M.t_lo_us <= t) & (M.t_hi_us > t)]
    return float(g[col].iloc[0]) if len(g) else math.nan


def shade_nonlinear(P, x0, t_lin, ylo, yhi, label=True):
    if np.isfinite(t_lin) and t_lin > x0:
        P.band([x0, t_lin], [ylo, ylo], [yhi, yhi], sd.GREY, alpha=0.16,
               tip='real pulses do not (yet) show a linear response: injection here is a best case only')


def cover(D, M, MF, LF, NI):
    cards = []
    for fam in ('PSS', 'WAL'):
        tu, tp = mfirst(MF, fam, 'ded', 'usable_from_us'), mfirst(MF, fam, 'par', 'usable_from_us')
        tl = mfirst(MF, fam, 'ded', 'linear_from_us')
        if fam == 'PSS':
            tg = mfirst(MF, fam, 'ded', 'gain_from_us')
            tlo = mfirst(MF, fam, 'ded', 'seen90_lo_us')
            note = (f'A MIP is seen ≥ 90 % from {fmt_t(tu)} (dedicated'
                    + (f'; {fmt_t(tlo)} if the gain sits at the low end of its range' if fmt_t(tlo) != fmt_t(tu) else '')
                    + f'). Real pulses respond linearly from {fmt_t(tl)}. '
                    f'Energy: gain measured from {fmt_t(tg)}, σ(E)/E ≈ {mat(M, fam, "ded", 2.5, "mip_sig_E"):.0%} at 2–3 µs; '
                    f'before, the gain is only bounded ({mat(M, fam, "ded", 1.1, "gain_lo"):.2f}–{mat(M, fam, "ded", 1.1, "gain_hi"):.2f}).')
            col = sd.DRED
        else:
            note = (f'Per bar end, dedicated; parasitic {fmt_t(tp)}. Real pulses respond linearly from {fmt_t(tl)} (spectra '
                    f'and bar ends). A 1-MIP end pulse stays poorly measured at dedicated intensity: σ(E)/E '
                    f'≈ {mat(M, fam, "ded", 10, "mip_sig_E"):.0%}, σ(t) ≈ {mat(M, fam, "ded", 10, "mip_sig_t_ns"):.0f} ns at 10 µs (pile-up).')
            col = sd.DBLUE
        cards.append(sd.bignum(fmt_t(tu), MIPN[fam], col, note, tip='usable = the later of (a MIP seen ≥ 90 % in every bin '
                                                                    'to 40 µs) and (real pulses show a linear response)'))
    cards.append(sd.bignum('—', 'liquid', sd.DGREEN,
                           'Not a lepton counter here: no MIP reference. A linear response is not established on real pulses '
                           '(dedicated and parasitic spectra differ by 10–20 % even at 60 µs). Injection-only numbers: appendix.'))
    head = 'When a MIP is seen after the γ-flash, and how much of that rests on real pulses'
    sub = (f'Plastics from ~{mfirst(MF, "PSS", "ded", "usable_from_us"):.1f} µs; SiPM walls ~{mfirst(MF, "WAL", "ded", "usable_from_us"):.0f} µs '
           f'at dedicated intensity, ~{mfirst(MF, "WAL", "par", "usable_from_us"):.0f} µs at parasitic. The efficiency comes from '
           'injected pulses, so it is only quoted where real pulses show the electronics to be linear')
    body = (sd.kicker('n_TOF EAR2 · X17 scintillators · the first microseconds after the γ-flash')
            + sd.title(f'<span style="color:{sd.DINK}">{head}</span>', f'<span style="color:{sd.DMUT}">{sub}</span>')
            + sd.row(*cards, gap=56)
            + sd.p(f'MIP = the measured cosmic-muon spectrum of each counter. Dedicated pulses (8.5×10¹² protons), time since the '
                   f'flash. {NI["n_bunch"]:,} bunches of {NI["n_runs"]} production runs.', 22, sd.DMUT))
    notes = ('<p>This version answers two concerns with the previous one. (1) The injection test adds pulses to the recorded '
             'traces, so it assumes the electronics are still linear after the flash and is a best case. Linearity is now '
             'tested on real pulses alone (slides 5–7), and the efficiency is only used where that test passes. (2) Results '
             'were shown as series of injected pulse sizes. They are now folded with the measured MIP spectrum: how often a '
             'MIP is lost, and how well it is measured.</p>'
             '<p>"Seen" = the pulse is found at its time. "Usable" = seen ≥ 90 % in every bin up to 40 µs, and from the time '
             'real pulses show a linear response. Correction found while reworking: the energy uncertainty of the previous '
             'version added the gain uncertainty in absolute instead of relative units, understating σ(E) at low gain by up '
             'to ×3 (7 % at 2 µs became 17–20 %).</p>')
    D.slide('cover', body, notes, dark=True, short='cover')


def definitions(D, LD, MC, z):
    panels = []
    med = MC.groupby('fam').a_mip_mV.mean()
    for fam in FAM2:
        g = LD[(LD.fam == fam) & (LD.kind == 'mip') & (LD.x < 3)]
        ymax = 2.0 if fam == 'PSS' else 1.6
        P = sd.Plot(800, 430, x=(0, 3), y=(0, ymax), xlabel="pulse size ÷ the counter's median MIP pulse",
                    ylabel='MIPs per unit size' if fam == 'PSS' else '', title=f'{MIPN[fam]}: measured MIP spectrum',
                    margin=(24, 24, 92, 96))
        if fam == 'PSS':
            a = LD[(LD.fam == fam) & (LD.kind == 'all') & (LD.x < 3)]
            xs, ys = sd.step_xy(list(np.r_[a.x - 0.025, a.x.iloc[-1] + 0.025]), list(a.density))
            P.band(xs, [0] * len(ys), clip(ys, 0, ymax), sd.GREY, alpha=0.25,
                   tip='every wall–plastic coincidence: the shoulder below ~0.6 is tracks clipping the bar, not a crossing MIP')
        xs, ys = sd.step_xy(list(np.r_[g.x - 0.025, g.x.iloc[-1] + 0.025]), list(g.density))
        if fam == 'WAL':
            P.band(xs, [0] * len(ys), clip(ys, 0, ymax), FCOL[fam], alpha=0.3)
        P.line(xs, clip(ys, 0, ymax), FCOL[fam], 3, markers=False,
               tip='through-going muons, fitted (Landau ⊗ Gauss)' if fam == 'PSS' else 'one end, all positions along the bar')
        for a_ in z[f'amps_{fam}']:
            x = a_ / med[fam]
            if x <= 3:
                P.vline(x, sd.INK, '3 5', 1.5, label=f'{a_:g}', tip=f'injected size {a_:g} mV = {x:.2f} × the median MIP')
        P.xticks([(v, f'{v:g}') for v in (0, 0.5, 1, 1.5, 2, 3)]).yticks([(v, f'{v:g}') for v in (0, 0.5, 1, 1.5)])
        panels.append(P.svg('L' + fam))
    rng = MC.groupby('fam').a_mip_mV.agg(['min', 'max'])
    rows = [['plastic bar, 20 mm', '≈ 4 MeV', f'{med["PSS"]:.0f} mV', f'{rng.loc["PSS", "min"]:.0f}–{rng.loc["PSS", "max"]:.0f} mV'],
            ['SiPM wall bar, 3 mm (one end)', f'{WALL_MIP_MEV:.2f} MeV, read at both ends', f'{med["WAL"]:.0f} mV',
             f'{rng.loc["WAL", "min"]:.0f}–{rng.loc["WAL", "max"]:.0f} mV'],
            ['liquid', 'no MIP reference', '—', '—']]
    body = (sd.title('What a pulse size means: a MIP is a spectrum of sizes, not one size',
                     'Cosmic muons 30–32 ms after each flash (no beam), wall and plastic in coincidence, accidentals subtracted. '
                     'Dotted: the sizes injected into the flash tails, in mV')
            + sd.row(*panels, gap=40)
            + sd.table(['counter', 'a MIP deposits', 'median MIP pulse (mean over counters)', 'range over counters'], rows,
                       size=21))
    notes = ('<p><b>A MIP</b> (minimum-ionizing particle: a cosmic muon here, an X17 e± in the experiment) loses ~2 MeV per cm '
             'of plastic, so ≈ 4 MeV in the 20 mm plastic bar and 0.47 MeV in the 3 mm wall bar. The deposit fluctuates '
             '(Landau) and the light collection adds a spread, so the pulse size is a distribution ~40 % wide: the efficiency '
             'for a MIP is the efficiency per size averaged over this distribution.</p>'
             '<p>The plastic coincidences also contain a shoulder of small pulses (grey, 15–33 % per bar): tracks that clip '
             'the edge of the bar. An X17 lepton from the target crosses the whole bar, so only the fitted through-going peak '
             'is used. For the walls the per-end size is the measured spectrum, all positions along the bar.</p>'
             '<p><b>What the earlier series were</b>: injected pulse sizes in mV (plastic 25–800, wall 8–120, liquid 25–400), '
             'labelled in MeVee or in MIP. MeVee is the light an electron of that energy would make (the appendix keeps that '
             'scale, anchored at 3.45 MeVee for the median of all coincidences, shoulder included). The MIP results fold the '
             'per-size efficiencies with this spectrum, after multiplying every size by the measured gain: at 2–3 µs a '
             'dedicated plastic MIP makes ~70 mV, not ~215 mV.</p>')
    D.slide('definitions', body, notes, short='what a MIP is')


def injection_scope(D, MF):
    tl = {f: mfirst(MF, f, 'ded', 'linear_from_us') for f in FAM2}
    rows = [
        ['is the pulse found? (efficiency)', 'injection', 'linear superposition'],
        ['reconstructed amplitude ÷ true (response, ±7 %)', 'injection', 'linear superposition'],
        ['spread of the amplitude from pile-up', 'injection', 'linear superposition'],
        ['time resolution', 'injection; walls checked on real bar ends', 'linear superposition'],
        ['PMT / SiPM gain (plastic ×3 low at 2–3 µs)', '<b>real pulses</b>: spectrum scale', 'same particles at both times'],
        ['is the response linear?', '<b>real pulses</b>: spectrum shape, bar ends', 'see slides 5–7'],
        ['MIP size distribution', '<b>real pulses</b>: cosmic muons', 'muon angles ≈ lepton angles'],
    ]
    body = (sd.title('What the injected pulses can and cannot tell',
                     'An injected pulse is added to the recorded trace: it tests the algorithm and the pile-up with the real '
                     'beam pulses, never how the detector and electronics respond to light after the flash')
            + sd.table(['quantity', 'measured with', 'assumes'], rows, size=24, align=['left', 'left', 'left'])
            + sd.row(sd.callout('Injection is the <b>best case</b>: if the PMT, SiPM, amplifier or digitizer were non-linear '
                                'after the flash (a compression, a baseline-dependent gain, a slower pulse), real pulses would do '
                                'worse than injected ones, and the test could not see it.', sd.RED, 23),
                     sd.callout(f'So linearity is tested <b>on real pulses only</b>, and the injection numbers are only used where '
                                f'that test passes: plastics from {fmt_t(tl["PSS"])}, walls from {fmt_t(tl["WAL"])}. Before '
                                'that, they are shaded grey.', sd.GREEN, 23), gap=40))
    notes = ('<p>Within that, the injection does more than exercise the algorithm: the pulses sit on the real flash tail of every '
             'bunch, among the real beam-induced pulses, so the losses and the spread it measures are the pile-up of this '
             'beam. A found fraction near 100 % for large pulses is expected and says little; the useful output is how the '
             'efficiency and the spread fall for small pulses as the pile-up density changes.</p>'
             '<p>A pure gain change is a linear response (every pulse scaled by the same factor). It is invisible to the '
             'injection and is measured from the real spectra instead (slide 8).</p>')
    D.slide('scope', body, notes, short='injection: scope')


def linearity_slide(D, S, MF):
    panels = []
    for fam in ('PSS', 'WAL', 'LIQ'):
        P = sd.Plot(540, 580, x=(0.3, 60, 'log'), y=(0, 0.5), xlabel='time since the flash',
                    ylabel='shape mismatch (max over q50–q99)' if fam == 'PSS' else '', title=FAMN[fam],
                    margin=(24, 24, 92, 96))
        P.band([0.3, 60], [0, 0], [LIN_TOL, LIN_TOL], sd.GREEN, alpha=0.10, tip=f'within ±{LIN_TOL:.0%}: a scaled copy')
        P.band([0.3, 2.5], [0, 0], [0.5, 0.5], sd.GREY, alpha=0.08,
               tip='before ~2.5 µs the particles differ from the 20–60 µs reference (fast-neutron recoils): the "time" test is not valid')
        for test, cls, col, dash, lab in (('flash', 'ded/par', sd.INK, None, 'dedicated vs parasitic, same time'),
                                          ('time', 'ded', CC['ded'], '6 5', 'dedicated vs its own 20–60 µs'),
                                          ('time', 'par', CC['par'], '6 5', 'parasitic vs its own 20–60 µs')):
            g = S[(S.fam == fam) & (S.test == test) & (S.cls == cls) & (S.n_ch >= 3)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.max_abs, 0, 0.5), col, 3.5 if test == 'flash' else 2.2, dash=dash,
                   markers=test == 'flash', r=4,
                   tips=[f'{lab}, {fmt_t(t)}: max mismatch {m:.1%}, scale {s:.2f} ({n} channels; {q0:.0f}–{q1:.0f} mV)'
                         for t, m, s, n, q0, q1 in zip(g.t_us, g.max_abs, g.scale, g.n_ch, g.q50_mV, g.q99_mV)], tip=lab)
        if fam in FAM2:
            t = mfirst(MF, fam, 'ded', 'linear_from_us')
            P.vline(t, sd.GREEN, '6 5', 2, label=f'linear from {fmt_t(t)}')
        P.xticks([(v, s) for v, s in ((0.3, '0.3 µs'), (1, '1 µs'), (3, '3 µs'), (10, '10 µs'), (30, '30 µs'))])
        P.yticks([(v, f'{v:.0%}') for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)])
        panels.append(P.svg('lin' + fam))
    body = (sd.title('Linearity on real pulses: plastics from 0.4 µs, walls from 2.5 µs, liquids not shown',
                     'Is the measured spectrum just a scaled copy of a reference? Thick: dedicated against parasitic bunches at '
                     'the same time (same particles, twice the flash). Dashed: against the same intensity at 20–60 µs')
            + sd.row(*panels, gap=24)
            + sd.legend([('dedicated ÷ parasitic, same time', sd.INK, 'line'), ('dedicated ÷ own 20–60 µs', CC['ded'], 'dash'),
                         ('parasitic ÷ own 20–60 µs', CC['par'], 'dash'), (f'within {LIN_TOL:.0%}', sd.GREEN, 'box')], size=20))
    notes = ('<p><b>The test.</b> A response that is linear up to a gain multiplies every amplitude by the same factor, so the '
             'spectrum keeps its shape. A non-linear response changes the shape: a compression of large pulses pulls the 99 % '
             'point in more than the median. Per channel and time bin, the spectrum of isolated real pulses (template fit, '
             'above 20 mV plastics/liquids, 15 mV walls) is fitted as a scaled copy of a reference, with the threshold applied '
             'consistently, and the mismatch of the 50/75/90/95/99 % points is kept; median over channels.</p>'
             '<p><b>Two references.</b> Dedicated against parasitic at the same time: the particles per proton are the same, '
             'only the flash is twice as large, so a flash-driven non-linearity shows while a change of particle mix cannot. '
             'Against the same intensity at 20–60 µs: valid only once the particles are the same capture-γ e± (from ~2.5 µs). '
             'Plastics: dedicated and parasitic agree in shape within 1–8 % from 0.4 µs, while their scale differs by ×0.3–0.7 '
             '(the dedicated gain is lower). Walls: the shapes differ by 30–45 % until 2.2 µs (the amplifier recovering from '
             'clipping) and agree within 2–5 % from 2.5 µs. Liquids: 10–20 % at every time to 60 µs, long after the flash, so '
             'the cause is probably not the flash (different run conditions behind the two intensities are possible), but '
             'linearity is not established.</p>'
             '<p><b>What it cannot see.</b> A non-linearity that is the same for both intensities (the dedicated vs parasitic '
             'test), or one that a change of particle mix happens to cancel. The amplitude range tested is the 50–99 % points: '
             '30–90 mV (plastics at 0.5 µs), 45–210 mV (walls), which contains the MIP pulse at the sagged gain. Code: '
             '<code>linearity.py</code>.</p>')
    D.slide('linearity', body, notes, short='linearity: spectra')


def shape_examples(D, E):
    panels = []
    for (fam, t), g in E.groupby(['fam', 't_us'], sort=False):
        ok = g.max_abs.iloc[0] <= LIN_TOL
        P = sd.Plot(540, 560, x=(10, 1000, 'log'), y=(0, float(g.frac.max()) * 1.15), xlabel='pulse amplitude [mV]',
                    ylabel='fraction per bin' if len(panels) == 0 else '',
                    title=f'{FAMN[fam]}, {fmt_t(t)}: ' + ('linear' if ok else 'not linear'), margin=(24, 24, 92, 96))
        d, p = g[g.cls == 'ded'].sort_values('a_mV'), g[g.cls == 'par'].sort_values('a_mV')
        P.line(d.a_mV.tolist(), d.frac.tolist(), CC['ded'], 3, markers=False, tip='dedicated')
        P.points(p.a_mV.tolist(), p.frac.tolist(), CC['par'], r=4, tips=None)
        P.text(P.X(12), P.Y(float(g.frac.max()) * 1.05), f'parasitic × {g.scale.iloc[0]:.2f}; mismatch {g.max_abs.iloc[0]:.0%}',
               size=19, color=sd.GREEN if ok else sd.RED)
        P.xticks([(v, f'{v:g}') for v in (10, 30, 100, 300, 1000)]).yticks([])
        panels.append(P.svg(f'ex{fam}{t:.1f}'))
    body = (sd.title('What the test looks like: the parasitic spectrum, scaled, on top of the dedicated one',
                     'Isolated real pulses of one channel (the one with most pulses), one time bin; parasitic amplitudes '
                     'multiplied by the fitted scale')
            + sd.row(*panels, gap=24)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic × scale', CC['par'], 'dot')]))
    F1 = E.groupby(['fam', 't_us'], as_index=False).first()
    p_ = F1[F1.fam == 'PSS'].iloc[0]
    w1, w2 = (r for _, r in F1[F1.fam == 'WAL'].sort_values('t_us').iterrows())
    notes = (f'<p>Plastic {p_.channel} at {fmt_t(p_.t_us)}: its dedicated pulses are {p_.scale:.2f} of the parasitic ones, and '
             'once scaled the two spectra lie on each other: the larger flash lowers the gain without changing the shape. '
             f'Wall {w1.channel} at {fmt_t(w1.t_us)}: the dedicated spectrum is narrower (fewer large pulses, more medium ones), '
             'which a scale cannot produce: the amplifier is still recovering from clipping. '
             f'Wall {w2.channel} at {fmt_t(w2.t_us)}: they agree again, at a scale of {w2.scale:.2f}; the dedicated/parasitic '
             'wall scale stays ~1.1 to 60 µs, so that offset is not a flash effect.</p>'
             '<p>The slide before uses the median over channels of exactly this comparison.</p>')
    D.slide('shape_examples', body, notes, short='linearity: examples')


def bar_ends_slide(D, B):
    B = B[(B.t_us > 0.3) & (B.t_us < 60)]
    P1 = sd.Plot(800, 560, x=(0.3, 60, 'log'), y=(-0.3, 0.3), xlabel='time since the flash',
                 ylabel='change of end ratio A₁/A₂', title='amplitude: ratio of the two ends at a fixed position',
                 margin=(24, 24, 92, 104))
    P2 = sd.Plot(800, 560, x=(0.3, 60, 'log'), y=(0, 12), xlabel='time since the flash',
                 ylabel='σ68(t₁ − t₂) [ns]', title='time: end-to-end difference at a fixed ratio', margin=(24, 24, 92, 96))
    P1.band([0.3, 60], [-0.05, -0.05], [0.05, 0.05], sd.GREEN, alpha=0.10, tip='within ±5 %')
    for cls in ('ded', 'par'):
        g = B[B.cls == cls].sort_values('t_us')
        r = np.exp(g.dlnr) - 1
        P1.line(g.t_us.tolist(), clip(r, -0.3, 0.3), CC[cls], 3, r=5,
                tips=[f'{CN[cls]}, {fmt_t(t)}: {v:+.1%} (bar-to-bar spread {s:.1%}; {n:,.0f} pairs)'
                      for t, v, s, n in zip(g.t_us, r, g.dlnr_spread, g.n_pairs)])
        P2.line(g.t_us.tolist(), clip(g.dt68_ns, 0, 12), CC[cls], 3, r=5,
                tips=[f'{CN[cls]}, {fmt_t(t)}: σ68 {s:.2f} ns (20–60 µs: {s0:.2f}), median shift {d:+.2f} ns'
                      for t, s, s0, d in zip(g.t_us, g.dt68_ns, g.dt68_ref_ns, g.ddt_ns)])
        P2.hline(float(g.dt68_ref_ns.median()), CC[cls], '2 5', 1.5, tip=f'{CN[cls]} at 20–60 µs')
    for P in (P1, P2):
        P.vline(2.5, sd.GREEN, '6 5', 2, label='2.5 µs')
        P.xticks([(v, s) for v, s in ((0.3, '0.3 µs'), (1, '1 µs'), (3, '3 µs'), (10, '10 µs'), (30, '30 µs'))])
    P1.yticks([(v, f'{v:+.0%}') for v in (-0.3, -0.15, 0, 0.15, 0.3)])
    P2.yticks([(v, f'{v:g}') for v in (0, 3, 6, 9, 12)])
    body = (sd.title('Walls on real pulses: the two ends of a bar agree in amplitude and time from 2.5 µs',
                     'Both ends see the same light. At a fixed position the ratio of their amplitudes, and at a fixed ratio '
                     'the spread of their time difference, against 20–60 µs. Median over 16 bars')
            + sd.row(P1.svg('be1'), P2.svg('be2'), gap=40)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'), ('20–60 µs', sd.MUT, 'dash')]))
    notes = ('<p>The two ends of a bar have different SiPM gains and so ride on different flash tails. If the gain of the '
             'electronics depended on the tail level, their ratio at a fixed position would move while the tails decay. It '
             'moves by ≤ 1 % from 2.5 µs (±3 % bar to bar), and by up to 4 % at 0.3–2 µs with a 7–24 % bar-to-bar scatter. '
             'The end-to-end time difference keeps its position within 0.2 ns from 2.5 µs; its spread is 3.3 / 2.8 ns at '
             '3 / 5 µs against 2.5 ns late, and 3.5–11 ns before 2.5 µs. This is the only test of timing on real pulses: '
             'the plastic timing is from the injection alone.</p>'
             '<p>Pairs within ±30 ns, accidentals from a 500 ns shifted window subtracted. Both ends share the flash, so a '
             'non-linearity common to both ends cancels here; the spectrum test on the previous slides covers that.</p>')
    D.slide('bar_ends', body, notes, short='linearity: wall bar ends')


def gain_slide(D, M, MF):
    panels = []
    for fam in FAM2:
        x0 = 0.3 if fam == 'PSS' else 2.0
        P = sd.Plot(800, 560, x=(x0, 40, 'log'), y=(0, 1.3), xlabel='time since the flash',
                    ylabel='gain (÷ its late value)' if fam == 'PSS' else '', title=MIPN[fam], margin=(24, 24, 92, 96))
        P.hline(1, sd.INK, None, 1.2)
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= x0) & (M.t_us <= 40)].sort_values('t_us')
            P.band(g.t_us.tolist(), clip(g.gain_lo, 0, 1.3), clip(g.gain_hi, 0, 1.3), CC[cls], alpha=0.22)
            m = g.gain_measured
            P.line(g.t_us[m].tolist(), clip(g.gain[m], 0, 1.3), CC[cls], 3.5, r=4,
                   tips=[f'{CN[cls]}, {fmt_t(t)}: {v:.2f} ({lo:.2f}–{hi:.2f})' for t, v, lo, hi in
                         zip(g.t_us[m], g.gain[m], g.gain_lo[m], g.gain_hi[m])])
            if (~m).any() and fam == 'PSS':
                P.points(g.t_us[~m].tolist(), clip(g.gain[~m], 0, 1.3), CC[cls], r=5, marker='open',
                         tips=[f'{CN[cls]}, {fmt_t(t)}: bounded only, {lo:.2f}–{hi:.2f}' for t, lo, hi in
                               zip(g.t_us[~m], g.gain_lo[~m], g.gain_hi[~m])])
        if fam == 'PSS':
            P.vline(2.0, sd.INK, '6 5', 1.5, label='measured from 2 µs')
        P.xticks([(v, s) for v, s in ((0.3, '0.3 µs'), (1, '1 µs'), (3, '3 µs'), (10, '10 µs'), (30, '30 µs')) if v >= x0])
        P.yticks([(v, f'{v:.1f}') for v in (0, 0.25, 0.5, 0.75, 1, 1.25)])
        panels.append(P.svg('g' + fam))
    body = (sd.title('The gain, from real pulses: plastic ×3 low at 2–3 µs; before 2 µs only bounded',
                     'Solid: measured (scale of the spectrum, gain_ref). Open circles and band before 2 µs: the range allowed by '
                     'the dedicated/parasitic scale (above) and the 99 % point of the spectrum (below)')
            + sd.row(*panels, gap=40)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'), ('range', sd.MUT, 'box')]))
    notes = ('<p>From 2 µs every plastic pulse above threshold is a capture-γ e±, so the scale of the spectrum against its late '
             'value is the gain (all quantiles move together, slide 5). Before 2 µs the pulses are fast-neutron recoils, '
             'whose spectrum has another shape, so the scale alone is not a gain. Two bounds remain. Below: the 99 % point '
             'against its late value (recoils are softer than capture γ, so this underestimates the gain). Above, for '
             'dedicated bunches: the dedicated/parasitic scale of the linearity test, since the dedicated gain is that '
             'factor times the parasitic gain and the parasitic gain is at most 1. Dedicated: 0.20–0.32 at 0.5–0.8 µs, '
             '0.21–0.64 at 0.8–1.6 µs. Parasitic: from the 99 % point up to 1.</p>'
             '<p>Walls: the spectrum scale is 1.05–1.18 at 2.5–10 µs against 20–60 µs, but the same 1.14 holds between '
             'dedicated and parasitic to 60 µs, so it is not taken as a gain; gain 1 with the spread as uncertainty.</p>')
    D.slide('gain', body, notes, short='gain')


def mip_eff_slide(D, M, MF):
    panels = []
    for fam in FAM2:
        x0 = 0.3 if fam == 'PSS' else 0.6
        P = sd.Plot(800, 600, x=(x0, 40, 'log'), y=(0, 1), xlabel='time since the flash',
                    ylabel='fraction of MIPs' if fam == 'PSS' else '', title=MIPN[fam], margin=(24, 24, 92, 96))
        shade_nonlinear(P, x0, mfirst(MF, fam, 'ded', 'linear_from_us'), 0, 1)
        P.hline(0.9, sd.INK, '2 5', 1.2, tip='90 %')
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= x0) & (M.t_us <= 40)].sort_values('t_us')
            P.band(g.t_us.tolist(), clip(g.found_lo, 0, 1), clip(g.found_hi, 0, 1), CC[cls], alpha=0.22)
            P.line(g.t_us.tolist(), clip(g.found, 0, 1), CC[cls], 4, r=4,
                   tips=[f'{CN[cls]}, {fmt_t(t)}: seen {f:.1%} ({lo:.1%}–{hi:.1%} over the gain range); within ±50 % {r:.1%}; '
                         f'MIP pulse {a:.0f} mV' for t, f, lo, hi, r, a in
                         zip(g.t_us, g.found, g.found_lo, g.found_hi, g.reco, g.mip_mV)])
            P.line(g.t_us.tolist(), clip(g.reco, 0, 1), CC[cls], 2.2, dash='6 5', markers=False,
                   tip=f'{CN[cls]}: seen and within ±50 % of its true size')
            t = mfirst(MF, fam, cls, 'usable_from_us')
            if np.isfinite(t):
                P.vline(t, CC[cls], '6 5', 1.5, label=fmt_t(t) if cls == 'ded' else None, tip=f'usable from {fmt_t(t)}, {CN[cls]}')
        P.xticks([(v, s) for v, s in ((0.3, '0.3 µs'), (1, '1 µs'), (3, '3 µs'), (10, '10 µs'), (30, '30 µs')) if v >= x0])
        P.yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 0.9, 1)])
        panels.append(P.svg('m' + fam))
    fp = M[(M.fam == 'PSS') & (M.cls == 'ded') & (M.t_us > 2) & (M.t_us < 30)]
    def rng(v):
        a, b = f'{v.min():.0%}', f'{v.max():.0%}'
        return a if a == b else f'{a[:-1]}–{b}'
    tw = mfirst(MF, 'WAL', 'ded', 'usable_from_us')
    fw = M[(M.fam == 'WAL') & (M.cls == 'ded') & (M.t_us > tw) & (M.t_us < 30)]
    body = (sd.title(f'MIP efficiency: a plastic MIP is seen {rng(fp.found)} of the time from 2 µs; '
                     f'a wall end {rng(fw.found)} from {fmt_t(tw)} (dedicated)',
                     'Measured MIP spectrum × measured gain × injected-pulse efficiency per size. Thick: seen; dashed: seen and '
                     'within ±50 %; band: over the gain range; grey: response not shown linear (best case)')
            + sd.row(*panels, gap=40)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'), ('within ±50 %', sd.MUT, 'dash'),
                         ('not shown linear', sd.GREY, 'box')]))
    notes = ('<p>For every time bin, each MIP of the measured spectrum makes a pulse of its size (mV, at the late gain) × '
             'the gain; the injected-pulse efficiency at that size (interpolated between the injected sizes) is averaged over '
             'the spectrum and over the counters. "Seen": the reconstruction finds a pulse at its time. "Within ±50 %": its '
             'amplitude is usable.</p>'
             '<p>Plastics: early the whole spectrum is pushed down by the gain (at 0.5–0.7 µs a dedicated MIP makes ~55 mV) and '
             'its low side drops into the pile-up. Within ±50 % is what the capture-γ pile-up spoils at 5–20 µs: the dashed '
             'curve stays at 0.92 (dedicated) and 0.96 (parasitic). Walls: a ~35 mV, 500 ns end pulse under the rain of real '
             'pulses; the parasitic intensity halves the rain.</p>'
             '<p>The fraction of MIPs below the smallest injected size (extrapolated): ≤ 2 % from 4 µs on the plastics, up to '
             '10 % at 0.5–0.7 µs (dedicated). Code: <code>mip_eff.py</code>.</p>')
    D.slide('mip_eff', body, notes, short='MIP efficiency')


def mip_res_slide(D, M, MF):
    panels = []
    for fam, col, ylab, yr, scale in (('PSS', 'mip_sig_E', 'σ(E)/E', (0, 0.8), 'lin'), ('WAL', 'mip_sig_E', '', (0, 0.8), 'lin'),
                                      ('PSS', 'mip_sig_t_ns', 'σ68(t) [ns]', (0.05, 20), 'log'), ('WAL', 'mip_sig_t_ns', '', (0.05, 20), 'log')):
        x0 = 0.3 if fam == 'PSS' else 2.0
        P = sd.Plot(420, 560, x=(x0, 40, 'log'), y=(*yr, scale), xlabel='time since the flash', ylabel=ylab,
                    title=f'{FAMN[fam]}: {"energy" if col == "mip_sig_E" else "time"}', margin=(24, 18, 92, 92))
        shade_nonlinear(P, x0, mfirst(MF, fam, 'ded', 'linear_from_us'), yr[0], yr[1])
        if fam == 'PSS' and col == 'mip_sig_E':
            P.band([x0, 2.0], [yr[0]] * 2, [yr[1]] * 2, sd.GOLD, alpha=0.12, tip='gain only bounded: the energy scale is uncertain')
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= max(x0, 0.4)) & (M.t_us <= 40)].sort_values('t_us')
            if col == 'mip_sig_E':
                P.line(g.t_us.tolist(), clip(g[col], *yr), CC[cls], 3.5, r=4,
                       tips=[f'{CN[cls]}, {fmt_t(t)}: {v:.0%} (pile-up spread {a:.0%}, gain {b:.0%}); MIP pulse {m:.0f} mV'
                             for t, v, a, b, m in zip(g.t_us, g[col], g.mip_sig_res, g.mip_sig_sys, g.mip_mV)])
            else:
                P.line(g.t_us.tolist(), clip(g[col], *yr), CC[cls], 3.5, r=4,
                       tips=[f'{CN[cls]}, {fmt_t(t)}: σ68 {v:.2f} ns; MIP pulse {m:.0f} mV' for t, v, m in
                             zip(g.t_us, g[col], g.mip_mV)])
        P.xticks([(v, s) for v, s in ((0.3, '0.3'), (1, '1 µs'), (3, '3'), (10, '10 µs'), (30, '30')) if v >= x0])
        P.yticks([(v, f'{v:.0%}') for v in (0, 0.2, 0.4, 0.6, 0.8)] if col == 'mip_sig_E' else
                 [(v, f'{v:g}') for v in (0.1, 1, 10)])
        panels.append(P.svg(f'r{fam}{col}'))
    body = (sd.title('How well a MIP is measured: plastic σ(E)/E ≈ 20 % and σ(t) ≈ 0.5 ns at 2–5 µs; a wall end ~40–60 % and 2–5 ns',
                     'At the median MIP, its pulse size reduced by the gain; σ(E) = pile-up spread (injection) ⊕ gain '
                     'uncertainty (data). Gold: gain only bounded; grey: not shown linear')
            + sd.row(*panels, gap=16)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line')]))
    notes = ('<p>The previous version quoted these for a 200 mV pulse at every time. At 2–3 µs a dedicated plastic MIP is ~70 mV '
             'because of the gain, and a smaller pulse has a larger relative spread under the same pile-up, so the MIP '
             'numbers are worse: σ(E)/E 20 % (11–13 % pile-up, 15–17 % gain) and σ(t) 0.4–0.5 ns. Before 2 µs the gain range '
             'dominates (±50 %). From ~10 µs, 13–14 % dedicated and 7–8 % parasitic.</p>'
             '<p>Walls, one end, a MIP: 37–60 % and 2.3–5 ns at dedicated intensity from 4.5 µs, 13–26 % and 0.9–2.3 ns parasitic, from the '
             'injection; the real bar ends (previous slide) give 3.3 → 2.5 ns for the end-to-end difference at 3 µs → late, for '
             'the whole spectrum of sizes.</p>')
    D.slide('mip_res', body, notes, short='MIP energy and time')


def mip_table(D, M, MF):
    rows, tips = [], []
    for fam in FAM2:
        for cls in ('ded', 'par'):
            tl = mfirst(MF, fam, cls, 'linear_from_us')
            for t in ((0.5, 1, 2, 3, 5, 10, 20) if fam == 'PSS' else (3, 5, 10, 20)):
                if not np.isfinite(mat(M, fam, cls, t, 'found')):
                    continue
                gm = mat(M, fam, cls, t, 'gain_measured')
                gtxt = (f'{mat(M, fam, cls, t, "gain"):.2f}' if gm else
                        f'{mat(M, fam, cls, t, "gain_lo"):.2f}–{mat(M, fam, cls, t, "gain_hi"):.2f}')
                rows.append([f'{MIPN[fam]}, {CN[cls]}', fmt_t(t), '✓' if t >= tl else '—', gtxt,
                             f'{mat(M, fam, cls, t, "mip_mV"):.0f} mV', f'{mat(M, fam, cls, t, "found"):.0%}',
                             f'{mat(M, fam, cls, t, "reco"):.0%}', f'{mat(M, fam, cls, t, "mip_sig_E"):.0%}',
                             f'{mat(M, fam, cls, t, "mip_sig_t_ns"):.2f} ns'])
                tips.append(f'σ(E): pile-up {mat(M, fam, cls, t, "mip_sig_res"):.0%}, gain {mat(M, fam, cls, t, "mip_sig_sys"):.0%}; '
                            f'seen over the gain range {mat(M, fam, cls, t, "found_lo"):.0%}–{mat(M, fam, cls, t, "found_hi"):.0%}')
    body = (sd.title('In numbers: a MIP after the flash',
                     'Linear = real pulses show a linear response (slides 5–7); gain range where it is only bounded; the MIP '
                     'pulse is the median MIP at that gain')
            + sd.table(['counter', 'time', 'linear', 'gain', 'MIP pulse', 'seen', '±50 %', 'σ(E)/E', 'σ(t)'], rows, size=16,
                       tips=tips))
    notes = ('<p>Every time bin: <code>out_dir("early")/mip_eff.csv</code>; first times: <code>mip_first.csv</code>; linearity: '
             '<code>out_dir("linearity")</code>. Hover a row for the split of σ(E) and the efficiency range over the gain.</p>')
    D.slide('mip_table', body, notes, short='numbers')


def implications(D, M, MF):
    items = [
        (f'<b>Plastics</b> see a MIP from ~{mfirst(MF, "PSS", "ded", "usable_from_us"):.1f} µs and respond linearly from 0.4 µs, but '
         'the PMT gain is ×3–5 low in the first µs (dedicated) and is only bounded before 2 µs. A light-pulser gain monitor '
         'firing in the first µs would turn the ±50 % energy scale before 2 µs into a measurement; a base built for high '
         'anode current would shrink the sag itself and raise the early MIP pulse out of the pile-up.'),
        ('<b>SiPM walls</b> are not linear until ~2.5 µs (the amplifier recovers from clipping), and after that a 1-MIP end '
         f'pulse sits in the pile-up of real pulses: ≥ 90 % seen from ~{mfirst(MF, "WAL", "ded", "usable_from_us"):.0f} µs at dedicated intensity, but measured within ±50 % only 64–77 % of the time. Faster shaping '
         '(shorter pulses) is the lever: the pile-up probability scales with the pulse length.'),
        ('<b>Liquids</b>: no MIP reference, and their linearity is not established on real pulses.'),
        ('<b>Everything early scales with intensity</b>: at parasitic intensity the plastic gain loss halves and a wall end '
         f'sees a MIP ≥ 90 % from ~{mfirst(MF, "WAL", "par", "usable_from_us"):.0f} µs.'),
    ]
    body = (sd.title('What this means for a measurement in the first microseconds',
                     'From the 2026 data; for this apparatus at 4 and 8.5×10¹² protons per pulse')
            + ''.join(sd.callout(t, c, 24) for t, c in zip(items, (sd.RED, sd.BLUE, sd.GREEN, sd.GOLD))))
    D.slide('implications', body, '', short='implications')


def caveats(D):
    items = [
        'The efficiency, the reconstruction response, the pile-up spread and the plastic timing come from injected pulses and '
        'assume linear superposition. Real pulses support that from 0.4 µs (plastics) and 2.5 µs (walls), within ~10 % in '
        'spectrum shape. A non-linearity equal at both intensities, or one a change of particle mix cancels, is not excluded.',
        'Before 2 µs the plastic gain is bounded (dedicated 0.2–0.64), not measured: energies there are uncertain by ~±50 %.',
        'The plastic timing is not tested on real pulses (no partner of sub-ns resolution); the wall timing is, on bar ends.',
        'The MIP spectrum is cosmic muons at an angle; a lepton at normal incidence makes a slightly smaller pulse, so the '
        'efficiencies are slightly optimistic for the walls, where the MIP sits at the edge of the pile-up. The plastic '
        'spectrum is the fitted through-going peak; counting the clipping-track shoulder as MIPs would lower the efficiencies.',
        'Liquids: linearity not established, no energy scale; their injection numbers (appendix) are a best case.',
        'Pile-up scales with the pulse rate: these numbers hold for this apparatus and these intensities.',
    ]
    body = (sd.kicker('What this does not rule out', dark=True) + ''.join(sd.p(f'· {t}', 24, sd.DINK) for t in items))
    D.slide('limits', body, '', dark=True, short='limits')


def appendix_sizes(D):
    body = (sd.kicker('Appendix A · per injected pulse size', dark=True)
            + sd.p('The injection results for each injected size, before folding with the MIP spectrum: the view of the previous '
                   'version. Best case (linear superposition), and in the old size units (wall MIP = 28.5 mV per end from the muon '
                   'calibration; the through-going in-situ median is 35 mV).', 26, sd.DINK))
    D.slide('appendix_sizes', body, '', dark=True, short='appendix A')


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
    body = (sd.kicker('Appendix B · the long-time recovery', dark=True)
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
    M = pd.read_csv(eo / 'mip_eff.csv')
    MF = pd.read_csv(eo / 'mip_first.csv')
    LD = pd.read_csv(eo / 'mip_landau.csv')
    lo = out_dir('linearity')
    S = pd.read_csv(lo / 'shape_tests.csv')
    B = pd.read_csv(lo / 'bar_ends.csv')
    LF = pd.read_csv(lo / 'linear_from.csv')
    E = pd.read_csv(lo / 'shape_examples.csv')
    X = dict(np.load(eo / 'examples.npz'))
    ES = energy_scale()
    z = np.load(out_dir('inject') / 'inject.npz')
    NI = dict(n_bunch=int(z['n_bunch'].sum()), n_runs=int(z['n_files']))
    ro = out_dir('reco')
    Hd = pd.read_csv(ro / 'headline.csv')
    Si = pd.read_csv(out_dir('inject') / 'inject_summary.csv')
    Q = pd.read_csv(out_dir('gain') / 'gain_quantiles.csv')
    R = pd.read_csv(out_dir('g4mix') / 'g4_rates.csv')
    D = sd.Deck('Scintillator pulses in the first microseconds after the γ-flash',
                'When each X17 scintillator sees a MIP after the n_TOF γ-flash, how well it measures it, and what rests on real pulses')
    cover(D, M, MF, LF, NI)
    traces(D, X)
    definitions(D, LD, pd.read_csv(eo / 'mip_counters.csv'), z)
    injection_scope(D, MF)
    linearity_slide(D, S, MF)
    shape_examples(D, E)
    bar_ends_slide(D, B)
    gain_slide(D, M, MF)
    mip_eff_slide(D, M, MF)
    mip_res_slide(D, M, MF)
    mip_table(D, M, MF)
    implications(D, M, MF)
    caveats(D)
    # appendix A: per injected size
    appendix_sizes(D)
    earliest(D, T, FT, ES)
    corrections(D, T, ES, FT)
    energy_unc(D, T, ES, FT)
    timing(D, T, ES, FT)
    small_signals(D, TH, ES)
    method(D, NI)
    # appendix B: the long-time recovery
    appendix_divider(D, Hd, ES)
    A.mix(D, R)
    A.gain_slide(D, Q)
    A.pile_clean(D, Si)
    A.good_slide(D, Si, A.energy_scale())
    A.model_slide(D, pd.read_csv(ro / 'occupancy_model.csv'), Si, Hd)
    A.bias_slide(D, pd.read_csv(ro / 'figures' / 'bias_vs_t.csv'))
    A.table_slide(D, pd.read_csv(ro / 'performance.csv'), A.energy_scale())
    path = D.write(out_dir('deck') / 'scint-flash-reconstruction.html', note_meta=dict(
        title='Scintillator pulses in the first microseconds after the γ-flash',
        summary='When each X17 scintillator sees a MIP after the n_TOF γ-flash (measured MIP spectrum × measured gain × '
                'injected-pulse efficiency), how well it measures its energy and time, and where real pulses show the '
                'electronics to be linear; per-size results and the long-time recovery in the appendix.',
        tags='X17, n_TOF, scintillators', date=dt.date.today().isoformat()))
    print(f'-> {path}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
