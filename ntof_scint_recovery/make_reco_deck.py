#!/usr/bin/env python3
"""
make_reco_deck.py -- the slide note of the final flash-recovery analysis: from when after the
γ-flash a scintillator pulse is reconstructed accurately, how biased it is and how well
resolved, in amplitude, energy and time.

    .venv/bin/python -m ntof_scint_recovery.make_reco_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        <out>/deck/scint-flash-reconstruction.html --slug scint-flash-reconstruction --force --deploy

Reads only ``headline.py``'s outputs (out_dir('reco')), the injection summary
(out_dir('inject')), the gain quantiles (out_dir('gain')) and the Geant4 mix (out_dir('g4mix')).
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.headline import FAMN, METHOD, REF_AMP, WALL_MIP_MEV, energy_scale, to_energy

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

FCOL = {'WAL': sd.PURPLE, 'PSS': sd.ORANGE, 'LIQ': sd.GREEN}
CN = {'par': 'parasitic', 'ded': 'dedicated'}
CC = {'par': sd.BLUE, 'ded': sd.ORANGE}
#: plots start here: the walls are switched off (divert gate) until ~0.65 µs; the plastics and liquids saturate
#: for the first ~0.2 µs
T0 = {'WAL': 0.7, 'PSS': 0.3, 'LIQ': 0.3}
FIRST_NOTE = 'https://dylan-neff.web.cern.ch/notes/scint-flash-recovery.html'


def fmt_t(us):
    if us is None or not np.isfinite(us):
        return '—'
    if us < 10:
        return f'{us:.1f} µs'
    if us < 1000:
        return f'{us:.0f} µs'
    return f'{us / 1e3:.1f} ms'


def tticks(lo, hi):
    labs = {0.1: '0.1 µs', 1: '1 µs', 10: '10 µs', 100: '100 µs', 1e3: '1 ms', 1e4: '10 ms'}
    return [(v, s) for v, s in labs.items() if lo <= v <= hi]


def clip(v, lo, hi):
    return [min(max(float(x), lo), hi) if (x is not None and np.isfinite(x)) else None for x in v]


def amp_colors(n):
    pal = ['#2b4c7e', '#3f7fb5', '#4fa3a5', '#7cbf6b', '#c9b13b', '#d9822b']
    return pal[:n] if n <= len(pal) else pal


# ------------------------------------------------------------------ slides
def cover(D, Hd, ES):
    big = []
    for fam in ('PSS', 'WAL', 'LIQ'):
        r = Hd[(Hd.fam == fam) & (Hd.cls == 'ded') & (Hd.amp_mV == REF_AMP[fam])]
        if not len(r):
            continue
        r = r.iloc[0]
        t = r.t_headline_us
        sub = (f'~1 MIP ({r.energy}): ≥ 90 % of pulses within 10 %'
               + (', gain ≥ 0.95' if fam == 'PSS' else '') + f'. Flash tail alone: {fmt_t(r.t_tail_loose_us)}')
        val = (fmt_t(t) + ('*' if r.headline_from_model else '')) if np.isfinite(t) else f'> {fmt_t(r.head_end_us)}'
        big.append(sd.bignum(val, FAMN[fam], {'PSS': sd.DRED, 'WAL': sd.DBLUE, 'LIQ': sd.DGREEN}[fam], sub,
                             tip=f'flash tail alone (clean pulses): from {fmt_t(r.t_tail_us)}; '
                                 f'gain ≥ 0.95 from {fmt_t(r.t_gain95_us)}'))
    head = 'When can a scintillator pulse be trusted again after the γ-flash?'
    sub = ('Pulses are found and unbiased (median) within a few µs, but the beam-induced pulse rate (pile-up) keeps '
           'a share of them off by > 10 % for much longer, and the plastic PMTs lose gain for ~10 µs')
    body = (sd.kicker('n_TOF EAR2 · X17 scintillators · γ-flash recovery, the full analysis')
            + sd.title(f'<span style="color:{sd.DINK}">{head}</span>', f'<span style="color:{sd.DMUT}">{sub}</span>')
            + sd.row(*big, gap=56)
            + sd.p('Dedicated pulses (8.5×10¹² protons); parasitic pulses are 1.5–3× faster. Time since the flash at the detector. '
                   '* beyond the recorded head block: from the measured pulse rate and the pile-up model (slide 6).', 24, sd.DMUT))
    notes = ('<p>"Trusted" here: a pulse of about one minimum-ionising particle is found and reconstructed within 10 % of '
             'its true amplitude in at least 90 % of cases, pile-up with real beam-induced pulses included, <i>and</i> the '
             'photodetector gain (measured from the data) is back within 5 %. The walls and liquids have no gain sag that the '
             'data can see or separate; for them only the reconstruction counts.</p>'
             '<p>Three independent pieces: (1) pulses injected into the real recorded flash tails and reconstructed '
             '(reconstruction bias and resolution); (2) the detector gain from the shape of the measured amplitude spectra '
             '(the injection cannot see gain); (3) Geant4 for what particles arrive when. Energy from the cosmic muons at '
             f'30–32 ms and the muon calibration. The first look (baselines, pulse shape) is the earlier note: '
             f'<a href="{FIRST_NOTE}">scint-flash-recovery</a>.</p>')
    D.slide('cover', body, notes, dark=True, short='cover')


def method(D, NI):
    steps = [('real flash trace', 'every channel, every bunch: the unsuppressed head block, 10 µs before to 30–200 µs after the flash'),
             ('+ injected pulses', 'family template, known amplitude and time (sub-ns), a comb every 0.6–2 µs, clipped at saturation'),
             ('reconstruct', 'running-median baseline → peak finder → template fit (local linear baseline, sub-ns shift); '
                             'single-pulse, and multi-pulse with every neighbour'),
             ('compare', 'found?  A_fit / A,  t_fit − t₀;  the same 1.5–8 µs BEFORE the flash is the control')]
    body = (sd.title('Method: put pulses of known size into the real flash tails, and reconstruct them',
                     f'{NI["n_inj"]:,} injected pulses in {NI["n_bunch"]:,} bunches of {NI["n_runs"]} production runs; '
                     'every number is split by beam intensity')
            + sd.flow([dict(label=a, sub=b, color=c) for (a, b), c in zip(steps, (sd.GREY, sd.ORANGE, sd.BLUE, sd.GREEN))],
                      size=26)
            + sd.row(sd.card(sd.p('<b>What it measures</b>: the electronics and the algorithm on the real tail: baseline, '
                                  'clipping, shape, and pile-up with the real beam-induced pulses.', 24), w=780),
                     sd.card(sd.p('<b>What it cannot</b>: a gain change in the PMT/SiPM (the pulse is added after it). '
                                  'That comes from the spectra (slide 7).', 24), w=780), gap=40))
    notes = ('<p>Amplitudes: walls 8–120 mV per bar end (0.3–4 MIP), plastics 25–800 mV (0.45–14 MeVee), liquids 25–400 mV. '
             'Each pass adds a comb of pulses with a random offset; 12 passes per amplitude, per channel, per bunch. '
             'A pulse is found if a peak lies within 40 ns (walls) / 6 ns (plastics, liquids) of it.</p>'
             '<p>"Clean" pulses: no real pulse at least 5 % of their size within a pulse width (template above 5 % of peak: '
             '217 / 42 / 18 ns). They isolate the flash tail itself; all pulses include the beam-induced pile-up, which is '
             'what an analysis gets.</p>'
             '<p>Code: <code>ntof_scint_recovery/inject.py</code> (condor, <code>flash_tail_inject/</code>), '
             '<code>inject_analysis.py</code>, <code>headline.py</code>.</p>')
    D.slide('method', body, notes, short='method')


def mix(D, R):
    P1 = sd.Plot(810, 560, x=(0.05, 1e4, 'log'), y=(0, 1), xlabel='time since the flash',
                 ylabel='share of proton recoils', title='Geant4: what makes a deposit > 0.1 MeVee')
    P2 = sd.Plot(810, 560, x=(0.05, 1e4, 'log'), y=(0.01, 1e5, 'log'), xlabel='time since the flash',
                 ylabel='pulses / µs / bunch (dedicated)', title='rate: Geant4 (solid) and data (dashed)')
    for fam, nm in (('wall', 'WAL'), ('plastic', 'PSS'), ('liquid', 'LIQ')):
        g = R[R.fam == fam]
        t = np.sqrt(g.t_lo_us * g.t_hi_us).to_numpy()
        P1.line(t.tolist(), clip(g.f_proton, 0, 1), FCOL[nm], 3.5, markers=True, r=4,
                tips=[f'{fam}, {fmt_t(a)}: {b:.0%} protons' for a, b in zip(t, g.f_proton)])
        P2.line(t.tolist(), clip(g.g4_rate_per_us, 0.01, 1e5), FCOL[nm], 3.5, markers=False,
                tip=f'{fam}, Geant4 per counter')
        P2.line(t.tolist(), clip(g.data_rate_per_us, 0.01, 1e5), FCOL[nm], 2.5, dash='8 6', markers=False,
                tip=f'{fam}, data per channel, > 4 mV')
    P1.xticks(tticks(0.05, 1e4)).yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
    P2.xticks(tticks(0.05, 1e4)).yticks([(10 ** k, f'10{sd.sup(k)}') for k in range(-2, 6)])
    pl = R[(R.fam == 'plastic') & (R.t_lo_us >= 3) & (R.t_hi_us <= 300)]
    rr = np.nanmedian(pl.g4_rate_per_us / pl.data_rate_per_us)
    late = R[(R.fam == 'plastic') & (R.t_lo_us >= 1000)]
    rl = np.nanmedian(late.g4_rate_per_us / late.data_rate_per_us)
    body = (sd.title('Fast-neutron recoils until ~2 µs, then capture γ: the population settles early',
                     'Geant4, 5×10⁸ EAR2 neutrons (1 meV–100 MeV) on the n_TOF apparatus, light with Birks quenching')
            + sd.row(P1.svg('mix'), P2.svg('rate'), gap=44)
            + sd.legend([('SiPM wall', FCOL['WAL'], 'line'), ('plastic', FCOL['PSS'], 'line'), ('liquid', FCOL['LIQ'], 'line'),
                         ('data', sd.MUT, 'dash')]))
    notes = (f'<p>At 19.5 m a neutron arriving 1 µs after the flash has ~2 MeV; at 2 µs ~0.5 MeV. Its proton recoils then give '
             'less than 0.1 MeVee (quenching), so from ~2 µs everything above threshold is e± from γ: capture γ in the '
             'apparatus (hydrogen 2.2 MeV dominates in the simulation). In the simulation the deposit spectrum keeps one shape '
             'from ~2 µs to ~1 ms (95 % quantile 1.99 MeVee, the 2.2 MeV Compton edge), then hardens slightly.</p>'
             f'<p>Rates: the simulation is within ×{1 / rr:.1f} of the data on the plastics at 3–300 µs (Geant4 / data = {rr:.2f}) '
             f'but ×{1 / rl:.0f} low after 1 ms, and its spectrum is softer than the data (data 95 % quantile behind a wall hit '
             '4.6–5.3 MeVee, simulation ~2 MeVee). Not simulated: the experimental hall (room-return neutrons, captures in '
             'concrete and the dump), the in-beam γ of the spallation target, the flash itself. The beam profile is the '
             'pre-2026-10-14 one (5× too narrow). So Geant4 explains what arrives early, not all of what arrives late.</p>'
             '<p>Code: <code>g4_mix.py</code> (condor over <code>full_sim/neutrons_fullrange</code>), '
             '<code>g4_mix_analysis.py</code>.</p>')
    D.slide('g4_mix', body, notes, short='Geant4 mix')


def good_slide(D, S, ES):
    s = S[(S.method == METHOD) & (S.tbin > 0)]
    panels = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        g0 = s[s.fam == fam]
        tmax = g0.t_hi_us.max()
        x0 = T0[fam]
        g0 = g0[g0.t_us >= x0]
        P = sd.Plot(540, 560, x=(x0, max(tmax, 20), 'log'), y=(0, 1), xlabel='time since the flash',
                    ylabel='within 10 % of true' if fam == 'WAL' else '', title=FAMN[fam], margin=(24, 24, 92, 90))
        P.band([x0, max(tmax, 20)], [0.9, 0.9], [1, 1], sd.GREEN, alpha=0.10)
        amps = sorted(g0.amp_mV.unique())
        for A, c in zip(amps, amp_colors(len(amps))):
            g = g0[(g0.cls == 'ded') & (g0.amp_mV == A)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.good10, 0, 1), c, 3, markers=False,
                   tip=f'{A:g} mV ≈ {to_energy(fam, A, ES)}, dedicated')
        g = g0[(g0.cls == 'par') & (g0.amp_mV == REF_AMP[fam])].sort_values('t_us')
        P.line(g.t_us.tolist(), clip(g.good10, 0, 1), sd.INK, 2.5, dash='8 6', markers=False,
               tip=f'{REF_AMP[fam]:g} mV, parasitic')
        P.xticks(tticks(x0, max(tmax, 20))).yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
        panels.append(P.svg(fam))
    leg = []
    for fam in ('PSS',):
        amps = sorted(s[s.fam == fam].amp_mV.unique())
        leg += [(f'{to_energy(fam, A, ES)}', c, 'line') for A, c in zip(amps, amp_colors(len(amps)))]
    body = (sd.title('Fraction of pulses reconstructed within 10 %: bigger pulses recover first; pile-up sets the pace',
                     'Injected pulses found and within 10 % of their amplitude, dedicated; dashed: ~1 MIP at parasitic '
                     'intensity. Colours: amplitude (plastic energies in the legend; walls 0.3–4 MIP, liquids 25–400 mV)')
            + sd.row(*panels, gap=24)
            + sd.legend(leg + [('~1 MIP, parasitic', sd.INK, 'dash'), ('≥ 90 %', sd.GREEN, 'box')], size=21))
    notes = ('<p>Every curve stops where the unsuppressed head block ends (walls ~40 µs, liquids 20–150 µs, plastics ~200 µs): '
             'only there is the full trace recorded, so only there can a pulse be injected. After it, the zero-suppression '
             'keeps blocks around pulses, and the rate (slide 3) is what changes.</p>'
             '<p>The losses are mostly pile-up: a real pulse within a pulse width that pulls the fitted amplitude. The fraction of '
             'pulses with such a neighbour is on the next slide; on clean pulses alone the reconstruction is already unbiased '
             'within a few µs.</p>')
    D.slide('good10', body, notes, short='accurate fraction')


def pile_clean(D, S):
    s = S[(S.method == METHOD) & (S.tbin > 0)]
    s = s[s.t_us >= s.fam.map(T0)]
    P1 = sd.Plot(810, 560, x=(0.3, 250, 'log'), y=(0, 1), xlabel='time since the flash',
                 ylabel='P(real pulse overlaps)', title='pile-up probability, ~1 MIP')
    P2 = sd.Plot(810, 560, x=(0.3, 250, 'log'), y=(-10, 10), xlabel='time since the flash',
                 ylabel='amplitude bias [%]', title='clean pulses only: the tail itself')
    P2.band([0.3, 250], [-3, -3], [3, 3], sd.GREEN, alpha=0.10)
    for fam in ('WAL', 'PSS', 'LIQ'):
        for cls, dash in (('ded', None), ('par', '8 6')):
            g = s[(s.fam == fam) & (s.cls == cls) & (s.amp_mV == REF_AMP[fam])].sort_values('t_us')
            P1.line(g.t_us.tolist(), clip(g.f_piled, 0, 1), FCOL[fam], 3, dash=dash, markers=False,
                    tip=f'{FAMN[fam]}, {CN[cls]}')
            m = g.n_clean >= 100
            P2.line(g.t_us[m].tolist(), clip(100 * g.bias_clean[m], -10, 10), FCOL[fam], 3, dash=dash, markers=True, r=4,
                    tips=[f'{FAMN[fam]}, {CN[cls]}, {fmt_t(t)}: {100 * b:+.1f} % ({int(n)} clean)'
                          for t, b, n in zip(g.t_us[m], g.bias_clean[m], g.n_clean[m])])
    for P in (P1, P2):
        P.xticks(tticks(0.3, 250))
    P1.yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
    P2.yticks([(v, f'{v:+d} %') for v in (-10, -5, 0, 5, 10)])
    body = (sd.title('Most of the loss is pile-up with real beam pulses; a clean pulse is unbiased after a few µs',
                     'Left: chance that a real pulse ≥ 5 % of the injected one sits within a pulse width. '
                     'Right: amplitude bias of the injected pulses that have no such neighbour')
            + sd.row(P1.svg('pile'), P2.svg('clean'), gap=44)
            + sd.legend([('SiPM wall', FCOL['WAL'], 'line'), ('plastic', FCOL['PSS'], 'line'), ('liquid', FCOL['LIQ'], 'line'),
                         ('parasitic', sd.MUT, 'dash'), ('±3 %', sd.GREEN, 'box')]))
    notes = ('<p>The pile-up probability is the beam-induced occupancy: the pulse rate times the pulse width. It falls with the '
             'rate, roughly in proportion to the beam intensity (dashed = parasitic, half the protons). It is not a property of '
             'the flash and does not go away with better electronics; a narrower pulse or a multi-pulse fit (used here) '
             'reduces what it costs.</p>'
             '<p>Clean pulses are rare where the occupancy is high, so their points start late on the walls and plastics; points '
             'with fewer than 100 clean pulses are not drawn.</p>')
    D.slide('pileup', body, notes, short='pile-up vs tail')


def model_slide(D, OM, S, Hd):
    from ntof_scint_recovery.headline import data_rate
    s = S[(S.method == METHOD) & (S.tbin > 0)]
    panels = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        P = sd.Plot(540, 560, x=(1, 1e4, 'log'), y=(0, 1), xlabel='time since the flash',
                    ylabel='within 10 % of true' if fam == 'WAL' else '', title=f'{FAMN[fam]}, ~1 MIP', margin=(24, 24, 92, 90))
        P.band([1, 1e4], [0.9, 0.9], [1, 1], sd.GREEN, alpha=0.10)
        for cls in ('ded', 'par'):
            m = OM[(OM.fam == fam) & (OM.cls == cls) & (OM.amp_mV == REF_AMP[fam])]
            if not len(m):
                continue
            a = float(m.a_us.iloc[0])
            R = data_rate(fam, cls)
            R = R[(R.t_us >= 1) & (R.t_us <= 1e4)]
            P.line(R.t_us.tolist(), clip(np.exp(-a * R.rate), 0, 1), CC[cls], 2.5, dash='8 6', markers=False,
                   tip=f'model, {CN[cls]}: exp(−{a:.3f} µs × rate)')
            g = s[(s.fam == fam) & (s.cls == cls) & (s.amp_mV == REF_AMP[fam]) & (s.t_us >= 1)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.good10, 0, 1), CC[cls], 3.5, markers=True, r=4,
                   tips=[f'injection, {CN[cls]}, {fmt_t(t)}: {v:.0%}' for t, v in zip(g.t_us, g.good10)])
            h = Hd[(Hd.fam == fam) & (Hd.cls == cls) & (Hd.amp_mV == REF_AMP[fam])]
            if len(h) and np.isfinite(h.t_good90_model_us.iloc[0]):
                P.vline(float(h.t_good90_model_us.iloc[0]), CC[cls], '3 5', 1.5, tip=f'model reaches 90 %, {CN[cls]}')
        P.xticks(tticks(1, 1e4)).yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
        panels.append(P.svg('m' + fam))
    body = (sd.title('Beyond the recorded head block: the accurate fraction follows the pulse rate',
                     'Injection (points) and exp(−a × measured pulse rate) fitted to it (dashed), carried to 10 ms with the '
                     'measured rate; vertical: where the model reaches 90 %')
            + sd.row(*panels, gap=24)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'), ('injection', sd.MUT, 'line'),
                         ('model', sd.MUT, 'dash'), ('≥ 90 %', sd.GREEN, 'box')]))
    notes = ('<p>If a pulse is spoiled only by a neighbour arriving at random, the fraction left intact is exp(−a R) with R the '
             'pulse rate per channel and a an effective width (µs). a is fitted on the injection bins after the tail itself has '
             'stopped mattering (walls ≥ 10 µs, plastics and liquids ≥ 5 µs), with R the measured rate of pulses above 4 mV in '
             'the same channels and bunches; fit residuals 0.02–0.05 for ~1 MIP. The rate is measured to 10 ms, so the model '
             'reaches times the injection cannot. Fitted a (~1 MIP): '
             + '; '.join(f'{FAMN[r.fam]} {r.cls} {r.a_us:.3f} µs' for r in OM[OM.apply(lambda r: r.amp_mV == REF_AMP[r.fam], axis=1)].itertuples())
             + '.</p>')
    D.slide('model', body, notes, short='pile-up model')


def gain_slide(D, Q):
    panels = []
    for pop, ttl in (('plastic|wall', 'plastic behind a wall hit'), ('wall bar', 'wall bars (both ends)')):
        g = Q[(Q['pop'] == pop) & (Q.cls == 'ded') & (Q.t_us > 1) & (Q.t_us < 2e4)].sort_values('t_us')
        P = sd.Plot(810, 560, x=(1, 1e4, 'log'), y=(0.2, 1.4), xlabel='time since the flash',
                    ylabel='quantile / its 1–10 ms value', title=f'{ttl}, dedicated')
        P.band([1, 1e4], [0.95, 0.95], [1.05, 1.05], sd.GREEN, alpha=0.10)
        for qn, c in zip(('q50', 'q75', 'q90', 'q95', 'q99'), ('#2b4c7e', '#3f7fb5', '#4fa3a5', '#c9b13b', sd.RED)):
            P.line(g.t_us.tolist(), clip(g[qn], 0.2, 1.4), c, 4 if qn in ('q50', 'q99') else 2.2, markers=qn == 'q99', r=4,
                   tips=[f'{qn} at {fmt_t(t)}: {v:.2f}' for t, v in zip(g.t_us, g[qn])] if qn == 'q99' else None,
                   tip=qn)
        P.hline(1, sd.INK, None, 1.2)
        P.xticks(tticks(1, 1e4)).yticks([(v, f'{v:.1f}') for v in (0.2, 0.6, 1.0, 1.4)])
        panels.append(P.svg(pop))
    body = (sd.title('Gain or particle mix? A gain change moves every quantile together',
                     'Quantiles of the amplitude spectrum over their 1–10 ms values. Plastics: all drop together before ~10 µs '
                     '(a gain sag); later q50 and q99 are back, q75–q95 stay low to ~0.3 ms (the mix)')
            + sd.row(*panels, gap=44)
            + sd.legend([('median (q50)', '#2b4c7e', 'line'), ('q75', '#3f7fb5', 'line'), ('q90', '#4fa3a5', 'line'),
                         ('q95', '#c9b13b', 'line'), ('q99 = the gain estimate', sd.RED, 'line'), ('±5 %', sd.GREEN, 'box')]))
    notes = ('<p>The earlier note read the 95 % edge of the plastic spectrum behind a wall hit as an upper bound on the gain sag '
             '(0.71 / 0.87 / 0.93 at 10 / 100 / 300 µs). The quantiles separate the two: from ~20 µs the median and the 99 % '
             'point are back within 2–4 % while the 75–95 % points are still 10–20 % low; a gain sag cannot do that, a change '
             'in the share of 2–5 MeVee pulses can. Before ~10 µs all five drop by the same factor (0.37 at 3 µs, ~0.6 at 5 µs, '
             '~0.8 at 8 µs), which is what a gain sag looks like. A PMT whose base is drained by the ~0.5 V of pile-up current '
             'in the first µs behaves this way.</p>'
             '<p>Walls: the top of the spectrum is within 1.07–1.00 from 3 µs on (a few % above 1 early: piled pulses are '
             'bigger), and dips 3 % at 0.2–1 ms. No gain sag the data can resolve beyond that. Liquids: their quantiles move by '
             '×1.4–2.9 at 2–100 µs, so the shape changes far too much to read a gain from them.</p>'
             '<p>Caveat: this assumes the top of the spectrum is made by the same particles at all times; the simulation says the '
             'population is capture γ from ~2 µs, but it does not reproduce the hardest part of the data spectrum.</p>'
             '<p>Code: <code>gain_ref.py</code>.</p>')
    D.slide('gain', body, notes, short='gain vs mix')


def bias_slide(D, B):
    panels = []
    for fam in ('PSS', 'WAL', 'LIQ'):
        g0 = B[B.fam == fam]
        P = sd.Plot(540, 560, x=(1, 250, 'log'), y=(-60, 20), xlabel='time since the flash',
                    ylabel='amplitude bias [%]' if fam == 'PSS' else '', title=f'{FAMN[fam]}, ~1 MIP', margin=(24, 24, 92, 90))
        P.band([1, 250], [-3, -3], [3, 3], sd.GREEN, alpha=0.10)
        for cls in ('ded', 'par'):
            g = g0[(g0.cls == cls) & (g0.t_us >= 1)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(100 * g.reco_bias, -60, 20), CC[cls], 2.5, markers=True, r=3,
                   tips=[f'reconstruction, {CN[cls]}, {fmt_t(t)}: {100 * v:+.1f} %' for t, v in zip(g.t_us, g.reco_bias)])
            if g.gain.notna().any():
                P.line(g.t_us.tolist(), clip(100 * (g.gain - 1), -60, 20), CC[cls], 2.5, dash='4 5', markers=False,
                       tip=f'detector gain, {CN[cls]}')
                P.line(g.t_us.tolist(), clip(100 * g.total_bias, -60, 20), CC[cls], 6, markers=False,
                       tip=f'total energy bias, {CN[cls]}')
        P.xticks(tticks(1, 250)).yticks([(v, f'{v:+d} %') for v in (-60, -40, -20, 0, 20)])
        panels.append(P.svg(fam))
    body = (sd.title('Energy bias = reconstruction × gain: the plastics are low by tens of % before ~10 µs, ≤ 4 % after ~20 µs',
                     'Median bias of reconstructed ~1-MIP pulses (thin), detector gain from the data (dashed), their product (thick)')
            + sd.row(*panels, gap=24)
            + sd.legend([('dedicated', CC['ded'], 'line'), ('parasitic', CC['par'], 'line'), ('reconstruction', sd.MUT, 'line'),
                         ('gain', sd.MUT, 'dash'), ('total', sd.INK, 'line'), ('±3 %', sd.GREEN, 'box')]))
    notes = ('<p>Reconstruction bias: median A_fit/A of the injected pulses over the same before the flash (all pulses, '
             'pile-up included, multi-pulse fit). Gain: q99 of the spectrum (previous slide), plastics behind a wall hit and '
             'wall bars; the liquids have none. The total is the bias of a reconstructed energy if nothing is corrected. '
             'Both parts are smooth in time and measured, so a correction g(t) can be applied; what remains is its '
             'uncertainty, ~the spread between the plastic populations (coincident vs singles, ~0.1 at 5–8 µs, &lt; 0.03 after '
             '15 µs).</p>')
    D.slide('bias', body, notes, short='energy bias')


def res_slide(D, S, ES):
    s = S[(S.method == METHOD) & (S.tbin > 0) & (S.cls == 'ded')]
    s = s[s.t_us >= s.fam.map(T0)]
    panels = []
    for fam in ('PSS', 'WAL', 'LIQ'):
        g0 = s[s.fam == fam]
        amps = sorted(g0.amp_mV.unique())
        P = sd.Plot(540, 330, x=(0.3, 250, 'log'), y=(0.3, 100, 'log'), xlabel='',
                    ylabel='σ68(A)/A [%]' if fam == 'PSS' else '', title=FAMN[fam], margin=(16, 24, 40, 90))
        Q = sd.Plot(540, 330, x=(0.3, 250, 'log'), y=(0.03, 30, 'log'), xlabel='time since the flash',
                    ylabel='σ68(t) [ns]' if fam == 'PSS' else '', margin=(16, 24, 92, 90))
        for A, c in zip(amps, amp_colors(len(amps))):
            g = g0[g0.amp_mV == A].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(100 * g.r_s68, 0.3, 100), c, 2.5, markers=False,
                   tip=f'{A:g} mV ≈ {to_energy(fam, A, ES)}')
            Q.line(g.t_us.tolist(), clip(g.d_s68, 0.03, 30), c, 2.5, markers=False, tip=f'{A:g} mV')
        P.xticks(tticks(0.3, 250)).yticks([(v, f'{v:g}') for v in (1, 10, 100)])
        Q.xticks(tticks(0.3, 250)).yticks([(v, f'{v:g}') for v in (0.1, 1, 10)])
        panels.append(sd.col(P.svg(fam + 'a'), Q.svg(fam + 't'), gap=0))
    body = (sd.title('Resolution: pile-up widens amplitude and time well past the tail\'s own recovery',
                     'Half the 16–84 % range of A_fit/A and of t_fit − t₀, dedicated, every found pulse; colour = amplitude (small → large)')
            + sd.row(*panels, gap=24))
    notes = ('<p>Before the flash (control) the amplitude resolution is 0.4 % (plastics, 200 mV), 1 % (liquids), 2 % (walls, 30 mV) '
             'and the time resolution 0.07 ns (plastics, liquids) and 0.4 ns (walls). The tail of the distributions is pile-up; '
             'σ68 grows with the occupancy and falls back as the rate falls. In energy units: <code>performance.csv</code>, and '
             'the next slide.</p>')
    D.slide('resolution', body, notes, short='resolution')


def table_slide(D, P, ES):
    p = P[(P.cls == 'ded') & P.apply(lambda r: r.amp_mV == REF_AMP[r.fam], axis=1)]
    rows, tips = [], []
    for fam, ts in (('PSS', (2, 5, 10, 20, 50, 100, 200)), ('WAL', (5, 10, 20, 50)), ('LIQ', (2, 5, 20, 100))):
        for t in ts:
            r = p[(p.fam == fam) & (p.t_us == t)]
            if not len(r) or not np.isfinite(r.good10.iloc[0]):
                continue
            r = r.iloc[0]
            unit = 'MeVee' if fam == 'PSS' else ('MeV' if fam == 'WAL' else '')
            sE = f'{r.sigma_E_MeV:.2f} {unit}' if fam != 'LIQ' else f'{100 * r.r_s68:.1f} %'
            rows.append([FAMN[fam], fmt_t(t), f'{r.good10:.0%}', f'{r.f_piled:.0%}', f'{100 * r.reco_bias:+.1f} %',
                         f'{r.gain:.2f}' if np.isfinite(r.gain) else '—',
                         f'{100 * r.total_bias:+.0f} %' if np.isfinite(r.total_bias) else '—', sE, f'{r.d_s68_ns:.2f} ns'])
            tips.append(f'{r.amp_mV:g} mV ≈ {to_energy(fam, r.amp_mV, ES)}; clean pulses: σ68(A)/A {100 * r.r_s68_clean:.1f} %, '
                        f'σ68(t) {r.d_s68_clean_ns:.2f} ns')
    body = (sd.title('In numbers: a ~1-MIP pulse against time since the flash (dedicated)',
                     f'Plastic 200 mV ≈ {200 / ES["pss_mean"]:.1f} MeVee; wall end 30 mV ≈ {30 / ES["wal_mean"]:.1f} MIP '
                     f'({30 / ES["wal_mean"] * WALL_MIP_MEV:.2f} MeV); liquid 100 mV (no energy scale)')
            + sd.table(['', 'time', 'within 10 %', 'piled', 'reco bias', 'gain', 'energy bias', 'σ68(E)', 'σ68(t)'],
                       rows, size=18, tips=tips))
    notes = (f'<p>Energy scale: plastics {ES["pss_mean"]:.0f} mV/MeVee on average ({ES["pss_lo"]:.0f}–{ES["pss_hi"]:.0f} over '
             'the 8 bars), from the muon calibration of the calorimetry study (6 bars) and the cosmic muons at 30–32 ms in this '
             f'same data (B bars; in-situ / calibration = {ES["pss_insitu_over_cal"]:.2f} on the others). Wall ends: '
             f'{ES["wal_mean"]:.0f} mV per MIP ({ES["wal_lo"]:.0f}–{ES["wal_hi"]:.0f}), a MIP being {WALL_MIP_MEV} MeV in the '
             '3 mm bar. Hover a row for the clean-pulse resolution (the tail alone).</p>'
             '<p>Every amplitude and the parasitic class are in <code>performance.csv</code> (out_dir("reco")).</p>')
    D.slide('numbers', body, notes, short='numbers')


def limits(D):
    items = [
        'The gain estimate assumes the top of the spectrum is made by the same particles at every time. Geant4 supports a '
        'stable capture-γ population from ~2 µs, but does not reproduce the hardest part of the data spectrum.',
        'Before ~2 µs (fast-neutron recoils) gain and mix cannot be separated on the plastics and liquids.',
        'Injection works only inside the unsuppressed head block (walls 40–60 µs, plastics ~250 µs, liquids 100–200 µs). '
        'Later times (the walls\' headline) come from the pile-up model: exp(−a × measured rate), fitted where both exist.',
        'Pulses are injected with the family template; per-channel shape differences cancel against the pre-flash control.',
        'Pile-up scales with the pulse rate: these numbers hold for this apparatus at 4 and 8.5×10¹² protons per pulse.',
    ]
    body = (sd.kicker('What this does not rule out', dark=True)
            + ''.join(sd.p(f'· {t}', 26, sd.DINK) for t in items))
    D.slide('limits', body, '', dark=True, short='limits')


def main() -> int:
    o = out_dir('reco')
    Hd = pd.read_csv(o / 'headline.csv')
    P = pd.read_csv(o / 'performance.csv')
    S = pd.read_csv(out_dir('inject') / 'inject_summary.csv')
    Q = pd.read_csv(out_dir('gain') / 'gain_quantiles.csv')
    B = pd.read_csv(o / 'figures' / 'bias_vs_t.csv')
    R = pd.read_csv(out_dir('g4mix') / 'g4_rates.csv')
    ES = energy_scale()
    z = np.load(out_dir('inject') / 'inject.npz')
    NI = dict(n_inj=int(z['N'].sum()), n_bunch=int(z['n_bunch'].sum()), n_runs=int(z['n_files']))
    D = sd.Deck('Scintillator pulses after the γ-flash',
                'From when after the n_TOF γ-flash a scintillator pulse is reconstructed accurately, and how well')
    cover(D, Hd, ES)
    method(D, NI)
    mix(D, R)
    good_slide(D, S, ES)
    pile_clean(D, S)
    model_slide(D, pd.read_csv(o / 'occupancy_model.csv'), S, Hd)
    gain_slide(D, Q)
    bias_slide(D, B)
    res_slide(D, S, ES)
    table_slide(D, P, ES)
    limits(D)
    path = D.write(out_dir('deck') / 'scint-flash-reconstruction.html', note_meta=dict(
        title='Scintillator pulses after the γ-flash: reconstruction, bias, resolution',
        summary='Pulses injected into the real n_TOF flash tails, the detector gain from the measured spectra, and Geant4: '
                'from when an X17 scintillator pulse is reconstructed accurately, and its bias and resolution in amplitude, '
                'energy and time against time since the flash.',
        tags='X17, n_TOF, scintillators', date=dt.date.today().isoformat()))
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
