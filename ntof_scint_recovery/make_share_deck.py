#!/usr/bin/env python3
"""
make_share_deck.py -- the short, picture-first version of the flash-recovery note, for collaborators without
the background: when do the X17 scintillators recover from the γ-flash, as a time and as a neutron energy.

    .venv/bin/python -m ntof_scint_recovery.make_share_deck [--plan <png>]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        <out>/deck/scint-flash-recovery-summary.html --slug scint-flash-recovery-summary --force --deploy

Same inputs as make_early_deck.py (run linearity.py and mip_eff.py first).  The set-up drawing is the MPGD26
plan view (mpgd26/make_ntof_plan.py --bare, from the Geant4 geometry), embedded in the page; it is regenerated
into out_dir('deck') when no --plan is given.  The detailed note is scint-flash-reconstruction.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import REPO, out_dir
from ntof_scint_recovery.make_reco_deck import clip, fmt_t, sd

#: EAR2 flight path (neutron_energy_vs_flight_time.py); times are since the flash AT the detector, so the
#: neutron time of flight is t + L/c
FLIGHT_M = 19.5
MN_MEV = 939.565
C_M_PER_US = 299.792458
COL = {'ded': sd.ORANGE, 'par': sd.BLUE}
NAME = {'ded': 'full-intensity beam pulse', 'par': 'half-intensity beam pulse'}
FAMS = {'WAL': 'SiPM wall', 'PSS': 'plastic scintillator', 'LIQ': 'liquid scintillator'}
#: the colours of the set-up drawing
FCOL = {'WAL': '#c99318', 'PSS': '#8a6cc0', 'LIQ': '#d0703f'}
TICKS = ((0.3, '0.3 µs'), (1, '1 µs'), (3, '3 µs'), (10, '10 µs'), (30, '30 µs'))
TXT = 27


def e_n(t_us):
    """Neutron kinetic energy (MeV) arriving t_us after the flash."""
    beta = FLIGHT_M / (C_M_PER_US * (t_us + FLIGHT_M / C_M_PER_US))
    return MN_MEV * (1 / math.sqrt(1 - beta ** 2) - 1)


def fmt_e(t_us):
    e = e_n(t_us)
    return f'{e:.1f} MeV' if e >= 1 else f'{e * 1e3:.0f} keV'


def first(MF, fam, cls, col):
    return float(MF[(MF.fam == fam) & (MF.cls == cls)][col].iloc[0])


def times(MF):
    """Recovery times shown everywhere: when a MIP is seen and the response is shown linear (usable_from), and
    when the plastic signal size is measured."""
    return dict(WAL=first(MF, 'WAL', 'ded', 'usable_from_us'), WAL_par=first(MF, 'WAL', 'par', 'usable_from_us'),
                PSS=first(MF, 'PSS', 'ded', 'usable_from_us'), PSS_par=first(MF, 'PSS', 'par', 'usable_from_us'),
                PSS_gain=first(MF, 'PSS', 'ded', 'gain_from_us'))


def say(text, size=TXT, color=sd.INK):
    return sd.p(text, size, color)


def card(name, col, lines, w=520):
    big = 'font-size:84px;font-weight:600;line-height:1.05'
    inner = f'<div style="font-size:30px;color:{sd.DMUT}">{name}</div>'
    for kind, s in lines:
        inner += (f'<div style="{big};color:{col}">{s}</div>' if kind == 'big' else
                  f'<div style="font-size:24px;color:{sd.DMUT};margin-top:10px">{s}</div>')
    return sd.card(inner, w=w, bg='#1f2633')


def cover(D, R):
    cards = [
        card('SiPM walls', sd.DBLUE, [('big', f'{R["WAL"]:.0f} µs'), ('big', fmt_e(R['WAL'])),
                                      ('s', 'after the flash · neutron energy'),
                                      ('s', f'half-intensity pulses: {R["WAL_par"]:.0f} µs · {fmt_e(R["WAL_par"])}')]),
        card('plastic scintillators', '#c9a4f0', [('big', f'{R["PSS"]:.1f} µs'), ('big', fmt_e(R['PSS'])),
                                                  ('s', 'after the flash · neutron energy'),
                                                  ('s', f'signal size known from {R["PSS_gain"]:.0f} µs · {fmt_e(R["PSS_gain"])}')]),
        card('liquid scintillators', sd.DGREEN, [('big', 'not'), ('big', 'clear'),
                                                 ('s', 'the liquid scintillators did not work as intended')]),
    ]
    body = (sd.kicker('n_TOF EAR2 · X17 · 2026 data')
            + sd.title(f'<span style="color:{sd.DINK}">X17 at n_TOF: when the scintillators recover from the γ-flash</span>',
                       f'<span style="color:{sd.DMUT}">Every n_TOF beam pulse starts with an intense flash of γ-rays that '
                       'blinds the scintillators for a moment. How soon can they be trusted again, and which neutron '
                       'energies does that leave us?</span>')
            + sd.row(*cards, gap=48))
    D.slide('cover', body, '', dark=True, short='cover')


def plan_png(path: Path | None) -> Path:
    if path:
        return path
    out = out_dir('deck') / 'ntof_plan_bare.png'
    if not out.exists():
        subprocess.run([sys.executable, 'make_ntof_plan.py', '--bare', '--no-pdf', '--out', str(out)],
                       cwd=REPO / 'mpgd26', check=True)
    return out


def setup(D, png: Path):
    b64 = base64.b64encode(png.read_bytes()).decode()
    img = (f'<img src="data:image/png;base64,{b64}" alt="The X17 set-up seen from above, with one simulated pair" '
           f'style="height:840px;display:block">')
    side = sd.col(
        say('The neutron beam comes out of the page and hits a small cell of ³He gas in the middle.'),
        say('A neutron captured there can produce an electron–positron pair (red lines). Each one crosses, in order:'),
        sd.legend([('a Micromegas tracker (its direction)', '#5fb8d6', 'box'), ('a wall of SiPM-read bars', FCOL['WAL'], 'box'),
                   ('two plastic scintillator bars', FCOL['PSS'], 'box'), ('a liquid scintillator', FCOL['LIQ'], 'box')], 25),
        say('The three scintillators tell us when the particle passed and how much energy it left.', TXT, sd.MUT),
        gap=28, w=640)
    body = sd.title('The set-up, seen from above') + sd.row(img, side, gap=56, align='center')
    D.slide('setup', body, '', short='set-up')


def waveforms(D, X, R):
    panels = []
    for fam, ymax in (('WAL', 1000), ('PSS', 2000), ('LIQ', 2000)):
        t, y = X[f'{fam}_t'], X[f'{fam}_orig_max']
        P = sd.Plot(540, 560, x=(-0.5, 10.5), y=(-50, ymax), xlabel='time after the flash [µs]',
                    ylabel='signal [mV]' if fam == 'WAL' else '', title=FAMS[fam], margin=(24, 24, 92, 96))
        if fam == 'WAL':
            # the divert gate blanks the SiPM signal around the flash; it opens at ~0.6 µs
            P.band([0, 0.6], [-50, -50], [ymax, ymax], sd.GREY, alpha=0.35)
            P.text(0.3, ymax * 0.55, 'switched off', 20, sd.INK, 'middle', 600)
        if fam in R:
            P.band([-0.5 if fam != 'WAL' else 0.6, R[fam]], [-50, -50], [ymax, ymax], sd.RED, alpha=0.07)
            P.vline(R[fam], sd.GREEN, None, 4)
            P.text(R[fam] + 0.3, ymax * 0.9, f'usable from {fmt_t(R[fam])}', 26, sd.GREEN, 'start', 600)
            P.text(R[fam] + 0.3, ymax * 0.8, f'neutrons below {fmt_e(R[fam])}', 22, sd.GREEN, 'start')
        else:
            P.text(6.5, ymax * 0.9, 'not clear', 30, sd.GOLD, 'middle', 600)
        P.line(t.tolist(), clip(y, -50, ymax), FCOL[fam], 2, markers=False)
        P.xticks([(v, f'{v:g}') for v in (0, 2, 4, 6, 8, 10)]).yticks([(v, f'{v:g}') for v in (0, ymax / 2, ymax)])
        panels.append(P.svg('w' + fam))
    body = (sd.title('One beam pulse: what the scintillators record in the first 10 µs')
            + sd.row(*panels, gap=24)
            + say('Time 0 is when the flash reaches the detectors. It drives every scintillator to its limit; the wall is switched '
                  'off on purpose during the flash and switched back on at ~0.6 µs, when the leftover flash charge appears. '
                  'What follows is the tail of the flash plus a rain of real particles from the beam. The green lines mark '
                  'when each one can be read again; the rest of these slides show how we decided that.', 26, sd.MUT))
    D.slide('waveforms', body, '', short='waveforms')


def _pulse(xs, x0, a, w):
    return [a * ((x - x0) / w) * math.exp(1 - (x - x0) / w) if x >= x0 else 0.0 for x in xs]


def idea(D):
    """Cartoon: a full- and a half-intensity beam pulse, flash and particles; what we compare."""
    W, H = 780, 330
    panels = []
    for cls, amp, n in (('ded', 1.0, 10), ('par', 0.5, 5)):
        xs = np.linspace(0, 10, 600)
        flash = [amp * 260 * math.exp(-x / 0.9) if x > 0.15 else (amp * 260 * x / 0.15) for x in xs]
        rng = np.random.default_rng(3 if cls == 'ded' else 5)
        t0s = np.sort(rng.uniform(1.0, 9.6, n))
        sig = np.zeros(len(xs))
        for t0 in t0s:
            sig += np.array(_pulse(xs, t0, rng.uniform(35, 70), 0.06))
        y = np.array(flash) + sig
        x0, y0, sx, sy = 60, H - 60, (W - 90) / 10, 0.9
        pts = ' '.join(f'{x0 + a * sx:.1f},{y0 - b * sy:.1f}' for a, b in zip(xs, y))
        g = [f'<polyline points="{pts}" fill="none" stroke="{COL[cls]}" stroke-width="3"/>',
             sd.line(x0, y0, W - 20, y0, sd.MUT, 2), sd.line(x0, y0, x0, 30, sd.MUT, 2),
             sd.T(W - 20, y0 + 34, 'time →', 22, sd.MUT, 'end'),
             sd.T(x0 + 0.6 * sx, y0 - amp * 260 * sy - 10, 'flash', 22, COL[cls], 'start', 600)]
        for t0 in t0s:
            g.append(sd.T(x0 + t0 * sx + 4, y0 - 80, '•', 26, COL[cls], 'middle'))
        panels.append(sd.col(sd.p(f'<b style="color:{COL[cls]}">{NAME[cls]}</b> · '
                                  + ('8.5×10¹² protons' if cls == 'ded' else '~4×10¹² protons'), 26),
                             sd.svg(W, H, ''.join(g), NAME[cls]), gap=4))
    body = (sd.title('How we test it: two kinds of beam pulse',
                     'n_TOF alternates full- and half-intensity proton pulses on the same target')
            + sd.row(*panels, gap=48)
            + sd.row(sd.callout('Half the protons give <b>half the flash</b> and <b>half as many particles</b>, but '
                                '<b>the same kinds of particles</b> at each moment after the flash.', sd.INK, TXT),
                     sd.callout('So at any given time, a detector that has recovered sees the same <b>mix of signal '
                                'sizes</b> after both pulses. If the bigger flash still disturbs it, the two mixes differ.',
                                sd.GREEN, TXT), gap=40))
    D.slide('idea', body, '', short='the idea')


def examples(D, E):
    panels = []
    labels = {('PSS', 0): 'plastic, 0.6 µs after the flash', ('WAL', 0): 'SiPM wall, 1.8 µs',
              ('WAL', 1): 'SiPM wall, 3.5 µs'}
    seen = {}
    for (fam, t), g in E.groupby(['fam', 't_us'], sort=False):
        i = seen.get(fam, 0)
        seen[fam] = i + 1
        ok = g.max_abs.iloc[0] <= 0.10
        ymax = float(g.frac.max()) * 1.25
        P = sd.Plot(540, 540, x=(10, 1000, 'log'), y=(0, ymax), xlabel='signal size [mV]',
                    ylabel='how often' if not panels else '', title=labels[(fam, i)], margin=(24, 24, 92, 80))
        for cls in ('ded', 'par'):
            d = g[g.cls == cls].sort_values('a_mV')
            P.line(d.a_mV.tolist(), d.frac.tolist(), COL[cls], 3, markers=True, r=3.5)
        P.text(900, ymax * 0.88, '✓ same shape' if ok else '✗ different shape', 30, sd.GREEN if ok else sd.RED, 'end', 600)
        P.xticks([(v, f'{v:g}') for v in (10, 30, 100, 300, 1000)]).yticks([])
        panels.append(P.svg(f'ex{fam}{i}'))
    body = (sd.title('What we compare: how often each signal size occurs')
            + sd.row(*panels, gap=24)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'] + ' (sizes rescaled to match)', COL['par'], 'line')], 24)
            + say('Each curve counts the real signals recorded in one short time window. The half-intensity signals are '
                  'stretched in size to line up with the full-intensity ones, since a detector may give smaller signals after '
                  'a bigger flash. On the plastic, 0.6 µs after the flash, the two shapes already agree. The wall still '
                  'disagrees at 1.8 µs and agrees by 3.5 µs.', 26, sd.MUT))
    D.slide('test', body, '', short='signal sizes')


def mismatch(D, S, R):
    panels = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        g = S[(S.fam == fam) & (S.test == 'flash') & (S.n_ch >= 3) & (S.t_us < 60)].sort_values('t_us')
        P = sd.Plot(540, 540, x=(0.3, 60, 'log'), y=(0, 0.5), xlabel='time after the flash',
                    ylabel='shape mismatch' if fam == 'WAL' else '', title=FAMS[fam], margin=(24, 24, 92, 96))
        P.band([0.3, 60], [0, 0], [0.1, 0.1], sd.GREEN, alpha=0.12)
        P.line(g.t_us.tolist(), clip(g.max_abs, 0, 0.5), FCOL[fam], 3.5, r=5,
               tips=[f'{fmt_t(t)}: {v:.0%}' for t, v in zip(g.t_us, g.max_abs)])
        if fam in ('WAL', 'PSS'):
            t = first(S.attrs['MF'], fam, 'ded', 'linear_from_us')
            P.vline(t, sd.GREEN, None, 3)
            x, y = P.X(t), P.Y(0.32)
            P.raw(sd.arrow(x + 110, y, x + 8, y, sd.GREEN, 4, 18))
            P.raw(sd.T(x + 118, y + 9, f'same from {fmt_t(t)}', 24, sd.GREEN, 'start', 600))
        else:
            P.text(3, 0.42, 'never quite matches', 24, sd.GOLD, 'start', 600)
        P.xticks(list(TICKS)).yticks([(v, f'{v:.0%}') for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)])
        panels.append(P.svg('mm' + fam))
    body = (sd.title('Shape mismatch between full- and half-intensity pulses, against time')
            + sd.row(*panels, gap=24)
            + say('For each moment after the flash we stretch the half-intensity curve to fit the full-intensity one as well '
                  'as possible, and record how far apart they still are: 0 % means identical shapes. Green band: within 10 %, '
                  'which we count as the same. The liquids stay 10–20 % apart even long after the flash, so this test cannot '
                  'clear them.', 26, sd.MUT))
    D.slide('mismatch', body, '', short='shape mismatch')


def size(D, M, R):
    panels = []
    for fam in ('WAL', 'PSS'):
        x0 = 2.0 if fam == 'WAL' else 0.3
        P = sd.Plot(800, 560, x=(x0, 40, 'log'), y=(0, 1.3), xlabel='time after the flash',
                    ylabel='signal size (1 = normal)' if fam == 'WAL' else '', title=FAMS[fam], margin=(24, 24, 92, 104))
        P.hline(1, sd.INK, None, 1.5)
        if fam == 'PSS':
            P.band([R['PSS'], R['PSS_gain']], [0, 0], [1.3, 1.3], sd.GOLD, alpha=0.15)
            P.text(math.sqrt(R['PSS'] * R['PSS_gain']), 1.2, 'seen, size uncertain', 21, sd.GOLD, 'middle', 600)
            P.vline(R['PSS_gain'], sd.GREEN, '8 6', 3)
            P.text(R['PSS_gain'] * 1.08, 0.3, f'size known from {fmt_t(R["PSS_gain"])}', 22, sd.GREEN, 'start', 600)
        P.vline(R[fam], sd.GREEN, None, 4)
        P.text(R[fam] * 1.08, 0.04 if fam == 'PSS' else 0.12, f'read from {fmt_t(R[fam])}', 22, sd.GREEN, 'start', 600)
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= x0) & (M.t_us <= 40)].sort_values('t_us')
            P.band(g.t_us.tolist(), clip(g.gain_lo, 0, 1.3), clip(g.gain_hi, 0, 1.3), COL[cls], alpha=0.25)
            m = g.gain_measured
            P.line(g.t_us[m].tolist(), clip(g.gain[m], 0, 1.3), COL[cls], 4, r=4)
        P.xticks([v for v in TICKS if v[0] >= x0]).yticks([(v, f'{v:g}') for v in (0, 0.5, 1)])
        panels.append(P.svg('s' + fam))
    body = (sd.title('Signal size: the plastics give ~3× smaller signals at first, back to normal by ~10 µs')
            + sd.row(*panels, gap=40)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'], COL['par'], 'line'),
                         ('possible range', sd.MUT, 'box')], 24)
            + say(f'The flash leaves the plastic photomultipliers giving smaller signals for a while. We can see particles '
                  f'from {fmt_t(R["PSS"])}, but only from {fmt_t(R["PSS_gain"])} can we measure by how much their signals '
                  f'are reduced, so energies measured before then are uncertain. The SiPM wall shows no such loss.', 26, sd.MUT))
    D.slide('size', body, '', short='signal size')


def seen(D, M, R):
    panels = []
    for fam in ('WAL', 'PSS'):
        x0 = 0.6 if fam == 'WAL' else 0.3
        P = sd.Plot(800, 540, x=(x0, 40, 'log'), y=(0, 1), xlabel='time after the flash',
                    ylabel='particles seen' if fam == 'WAL' else '', title=FAMS[fam], margin=(24, 24, 92, 104))
        P.band([x0, R[fam]], [0, 0], [1, 1], sd.GREY, alpha=0.18)
        P.hline(0.9, sd.INK, '3 5', 1.5, label='90 %')
        P.vline(R[fam], sd.GREEN, None, 4)
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= x0) & (M.t_us <= 40)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.found, 0, 1), COL[cls], 4, r=4,
                   tips=[f'{fmt_t(a)}: {v:.0%}' for a, v in zip(g.t_us, g.found)])
        P.xticks([v for v in TICKS if v[0] >= x0]).yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
        panels.append(P.svg('e' + fam))
    body = (sd.title('How often a particle crossing the scintillator is seen')
            + sd.row(*panels, gap=40)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'], COL['par'], 'line'),
                         ('not recovered yet', sd.GREY, 'box')], 24)
            + say('The particles we care about, the X17 electron and positron, are fast and leave a characteristic amount of '
                  'energy, like the cosmic muons we use as a reference (a "MIP", minimum-ionizing particle). Since that amount '
                  'varies from particle to particle, we take the real spread measured with cosmic muons and ask what fraction '
                  'would still be picked out of the busy signal after the flash.', 26, sd.MUT))
    D.slide('seen', body, '', short='particles seen')


def precision(D, M, R):
    panels = []
    for fam, col, ttl, yr, sc in (('WAL', 'mip_sig_E', 'SiPM wall: energy', (0, 0.8), 'lin'),
                                  ('PSS', 'mip_sig_E', 'plastic: energy', (0, 0.8), 'lin'),
                                  ('WAL', 'mip_sig_t_ns', 'SiPM wall: time', (0.05, 20), 'log'),
                                  ('PSS', 'mip_sig_t_ns', 'plastic: time', (0.05, 20), 'log')):
        x0 = 2.0 if fam == 'WAL' else 0.3
        P = sd.Plot(420, 540, x=(x0, 40, 'log'), y=(*yr, sc), xlabel='time after the flash',
                    ylabel=('typical energy error' if col == 'mip_sig_E' else 'typical time error [ns]') if fam == 'WAL' else '',
                    title=ttl, margin=(24, 18, 92, 96))
        P.band([x0, R[fam]], [yr[0]] * 2, [yr[1]] * 2, sd.GREY, alpha=0.18)
        if fam == 'PSS' and col == 'mip_sig_E':
            P.band([R['PSS'], R['PSS_gain']], [yr[0]] * 2, [yr[1]] * 2, sd.GOLD, alpha=0.15)
        P.vline(R[fam], sd.GREEN, None, 3)
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= max(x0, 0.4)) & (M.t_us <= 40)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g[col], *yr), COL[cls], 3.5, r=4)
        P.xticks([(v, s) for v, s in ((0.3, '0.3'), (1, '1 µs'), (3, '3'), (10, '10 µs'), (30, '30')) if v >= x0])
        P.yticks([(v, f'{v:.0%}') for v in (0, 0.2, 0.4, 0.6, 0.8)] if col == 'mip_sig_E' else
                 [(v, f'{v:g}') for v in (0.1, 1, 10)])
        panels.append(P.svg(f'p{fam}{col}'))
    body = (sd.title('How precisely a particle is measured')
            + sd.row(*panels, gap=16)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'], COL['par'], 'line'),
                         ('not recovered yet', sd.GREY, 'box'), ('size uncertain', sd.GOLD, 'box')], 24)
            + say('How far off a typical measurement of a MIP\'s energy and arrival time is, against time after the flash. '
                  'Measured by adding artificial signals of known size and time to the real recorded traces of thousands of '
                  'beam pulses and seeing how well they come back, corrected for the signal-size loss measured from the real '
                  'data. The plastics measure time to well under a nanosecond; the wall is held back by the many other '
                  'particles arriving at the same time.', 26, sd.MUT))
    D.slide('precision', body, '', short='precision')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--plan', type=Path, default=None, help='the set-up drawing (png); default: regenerate')
    a = ap.parse_args(argv)
    eo, lo = out_dir('early'), out_dir('linearity')
    M = pd.read_csv(eo / 'mip_eff.csv')
    MF = pd.read_csv(eo / 'mip_first.csv')
    X = dict(np.load(eo / 'examples.npz'))
    S = pd.read_csv(lo / 'shape_tests.csv')
    S.attrs['MF'] = MF
    E = pd.read_csv(lo / 'shape_examples.csv')
    R = times(MF)
    D = sd.Deck('X17 at n_TOF: when the scintillators recover from the γ-flash',
                'When the X17 scintillators recover from the n_TOF γ-flash, in time and in neutron energy')
    cover(D, R)
    setup(D, plan_png(a.plan))
    waveforms(D, X, {k: R[k] for k in ('WAL', 'PSS')})
    idea(D)
    examples(D, E)
    mismatch(D, S, R)
    size(D, M, R)
    seen(D, M, R)
    precision(D, M, R)
    path = D.write(out_dir('deck') / 'scint-flash-recovery-summary.html', note_meta=dict(
        title='X17 at n_TOF: when the scintillators recover from the γ-flash',
        summary='Short, picture-first summary: when the X17 scintillators recover from the n_TOF γ-flash, as a time and '
                'as a neutron energy. Details: scint-flash-reconstruction.',
        tags='X17, n_TOF, scintillators', date=dt.date.today().isoformat()))
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
