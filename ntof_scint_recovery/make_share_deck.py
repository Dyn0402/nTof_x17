#!/usr/bin/env python3
"""
make_share_deck.py -- the short, picture-first version of the flash-recovery note, for collaborators without
the background: when do the X17 scintillators recover from the γ-flash, as a time and as a neutron energy.

    .venv/bin/python -m ntof_scint_recovery.make_share_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        <out>/deck/scint-flash-recovery-summary.html --slug scint-flash-recovery-summary --force --deploy

Same inputs as make_early_deck.py (run linearity.py and mip_eff.py first).  The detailed note is
scint-flash-reconstruction.
"""
from __future__ import annotations

import datetime as dt
import math
import sys

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.make_reco_deck import clip, fmt_t, sd

#: EAR2 flight path (neutron_energy_vs_flight_time.py); times are since the flash AT the detector, so the
#: neutron time of flight is t + L/c
FLIGHT_M = 19.5
MN_MEV = 939.565
C_M_PER_US = 299.792458
COL = {'ded': sd.ORANGE, 'par': sd.BLUE}
NAME = {'ded': 'full-intensity beam pulse', 'par': 'half-intensity beam pulse'}
FAMS = {'PSS': 'plastic scintillator', 'WAL': 'SiPM wall'}
FCOL = {'PSS': '#9a7fc4', 'WAL': '#5f8f6b'}
TICKS = ((0.3, '0.3 µs'), (1, '1 µs'), (3, '3 µs'), (10, '10 µs'), (30, '30 µs'))


def e_n(t_us):
    """Neutron kinetic energy (MeV) arriving t_us after the flash."""
    beta = FLIGHT_M / (C_M_PER_US * (t_us + FLIGHT_M / C_M_PER_US))
    return MN_MEV * (1 / math.sqrt(1 - beta ** 2) - 1)


def fmt_e(t_us):
    e = e_n(t_us)
    return f'{e:.1f} MeV' if e >= 1 else f'{e * 1e3:.0f} keV'


def first(MF, fam, cls, col):
    return float(MF[(MF.fam == fam) & (MF.cls == cls)][col].iloc[0])


def cover(D, MF):
    tw, tp = first(MF, 'WAL', 'ded', 'usable_from_us'), first(MF, 'WAL', 'par', 'usable_from_us')
    big = 'font-size:96px;font-weight:600;line-height:1.05'
    wall = sd.card(f'<div style="font-size:30px;color:{sd.DMUT}">SiPM walls</div>'
                   f'<div style="{big};color:{sd.DBLUE}">{tw:.0f} µs</div>'
                   f'<div style="{big};color:{sd.DBLUE}">{fmt_e(tw)}</div>'
                   f'<div style="font-size:26px;color:{sd.DMUT}">after the flash · neutron energy</div>'
                   f'<div style="font-size:26px;color:{sd.DMUT};margin-top:16px">half-intensity pulses: {tp:.0f} µs · {fmt_e(tp)}</div>',
                   w=760, bg='#1f2633')
    plas = sd.card(f'<div style="font-size:30px;color:{sd.DMUT}">plastic scintillators</div>'
                   f'<div style="{big};color:{sd.DRED}">not yet</div><div style="{big};color:{sd.DRED}">clear</div>'
                   f'<div style="font-size:26px;color:{sd.DMUT}">signals look normal from 0.4 µs ({fmt_e(0.4)}),<br>'
                   f'but are smaller until ~10 µs ({fmt_e(10)})</div>', w=760, bg='#1f2633')
    body = (sd.kicker('n_TOF EAR2 · X17 · 2026 data')
            + sd.title(f'<span style="color:{sd.DINK}">X17 at n_TOF: when the scintillators recover from the γ-flash</span>')
            + sd.row(wall, plas, gap=64))
    D.slide('cover', body, '', dark=True, short='cover')


def setup(D):
    W, H = 900, 720
    cx, cy, k = 450, 360, 7.0
    layers = [('Micromegas tracker', 25.0, 28.5, '#8fb3dc'), ('SiPM wall', 31.0, 32.0, FCOL['WAL']),
              ('liquid scintillator', 34.0, 37.5, '#c9b38a'), ('plastic scintillator', 39.0, 41.0, FCOL['PSS'])]
    g = []
    for ang in (0, 90, 180, 270):
        for name, r0, r1, col in layers:
            half = 20.0 * k
            if ang in (0, 180):
                sx = 1 if ang == 0 else -1
                x0, x1 = cx + sx * r0 * k, cx + sx * r1 * k
                g.append(f'<rect x="{min(x0, x1):.0f}" y="{cy - half:.0f}" width="{abs(x1 - x0):.0f}" '
                         f'height="{2 * half:.0f}" fill="{col}" rx="3"{sd.tipattr(name)}/>')
            else:
                sy = 1 if ang == 90 else -1
                y0, y1 = cy + sy * r0 * k, cy + sy * r1 * k
                g.append(f'<rect x="{cx - half:.0f}" y="{min(y0, y1):.0f}" width="{2 * half:.0f}" '
                         f'height="{abs(y1 - y0):.0f}" fill="{col}" rx="3"{sd.tipattr(name)}/>')
    # an X17 -> e+e- pair crossing two arms
    g.append(sd.arrow(cx, cy, cx + 42 * k, cy - 9 * k, sd.RED, 4, 16))
    g.append(sd.arrow(cx, cy, cx - 10 * k, cy - 42 * k, sd.RED, 4, 16))
    g.append(sd.T(cx + 30 * k, cy - 12 * k, 'e⁺', 28, sd.RED, 'middle', 600))
    g.append(sd.T(cx - 12 * k, cy - 30 * k, 'e⁻', 28, sd.RED, 'end', 600))
    g.append(f'<circle cx="{cx}" cy="{cy}" r="{3 * k:.0f}" fill="#d9d4c7" stroke="{sd.INK}" stroke-width="2"/>')
    g.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{sd.BG}" stroke="{sd.INK}" stroke-width="2.5"/>'
             f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{sd.INK}"/>')
    g.append(sd.T(cx + 28, cy + 44, '³He target', 22, sd.INK, 'start'))
    g.append(sd.T(cx + 28, cy + 70, 'beam into page', 22, sd.INK, 'start'))
    diagram = sd.svg(W, H, ''.join(g), 'top-down view of the X17 set-up')
    beam = sd.flow([dict(label='γ-flash', color=sd.GOLD, sub='t = 0'),
                    dict(label='fast neutrons', color=sd.RED, sub='~1 µs'),
                    dict(label='slow neutrons', color=sd.BLUE, sub='µs → ms')], size=24)
    side = sd.col(sd.legend([(n, c, 'box') for n, _a, _b, c in layers], 26),
                  sd.p('Every beam pulse:', 26, sd.MUT), beam, gap=40)
    body = sd.title('The set-up, seen from above') + sd.row(diagram, side, gap=56, align='center')
    D.slide('setup', body, '', short='set-up')


def waveforms(D, X, MF):
    tw = first(MF, 'WAL', 'ded', 'usable_from_us')
    panels = []
    for fam, ymax in (('WAL', 1000), ('PSS', 2000)):
        t, y = X[f'{fam}_t'], X[f'{fam}_orig_max']
        P = sd.Plot(800, 600, x=(-0.5, 10.5), y=(-50, ymax), xlabel='time after the flash [µs]',
                    ylabel='signal [mV]', title=FAMS[fam], margin=(24, 24, 92, 104))
        if fam == 'WAL':
            P.band([-0.5, tw], [-50, -50], [ymax, ymax], sd.RED, alpha=0.08)
            P.vline(tw, sd.GREEN, None, 4)
            P.text(tw + 0.3, ymax * 0.9, f'recovered: {tw:.0f} µs · {fmt_e(tw)}', 30, sd.GREEN, 'start', 600)
        else:
            P.band([0.4, 10], [-50, -50], [ymax, ymax], sd.GOLD, alpha=0.12)
            P.text(5.2, ymax * 0.9, 'not yet clear', 34, sd.GOLD, 'middle', 600)
        P.line(t.tolist(), clip(y, -50, ymax), FCOL[fam], 2, markers=False)
        P.xticks([(v, f'{v:g}') for v in (0, 2, 4, 6, 8, 10)]).yticks([(v, f'{v:g}') for v in (0, ymax / 2, ymax)])
        panels.append(P.svg('w' + fam))
    body = (sd.title('One beam pulse: what the scintillators record in the first 10 µs')
            + sd.row(*panels, gap=40))
    D.slide('waveforms', body, '', short='waveforms')


def examples(D, E):
    panels = []
    labels = {('PSS', 0): 'plastic, 0.6 µs', ('WAL', 0): 'SiPM wall, 1.8 µs', ('WAL', 1): 'SiPM wall, 3.5 µs'}
    seen = {}
    for (fam, t), g in E.groupby(['fam', 't_us'], sort=False):
        i = seen.get(fam, 0)
        seen[fam] = i + 1
        ok = g.max_abs.iloc[0] <= 0.10
        ymax = float(g.frac.max()) * 1.25
        P = sd.Plot(540, 560, x=(10, 1000, 'log'), y=(0, ymax), xlabel='signal size [mV]',
                    ylabel='how often' if not panels else '', title=labels[(fam, i)], margin=(24, 24, 92, 80))
        d, p = g[g.cls == 'ded'].sort_values('a_mV'), g[g.cls == 'par'].sort_values('a_mV')
        P.line(d.a_mV.tolist(), d.frac.tolist(), COL['ded'], 3.5, markers=False)
        P.points(p.a_mV.tolist(), p.frac.tolist(), COL['par'], r=5)
        P.text(900, ymax * 0.88, '✓ same shape' if ok else '✗ different', 30,
               sd.GREEN if ok else sd.RED, 'end', 600)
        P.xticks([(v, f'{v:g}') for v in (10, 30, 100, 300, 1000)]).yticks([])
        panels.append(P.svg(f'ex{fam}{i}'))
    body = (sd.title('The test: does a 2× bigger flash change the signals?',
                     'Same particles, twice the flash. If the detector has recovered, only the size changes, not the shape')
            + sd.row(*panels, gap=24)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'] + ', rescaled', COL['par'], 'dot')], 24))
    D.slide('test', body, '', short='the test')


def mismatch(D, S, MF):
    panels = []
    for fam in ('WAL', 'PSS'):
        g = S[(S.fam == fam) & (S.test == 'flash') & (S.n_ch >= 3) & (S.t_us < 60)].sort_values('t_us')
        P = sd.Plot(800, 580, x=(0.3, 60, 'log'), y=(0, 0.5), xlabel='time after the flash',
                    ylabel='shape mismatch', title=FAMS[fam], margin=(24, 24, 92, 104))
        P.band([0.3, 60], [0, 0], [0.1, 0.1], sd.GREEN, alpha=0.12)
        P.line(g.t_us.tolist(), clip(g.max_abs, 0, 0.5), sd.INK, 3.5, r=5)
        t = first(MF, fam, 'ded', 'linear_from_us')
        P.vline(t, sd.GREEN, None, 3)
        x, y = P.X(t), P.Y(0.3)
        P.raw(sd.arrow(x + 150, y, x + 8, y, sd.GREEN, 4, 18))
        P.raw(sd.T(x + 160, y + 9, f'same shape from {fmt_t(t)}', 26, sd.GREEN, 'start', 600))
        P.xticks(list(TICKS)).yticks([(v, f'{v:.0%}') for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)])
        panels.append(P.svg('mm' + fam))
    body = sd.title('Shape mismatch, full vs half intensity') + sd.row(*panels, gap=40)
    D.slide('mismatch', body, '', short='shape mismatch')


def size(D, M):
    panels = []
    for fam in ('WAL', 'PSS'):
        x0 = 2.0 if fam == 'WAL' else 0.3
        P = sd.Plot(800, 580, x=(x0, 40, 'log'), y=(0, 1.3), xlabel='time after the flash',
                    ylabel='signal size (1 = normal)', title=FAMS[fam], margin=(24, 24, 92, 104))
        P.hline(1, sd.INK, None, 1.5)
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= x0) & (M.t_us <= 40)].sort_values('t_us')
            P.band(g.t_us.tolist(), clip(g.gain_lo, 0, 1.3), clip(g.gain_hi, 0, 1.3), COL[cls], alpha=0.25)
            m = g.gain_measured
            P.line(g.t_us[m].tolist(), clip(g.gain[m], 0, 1.3), COL[cls], 4, r=4)
        if fam == 'PSS':
            P.text(0.7, 1.15, '?', 44, sd.GOLD, 'middle', 600)
        P.xticks([v for v in TICKS if v[0] >= x0]).yticks([(v, f'{v:g}') for v in (0, 0.5, 1)])
        panels.append(P.svg('s' + fam))
    body = (sd.title('Signal size: the plastics give ~3× smaller signals at first, back to normal by ~10 µs')
            + sd.row(*panels, gap=40)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'], COL['par'], 'line'),
                         ('range (not measured exactly)', sd.MUT, 'box')], 24))
    D.slide('size', body, '', short='signal size')


def seen(D, M, MF):
    panels = []
    for fam in ('WAL', 'PSS'):
        x0 = 0.6 if fam == 'WAL' else 0.3
        P = sd.Plot(800, 580, x=(x0, 40, 'log'), y=(0, 1), xlabel='time after the flash',
                    ylabel='particles seen' if fam == 'WAL' else '', title=FAMS[fam], margin=(24, 24, 92, 104))
        t = first(MF, fam, 'ded', 'linear_from_us')
        P.band([x0, t], [0, 0], [1, 1], sd.GREY, alpha=0.18)
        P.hline(0.9, sd.INK, '3 5', 1.5, label='90 %')
        for cls in ('ded', 'par'):
            g = M[(M.fam == fam) & (M.cls == cls) & (M.t_us >= x0) & (M.t_us <= 40)].sort_values('t_us')
            P.line(g.t_us.tolist(), clip(g.found, 0, 1), COL[cls], 4, r=4,
                   tips=[f'{fmt_t(a)}: {v:.0%}' for a, v in zip(g.t_us, g.found)])
        P.xticks([v for v in TICKS if v[0] >= x0]).yticks([(v, f'{v:.0%}') for v in (0, 0.25, 0.5, 0.75, 1)])
        panels.append(P.svg('e' + fam))
    body = (sd.title('How often a particle crossing the scintillator is seen')
            + sd.row(*panels, gap=40)
            + sd.legend([(NAME['ded'], COL['ded'], 'line'), (NAME['par'], COL['par'], 'line'),
                         ('not recovered yet', sd.GREY, 'box')], 24))
    D.slide('seen', body, '', short='efficiency')


def precision(D, M, MF):
    panels = []
    for fam, col, ttl, yr, sc in (('WAL', 'mip_sig_E', 'SiPM wall: energy', (0, 0.8), 'lin'),
                                  ('PSS', 'mip_sig_E', 'plastic: energy', (0, 0.8), 'lin'),
                                  ('WAL', 'mip_sig_t_ns', 'SiPM wall: time', (0.05, 20), 'log'),
                                  ('PSS', 'mip_sig_t_ns', 'plastic: time', (0.05, 20), 'log')):
        x0 = 2.0 if fam == 'WAL' else 0.3
        P = sd.Plot(420, 560, x=(x0, 40, 'log'), y=(*yr, sc), xlabel='time after the flash',
                    ylabel=('energy uncertainty' if col == 'mip_sig_E' else 'time uncertainty [ns]') if fam == 'WAL' else '',
                    title=ttl, margin=(24, 18, 92, 96))
        P.band([x0, first(MF, fam, 'ded', 'linear_from_us')], [yr[0]] * 2, [yr[1]] * 2, sd.GREY, alpha=0.18)
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
                         ('not recovered yet', sd.GREY, 'box')], 24))
    D.slide('precision', body, '', short='precision')


def main() -> int:
    eo, lo = out_dir('early'), out_dir('linearity')
    M = pd.read_csv(eo / 'mip_eff.csv')
    MF = pd.read_csv(eo / 'mip_first.csv')
    X = dict(np.load(eo / 'examples.npz'))
    S = pd.read_csv(lo / 'shape_tests.csv')
    E = pd.read_csv(lo / 'shape_examples.csv')
    D = sd.Deck('X17 at n_TOF: when the scintillators recover from the γ-flash',
                'When the X17 scintillators recover from the n_TOF γ-flash, in time and in neutron energy')
    cover(D, MF)
    setup(D)
    waveforms(D, X, MF)
    examples(D, E)
    mismatch(D, S, MF)
    size(D, M)
    seen(D, M, MF)
    precision(D, M, MF)
    path = D.write(out_dir('deck') / 'scint-flash-recovery-summary.html', note_meta=dict(
        title='X17 at n_TOF: when the scintillators recover from the γ-flash',
        summary='Short, picture-first summary: when the X17 scintillators recover from the n_TOF γ-flash, as a time and '
                'as a neutron energy. Details: scint-flash-reconstruction.',
        tags='X17, n_TOF, scintillators', date=dt.date.today().isoformat()))
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
